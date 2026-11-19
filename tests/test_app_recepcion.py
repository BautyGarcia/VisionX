"""Recepción de entregas externas: CLI/replay real en disco; sin pesos ni dispositivos."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import fakes_recepcion
import numpy as np
import pytest
import yaml

from visionx.app.__main__ import main
from visionx.app.recepcion import locate_marks, validate_result
from visionx.calibration import StereoCalibration
from visionx.calibration.models import Intrinsics, LensModel, rectify
from visionx.camera import Recorder
from visionx.shared.contracts import Detection, StereoFrame


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    size = (64, 48)
    k = np.array([[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]])
    intr = Intrinsics(k, np.zeros(5))
    rotation, translation = np.eye(3), np.array([[-0.06], [0.0], [0.0]])
    rect = rectify(LensModel.PINHOLE5, intr, intr, size, rotation, translation, 0)
    calibration = StereoCalibration(
        size, LensModel.PINHOLE5, intr, intr, rotation, translation, rect, 0.0
    )
    calibration_path = tmp_path / "calib.yaml"
    calibration.save(calibration_path)
    clip = tmp_path / "clip-falso"
    image = np.zeros((48, 64, 3), np.uint8)
    with Recorder(clip) as recorder:
        for i in range(8):
            recorder.write(StereoFrame("fake-recepcion", i, image, image, received_ns=i * 10**9))
    config = {
        "camera": {"calibration": str(calibration_path)},
        "depth": {
            "estimadores": {
                "ejemplo": {"fabrica": "fakes_recepcion:build_example", "uncertainty": True},
                "roto": {"fabrica": "fakes_recepcion:build_broken"},
            }
        },
    }
    config_path = tmp_path / "override.yaml"
    output = tmp_path / "salida"

    def run(name="ejemplo", extra=(), external=False):
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        argv = [
            "recibir-profundidad",
            name,
            "--clip",
            str(clip),
            "--config",
            str(config_path),
            "--pares",
            "3",
            "--calentamiento",
            "1",
            "--salida",
            str(output),
            *extra,
        ]
        if external:
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src")]
                ),
                "PYTHONIOENCODING": "utf-8",
            }
            process = subprocess.run(
                [sys.executable, "-m", "visionx.app", *argv],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            code = process.returncode
            assert "Traceback" not in process.stderr, process.stderr
        else:
            code = main(argv)
        return code, json.loads((output / "recepcion.json").read_text(encoding="utf-8"))

    return config, run, calibration, output


def test_cli_external_factory_replay_json_and_comparison(delivery):
    _, run, _, output = delivery
    code, report = run(extra=("--contra", "roto"), external=True)
    assert code == 1
    good, bad = report["metodos"]
    assert good["ok"] and good["close_ok"]
    assert good["latencia"]["n"] == 3
    assert [p["frame_id"] for p in good["pares"]] == [0, 3, 7]
    assert good["latencia"]["p95_ms"] >= good["latencia"]["p50_ms"] >= 0
    assert good["pares"][0]["incertidumbre"]["rango_px"] == pytest.approx([0.2, 0.2])
    assert not bad["ok"] and bad["close_ok"]
    assert "finita" in bad["errores"][0]
    assert "FALLÓ" in (output / "tabla.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"broken": "nan"}, "finita"),
        ({"broken": "inf"}, "finita"),
        ({"broken": "dtype"}, "float32"),
        ({"size": [32, 24], "declare_geometry": False, "broken": None}, "sin declarar geometry"),
        ({"broken": "units"}, "depth_m no corresponde"),
        ({"broken": "frame"}, "frames distintos"),
        ({"broken": "model"}, "model_id"),
        ({"broken": "reason"}, "reason"),
        ({"broken": "uncertainty"}, "NaN"),
        ({"broken": "close"}, "close()"),
    ],
)
def test_rejects_invalid_delivery_and_closes(delivery, params, expected):
    config, run, _, _ = delivery
    config["depth"]["estimadores"]["roto"].update(params)
    code, report = run("roto")
    method = report["metodos"][0]
    assert code == 1 and not method["ok"]
    assert any(expected in error for error in method["errores"])
    assert fakes_recepcion.INSTANCES[-1].closed


@pytest.mark.parametrize(
    "factory, expected",
    [
        ("paquete_que_no_existe:fabrica", "no se puede importar"),
        ("fakes_recepcion:ausente", "no se puede importar"),
        ("fakes_recepcion:INSTANCES", "no es invocable"),
        (None, "no se puede importar"),
    ],
)
def test_factory_errors_are_clear(delivery, factory, expected):
    config, run, _, _ = delivery
    config["depth"]["estimadores"]["ejemplo"]["fabrica"] = factory
    code, report = run()
    assert code == 1
    assert expected in report["metodos"][0]["errores"][0]


def test_weight_preflight_missing_hash_and_good_hash(delivery, tmp_path):
    import hashlib

    config, run, _, _ = delivery
    entry = config["depth"]["estimadores"]["ejemplo"]
    weights = tmp_path / "fake-weights.txt"
    entry.update(pesos=str(weights), sha256="0" * 64)
    code, report = run()
    assert code == 1 and "pesos faltantes" in report["metodos"][0]["errores"][0]
    weights.write_bytes(b"fake")
    code, report = run()
    assert code == 1 and "SHA-256 distinto" in report["metodos"][0]["errores"][0]
    entry["sha256"] = hashlib.sha256(b"fake").hexdigest()
    code, report = run()
    assert code == 0 and report["metodos"][0]["pesos"]["verificado"]


@pytest.mark.parametrize(
    "extra, expected",
    [
        (("--pares", "0"), "--pares"),
        (("--pares", "99"), "contiene 8"),
        (("--calentamiento", "0"), "--calentamiento"),
    ],
)
def test_invalid_args_leave_failure_json(delivery, extra, expected):
    _, run, _, _ = delivery
    code, report = run(extra=extra)
    assert code == 1 and expected in report["errores"][0]


def test_accepts_scaled_geometry_disparity_only_and_nan_outside_mask(delivery):
    config, run, _, _ = delivery
    config["depth"]["estimadores"]["ejemplo"].update(
        size=[32, 24], disparity_only=True, outside_nan=True
    )
    code, report = run()
    assert code == 0
    pair = report["metodos"][0]["pares"][0]
    assert pair["size"] == [32, 24] and not pair["depth_m_recibida"]
    assert pair["depth_m"] == pytest.approx([1, 1])


def test_parity_uses_same_masks_spatial_settings_and_catches_units(delivery, monkeypatch):
    config, run, _, _ = delivery
    config["recepcion"] = {
        "marcas": [{"t_s": 2, "objeto": "taza falsa", "clase": "cup", "cinta_m": 1.0}]
    }
    config["spatial_provisional"] = {"min_valid": 0.5, "front_percentile": 20}

    class FakeDetector:
        def __init__(self, _):
            pass

        def predict(self, frame):
            mask = np.zeros((1, 48, 64), bool)
            mask[:, 8:40, 8:56] = True
            return Detection(
                frame.session_id,
                frame.frame_id,
                "fake-detector",
                frame.size,
                np.array([[8, 8, 56, 40]], np.float32),
                ("cup",),
                np.array([1], np.float32),
                0,
                geometry_id=frame.geometry_id,
                calibration_id=frame.calibration_id,
                masks=mask,
            )

        def close(self):
            pass

    monkeypatch.setattr("visionx.detection.YoloDetector", FakeDetector)
    code, report = run()
    assert code == 0
    obj = report["metodos"][0]["objetos"][0]
    assert obj["z_m"] == pytest.approx(1.0) and obj["politica"] == "speak_meters"
    assert obj["frame_id"] == 2 and obj["detector_model_id"] == "fake-detector"
    entry = config["depth"]["estimadores"]["ejemplo"]
    entry.update(broken="meters_as_disparity", z_m=0.1)
    code, report = run()
    # Solo disparidad consistente es contractual; sospecha de unidades contra cinta es aviso.
    assert code == 0
    assert any("metros/disparidad" in aviso for aviso in report["metodos"][0]["pares"][0]["avisos"])
    # La provisoria se abstenía sin depth_m; la de A mide en disparidad y da metros absurdos
    # (36 m contra 1 m de cinta): lo que importa es que la recepción lo marque.
    z_m = report["metodos"][0]["objetos"][0]["z_m"]
    assert z_m is None or z_m > 4.0

    entry.update(broken=None, disparity_only=True, z_m=1.0, uncertainty=False)
    code, report = run()
    assert code == 0
    # La ubicación de A mide en disparidad: sin depth_m también da metros.
    obj = report["metodos"][0]["objetos"][0]
    assert obj["politica"] == "speak_meters" and obj["z_m"] == pytest.approx(1.0, abs=0.05)
    assert not report["metodos"][0]["pares"][0]["incertidumbre"]["presente"]


def test_spatial_declared_resize_and_missing_or_ambiguous_objects(delivery):
    _, _, calibration, _ = delivery
    geometry = calibration.geometry()
    image = np.zeros((48, 64, 3), np.uint8)
    frame = StereoFrame(
        "fake",
        1,
        image,
        image,
        0,
        rectified=True,
        geometry_id=geometry.geometry_id,
        calibration_id=calibration.calibration_id,
    )
    estimator = fakes_recepcion.build_example({"size": [32, 24]}, geometry)
    result, _ = validate_result(
        frame, estimator.predict(frame), estimator, geometry, estimator.model_id
    )
    masks = np.zeros((2, 48, 64), bool)
    masks[0, 8:40, 8:26] = masks[1, 8:40, 38:56] = True
    detection = Detection(
        "fake",
        1,
        "fake-d",
        frame.size,
        np.array([[8, 8, 26, 40], [38, 8, 56, 40]], np.float32),
        ("cup", "cup"),
        np.ones(2, np.float32),
        0,
        geometry_id=frame.geometry_id,
        calibration_id=frame.calibration_id,
        masks=masks,
    )
    marks = [{"objeto": "taza", "clase": "cup", "cinta_m": 1, "lado": "izquierda"}]
    rows = locate_marks(frame, 0, marks, detection, result, {}, geometry, estimator.geometry)
    assert rows[0]["z_m"] == pytest.approx(1) and rows[0]["politica"] == "speak_meters"
    marks[0].pop("lado")
    assert (
        locate_marks(frame, 0, marks, detection, result, {}, geometry, estimator.geometry)[0][
            "politica"
        ]
        == "ambiguo"
    )
    detection = replace(detection, classes=("bottle", "bottle"))
    assert (
        locate_marks(frame, 0, marks, detection, result, {}, geometry, estimator.geometry)[0][
            "politica"
        ]
        == "ausente"
    )
