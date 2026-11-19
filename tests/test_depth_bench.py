"""Banco ELP: lectura de escenas, medición por punto y resumen. Sin cámara ni redes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from visionx.depth import bench
from visionx.depth.bench import BenchPoint, load_scene, measure_point, save_points, summarize
from visionx.depth.stereo import build_result
from visionx.shared.contracts import StereoFrame, UncertaintyKind
from visionx.shared.errors import InvalidInput, SourceError
from visionx.shared.geometry import StereoGeometry

GEOM = StereoGeometry(width=200, height=120, fx=400.0, fy=400.0, cx=100.0, cy=60.0, baseline_m=0.06)


def write_scene(
    folder, session="s1", points=((1, 0.987), (2, 2.0)), marks=((1, 50, 40), (2, 150, 80))
):
    folder.mkdir(parents=True, exist_ok=True)
    sheet = {
        "formato": "visionx-banco-elp",
        "version": 1,
        "escena": folder.name,
        "sesion": session,
        "puntos": [
            {"id": i, "sobre": "pared", "superficie": "lisa", "clase_app": None, "cinta_m": tape}
            for i, tape in points
        ],
    }
    (folder / "escena.yaml").write_text(yaml.safe_dump(sheet, allow_unicode=True), "utf-8")
    save_points(folder, GEOM.size, {i: (u, v) for i, u, v in marks}, frame_id=4)
    return folder


def result_at(depth_m: float, size=None, b_px: float | None = None, holes=None):
    """DepthResult de un plano a `depth_m`, opcionalmente a otra resolución y con b."""
    geom = GEOM if size is None else GEOM.resized(*size)
    width, height = geom.size
    disparity = np.full((height, width), geom.fx * geom.baseline_m / depth_m, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    if holes is not None:
        valid[holes] = False
    left = np.zeros((GEOM.height, GEOM.width, 3), dtype=np.uint8)
    frame = StereoFrame(
        session_id="s",
        frame_id=0,
        left=left,
        right=left,
        received_ns=1,
        rectified=True,
        calibration_id="c",
        geometry_id=GEOM.geometry_id,
    )
    extra = {}
    if b_px is not None:
        extra = {
            "uncertainty": np.full((height, width), b_px, dtype=np.float32),
            "uncertainty_kind": UncertaintyKind.LAPLACE_B,
        }
    return build_result(frame, "falso", geom, disparity, valid, 1.0, 0.0, **extra)


def point(tape_m: float = 1.987, u: float = 50.0, v: float = 40.0) -> BenchPoint:
    return BenchPoint(id=1, u=u, v=v, tape_m=tape_m, surface="lisa", app_class=None)


def test_leer_una_escena(tmp_path) -> None:
    scene = load_scene(write_scene(tmp_path / "banco-s1-e01"))
    assert scene.name == "banco-s1-e01" and scene.session == "s1" and not scene.is_test
    assert scene.size == GEOM.size
    assert [(p.id, p.u, p.v, p.tape_m) for p in scene.points] == [
        (1, 50.0, 40.0, 0.987),
        (2, 150.0, 80.0, 2.0),
    ]
    assert load_scene(write_scene(tmp_path / "banco-s3-e01", session="s3")).is_test


def test_escena_incompleta_o_inconsistente(tmp_path) -> None:
    with pytest.raises(SourceError):
        load_scene(tmp_path / "no-existe")
    # Un sticker marcado que no está en la planilla.
    folder = write_scene(tmp_path / "a", marks=((1, 50, 40), (3, 150, 80)))
    with pytest.raises(InvalidInput, match="difieren"):
        load_scene(folder)
    # La planilla sin completar (cinta en cero).
    folder = write_scene(tmp_path / "b", points=((1, 0.0), (2, 2.0)))
    with pytest.raises(InvalidInput, match="lectura de cinta"):
        load_scene(folder)
    folder = write_scene(tmp_path / "c")
    (folder / "escena.yaml").write_text("formato: otro\npuntos: []\n", "utf-8")
    with pytest.raises(InvalidInput, match="formato"):
        load_scene(folder)


def test_medir_un_punto_suma_el_cero_de_la_cinta() -> None:
    # Cinta 1,987 m desde el vidrio + 13 mm hasta el centro óptico = 2,000 m.
    row = measure_point(result_at(2.0), point(), GEOM.size, zero_m=0.013)
    assert row["referencia_m"] == pytest.approx(2.0)
    assert row["medido_m"] == pytest.approx(2.0, abs=1e-3)
    assert row["error_pct"] == pytest.approx(0.0, abs=0.05)
    assert row["cubierto"] == 1.0
    far = measure_point(result_at(2.2), point(), GEOM.size, zero_m=0.013)
    assert far["error_pct"] == pytest.approx(10.0, abs=0.1)


def test_medir_a_otra_resolucion_y_con_huecos() -> None:
    half = measure_point(result_at(2.0, size=(100, 60)), point(), GEOM.size, zero_m=0.013)
    assert half["medido_m"] == pytest.approx(2.0, abs=1e-3)
    holes = (slice(36, 41), slice(0, 200))  # media ventana sin dato
    partial = measure_point(result_at(2.0, holes=holes), point(), GEOM.size, 0.013)
    assert 0.0 < partial["cubierto"] < 1.0 and partial["medido_m"] == pytest.approx(2.0, abs=1e-3)
    empty = measure_point(
        result_at(2.0, holes=(slice(30, 51), slice(0, 200))), point(), GEOM.size, 0.013
    )
    assert empty["cubierto"] == 0.0 and math.isnan(empty["error_pct"])


def test_intervalo_de_incertidumbre_en_metros() -> None:
    # A 2 m la disparidad es 12 px. Con b = 0,2 px el intervalo del 90 % es ±0,46 px.
    inside = measure_point(result_at(2.05, b_px=0.2), point(), GEOM.size, 0.013)
    low, high = inside["intervalo90_m"]
    assert low < 2.05 < high and high - 2.05 > 2.05 - low  # asimétrico: más ancho hacia lejos
    assert inside["dentro_intervalo90"] is True
    outside = measure_point(result_at(2.3, b_px=0.05), point(), GEOM.size, 0.013)
    assert outside["dentro_intervalo90"] is False
    assert "b_px" not in measure_point(result_at(2.0), point(), GEOM.size, 0.013)
    # A 2,05 m la disparidad es 11,71 px y la de referencia (2 m) 12 px: error de -0,29 px.
    assert inside["error_px"] == pytest.approx(24.0 / 2.05 - 12.0, abs=0.01)
    # |error| / b: con b = 0,2 px, la red confía de más por un factor ~1,5.
    assert summarize([inside])["escala_b_sugerida"] == pytest.approx(0.29 / 0.2, abs=0.05)


def test_resumen_cuenta_los_puntos_sin_medir_como_no_acertados() -> None:
    rows = [
        {"error_pct": 1.0, "referencia_m": 0.8},
        {"error_pct": -4.0, "referencia_m": 1.5},
        {"error_pct": 12.0, "referencia_m": 2.5},
        {"error_pct": math.nan, "referencia_m": 2.8},
    ]
    total = summarize(rows)
    assert total["puntos"] == 4 and total["medidos"] == 3 and total["cobertura"] == 0.75
    assert total["dentro_5pct"] == 50.0  # 2 de 4, no 2 de 3
    assert total["error_abs_mediana_pct"] == 4.0 and total["sesgo_mediana_pct"] == 1.0
    by_range = bench.summarize_by_range(rows)
    assert by_range["2-3.5 m"]["puntos"] == 2 and by_range["0-1 m"]["dentro_5pct"] == 100.0
    assert summarize([{"error_pct": math.nan, "referencia_m": 1.0}])["medidos"] == 0
    with_interval = summarize(
        [
            {"error_pct": 1.0, "referencia_m": 1.0, "dentro_intervalo90": True},
            {"error_pct": 9.0, "referencia_m": 1.0, "dentro_intervalo90": False},
        ]
    )
    assert with_interval["cobertura_intervalo90"] == 0.5


def test_evaluar_rechaza_sesiones_de_test_sin_la_bandera(tmp_path, capsys) -> None:
    # La calibración real tiene otro tamaño, pero la reserva se controla antes que todo.
    folder = write_scene(tmp_path / "banco-s4-e02", session="s4")
    assert bench.main(["evaluar", str(folder)]) == 1
    assert "reservadas" in capsys.readouterr().err
