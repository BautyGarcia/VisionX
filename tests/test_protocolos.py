"""Protocolos B sin dispositivos, internet ni pesos: CSV -> evaluación -> artefactos."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visionx.app.fakes import FakeDepthEstimator, ScriptedDetector, SyntheticStereoSource
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.protocolos import objects, scenes, text, usage, voice
from visionx.protocolos.__main__ import main
from visionx.protocolos.common import read_csv, sha, write_csv, write_json
from visionx.protocolos.templates import plantilla
from visionx.reading.core import FakeReader, TextBlock
from visionx.shared.contracts import DistancePolicy


class FakeVision:
    def __init__(self, *, abstain=False, duplicate=False):
        self.source = SyntheticStereoSource(fps=None, n_frames=3)
        self.geometry = self.source.geometry
        self.inputs = []
        self.detector = ScriptedDetector(self.source)
        self.depth = FakeDepthEstimator(self.geometry, source=self.source)
        self.abstain = abstain
        self.duplicate = duplicate
        self.closed = False

    def frames(self, *args, **kwargs):
        for i in range(3):
            yield replace(self.source.read(), received_ns=int((10 + i * 0.2) * 1e9))

    def observe(self, frame, methods):
        detection = self.detector.predict(frame)
        depth = self.depth.predict(frame)
        spatial = ProvisionalSpatialEstimator(self.geometry).locate(frame, depth, detection)
        if self.abstain:
            spatial = [replace(o, policy=DistancePolicy.DIRECTION_ONLY) for o in spatial]
        if self.duplicate:
            detection = SimpleNamespace(classes=("cup", "cup"))
        return detection, {m: (depth, spatial) for m in methods}

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "metric,n", [("m7", 120), ("m8", 63), ("voz", 61), ("ocr", 60), ("describir", 30)]
)
def test_templates_cli(tmp_path, metric, n):
    out = tmp_path / metric
    assert main([metric, "plantilla", "--salida", str(out)]) == 0
    assert len(read_csv(out / "planilla.csv")) == n
    assert (out / "GRABACION.md").is_file()
    assert main([metric, "plantilla", "--salida", str(out)]) == 1


def m7_sheet(tmp_path):
    sheet = plantilla("m7", tmp_path / "m7")
    row = read_csv(sheet)[0]
    row.update(clase_coco="cup", clip="clip", t_s=1, distancia_cinta_m=1)
    write_csv(sheet, [row])
    return sheet


@pytest.mark.parametrize("abstain,duplicate", [(False, False), (True, False), (False, True)])
def test_m7_chain(tmp_path, abstain, duplicate):
    backend = FakeVision(abstain=abstain, duplicate=duplicate)
    sheet = m7_sheet(tmp_path)
    result = objects.evaluate(sheet, tmp_path / "r", {}, ["B0", "B4"], backend=backend)
    first = result["resumen"]["tabla"][0]
    assert first["n"] == 1  # no contar los tres frames como tres repeticiones
    assert first["cobertura"] == (0 if abstain or duplicate else 1)
    if not abstain and not duplicate:
        assert first["sesgo_m"] == pytest.approx(0.2, abs=1e-5)
        assert first["mae_m"] == pytest.approx(0.2, abs=1e-5)
    else:
        assert first["mae_m"] is None
    assert backend.closed
    assert (tmp_path / "r/detalle.csv").is_file()


def test_m7_offsets_and_duplicates(tmp_path):
    sheet = m7_sheet(tmp_path)
    row = read_csv(sheet)[0]
    row["origen_cinta"] = "carcasa"
    write_csv(sheet, [row])
    with pytest.raises(ValueError, match="corrección"):
        objects.evaluate(sheet, tmp_path / "r", {}, ["B0"], backend=FakeVision())
    row["z_offset_m"] = -0.05
    write_csv(sheet, [row, row])
    with pytest.raises(ValueError, match="duplicada"):
        objects.evaluate(sheet, tmp_path / "r", {}, ["B0"], backend=FakeVision())


def test_m7_paired_reference_and_percentile():
    rows = [{"metodo": "B0", "referencia_z_m": 1, "z_pred_m": z} for z in (0.8, 1.4, None)]
    r = objects.summarize(rows)[0]
    assert r["sesgo_m"] == pytest.approx(0.1)
    assert r["mae_m"] == pytest.approx(0.3)
    assert r["p95_error_absoluto_m"] == pytest.approx(0.39)
    assert r["cobertura"] == 2 / 3


def trace_file(tmp_path, events):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(json.dumps(dict(t_ns=int(t * 1e9), tipo=k, **data)) for t, k, data in events),
        encoding="utf-8",
    )
    return path


def test_m8_chain_and_unknown_audio(tmp_path):
    path = trace_file(
        tmp_path,
        [
            (0, "orden", {}),
            (1, "audio", dict(request_id=1, texto="sonido:bip", resultado="reproducido")),
            (3, "cancelado", dict(request_id=2)),
            (4, "audio", dict(request_id=2, texto="hola", resultado="reproducido")),
            (5, "buscar-adquirido", dict(request_id=3)),
            (10, "fin", {}),
        ],
    )
    sheet = tmp_path / "m8.csv"
    base = dict(
        ensayo=1,
        ambiente="casa",
        objeto="cup",
        resultado_real="ausente",
        traza=path.name,
        request_id=1,
        inicio_s=0,
        fin_s=2,
        exito_s="",
        notas="",
    )
    write_csv(
        sheet,
        [
            base,
            dict(base, ensayo=2, resultado_real="cancelar", request_id=2, inicio_s=2, fin_s=5),
            dict(
                base,
                ensayo=3,
                resultado_real="exito",
                request_id=3,
                inicio_s=5,
                fin_s=10,
                exito_s=4,
            ),
        ],
    )
    assert (
        main(
            [
                "m8",
                "evaluar",
                "--planilla",
                str(sheet),
                "--salida",
                str(tmp_path / "r"),
                "--traza",
                str(path),
            ]
        )
        == 0
    )
    s = json.loads((tmp_path / "r/resumen.json").read_text(encoding="utf-8"))["resumen"]
    assert s["falsos_bips_por_min"] == 30
    assert s["exito_60s_fraccion"] == 1
    assert s["respuestas_tras_cancelar"] is None
    assert s["guia_vencida"] is None
    assert s["callbacks_posteriores_cancelar_a_revisar"] == 1


def test_m8_started_time_not_callback(tmp_path):
    path = trace_file(
        tmp_path,
        [
            (0, "orden", {}),
            (1, "cancelado", dict(request_id=1)),
            (
                2,
                "audio",
                dict(
                    request_id=1,
                    texto="Taza a la izquierda.",
                    resultado="reproducido",
                    started_ns=int(0.5e9),
                    valid_until_ns=int(0.4e9),  # guía que empezó a sonar ya vencida
                ),
            ),
            (
                3,
                "audio",
                dict(request_id=1, texto="tarde", resultado="reproducido", started_ns=int(2e9)),
            ),
            (4, "fin", {}),
        ],
    )
    sheet = tmp_path / "s.csv"
    write_csv(
        sheet,
        [
            dict(
                ambiente="casa",
                objeto="cup",
                resultado_real="cancelar",
                traza=path.name,
                request_id=1,
                inicio_s=0,
                fin_s=4,
                exito_s="",
            )
        ],
    )
    s = usage.evaluate(sheet, tmp_path / "r")["resumen"]
    assert s["respuestas_tras_cancelar"] == 1
    assert s["guia_vencida"] == 1
    assert s["violaciones_inicio_observadas"] == 2


def test_ocr_reuses_numeric_and_text_metrics(tmp_path):
    sheet = plantilla("ocr", tmp_path / "ocr")
    r = read_csv(sheet)[0]
    r.update(clip="clip", t_s=1, texto_referencia="Oferta 120 pesos")
    write_csv(sheet, [r])
    reader = FakeReader([TextBlock("Oferta 12 pesos", 0.99, (10, 10, 300, 40))])
    result = text.evaluate(sheet, tmp_path / "r", {}, reader=reader, backend=FakeVision())
    assert result["resumen"]["n"] == 1
    assert result["resumen"]["cer_micro"] == 1 / 16
    assert result["resumen"]["errores_numericos"] == 1
    assert result["resumen"]["recall_palabras_micro"] == 1


def test_describe_chain_executes_handler(tmp_path):
    sheet = plantilla("describir", tmp_path / "escenas")
    r = read_csv(sheet)[0]
    r.update(
        clip="clip",
        t_s=1,
        objetos_json=json.dumps([dict(clase="cup", zona="a la izquierda", distancia_m=1.2)]),
    )
    write_csv(sheet, [r])
    result = scenes.evaluate(sheet, tmp_path / "r", {}, backend=FakeVision())
    assert result["resumen"]["n_con_respuesta"] == 1
    assert result["resumen"]["objetos_reales_mencionados"] == 1
    assert result["resumen"]["metros_inventados"] == 0
    assert result["resumen"]["afirmaciones_sin_respaldo"] == 0


def test_describe_audit_text_not_all_facts():
    from visionx.detection.catalog import COCO_CLASSES, ClassCatalog

    facts = {
        "presence": [
            dict(label="cup", zone="al frente", detection_index=0, confidence="high"),
            dict(label="chair", zone="al frente", detection_index=1, confidence="high"),
        ],
        "measured": [dict(detection_index=0, z_m=1.23)],
    }
    result = scenes.audit(
        ["Al frente, una taza a 1,2 metros y dos sillas a 8 metros."],
        facts,
        [dict(clase="cup", zona="al frente", distancia_m=1)],
        ClassCatalog(COCO_CLASSES).spoken,
    )
    assert result["objetos_reales_mencionados"] == 1
    assert result["afirmaciones_sin_respaldo"] == 2
    assert result["metros_inventados"] == 1


def test_phrases_same_as_t08():
    from visionx.voice.__main__ import read_phrases

    original = [
        r for r in read_phrases(Path("tests/fixtures/voz-frases.csv")) if r["frase"].strip()
    ]
    assert [(r["frase"], r["modo"], r["objetivo"]) for r in voice.phrases()] == [
        (r["frase"], r["modo"], r["objetivo"]) for r in original
    ]


def test_voice_record_retry_skip_resume_and_input_level(tmp_path):
    sheet = plantilla("voz", tmp_path / "voz")
    write_csv(sheet, read_csv(sheet)[:3])
    answers = iter(["", "r", "", "a", "s", "q"])
    calls = []

    def recorder(seconds, device):
        calls.append(device)
        return np.full(1600, 0.1, np.float32), {"overflows": []}

    state = voice.record(
        sheet, countdown=0, recorder=recorder, device=4, ask=lambda _: next(answers)
    )
    assert calls == [4, 4]
    assert state["frases"]["1"]["estado"] == "ok"
    assert state["frases"]["1"]["intentos"] == 2
    assert state["frases"]["1"]["rms"] == pytest.approx(0.1)
    assert state["frases"]["2"]["estado"] == "saltada"
    assert len(list((sheet.parent / "wav").glob("*.wav"))) == 2
    answers = iter(["", "a"])
    state = voice.record(sheet, countdown=0, recorder=recorder, ask=lambda _: next(answers))
    assert state["frases"]["3"]["estado"] == "ok"
    assert len(calls) == 3


def test_voice_interrupt_error_and_silence(tmp_path):
    sheet = plantilla("voz", tmp_path / "voz")
    write_csv(sheet, read_csv(sheet)[:1])
    answers = iter(["", "", "", "q"])
    n = 0

    def recorder(seconds, device):
        nonlocal n
        n += 1
        if n == 1:
            raise OSError("dispositivo desconectado")
        return np.zeros(1600, np.float32), {}

    state = voice.record(sheet, countdown=0, recorder=recorder, ask=lambda _: next(answers))
    assert state["frases"]["1"]["estado"] == "pendiente"  # silencio no se acepta con Enter
    assert state["frases"]["1"]["advertencias"] == ["casi silencio"]
    state = voice.record(
        sheet,
        countdown=0,
        recorder=recorder,
        ask=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert (sheet.parent / "sesion.json").is_file()
    assert state["frases"]["1"]["estado"] == "pendiente"


def test_voice_evaluation_two_models_and_cache(tmp_path):
    sheet = plantilla("voz", tmp_path / "voz")
    write_csv(sheet, read_csv(sheet)[:1])
    answers = iter(["", "a"])
    voice.record(
        sheet,
        countdown=0,
        ask=lambda _: next(answers),
        recorder=lambda *_: (np.full(1600, 0.1, np.float32), {}),
    )
    calls = []

    class Transcriber:
        def __init__(self, name):
            self.model_id = "fake-" + name

        def transcribe(self, samples):
            calls.append(len(samples))
            return SimpleNamespace(text="buscá la taza", elapsed_ms=12)

    factories = {m: lambda m=m: Transcriber(m) for m in ("base", "small")}
    result = voice.evaluate(sheet, tmp_path / "r", {}, factories=factories)
    assert len(result["resumen"]["modelos"]) == 2
    for r in result["resumen"]["modelos"]:
        assert r["intencion_pct"] == r["objeto_pct"] == 100
        assert r["stt_p50_ms"] == 12
    voice.evaluate(sheet, tmp_path / "r", {}, factories=factories)
    assert len(calls) == 6  # 2 warmups por corrida, transcribir WAV solo la primera


def test_voice_resume_rejects_changed_annotations(tmp_path):
    sheet = plantilla("voz", tmp_path / "voz")
    write_json(sheet.parent / "sesion.json", {"planilla_sha256": "otro", "frases": {}})
    with pytest.raises(ValueError, match="cambió"):
        voice.record(sheet, recorder=lambda *_: pytest.fail("no abrir micrófono"))
    assert sha(sheet)


@pytest.mark.parametrize("metric,module", [("m7", objects), ("ocr", text), ("describir", scenes)])
def test_evaluate_cli_lazy_backend(tmp_path, monkeypatch, metric, module):
    called = []
    monkeypatch.setattr(module, "evaluate", lambda *a, **kw: called.append((a, kw)))
    assert main([metric, "evaluar", "--planilla", "ejemplo.csv", "--salida", str(tmp_path)]) == 0
    assert len(called) == 1


def test_cli_voice_never_records_during_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "evaluate", lambda *a, **kw: None)
    monkeypatch.setattr(voice, "record", lambda *a, **kw: pytest.fail("micrófono"))
    assert main(["voz", "evaluar", "--planilla", "ejemplo.csv", "--salida", str(tmp_path)]) == 0


def test_sparse_recording_window_and_hash_validation(tmp_path):
    from visionx.camera.recording import Recorder
    from visionx.protocolos.vision import OfflineVision, sample_entries

    backend = FakeVision()
    folder = tmp_path / "clip"
    with Recorder(folder) as recorder:
        for frame in backend.frames():
            recorder.write(frame)
    _, entries = sample_entries(folder, 0.2, 0.4, 3)
    assert len(entries) == 3
    assert sample_entries(folder, 0.2, 0.4, 1)[1][0]["frame_id"] == 1
    with pytest.raises(ValueError, match="fuera del clip"):
        sample_entries(folder, 100, 0.4, 3)
    reader = object.__new__(OfflineVision)
    reader.geometry = backend.geometry
    reader.inputs = []
    reader.calibration = SimpleNamespace(image_size=backend.geometry.size, calibration_id="fake")
    frames = list(reader.frames(folder, 0.2, 0.4, 3))
    assert [f.frame_id for f in frames] == [0, 1, 2]
    (folder / entries[0]["left"]).write_bytes(b"imagen rota")
    with pytest.raises(ValueError, match="hash"):
        list(reader.frames(folder, 0.2, 0.4, 3))


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3", "B4"])
def test_estimator_registry_no_weights(name, monkeypatch):
    from visionx.depth import mono
    from visionx.protocolos.vision import build_estimator
    from visionx.shared.config import load_config

    monkeypatch.setattr(mono, "DepthAnythingV2", lambda cfg: SimpleNamespace(model_id="fake-mono"))
    estimator = build_estimator(name, load_config(None), FakeVision().geometry)
    assert estimator.geometry.size == (1280, 720)
    assert name.lower() in estimator.model_id.lower()
    estimator.close()


def test_m8_rejects_missing_time_and_overlap(tmp_path):
    path = trace_file(tmp_path, [(0, "inicio", dict(request_id=1)), (10, "fin", {})])
    sheet = tmp_path / "s.csv"
    row = dict(
        ambiente="a",
        objeto="cup",
        resultado_real="ausente",
        traza=path.name,
        request_id=1,
        inicio_s=0,
        fin_s=600,
        exito_s="",
    )
    write_csv(sheet, [row])
    with pytest.raises(ValueError, match="cobertura temporal"):
        usage.evaluate(sheet, tmp_path / "r")
    row["fin_s"] = 9
    write_csv(sheet, [row, row])
    with pytest.raises(ValueError, match="superpuestos"):
        usage.evaluate(sheet, tmp_path / "r")
    row["request_id"] = 999
    write_csv(sheet, [row])
    with pytest.raises(ValueError, match="no existe en la traza"):
        usage.evaluate(sheet, tmp_path / "r")


def test_voice_synthetic_session_without_device(tmp_path, monkeypatch):
    from visionx.voice import engines

    sheet = plantilla("voz", tmp_path / "voz")
    write_csv(sheet, read_csv(sheet)[:1])

    class Synth:
        voice_id = "fake-piper"
        sample_rate = 22050

        def synthesize(self, phrase):
            return np.full(22050, 0.1, np.float32)

    monkeypatch.setattr(engines, "PiperSynth", lambda cfg: Synth())
    state = voice.synthesize_session(sheet, {})
    assert state["origen_audio"] == "sintetico_piper"
    assert (sheet.parent / state["frases"]["1"]["wav"]).is_file()
    with pytest.raises(ValueError, match="sintética"):
        voice.record(sheet)


def test_capture_level_and_stream_cleanup_without_microphone(monkeypatch, capsys):
    import sys

    closed = []

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(np.full((1600, 1), 0.25, np.float32), 1600, None, "input overflow")
            return self

        def __exit__(self, *args):
            closed.append(True)

    fake_sd = SimpleNamespace(
        query_devices=lambda *_: dict(default_samplerate=16000, name="fake-input"),
        InputStream=Stream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    times = iter([0, 0.1, 0.15, 0.3])
    monkeypatch.setattr(voice.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(voice.time, "sleep", lambda _: None)
    audio, metadata = voice.capture(0.2, 4)
    assert np.all(audio == 0.25)
    assert "pico 0.250" in capsys.readouterr().out
    assert metadata["overflows"] == ["input overflow"]
    assert closed == [True]


@pytest.mark.parametrize("kind", ["laplace_b", "lr_consistency"])
def test_m7_config_factory_cli(tmp_path, monkeypatch, kind):
    import sys

    import yaml

    from visionx.protocolos import vision
    from visionx.shared.config import load_config
    from visionx.shared.contracts import UncertaintyKind

    fake = FakeVision()
    adapter = object.__new__(vision.OfflineVision)
    adapter.geometry = fake.geometry
    adapter.detector = fake.detector
    adapter.inputs = []
    adapter.estimators = {}
    adapter.frames = fake.frames
    calls = []

    class Estimator(FakeDepthEstimator):
        def predict(self, frame):
            depth = super().predict(frame)
            return replace(
                depth,
                uncertainty=np.full(depth.valid_mask.shape, 0.4, np.float32),
                uncertainty_kind=UncertaintyKind(kind),
            )

    def factory(cfg, geometry):
        calls.append(cfg)
        return Estimator(geometry, source=fake.source)

    monkeypatch.setitem(sys.modules, "fake_m7_factory", SimpleNamespace(build=factory))
    config = tmp_path / "local.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "depth": {
                    "estimadores": {"fusion": {"fabrica": "fake_m7_factory:build", "parametro": 7}}
                }
            }
        ),
        encoding="utf-8",
    )
    adapter.cfg = load_config(config)
    monkeypatch.setattr(vision, "OfflineVision", lambda cfg: adapter)
    sheet = m7_sheet(tmp_path)
    assert (
        main(
            [
                "m7",
                "evaluar",
                "--planilla",
                str(sheet),
                "--salida",
                str(tmp_path / "r"),
                "--metodos",
                "fusion",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert calls == [{"parametro": 7}]
    rows = read_csv(tmp_path / "r/detalle.csv")
    assert float(rows[0]["incertidumbre_px"]) == pytest.approx(0.4)
    table = read_csv(tmp_path / "r/tabla.csv")
    assert table[0]["tipo_incertidumbre"] == kind
    assert float(table[0]["mae_confianza_50_m"]) == pytest.approx(0.2, abs=1e-5)
    assert float(table[0]["cobertura_confianza_50"]) == 1  # ceil con una fila
    # El import usado por protocolos es exactamente el constructor público.
    from visionx.app.depth_registry import build_estimator

    assert vision.build_estimator is build_estimator


def test_m7_risk_coverage_order_missing_ties_and_abstention():
    rows = [
        dict(
            metodo="fusion",
            referencia_z_m=1,
            z_pred_m=1 + error,
            incertidumbre_px=u,
            tipo_incertidumbre="laplace_b",
        )
        for error, u in [(0.1, 1), (0.2, 1), (0.4, 3), (0.8, 4)]
    ]
    rows += [
        dict(metodo="fusion", referencia_z_m=1, z_pred_m=z, incertidumbre_px=None)
        for z in (1.9, None)
    ]
    result = objects.summarize(rows)[0]
    assert result["n_con_incertidumbre"] == 4
    assert result["cobertura_confianza_100"] == 4 / 6
    assert result["mae_confianza_100_m"] == pytest.approx(0.375)
    assert result["cobertura_confianza_90"] == 4 / 6
    assert result["cobertura_confianza_75"] == 3 / 6
    assert result["mae_confianza_75_m"] == pytest.approx(0.7 / 3)
    assert result["cobertura_confianza_50"] == 2 / 6
    assert result["mae_confianza_50_m"] == pytest.approx(0.15)
    missing = objects.summarize(rows[-2:])[0]
    assert missing["mae_confianza_100_m"] is None
    assert missing["cobertura_confianza_100"] is None
    rows[0]["tipo_incertidumbre"] = "lr_consistency"
    assert objects.summarize(rows)[0]["mae_confianza_100_m"] is None


def test_m7_uncertainty_uses_resized_object_mask_only():
    from visionx.shared.geometry import resize_mask

    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    small = resize_mask(mask, 4, 4)
    valid = np.ones((4, 4), bool)
    valid[1, 1] = False
    uncertainty = np.full((4, 4), 100, np.float32)
    uncertainty[small] = 0.2
    uncertainty[~valid] = np.nan
    depth = SimpleNamespace(uncertainty=uncertainty, size=(4, 4), valid_mask=valid)
    detection = SimpleNamespace(masks=np.array([mask]))
    assert objects.object_uncertainty(depth, detection, 0) == pytest.approx(0.2)
    assert objects.object_uncertainty(depth, SimpleNamespace(masks=None), 0) is None


def fake_audio_devices():
    devices = [
        dict(name="Micr�fono (Yeti Nano)", hostapi=0, max_input_channels=1),
        dict(name="Micrófono (Yeti Nano)", hostapi=1, max_input_channels=1),
        dict(name="Micrófono (Yeti Nano)", hostapi=2, max_input_channels=1),
        dict(name="MicrÃ³fono (USB)", hostapi=0, max_input_channels=1),
        dict(name="Parlantes", hostapi=2, max_input_channels=0),
    ]
    return SimpleNamespace(
        query_devices=lambda: devices,
        query_hostapis=lambda: [
            dict(name=n) for n in ("MME", "Windows DirectSound", "Windows WASAPI")
        ],
    )


def test_voice_device_names_and_matching_without_audio(monkeypatch, capsys):
    import sys

    from visionx.protocolos.audio_devices import input_devices, resolve_device

    sd = fake_audio_devices()
    devices = input_devices(sd)
    assert [d["name"] for d in devices] == ["Micrófono (Yeti Nano)"] * 3 + ["Micrófono (USB)"]
    assert resolve_device("yeti nano", sd) == 2
    assert resolve_device("MICROFONO (YETI", sd) == 2
    assert resolve_device("microfono (usb)", sd) == 3
    assert resolve_device("0", sd) == 0
    assert resolve_device(None, sd) is None
    with pytest.raises(ValueError, match="ambiguo"):
        resolve_device("microfono", sd)
    with pytest.raises(ValueError, match="no hay entrada"):
        resolve_device("ausente", sd)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert main(["voz", "dispositivos"]) == 0
    output = capsys.readouterr().out
    assert "Micrófono (Yeti Nano)" in output
    assert "�" not in output


@pytest.mark.parametrize("cut", [True, False])
def test_voice_end_energy_warning_and_default_duration(tmp_path, capsys, cut):
    sheet = plantilla("voz", tmp_path / "voz")
    write_csv(sheet, read_csv(sheet)[:1])
    audio = np.zeros(16000 * 4, np.float32)
    audio[16000:32000] = 0.15
    if cut:
        audio[-3200:] = 0.15
    durations = []
    answers = iter(["", "q"])

    def recorder(seconds, device):
        durations.append(seconds)
        return audio, {}

    state = voice.record(sheet, countdown=0, recorder=recorder, ask=lambda _: next(answers))
    assert durations == [4]
    item = state["frases"]["1"]
    assert bool(item["advertencias"]) == cut
    assert item["rms_ultimos_200ms"] == pytest.approx(0.15 if cut else 0)
    assert ("posible frase cortada" in capsys.readouterr().out) == cut
    assert item["estado"] == "pendiente"
