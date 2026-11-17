"""T10 sin cámara, pesos, internet ni dispositivos: fakes explícitos y reloj manual."""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from visionx.app.controller import Controller
from visionx.app.fakes import ScriptedDetector, SyntheticStereoSource
from visionx.app.runtime import Pipeline, View
from visionx.camera.rectified import RectifiedSource
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes.read import ReadHandler
from visionx.reading import FakeReader, RapidReader, ReadingSettings, TextBlock
from visionx.reading.core import (
    framing_hint,
    normalize_text,
    reading_order,
    sharpness,
    speech_chunks,
    spoken_chunks,
)
from visionx.reading.evaluate import evaluate, reference_for, references, text_metrics
from visionx.reading.rapid import file_hash
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import Command, Mode, Priority, StereoFrame
from visionx.shared.errors import EstimatorError
from visionx.voice.commands import CommandParser

SEC = 1_000_000_000


def block(text="Hola", score=0.9, box=(10, 10, 100, 30)):
    return TextBlock(text, score, box)


def frame(t=0.1, frame_id=1, image=None, *, rectified=False):
    if image is None:
        image = np.zeros((120, 160, 3), np.uint8)
    return StereoFrame(
        "reading-test",
        frame_id,
        image,
        np.zeros_like(image),
        int(t * SEC),
        rectified=rectified,
        calibration_id="test" if rectified else None,
        geometry_id="test" if rectified else None,
    )


def command(request_id=1, deadline=30):
    return Command(request_id, Mode.READ, "leé", 0, deadline_ns=int(deadline * SEC))


def wait_for(condition, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.002)
    assert condition(), "el worker no completó el paso esperado"


@pytest.fixture
def handlers():
    created = []

    def build(reader=None, **overrides):
        overrides.setdefault("image_source", "original")  # frames sintéticos de estos tests
        handler = ReadHandler(
            reader if reader is not None else FakeReader([block()]), ReadingSettings(**overrides)
        )
        created.append(handler)
        return handler

    yield build
    for handler in created:
        handler.close()


def finish_selection(handler, cmd, view=None):
    handler.tick(cmd, view or View(), 2 * SEC)
    wait_for(lambda: handler._job.result.done())
    return handler.tick(cmd, View(), 2 * SEC)


def test_reading_order_groups_by_height_and_filters_score():
    source = [
        block("abajo", box=(10, 60, 150, 80)),
        block("derecha", box=(80, 14, 150, 34)),
        block("  izquie\u0301rda\n", box=(10, 10, 70, 30)),
        block("ruido", score=0.64),
        block("  ", score=1),
    ]
    result = reading_order(source, ReadingSettings())
    assert [b.text for b in result] == ["izquiérda", "derecha", "abajo"]
    assert result[0].box == source[2].box
    assert normalize_text("19 g\tC/U.\nÁZÚCAR") == "19 g C/U. ÁZÚCAR"
    assert reading_order([block(score=0.65)], ReadingSettings())


def test_tall_title_does_not_absorb_small_lines():
    source = [
        block("título", box=(1, 1, 150, 51)),
        block("arriba", box=(151, 10, 159, 20)),
        block("abajo", box=(151, 31, 159, 41)),
    ]
    assert [b.text for b in reading_order(source, ReadingSettings())] == [
        "arriba",
        "título",
        "abajo",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_score": -1},
        {"speech_min_score": 0.64},
        {"speech_min_score": 1.01},
        {"speech_min_score": float("nan")},
        {"window_s": 0},
        {"max_age_s": float("nan")},
        {"image_source": "auto"},
        {"threads": 0},
        {"max_chars": 1.5},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ReadingSettings(**kwargs)


def test_chunks_preserve_words_and_limit_length():
    chunks = speech_chunks([block("hola mundo adiós " + "x" * 50)], 12)
    assert all(len(s) <= 12 for s in chunks)
    assert "".join(chunks).replace(" ", "") == "holamundoadiós" + "x" * 50


def test_selects_sharpest_left_after_command(handlers):
    image = np.zeros((120, 160, 3), np.uint8)
    cv2.putText(image, "LEER", (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blur = cv2.GaussianBlur(image, (15, 15), 4)
    assert sharpness(image) > sharpness(blur)

    class Inspect(FakeReader):
        def read(self, selected):
            self.selected = selected
            return super().read(selected)

    reader = Inspect([block()])
    handler, cmd = handlers(reader), command()
    # Anterior a la orden: más nítido pero no debe entrar.
    old = replace(frame(0, 0, image), received_ns=0)
    handler.start(replace(cmd, issued_ns=1), View(captured=old), 1)
    handler.tick(cmd, View(captured=frame(0.2, 1, blur)), int(0.2 * SEC))
    wait_for(lambda: handler._job.selected_key is not None)
    handler.tick(cmd, View(captured=frame(0.5, 2, image)), int(0.5 * SEC))
    wait_for(lambda: handler._job.selected_key.frame_id == 2)
    says, done = finish_selection(handler, cmd)
    assert [s.text_or_sound for s in says] == ["Hola"] and not done
    assert np.array_equal(reader.selected, image)
    assert says[0].priority is Priority.LONG and says[0].cancelable


def test_original_and_rectified_are_explicit_and_not_interchanged(handlers):
    cmd = command()
    rect = frame(0.1, rectified=True)
    original = frame(0.1)
    handler = handlers()
    handler.start(cmd, View(captured=rect), SEC // 10)
    assert handler._last_key is None
    handler.tick(cmd, View(captured=rect, original=original), SEC // 10)
    assert handler._last_key == original.key
    rect_handler = handlers(image_source="rectified")
    rect_handler.start(cmd, View(captured=original), SEC // 10)
    assert rect_handler._last_key is None
    rect_handler.tick(cmd, View(captured=rect, original=original), SEC // 10)
    assert rect_handler._last_key == rect.key


def test_stale_future_and_outside_window_frames_are_ignored(handlers):
    handler, cmd = handlers(), command()
    handler.start(cmd, View(captured=frame(0.1)), 2 * SEC)  # demasiado viejo
    assert handler._last_key is None
    handler.tick(cmd, View(captured=frame(3)), 2 * SEC)
    assert handler._last_key is None
    assert finish_selection(handler, cmd)[0][0].text_or_sound == "No encuentro texto."


def test_no_text_and_evidence_based_framing(handlers):
    handler, cmd = handlers(FakeReader([block(score=0.3, box=(0, 10, 80, 30))])), command()
    handler.start(cmd, View(captured=frame()), SEC // 10)
    says, done = finish_selection(handler, cmd)
    assert done and says[0].text_or_sound == (
        "No encuentro texto. Probá encuadrar un poco hacia la izquierda."
    )
    assert framing_hint([], (120, 160, 3), ReadingSettings()) == ""
    assert (
        framing_hint([block(box=(20, 30, 60, 40))], (120, 160, 3), ReadingSettings()) == "Acercalo."
    )


def test_reader_failure_is_reported_without_stopping_controller(handlers):
    class Broken(FakeReader):
        def read(self, image):
            raise RuntimeError("modelo roto")

    handler, cmd = handlers(Broken()), command()
    handler.start(cmd, View(captured=frame()), SEC // 10)
    says, done = finish_selection(handler, cmd)
    assert done and says[0].text_or_sound.startswith("No pude leer.")


class Sink:
    def __init__(self):
        self.events, self.cancelled = [], []

    def submit(self, event):
        self.events.append(event)

    def cancel(self, request_id):
        self.cancelled.append(request_id)

    def cancel_all(self):
        self.cancelled.extend(e.request_id for e in self.events)


def controller(handler, view):
    sink, clock = Sink(), ManualClock(0)
    ctl = Controller(
        CommandParser(ClassCatalog(COCO_CLASSES)),
        sink,
        latest_view=lambda: view,
        handlers={Mode.READ: handler},
        clock=clock,
        deadlines_s={"read": 30},
    )
    return ctl, sink, clock


@pytest.mark.parametrize("next_order", ["cancelar", "describí", "leé"])
def test_cancel_or_new_order_halfway_through_reading(handlers, next_order):
    handler = handlers(
        FakeReader([block("primero"), block("segundo", box=(10, 40, 100, 60))]), max_chars=7
    )
    ctl, sink, clock = controller(handler, View(captured=frame(0)))
    ctl.handle_text("leé")
    old = ctl.active.request_id
    clock.set(2 * SEC)
    ctl.tick()
    wait_for(lambda: handler._job.result.done())
    ctl.tick()
    assert sink.events[-1].text_or_sound == "primero"
    job = handler._job
    ctl.handle_text(next_order)
    assert job.cancelled.is_set() and old in sink.cancelled
    clock.set(3 * SEC)
    ctl.tick()
    assert not any(e.request_id == old and e.text_or_sound == "segundo" for e in sink.events)
    ctl.close()


def test_expiry_cancels_audio_and_worker(handlers):
    handler = handlers()
    ctl, sink, clock = controller(handler, View(captured=frame(0)))
    ctl.handle_text("leé")
    request = ctl.active.request_id
    clock.set(30 * SEC)
    ctl.tick()
    assert ctl.active is None and handler._job.cancelled.is_set() and request in sink.cancelled
    assert handler.tick(command(deadline=30), View(), 30 * SEC) == ([], True)
    ctl.close()


def test_tick_and_capture_continue_while_ocr_is_blocked(handlers):
    started, release = threading.Event(), threading.Event()

    class Slow(FakeReader):
        def read(self, image):
            started.set()
            assert release.wait(3)
            return [block("viejo")]

    handler = handlers(Slow())
    ctl, sink, clock = controller(handler, View(captured=frame(0)))
    ctl.handle_text("leé")
    clock.set(2 * SEC)
    ctl.tick()
    assert started.wait(2)
    try:
        t0 = time.perf_counter()
        for _ in range(100):
            ctl.tick()
        assert time.perf_counter() - t0 < 0.1  # OCR continúa esperando un Event
        ctl.handle_text("cancelar")
        assert handler._job.cancelled.is_set()
    finally:
        release.set()
    wait_for(lambda: handler._job.result.done())
    ctl.tick()
    assert "viejo" not in [e.text_or_sound for e in sink.events]
    ctl.close()


def test_new_requests_replace_pending_work_and_ignore_old_result(handlers):
    started, release = threading.Event(), threading.Event()

    class Gated(FakeReader):
        calls = 0

        def read(self, image):
            self.calls += 1
            if self.calls == 1:
                started.set()
                release.wait(3)
                return [block("viejo")]
            return [block("nuevo")]

    reader = Gated()
    handler = handlers(reader)
    handler.start(command(1), View(captured=frame(0)), 0)
    handler.tick(command(1), View(), 2 * SEC)
    assert started.wait(2)
    try:
        handler.start(command(2), View(captured=frame(0)), 0)
        pending = handler._job
        handler.start(command(3), View(captured=frame(0)), 0)
        assert pending.cancelled.is_set()
        handler.tick(command(3), View(), 2 * SEC)
    finally:
        release.set()
    wait_for(lambda: handler._job.result.done())
    says, _ = handler.tick(command(3), View(), 2 * SEC)
    assert [s.text_or_sound for s in says] == ["nuevo"]
    assert reader.calls == 2  # una nativa vieja y la última orden, sin inferencias intermedias


def test_completed_read_audio_is_cancelled_by_next_order(handlers):
    handler = handlers()
    ctl, sink, clock = controller(handler, View(captured=frame(0)))
    ctl.handle_text("leé")
    request = ctl.active.request_id
    clock.set(2 * SEC)
    ctl.tick()
    wait_for(lambda: handler._job.result.done())
    ctl.tick()
    clock.set(3 * SEC)
    ctl.tick()
    assert ctl.active is None
    ctl.handle_text("describí")
    assert request in sink.cancelled  # podría seguir sonando el último Say
    ctl.close()


def test_deadline_also_cuts_last_audio_after_handler_finished(handlers):
    handler = handlers()
    ctl, sink, clock = controller(handler, View(captured=frame(0)))
    ctl.handle_text("leé")
    request = ctl.active.request_id
    clock.set(2 * SEC)
    ctl.tick()
    wait_for(lambda: handler._job.result.done())
    ctl.tick()
    clock.set(3 * SEC)
    ctl.tick()
    assert ctl.active is None and request not in sink.cancelled
    clock.set(30 * SEC)
    ctl.tick()
    assert request in sink.cancelled
    ctl.close()


def test_capture_advances_during_ocr_with_actual_pipeline(handlers):
    started, release = threading.Event(), threading.Event()

    class Slow(FakeReader):
        def read(self, image):
            started.set()
            release.wait(3)
            return [block()]

    source = SyntheticStereoSource(fps=100)
    pipeline = Pipeline(source, ScriptedDetector(source))
    handler = handlers(Slow(), image_source="rectified", window_s=0.03)
    ctl = Controller(
        CommandParser(ClassCatalog(COCO_CLASSES)),
        Sink(),
        latest_view=pipeline.view,
        handlers={Mode.READ: handler},
        deadlines_s={"read": 30},
    )
    pipeline.start()
    try:
        wait_for(lambda: pipeline.view().captured is not None)
        ctl.handle_text("leé")

        def tick_until_started():
            ctl.tick()
            return started.is_set()

        wait_for(tick_until_started)
        before = pipeline.capture_status.processed
        wait_for(lambda: pipeline.capture_status.processed >= before + 10)
        assert not release.is_set()
        ctl.handle_text("cancelar")
    finally:
        release.set()
        ctl.close()
        pipeline.stop()


def test_pipeline_preserves_original_without_waiting_for_detector():
    source = SyntheticStereoSource(fps=None, n_frames=1)

    class Raw:
        def read(self):
            f = source.read()
            return replace(f, rectified=False, calibration_id=None, geometry_id=None) if f else None

        def close(self):
            source.close()

    class Calibration:
        image_size = source.geometry.size
        calibration_id = "test"

        def geometry(self):
            return source.geometry

        def rectify_maps(self):
            return ()

        def rectify(self, left, right, maps):
            return np.full_like(left, 99), np.full_like(right, 99)

    rectified = RectifiedSource(Raw(), Calibration())
    pipeline = Pipeline(rectified, ScriptedDetector(source))
    pipeline._capture_loop()  # captura única; nunca se inicia el detector
    view = pipeline.view()
    assert view.seen is None and view.measured is None
    assert view.captured.key == view.original.key
    assert view.captured.rectified and not view.original.rectified
    assert np.all(view.captured.left == 99) and not np.all(view.original.left == 99)
    pipeline.stop()


def test_metrics_penalize_missing_extra_text_and_numbers():
    metrics = text_metrics("Azúcar 120", "AZUCAR 12")
    assert metrics["ediciones"] == 2 and metrics["cer"] == 0.2
    assert metrics["cer_sin_acentos"] == 0.1
    assert metrics["errores_numericos"] == 1
    assert text_metrics("Hola", "")["cer"] == 1
    assert text_metrics("a", "aaaa")["cer"] == 3  # no limitar CER a 100 %


def test_missing_model_fails_without_import_or_network(tmp_path):
    reader = RapidReader(ReadingSettings(weights_dir=str(tmp_path)))
    with pytest.raises(EstimatorError, match="descargar"):
        reader.read(np.zeros((30, 30, 3), np.uint8))


def test_rapid_adapter_is_cpu_and_returns_input_pixel_boxes(tmp_path, monkeypatch):
    from visionx.reading import rapid

    models = {}
    for stage in ("det", "rec", "cls"):
        path = tmp_path / f"{stage}.onnx"
        path.write_bytes(b"fake model - not ONNX")
        models[stage] = (path.name, file_hash(path))
    monkeypatch.setattr(rapid, "MODELS", models)
    params_seen = []

    class Engine:
        def __init__(self, *, params):
            params_seen.append(params)

        def __call__(self, image, **kwargs):
            assert kwargs == {"use_det": True, "use_cls": True, "use_rec": True}
            return SimpleNamespace(
                boxes=np.array([[[-2, 1], [18, 1], [18, 40], [-2, 40]]]),
                txts=["español"],
                scores=[0.8],
            )

    fake_module = SimpleNamespace(
        RapidOCR=Engine,
        EngineType=SimpleNamespace(ONNXRUNTIME="cpu"),
        LangRec=SimpleNamespace(LATIN="latin"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
    )
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)
    result = RapidReader(ReadingSettings(weights_dir=str(tmp_path), crop="none")).read(
        np.zeros((30, 30, 3), np.uint8)
    )
    assert result == [block("español", 0.8, (0, 1, 18, 30))]
    params = params_seen[0]
    assert params["Rec.lang_type"] == "latin"
    assert not params["EngineConfig.onnxruntime.use_cuda"]
    assert not params["EngineConfig.onnxruntime.use_coreml"]


def test_evaluation_uses_only_window_and_preserves_sources(tmp_path):
    folder = tmp_path / "clip"
    folder.mkdir()
    (folder / "textos.txt").write_text('1. Test (plano): "Hola" "120 g"\n', encoding="utf-8")
    (folder / "segmentos.csv").write_text(
        "objeto,distancia_m,inicio_ns,fin_ns\nTest,0.5,0,5000000000\n", encoding="utf-8"
    )
    entries = []
    for i, t in enumerate((0, SEC, 4 * SEC)):
        path = folder / f"{i}.jpg"
        cv2.imwrite(str(path), np.zeros((40, 80, 3), np.uint8))
        entries.append(
            {"frame_id": i, "received_ns": t, "left": path.name, "sha256_left": file_hash(path)}
        )
    (folder / "frames.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8"
    )
    (folder / "manifest.yaml").write_text(
        yaml.safe_dump(
            {"formato": "visionx-grabacion", "version": 1, "pares": 3, "session_id": "s"}
        ),
        encoding="utf-8",
    )
    before = {p.name: file_hash(p) for p in folder.iterdir()}
    refs = references(folder / "textos.txt")
    assert reference_for("Test", refs) == "Hola 120 g"
    with pytest.raises(ValueError):
        reference_for("no existe", refs)
    rows = evaluate(
        [folder],
        tmp_path / "result.csv",
        FakeReader([block("Hola"), block("120 g", box=(10, 31, 70, 39))]),
        ReadingSettings(speech_max_blocks=1),
        {},
        warmup=True,
    )
    assert rows[0]["n_frames"] == 2 and rows[0]["cer"] == 0
    assert rows[0]["recall_palabras"] == rows[0]["precision_palabras"] == 1
    assert rows[0]["aciertos_numericos"] == 1
    assert rows[0]["primera_palabra_lista_s"] >= 1.5
    assert rows[0]["primera_palabra_estimada_s"] is None  # no inventar audio
    assert before == {p.name: file_hash(p) for p in folder.iterdir()}
    assert (tmp_path / "result.json").is_file()
    detail = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert detail["calentamiento"]
    assert detail["resultados"][0]["hablado"] == ["Hola 120 g"]
    silent_rows = evaluate(
        [folder],
        tmp_path / "silent.csv",
        FakeReader([block("Hola"), block("120 g", box=(10, 31, 70, 39))]),
        ReadingSettings(speech_min_score=1),
        {},
    )
    # Solo cambia el habla; las métricas siguen usando incluso bloques no hablables.
    for key in ("ocr", *text_metrics("Hola 120 g", "Hola 120 g")):
        assert silent_rows[0][key] == rows[0][key]
    silent = json.loads((tmp_path / "silent.json").read_text(encoding="utf-8"))
    assert silent["resultados"][0]["hablado"] == []
    with (tmp_path / "result.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1

    # Una copia incompleta no se convierte silenciosamente en una evaluación completa.
    (folder / "1.jpg").unlink()
    from visionx.shared.errors import SourceError

    with pytest.raises(SourceError, match="faltan"):
        evaluate([folder], tmp_path / "missing.csv", FakeReader(), ReadingSettings(), {})
    partial = evaluate(
        [folder],
        tmp_path / "partial.csv",
        FakeReader(),
        ReadingSettings(),
        {},
        allow_incomplete=True,
    )
    assert partial[0]["n_frames"] == 1 and partial[0]["n_frames_esperados"] == 2
    assert not partial[0]["copia_completa_ventana"]
    meta = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
    assert meta["copias_incompletas"][0]["faltantes"] == ["1.jpg"]


def test_word_recall_tolerance_precision_and_occurrences():
    from visionx.reading.evaluate import word_matches, word_tokens

    m = text_metrics("ÁZÚCAR azúcar leche 120 g 24%", "azucar azúcr LECHE 12 g 24% 24% ruido")
    assert word_tokens("AZÚCAR-LECHE, 24% g") == ["azucar", "leche", "g"]
    assert m["n_palabras_ref"] == 4 and m["n_palabras_ocr"] == 5
    assert m["aciertos_palabras"] == 4
    assert m["recall_palabras"] == 1
    assert m["precision_palabras"] == 3 / 5  # aproximado no suma precisión estricta
    assert m["aciertos_numericos"] == 1 and m["recall_numeros"] == 0.5
    assert text_metrics("leche leche", "leche")["recall_palabras"] == 0.5
    assert text_metrics("hola", "hol")["recall_palabras"] == 0  # corta exige exacto
    assert text_metrics("leche", "leche leche")["precision_palabras"] == 0.5
    assert text_metrics("1,5% 120", "120 1.5%")["aciertos_numericos"] == 2
    assert text_metrics("hola", "")["precision_palabras"] == 0
    assert text_metrics("120", "")["recall_palabras"] is None
    assert text_metrics("hola", "")["recall_numeros"] is None
    # El primer aproximado no puede robar el único match de la segunda palabra.
    assert word_matches(["abcde", "xbcde"], ["abcde", "abcdf"]) == 2
    assert word_matches(["abcde"], ["abcd"]) == 1
    assert word_matches(["abcde"], ["abcdef"]) == 1
    assert word_matches(["abcde"], ["abxde"]) == 1
    assert word_matches(["abcde"], ["abxxe"]) == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_regions_keep_seals_in_columns_with_spanning_heading_and_footer(reverse):
    blocks = [
        block("TÍTULO", box=(0, 0, 260, 45)),
        block("EXCESO EN", box=(5, 80, 105, 100)),
        block("AZÚCARES", box=(10, 105, 100, 125)),
        block("EXCESO EN", box=(155, 80, 255, 100)),
        block("GRASAS", box=(160, 105, 250, 125)),
        block("PIE", box=(0, 170, 260, 190)),
    ]
    result = reading_order(blocks[::-1] if reverse else blocks, ReadingSettings())
    assert [b.text for b in result] == [
        "TÍTULO",
        "EXCESO EN",
        "AZÚCARES",
        "EXCESO EN",
        "GRASAS",
        "PIE",
    ]


def test_speech_title_first_then_visual_order_without_mutating_metrics_input():
    blocks = [
        block("sello", box=(0, 0, 100, 20)),
        block("TÍTULO", box=(0, 80, 100, 130)),
        block("detalle", box=(0, 220, 100, 240)),
    ]
    assert spoken_chunks(blocks, ReadingSettings()) == ["TÍTULO", "sello", "detalle"]
    assert [b.text for b in blocks] == ["sello", "TÍTULO", "detalle"]
    assert spoken_chunks([], ReadingSettings()) == []
    assert spoken_chunks(blocks, ReadingSettings(speech_max_blocks=1)) == [
        "TÍTULO",
        "…y más texto chico.",
    ]


def test_speech_character_limit_includes_suffix_and_long_words():
    blocks = [block("TÍTULO " + "detalle " * 40)]
    chunks = spoken_chunks(blocks, ReadingSettings(speech_max_chars=60, max_chars=20))
    assert sum(map(len, chunks)) <= 60 and all(len(c) <= 20 for c in chunks)
    assert chunks[0].startswith("TÍTULO") and chunks[-1] == "…y más texto chico."
    assert not any("¿" in c for c in chunks)
    assert spoken_chunks([block("texto")], ReadingSettings(speech_max_chars=24)) == ["texto"]
    chunks = spoken_chunks([block("x" * 500)], ReadingSettings(speech_max_chars=24, max_chars=4))
    assert sum(map(len, chunks)) <= 24 and all(len(c) <= 4 for c in chunks)


def test_speech_joins_seal_lines_without_joining_neighboring_regions():
    blocks = [
        block("Terrabust", box=(912, 296, 997, 345)),
        block("TITA", box=(1006, 295, 1179, 360)),
        block("EXCESO EN", box=(918, 439, 1001, 461)),
        block("AZÚCARES", box=(919, 456, 1001, 475)),
        block("EXCESO EN", box=(1010, 432, 1098, 453)),
        block("GRASAS", box=(1020, 446, 1086, 467)),
        block("TOTALES", box=(1019, 462, 1088, 484)),
    ]
    ordered = reading_order(blocks, ReadingSettings())
    assert spoken_chunks(ordered, ReadingSettings()) == [
        "TITA",
        "Terrabust",  # no corregir marcas
        "EXCESO EN AZÚCARES",
        "EXCESO EN GRASAS TOTALES",
    ]
    assert spoken_chunks(ordered, ReadingSettings(speech_max_blocks=3)) == [
        "TITA",
        "Terrabust",
        "EXCESO EN AZÚCARES",
        "…y más texto chico.",
    ]


def test_speech_region_splits_at_say_limit_and_keeps_words_and_region_boundary():
    blocks = [
        block("EXCESO EN", box=(0, 0, 80, 20)),
        block("GRASAS TOTALES", box=(0, 21, 80, 41)),
        block("OTRO SELLO", box=(100, 0, 180, 20)),
    ]
    chunks = spoken_chunks(blocks, ReadingSettings(max_chars=16))
    assert chunks == ["EXCESO EN GRASAS", "TOTALES", "OTRO SELLO"]
    assert all(len(c) <= 16 for c in chunks)


@pytest.mark.parametrize(
    "text,score,expected",
    [
        ("HEWLOOE", 0.66, []),
        ("haicskin&na", 0.99, []),
        ("bcdfgh", 0.99, []),
        ("a" * 31, 0.99, []),
        ("T1TA leche@ ###", 0.99, []),
        ("a de g", 0.99, []),
        ("EXCESO EN", 0.90, ["EXCESO EN"]),
        ("Ácido hialurónico", 0.99, ["Ácido hialurónico"]),
        ("Terrabust", 0.99, ["Terrabust"]),
        ("120 g 1,5% 24% 36 0.5 19 %", 0.99, ["120 g 1,5% 24% 36 0.5 19 %"]),
        ("leche de 120 g", 0.99, ["leche de 120 g"]),
        ("%", 0.99, ["%"]),
        ("anti-ácido d'angelo", 0.99, ["anti-ácido d'angelo"]),
        ("-", 0.99, []),
    ],
)
def test_speech_filter_plausibility_score_and_numbers(text, score, expected):
    assert spoken_chunks([block(text, score)], ReadingSettings()) == expected


def test_short_speech_words_need_context_in_same_line_not_just_same_region():
    blocks = [
        block("de", box=(0, 0, 20, 20)),
        block("leche", box=(21, 0, 90, 20)),
        block("la", box=(0, 21, 90, 41)),
    ]
    assert spoken_chunks(blocks, ReadingSettings()) == ["de leche"]


def test_only_speech_rejected_reports_confidence_instead_of_finishing_silently(handlers):
    handler = handlers(FakeReader([block("HEWLOOE", score=0.7)]))
    handler.start(command(), View(captured=frame()), SEC // 10)
    says, done = finish_selection(handler, command())
    assert done and [s.text_or_sound for s in says] == [
        "No pude leer con suficiente confianza. Probá acercarlo."
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop": "auto"},
        {"center_fraction": 0},
        {"center_fraction": 1.1},
        {"crop_margin": -1},
        {"text_probe_px": 0},
        {"region_gap_ratio": 0},
        {"speech_max_blocks": 0},
        {"speech_max_chars": 23},
    ],
)
def test_invalid_round2_settings(kwargs):
    with pytest.raises(ValueError):
        ReadingSettings(**kwargs)


class CropEngine:
    """Fake rotulado, sin modelos: caja conocida en cada imagen entregada."""

    def __init__(self, probe_boxes=True):
        self.calls = []
        self.probe_boxes = probe_boxes

    def __call__(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        if not kwargs["use_rec"]:
            # La prueba usa 200x100 de entrada, probe 100x50: centro 50%.
            return SimpleNamespace(
                boxes=np.array([[[25, 10], [75, 10], [75, 40], [25, 40]]])
                if self.probe_boxes
                else None
            )
        h, w = image.shape[:2]
        return SimpleNamespace(
            boxes=np.array([[[0, 0], [w, 0], [w, h], [0, h]]]), txts=["título"], scores=[0.99]
        )


@pytest.mark.parametrize(
    "mode,roi,n_calls",
    [
        ("none", (0, 0, 200, 100), 1),
        ("center", (40, 20, 160, 80), 1),
        ("text", (50, 20, 150, 80), 2),
        ("object", (50, 20, 150, 80), 2),  # sin detección de objeto: text
    ],
)
def test_crop_coordinate_mapping_and_two_pass_flags(mode, roi, n_calls):
    reader = RapidReader(
        ReadingSettings(crop=mode, crop_margin=0, text_probe_px=100, det_limit_px=400)
    )
    engine = reader._engine = CropEngine()
    result = reader.read(np.zeros((100, 200, 3), np.uint8))
    assert result[0].box == pytest.approx(roi)
    assert len(engine.calls) == n_calls
    assert engine.calls[-1][1] == dict(use_det=True, use_cls=True, use_rec=True)
    assert reader.last_crop["roi"] == roi
    if n_calls == 2:
        assert engine.calls[0] == ((50, 100, 3), dict(use_det=True, use_cls=False, use_rec=False))
    if mode != "none":
        assert max(engine.calls[-1][0][:2]) == 400


def test_text_crop_missing_boxes_uses_full_image_and_object_margin_clips():
    image = np.zeros((100, 200, 3), np.uint8)
    reader = RapidReader(ReadingSettings(crop="text", det_limit_px=400))
    reader._engine = CropEngine(probe_boxes=False)
    assert reader.read(image)[0].box == pytest.approx((0, 0, 200, 100))
    reader = RapidReader(ReadingSettings(crop="object", crop_margin=0.2, det_limit_px=400))
    engine = reader._engine = CropEngine()
    assert reader.read_region(image, (0, 10, 100, 90))[0].box == pytest.approx((0, 0, 120, 100))
    assert len(engine.calls) == 1


@pytest.mark.parametrize("mismatch", ["none", "frame", "geometry", "calibration", "size"])
def test_object_crop_only_uses_same_frame_geometry_and_keeps_selected_box(handlers, mismatch):
    from visionx.app.runtime import Joint
    from visionx.shared.contracts import Detection

    f = frame(rectified=True)
    detection = Detection(
        f.session_id,
        f.frame_id + (mismatch == "frame"),
        "fake",
        geometry_id="other" if mismatch == "geometry" else f.geometry_id,
        calibration_id="other" if mismatch == "calibration" else f.calibration_id,
        image_size=(80, 60) if mismatch == "size" else (160, 120),
        boxes=np.array([[0, 0, 20, 20], [60, 40, 100, 80]], np.float32),
        classes=("box", "box"),
        scores=np.array([0.9, 0.9], np.float32),
        inference_ms=0,
    )

    class Inspect(FakeReader):
        def read_region(self, image, box):
            self.selected_box = box
            return [block()]

    reader = Inspect()
    h = handlers(reader, crop="object", image_source="rectified")
    view = View(captured=f, seen=Joint(f, detection, None, ()))
    h.start(command(), view, SEC // 10)
    finish_selection(h, command())
    assert reader.selected_box == ((60, 40, 100, 80) if mismatch == "none" else None)


def test_real_seal_geometry_from_round1_does_not_interleave_lines():
    # Coordenadas del OCR Tita rectificada, 0,25 m (ronda 1). Las cajas de
    # GRASAS/TOTALES se solapan verticalmente y los gutters miden solo 6–10 px.
    source = [
        block("TITA", box=(1006, 289, 1178, 361)),
        block("Terrabusi", box=(907, 290, 999, 348)),
        block("AZÚCARES", box=(916, 451, 1004, 477)),
        block("EXCESO EN", box=(919, 440, 1002, 460)),
        block("EXCESO EN", box=(1014, 431, 1095, 453)),
        block("TOTALES", box=(1017, 455, 1089, 485)),
        block("GRASAS", box=(1018, 445, 1088, 471)),
        block("SATURADAS", box=(1101, 459, 1194, 485)),
        block("EXCESO EN", box=(1108, 431, 1186, 451)),
        block("GRASAS", box=(1118, 445, 1176, 466)),
        block("CALORIAS", box=(1200, 453, 1276, 474)),
        block("EXCESO EN", box=(1202, 439, 1278, 460)),
    ]
    ordered = reading_order(source, ReadingSettings())
    assert [b.text for b in ordered] == [
        "Terrabusi",
        "TITA",
        "EXCESO EN",
        "AZÚCARES",
        "EXCESO EN",
        "GRASAS",
        "TOTALES",
        "EXCESO EN",
        "GRASAS",
        "SATURADAS",
        "EXCESO EN",
        "CALORIAS",
    ]


def test_heading_is_read_before_columns_even_when_it_leaves_a_narrow_vertical_gutter():
    source = [
        block("TITA", box=(1006, 295, 1179, 360)),
        block("Terrabusi", box=(912, 296, 997, 345)),
        block("EXCESO EN", box=(918, 439, 1001, 461)),
        block("AZÚCARES", box=(919, 456, 1001, 475)),
        block("EXCESO EN", box=(1010, 432, 1098, 453)),
        block("GRASAS", box=(1020, 446, 1086, 467)),
        block("TOTALES", box=(1019, 462, 1088, 484)),
    ]
    assert [b.text for b in reading_order(source, ReadingSettings())] == [
        "Terrabusi",
        "TITA",
        "EXCESO EN",
        "AZÚCARES",
        "EXCESO EN",
        "GRASAS",
        "TOTALES",
    ]
