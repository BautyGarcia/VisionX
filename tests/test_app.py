"""Tests de la ejecución (T07) con fuente sintética y modelos falsos.

Comprueban lo que T07 promete: los modelos no acumulan colas cuando no dan abasto, nunca se
juntan resultados de frames distintos, las órdenes nuevas invalidan las viejas y cancelar corta
el audio. No miden FPS de la ELP ni latencia audible.
"""

from __future__ import annotations

import csv
import json
import threading
import time

import numpy as np
import pytest

from visionx.app.__main__ import main as app_main
from visionx.app.controller import Controller, Say, UnavailableHandler
from visionx.app.dashboard import DashboardState, depth_colormap, render, top_view
from visionx.app.fakes import (
    FakeDepthEstimator,
    ScriptedDetector,
    SyntheticObject,
    SyntheticStereoSource,
)
from visionx.app.runtime import FrameCache, LatestMailbox, Pipeline, ResultBoard
from visionx.app.science import TraceWriter, compare
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import (
    DistancePolicy,
    FrameKey,
    Mode,
    check_depth_geometry,
    check_detection_frame,
    check_same_frame,
)
from visionx.shared.errors import EstimatorError, IdentityMismatch
from visionx.shared.interfaces import DepthEstimator, Detector, FrameSource, SpatialEstimator
from visionx.voice.commands import CommandParser
from visionx.voice.output import AudioOutput, FakePlayer, Outcome


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# --- piezas -----------------------------------------------------------------------------


def test_mailbox_keeps_only_latest_and_counts_skips():
    box: LatestMailbox[int] = LatestMailbox()
    for i in range(5):
        box.put(i)
    assert box.take(0.1) == 4
    assert box.overwritten == 4
    assert box.take(0.05) is None  # nada nuevo
    box.put(5)
    box.close()
    assert box.take(0.1) == 5  # lo pendiente se entrega aunque esté cerrado
    assert box.take(0.1) is None and box.closed


def test_frame_cache_and_board_are_bounded():
    source = SyntheticStereoSource(fps=None, n_frames=20)
    cache = FrameCache(5)
    board = ResultBoard(keep=5)
    det = ScriptedDetector(source)
    for _ in range(20):
        frame = source.read()
        cache.put(frame)
        board.put_detection(det.predict(frame))
    assert len(cache) == 5 and len(board.detections) == 5
    assert cache.get(FrameKey("sintetica", 0)) is None
    assert cache.get(FrameKey("sintetica", 19)) is not None


def test_board_joins_only_the_same_frame():
    source = SyntheticStereoSource(fps=None, n_frames=5)
    frames = [source.read() for _ in range(5)]
    det = ScriptedDetector(source)
    depth = FakeDepthEstimator(source.geometry, source=source)
    board = ResultBoard()
    board.put_depth(depth.predict(frames[1]))
    board.put_detection(det.predict(frames[1]))
    board.put_detection(det.predict(frames[3]))  # detección más nueva, sin profundidad
    board.put_depth(depth.predict(frames[4]))  # profundidad más nueva, sin detección
    assert board.latest_common_key() == FrameKey("sintetica", 1)


def test_fakes_satisfy_interfaces_and_contract_checks():
    source = SyntheticStereoSource((SyntheticObject("cup", 1.2),), fps=None, n_frames=1)
    frame = source.read()
    det = ScriptedDetector(source)
    depth = FakeDepthEstimator(source.geometry, source=source)
    spatial = ProvisionalSpatialEstimator(source.geometry)
    assert isinstance(source, FrameSource) and isinstance(det, Detector)
    assert isinstance(depth, DepthEstimator) and isinstance(spatial, SpatialEstimator)
    d = det.predict(frame)
    z = depth.predict(frame)
    check_detection_frame(frame, d)
    check_depth_geometry(z, source.geometry)
    (obj,) = spatial.locate(frame, z, d)
    assert obj.label == "cup" and obj.z_m == pytest.approx(1.2, rel=1e-3)
    assert obj.policy is DistancePolicy.SPEAK_METERS and obj.valid_fraction == 1.0
    assert det.model_id.startswith("fake-") and z.model_id.startswith("fake-")


def test_spatial_accepts_depth_in_a_resized_geometry():
    # SGBM calcula a 1280x720 aunque la captura sea 1920x1080: las máscaras del detector se
    # reducen a la resolución de la profundidad y la unión sigue siendo del mismo frame.
    source = SyntheticStereoSource((SyntheticObject("cup", 1.2),), fps=None, n_frames=1)
    frame = source.read()
    w, h = source.geometry.size
    half = source.geometry.resized(w // 2, h // 2)
    depth = FakeDepthEstimator(half, background_m=2.0).predict(frame)
    d = ScriptedDetector(source).predict(frame)
    with pytest.raises(IdentityMismatch):  # sin declarar la geometría reducida, se rechaza
        ProvisionalSpatialEstimator(source.geometry).locate(frame, depth, d)
    (obj,) = ProvisionalSpatialEstimator(source.geometry, depth_geometry=half).locate(
        frame, depth, d
    )
    assert obj.z_m == pytest.approx(2.0, rel=1e-3) and obj.valid_fraction == 1.0
    # el ángulo sale de la caja en la geometría del frame, no de la reducida
    x1, y1, x2, y2 = d.boxes[0]
    expected_h, _ = source.geometry.pixel_to_angles((x1 + x2) / 2, (y1 + y2) / 2)
    assert obj.angle_h_deg == pytest.approx(expected_h)


def test_synthetic_right_eye_is_shifted_by_disparity():
    source = SyntheticStereoSource((SyntheticObject("cup", 1.0),), fps=None, n_frames=1)
    frame = source.read()
    d = source.geometry.disparity_at(1.0)
    row = frame.left.shape[0] * 55 // 100
    left_x = np.flatnonzero(frame.left[row, :, 2] == 230)[0]
    right_x = np.flatnonzero(frame.right[row, :, 2] == 230)[0]
    assert left_x - right_x == round(d)


# --- pipeline ---------------------------------------------------------------------------


def test_pipeline_under_overload_keeps_bounded_queues_and_never_mixes_frames():
    source = SyntheticStereoSource(fps=200, n_frames=300)  # cámara más rápida que los modelos
    det = ScriptedDetector(source, delay_ms=20)
    depth = FakeDepthEstimator(source.geometry, source=source, delay_ms=35)
    events = []
    pipeline = Pipeline(
        source,
        det,
        depth,
        ProvisionalSpatialEstimator(source.geometry),
        cache_frames=10,
        on_event=lambda kind, data: events.append((kind, data)),
    )
    joints = []
    pipeline.start()
    while not pipeline.finished.is_set() or pipeline.running:
        joint = pipeline.latest_joint()
        if joint is not None:
            joints.append(joint)
            assert max(pipeline.queue_sizes().values()) <= 10
        time.sleep(0.005)
    pipeline.stop()
    assert pipeline.capture_status.processed == 300
    assert pipeline.detector_status.processed < 300  # salteó frames en vez de encolarlos
    assert pipeline.dropped()["detector"] > 0 and pipeline.dropped()["profundidad"] > 0
    assert joints, "nunca hubo una unión"
    for joint in joints:
        check_same_frame(joint.frame, joint.detection, joint.depth, *joint.objects)
    frame_ids = [j.frame.frame_id for j in joints]
    assert frame_ids == sorted(frame_ids)  # la unión publicada nunca retrocede
    assert any(kind == "union" for kind, _ in events)


def test_view_gives_detection_before_slow_depth():
    # Buscar (T09) avisa presencia con la detección sola: la vista la trae aunque la
    # profundidad todavía no haya terminado ese frame.
    source = SyntheticStereoSource(fps=60, n_frames=60)
    depth = FakeDepthEstimator(source.geometry, source=source, delay_ms=400)
    pipeline = Pipeline(
        source, ScriptedDetector(source), depth, ProvisionalSpatialEstimator(source.geometry)
    )
    pipeline.start()
    assert wait_for(lambda: pipeline.view().seen is not None, timeout=2)
    first = pipeline.view()
    assert first.measured is None or first.measured.frame.frame_id <= first.seen.frame.frame_id
    assert wait_for(lambda: pipeline.view().measured is not None, timeout=5)
    view = pipeline.view()
    pipeline.stop()
    assert view.seen.depth is None and view.measured.depth is not None
    assert view.seen.frame.frame_id >= view.measured.frame.frame_id
    check_same_frame(view.measured.frame, view.measured.detection, view.measured.depth)


def test_model_failure_is_counted_and_pipeline_continues():
    source = SyntheticStereoSource(fps=100, n_frames=40)

    class Flaky(ScriptedDetector):
        def predict(self, frame):
            if frame.frame_id % 2:
                raise EstimatorError("falla simulada")
            return super().predict(frame)

    pipeline = Pipeline(source, Flaky(source, delay_ms=5))
    pipeline.start()
    assert wait_for(lambda: pipeline.finished.is_set() and not pipeline.running)
    pipeline.stop()
    assert pipeline.detector_status.errors > 0
    assert pipeline.detector_status.processed > 0
    assert pipeline.latest_joint() is not None and pipeline.latest_joint().depth is None


def test_source_failure_stops_pipeline_with_error():
    from visionx.shared.errors import SourceError

    class Broken:
        def read(self):
            raise SourceError("cámara desconectada")

        def close(self):
            pass

    pipeline = Pipeline(Broken(), ScriptedDetector(SyntheticStereoSource(fps=None)))
    pipeline.start()
    assert wait_for(lambda: pipeline.finished.is_set())
    pipeline.stop()
    assert "desconectada" in pipeline.error


# --- controlador ------------------------------------------------------------------------


def make_controller(handlers=None, clock=None):
    output = AudioOutput(TextSynth(), FakePlayer())
    controller = Controller(
        CommandParser(ClassCatalog(COCO_CLASSES)),
        output,
        handlers=handlers,
        deadlines_s={"search": 30, "distance": 5},
        clock=clock,
    )
    return controller, output


class TextSynth:
    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(len(text), np.float32)


def spoken(output: AudioOutput) -> list[str]:
    output.wait_idle(2)
    return [r.text_or_sound for r in output.records]


def test_new_command_invalidates_previous_and_sets_deadline():
    clock = ManualClock(1_000)
    controller, output = make_controller(clock=clock)
    first = controller.handle(controller.parser.parse("buscá la taza"))
    second = controller.handle(controller.parser.parse("¿a qué distancia está la botella?"))
    assert second.request_id > first.request_id
    assert controller.active is second and second.mode is Mode.DISTANCE
    assert second.deadline_ns == 1_000 + 5 * 10**9
    texts = spoken(output)
    # distancia confirma con un sonido: la respuesta ya nombra el objeto (T09)
    assert "Busco taza." in texts and "sonido:ok" in texts
    assert "Esa función todavía no está disponible." in texts  # sin handler: no inventa
    output.close()


def test_clarification_does_not_cancel_active_command():
    controller, output = make_controller()
    command = controller.handle(controller.parser.parse("buscá la taza"))
    controller.handle_text("buscá las llaves")
    assert controller.active is command
    assert any("No sé buscar llaves" in t for t in spoken(output))
    output.close()


def test_cancel_cuts_audio_and_clears_active():
    controller, output = make_controller()
    output.player = FakePlayer(realtime=True, block_ms=5)

    class Talkative(UnavailableHandler):
        def start(self, command, view, now_ns):
            return [Say("x" * 3000, cancelable=True)]  # 3 s a 1 kHz

        def tick(self, command, view, now_ns):
            return [], False

    controller.handlers = {Mode.SEARCH: Talkative()}
    controller.handle_text("buscá la taza")
    assert wait_for(lambda: output.player.started.is_set())
    controller.handle_text("pará")
    assert controller.active is None
    output.wait_idle(3)
    long_one = next(r for r in output.records if r.text_or_sound == "x" * 3000)
    assert long_one.outcome in (Outcome.CANCELLED, Outcome.PREEMPTED)
    assert output.records[-1].text_or_sound == "sonido:cancelado"
    output.close()


def test_expired_command_stops_ticking_and_repeat_works():
    clock = ManualClock(0)
    calls = []

    class Counting(UnavailableHandler):
        def start(self, command, view, now_ns):
            return []

        def tick(self, command, view, now_ns):
            calls.append(now_ns)
            return [], False

    controller, output = make_controller(handlers={Mode.SEARCH: Counting()}, clock=clock)
    controller.handle_text("buscá la taza")
    controller.tick()
    clock.advance(31 * 10**9)  # vence a los 30 s
    controller.tick()
    assert len(calls) == 1 and controller.active is None
    controller.handle_text("repetí")
    assert spoken(output)[-1] == "Busco taza."
    output.close()


# --- dashboard, comparación y CLI -------------------------------------------------------


def test_dashboard_renders_with_and_without_results():
    source = SyntheticStereoSource(fps=None, n_frames=1)
    pipeline = Pipeline(
        source,
        ScriptedDetector(source),
        FakeDepthEstimator(source.geometry, source=source),
        ProvisionalSpatialEstimator(source.geometry),
    )
    empty = render(DashboardState(None, (pipeline.detector_status,), now_ns=0))
    assert empty.ndim == 3
    pipeline.start()
    assert wait_for(lambda: pipeline.latest_joint() is not None)
    pipeline.stop()
    image = render(
        DashboardState(
            pipeline.latest_joint(),
            (pipeline.capture_status, pipeline.detector_status, pipeline.depth_status),
            now_ns=pipeline.clock.now_ns(),
            geometry=source.geometry,
        )
    )
    assert image.shape[1] == 3 * 480 and image.dtype == np.uint8


def test_depth_colormap_uses_fixed_scale_and_black_for_missing():
    z = np.array([[0.5, 3.0, np.nan]], np.float32)
    valid = np.array([[True, True, False]])
    a = depth_colormap(z, valid, (0.5, 3.0))
    b = depth_colormap(z * 1.0, valid, (0.5, 3.0))
    assert np.array_equal(a, b) and a[0, 2].tolist() == [0, 0, 0]
    assert not np.array_equal(a[0, 0], a[0, 1])
    view = top_view(np.full((10, 10), 1.0, np.float32), np.ones((10, 10), bool), 9.5, 5, (0.5, 3))
    assert view.any()


def test_compare_processes_every_frame_with_every_method(tmp_path):
    source = SyntheticStereoSource(fps=None, n_frames=12)

    class FailsOnce(ScriptedDetector):
        def predict(self, frame):
            if frame.frame_id == 3:
                raise EstimatorError("falla")
            return super().predict(frame)

    methods = {
        "det": FailsOnce(source),
        "prof": FakeDepthEstimator(source.geometry, source=source),
    }
    result = compare(source, methods, tmp_path / "c.csv")
    rows = list(csv.DictReader(open(tmp_path / "c.csv", encoding="utf-8")))
    assert len(rows) == 24  # 12 frames x 2 métodos, ninguno descartado
    assert result["det"]["fallos"] == 1 and result["prof"]["fallos"] == 0
    assert result["det"]["frames"] == 12
    assert next(r for r in rows if r["frame_id"] == "3" and r["metodo"] == "det")["error"]


def test_trace_writer(tmp_path):
    trace = TraceWriter(tmp_path / "t.jsonl")
    threads = [threading.Thread(target=lambda: [trace("x", {"i": i}) for i in range(50)])]
    threads += [threading.Thread(target=lambda: [trace("y", {"i": i}) for i in range(50)])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    trace.close()
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100 and all(json.loads(line)["tipo"] in "xy" for line in lines)


def test_cli_runs_headless_with_scripted_orders(tmp_path, capsys):
    orders = tmp_path / "ordenes.jsonl"
    orders.write_text(
        '{"t_s": 0.1, "texto": "buscá la taza"}\n{"t_s": 0.3, "texto": "cancelar"}\n',
        encoding="utf-8",
    )
    code = app_main(
        ["ejecutar", "--sin-ventana", "--frames", "30", "--fps", "60",
         "--ordenes", str(orders), "--traza", str(tmp_path / "traza.jsonl"),
         "--captura", str(tmp_path / "dashboard.png")]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "frames_capturados: 30" in out
    events = [json.loads(line) for line in open(tmp_path / "traza.jsonl", encoding="utf-8")]
    kinds = [e["tipo"] for e in events]
    assert "orden" in kinds and "cancelado" in kinds and kinds[-1] == "resumen"
    # cada unión deja en la traza qué objeto se ubicó, a qué distancia y con qué política
    union_ids = [e["frame_id"] for e in events if e["tipo"] == "union"]
    assert len(union_ids) == len(set(union_ids))  # cada unión se registra una sola vez
    objs = [o for e in events if e["tipo"] == "union" for o in e["objetos"]]
    assert objs and {"clase", "z_m", "fraccion_valida", "politica"} <= set(objs[0])
    assert (tmp_path / "dashboard.png").stat().st_size > 0
    assert app_main(["comparar", "--frames", "5", "--csv", str(tmp_path / "c.csv")]) == 0
