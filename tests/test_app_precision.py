"""Profundidad a pedido sin pesos/red/hardware: workers lentos sincronizados por eventos."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from visionx.app.__main__ import build_precision, main
from visionx.app.controller import Controller
from visionx.app.depth_registry import build_estimator
from visionx.app.fakes import FakeDepthEstimator, ScriptedDetector, SyntheticStereoSource
from visionx.app.precision import PrecisionWorker
from visionx.app.runtime import Joint, View
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes.common import ModeSettings, precision_view, request_precision
from visionx.modes.describe import DescribeHandler, DescribeSettings
from visionx.modes.distance import DistanceHandler
from visionx.modes.simulate import RecordingSink
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import Command, Mode, check_same_frame
from visionx.shared.errors import InvalidInput
from visionx.voice.commands import CommandParser

SEC = 1_000_000_000


class SlowDepth(FakeDepthEstimator):
    def __init__(self, geometry, *, wrong_frame=False, fail=False):
        super().__init__(geometry, background_m=2.0)
        self.model_id = "fake-precisa"
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.wrong_frame = wrong_frame
        self.fail = fail
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5), "el test debe liberar el cálculo nativo falso"
        if self.fail:
            raise RuntimeError("fallo de fábrica externa")
        result = super().predict(frame)
        return replace(result, frame_id=frame.frame_id + 1) if self.wrong_frame else result

    def close(self):
        self.closed.set()


@pytest.fixture
def setup_precision():
    clock = ManualClock(10 * SEC)
    source = SyntheticStereoSource(fps=None, clock=clock)
    frame = source.read()
    det = ScriptedDetector(source).predict(frame)
    continuous = FakeDepthEstimator(source.geometry, background_m=1.0)
    continuous.model_id = "fake-continua"
    depth = continuous.predict(frame)
    spatial = ProvisionalSpatialEstimator(source.geometry)
    measured = Joint(frame, det, depth, tuple(spatial.locate(frame, depth, det)))
    slow = SlowDepth(source.geometry)
    events = []
    worker = PrecisionWorker(
        slow, spatial, clock=clock, on_event=lambda kind, data: events.append((kind, dict(data)))
    )
    view = View(Joint(frame, det, None, ()), measured, precise=worker)
    try:
        yield SimpleNamespace(
            clock=clock,
            source=source,
            slow=slow,
            worker=worker,
            view=view,
            events=events,
            spatial=spatial,
        )
    finally:
        slow.release.set()
        worker.close()
        assert slow.closed.wait(5)


def command(request_id=1, mode=Mode.DISTANCE, target="cup"):
    return Command(request_id, mode, "x", 10 * SEC, target=target, deadline_ns=15 * SEC)


def wait_event(env, kind, request_id=1):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if any(k == kind and d.get("request_id") == request_id for k, d in env.events):
            return
        time.sleep(0.002)
    pytest.fail(f"no llegó {kind} para {request_id}: {env.events}")


def handler(env):
    catalog = ClassCatalog(COCO_CLASSES)
    return DistanceHandler(env.source.geometry, catalog.spoken)


def test_precise_arrives_in_time_and_uses_frozen_detection(setup_precision):
    env = setup_precision
    mode = handler(env)
    assert not mode.start(command(), env.view, env.clock.now_ns())
    assert env.slow.started.wait(5)
    env.clock.advance(100_000_000)
    env.slow.release.set()
    wait_event(env, "precisa-resultado")
    # Mientras calcula, la cámara/detección puede avanzar. No se aplica esta máscara nueva.
    frame2 = env.source.read()
    det2 = ScriptedDetector(env.source).predict(frame2)
    newer_view = replace(env.view, seen=Joint(frame2, det2, None, ()))
    chosen = precision_view(
        newer_view, command(), env.clock.now_ns(), ModeSettings(), env.source.geometry, "cup"
    )
    assert chosen.measured.precise_request_id == 1
    assert chosen.measured.frame.key == env.view.seen.frame.key
    check_same_frame(chosen.measured.frame, chosen.measured.detection, chosen.measured.depth)
    says, done = mode.tick(command(), newer_view, env.clock.now_ns())
    assert done and "2 metros" in says[0].text_or_sound
    assert env.worker.selection["model_id"] == "fake-precisa"
    assert env.worker.selection["latencia_ms"] >= 0
    assert env.slow.calls == 1


def test_timeout_falls_back_and_discards_late_result(setup_precision):
    env = setup_precision
    mode = handler(env)
    assert not mode.start(command(), env.view, env.clock.now_ns())
    assert env.slow.started.wait(5)
    env.clock.advance(2 * SEC)
    says, done = mode.tick(command(), env.view, env.clock.now_ns())
    assert done and "1 metro" in says[0].text_or_sound
    assert env.worker.selection["motivo"] == "espera-vencida"
    assert env.worker.selection["model_id"] == "fake-continua"
    env.slow.release.set()
    wait_event(env, "precisa-descartada")
    assert mode.tick(command(), env.view, env.clock.now_ns()) == ([], True)


@pytest.mark.parametrize("failure", ["invalid", "wrong_frame", "exception"])
def test_invalid_or_failed_precise_falls_back(setup_precision, failure):
    env = setup_precision
    if failure == "invalid":
        env.slow.background_m = 20.0  # fuera del rango de metros aceptados
    env.slow.wrong_frame = failure == "wrong_frame"
    env.slow.fail = failure == "exception"
    mode = handler(env)
    mode.start(command(), env.view, env.clock.now_ns())
    env.slow.release.set()
    wait_event(env, "precisa-resultado")
    says, done = mode.tick(command(), env.view, env.clock.now_ns())
    assert done and "1 metro" in says[0].text_or_sound
    assert env.worker.selection["model_id"] == "fake-continua"
    assert env.worker.selection["motivo"] != "precisa-valida"


@pytest.mark.parametrize("new_order", ["cancelar", "leé", "¿a cuánto está la taza?"])
def test_controller_cancels_pending_inference_without_old_speech(setup_precision, new_order):
    env = setup_precision
    output = RecordingSink(env.clock, lambda: 0)
    catalog = ClassCatalog(COCO_CLASSES)
    controller = Controller(
        CommandParser(catalog),
        output,
        latest_view=lambda: env.view,
        handlers={Mode.DISTANCE: handler(env)},
        clock=env.clock,
    )
    controller.handle_text("¿a cuánto está la taza?")
    old_id = controller.active.request_id
    assert env.slow.started.wait(5)
    controller.handle_text(new_order)
    assert env.worker.status.processed == 0  # la cancelación volvió con inferencia aún bloqueada
    cutoff = len(output.spoken)
    env.slow.release.set()
    wait_event(env, "precisa-descartada", old_id)
    if new_order == "¿a cuánto está la taza?":
        new_id = controller.active.request_id
        wait_event(env, "precisa-resultado", new_id)
    controller.tick()
    assert not any(e.request_id == old_id for e in output.spoken[cutoff:])
    if new_order == "¿a cuánto está la taza?":
        assert any(e.request_id == new_id and "2 metros" in e.text for e in output.spoken[cutoff:])
    controller.close()


def test_description_waits_and_uses_precise_facts(setup_precision):
    env = setup_precision
    catalog = ClassCatalog(COCO_CLASSES)
    describe = DescribeHandler(
        env.source.geometry, catalog.spoken, describe=DescribeSettings(persist_hits=1)
    )
    cmd = command(mode=Mode.DESCRIBE, target=None)
    assert not describe.start(cmd, env.view, env.clock.now_ns())
    env.slow.release.set()
    wait_event(env, "precisa-resultado")
    says, done = describe.tick(cmd, env.view, env.clock.now_ns())
    assert done and says
    assert describe.facts.depth_model_id == "fake-precisa"
    assert describe.facts.measured[0].z_m == pytest.approx(2.0)


def test_description_timeout_uses_continuous_facts(setup_precision):
    env = setup_precision
    catalog = ClassCatalog(COCO_CLASSES)
    describe = DescribeHandler(
        env.source.geometry, catalog.spoken, describe=DescribeSettings(persist_hits=1)
    )
    cmd = command(mode=Mode.DESCRIBE, target=None)
    assert not describe.start(cmd, env.view, env.clock.now_ns())
    assert env.slow.started.wait(5)
    env.clock.advance(2 * SEC)
    says, done = describe.tick(cmd, env.view, env.clock.now_ns())
    assert done and says
    assert describe.facts.depth_model_id == "fake-continua"
    assert describe.facts.measured[0].z_m == pytest.approx(1.0)


def test_precise_wait_is_limited_by_command_deadline(setup_precision):
    env = setup_precision
    cmd = replace(command(), deadline_ns=11 * SEC)
    mode = handler(env)
    says = mode.start(cmd, env.view, env.clock.now_ns())
    # Queda 1 s: ni la respuesta ni el aviso mínimo caben completos. No empezar
    # audio destinado al corte; el selector sí libera la espera de precisión.
    assert says == []
    assert env.worker.selection["motivo"] == "espera-vencida"


def test_observing_views_does_not_run_precise_without_order(setup_precision):
    env = setup_precision
    for _ in range(10):
        env.worker.select(command(), env.view, env.clock.now_ns(), lambda joint: None)
    assert not env.slow.started.is_set()
    assert env.slow.calls == 0


def test_request_without_detection_uses_next_detected_frame(setup_precision):
    env = setup_precision
    request_precision(View(precise=env.worker), command(), env.clock.now_ns(), ModeSettings())
    assert not env.slow.started.is_set()
    assert (
        precision_view(
            env.view, command(), env.clock.now_ns(), ModeSettings(), env.source.geometry, "cup"
        )
        is None
    )
    assert env.slow.started.wait(5)


def test_pending_queue_keeps_only_newest_order(setup_precision):
    env = setup_precision
    request_precision(env.view, command(), env.clock.now_ns(), ModeSettings())
    assert env.slow.started.wait(5)
    for request_id in (2, 3):
        env.worker.cancel(request_id - 1)
        request_precision(env.view, command(request_id), env.clock.now_ns(), ModeSettings())
    env.slow.release.set()
    wait_event(env, "precisa-descartada", 1)
    wait_event(env, "precisa-resultado", 3)
    assert env.slow.calls == 2
    assert not any(k == "precisa-resultado" and d["request_id"] in (1, 2) for k, d in env.events)


def test_old_detection_waits_for_fresh_one_before_computing(setup_precision):
    env = setup_precision
    old = replace(env.view.seen, frame=replace(env.view.seen.frame, received_ns=8 * SEC))
    old_view = replace(env.view, seen=old)
    request_precision(old_view, command(), env.clock.now_ns(), ModeSettings())
    assert not env.slow.started.is_set()
    precision_view(
        env.view, command(), env.clock.now_ns(), ModeSettings(), env.source.geometry, "cup"
    )
    assert env.slow.started.wait(5)


def test_cli_accepts_registry_names_for_both_paths(monkeypatch, tmp_path):
    module = ModuleType("fake_depth_cli")
    module.create = lambda cfg, geometry: FakeDepthEstimator(geometry)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = tmp_path / "registry.yaml"
    cfg.write_text(
        'depth:\n  estimadores:\n    plugin:\n      fabrica: "fake_depth_cli:create"\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "ejecutar",
                "--sin-ventana",
                "--frames",
                "2",
                "--config",
                str(cfg),
                "--profundidad",
                "plugin",
                "--profundidad-precisa",
                "plugin",
            ]
        )
        == 0
    )


def test_registry_factory_receives_free_parameters(monkeypatch, setup_precision):
    env = setup_precision
    module = ModuleType("fake_depth_plugin")
    received = []

    def factory(cfg, geometry):
        received.append((cfg, geometry))
        return FakeDepthEstimator(geometry, background_m=cfg["metros"])

    module.create = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = {
        "depth": {
            "estimadores": {
                "future": {"fabrica": "fake_depth_plugin:create", "metros": 1.75, "libre": [1, 2]},
            }
        }
    }
    est = build_estimator("future", cfg, env.source.geometry)
    assert received == [({"metros": 1.75, "libre": [1, 2]}, env.source.geometry)]
    assert est.predict(env.view.seen.frame).depth_m[0, 0] == pytest.approx(1.75)
    # Misma fábrica en la CLI y config de la precisa, sin ramas específicas para Fusion.
    cfg["depth"]["precisa"] = {"estimador": "future", "espera_s": 0.25}
    worker = build_precision(SimpleNamespace(profundidad_precisa=None), cfg, env.source.geometry)
    assert worker.wait_s == 0.25
    worker.close()
    assert (
        build_precision(SimpleNamespace(profundidad_precisa="ninguna"), cfg, env.source.geometry)
        is None
    )


@pytest.mark.parametrize(
    "factory", [None, "sin_dos_puntos", "no.existe:crear", "math:no_existe", "math:pi", "math:sqrt"]
)
def test_registry_invalid_factory(factory, setup_precision):
    cfg = {"depth": {"estimadores": {"bad": {"fabrica": factory}}}}
    with pytest.raises(InvalidInput, match="bad.*fabrica"):
        build_estimator("bad", cfg, setup_precision.source.geometry)


def test_registry_unknown_name(setup_precision):
    with pytest.raises(InvalidInput, match="desconocido.*desconocido"):
        build_estimator("desconocido", {}, setup_precision.source.geometry)


@pytest.mark.parametrize("name", ["B0", "B1", "B2"])
def test_registry_stereo(name, setup_precision):
    from visionx.depth.stereo import SgbmEstimator

    est = build_estimator(name, {}, setup_precision.source.geometry)
    assert isinstance(est, SgbmEstimator)
    est.close()


@pytest.mark.parametrize("name", ["B3", "B4"])
def test_registry_hybrid_without_weights(monkeypatch, name, setup_precision):
    from visionx.depth.hybrid import HybridEstimator

    monkeypatch.setattr(
        "visionx.depth.mono.DepthAnythingV2", lambda cfg: SimpleNamespace(model_id="fake-mono")
    )
    est = build_estimator(name, {}, setup_precision.source.geometry)
    assert isinstance(est, HybridEstimator)
    assert est.model_id.startswith(name.lower())


def test_real_b4_from_local_weights_only():
    """Recepción opcional: usa caché local pública de DA; nunca baja pesos en CI."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from visionx.depth.hybrid import HybridEstimator
    from visionx.shared.config import load_config

    cfg = load_config()
    mono = cfg["depth"]["mono"]
    try:
        hf_hub_download(
            mono["repo"], "model.safetensors", revision=mono["revision"], local_files_only=True
        )
    except LocalEntryNotFoundError:
        pytest.skip("pesos DA V2 Small ausentes en caché local")
    source = SyntheticStereoSource(fps=None)
    cfg["depth"]["compute"] = [320, 180]
    est = build_estimator("B4", cfg, source.geometry)
    try:
        assert isinstance(est, HybridEstimator)
        frame = source.read()
        result = est.predict(frame)
        check_same_frame(frame, result)
        assert result.size == (320, 180)
        assert np.isfinite(result.inference_ms)
    finally:
        est.close()


def test_registry_b5_uses_b5_section(monkeypatch, setup_precision):
    from visionx.depth.hybrid import HybridEstimator
    from visionx.shared.config import load_config

    monkeypatch.setattr(
        "visionx.depth.mono.DepthAnythingV2", lambda cfg: SimpleNamespace(model_id="fake-mono")
    )
    est = build_estimator("B5", load_config(), setup_precision.source.geometry)
    assert isinstance(est, HybridEstimator)
    assert est.matcher.variant == "B5" and est.matcher.local_params is not None


def test_registry_b5_without_section_is_clear(monkeypatch, setup_precision):
    monkeypatch.setattr(
        "visionx.depth.mono.DepthAnythingV2", lambda cfg: SimpleNamespace(model_id="fake-mono")
    )
    with pytest.raises(InvalidInput, match="depth.b5"):
        build_estimator("B5", {}, setup_precision.source.geometry)


def test_registry_fusion_goes_to_a_loader_with_full_config(monkeypatch, setup_precision):
    calls = []

    def load_estimator(cfg, geometry, *, package=None):
        calls.append((cfg, geometry, package))
        return FakeDepthEstimator(geometry)

    module = ModuleType("visionx.depth.fusion.estimator")
    module.load_estimator = load_estimator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    geometry = setup_precision.source.geometry
    entry = {"tipo": "fusion", "paquete": "b"}
    cfg = {"depth": {"fusion": {"paquete": "a"}, "estimadores": {"f-b5": entry}}}
    build_estimator("fusion", cfg, geometry)
    build_estimator("f-b5", cfg, geometry)
    assert [(c is cfg, g is geometry, p) for c, g, p in calls] == [
        (True, True, None),
        (True, True, "b"),
    ]


def test_registry_crestereo_goes_to_a_loader_with_full_config(monkeypatch, setup_precision):
    calls = []

    def load_estimator(cfg, geometry, mode="continuo"):
        calls.append((cfg, geometry, mode))
        return FakeDepthEstimator(geometry)

    module = ModuleType("visionx.depth.crestereo_adapt.estimator")
    module.load_estimator = load_estimator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    geometry = setup_precision.source.geometry
    cfg = {"depth": {"crestereo": {}}}
    build_estimator("crestereo", cfg, geometry)
    build_estimator("crestereo-preciso", cfg, geometry)
    assert [(c is cfg, g is geometry, m) for c, g, m in calls] == [
        (True, True, "continuo"),
        (True, True, "preciso"),
    ]
    with pytest.raises(InvalidInput, match="depth.crestereo"):
        build_estimator("crestereo", {"depth": {}}, geometry)


def test_registry_fusion_without_package_is_invalid(setup_precision):
    pytest.importorskip("torch")
    from visionx.shared.config import load_config

    with pytest.raises(InvalidInput, match="paquete"):
        build_estimator("fusion", load_config(), setup_precision.source.geometry)


def test_app_uses_a_locator_and_can_go_back_to_provisional(setup_precision):
    from visionx.app.__main__ import build_spatial
    from visionx.depth.spatial import ObjectLocator
    from visionx.shared.config import load_config

    cfg = load_config()
    geometry = setup_precision.source.geometry
    assert isinstance(build_spatial(cfg, geometry), ObjectLocator)
    cfg["spatial_provisional"]["usar"] = True
    assert isinstance(build_spatial(cfg, geometry), ProvisionalSpatialEstimator)
