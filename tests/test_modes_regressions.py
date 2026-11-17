"""Selección ambigua y cierre de simulaciones: regresiones R1 sin dispositivos."""

import pytest
from test_modes import CATALOG, SEC, G, cmd, joint, texts

from visionx.app.controller import Controller
from visionx.app.fakes import ScriptedDetector, SyntheticStereoSource
from visionx.app.runtime import View
from visionx.modes import build_handlers
from visionx.modes.search import CHOOSING, SearchHandler
from visionx.modes.simulate import simulate
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import DistancePolicy, Mode
from visionx.voice.commands import CommandParser


def test_nearest_does_not_skip_an_ambiguous_nearer_instance():
    handler = SearchHandler(G, CATALOG.spoken)
    command = cmd(Mode.SEARCH, "cup")
    handler.start(command, View(), 0)
    objects = [("cup", -2, None), ("cup", 2, None), ("cup", 25, None)]
    handler.tick(command, View(joint(1.9, 19, objects)), int(1.9 * SEC))
    seen = joint(2, 20, objects)
    handler.tick(command, View(seen), 2 * SEC)
    assert handler.state == CHOOSING
    measured = joint(
        1.8,
        18,
        [
            ("cup", 0, None, 0.7, DistancePolicy.SPEAK_METERS),
            ("cup", 25, None, 2.5, DistancePolicy.SPEAK_METERS),
        ],
        measured=True,
    )
    says = texts(handler.choose(command, "cerca", View(seen, measured), 2 * SEC))
    assert handler.state == CHOOSING
    assert handler.target is None
    assert says == ["No puedo saber cuál está más cerca. Decí izquierda, derecha o centro."]


@pytest.mark.parametrize("failure", [None, "source", "detector", "frame", "tail"])
def test_simulate_closes_ocr_workers_even_without_read_orders_or_after_failure(failure):
    controllers, workers = [], []
    try:
        for _ in range(3):
            clock = ManualClock(SEC)

            class Source(SyntheticStereoSource):
                reads = 0

                def read(self):
                    self.reads += 1
                    if failure == "source" and self.reads == 3:
                        raise RuntimeError("fuente rota")
                    return super().read()

            source = Source(fps=None, n_frames=3, clock=clock)

            class Detector(ScriptedDetector):
                def predict(self, frame):
                    if failure == "detector" and frame.frame_id == 2:
                        raise RuntimeError("detector roto")
                    return super().predict(frame)

            def make_controller(view, sink, event, clock=clock):
                ctl = Controller(
                    CommandParser(CATALOG),
                    sink,
                    latest_view=view,
                    clock=clock,
                    handlers=build_handlers({}, G, CATALOG.spoken, event),
                )
                controllers.append(ctl)
                return ctl

            def on_frame(t_s, frame, view, sink):
                worker = controllers[-1].handlers[Mode.READ]._worker
                if worker not in workers:
                    workers.append(worker)
                if failure == "frame" or (failure == "tail" and frame is None):
                    raise RuntimeError("callback roto")

            def run(source=source, clock=clock):
                return simulate(
                    source,
                    Detector(source),
                    make_controller,
                    clock,
                    [],
                    tick_s=1 / 30,
                    tail_s=0.2,
                    on_frame=on_frame,
                )

            if failure:
                with pytest.raises(RuntimeError):
                    run()
            else:
                assert run().frames == 3
        for worker in workers:
            worker._thread.join(timeout=1)
        assert len(workers) == 3
        assert not [w._thread.name for w in workers if w._thread.is_alive()]
    finally:
        # También limpiar al fallar contra el código anterior, sin contaminar la suite.
        for ctl in controllers:
            ctl.close()
        for worker in workers:
            worker._thread.join(timeout=1)
