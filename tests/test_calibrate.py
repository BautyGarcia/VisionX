"""«Calibrar distancia» (app/calibrate.py): intérprete, controlador y método opcional."""

from types import SimpleNamespace

import numpy as np

from visionx.app.calibrate import build_calibrator, parse_meters
from visionx.app.controller import Controller
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.voice.commands import CommandParser

CATALOG = ClassCatalog(COCO_CLASSES)


def test_parse_meters_in_digits_words_and_centimeters():
    assert parse_meters("Calibrá, está a 1,5 metros.") == 1.5
    assert parse_meters("calibrar a 150 centímetros") == 1.5
    assert parse_meters("Calibrá, está a dos metros y medio") == 2.5
    assert parse_meters("calibrá a un metro y medio") == 1.5
    assert parse_meters("calibrá a medio metro") == 0.5
    assert parse_meters("calibrá") is None


def test_parser_turns_calibrate_into_a_control_order():
    p = CommandParser(CATALOG)
    i = p.parse("Calibrá, está a 2 metros.")
    assert (i.mode.value, i.target, i.rule) == ("control", "calibrar:2", "calibrar")
    assert p.parse("Calibrá.").reply.startswith("Decime a cuántos metros")


class Sink:
    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)

    def cancel(self, request_id):
        pass

    def cancel_all(self):
        pass


def _view(angle_ok=True):
    det = SimpleNamespace(masks=np.ones((1, 4, 4), bool))
    frame = SimpleNamespace(received_ns=10**9)
    joint = SimpleNamespace(frame=frame, detection=det)
    return SimpleNamespace(measured=joint)


def test_controller_calls_the_estimator_method_when_it_exists(monkeypatch):
    import visionx.app.calibrate as cal

    cand = SimpleNamespace(label="cup", angle_h_deg=2.0, index=0)
    monkeypatch.setattr(cal, "candidates", lambda joint, label, geometry: [cand])
    monkeypatch.setattr(cal, "age_s", lambda joint, now: 0.1)
    calls = []
    depth = SimpleNamespace(calibrar_distancia=lambda frame, mask, z: calls.append(z) or True)
    view = _view()
    sink = Sink()
    ctl = Controller(
        CommandParser(CATALOG), sink, latest_view=lambda: view, clock=ManualClock(10**9),
        calibrate=build_calibrator(depth, None, CATALOG.spoken),
    )  # fmt: skip
    ctl.handle_text("Calibrá, está a 1,5 metros.")
    assert calls == [1.5]
    assert [e.text_or_sound for e in sink.events] == ["Listo, calibré con taza a 1,5 metros."]


def test_without_the_method_the_app_says_so():
    sink = Sink()
    ctl = Controller(
        CommandParser(CATALOG), sink, latest_view=lambda: None, clock=ManualClock(0),
        calibrate=build_calibrator(SimpleNamespace(), None, CATALOG.spoken),
    )  # fmt: skip
    ctl.handle_text("Calibrá a 2 metros.")
    assert "no se calibra" in sink.events[0].text_or_sound
