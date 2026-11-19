"""Estado global de la cámara dicho por voz (app/camera_status.py), sin modelos ni cámara."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visionx.app.camera_status import (
    DEFAULT_PHRASES,
    OK,
    CameraStatus,
    CameraStatusSettings,
    status_code,
)
from visionx.app.controller import Controller
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.shared.config import load_config
from visionx.shared.contracts import DepthResult, DepthStatus
from visionx.voice.commands import CommandParser

S = 1_000_000_000


def depth(status=DepthStatus.VALID, reason=None) -> DepthResult:
    d = np.zeros((4, 4), np.float32)
    mask = np.full((4, 4), status is not DepthStatus.UNMEASURABLE)  # el contrato lo exige
    return DepthResult("s", 1, "m", "g", d + 1, mask, status, 1.0, reason)


def view(status=DepthStatus.VALID, reason=None):
    return SimpleNamespace(measured=SimpleNamespace(depth=depth(status, reason)))


U = DepthStatus.UNMEASURABLE


def texts(says) -> list[str]:
    return [s.text_or_sound for s in says]


def test_status_code_reads_the_agreed_codes_with_optional_detail():
    p = DEFAULT_PHRASES
    assert status_code(depth(U, "alineando"), p) == "alineando"
    assert status_code(depth(U, "camara-desconocida: ningún perfil"), p) == "camara-desconocida"
    assert status_code(depth(U, "no-medible"), p) == "no-medible"
    assert status_code(depth(U, "sin-textura"), p) == OK  # otro motivo: no es de la cámara
    assert status_code(depth(DepthStatus.PARTIAL, "alineando"), p) == OK  # mide algo
    assert status_code(None, p) == OK


def test_announces_once_after_hold_and_ready_when_it_measures_again():
    cam = CameraStatus(CameraStatusSettings(hold_s=1.0, repeat_s=0))
    assert cam.observe(view(), 0) == []  # al arrancar midiendo no dice nada
    assert cam.observe(view(U, "alineando"), 1 * S) == []
    assert cam.observe(view(U, "alineando"), int(1.5 * S)) == []  # todavía no duró hold_s
    said = texts(cam.observe(view(U, "alineando"), 2 * S))
    assert said == [DEFAULT_PHRASES["alineando"]]
    assert cam.observe(view(U, "alineando"), 50 * S) == []  # repeat_s 0: no recuerda
    assert cam.observe(view(), 51 * S) == []
    assert texts(cam.observe(view(), 52 * S)) == ["Listo, ya puedo medir distancias."]


def test_a_flicker_shorter_than_hold_is_not_announced():
    cam = CameraStatus(CameraStatusSettings(hold_s=1.0))
    cam.observe(view(U, "no-medible"), 0)
    cam.observe(view(), int(0.5 * S))
    assert cam.observe(view(), 2 * S) == [] and cam.announced == OK


def test_reminds_while_the_state_lasts():
    cam = CameraStatus(CameraStatusSettings(hold_s=0, repeat_s=30))
    first = texts(cam.observe(view(U, "camara-desconocida"), 0))
    assert first == [DEFAULT_PHRASES["camara-desconocida"]]
    assert cam.observe(view(U, "camara-desconocida"), 29 * S) == []
    assert texts(cam.observe(view(U, "camara-desconocida"), 30 * S)) == first


def test_changing_bad_state_announces_the_new_one():
    cam = CameraStatus(CameraStatusSettings(hold_s=0))
    cam.observe(view(U, "alineando"), 0)
    assert texts(cam.observe(view(U, "camara-desconocida"), S)) == [
        DEFAULT_PHRASES["camara-desconocida"]
    ]


def test_without_a_new_measurement_nothing_is_said():
    cam = CameraStatus(CameraStatusSettings(hold_s=0))
    assert cam.observe(SimpleNamespace(measured=None), 0) == []
    assert cam.observe(None, 0) == []


def test_codes_and_phrases_come_from_config():
    s = CameraStatusSettings.from_config(load_config())
    assert set(s.phrases) == {"alineando", "camara-desconocida", "no-medible"}
    custom = CameraStatusSettings.from_config({"camera_status": {"frases": {"x": "Equis."}}})
    cam = CameraStatus(CameraStatusSettings(phrases=custom.phrases, hold_s=0))
    assert texts(cam.observe(view(U, "x"), 0)) == ["Equis."]


class Sink:
    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)

    def cancel(self, request_id):
        pass

    def cancel_all(self):
        pass


def test_controller_speaks_watchers_with_their_own_request():
    clock, sink = ManualClock(), Sink()
    state = {"view": view(U, "alineando")}
    controller = Controller(
        CommandParser(ClassCatalog(COCO_CLASSES)),
        sink,
        latest_view=lambda: state["view"],
        watchers=[CameraStatus(CameraStatusSettings(hold_s=0))],
        clock=clock,
    )
    controller.tick()
    assert [e.text_or_sound for e in sink.events] == [DEFAULT_PHRASES["alineando"]]
    assert controller.active is None and controller.last_spoken == DEFAULT_PHRASES["alineando"]
    controller.tick()
    assert len(sink.events) == 1  # una sola vez


def test_controller_replace_cancels_the_same_request_before_speaking():
    from visionx.app.controller import Say

    sink = Sink()
    sink.cancelled = []
    sink.cancel = lambda rid: sink.cancelled.append(rid)
    controller = Controller(CommandParser(ClassCatalog(COCO_CLASSES)), sink, clock=ManualClock())
    controller._say(Say("A la derecha.", replace=True), 7, 0)
    controller._say(Say("A 2 metros."), 7, 0)
    assert sink.cancelled == [7] and [e.text_or_sound for e in sink.events] == [
        "A la derecha.",
        "A 2 metros.",
    ]
