"""El vencimiento es un plazo para EMPEZAR a responder, no para cortar (vivo 19/11).

La frase de distancia se cortaba a la mitad, antes de los metros, porque al vencer la orden
(10 s) el controlador cortaba lo que sonaba. Ahora una respuesta que ya empezó termina, con un
tope; la confirmación («Leo.») y los sonidos no cuentan como respuesta.
"""

from __future__ import annotations

import numpy as np

from visionx.app.controller import Controller, Say
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import AudioEvent, Mode, Priority
from visionx.voice.commands import CommandParser
from visionx.voice.output import AudioOutput, FakePlayer, Outcome

SEC = 1_000_000_000


class _Synth:
    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(len(text), np.float32)


def _out():
    clock = ManualClock(0)
    return AudioOutput(_Synth(), FakePlayer(), clock=clock), clock


def _say(out, rid, text, deadline_s=5):
    out.submit(AudioEvent(rid, Priority.SHORT, text, created_ns=0, deadline_ns=deadline_s * SEC))
    out.wait_idle(2)
    return out.records[-1].outcome


def test_respuesta_empezada_termina_aunque_venza():
    out, clock = _out()
    clock.set(1 * SEC)
    assert _say(out, 1, "La taza está a 1,4 metros,") == Outcome.PLAYED
    assert out.answering(1)
    clock.set(6 * SEC)  # vencida, pero ya estaba respondiendo: el tramo siguiente suena
    assert _say(out, 1, "sobre el taburete.") == Outcome.PLAYED
    clock.set(26 * SEC)  # pasada la gracia (20 s): ya no
    assert _say(out, 1, "Y algo más.") == Outcome.EXPIRED
    out.close()


def test_la_confirmacion_no_cuenta_como_respuesta():
    out, clock = _out()
    out.note_ack(2, "Leo.")
    assert _say(out, 2, "Leo.") == Outcome.PLAYED
    assert _say(out, 2, "sonido:ok") == Outcome.PLAYED
    assert not out.answering(2)
    clock.set(6 * SEC)  # vencida sin haber empezado a responder: no habla
    assert _say(out, 2, "Dice ibuprofeno.") == Outcome.EXPIRED
    out.close()


class _Lento:
    """Modo que responde en dos tramos: el segundo llega después del vencimiento."""

    def __init__(self):
        self.ticks = 0
        self.cancelado = False

    def start(self, command, view, now_ns):
        return []

    def tick(self, command, view, now_ns):
        self.ticks += 1
        if now_ns >= 9 * SEC and self.ticks == 1:
            return [Say("La taza está a 1,4 metros,")], False
        if now_ns >= 12 * SEC:
            return [Say("sobre el taburete.")], True
        return [], False

    def cancel(self):
        self.cancelado = True


def _ctl(handler):
    out, clock = _out()
    ctl = Controller(
        CommandParser(ClassCatalog(COCO_CLASSES)), out, handlers={Mode.DISTANCE: handler},
        deadlines_s={"distance": 10}, clock=clock,
    )  # fmt: skip
    return ctl, out, clock


def test_controlador_no_corta_una_respuesta_que_ya_suena():
    h = _Lento()
    ctl, out, clock = _ctl(h)
    ctl.handle_text("¿A qué distancia está la taza?")
    for t in (9, 10.5, 12):  # empieza a 9 s; vence a 10 s; el segundo tramo llega a 12 s
        clock.set(int(t * SEC))
        ctl.tick()
        out.wait_idle(2)
    dicho = {r.text_or_sound: r.outcome for r in out.records}
    assert dicho["La taza está a 1,4 metros,"] == Outcome.PLAYED
    assert dicho["sobre el taburete."] == Outcome.PLAYED
    assert ctl.active is None and not h.cancelado  # terminó, no venció
    out.close()


def test_controlador_vence_si_no_empezo_a_responder():
    class Mudo(_Lento):
        def tick(self, command, view, now_ns):
            return [], False

    h = Mudo()
    ctl, out, clock = _ctl(h)
    ctl.handle_text("¿A qué distancia está la taza?")
    clock.set(10 * SEC)
    ctl.tick()
    assert ctl.active is None and h.cancelado
    out.close()


def test_maus_es_mouse():
    parser = CommandParser(ClassCatalog(COCO_CLASSES))
    assert parser.parse("¿A qué distancia está el maus?").target == "mouse"
    assert parser.parse("Buscá el mause").target == "mouse"
