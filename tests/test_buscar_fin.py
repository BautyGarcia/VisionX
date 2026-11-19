"""La búsqueda termina cuando la persona lo dice («la encontré», «listo», «terminá»)."""

from __future__ import annotations

import numpy as np
import pytest

from visionx.app.controller import Controller
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.shared.config import load_config
from visionx.shared.contracts import Mode
from visionx.voice.commands import CANCEL, END_SEARCH, FOUND, CommandParser
from visionx.voice.output import AudioOutput, FakePlayer

PARSER = CommandParser(ClassCatalog(COCO_CLASSES))


@pytest.mark.parametrize(
    "frase",
    ["La encontré.", "Ya lo tengo", "Listo", "Ya está.", "Llegué", "listo, gracias",
     "Encontrado", "la agarré"],
)  # fmt: skip
def test_frases_de_encontrado(frase):
    i = PARSER.parse(frase)
    assert (i.mode, i.target) == (Mode.CONTROL, FOUND)


@pytest.mark.parametrize(
    "frase", ["Terminá la búsqueda.", "terminar", "Fin de la búsqueda", "No busques más"]
)
def test_frases_de_terminar(frase):
    assert PARSER.parse(frase).target == END_SEARCH


def test_buscar_y_cancelar_siguen_igual():
    assert PARSER.parse("Encontrá la taza").mode is Mode.SEARCH
    assert PARSER.parse("Encontré la taza").mode is Mode.SEARCH  # sin contexto: Whisper
    assert PARSER.parse("Dejá de buscar").target == CANCEL
    assert PARSER.parse("¿Ya está lista la taza?").target != FOUND


class _Synth:
    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(len(text), np.float32)


class _Buscar:
    def __init__(self):
        self.cancelado = False

    def start(self, command, view, now_ns):
        return []

    def tick(self, command, view, now_ns):
        return [], False

    def cancel(self):
        self.cancelado = True


def _ctl():
    out = AudioOutput(_Synth(), FakePlayer())
    h = _Buscar()
    ctl = Controller(
        PARSER, out, handlers={Mode.SEARCH: h}, deadlines_s={"search": 0},
        clock=ManualClock(1_000),
    )  # fmt: skip
    return ctl, out, h


def _dicho(out):
    out.wait_idle(2)
    return [r.text_or_sound for r in out.records]


@pytest.mark.parametrize("frase", ["La encontré.", "Encontré la taza.", "Terminá la búsqueda."])
def test_la_persona_termina_la_busqueda(frase):
    ctl, out, h = _ctl()
    c = ctl.handle(PARSER.parse("Buscá la taza."))
    assert c.deadline_ns is None  # sin vencimiento: no termina sola
    ctl.handle_text(frase)
    assert ctl.active is None and h.cancelado
    assert _dicho(out)[-1] == "Listo, terminé de buscar taza."
    out.close()


def test_encontrado_sin_busqueda():
    ctl, out, _ = _ctl()
    ctl.handle_text("Listo.")
    assert _dicho(out) == ["No estoy buscando nada ahora."]
    out.close()


def test_encontre_otra_cosa_durante_la_busqueda_es_buscarla():
    ctl, out, _ = _ctl()
    ctl.handle_text("Buscá la taza.")
    c = ctl.handle_text("Encontré la botella.")
    assert c.mode is Mode.SEARCH and ctl.active is not None and ctl.active.target == "bottle"
    out.close()


def test_config_por_defecto_busqueda_sin_plazo_ni_cierre_al_llegar():
    from visionx.modes.common import ModeSettings

    cfg = load_config()
    assert float(cfg["audio"]["deadlines_s"]["search"]) == 0
    assert ModeSettings.from_config(cfg).end_on_arrival is False
