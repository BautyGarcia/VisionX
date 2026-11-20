"""Respuestas cortas por defecto y «¿Querés más detalle?» solo cuando hay más para decir."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visionx.app.controller import Controller, Say
from visionx.assistant import Assistant, AssistantSettings
from visionx.assistant.core import KINDS, MORE_OFFER, offer_more, system_prompt
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.contracts import Mode
from visionx.voice.commands import EXPAND, NO, YES, CommandParser
from visionx.voice.output import AudioOutput, FakePlayer

PARSER = CommandParser(ClassCatalog(COCO_CLASSES), questions=True)
IMG = np.zeros((72, 128, 3), np.uint8)


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.chunks

    def get_final_message(self):
        usage = SimpleNamespace(input_tokens=1000, output_tokens=50)
        return SimpleNamespace(stop_reason="end_turn", usage=usage)


class _Client:
    def __init__(self, chunks):
        outer = self
        self.calls = []

        class Messages:
            def stream(self, **kw):
                outer.calls.append(kw)
                return _Stream(chunks)

        self.messages = Messages()
        self.beta = SimpleNamespace(messages=Messages())


def _job(chunks, kind="describir"):
    a = Assistant(AssistantSettings(ledger=""), client=_Client(chunks))
    job = a.submit(1, kind, IMG)
    assert job.done.wait(5)
    return job, a


def test_la_marca_no_se_dice_y_enciende_el_ofrecimiento():
    job, _ = _job(["Adelante hay una mesa con una taza. ", "[MÁS]"])
    assert [s for _, s in job.sentences] == ["Adelante hay una mesa con una taza."]
    assert job.more and offer_more(job)
    job, _ = _job(["Hay una silla al frente. [más]"])
    assert [s for _, s in job.sentences] == ["Hay una silla al frente."] and job.more


def test_sin_marca_no_se_ofrece_y_no_en_pedidos_internos():
    job, _ = _job(["Hay una silla al frente."])
    assert not job.more and not offer_more(job)
    job, _ = _job(["Está sobre la mesa. [MÁS]"], kind="apoyo")
    assert job.more and not offer_more(job)  # buscar no pregunta «¿más detalle?»


def test_prompt_corto_por_defecto_y_ampliar():
    p = system_prompt()
    assert "1 o 2 frases cortas" in p and "[MÁS]" in p
    assert "AMPLIAR" in KINDS["ampliar"] and "sin repetir" in KINDS["ampliar"]
    assert "2 frases como máximo" in KINDS["describir"]


@pytest.mark.parametrize("frase", ["Sí", "Sí, dale", "Dale", "Contame más", "Más detalle", "Ok"])
def test_si(frase):
    assert PARSER.parse(frase).target == YES


@pytest.mark.parametrize("frase", ["No", "No, gracias", "Así está bien"])
def test_no(frase):
    assert PARSER.parse(frase).target == NO


class _Describe:
    """Modo de prueba: responde y ofrece más; guarda las órdenes que recibe."""

    def __init__(self):
        self.commands = []

    def start(self, command, view, now_ns):
        self.commands.append(command)
        if command.target == EXPAND:
            return [Say("La taza es blanca y tiene un logo azul.")]
        return [Say("Adelante hay una taza."), Say(MORE_OFFER)]

    def tick(self, command, view, now_ns):
        return [], True


class _Synth:
    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(len(text), np.float32)


def _ctl():
    out = AudioOutput(_Synth(), FakePlayer())
    h = _Describe()
    return Controller(PARSER, out, handlers={Mode.DESCRIBE: h}), out, h


def _dicho(out):
    out.wait_idle(2)
    return [r.text_or_sound for r in out.records]


def test_si_despues_del_ofrecimiento_amplia():
    ctl, out, h = _ctl()
    ctl.handle_text("Describí")
    ctl.handle_text("Sí")
    assert [c.target for c in h.commands] == [None, EXPAND]
    assert _dicho(out)[-1] == "La taza es blanca y tiene un logo azul."
    out.close()


def test_no_cierra_y_si_suelto_no_hace_nada():
    ctl, out, h = _ctl()
    ctl.handle_text("Describí")
    ctl.handle_text("No")
    assert _dicho(out)[-1] == "Dale."
    ctl.handle_text("Sí")  # ya no hay ofrecimiento abierto
    assert len(h.commands) == 1 and _dicho(out)[-1] == "Dale."
    out.close()
