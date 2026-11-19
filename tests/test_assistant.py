"""Asistente con Claude (visionx.assistant) con un cliente falso: sin red ni gasto."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from visionx.assistant import (
    Assistant,
    AssistantSettings,
    Conversation,
    Delivery,
    split_sentences,
    verify_quotes,
)
from visionx.assistant.core import Ledger, user_text
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.voice.commands import READ_ALL, CommandParser, acknowledgement

IMG = np.zeros((72, 128, 3), np.uint8)


class FakeStream:
    def __init__(self, chunks, stop="end_turn", delay=0.0):
        self.chunks, self.stop, self.delay = chunks, stop, delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for c in self.chunks:
            time.sleep(self.delay)
            yield c

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self.stop, usage=SimpleNamespace(input_tokens=1500, output_tokens=100)
        )


class FakeClient:
    def __init__(self, chunks, **kw):
        self.calls = []
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                if isinstance(chunks, Exception):
                    raise chunks
                return FakeStream(chunks, **kw)

        self.messages = Messages()
        self.beta = SimpleNamespace(messages=Messages())


def settings(**kw):
    return AssistantSettings(ledger="", **kw)


def wait(job, timeout=5.0):
    assert job.done.wait(timeout)
    return job


# --- texto ---------------------------------------------------------------------------------


def test_quotes_confirmed_by_ocr_stay_and_the_rest_become_parece_decir():
    ocr = ["Terrabus!", "TITA", "EXCESO EN", "AZÚCARES", "CRASAS"]
    text = 'Es una caja que dice «Tita». Tiene un sello que dice "exceso en grasas".'
    out, bad = verify_quotes(text, ocr)
    assert "«Tita»" in out and bad == 0  # «GRASAS» está a una letra de «CRASAS»
    out, bad = verify_quotes("El frasco dice «60 cápsulas» y es de «Force Factor».", ["FORCE"])
    assert out == "El frasco parece decir 60 cápsulas y es de Force Factor." and bad == 2
    out, bad = verify_quotes("Los sellos dicen «exceso en sodio».", ["EXCESO EN"])
    assert out == "Los sellos parecen decir exceso en sodio." and bad == 1


def test_sentences_are_split_when_complete():
    ready, rest = split_sentences("Adelante tenés una caja. Dice Tita y tiene")
    assert ready == ["Adelante tenés una caja."] and rest.strip() == "Dice Tita y tiene"
    ready, rest = split_sentences("A 1,4 metros hay una silla. ", final=True)
    assert ready == ["A 1,4 metros hay una silla."] and rest == ""


def test_user_text_lists_measured_objects_and_ocr():
    text = user_text("pregunta", "¿Qué es?", ["TITA"], ["silla al frente, a 1,4 metros"])
    assert "«¿Qué es?»" in text and "- silla al frente, a 1,4 metros" in text and "- TITA" in text
    assert "no leyó texto" in user_text("leer", None, [], [])


def test_conversation_forgets_after_memory_s():
    c = Conversation(memory_s=120, max_turns=2)
    c.add("¿Qué es?", "Una caja.", now_s=0)
    assert len(c.history(100)) == 2
    assert c.history(121 + 0) == []  # pasó más de memory_s desde el último intercambio


def test_ledger_adds_cost_and_stops_at_budget(tmp_path):
    path = tmp_path / "gasto.jsonl"
    ledger = Ledger(str(path), budget_usd=0.01)
    usd = ledger.add("claude-sonnet-5-5", 1500, 100, "leer")
    assert usd == pytest.approx(0.004) and not ledger.exhausted()
    ledger.add("claude-sonnet-5-5", 1500, 100, "leer")
    ledger.add("claude-sonnet-5-5", 1500, 100, "leer")
    assert ledger.exhausted()
    assert Ledger(str(path), 0.01).spent == pytest.approx(0.012)  # sobrevive a reinicios
    assert json.loads(path.read_text().splitlines()[0])["tipo"] == "leer"


# --- pedidos -------------------------------------------------------------------------------


def test_assistant_streams_sentences_verifies_quotes_and_remembers():
    client = FakeClient(["Adelante tenés una caja ", "que dice «TITA». Es dulce", "."])
    a = Assistant(settings(), client=client)
    job = wait(a.submit(1, "leer", IMG, ocr=["TITA"]))
    assert job.error is None
    assert [s for _, s in job.sentences] == [
        "Adelante tenés una caja que dice «TITA».",
        "Es dulce.",
    ]
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"][-1]["content"][0]["type"] == "image"
    job2 = wait(a.submit(2, "pregunta", IMG, request="¿Y qué más dice?"))
    history = client.calls[1]["messages"]
    assert len(history) == 3 and history[1]["content"] == job.text  # recuerda lo anterior
    assert job2.error is None


def test_errors_and_refusals_mark_the_job_failed_for_the_fallback():
    a = Assistant(settings(), client=FakeClient(ConnectionError("sin red")))
    job = wait(a.submit(1, "describir", IMG))
    assert "sin red" in job.error and Delivery(job, 0).failed
    a = Assistant(settings(), client=FakeClient(["No."], stop="refusal"))
    assert "refusal" in wait(a.submit(1, "pregunta", IMG, request="x")).error


def test_budget_exhausted_means_unavailable(tmp_path):
    ledger = Ledger(str(tmp_path / "g.jsonl"), budget_usd=0.001)
    ledger.add("claude-sonnet-5-5", 1500, 100, "leer")
    a = Assistant(settings(), client=FakeClient(["Hola."]), ledger=ledger)
    assert not a.available
    assert "tope" in wait(a.submit(1, "leer", IMG)).error


def test_new_request_cancels_the_previous_one():
    a = Assistant(settings(), client=FakeClient(["Uno. ", "Dos. ", "Tres."], delay=0.05))
    first = a.submit(1, "leer", IMG)
    second = a.submit(2, "leer", IMG)
    wait(first), wait(second)
    assert first.cancelled.is_set() and not second.cancelled.is_set()


def test_question_about_text_runs_ocr_first():
    seen = []
    a = Assistant(
        settings(),
        client=FakeClient(["Vence en marzo."]),
        ocr=lambda im: seen.append(1) or ["VTO 03/27"],
    )
    job = wait(a.submit(1, "pregunta", IMG, request="¿Cuándo vence?", read_text=True))
    assert seen == [1] and job.ocr == ["VTO 03/27"]


def test_delivery_respects_the_mode_clock():
    job = SimpleNamespace(
        sentences=[(0.5, "Uno."), (1.5, "Dos.")],
        done=SimpleNamespace(is_set=lambda: True),
        error=None,
    )
    d = Delivery(job, start_ns=10 * 10**9)
    assert d.ready(int(10.4e9)) == []
    assert d.ready(int(10.6e9)) == ["Uno."]
    assert d.ready(int(12e9)) == ["Dos."] and d.finished


# --- intérprete ----------------------------------------------------------------------------


def test_free_questions_only_with_assistant():
    catalog = ClassCatalog(COCO_CLASSES)
    on, off = CommandParser(catalog, questions=True), CommandParser(catalog)
    q = on.parse("¿De qué color es la taza?")
    assert (q.mode.value, q.target, q.rule) == ("describe", "¿De qué color es la taza?", "pregunta")
    assert acknowledgement(q, catalog) == "sonido:ok"
    assert off.parse("¿De qué color es la taza?").rule == "sin-regla"
    assert on.parse("¿Para qué sirve este remedio?").rule == "pregunta"  # no es «pará»
    assert off.parse("¿Para qué sirve este remedio?").rule == "cancelar"
    assert on.parse("Pará, pará.").rule == "cancelar"
    assert on.parse("Leé todo.").target == READ_ALL
    assert (
        on.parse("¿Qué dice?").rule == "leer" and on.parse("¿Qué dice la etiqueta?").rule == "leer"
    )
    assert on.parse("¿Cuándo vence lo que dice acá?").rule == "pregunta"
    assert on.parse("¿Hay algún lugar para sentarme?").rule == "pregunta"
    assert on.parse("¿Hay una taza?").rule == "buscar"
    assert on.parse("¿Dónde está la taza?").rule == "buscar"
    assert on.parse("¿Qué hay adelante?").rule == "describir"


def test_frequent_phrases_match_what_the_guide_says():
    from visionx.shared.config import load_config
    from visionx.voice.warm import frequent_phrases

    phrases = frequent_phrases(load_config(), ClassCatalog(COCO_CLASSES).spoken)
    assert len(phrases) == len(set(phrases)) > 500
    for expected in (
        "Busco taza.",
        "Taza a la izquierda.",
        "Un poco a la derecha.",
        "Al frente, a 1,2 metros.",
        "Cancelado.",
        "Listo, ya puedo medir distancias.",
    ):
        assert expected in phrases, expected
