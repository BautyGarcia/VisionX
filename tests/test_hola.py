"""«Hola VisionX», cómo habla (velocidad y largo) y la alerta de obstáculos mientras guía."""

from __future__ import annotations

import numpy as np
import pytest

from visionx.app.controller import Controller
from visionx.app.onboarding import apply_preference, intro_phrases
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes import body
from visionx.voice.commands import DETAIL, INTRO, VOICE, CommandParser
from visionx.voice.output import AudioOutput, FakePlayer

PARSER = CommandParser(ClassCatalog(COCO_CLASSES), questions=True)


@pytest.fixture(autouse=True)
def _persona(tmp_path, monkeypatch):
    monkeypatch.setattr(body.Path, "home", lambda: tmp_path)
    body.reset()
    yield
    body.reset()


@pytest.mark.parametrize("frase", ["Hola VisionX", "hola", "Hola, Visión X.", "¿Qué podés hacer?",
                                   "Ayuda", "Buenas, visionx"])  # fmt: skip
def test_hola(frase):
    assert PARSER.parse(frase).target == INTRO


def test_hola_con_una_orden_es_la_orden():
    i = PARSER.parse("Hola VisionX, buscá la taza")
    assert i.target == "cup"


@pytest.mark.parametrize(
    "frase,esperado",
    [("Hablá más rápido", f"{VOICE}:rapido"), ("Más despacio, por favor", f"{VOICE}:lento"),
     ("Velocidad normal", f"{VOICE}:normal"), ("Respuestas más cortas", f"{DETAIL}:corto"),
     ("Respondé más corto", f"{DETAIL}:corto"), ("Más detallado", f"{DETAIL}:largo"),
     ("Respuestas normales", f"{DETAIL}:normal")],
)  # fmt: skip
def test_preferencias(frase, esperado):
    assert PARSER.parse(frase).target == esperado


def test_preguntas_no_son_preferencias():
    assert PARSER.parse("¿Cuál está más lejos, la taza o el vaso?").target not in (
        f"{DETAIL}:largo", f"{VOICE}:lento"
    )  # fmt: skip
    assert not str(PARSER.parse("Contame más sobre la caja").target).startswith(DETAIL)


class _Synth:
    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(len(text), np.float32)


def _ctl():
    out = AudioOutput(_Synth(), FakePlayer())
    ctl = Controller(PARSER, out, intro=lambda: intro_phrases(body.current(), {}))
    return ctl, out


def _dicho(out):
    out.wait_idle(2)
    return [r.text_or_sound for r in out.records]


def test_presentacion_pide_la_altura_y_entiende_un_numero_suelto():
    ctl, out = _ctl()
    ctl.handle_text("Hola VisionX")
    dicho = _dicho(out)
    assert dicho[0].startswith("Hola, soy VisionX") and dicho[-1].endswith("¿cuánto medís?")
    ctl.handle_text("1,75")
    assert body.current().height_m == 1.75 and _dicho(out)[-1].startswith("Listo, medís 1,75")
    out.close()


def test_presentacion_con_altura_guardada_no_espera_un_numero():
    body.set_height(1.80)
    ctl, out = _ctl()
    ctl.handle_text("Hola VisionX")
    assert _dicho(out)[-1].startswith("Tengo guardado que medís 1,8 metros")
    assert ctl._awaiting_height_ns is None
    out.close()


def test_velocidad_cambia_la_voz_y_se_guarda(tmp_path):
    from visionx.voice.engines import DiskCachedSynth, EdgeSynth, VoiceSettings
    from visionx.voice.output import CachedSynth

    edge = EdgeSynth(VoiceSettings(speech_rate=1.15), fetch=lambda *a: b"")
    synth = CachedSynth(DiskCachedSynth(edge, tmp_path / "voz"))
    assert apply_preference("voz:rapido", synth, None, 1.15) == [
        "Listo, hablo más rápido. ¿Así está bien?"
    ]
    assert edge.rate == "+30%" and "30" in synth.synth.folder.parent.name
    assert body.preferences()["velocidad"] == 1.3
    apply_preference("voz:rapido", synth, None, 1.15)
    assert apply_preference("voz:rapido", synth, None, 1.15)[0].startswith(
        "Ya estoy hablando lo más"
    )
    apply_preference("voz:normal", synth, None, 1.15)
    assert edge.rate == "+15%"


def test_largo_de_las_respuestas():
    from types import SimpleNamespace

    from visionx.assistant.core import system_prompt

    a = SimpleNamespace(detail="normal")
    assert apply_preference("detalle:corto", None, a, 1.15)[0].endswith("más cortas.")
    assert a.detail == "corto" and body.preferences()["detalle"] == "corto"
    assert "UNA sola frase" in system_prompt("corto") and "hasta 5" in system_prompt("largo")
    assert system_prompt("normal").endswith("pedí que lo repita.")


def test_alerta_mientras_guia_ignora_al_objetivo():
    from visionx.app.proximity import ProximityAlert, ProximitySettings

    alert = ProximityAlert(ProximitySettings())
    depth = np.full((90, 160), 3.0, np.float32)
    depth[20:60, 60:100] = 0.5  # algo cerca al centro
    frame = None
    assert alert.check(depth, frame)  # sin excluir nada: obstáculo
    exclude = np.zeros_like(depth, bool)
    exclude[18:62, 58:102] = True  # es el objetivo que se está buscando
    assert not alert.check(depth, frame, exclude)


def test_alerta_solo_mientras_busca():
    from types import SimpleNamespace

    from visionx.app.proximity import ProximityAlert, ProximitySettings

    alert = ProximityAlert(ProximitySettings())
    alert.active_mode = None
    assert alert.observe(SimpleNamespace(measured=None), 0) == []  # sin buscar: callada
