"""La incertidumbre hasta la voz (docs/INCERTIDUMBRE-Y-GT.md, sección 1).

El código decide qué tan segura es una distancia (precisa / aproximada / no segura); Claude solo
la dice, y lo que dice se verifica: un número que no medimos no suena.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from visionx.assistant import Assistant, Delivery
from visionx.assistant.core import distances_in, verify_distances
from visionx.modes.common import (
    Measure,
    ModeSettings,
    distance_fact,
    measure,
    measure_phrase,
)
from visionx.shared.contracts import DistancePolicy, SpatialObject
from visionx.shared.errors import InvalidInput
from visionx.shared.uncertainty import LN10, laplace_interval, relative_width


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
        return SimpleNamespace(
            stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1500, output_tokens=100)
        )


class FakeClient:
    """Cliente de Claude falso: sin red ni gasto."""

    def __init__(self, chunks):
        self.calls = []
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return _Stream(chunks)

        self.messages = Messages()
        self.beta = SimpleNamespace(messages=Messages())


def settings(**kw):
    from visionx.assistant import AssistantSettings

    return AssistantSettings(ledger="", **kw)


def wait(job, timeout=5.0):
    assert job.done.wait(timeout)
    return job


S = ModeSettings()
IMG = np.zeros((72, 128, 3), np.uint8)


def obj(z=1.4, bajo=None, alto=None, policy=DistancePolicy.SPEAK_METERS):
    return SpatialObject(
        session_id="s", frame_id=1, detection_index=0, track_id=None, label="cup",
        depth_model_id="m", angle_h_deg=0.0, angle_v_deg=0.0, z_m=z, z_spread_m=0.0,
        valid_fraction=1.0, detector_score=0.9, policy=policy, z_bajo_m=bajo, z_alto_m=alto,
    )  # fmt: skip


# --- intervalo --------------------------------------------------------------------------------


def test_intervalo_de_laplace():
    fb = 400.0 * 0.06  # fx·B
    d = fb / 2.0  # objeto a 2 m
    bajo, alto = laplace_interval(d, 0.5, fb)
    assert bajo == pytest.approx(fb / (d + LN10 * 0.5)) and alto == pytest.approx(
        fb / (d - LN10 * 0.5)
    )
    assert bajo < 2.0 < alto
    assert laplace_interval(1.0, 1.0, fb)[1] is None  # cruza disparidad 0: sin cota superior
    assert relative_width(2.0, 1.8, 2.2) == pytest.approx(0.1)
    assert relative_width(2.0, 1.8, None) == math.inf


def test_contrato_valida_el_intervalo():
    obj(1.4, 1.3, 1.5)
    obj(1.4, 1.3, None)  # sin cota superior
    with pytest.raises(InvalidInput):
        obj(1.4, None, 1.5)
    with pytest.raises(InvalidInput):
        obj(1.4, 1.5, 1.6)  # z fuera del intervalo


# --- niveles ----------------------------------------------------------------------------------


def test_niveles_de_certeza():
    assert measure(obj(1.4), S) == Measure(1.4, None, "precisa")  # sin intervalo: como antes
    assert measure(obj(1.4, 1.32, 1.48), S).certainty == "precisa"  # ±5,7 %
    m = measure(obj(2.3, 1.9, 2.75), S)  # ±18 %
    assert m.ok and m.approximate
    assert measure_phrase(m, S) == "a unos 2,5 metros"
    assert measure(obj(2.3, 1.6, 3.4), S) == Measure(None, "no-segura")  # ±39 %
    assert measure(obj(2.3, 1.6, None), S) == Measure(None, "no-segura")
    assert measure_phrase(Measure(None, "no-segura"), S).endswith(
        "no puedo medir bien a qué distancia está"
    )


def test_dato_para_claude_con_su_nivel():
    assert distance_fact(measure(obj(1.4, 1.32, 1.48), S), S) == "a 1,4 metros (medida precisa)"
    assert distance_fact(measure(obj(2.3, 1.9, 2.75), S), S) == (
        "a unos 2,5 metros (medida aproximada: decilo con «unos»)"
    )
    assert distance_fact(measure(obj(2.3, 1.6, 3.4), S), S) == (
        "la distancia no se pudo medir bien (no digas ningún número)"
    )
    assert distance_fact(Measure(None, "sin-profundidad"), S) == ""


# --- verificación ---------------------------------------------------------------------------


def test_distancias_en_el_texto():
    assert distances_in("La taza está a 1,4 metros, unos 20 cm a tu izquierda.") == {
        ("m", 1.4), ("cm", 20.0)
    }  # fmt: skip
    assert distances_in("Está a dos metros y medio, a un metro de la silla.") == {
        ("m", 2.5), ("m", 1.0)
    }  # fmt: skip
    assert distances_in("Hay 3 tazas en la mesa.") == set()  # sin unidad: no es distancia


def test_verificar_distancias():
    allowed = distances_in(
        "taza: a 1,4 metros (medida precisa); unos 20 centímetros a tu izquierda"
    )
    assert verify_distances("La taza blanca está a 1,4 metros, unos 20 cm a tu izquierda.", allowed)
    assert not verify_distances("La taza está a 1,43 metros.", allowed)  # afinada
    assert not verify_distances("La silla está a 3 metros.", allowed)  # inventada
    assert verify_distances("La silla está a tu derecha.", allowed)


def test_el_asistente_no_dice_un_numero_que_no_medimos():
    client = FakeClient(["La taza está a 1,4 metros. ", "La silla está a 3 metros. ", "Listo."])
    a = Assistant(settings(), client=client)
    job = wait(a.submit(1, "distancia", IMG, request="¿A qué distancia?",
                        objects=["taza: a 1,4 metros (medida precisa)",
                                 "silla: a tu derecha; la distancia no se pudo medir bien "
                                 "(no digas ningún número)"]))  # fmt: skip
    dichas = [s for _, s in job.sentences]
    assert dichas == ["La taza está a 1,4 metros.", "Listo."]
    assert job.dropped == ["La silla está a 3 metros."]
    prompt = client.calls[0]["system"]
    assert "medida aproximada" in prompt and "no se pudo medir bien" in prompt


def test_si_todo_se_descarta_el_modo_usa_su_respaldo():
    client = FakeClient(["La taza está a 2 metros."])
    a = Assistant(settings(), client=client)
    job = wait(a.submit(1, "distancia", IMG, objects=["taza: a 1,4 metros (medida precisa)"]))
    d = Delivery(job, 0)
    assert d.ready(10**12) == [] and d.failed  # nada verificado: respaldo local


def test_sin_objetos_no_se_verifica():
    client = FakeClient(["Adelante hay una mesa a un metro de la pared."])
    a = Assistant(settings(), client=client)
    job = wait(a.submit(1, "describir", IMG))
    assert [s for _, s in job.sentences] == ["Adelante hay una mesa a un metro de la pared."]


def test_frases_corregidas():
    from visionx.modes.common import lateral_phrase, meters

    assert meters(1.1, S, force=True) == "más o menos 1 metro"  # no «unos 1 metros»
    assert lateral_phrase(1.0, 44.6, S) == "1 metro a tu derecha"  # no «unos 100 centímetros»


def test_banco_60_casos_y_metricas():
    from visionx.assistant.banco import casos, evaluar

    cs = casos(0)
    assert len(cs) == 60
    assert {c.grupo for c in cs} == {"precisa", "aproximada", "no-segura", "varios", "camara",
                                     "pregunta"}  # fmt: skip
    c = next(c for c in cs if c.grupo == "precisa")
    (z,) = c.numeros.values()
    bien = f"La caja está a {z:g} metros.".replace(".", ",", 1)
    assert evaluar(c, bien, [bien])["nivel_ok"]
    malo = evaluar(c, "Está a 9 metros.", [])
    assert not malo["fiel_crudo"] and malo["inventadas"] == ["9 m"]
    nc = next(c for c in cs if c.grupo == "camara")
    r = evaluar(nc, "Estoy ajustando la cámara. La taza está a tu izquierda.", ["x"])
    assert r["nivel_ok"] and r["camara_ok"]


def test_aproximada_de_1_metro_pide_mas_o_menos():
    m = measure(obj(1.05, 0.9, 1.2), S)  # ±14 %
    assert (
        distance_fact(m, S) == "a más o menos 1 metro (medida aproximada: decilo con «más o menos»)"
    )
