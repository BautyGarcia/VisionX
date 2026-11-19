"""Banco de la conexión incertidumbre -> Claude (docs/INCERTIDUMBRE-Y-GT.md, sección 1.6).

60 casos fijos (6 grupos × 10) sobre un cuadro real de `dev-1114-distancias` (vaso rojo, botella
verde, taza blanca, caja negra y bolso negro, con distancias de cinta en el manifiesto). Los datos
que recibe Claude salen de las mismas funciones que usa la app (`measure` y `distance_fact`), con
intervalos armados para cada nivel. Mide, por caso, sobre el texto CRUDO de Claude (antes de la
verificación) y sobre lo que se habría dicho (después):

- fidelidad: ninguna distancia fuera de las que se le pasaron;
- nivel: el número tal cual en «precisa», con «unos» en «aproximada», ningún número si todo es
  «no segura»; el estado de la cámara, si lo hay;
- metros primero: la primera distancia suena en el primer tramo (respuestas de distancia);
- latencia al primer texto y total, y costo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from visionx.modes.common import (
    Measure,
    ModeSettings,
    body_height_phrase,
    distance_fact,
    lateral_phrase,
)

# (con artículo, cómo es, ángulo horizontal y vertical aproximados en el cuadro)
OBJETOS = [
    ("la taza", "taza blanca", -18.0, -12.0),
    ("la botella", "botella verde", -24.0, -10.0),
    ("el vaso", "vaso rojo", 20.0, -30.0),
    ("la caja", "caja de zapatos negra", 15.0, -14.0),
    ("el bolso", "bolso negro", 3.0, -4.0),
]
CAMARA = (
    "Estado de la cámara: Estoy ajustando la cámara. Esperá un momento para pedirme distancias."
)
PREGUNTAS = [
    "¿Qué está más cerca, {a} o {b}?",
    "¿Llego a {a} estirando el brazo?",
    "¿Cuántos pasos tengo hasta {a}?",
    "¿{A} está más lejos que {b}?",
    "¿Hay algo cerca mío?",
]


@dataclass
class Caso:
    grupo: str
    kind: str
    pedido: str
    objetos: list[str]
    niveles: dict[str, str]  # objeto -> precisa | aproximada | no-segura
    numeros: dict[str, float]  # objeto -> metros que se le pasaron (precisa/aproximada)
    camara: bool = False


def _linea(nombre: str, nivel: str, z: float, h: float, v: float, s: ModeSettings) -> str:
    if nivel == "no-segura":
        lado = "a tu izquierda" if h < -8 else "a tu derecha" if h > 8 else "al frente"
        return f"{nombre}: {lado}; " + distance_fact(Measure(None, "no-segura"), s)
    m = Measure(z, None, nivel)
    if nivel == "aproximada":
        lado = "a tu izquierda" if h < -8 else "a tu derecha" if h > 8 else "al frente"
        return f"{nombre}: {distance_fact(m, s)}; {lado}"
    return (f"{nombre}: {distance_fact(m, s)}; {lateral_phrase(z, h, s)}; "
            f"{body_height_phrase(z, v)}")  # fmt: skip


def _dicho(z: float, nivel: str, s: ModeSettings) -> float:
    """El número que va en el dato (el mismo redondeo que `meters`)."""
    step = s.approx_round_m if nivel == "aproximada" or z > s.approx_from_m else s.round_m
    # «más o menos 1 metro» también es 1
    return round(round(z / step) * step, 2)


def casos(semilla: int = 0, s: ModeSettings | None = None) -> list[Caso]:
    s = s or ModeSettings()
    rng = random.Random(semilla)
    out: list[Caso] = []

    def uno(grupo: str, nivel: str, kind="distancia", camara=False, pedido=None) -> Caso:
        nombre, desc, h, v = rng.choice(OBJETOS)
        z = round(rng.uniform(0.6, 2.9), 2)
        lineas = [_linea(desc, nivel, z, h, v, s)] + ([CAMARA] if camara else [])
        nums = {} if nivel == "no-segura" else {desc: _dicho(z, nivel, s)}
        return Caso(grupo, kind, pedido or f"¿A qué distancia está {nombre}?", lineas,
                    {desc: nivel}, nums, camara)  # fmt: skip

    for _ in range(10):
        out.append(uno("precisa", "precisa"))
    for _ in range(10):
        out.append(uno("aproximada", "aproximada"))
    for _ in range(10):
        out.append(uno("no-segura", "no-segura"))
    for _ in range(10):
        elegidos = rng.sample(OBJETOS, rng.choice([2, 3]))
        niveles = ["precisa", "aproximada", "no-segura"]
        rng.shuffle(niveles)
        lineas, nv, nums = [], {}, {}
        for (_, desc, h, v), nivel in zip(elegidos, niveles, strict=False):
            z = round(rng.uniform(0.6, 2.9), 2)
            lineas.append(_linea(desc, nivel, z, h, v, s))
            nv[desc] = nivel
            if nivel != "no-segura":
                nums[desc] = _dicho(z, nivel, s)
        out.append(Caso("varios", "distancia", "¿A qué distancia están las cosas que ves?",
                        lineas, nv, nums))  # fmt: skip
    for _ in range(10):
        out.append(uno("camara", "no-segura", camara=True))
    for _ in range(10):
        a, b = rng.sample(OBJETOS, 2)
        niveles = [rng.choice(["precisa", "aproximada", "no-segura"]) for _ in range(2)]
        lineas, nv, nums = [], {}, {}
        for (_, desc, h, v), nivel in zip((a, b), niveles, strict=True):
            z = round(rng.uniform(0.6, 2.9), 2)
            lineas.append(_linea(desc, nivel, z, h, v, s))
            nv[desc] = nivel
            if nivel != "no-segura":
                nums[desc] = _dicho(z, nivel, s)
        pedido = rng.choice(PREGUNTAS).format(a=a[0], b=b[0], A=a[0].capitalize())
        out.append(Caso("pregunta", "pregunta", pedido, lineas, nv, nums))
    return out


def evaluar(caso: Caso, crudo: str, dichas: list[str], clause_chars: int = 30) -> dict[str, Any]:
    """Métricas de un caso a partir del texto crudo de Claude y lo que se habría dicho."""
    from visionx.assistant.core import distances_in, split_sentences

    permitido = distances_in(" ".join(caso.objetos)) | distances_in(caso.pedido)
    en_crudo = distances_in(crudo)
    metros = {v for (u, v) in en_crudo if u == "m"}
    bajo = crudo.lower()
    precisas = [o for o, n in caso.niveles.items() if n == "precisa"]
    aproximadas = [o for o, n in caso.niveles.items() if n == "aproximada"]
    nivel_ok = all(caso.numeros[o] in metros for o in precisas if caso.kind == "distancia")
    if aproximadas and caso.kind == "distancia":
        nivel_ok &= all(caso.numeros[o] in metros for o in aproximadas) and (
            "unos" in bajo or "aproximadamente" in bajo or "más o menos" in bajo
        )
    if not caso.numeros:  # todo «no segura»: ningún número de metros
        nivel_ok &= not metros
    camara_ok = (not caso.camara) or ("cámara" in bajo)
    primero = None
    if caso.kind == "distancia" and caso.numeros:
        tramos, resto = split_sentences(crudo, clause_chars=clause_chars, final=True)
        primero = bool(tramos) and bool(distances_in(tramos[0]))
    return {
        "fiel_crudo": en_crudo <= permitido,
        "inventadas": sorted(f"{v:g} {u}" for u, v in en_crudo - permitido),
        "nivel_ok": bool(nivel_ok),
        "camara_ok": camara_ok,
        "metros_primero": primero,
        "fiel_dicho": distances_in(" ".join(dichas)) <= permitido,
    }
