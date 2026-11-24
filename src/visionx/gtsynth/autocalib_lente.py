"""Autocalibración con lentes: los recorridos de `gt-video` vistos a través de una lente sorteada.

Los renders son pinhole con lienzo de margen, así que se puede simular una lente real
(`camara_net.lente.mapa_lente`, el mismo remuestreo que usa la red de entrada al entrenar). Por
recorrido se elige una k1 de `K1S` (determinística, la más fuerte que entra en el lienzo sin
bordes negros) y se compara:

    pinhole   `focal_por_movimiento` (supone que no hay lente: lo de antes)
    lente     `lente_por_movimiento` (busca focal y k1 juntas)

Las dos parten de lo que dice la red de entrada mirando los cuadros **distorsionados** (como
pasaría con una cámara real).

    python -m visionx.gtsynth autocalib-lente data/gt-video
"""

from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

K1S = (-0.30, -0.20, -0.12, -0.05, 0.0)


def elegir_k1(v0) -> float:
    """La k1 del recorrido: una de `K1S` según la escena, la más fuerte que entra en el lienzo."""
    from visionx.depth.camara_net.lente import Camara, cabe

    c, n = v0.meta["camara"], v0.nominal
    pedida = K1S[zlib.crc32(f"{v0.variante}/{v0.escena}".encode()) % len(K1S)]
    for k1 in [k for k in K1S if k >= pedida]:
        cam = Camara(n["ancho"], n["alto"], c["fx"], n["cx"], n["cy"], k1, 0.0)
        if cabe(cam, (c["ancho"], c["alto"]), (c["cx"], c["cy"])):
            return k1
    return 0.0


def distorsionar_recorrido(vs, k1: float) -> tuple[list, list, dict]:
    """Ojos izquierdo y derecho vistos con la lente k1 (tamaño nominal) y la cámara real."""
    from visionx.depth.camara_net.lente import Camara, mapa_lente

    c, n = vs[0].meta["camara"], vs[0].nominal
    cam = Camara(n["ancho"], n["alto"], c["fx"], n["cx"], n["cy"], k1, 0.0)
    mx, my = mapa_lente(cam, (cam.ancho, cam.alto), (c["cx"], c["cy"]), 1.0)
    izq = [cv2.remap(v.izq(), mx, my, cv2.INTER_LINEAR) for v in vs]
    der = [cv2.remap(v.der(), mx, my, cv2.INTER_LINEAR) for v in vs]
    return izq, der, {"fx": c["fx"], "cx": n["cx"], "cy": n["cy"], "ancho": n["ancho"],
                      "alto": n["alto"], "k1": k1}  # fmt: skip


def evaluar(vs, estimador, cuadros: int = 60) -> dict:
    from visionx.app.autocalibracion import focal_por_movimiento, lente_por_movimiento

    vs = vs[:cuadros]
    k1 = elegir_k1(vs[0])
    izq, der, real = distorsionar_recorrido(vs, k1)
    est = estimador.estimar([(izq[i], der[i]) for i in range(min(8, len(izq)))])
    w, h = real["ancho"], real["alto"]
    f0 = (w / 2) / math.tan(math.radians(est.campo_horizontal_deg) / 2)
    cx = (0.5 + est.centro_rel[0]) * w - 0.5
    cy = (0.5 + est.centro_rel[1]) * h - 0.5
    err = lambda f: None if f is None else round(100 * (f / real["fx"] - 1), 2)  # noqa: E731
    out = {"variante": vs[0].variante, "escena": vs[0].escena, "k1_real": k1,
           "red_err_pct": err(f0), "red_k1": round(est.k1, 3)}  # fmt: skip
    t = time.time()
    a = focal_por_movimiento(izq, cx, cy, f0, saltos=(3, 6, 12))
    out.update({"pinhole_err_pct": err(a.fx), "pinhole_sigma_pct": a.sigma_log and
                round(100 * a.sigma_log, 2), "pinhole_s": round(time.time() - t, 1)})  # fmt: skip
    t = time.time()
    b = lente_por_movimiento(izq, cx, cy, f0)
    out.update({"lente_err_pct": err(b.fx), "lente_sigma_pct": b.sigma_log and
                round(100 * b.sigma_log, 2), "lente_k1": b.k1 and round(b.k1, 3),
                "lente_motivo": b.motivo, "lente_s": round(time.time() - t, 1)})  # fmt: skip
    return out


def resumen(filas: list[dict]) -> dict:
    out = {}
    for k in ("red", "pinhole", "lente"):
        e = np.array([abs(f[f"{k}_err_pct"]) for f in filas if f.get(f"{k}_err_pct") is not None])
        if len(e):
            out[k] = {"n": len(e), "mediana_pct": round(float(np.median(e)), 2),
                      "p90_pct": round(float(np.percentile(e, 90)), 2)}  # fmt: skip
    dk = [abs(f["lente_k1"] - f["k1_real"]) for f in filas if f.get("lente_k1") is not None]
    if dk:
        out["lente_k1_err_mediana"] = round(float(np.median(dk)), 3)
    for nombre, cond in (("con_lente", lambda f: f["k1_real"] <= -0.1),
                         ("sin_lente", lambda f: f["k1_real"] == 0.0)):  # fmt: skip
        sub = [f for f in filas if cond(f)]
        for k in ("pinhole", "lente"):
            e = [abs(f[f"{k}_err_pct"]) for f in sub if f.get(f"{k}_err_pct") is not None]
            if e:
                out[f"{nombre}_{k}_mediana_pct"] = round(float(np.median(e)), 2)
    return out


def correr_todo(raiz, salida, red: str, variantes=None, cuadros: int = 60) -> dict:
    from visionx.app.estimador_red import EstimadorRed
    from visionx.gtsynth.dataset import vistas

    est = EstimadorRed(red)
    raiz = Path(raiz)
    variantes = variantes or sorted(p.name for p in raiz.iterdir()
                                    if p.is_dir() and p.name != "video-elp")  # fmt: skip
    filas = []
    for var in variantes:
        por: dict[str, list] = {}
        for v in vistas(raiz, var):
            por.setdefault(v.escena, []).append(v)
        for esc in sorted(por):
            vs = sorted(por[esc], key=lambda v: v.meta["indice"])
            filas.append(evaluar(vs, est, cuadros))
            print(json.dumps(filas[-1], ensure_ascii=False), flush=True)
    r = {"cuadros": cuadros, "resumen": resumen(filas), "filas": filas}
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    Path(salida).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r
