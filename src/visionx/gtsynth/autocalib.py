"""Evaluar la autocalibración por movimiento en `gt-video` contra la cámara real.

Por recorrido (una cámara, una escena): la red de entrada mira los primeros cuadros
(`sin_calibrar.camara_de_la_red`, CSV de `camara_net evaluar`) y da focal, centro y su b; el
movimiento de los primeros `cuadros` del ojo izquierdo mide la focal partiendo de ahí
(`app.autocalibracion`), y las dos se combinan. Sin CSV de la red, arranca de 90° y del centro
de la imagen (el valor por defecto de la pantalla «Cámara nueva»).

    python -m visionx.gtsynth autocalib data/gt-video --red-camara outputs/camara_net/e2-v2/eval-pc
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from visionx.app.autocalibracion import combinar, focal_por_movimiento
from visionx.gtsynth.dataset import VistaNominal, vistas
from visionx.gtsynth.sin_calibrar import camara_de_la_red, leer_red

# 1,79 es el de la v2 para cubrir el 90 % en val; fuera de val (gt-video) la red cubre el 42 %
# y hace falta el doble para combinarla con el movimiento sin que la arrastre (BITACORA 22/11).
FACTOR_B_RED = 1.79 * 2


def evaluar(vs, red: dict | None, cuadros: int = 60, saltos=(3, 6, 12)) -> dict:
    vn = [VistaNominal(v.carpeta) for v in vs[:cuadros]]
    g = vn[0].geometria
    cam = camara_de_la_red(vs, red) if red else None
    if cam is not None:
        q = g.width / cam["ancho"]
        f0, cx, cy = cam["fx"] * q, cam["cx"] * q, cam["cy"] * q
        s_red = math.sqrt(2) * cam["b_log_f"] * FACTOR_B_RED  # Laplace: σ = √2·b
    else:
        f0 = (g.width / 2) / math.tan(math.radians(45))
        cx, cy, s_red = g.width / 2, g.height / 2, 0.3
    t = time.time()
    m = focal_por_movimiento([v.izq() for v in vn], cx, cy, f0, saltos=saltos)
    lf, s = combinar(math.log(f0), s_red, m)
    err = lambda f: None if f is None else round(100 * (f / g.fx - 1), 2)  # noqa: E731
    return {
        "escena": vs[0].escena, "variante": vs[0].variante, "fx_real": round(g.fx, 1),
        "red_err_pct": err(f0), "red_sigma_pct": round(100 * s_red, 2),
        "mov_err_pct": err(m.fx), "mov_sigma_pct": m.sigma_log and round(100 * m.sigma_log, 2),
        "mov_pares": m.pares, "mov_motivo": m.motivo,
        "comb_err_pct": err(math.exp(lf)), "comb_sigma_pct": round(100 * s, 2),
        "cx_err_px": round(cx - g.cx, 1), "segundos": round(time.time() - t, 1),
    }  # fmt: skip


def resumen(filas: list[dict]) -> dict:
    out = {}
    for k in ("red", "mov", "comb"):
        e = np.array([abs(f[f"{k}_err_pct"]) for f in filas if f[f"{k}_err_pct"] is not None])
        s = np.array([f[f"{k}_sigma_pct"] for f in filas if f[f"{k}_err_pct"] is not None])
        cob = float(np.mean(e <= 1.645 * s)) if len(e) else None
        out[k] = {"n": len(e), "mediana_pct": round(float(np.median(e)), 2) if len(e) else None,
                  "p90_pct": round(float(np.percentile(e, 90)), 2) if len(e) else None,
                  "cobertura_90": cob and round(cob, 3)}  # fmt: skip
    return out


def correr_todo(raiz, salida, red: str | None = None, variantes=None, cuadros: int = 60,
                saltos=(3, 6, 12)) -> dict:  # fmt: skip
    raiz = Path(raiz)
    est = leer_red(red) if red else None
    variantes = variantes or sorted(p.name for p in raiz.iterdir()
                                    if p.is_dir() and p.name != "video-elp")  # fmt: skip
    filas = []
    for var in variantes:
        por: dict[str, list] = {}
        for v in vistas(raiz, var):
            por.setdefault(v.escena, []).append(v)
        for esc in sorted(por):
            vs = sorted(por[esc], key=lambda v: v.meta["indice"])
            filas.append(evaluar(vs, est, cuadros, saltos))
            print(json.dumps(filas[-1], ensure_ascii=False), flush=True)
    r = {"cuadros": cuadros, "saltos": list(saltos), "resumen": resumen(filas), "filas": filas}
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    Path(salida).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r
