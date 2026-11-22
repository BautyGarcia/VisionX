"""E3 con una red de entrada simulada: qué precisión de focal hace falta (PLUG-AND-PLAY, sección 8).

Antes de tener la red de Bauty: se toma la medición por objeto con la cámara conocida
(`gtsynth evaluar --geometria vista`, un CSV por cámara) y se le suma el error que dejaría una
red que estima la focal con un desvío `σ`. Con la **separación conocida**, un error ε en la focal
multiplica toda la distancia por (1 + ε): es el mismo para todos los objetos de esa unidad, así
que se sortea **uno por cámara** y se repite muchas veces.

Mide, para cada σ: el error de distancia por objeto (mediana y P95) y la cobertura del intervalo
del 90 % de dos maneras: solo con la duda de la U-Net (lo de hoy) y sumando la duda de la
cámara como pide la sección 6, `(σ_Z/Z)² = (U-Net)² + σ_escala²`. Si la red de entrada informa
bien su σ, la segunda tiene que quedar cerca del 90 %.

    python -m visionx.gtsynth e3-simulado outputs/gtsynth/inc
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

LN10 = math.log(10.0)  # Laplace: P(|e| > b·ln 10) = 0,1
Z90 = 1.6449  # normal: P(|e| > 1,645·σ) = 0,1
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)


def leer(path: Path) -> dict[str, np.ndarray]:
    """Error relativo y semiancho relativo del 90 % de la U-Net (k = 1) de cada objeto medido."""
    e, h = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("calibra") == "True" or not r["error_rel"] or not r["b_px"]:
                continue
            d = float(r["d_frente_px"] or 0)
            if d <= 0:
                continue
            e.append(float(r["error_rel"]))
            h.append(LN10 * float(r["b_px"]) / d)  # error relativo de Z ≈ error relativo de d
    return {"e": np.array(e), "h": np.array(h)}


def simular(objs: dict[str, np.ndarray], sigma: float, k: float, sorteos: int,
            rng: np.random.Generator) -> dict[str, float]:  # fmt: skip
    """Promedio sobre `sorteos` unidades: cada una con su error de focal ε ~ N(0, σ)."""
    eps = rng.normal(0.0, sigma, sorteos) if sigma > 0 else np.zeros(1)
    tot = (1.0 + objs["e"])[None, :] * (1.0 + eps)[:, None] - 1.0
    a = np.abs(tot)
    h_unet = k * objs["h"][None, :]
    h_total = np.sqrt(h_unet**2 + (Z90 * sigma) ** 2)
    return {
        "error_mediano_pct": round(100 * float(np.median(a)), 2),
        "p95_pct": round(100 * float(np.quantile(a, 0.95)), 2),
        "cobertura_solo_unet": round(float(np.mean(a <= h_unet)), 3),
        "cobertura_con_camara": round(float(np.mean(a <= h_total)), 3),
    }


def correr(carpeta: str | Path, salida: str | Path | None = None, k: float = 1.0,
           sorteos: int = 500, semilla: int = 0) -> dict:  # fmt: skip
    carpeta = Path(carpeta)
    rng = np.random.default_rng(semilla)
    camaras = {p.stem: leer(p) for p in sorted(carpeta.glob("video-*.csv"))}
    camaras = {c: o for c, o in camaras.items() if len(o["e"])}
    out: dict = {"k": k, "sorteos": sorteos, "camaras": sorted(camaras), "por_sigma": {}}
    for s in SIGMAS:
        filas = {c: simular(o, s, k, sorteos, rng) for c, o in camaras.items()}
        claves = next(iter(filas.values())).keys()
        out["por_sigma"][f"{100 * s:g} %"] = {
            "media": {c: round(float(np.mean([f[c] for f in filas.values()])), 3) for c in claves},
            "por_camara": filas,
        }
    if salida:
        Path(salida).parent.mkdir(parents=True, exist_ok=True)
        Path(salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
