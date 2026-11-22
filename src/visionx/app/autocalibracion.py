"""Autocalibración por movimiento: medir la focal mientras la persona gira la cámara.

La red de entrada (`camara_net`) adivina la focal por cómo se ve la escena, con un error del
5–14 % que depende del cuarto (BITACORA 22/11). La geometría la puede **medir**: entre dos
cuadros del mismo ojo tomados en momentos distintos, los puntos emparejados cumplen
`x2ᵀ F x1 = 0`. Con la cámara correcta `K`, la matriz esencial `E = Kᵀ F K` tiene sus dos
valores singulares no nulos iguales; con una focal equivocada, no. La focal es la que deja
todas las `E` de un recorrido lo más «esenciales» posible (Mendonça y Cipolla, 1999).

- Sirve con movimiento general (avanzar y girar). Necesita **giro**: avanzar en línea recta
  no dice nada de la focal, y por eso se le pide a la persona que gire despacio.
- Supone píxeles cuadrados y el centro que da la red (o el centro de la imagen).
- La duda sale de remuestrear los pares (bootstrap); con la de la red se combinan pesando por
  la inversa de la varianza, en log f.

    from visionx.app.autocalibracion import focal_por_movimiento
    r = focal_por_movimiento(cuadros_izq, cx, cy, f_inicial)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

MAX_ANCHO = 960  # se trabaja a esta resolución como máximo (más rápido, misma focal relativa)
# El bootstrap sobre pares que comparten cuadros subestima la duda: en gt-video (72 recorridos)
# cubría el 56 % con ±1,645 σ; ×2 lleva al ~90 % (BITACORA 22/11).
FACTOR_SIGMA = 2.0


@dataclass(frozen=True, slots=True)
class FocalMedida:
    """Resultado de la autocalibración, en px de la imagen original."""

    fx: float | None
    sigma_log: float | None  # desvío de log f (≈ error relativo)
    pares: int  # pares de cuadros usados
    motivo: str | None = None  # por qué no se pudo medir (None si se pudo)

    @property
    def ok(self) -> bool:
        return self.fx is not None


def _gris(img: np.ndarray, q: float) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if q < 1:
        g = cv2.resize(g, (round(g.shape[1] * q), round(g.shape[0] * q)),
                       interpolation=cv2.INTER_AREA)  # fmt: skip
    return g


def fundamentales(
    cuadros: Sequence[np.ndarray],
    saltos: Sequence[int] = (3, 6),
    min_inliers: int = 60,
    min_movimiento_px: float = 6.0,
) -> tuple[list[np.ndarray], list[int], float]:
    """Matrices F entre cuadros separados `saltos`, sus inliers y la escala de trabajo."""
    q = min(1.0, MAX_ANCHO / cuadros[0].shape[1])
    sift = cv2.SIFT_create(nfeatures=3000)
    rasgos = [sift.detectAndCompute(_gris(c, q), None) for c in cuadros]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fs, ns = [], []
    for s in saltos:
        for i in range(len(cuadros) - s):
            (k1, d1), (k2, d2) = rasgos[i], rasgos[i + s]
            if d1 is None or d2 is None or len(k1) < min_inliers or len(k2) < min_inliers:
                continue
            buenos = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2)
                      if m.distance < 0.75 * n.distance]  # fmt: skip
            if len(buenos) < min_inliers:
                continue
            p1 = np.float32([k1[m.queryIdx].pt for m in buenos])
            p2 = np.float32([k2[m.trainIdx].pt for m in buenos])
            if float(np.median(np.linalg.norm(p1 - p2, axis=1))) < min_movimiento_px:
                continue  # casi quieta: no informa
            f, m = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
            if f is None or f.shape != (3, 3) or m is None or int(m.sum()) < min_inliers:
                continue
            fs.append(f / np.linalg.norm(f))
            ns.append(int(m.sum()))
    return fs, ns, q


def costo(fx: float, cx: float, cy: float, fs: Sequence[np.ndarray],
          pesos: np.ndarray | None = None) -> float:  # fmt: skip
    """Cuánto se alejan las E = Kᵀ F K de ser esenciales: media de (σ1 − σ2) / σ1."""
    k = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
    c = np.empty(len(fs))
    for j, f in enumerate(fs):
        s = np.linalg.svd(k.T @ f @ k, compute_uv=False)
        c[j] = (s[0] - s[1]) / max(s[0], 1e-12)
    w = np.ones(len(fs)) if pesos is None else pesos
    return float((w * c).sum() / w.sum())


def _mejor_focal(cx: float, cy: float, fs: Sequence[np.ndarray], f0: float,
                 rango: float, pesos: np.ndarray | None = None) -> float:  # fmt: skip
    """Grilla en log f alrededor de `f0` (× 1/rango … × rango) y refinamiento fino."""
    grilla = f0 * np.exp(np.linspace(-math.log(rango), math.log(rango), 121))
    cs = [costo(f, cx, cy, fs, pesos) for f in grilla]
    j = int(np.argmin(cs))
    lo, hi = grilla[max(j - 1, 0)], grilla[min(j + 1, len(grilla) - 1)]
    fino = np.exp(np.linspace(math.log(lo), math.log(hi), 41))
    cs = [costo(f, cx, cy, fs, pesos) for f in fino]
    return float(fino[int(np.argmin(cs))])


def focal_por_movimiento(
    cuadros: Sequence[np.ndarray],
    cx: float,
    cy: float,
    f_inicial: float,
    saltos: Sequence[int] = (3, 6),
    rango: float = 2.0,
    min_pares: int = 6,
    remuestreos: int = 30,
    semilla: int = 0,
) -> FocalMedida:
    """Focal (px de la imagen original) que mejor explica el movimiento entre los cuadros.

    `cuadros`: imágenes de un mismo ojo en orden temporal. `cx, cy`: centro supuesto (px de la
    imagen original). `f_inicial`: punto de partida (la red); se busca entre ×1/rango y ×rango.
    """
    if len(cuadros) < max(saltos) + 2:
        return FocalMedida(None, None, 0, "muy pocos cuadros")
    fs, ns, q = fundamentales(cuadros, saltos)
    if len(fs) < min_pares:
        return FocalMedida(None, None, len(fs), "poco movimiento o poca textura")
    cxq, cyq, f0q = cx * q, cy * q, f_inicial * q
    pesos = np.asarray(ns, float)
    f = _mejor_focal(cxq, cyq, fs, f0q, rango, pesos)
    rng = np.random.default_rng(semilla)
    logs = []
    for _ in range(remuestreos):
        idx = rng.integers(0, len(fs), len(fs))
        logs.append(math.log(_mejor_focal(cxq, cyq, [fs[i] for i in idx], f, 1.25, pesos[idx])))
    sigma = FACTOR_SIGMA * float(np.std(logs)) if logs else None
    if abs(math.log(f / f0q)) > 0.98 * math.log(rango):
        return FocalMedida(None, sigma, len(fs), "la focal cae en el borde de la búsqueda")
    return FocalMedida(f / q, sigma, len(fs))


def combinar(log_f_red: float, sigma_red: float, medida: FocalMedida) -> tuple[float, float]:
    """Red y movimiento en una sola focal (log f) con su desvío: inversa de la varianza."""
    if not medida.ok or not medida.sigma_log:
        return log_f_red, sigma_red
    s_m = max(medida.sigma_log, 1e-3)
    w_r, w_m = 1 / sigma_red**2, 1 / s_m**2
    lf = (w_r * log_f_red + w_m * math.log(medida.fx)) / (w_r + w_m)
    return lf, 1 / math.sqrt(w_r + w_m)
