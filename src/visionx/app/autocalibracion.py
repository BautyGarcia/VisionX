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
    k1: float | None = None  # distorsión radial medida (`lente_por_movimiento`); None = pinhole

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


# ---------------------------------------------------------------- con distorsión (lente)
#
# Con una lente gran angular las rectas se curvan y la geometría de pinhole deja de valer: la F
# que ajusta RANSAC sobre los píxeles crudos sale sesgada y la focal también. Para medir la lente
# se buscan a la vez la focal y k1 (modelo radial de OpenCV, k2 = 0): con cada par (f, k1) los
# puntos emparejados se llevan a coordenadas normalizadas sin distorsión, se estima la E de cada
# par de cuadros por mínimos cuadrados (8 puntos, lineal) y se mide cuán esencial es. La
# respuesta es el (f, k1) que deja todas las E más esenciales.

K1_RANGO = (-0.45, 0.15)


def emparejar(
    cuadros: Sequence[np.ndarray],
    saltos: Sequence[int] = (3, 6, 12),
    min_inliers: int = 60,
    min_movimiento_px: float = 6.0,
    ransac_px: float = 3.0,
    max_puntos: int = 300,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    """Puntos emparejados (inliers de RANSAC, a la escala de trabajo) entre cuadros separados.

    El umbral de RANSAC es más ancho que en `fundamentales`: con distorsión, los puntos del borde
    no cumplen la geometría de pinhole y se descartarían justo donde la lente se nota.
    """
    q = min(1.0, MAX_ANCHO / cuadros[0].shape[1])
    sift = cv2.SIFT_create(nfeatures=3000)
    rasgos = [sift.detectAndCompute(_gris(c, q), None) for c in cuadros]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    pares = []
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
                continue
            f, m = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.999)
            if f is None or m is None or int(m.sum()) < min_inliers:
                continue
            ok = m.ravel().astype(bool)
            a, b = p1[ok].astype(np.float64), p2[ok].astype(np.float64)
            if len(a) > max_puntos:  # que todos los pares pesen parecido
                j = rng.choice(len(a), max_puntos, replace=False)
                a, b = a[j], b[j]
            pares.append((a, b))
    return pares, q


class _Puntos:
    """Los puntos de todos los pares apilados, para evaluar el costo de un (f, k1) de una vez."""

    def __init__(self, pares: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.p1 = np.concatenate([a for a, _ in pares])
        self.p2 = np.concatenate([b for _, b in pares])
        n = np.array([len(a) for a, _ in pares])
        self.cortes = np.concatenate([[0], np.cumsum(n)[:-1]])
        self.pesos = n.astype(float)

    def subconjunto(self, idx: np.ndarray) -> _Puntos:
        nuevo = object.__new__(_Puntos)
        trozos = [(int(self.cortes[i]), int(self.cortes[i] + self.pesos[i])) for i in idx]
        nuevo.p1 = np.concatenate([self.p1[a:b] for a, b in trozos])
        nuevo.p2 = np.concatenate([self.p2[a:b] for a, b in trozos])
        n = np.array([b - a for a, b in trozos])
        nuevo.cortes = np.concatenate([[0], np.cumsum(n)[:-1]])
        nuevo.pesos = n.astype(float)
        return nuevo


def costo_lente(f: float, k1: float, pts: _Puntos, cx: float, cy: float) -> float:
    """Media (pesada por puntos) de (σ1 − σ2)/σ1 de las E estimadas con (f, k1)."""
    from visionx.depth.camara_net.lente import sin_distorsion

    x1, y1 = sin_distorsion((pts.p1[:, 0] - cx) / f, (pts.p1[:, 1] - cy) / f, k1, 0.0)
    x2, y2 = sin_distorsion((pts.p2[:, 0] - cx) / f, (pts.p2[:, 1] - cy) / f, k1, 0.0)
    ok = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    if ok.mean() < 0.8:
        return math.inf  # la lente se pliega en buena parte de la imagen: imposible
    a = np.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)], 1)
    a[~ok] = 0.0  # puntos más allá del pliegue (esquinas): no cuentan
    m = np.add.reduceat(np.einsum("ni,nj->nij", a, a).reshape(len(a), 81), pts.cortes, axis=0)
    _, v = np.linalg.eigh(m.reshape(-1, 9, 9))
    e = v[:, :, 0].reshape(-1, 3, 3)
    s = np.linalg.svd(e, compute_uv=False)
    c = (s[:, 0] - s[:, 1]) / np.maximum(s[:, 0], 1e-12)
    return float((pts.pesos * c).sum() / pts.pesos.sum())


def _mejor_lente(pts: _Puntos, cx: float, cy: float, f0: float, rango_f: float,
                 rango_k1: tuple[float, float], nf: int = 41, nk: int = 13,
                 vueltas: int = 3) -> tuple[float, float]:  # fmt: skip
    """Grilla en (log f, k1) y refinamientos alrededor del mínimo."""
    lf0, lr = math.log(f0), math.log(rango_f)
    lfs = np.linspace(lf0 - lr, lf0 + lr, nf)
    k1s = np.linspace(rango_k1[0], rango_k1[1], nk)
    for _ in range(vueltas):
        cs = np.array([[costo_lente(math.exp(lf), k, pts, cx, cy) for k in k1s] for lf in lfs])
        i, j = np.unravel_index(int(np.argmin(cs)), cs.shape)
        dlf, dk = (lfs[1] - lfs[0]), (k1s[1] - k1s[0])
        lfs = np.linspace(lfs[i] - dlf, lfs[i] + dlf, 9)
        k1s = np.linspace(k1s[j] - dk, k1s[j] + dk, 9)
    return math.exp(float(lfs[4])), float(k1s[4])


def lente_por_movimiento(
    cuadros: Sequence[np.ndarray],
    cx: float,
    cy: float,
    f_inicial: float,
    saltos: Sequence[int] = (3, 6, 12),
    rango: float = 2.0,
    rango_k1: tuple[float, float] = K1_RANGO,
    min_pares: int = 6,
    remuestreos: int = 20,
    semilla: int = 0,
) -> FocalMedida:
    """Focal y k1 (px de la imagen original) que mejor explican el movimiento entre los cuadros.

    Como `focal_por_movimiento`, pero sin suponer pinhole: sirve para cámaras con lente (gran
    angular baratas). La duda de log f sale de remuestrear los pares (×`FACTOR_SIGMA`).
    """
    if len(cuadros) < max(saltos) + 2:
        return FocalMedida(None, None, 0, "muy pocos cuadros")
    pares, q = emparejar(cuadros, saltos)
    if len(pares) < min_pares:
        return FocalMedida(None, None, len(pares), "poco movimiento o poca textura")
    pts = _Puntos(pares)
    cxq, cyq, f0q = cx * q, cy * q, f_inicial * q
    f, k1 = _mejor_lente(pts, cxq, cyq, f0q, rango, rango_k1)
    rng = np.random.default_rng(semilla)
    logs = []
    for _ in range(remuestreos):
        sub = pts.subconjunto(rng.integers(0, len(pares), len(pares)))
        fb, _ = _mejor_lente(sub, cxq, cyq, f, 1.1, (k1 - 0.06, k1 + 0.06), nf=9, nk=7,
                             vueltas=2)  # fmt: skip
        logs.append(math.log(fb))
    sigma = FACTOR_SIGMA * float(np.std(logs)) if logs else None
    if abs(math.log(f / f0q)) > 0.98 * math.log(rango):
        return FocalMedida(None, sigma, len(pares), "la focal cae en el borde de la búsqueda")
    return FocalMedida(f / q, sigma, len(pares), k1=k1)
