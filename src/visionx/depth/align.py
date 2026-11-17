"""Alineación robusta del mapa monocular a la escala estéreo (T05, TECNICO "Alineación robusta").

El mapa relativo m de Depth Anything se lleva a disparidad con una recta: d ≈ s·m + t. Los
puntos de apoyo (**anclas**) son los píxeles donde el estéreo dio una medida fiable. Nunca la
GT: en la app no existe, y en evaluación usarla sería hacer trampa.

Por qué no alcanza con contar anclas: quinientos píxeles de la misma pared no dicen nada de la
escala del resto de la escena. Por eso el diagnóstico mira también:
- **Cobertura de grilla:** en qué fracción de las celdas de una grilla hay anclas suficientes.
- **Rango de disparidad:** si todas las anclas están a la misma distancia, la pendiente s no
  queda determinada (cualquier recta que pase por ese punto sirve).
- **Consenso (inliers):** qué fracción de anclas queda cerca de la recta elegida por RANSAC.
- **Celdas reservadas:** antes de ajustar se apartan algunas celdas enteras. El ajuste no las
  ve; después se mide qué tan bien las predice. Se reservan celdas y no píxeles sueltos porque
  un píxel vecino de una ancla usada no comprueba nada. Siguen siendo medidas estéreo, no GT.

**RANSAC:** prueba muchas rectas armadas con dos anclas al azar, se queda con la que más
anclas explica y la reajusta por mínimos cuadrados solo con esas. Así, una zona del estéreo
equivocada (reflejo, patrón repetido) no tuerce la recta.

**Huber** (`AlignParams.method = "huber"`, la recta global de B5, T12): en vez de elegir por
votos, minimiza una pérdida que es cuadrática para residuos chicos y lineal para los grandes
(IRLS: mínimos cuadrados repetidos, bajando el peso de las anclas lejanas a la recta). Usa
todas las anclas de ajuste, no solo las de una recta sorteada: en InStereo2K val RANSAC a veces
elige una pendiente anómala que corre decenas de px una región entera (BITACORA 2026-11-16).
Submuestreo, grilla, celdas reservadas y criterios de aceptación son los mismos que con RANSAC.

**Corrección afín local** (`local_affine`, B5): la recta global es una sola para toda la
imagen, pero la relación monocular → disparidad no es exactamente una recta. En cada píxel de
un hueco del estéreo se reajusta una recta propia con las anclas cercanas, pesadas con una
ventana gaussiana, y con la pendiente atada a la global. Ver el docstring de `local_affine`.

Los umbrales en píxeles dependen de la resolución de cómputo: son provisorios hasta el ajuste
de T05 PR 3. Si la alineación no se acepta, nadie usa s y t: B4 se queda con el estéreo solo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import cv2
import numpy as np

from visionx.shared import _validate as v
from visionx.shared.errors import InvalidInput

ALIGN_METHODS = ("ransac", "huber")
# Huber: umbral = HUBER_K · σ robusta de los residuos. 1,345 es la constante clásica: con ruido
# gaussiano conserva el 95 % de la eficiencia de mínimos cuadrados y, con outliers, las anclas
# a más de ~1,3 σ pesan c/|r| en vez de 1.
HUBER_K = 1.345
# σ robusta = 1,4826 · MAD (mediana de |r − mediana(r)|): con ruido gaussiano coincide con el
# desvío estándar, pero un 20 % de anclas equivocadas casi no la mueve.
MAD_TO_SIGMA = 1.4826
HUBER_ITERATIONS = 50  # tope de IRLS; converge en ~10 pasadas en InStereo2K
HUBER_TOLERANCE = 1e-7  # cambio relativo de (s, t) para cortar
HUBER_MIN_SIGMA_PX = 1e-3  # piso de σ: con anclas exactas el MAD es 0 y el umbral también


@dataclass(frozen=True)
class AlignParams:
    grid_cols: int = 8  # celdas horizontales de la grilla de diagnóstico
    grid_rows: int = 6
    min_anchors_per_cell: int = 20  # anclas para que una celda cuente como poblada
    min_grid_coverage: float = 0.25  # fracción mínima de celdas pobladas
    holdout_fraction: float = 0.2  # fracción de celdas pobladas que se reservan
    max_fit_anchors: int = 5000  # submuestreo para que RANSAC sea rápido
    ransac_iterations: int = 200
    inlier_threshold_px: float = 1.0  # |s·m + t - d| para contar como inlier
    min_inlier_ratio: float = 0.5  # en las anclas de ajuste
    min_holdout_inlier_ratio: float = 0.5  # en las celdas reservadas
    min_disparity_spread_px: float = 2.0  # P95 - P5 de la disparidad de las anclas
    seed: int = 0  # misma entrada, misma alineación
    # Cómo se ajusta la recta: "ransac" (B4, congelado en T05) o "huber" (IRLS, recta de B5).
    # No entra en la clave de la caché de Fusion cuando es "ransac" (ver cache.key_material).
    method: str = "ransac"

    def __post_init__(self) -> None:
        if self.method not in ALIGN_METHODS:
            raise InvalidInput(f"method {self.method!r}: se esperaba uno de {ALIGN_METHODS}")
        for name in ("grid_cols", "grid_rows", "max_fit_anchors", "ransac_iterations"):
            if v.non_negative_int(name, getattr(self, name)) == 0:
                raise InvalidInput(f"{name} tiene que ser > 0")
        v.non_negative_int("min_anchors_per_cell", self.min_anchors_per_cell)
        v.non_negative_int("seed", self.seed)
        for name in ("min_grid_coverage", "holdout_fraction", "min_inlier_ratio",
                     "min_holdout_inlier_ratio"):  # fmt: skip
            v.finite(name, getattr(self, name), low=0.0, high=1.0)
        v.finite("inlier_threshold_px", self.inlier_threshold_px, low=0.0)
        v.finite("min_disparity_spread_px", self.min_disparity_spread_px, low=0.0)

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> AlignParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: val for k, val in section.items() if k in names})


@dataclass(frozen=True)
class Alignment:
    """Resultado auxiliar de la alineación. No va dentro de DepthResult (schema 3)."""

    accepted: bool
    reason: str | None  # por qué no se aceptó; None si se aceptó
    scale: float  # s; NaN si no se llegó a ajustar
    offset: float  # t, en px
    n_anchors: int
    n_fit: int
    n_holdout: int
    grid_coverage: float  # fracción de celdas pobladas
    disparity_spread_px: float
    inlier_ratio: float
    holdout_inlier_ratio: float
    fit_p50_px: float  # residuo absoluto en anclas de ajuste
    fit_p95_px: float
    holdout_p50_px: float  # residuo absoluto en celdas reservadas
    holdout_p95_px: float

    def apply(self, mono: np.ndarray) -> np.ndarray:
        if not self.accepted:
            raise InvalidInput(f"alineación no aceptada: {self.reason}")
        return (np.float32(self.scale) * mono + np.float32(self.offset)).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(x, float) and math.isnan(x) else x)
                for k, x in asdict(self).items()}  # fmt: skip


def _rejected(reason: str, **known: Any) -> Alignment:
    base = dict(scale=math.nan, offset=math.nan, n_anchors=0, n_fit=0, n_holdout=0,
                grid_coverage=0.0, disparity_spread_px=math.nan, inlier_ratio=math.nan,
                holdout_inlier_ratio=math.nan, fit_p50_px=math.nan, fit_p95_px=math.nan,
                holdout_p50_px=math.nan, holdout_p95_px=math.nan)  # fmt: skip
    base.update(known)
    return Alignment(accepted=False, reason=reason, **base)


def _cells(shape: tuple[int, int], p: AlignParams) -> np.ndarray:
    """Número de celda de cada píxel (fila de celda * columnas + columna de celda)."""
    h, w = shape
    rows = (np.arange(h) * p.grid_rows // h)[:, None]
    cols = (np.arange(w) * p.grid_cols // w)[None, :]
    return rows * p.grid_cols + cols


def _lstsq(m: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    a = np.stack([m, np.ones_like(m)], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, d, rcond=None)
    return float(s), float(t)


def _ransac(m: np.ndarray, d: np.ndarray, p: AlignParams, rng: np.random.Generator):
    """Mejor recta con pendiente positiva. Devuelve (s, t) o None."""
    n = m.size
    pairs = rng.integers(0, n, size=(p.ransac_iterations, 2))
    m1, m2 = m[pairs[:, 0]], m[pairs[:, 1]]
    d1, d2 = d[pairs[:, 0]], d[pairs[:, 1]]
    dm = m2 - m1
    usable = np.abs(dm) > 1e-9
    s = np.where(usable, (d2 - d1) / np.where(usable, dm, 1.0), np.nan)
    t = d1 - s * m1
    # Más cerca = m más grande = d más grande: una pendiente <= 0 no tiene sentido físico.
    usable &= np.isfinite(s) & (s > 0)
    if not usable.any():
        return None
    s, t = s[usable], t[usable]
    residual = np.abs(s[:, None] * m[None, :] + t[:, None] - d[None, :])
    best = int(np.argmax((residual <= p.inlier_threshold_px).sum(axis=1)))
    inliers = residual[best] <= p.inlier_threshold_px
    if inliers.sum() < 2:
        return None
    # Reajuste por mínimos cuadrados con los inliers, dos pasadas (los inliers se recalculan).
    s_fit, t_fit = _lstsq(m[inliers], d[inliers])
    inliers = np.abs(s_fit * m + t_fit - d) <= p.inlier_threshold_px
    if inliers.sum() >= 2:
        s_fit, t_fit = _lstsq(m[inliers], d[inliers])
    return s_fit, t_fit


def _weighted_lstsq(m: np.ndarray, d: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    root = np.sqrt(w)
    a = np.stack([m * root, root], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, d * root, rcond=None)
    return float(s), float(t)


def _huber(m: np.ndarray, d: np.ndarray):
    """Recta de Huber por IRLS. Devuelve (s, t) con s > 0, o None.

    Cada pasada: residuos r = d − (s·m + t); σ = 1,4826 · MAD(r) (con piso); umbral
    c = HUBER_K · σ; pesos w = 1 si |r| <= c y c/|r| si no; mínimos cuadrados ponderados.
    Arranca de mínimos cuadrados comunes. La pérdida de Huber es convexa para σ fija, así que
    no depende de un sorteo como RANSAC (sí de las anclas submuestreadas, con semilla fija).
    """
    if m.size < 2 or np.ptp(m) <= 1e-9:
        return None
    s, t = _lstsq(m, d)
    for _ in range(HUBER_ITERATIONS):
        r = d - (s * m + t)
        sigma = max(MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r)))),
                    HUBER_MIN_SIGMA_PX)  # fmt: skip
        c = HUBER_K * sigma
        a = np.abs(r)
        w = np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))
        s_new, t_new = _weighted_lstsq(m, d, w)
        change = abs(s_new - s) / max(abs(s), 1e-12) + abs(t_new - t) / max(abs(t), 1.0)
        s, t = s_new, t_new
        if change < HUBER_TOLERANCE:
            break
    if not (math.isfinite(s) and math.isfinite(t) and s > 0):
        return None
    return s, t


def align(
    mono: np.ndarray, disparity: np.ndarray, anchors: np.ndarray, params: AlignParams
) -> Alignment:
    """Ajusta d ≈ s·m + t con las anclas estéreo y diagnostica si el ajuste es confiable.

    `params.method` elige RANSAC (B4) o Huber (B5); todo lo demás (grilla, celdas reservadas,
    submuestreo y criterios de aceptación) es igual para los dos.
    """
    m = v.array("mono", mono, np.float32, 2)
    d = v.array("disparity", disparity, np.float32, 2, m.shape)
    v.array("anchors", anchors, np.bool_, 2, m.shape)
    p = params
    usable = anchors & np.isfinite(d) & np.isfinite(m)
    n_anchors = int(usable.sum())

    cell = _cells(m.shape, p)
    n_cells = p.grid_rows * p.grid_cols
    counts = np.bincount(cell[usable], minlength=n_cells)
    populated = np.flatnonzero(counts >= max(p.min_anchors_per_cell, 1))
    coverage = populated.size / n_cells
    known = dict(n_anchors=n_anchors, grid_coverage=coverage)
    if populated.size < 2 or coverage < p.min_grid_coverage:
        return _rejected(f"anclas concentradas: {coverage:.0%} de la grilla con anclas", **known)

    rng = np.random.default_rng(p.seed)
    n_hold = min(max(1, round(p.holdout_fraction * populated.size)), populated.size - 1)
    hold_cells = rng.choice(populated, size=n_hold, replace=False)
    in_holdout = np.isin(cell, hold_cells)
    fit_mask = usable & ~in_holdout
    hold_mask = usable & in_holdout
    m_fit, d_fit = m[fit_mask].astype(np.float64), d[fit_mask].astype(np.float64)
    m_hold, d_hold = m[hold_mask].astype(np.float64), d[hold_mask].astype(np.float64)
    known.update(n_fit=int(m_fit.size), n_holdout=int(m_hold.size))

    spread = float(np.percentile(d_fit, 95) - np.percentile(d_fit, 5))
    known.update(disparity_spread_px=spread)
    if spread < p.min_disparity_spread_px:
        return _rejected(f"anclas sin variación de disparidad ({spread:.1f} px): "
                         "la escala no queda determinada", **known)  # fmt: skip

    if m_fit.size > p.max_fit_anchors:
        pick = rng.choice(m_fit.size, size=p.max_fit_anchors, replace=False)
        m_sample, d_sample = m_fit[pick], d_fit[pick]
    else:
        m_sample, d_sample = m_fit, d_fit
    if p.method == "huber":
        fitted = _huber(m_sample, d_sample)
        if fitted is None:
            return _rejected("Huber sin recta de pendiente positiva", **known)
    else:
        fitted = _ransac(m_sample, d_sample, p, rng)
        if fitted is None:
            return _rejected("RANSAC sin recta de pendiente positiva", **known)
    s, t = fitted
    if not (math.isfinite(s) and math.isfinite(t) and s > 0):
        return _rejected("escala no positiva tras el reajuste", **known)

    res_fit = np.abs(s * m_fit + t - d_fit)
    res_hold = np.abs(s * m_hold + t - d_hold)
    inlier_ratio = float((res_fit <= p.inlier_threshold_px).mean())
    holdout_ratio = float((res_hold <= p.inlier_threshold_px).mean())
    stats = dict(scale=s, offset=t, inlier_ratio=inlier_ratio, holdout_inlier_ratio=holdout_ratio,
                 fit_p50_px=float(np.percentile(res_fit, 50)),
                 fit_p95_px=float(np.percentile(res_fit, 95)),
                 holdout_p50_px=float(np.percentile(res_hold, 50)),
                 holdout_p95_px=float(np.percentile(res_hold, 95)))  # fmt: skip
    known.update(stats)
    if inlier_ratio < p.min_inlier_ratio:
        return _rejected(f"poco consenso: {inlier_ratio:.0%} de anclas cerca de la recta", **known)
    if holdout_ratio < p.min_holdout_inlier_ratio:
        return _rejected(f"no generaliza: {holdout_ratio:.0%} de las anclas reservadas "
                         "cerca de la recta", **known)  # fmt: skip
    return Alignment(accepted=True, reason=None, **known)


# --- corrección afín local (B5) ------------------------------------------------------------


@dataclass(frozen=True)
class LocalAffineParams:
    """Parámetros de `local_affine` (sección `depth.b5` de la config, salvo `method`)."""

    sigma_px: float = 120.0  # σ de la ventana gaussiana, en px de la imagen de cómputo
    slope_lambda_px2: float = 100.0  # λ: ata la pendiente local a la global (ver local_affine)
    exclude_px: float = 0.0  # anclas a menos de esto de un hueco no se usan (0 = todas)
    samples_per_sigma: float = 8.0  # grilla reducida: paso = σ / esto (σ en la grilla ~ 8 celdas)
    min_density: float = 1e-4  # bajo esta fracción de anclas en la ventana, recta global
    max_slope_ratio: float = 2.0  # la pendiente local queda en [global / r, global · r]

    def __post_init__(self) -> None:
        for name in ("sigma_px", "samples_per_sigma"):
            if v.finite(name, getattr(self, name)) <= 0:
                raise InvalidInput(f"{name} tiene que ser > 0")
        v.finite("slope_lambda_px2", self.slope_lambda_px2, low=0.0)
        v.finite("exclude_px", self.exclude_px, low=0.0)
        v.finite("min_density", self.min_density, low=0.0, high=1.0)
        v.finite("max_slope_ratio", self.max_slope_ratio, low=1.0)

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> LocalAffineParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(val) for k, val in section.items() if k in names})


@dataclass(frozen=True)
class AnchorSupport:
    """Qué anclas usa la corrección local y cuánto apoyo tiene cada píxel."""

    used: np.ndarray  # H x W bool: anclas tras la exclusión
    distance_px: np.ndarray  # H x W float32: px al ancla usada más cercana (inf sin anclas)
    density: np.ndarray  # H x W float32 en [0, 1]: Σw normalizada (fracción de anclas)


@dataclass(frozen=True)
class LocalCorrection:
    """Salida de `local_affine`: disparidad corregida en los huecos y el apoyo de cada píxel."""

    disparity_px: np.ndarray  # H x W float32: corregida donde `valid`; NaN en el resto
    valid: np.ndarray  # H x W bool: hueco, recta global aceptada, d finita y d + doffs > 0
    support: AnchorSupport
    stats: dict[str, float] = field(default_factory=dict)


def _distance_to_false(mask: np.ndarray) -> np.ndarray:
    """Distancia euclídea (px) de cada píxel al píxel False más cercano; inf si no hay ninguno.

    cv2.distanceTransform mide hasta el cero más cercano; DIST_MASK_PRECISE es la distancia
    exacta (no la aproximación de máscara 3x3/5x5).
    """
    if mask.all():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


class _LowResGaussian:
    """Desenfoque gaussiano de σ grande a resolución reducida.

    Con σ = 120 px una convolución directa a 1080x860 tarda segundos. Las sumas ponderadas
    varían despacio (escala σ), así que alcanza con calcularlas en una grilla de paso σ / k:
    1. `down`: bajar los mapas promediando cada celda (INTER_AREA: conserva la masa de las
       anclas) y convolucionar con una gaussiana de σ / paso celdas (borde = 0: fuera de la
       imagen no hay anclas);
    2. `up`: subir bilineal al tamaño original.
    Error de aproximación: del orden del paso frente a σ (1/k), mucho menor que la escala de la
    corrección. Varios mapas viajan juntos como canales de una sola imagen (una llamada).
    """

    def __init__(self, shape: tuple[int, int], sigma_px: float, samples_per_sigma: float):
        h, w = shape
        step = max(1.0, sigma_px / samples_per_sigma)
        self.size = (w, h)
        self.low = (max(1, round(w / step)), max(1, round(h / step)))
        sx = sigma_px * self.low[0] / w  # σ en celdas de la grilla reducida, por eje
        sy = sigma_px * self.low[1] / h
        self.kx = cv2.getGaussianKernel(2 * math.ceil(3 * sx) + 1, sx, cv2.CV_64F)
        self.ky = cv2.getGaussianKernel(2 * math.ceil(3 * sy) + 1, sy, cv2.CV_64F)

    def down(self, maps: np.ndarray) -> np.ndarray:
        """H x W (x C) a resolución completa -> h x w (x C) desenfocado en la grilla reducida."""
        if maps.ndim == 3 and maps.shape[2] > 4:  # cv2.resize acepta hasta 4 canales
            return np.dstack([self.down(maps[..., i : i + 4]) for i in range(0, maps.shape[2], 4)])
        small = cv2.resize(maps, self.low, interpolation=cv2.INTER_AREA)
        return cv2.sepFilter2D(small, cv2.CV_64F, self.kx, self.ky,
                               borderType=cv2.BORDER_CONSTANT)  # fmt: skip

    def up(self, maps: np.ndarray) -> np.ndarray:
        return cv2.resize(maps, self.size, interpolation=cv2.INTER_LINEAR)


def _used_anchors(anchors: np.ndarray, holes: np.ndarray, p: LocalAffineParams) -> np.ndarray:
    """Anclas a `exclude_px` o más de cualquier hueco (transformada de distancia)."""
    if p.exclude_px <= 0:
        return anchors.copy()
    return anchors & (_distance_to_false(~holes) >= p.exclude_px)


def anchor_support(
    anchors: np.ndarray, holes: np.ndarray, params: LocalAffineParams
) -> AnchorSupport:
    """Anclas que usa la corrección local, su distancia y su densidad (no necesita la recta).

    Excluye las anclas a menos de `exclude_px` de un hueco: el estéreo pegado a un hueco es el
    menos confiable (EPE 2,47 px a 0–4 px de un hueco contra 0,23 px a más de 64 px en
    InStereo2K val, BITACORA 2026-11-16). La densidad es Σw de la ventana gaussiana con pesos
    normalizados (Σ G_σ = 1): la fracción de píxeles de la ventana que son anclas usadas.
    """
    v.array("anchors", anchors, np.bool_, 2)
    v.array("holes", holes, np.bool_, 2, anchors.shape)
    used = _used_anchors(anchors, holes, params)
    blur = _LowResGaussian(anchors.shape, params.sigma_px, params.samples_per_sigma)
    density = blur.up(blur.down(used.astype(np.float64)))
    return AnchorSupport(used, _distance_to_false(~used),
                         np.clip(density, 0.0, 1.0).astype(np.float32))  # fmt: skip


def local_affine(
    mono: np.ndarray,
    disparity: np.ndarray,
    anchors: np.ndarray,
    alignment: Alignment,
    holes: np.ndarray,
    params: LocalAffineParams,
    doffs: float = 0.0,
) -> LocalCorrection:
    """Disparidad en los huecos con una recta local por píxel, atada a la recta global.

    Se trabaja sobre el monocular ya llevado a px con la recta global, g = s·m + t (así λ queda
    en px² y no depende de la escala arbitraria del monocular). En cada píxel p se busca
    d ≈ a·g + b minimizando

        Σ_i ŵ_i (a·g_i + b − d_i)² + λ (a − 1)²,   ŵ_i = G_σ(p − x_i) / Σ_j G_σ(p − x_j),

    con i sobre las anclas usadas. Con medias y covarianza ponderadas (ḡ, d̄, Var_w g,
    Cov_w(g, d)) la solución cerrada es

        a = (Cov_w(g, d) + λ) / (Var_w(g) + λ),     b = d̄ − a·ḡ,

    y en términos del monocular: s_local = a·s, t_local = a·t + b. λ = 0 es mínimos cuadrados
    locales puros; λ → ∞ deja a = 1 (solo corrige el desplazamiento). `a` se recorta a
    [1/r, r] (r = max_slope_ratio) para que una ventana sin variación de g no invente escalas.
    Las cinco sumas (Σw, Σw·g, Σw·d, Σw·g², Σw·g·d) son convoluciones gaussianas de mapas que
    valen 0 fuera de las anclas. Se calculan en la grilla reducida de `_LowResGaussian`, donde
    también se resuelven a y b; a resolución completa se suben a, b y Σw (bilineal) y se
    evalúa a·g + b solo en los huecos.

    Donde la fracción de anclas en la ventana (Σw normalizada) es menor que `min_density`, no
    hay con qué corregir y queda la recta global. Un valor con d + doffs <= 0 se invalida,
    no se recorta. Si la recta global no se aceptó, no se corrige nada (valid todo False),
    pero el apoyo de anclas se calcula igual (Fusion F3 lo usa como canal).
    """
    m = v.array("mono", mono, np.float32, 2)
    d = v.array("disparity", disparity, np.float32, 2, m.shape)
    v.array("anchors", anchors, np.bool_, 2, m.shape)
    v.array("holes", holes, np.bool_, 2, m.shape)
    doffs = v.finite("doffs", doffs)
    p = params
    usable = anchors & np.isfinite(d) & np.isfinite(m)
    out = np.full(m.shape, np.nan, np.float32)
    if not alignment.accepted:
        support = anchor_support(usable, holes, p)
        stats = {"fraccion_global": math.nan, "pendiente_local_mediana": math.nan}
        return LocalCorrection(out, np.zeros(m.shape, bool), support, stats)

    used = _used_anchors(usable, holes, p)
    blur = _LowResGaussian(m.shape, p.sigma_px, p.samples_per_sigma)
    g = alignment.scale * m.astype(np.float64) + alignment.offset  # recta global, en px
    if used.any():
        # Centrar g y d en sus medias globales no cambia a ni b, pero evita restar números
        # grandes parecidos (Σw·g²/Σw − ḡ² con g ~ 270 px) al calcular la varianza.
        g0, d0 = float(g[used].mean()), float(d[used].mean())
        gc = np.where(used, g - g0, 0.0)
        dc = np.where(used, d.astype(np.float64) - d0, 0.0)
        sums = blur.down(np.dstack([used.astype(np.float64), gc, dc, gc * gc, gc * dc]))
        s0 = sums[..., 0]
        safe = np.maximum(s0, 1e-12)
        g_mean, d_mean = sums[..., 1] / safe, sums[..., 2] / safe
        var = np.maximum(sums[..., 3] / safe - g_mean**2, 0.0)
        cov = sums[..., 4] / safe - g_mean * d_mean
        den = var + p.slope_lambda_px2
        a = np.where(den > 1e-12, (cov + p.slope_lambda_px2) / np.maximum(den, 1e-12), 1.0)
        a = np.clip(a, 1.0 / p.max_slope_ratio, p.max_slope_ratio)
        # b en px absolutos para la g sin centrar: d = d0 + d̄c + a·(g − g0 − ḡc).
        b = d0 + d_mean - a * (g0 + g_mean)
        full = blur.up(np.dstack([s0, a, b]))
        density, a_full, b_full = full[..., 0], full[..., 1], full[..., 2]
    else:
        density = np.zeros(m.shape)
        a_full = np.ones(m.shape)
        b_full = np.zeros(m.shape)
    supported = density >= max(p.min_density, 1e-12)

    in_hole = holes
    gh = g[in_hole]
    local = np.where(supported[in_hole], a_full[in_hole] * gh + b_full[in_hole], gh)
    with np.errstate(invalid="ignore"):
        ok_h = np.isfinite(local) & ((local + doffs) > 0)
    ok = np.zeros(m.shape, bool)
    ok[in_hole] = ok_h
    out[ok] = local[ok_h].astype(np.float32)

    support = AnchorSupport(used, _distance_to_false(~used),
                            np.clip(density, 0.0, 1.0).astype(np.float32))  # fmt: skip
    n_holes = int(holes.sum())
    corrected = holes & supported
    stats = {
        "fraccion_global": float(n_holes - corrected.sum()) / n_holes if n_holes else 0.0,
        "pendiente_local_mediana": float(np.median(a_full[corrected]))
        if corrected.any() else math.nan,
    }  # fmt: skip
    return LocalCorrection(out, ok, support, stats)
