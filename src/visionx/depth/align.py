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

Los umbrales en píxeles dependen de la resolución de cómputo: son provisorios hasta el ajuste
de T05 PR 3. Si la alineación no se acepta, nadie usa s y t: B4 se queda con el estéreo solo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from visionx.shared import _validate as v
from visionx.shared.errors import InvalidInput


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

    def __post_init__(self) -> None:
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


def align(
    mono: np.ndarray, disparity: np.ndarray, anchors: np.ndarray, params: AlignParams
) -> Alignment:
    """Ajusta d ≈ s·m + t con las anclas estéreo y diagnostica si el ajuste es confiable."""
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
