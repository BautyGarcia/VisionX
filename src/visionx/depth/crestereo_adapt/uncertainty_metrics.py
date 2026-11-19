"""Cómo se juzga una incertidumbre: si ordena bien los errores y si sus intervalos se cumplen.

Una incertidumbre sirve para dos cosas distintas, y cada una tiene su medida:

- **Ordenar**: que los píxeles con más incertidumbre sean los que más erran. Se mide con la
  curva de *sparsification*: se van descartando los píxeles menos confiables y se mira cómo
  baja el error de los que quedan. La mejor curva posible descarta por error real (el
  "oráculo"). El área entre las dos es el **AUSE**: 0 es perfecto; cuanto más chico, mejor.
- **Calibrar**: que el tamaño de la incertidumbre sea creíble. Si b describe una Laplace, el
  error cae dentro de ±t con probabilidad 1 − exp(−t/b). Se mide qué fracción de los errores
  queda dentro del intervalo que debería contener el 50 % y el 90 %: la **cobertura**.

Ordenar bien no implica estar calibrado, ni al revés: por eso van las dos.

Todo acá es NumPy sobre arreglos de error e incertidumbre ya calculados; no usa la red.
"""

from __future__ import annotations

import math

import numpy as np

from visionx.shared.errors import InvalidInput

FRACTIONS = np.linspace(0.0, 0.95, 20)  # qué parte de los píxeles se descarta en cada paso


def _flat(error: np.ndarray, uncertainty: np.ndarray, mask: np.ndarray) -> tuple:
    if error.shape != uncertainty.shape or error.shape != mask.shape:
        raise InvalidInput("error, incertidumbre y máscara deben tener la misma forma")
    keep = mask & np.isfinite(error) & np.isfinite(uncertainty)
    return np.abs(error[keep]).astype(np.float64), uncertainty[keep].astype(np.float64)


def sparsification_curve(
    error: np.ndarray, uncertainty: np.ndarray, mask: np.ndarray, fractions: np.ndarray = FRACTIONS
) -> np.ndarray:
    """Error medio que queda al descartar la fracción más incierta, relativo al error total.

    Arranca en 1 (no se descartó nada). Una buena incertidumbre la hace bajar rápido.
    """
    err, unc = _flat(error, uncertainty, mask)
    if err.size == 0 or err.mean() == 0:
        return np.full(len(fractions), np.nan)
    order = np.argsort(-unc, kind="stable")  # del más incierto al menos
    # Suma acumulada desde el final: error total de los n píxeles más confiables.
    tail = np.cumsum(err[order][::-1])
    curve = []
    for fraction in fractions:
        kept = max(int(round(err.size * (1.0 - fraction))), 1)
        curve.append(tail[kept - 1] / kept / err.mean())
    return np.asarray(curve)


def ause(
    error: np.ndarray, uncertainty: np.ndarray, mask: np.ndarray, fractions: np.ndarray = FRACTIONS
) -> float:
    """Área entre la curva de la incertidumbre y la del oráculo (0 = ordena perfecto)."""
    curve = sparsification_curve(error, uncertainty, mask, fractions)
    oracle = sparsification_curve(error, np.abs(error), mask, fractions)
    if not np.isfinite(curve).all():
        return float("nan")
    return float(np.mean(curve - oracle))


def interval_coverage(error: np.ndarray, b: np.ndarray, mask: np.ndarray, level: float) -> float:
    """Fracción de errores dentro del intervalo de Laplace que debería contener `level`.

    Calibrada: devuelve ~`level`. Menos: b es chico (demasiada confianza). Más: b es grande.
    """
    if not 0.0 < level < 1.0:
        raise InvalidInput("level debe estar entre 0 y 1")
    err, scale = _flat(error, b, mask)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err <= -scale * math.log(1.0 - level)))


def laplace_nll(error: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Verosimilitud negativa media de Laplace: |error|/b + log(2b). Más chica, mejor."""
    err, scale = _flat(error, b, mask)
    if err.size == 0:
        return float("nan")
    scale = np.maximum(scale, 1e-6)
    return float(np.mean(err / scale + np.log(2.0 * scale)))
