"""Del b de Laplace (U-Net de incertidumbre, px de disparidad) a un intervalo en metros.

Contrato de docs/INCERTIDUMBRE-Y-GT.md (sección 1.2). Lo usan el ubicador (para llenar
`SpatialObject.z_bajo_m` / `z_alto_m`), la evaluación del GT sintético y la app.

Laplace: P(|error| > t) = e^(−t/b), así que el 90 % central de la disparidad es ±b·ln 10.
"""

from __future__ import annotations

import math

LN10 = math.log(10.0)


def laplace_interval(
    d_front_px: float, b_px: float, fb: float, doffs: float = 0.0, k: float = 1.0
) -> tuple[float, float | None]:
    """(z_bajo, z_alto) del 90 % en metros. z_alto None = sin cota superior.

    `fb` = fx·B (m·px) de la geometría de la profundidad; `k` = factor de calibración de b
    (se ajusta con la validación del GT, para que el intervalo cubra ~90 %).
    """
    d = d_front_px + doffs
    t = LN10 * b_px * k
    bajo = fb / (d + t)
    alto = fb / (d - t) if d - t > 0 else None
    return bajo, alto


def relative_width(z_m: float, z_bajo_m: float, z_alto_m: float | None) -> float:
    """Medio ancho relativo del intervalo: (alto − bajo) / (2·z). inf sin cota superior."""
    if z_alto_m is None:
        return math.inf
    return (z_alto_m - z_bajo_m) / (2.0 * z_m)
