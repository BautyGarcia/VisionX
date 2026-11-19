"""Cuánto cambia la profundidad cuando el par está mal alineado, y cuánto se recupera.

Es la medición central de la compuerta de T12. No hay profundidad de referencia en los clips
de la ELP, así que la referencia es **la propia red sobre el par bien rectificado**: se mide
cuánto se aleja el resultado cuando se desalinea el ojo derecho. Eso no dice si la referencia
es correcta en metros (para eso está la cinta); dice cuánto daño hace la desalineación.

Para cada nivel de error se comparan tres caminos:

    sin corregir   el par desalineado, tal cual (el "piso")
    clásico        se estima la corrección con puntos emparejados y se aplica
    oráculo        se aplica la corrección exacta (el "techo" de cualquier alineación)

El oráculo no vuelve exactamente a cero: al mover la imagen se pierde un borde y se interpola.
Esa diferencia es el costo mínimo de corregir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visionx.depth.crestereo_adapt.correction import Correction

# Niveles de cada error, de a uno por vez. Los de roll y pitch llegan a los rangos iniciales
# de la nota del 18/11; los chicos rodean lo medido en la ELP (0,24° con el precinto, 09/11).
SWEEPS: dict[str, list[float]] = {
    "roll_deg": [0.05, 0.1, 0.25, 0.5, 1.0, 1.5],
    "pitch_deg": [0.05, 0.1, 0.25, 0.5, 1.0],
    "yaw_deg": [0.05, 0.1, 0.25, 0.5],
    "log_focal": [0.002, 0.005, 0.01, 0.02],
}


def error_for(parameter: str, value: float) -> Correction:
    """El error de un solo parámetro, con los demás en cero."""
    if parameter not in SWEEPS:
        raise KeyError(parameter)
    return Correction(**{parameter: value})


def depth_change(
    depth: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
    range_m: tuple[float, float],
) -> dict[str, float]:
    """Diferencia relativa de profundidad contra la referencia, donde las dos tienen dato.

    Solo cuenta píxeles cuya referencia está dentro de `range_m`, el rango en el que VisionX
    anuncia metros. `cobertura` es qué parte de esos píxeles sigue teniendo dato.
    """
    with np.errstate(invalid="ignore"):
        wanted = reference_valid & (reference >= range_m[0]) & (reference <= range_m[1])
    both = wanted & valid
    if not wanted.any() or not both.any():
        nan = float("nan")
        return {"mediana_pct": nan, "p90_pct": nan, "mas_de_5_pct": nan, "cobertura": 0.0}
    relative = np.abs(depth[both] - reference[both]) / reference[both] * 100.0
    return {
        "mediana_pct": float(np.median(relative)),
        "p90_pct": float(np.percentile(relative, 90)),
        "mas_de_5_pct": float(np.mean(relative > 5.0) * 100.0),
        "cobertura": float(both.sum() / wanted.sum()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Promedio por par de cada número (ignora los pares donde no se pudo calcular)."""
    out: dict[str, Any] = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return out
