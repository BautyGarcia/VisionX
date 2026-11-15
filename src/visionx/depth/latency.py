"""Latencia de profundidad en la Mac sobre un clip de la ELP (T05, M9).

Qué se mide: el componente de profundidad completo de un par ya rectificado, como lo usa la
app (`HybridEstimator.predict_detailed` o `SgbmEstimator.predict`): reescalar a la resolución
de cómputo, SGBM, monocular, alineación y armado del DepthResult. No entra la captura, el
JPEG ni la rectificación (eso es de T02 y se mide aparte en el pipeline completo).

Protocolo M9 (TECNICO): 50 pares de calentamiento descartados y al menos 300 medidos. El clip
se recorre en ciclo si tiene menos pares. La GPU (MPS) queda sincronizada porque el monocular
pasa su salida a CPU antes de devolver. P99 no se informa con n < 1000 (sería un par suelto).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from visionx.shared.errors import InvalidInput

WARMUP = 50
MEASURED = 300


def percentiles(values: Iterable[float]) -> dict[str, float]:
    """P50/P95 (y P99 solo con n >= 1000), media y n, en las unidades de entrada."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        raise InvalidInput("no hay mediciones")
    out = {"n": float(x.size), "media": float(x.mean()),
           "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))}  # fmt: skip
    if x.size >= 1000:
        out["p99"] = float(np.percentile(x, 99))
    return out


def peak_rss_mb() -> float:
    """Memoria residente máxima del proceso. macOS informa bytes; Linux, kilobytes.
    En Windows no existe `resource`: NaN (la medición M9 es en la Mac)."""
    try:
        import resource
    except ImportError:
        return math.nan
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def measure(
    frames: list[Any],
    run: Callable[[Any], dict[str, float]],
    warmup: int = WARMUP,
    measured: int = MEASURED,
) -> dict[str, dict[str, float]]:
    """Corre `run` sobre los pares en ciclo; descarta `warmup` y resume cada tiempo devuelto.

    `run(frame)` devuelve un diccionario de tiempos en ms (p. ej. total, estéreo, mono) y
    cualquier otro número por par (p. ej. alineación aceptada), que se resume igual.
    """
    if not frames:
        raise InvalidInput("no hay pares para medir")
    if measured <= 0 or warmup < 0:
        raise InvalidInput("warmup >= 0 y measured > 0")
    samples: dict[str, list[float]] = {}
    for i in range(warmup + measured):
        values = run(frames[i % len(frames)])
        if i < warmup:
            continue
        for key, val in values.items():
            samples.setdefault(key, []).append(float(val))
    return {key: percentiles(vals) for key, vals in samples.items()}
