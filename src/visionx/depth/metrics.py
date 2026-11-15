"""Métricas de profundidad (TECNICO, "Entender las métricas"). Las reutilizan T12, T14 y T15.

Reglas que aplican todas las funciones:
- **Error y cobertura van juntos.** El error se mide solo donde hay GT y el método respondió;
  la cobertura dice qué fracción de la GT recibió respuesta. Un método que no responde no
  tiene error cero: tiene cobertura cero y error NaN.
- **Región:** "all" usa toda la GT válida; "noc" solo los píxeles visibles en ambos ojos
  (protocolo de Middlebury y FoundationStereo). `support` restringe además a un soporte común
  cuando se comparan métodos sobre los mismos píxeles.
- **Agregado:** el reporte principal es la media por escena (cada escena pesa igual). La media
  por píxel se informa aparte para que una imagen grande no domine.

Unidades: EPE en píxeles de la resolución evaluada; bad-2 en % de píxeles con error > 2 px;
acierto-2 en % de los píxeles con GT (respondidos y con error <= 2 px);
AbsRel = |Z - Z_GT| / Z_GT (sin unidad); MAE_Z y sesgo en metros.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from visionx.shared import _validate as v
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

BAD_THRESHOLDS_PX = (1.0, 2.0, 3.0)  # bad-2 es la principal; 1 y 3 ayudan a leer la curva


def disparity_metrics(
    pred: np.ndarray,
    pred_valid: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray,
    support: np.ndarray | None = None,
    geometry: StereoGeometry | None = None,
) -> dict[str, float]:
    """Métricas de una muestra en una región. Devuelve floats; NaN si no hay con qué medir.

    `gt_valid` ya tiene que estar restringida a la región (all/noc). Con `geometry` se agregan
    las métricas en metros, convirtiendo predicción y GT con la misma geometría.
    """
    p = v.array("pred", pred, np.float32, 2)
    v.array("pred_valid", pred_valid, np.bool_, 2, p.shape)
    g = v.array("gt", gt, np.float32, 2, p.shape)
    v.array("gt_valid", gt_valid, np.bool_, 2, p.shape)
    reference = gt_valid.copy()
    if support is not None:
        v.array("support", support, np.bool_, 2, p.shape)
        reference &= support
    if not np.isfinite(g[reference]).all():
        raise InvalidInput("GT no finita dentro de la región evaluada")
    if not np.isfinite(p[pred_valid]).all():
        raise InvalidInput("predicción no finita dentro de su máscara")

    n_gt = int(reference.sum())
    both = reference & pred_valid
    n = int(both.sum())
    out: dict[str, float] = {
        "n_gt": float(n_gt),
        "n": float(n),
        "cobertura": n / n_gt if n_gt else math.nan,
    }
    err = np.abs(p[both].astype(np.float64) - g[both].astype(np.float64))
    out["epe"] = float(err.mean()) if n else math.nan
    for t in BAD_THRESHOLDS_PX:
        out[f"bad{t:g}"] = float((err > t).mean() * 100.0) if n else math.nan
    # Acierto-2: % de la GT respondida con error <= 2 px. Junta cobertura y error en un número
    # para elegir parámetros (T05 PR 3): no responder resta igual que responder mal.
    out["acierto2"] = float((err <= 2.0).sum() / n_gt * 100.0) if n_gt else math.nan

    if geometry is not None:
        z_pred, ok_pred = geometry.disparity_to_depth(p, pred_valid)
        z_gt, ok_gt = geometry.disparity_to_depth(g, reference)
        ok = both & ok_pred & ok_gt
        zp = z_pred[ok].astype(np.float64)
        zg = z_gt[ok].astype(np.float64)
        has = bool(ok.any())
        out["absrel"] = float(np.mean(np.abs(zp - zg) / zg)) if has else math.nan
        out["mae_z_m"] = float(np.mean(np.abs(zp - zg))) if has else math.nan
        out["sesgo_z_m"] = float(np.mean(zp - zg)) if has else math.nan
        out["p95_abs_z_m"] = float(np.percentile(np.abs(zp - zg), 95)) if has else math.nan
    return out


# Métricas que se promedian directamente; las cuentas (n, n_gt) se suman.
_COUNTS = ("n", "n_gt")
# Por píxel, el acierto se pondera por los píxeles con GT (n_gt), no por los respondidos (n).
_GT_WEIGHTED = ("acierto2",)


def aggregate(rows: Iterable[dict[str, Any]], key: str = "escena") -> dict[str, dict[str, float]]:
    """Resume filas por muestra en dos agregados.

    - "por_escena": media de las muestras dentro de cada escena y después media entre escenas
      (reporte principal). Las variantes de una misma escena no cuentan doble.
    - "por_pixel": cada métrica de error ponderada por los píxeles evaluados (n); la cobertura
      se recalcula como suma de n / suma de n_gt.

    Un NaN (muestra sin respuesta) se excluye del error pero no de la cobertura, que lo
    refleja con 0. `n_escenas` y `n_muestras` dejan visible el tamaño del resumen.
    """
    rows = list(rows)
    if not rows:
        raise InvalidInput("no hay filas para resumir")
    metrics = [k for k, val in rows[0].items() if isinstance(val, float) and k not in _COUNTS]

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row[key])].append(row)
    scene_means = [
        {m: _nanmean(r[m] for r in group) for m in metrics} for group in by_scene.values()
    ]
    per_scene = {m: _nanmean(s[m] for s in scene_means) for m in metrics}
    per_scene.update(n_escenas=float(len(by_scene)), n_muestras=float(len(rows)))

    total_n = sum(r["n"] for r in rows)
    total_gt = sum(r["n_gt"] for r in rows)
    per_pixel = {"cobertura": total_n / total_gt if total_gt else math.nan}
    for m in metrics:
        if m == "cobertura":
            continue
        w = "n_gt" if m in _GT_WEIGHTED else "n"
        pairs = [(r[m], r[w]) for r in rows if not math.isnan(r[m]) and r[w] > 0]
        weight = sum(n for _, n in pairs)
        per_pixel[m] = sum(val * n for val, n in pairs) / weight if weight else math.nan
    per_pixel.update(n_escenas=float(len(by_scene)), n_muestras=float(len(rows)))
    return {"por_escena": per_scene, "por_pixel": per_pixel}


def _nanmean(values: Iterable[float]) -> float:
    vals = [x for x in values if not math.isnan(x)]
    return sum(vals) / len(vals) if vals else math.nan
