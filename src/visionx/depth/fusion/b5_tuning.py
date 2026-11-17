"""Ajuste de B5 con el protocolo de T05, leyendo los ingredientes de la caché de Fusion (T12).

Mismo protocolo que `visionx.depth.tuning` (T05 PR 3):
- **Objetivo único: acierto-2**, media por muestra, sobre toda la GT (no responder resta).
  EPE, bad-2 y cobertura se informan.
- Se elige con **InStereo2K val**. Los mejores se repiten en pares repartidos de **train**:
  la confirmación no elige nada, solo avisa si el orden cambia en escenas que val no tiene.
- El test público no entra: este módulo solo recibe las muestras que le pasa el CLI.

Para que sea rápido no se recalcula SGBM ni el monocular: estéreo B2, mapa monocular y
alineación de B4 salen de la caché de Fusion (solo lectura, con su clave verificada). Por
muestra se ajusta una vez la recta global de cada método y después se prueban todas las
correcciones locales con esa recta. Nada se escribe en la config: el CLI deja la propuesta.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from visionx.depth.align import Alignment, AlignParams, LocalAffineParams, align
from visionx.depth.fusion import cache
from visionx.depth.fusion.cache import CacheSpec
from visionx.depth.hybrid import B5Params, combine
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.shared.errors import InvalidInput

OBJECTIVE = "acierto2"
# Grilla de la consigna (T12, 16/11): 2 x 3 x 2 x 2 = 24 combinaciones.
DEFAULT_GRID: dict[str, list[Any]] = {
    "method": ["ransac", "huber"],
    "sigma_px": [120.0, 180.0, 240.0],
    "slope_lambda_px2": [100.0, 1000.0],
    "exclude_px": [0.0, 16.0],
}
SUMMARY_KEYS = ("acierto2", "epe", "bad2", "cobertura", "ms")


@dataclass(frozen=True)
class Candidate:
    """Una combinación de la grilla (o una referencia: B4, o la recta global sola)."""

    name: str
    kind: str  # "B4" | "global" | "B5"
    params: B5Params | None = None  # None para B4

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"candidato": self.name, "tipo": self.kind}
        if self.params is not None:
            row["method"] = self.params.align.method
            row.update({k: val for k, val in asdict(self.params.local).items()})
        return row


def candidates(base: B5Params, grid: dict[str, list[Any]] | None = None) -> list[Candidate]:
    """B4, la recta global de cada método sola y la grilla de B5, en orden estable."""
    grid = grid or DEFAULT_GRID
    unknown = set(grid) - {"method", *LocalAffineParams.__dataclass_fields__}
    if unknown:
        raise InvalidInput(f"claves desconocidas en la grilla de B5: {sorted(unknown)}")
    methods = list(grid.get("method", [base.align.method]))
    out = [Candidate("B4", "B4")]
    for method in methods:
        out.append(Candidate(f"global-{method}", "global",
                             replace(base, align=replace(base.align, method=method))))  # fmt: skip
    keys = [k for k in grid if k != "method"]
    for method in methods:
        for values in itertools.product(*(grid[k] for k in keys)):
            local = replace(base.local, **{k: float(x) for k, x in zip(keys, values, strict=True)})
            params = B5Params(replace(base.align, method=method), local)
            pairs = zip(keys, values, strict=True)
            label = "-".join(f"{k.split('_')[0][:3]}{x:g}" for k, x in pairs)
            out.append(Candidate(f"B5-{method}-{label}", "B5", params))
    return out


def evaluate_sample(
    sample: Any, ing: cache.Ingredients, b4_align: AlignParams, cands: Sequence[Candidate]
) -> list[dict[str, Any]]:
    """Filas de métricas (todas las candidatas) para una muestra ya leída de la caché."""
    doffs = cache.sample_doffs(sample)
    stereo = ing.stereo()
    mono = ing.mono_relative
    gt = sample.disparity.astype(np.float32)
    lines: dict[AlignParams, Alignment] = {b4_align: ing.alignment}  # la de B4, de la caché
    rows = []
    for cand in cands:
        start = time.perf_counter()
        if cand.kind == "B4":
            out = combine("B4", stereo, mono, doffs, b4_align, alignment=ing.alignment)
        else:
            ap = cand.params.align
            if ap not in lines:  # una recta global por método y muestra
                lines[ap] = align(mono, stereo.disparity_px, stereo.valid, ap)
            variant = "B4" if cand.kind == "global" else "B5"  # "global": B4 con esa recta
            out = combine(variant, stereo, mono, doffs, ap, alignment=lines[ap],
                          local=cand.params.local)  # fmt: skip
        ms = (time.perf_counter() - start) * 1e3
        m = disparity_metrics(out.disparity_px, out.valid, gt, sample.valid)
        rows.append({"candidato": cand.name, "muestra": sample.sample_id,
                     "escena": sample.scene_id, **m, "ms": ms,
                     "alin_aceptada": float(out.alignment.accepted)})  # fmt: skip
    return rows


def run(
    refs: Iterable[Any],
    num_disparities: Callable[[Any], int],
    spec: CacheSpec,
    data_root: Any,
    cands: Sequence[Candidate],
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Recorre las muestras (de a una: no guarda los mapas) y devuelve todas las filas."""
    rows: list[dict[str, Any]] = []
    refs = list(refs)
    start = time.perf_counter()
    for i, ref in enumerate(refs, 1):
        sample = ref.load()
        ing = cache.read(sample, num_disparities(sample), spec, data_root)
        rows.extend(evaluate_sample(sample, ing, spec.align, cands))
        if i % 10 == 0 or i == len(refs):
            log(f"  [{i}/{len(refs)}] {time.perf_counter() - start:.0f} s")
    return rows


def summarize(rows: list[dict[str, Any]], cands: Sequence[Candidate]) -> list[dict[str, Any]]:
    """Una fila por candidata: media por muestra (elige) y acierto-2 por escena (se informa)."""
    table = []
    for cand in cands:
        mine = [r for r in rows if r["candidato"] == cand.name]
        if not mine:
            continue
        per_sample = aggregate(mine, key="muestra")["por_escena"]
        per_scene = aggregate(mine, key="escena")["por_escena"]
        table.append({**cand.flat(), **{k: per_sample[k] for k in SUMMARY_KEYS},
                      "acierto2_por_escena": per_scene["acierto2"],
                      "alin_aceptada": per_sample["alin_aceptada"],
                      "n_muestras": per_sample["n_muestras"]})  # fmt: skip
    return table


def ranking(table: list[dict[str, Any]], kind: str = "B5") -> list[dict[str, Any]]:
    """Candidatas de un tipo, de mejor a peor por acierto-2 (desempate: menor EPE)."""
    rows = [r for r in table if r["tipo"] == kind]
    return sorted(rows, key=lambda r: (-r[OBJECTIVE], r["epe"]))
