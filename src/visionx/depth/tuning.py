"""Ajuste por etapas de los parámetros de T05 sobre InStereo2K val (PR 3).

Qué se elige y con qué:
- **Objetivo único: acierto-2** (metrics.py), media por muestra. Un filtro que descarta
  píxeles buenos baja el acierto igual que uno que deja pasar píxeles malos, así que no hace
  falta fijar a mano una cobertura objetivo.
- **Por etapas, cada una con lo anterior ya fijado** (grillas en configs/ajuste-t05.yaml):
  1. base de SGBM (bloque, P1/P2) con B0;
  2. filtros de B1 x alineación, mirando B4 con anclas B1 (B1 existe para dar anclas y estéreo
     fiable a B4: se juzga por lo que produce ahí);
  3. WLS con B2;
  4. alineación con B4 y anclas B2. Entre 2 y 4 gana el mejor B4.
  Una grilla completa de todo junto serían miles de corridas; por etapas son ~200.
- **Confirmación:** los mejores B4 se repiten en muestras repartidas de train. No elige nada:
  avisa si el orden cambia en escenas que val no tiene.

El test público no entra acá: este módulo solo recibe los pares que le pasa el CLI (val y
train). Nada se escribe en la config: el CLI deja una propuesta y la persona la copia.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from visionx.depth.align import AlignParams, align
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.hybrid import combine
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.stereo import SgbmMatcher, SgbmParams
from visionx.shared.errors import InvalidInput

OBJECTIVE = "acierto2"
SUMMARY_KEYS = ("acierto2", "epe", "bad2", "cobertura", "ms")


@dataclass
class Pair:
    """Un par listo para el ajuste: la muestra, su rango y el monocular ya calculado."""

    sample: StereoSample
    num_disparities: int
    mono: np.ndarray | None = None  # mapa relativo de DA V2 (None si no hay híbridos)
    mono_ms: float = 0.0

    @property
    def doffs(self) -> float:
        return self.sample.geometry.doffs if self.sample.geometry else 0.0


@dataclass
class Trial:
    """Una combinación de parámetros evaluada en todos los pares."""

    stage: str
    method: str  # B0..B4
    sgbm: dict[str, Any]  # SgbmParams sin num_disparities (va por par)
    stereo_variant: str | None = None  # anclas de B3/B4
    alignment: dict[str, Any] | None = None  # AlignParams de B3/B4
    summary: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        """Una fila de CSV: identificación, parámetros y resumen."""
        row: dict[str, Any] = {"etapa": self.stage, "metodo": self.method,
                               "anclas": self.stereo_variant or ""}  # fmt: skip
        row.update({f"sgbm.{k}": val for k, val in self.sgbm.items()})
        row.update({f"alin.{k}": val for k, val in (self.alignment or {}).items()})
        row.update(self.summary)
        return row


def expand(spec: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Producto cartesiano de una grilla, en orden estable. `p1_p2_factor` se abre en dos."""
    keys = list(spec)
    combos = []
    for values in itertools.product(*(spec[k] for k in keys)):
        combo: dict[str, Any] = {}
        for key, val in zip(keys, values, strict=True):
            if key == "p1_p2_factor":
                combo["p1_factor"], combo["p2_factor"] = (int(x) for x in val)
            else:
                combo[key] = val
        combos.append(combo)
    return combos or [{}]


def _row(pair: Pair, pred: np.ndarray, valid: np.ndarray, ms: float, **extra: Any) -> dict:
    s = pair.sample
    m = disparity_metrics(pred, valid, s.disparity, s.valid)
    return {"muestra": s.sample_id, "escena": s.scene_id, **m, "ms": ms, **extra}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Media por muestra (la que elige) y acierto por escena (solo se informa)."""
    per_sample = aggregate(rows, key="muestra")["por_escena"]
    per_scene = aggregate(rows, key="escena")["por_escena"]
    out = {k: per_sample[k] for k in SUMMARY_KEYS}
    out["acierto2_por_escena"] = per_scene["acierto2"]
    if "alin_aceptada" in per_sample:
        out["alin_aceptada"] = per_sample["alin_aceptada"]
        out["fraccion_mono"] = per_sample["fraccion_mono"]
    out["n_muestras"] = per_sample["n_muestras"]
    return out


def _sgbm(base: SgbmParams, pair: Pair) -> SgbmParams:
    return replace(base, num_disparities=pair.num_disparities)


def _public(params: SgbmParams) -> dict[str, Any]:
    d = asdict(params)
    d.pop("num_disparities")  # depende del dataset/geometría, no se congela acá
    return d


def run_stereo(pairs: list[Pair], variant: str, params: SgbmParams) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        matcher = SgbmMatcher(variant, _sgbm(params, pair))
        start = time.perf_counter()
        out = matcher.compute(pair.sample.left, pair.sample.right, pair.doffs)
        ms = (time.perf_counter() - start) * 1e3
        rows.append(_row(pair, out.disparity_px, out.valid, ms))
    return rows


def run_hybrid(
    pairs: list[Pair], stereo_variant: str, params: SgbmParams, aligns: list[AlignParams]
) -> tuple[list[dict[str, Any]], list[dict[str, list[dict[str, Any]]]]]:
    """Filas del estéreo y, para cada AlignParams, filas de B3 y B4 (el estéreo se calcula
    una vez por par y se reusa en todas las alineaciones)."""
    stereo_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, list[dict[str, Any]]]] = [{"B3": [], "B4": []} for _ in aligns]
    for pair in pairs:
        if pair.mono is None:
            raise InvalidInput(f"{pair.sample.sample_id}: falta el mapa monocular")
        matcher = SgbmMatcher(stereo_variant, _sgbm(params, pair))
        start = time.perf_counter()
        stereo = matcher.compute(pair.sample.left, pair.sample.right, pair.doffs)
        stereo_ms = (time.perf_counter() - start) * 1e3
        stereo_rows.append(_row(pair, stereo.disparity_px, stereo.valid, stereo_ms))
        for j, ap in enumerate(aligns):
            start = time.perf_counter()
            al = align(pair.mono, stereo.disparity_px, stereo.valid, ap)
            align_ms = (time.perf_counter() - start) * 1e3
            for variant in ("B3", "B4"):
                out = combine(variant, stereo, pair.mono, pair.doffs, ap, alignment=al)
                ms = stereo_ms + pair.mono_ms + align_ms
                hybrid_rows[j][variant].append(
                    _row(
                        pair,
                        out.disparity_px,
                        out.valid,
                        ms,
                        alin_aceptada=float(al.accepted),
                        fraccion_mono=out.stats["fraccion_mono"],
                    )  # fmt: skip
                )
    return stereo_rows, hybrid_rows


def best(trials: list[Trial], method: str, stages: tuple[str, ...] | None = None) -> Trial:
    """El de mayor acierto-2. Empate: el primero de la grilla (orden estable)."""
    pool = [t for t in trials if t.method == method and (stages is None or t.stage in stages)]
    if not pool:
        raise InvalidInput(f"no hay ensayos de {method}")
    return max(pool, key=lambda t: (t.summary[OBJECTIVE], -pool.index(t)))


def tune(
    val: list[Pair],
    base: SgbmParams,
    align_base: AlignParams,
    grids: dict[str, Any],
    log: Callable[[str], None] = print,
) -> list[Trial]:
    """Las cuatro etapas sobre val. Devuelve todos los ensayos, en orden."""
    trials: list[Trial] = []

    def record(trial: Trial) -> Trial:
        trials.append(trial)
        s = trial.summary
        log(f"  {trial.stage} {trial.method}: acierto-2 {s['acierto2']:.2f} %, "
            f"EPE {s['epe']:.2f} px, cobertura {s['cobertura']:.0%}")  # fmt: skip
        return trial

    # 1. Base de SGBM con B0.
    log("etapa 1: base SGBM (B0)")
    for combo in expand(grids["base"]):
        params = replace(base, **combo)
        record(Trial("1-base", "B0", _public(params),
                     summary=summarize(run_stereo(val, "B0", params))))  # fmt: skip
    base = SgbmParams(**{**best(trials, "B0").sgbm, "num_disparities": base.num_disparities})

    # 2. Filtros de B1 x alineación, juzgados por B4 con anclas B1.
    log("etapa 2: filtros B1 x alineación (B4 con anclas B1)")
    aligns = [replace(align_base, **c) for c in expand(grids["alineacion"])]
    for combo in expand(grids["b1"]):
        params = replace(base, **combo)
        stereo_rows, hybrid = run_hybrid(val, "B1", params, aligns)
        record(Trial("2-b1", "B1", _public(params), summary=summarize(stereo_rows)))
        for ap, rows in zip(aligns, hybrid, strict=True):
            for variant in ("B3", "B4"):
                record(Trial("2-b1", variant, _public(params), "B1", asdict(ap),
                             summarize(rows[variant])))  # fmt: skip
    chosen_b4 = best(trials, "B4", ("2-b1",))
    base = SgbmParams(**{**chosen_b4.sgbm, "num_disparities": base.num_disparities})

    # 3. WLS con B2 (B2 = B1 elegido + WLS).
    log("etapa 3: WLS (B2)")
    for combo in expand(grids["b2"]):
        params = replace(base, **combo)
        record(Trial("3-b2", "B2", _public(params),
                     summary=summarize(run_stereo(val, "B2", params))))  # fmt: skip
    base = SgbmParams(**{**best(trials, "B2").sgbm, "num_disparities": base.num_disparities})

    # 4. Alineación con anclas B2.
    log("etapa 4: alineación (B4 con anclas B2)")
    _, hybrid = run_hybrid(val, "B2", base, aligns)
    for ap, rows in zip(aligns, hybrid, strict=True):
        for variant in ("B3", "B4"):
            record(Trial("4-b2", variant, _public(base), "B2", asdict(ap),
                         summarize(rows[variant])))  # fmt: skip
    return trials


def frozen(trials: list[Trial]) -> dict[str, Any]:
    """Propuesta para la config. El mejor B2 ya arrastra la base (etapa 1) y los filtros de B1
    (etapa 2); la alineación y la variante de anclas salen del mejor B4."""
    b4 = best(trials, "B4")
    return {"sgbm": dict(best(trials, "B2").sgbm), "alineacion": b4.alignment,
            "b4": {"estereo": b4.stereo_variant}}  # fmt: skip


def table(trials: list[Trial]) -> dict[str, dict[str, float]]:
    """B0-B4 con lo elegido, sacados de los ensayos ya corridos (mismos pares de val)."""
    b4 = best(trials, "B4")
    chosen_b1 = best(trials, "B4", ("2-b1",))
    b1 = next(t for t in trials if t.stage == "2-b1" and t.method == "B1"
              and t.sgbm == chosen_b1.sgbm)  # fmt: skip
    b3 = next(t for t in trials if t.method == "B3" and t.stage == b4.stage
              and t.sgbm == b4.sgbm and t.alignment == b4.alignment)  # fmt: skip
    picks = {"B0": best(trials, "B0"), "B1": b1, "B2": best(trials, "B2"), "B3": b3, "B4": b4}
    return {name: t.summary for name, t in picks.items()}


def ranking(trials: list[Trial], method: str, k: int) -> list[Trial]:
    pool = [t for t in trials if t.method == method]
    return sorted(pool, key=lambda t: -t.summary[OBJECTIVE])[:k]


def confirm(pairs: list[Pair], candidates: list[Trial]) -> list[dict[str, Any]]:
    """Repite B4 de cada candidato en otros pares. Devuelve acierto en val y en esos pares."""
    out = []
    for trial in candidates:
        params = SgbmParams(**trial.sgbm)  # num_disparities lo pone cada par
        ap = AlignParams(**(trial.alignment or {}))
        _, hybrid = run_hybrid(pairs, trial.stereo_variant or "B1", params, [ap])
        s = summarize(hybrid[0]["B4"])
        out.append({"etapa": trial.stage, "anclas": trial.stereo_variant,
                    "alineacion": trial.alignment, "acierto2_val": trial.summary[OBJECTIVE],
                    "acierto2_confirmacion": s[OBJECTIVE],
                    "alin_aceptada_confirmacion": s.get("alin_aceptada")})  # fmt: skip
    return out
