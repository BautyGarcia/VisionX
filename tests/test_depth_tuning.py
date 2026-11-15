"""Tests de T05 parte 3: acierto-2, ajuste por etapas y medición de latencia, sin datos ni GPU."""

import math

import numpy as np
import pytest

from visionx.depth.align import AlignParams, align
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.hybrid import combine
from visionx.depth.latency import measure, percentiles
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.stereo import SgbmParams, StereoOutput
from visionx.depth.tuning import Pair, best, expand, frozen, table, tune
from visionx.shared.errors import InvalidInput

H, W = 72, 128


def synthetic_pair(seed: int) -> Pair:
    """Textura al azar con disparidad que crece hacia abajo (8 a 19 px) y monocular exacto."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (H // 2, W // 2, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left, 2, 0), 2, 1)
    d_rows = 8 + (np.arange(H) * 12 // H)
    right = np.stack([np.roll(left[y], -int(d_rows[y]), axis=0) for y in range(H)])
    disparity = np.repeat(d_rows[:, None], W, 1).astype(np.float32)
    valid = np.arange(W)[None, :] >= d_rows[:, None]  # sin correspondencia en el borde izq.
    disparity[~valid] = np.nan
    sample = StereoSample("sint", f"s{seed}", f"escena{seed % 2}", left, right, disparity, valid)
    mono = ((np.repeat(d_rows[:, None], W, 1) - 10.0) / 4.0).astype(np.float32)
    return Pair(sample, 32, mono, mono_ms=1.0)


GRIDS = {
    "base": {"block_size": [3, 5], "p1_p2_factor": [[8, 32]]},
    "b1": {"uniqueness_ratio": [10], "speckle_window": [0, 100], "lr_max_diff_px": [1.0]},
    "alineacion": {
        "inlier_threshold_px": [1.0, 2.0],
        "min_inlier_ratio": [0.5],
        "min_holdout_inlier_ratio": [0.5],
    },  # fmt: skip
    "b2": {"wls_lambda": [8000.0], "wls_sigma": [1.5], "wls_min_confidence": [128.0]},
}
ALIGN = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5)


# --- acierto-2 ----------------------------------------------------------------------------


def test_acierto2_joins_coverage_and_error():
    gt = np.full((4, 4), 10.0, np.float32)
    gt_valid = np.ones((4, 4), bool)
    pred = gt.copy()
    pred[0] += 5.0  # una fila respondida pero mal
    valid = np.ones((4, 4), bool)
    valid[3] = False  # una fila sin responder
    m = disparity_metrics(pred, valid, gt, gt_valid)
    assert m["acierto2"] == pytest.approx(50.0)  # 8 de 16: ni la fila mal ni la vacía suman
    silent = disparity_metrics(pred, np.zeros((4, 4), bool), gt, gt_valid)
    assert silent["acierto2"] == 0.0 and math.isnan(silent["epe"])  # no responder no es NaN


def test_acierto2_per_pixel_is_weighted_by_gt_pixels():
    rows = [
        {"muestra": "a", "escena": "a", "n": 0.0, "n_gt": 300.0, "cobertura": 0.0,
         "epe": math.nan, "acierto2": 0.0},
        {"muestra": "b", "escena": "b", "n": 100.0, "n_gt": 100.0, "cobertura": 1.0,
         "epe": 0.5, "acierto2": 100.0},
    ]  # fmt: skip
    agg = aggregate(rows)
    assert agg["por_pixel"]["acierto2"] == pytest.approx(25.0)  # 100 de 400 píxeles con GT
    assert agg["por_escena"]["acierto2"] == pytest.approx(50.0)


# --- grillas y etapas --------------------------------------------------------------------


def test_expand_opens_p1_p2_and_keeps_order():
    combos = expand({"block_size": [3, 5], "p1_p2_factor": [[4, 16], [8, 32]]})
    assert combos[0] == {"block_size": 3, "p1_factor": 4, "p2_factor": 16}
    assert combos[-1] == {"block_size": 5, "p1_factor": 8, "p2_factor": 32}
    assert len(combos) == 4
    assert expand({}) == [{}]


def test_combine_reuses_a_precomputed_alignment():
    pair = synthetic_pair(0)
    d = np.where(pair.sample.valid, pair.sample.disparity, np.nan).astype(np.float32)
    stereo = StereoOutput(d, pair.sample.valid.copy(), {})
    al = align(pair.mono, d, stereo.valid, ALIGN)
    a = combine("B4", stereo, pair.mono, 0.0, ALIGN)
    b = combine("B4", stereo, pair.mono, 0.0, ALIGN, alignment=al)
    assert a.alignment == b.alignment
    assert np.array_equal(a.valid, b.valid)


def test_tune_runs_all_stages_and_freezes_from_val_only():
    pairs = [synthetic_pair(s) for s in range(3)]
    base = SgbmParams(num_disparities=32)
    trials = tune(pairs, base, ALIGN, GRIDS, log=lambda _: None)
    stages = {t.stage for t in trials}
    assert stages == {"1-base", "2-b1", "3-b2", "4-b2"}
    # 2 bases + (2 B1 + 2x2 alineaciones x B3/B4) + 1 WLS + 2 alineaciones x B3/B4
    assert len(trials) == 2 + (2 + 2 * 2 * 2) + 1 + 2 * 2
    proposal = frozen(trials)
    assert set(proposal) == {"sgbm", "alineacion", "b4"}
    assert "num_disparities" not in proposal["sgbm"]  # va por dataset/geometría
    assert proposal["b4"]["estereo"] in ("B1", "B2")
    assert proposal["sgbm"]["block_size"] == best(trials, "B0").sgbm["block_size"]
    summary = table(trials)
    assert list(summary) == ["B0", "B1", "B2", "B3", "B4"]
    # Escena con monocular exacto: B4 completa los huecos del estéreo y acierta más que B1.
    assert summary["B4"]["acierto2"] > summary["B1"]["acierto2"]
    assert summary["B4"]["alin_aceptada"] == 1.0


def test_tune_is_deterministic():
    pairs = [synthetic_pair(s) for s in range(2)]
    base = SgbmParams(num_disparities=32)
    a = tune(pairs, base, ALIGN, GRIDS, log=lambda _: None)
    b = tune(pairs, base, ALIGN, GRIDS, log=lambda _: None)
    strip = lambda ts: [{k: v for k, v in t.flat().items() if k != "ms"} for t in ts]  # noqa: E731
    assert strip(a) == strip(b)


# --- latencia ---------------------------------------------------------------------------


def test_measure_discards_warmup_and_cycles_frames():
    seen = []

    def run(frame):
        seen.append(frame)
        return {"total_ms": 100.0 if len(seen) <= 2 else 10.0}  # el calentamiento es lento

    stats = measure(["a", "b", "c"], run, warmup=2, measured=5)
    assert seen == ["a", "b", "c", "a", "b", "c", "a"]
    assert stats["total_ms"]["n"] == 5 and stats["total_ms"]["p95"] == 10.0


def test_percentiles_hide_p99_below_1000_samples():
    assert "p99" not in percentiles(range(300))
    assert "p99" in percentiles(range(1000))
    with pytest.raises(InvalidInput):
        percentiles([])
