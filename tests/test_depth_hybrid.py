"""Tests de T05 parte 2 (alineación, B3 y B4) sin pesos ni GPU: monocular falso y determinista."""

import math

import numpy as np
import pytest

from visionx.depth.align import AlignParams, align
from visionx.depth.hybrid import (
    SOURCE_MONO,
    SOURCE_STEREO,
    HybridEstimator,
    HybridMatcher,
    combine,
)
from visionx.depth.mono import MonoConfig, MonoModel, check_relative
from visionx.depth.stereo import SgbmParams, StereoOutput
from visionx.shared.contracts import DepthStatus, StereoFrame, check_depth_geometry
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

S_TRUE, T_TRUE = 4.0, 10.0  # d = 4·m + 10
H, W = 60, 80
PARAMS = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5)


def scene():
    """Plano inclinado: la disparidad crece hacia abajo (más cerca), de 10 a ~40 px."""
    rows = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
    d = (10 + 30 * rows).astype(np.float32)
    m = ((d - T_TRUE) / S_TRUE).astype(np.float32)  # monocular relativo exacto
    return d, m


def stereo_output(d: np.ndarray, valid: np.ndarray) -> StereoOutput:
    disp = np.where(valid, d, np.nan).astype(np.float32)
    return StereoOutput(disp, valid, {"cobertura": float(valid.mean())})


# --- Alineación ---------------------------------------------------------------------------


def test_align_recovers_scale_and_offset_despite_outliers():
    d, m = scene()
    noisy = d.copy()
    rng = np.random.default_rng(1)
    bad = rng.random(d.shape) < 0.2  # 20 % de anclas estéreo equivocadas (reflejos, etc.)
    noisy[bad] += rng.uniform(-15, 15, bad.sum()).astype(np.float32)
    a = align(m, noisy, np.ones(d.shape, bool), PARAMS)
    assert a.accepted, a.reason
    assert a.scale == pytest.approx(S_TRUE, rel=0.01)
    assert a.offset == pytest.approx(T_TRUE, abs=0.2)
    assert 0.7 < a.inlier_ratio < 0.9  # ~80 % buenas
    assert a.n_fit + a.n_holdout == a.n_anchors and a.n_holdout > 0
    assert a.holdout_p50_px < 0.5  # las celdas reservadas también quedan bien


def test_align_rejects_anchors_concentrated_in_one_region():
    d, m = scene()
    anchors = np.zeros(d.shape, bool)
    anchors[:15, :20] = True  # una sola celda de la grilla
    a = align(m, d, anchors, PARAMS)
    assert not a.accepted and "concentradas" in a.reason
    assert math.isnan(a.scale)
    with pytest.raises(InvalidInput):
        a.apply(m)


def test_align_rejects_flat_scene_without_disparity_spread():
    d = np.full((H, W), 20.0, np.float32)  # pared frontal: todas las anclas a la misma d
    m = np.random.default_rng(0).random((H, W)).astype(np.float32)
    a = align(m, d, np.ones(d.shape, bool), PARAMS)
    assert not a.accepted and "variación" in a.reason


def test_align_rejects_inverted_relation():
    d, m = scene()
    a = align(-m, d, np.ones(d.shape, bool), PARAMS)  # más cerca = m más chico: imposible
    assert not a.accepted


def test_align_rejects_when_holdout_cells_do_not_follow_the_fit():
    # El monocular acierta en la mitad de arriba y se equivoca en la de abajo: según qué celdas
    # se reserven, el ajuste no generaliza. Con min_holdout alto tiene que rechazarse.
    d, m = scene()
    m_bad = m.copy()
    m_bad[H // 2 :] = m_bad[H // 2 :][::-1]
    strict = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5,
                         min_inlier_ratio=0.0, min_holdout_inlier_ratio=0.9)  # fmt: skip
    assert not align(m_bad, d, np.ones(d.shape, bool), strict).accepted


def test_align_is_deterministic_with_seed():
    d, m = scene()
    a1 = align(m, d + 0.3, np.ones(d.shape, bool), PARAMS)
    a2 = align(m, d + 0.3, np.ones(d.shape, bool), PARAMS)
    assert a1 == a2
    assert a1.to_dict()["accepted"] is True


# --- B3 / B4 ------------------------------------------------------------------------------


def test_b4_keeps_stereo_and_fills_holes_with_aligned_mono():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False  # franja sin estéreo (pared lisa)
    stereo_d = d + 0.25  # el estéreo tiene su propio pequeño sesgo
    out = combine("B4", stereo_output(stereo_d, valid), m, 0.0, PARAMS)
    assert out.fallback is None
    assert out.valid.all()
    assert (out.source[valid] == SOURCE_STEREO).all()
    assert (out.source[~valid] == SOURCE_MONO).all()
    assert np.array_equal(out.disparity_px[valid], stereo_d[valid])  # estéreo intacto
    assert np.allclose(out.disparity_px[~valid], d[~valid] + 0.25, atol=0.1)


def test_b3_is_aligned_mono_everywhere():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False
    out = combine("B3", stereo_output(d, valid), m, 0.0, PARAMS)
    assert (out.source == SOURCE_MONO).all()
    assert np.allclose(out.disparity_px, d, atol=0.05)


def test_b4_falls_back_to_stereo_when_alignment_rejected():
    d, m = scene()
    valid = np.zeros(d.shape, bool)
    valid[:15, :20] = True  # anclas en una sola celda
    out = combine("B4", stereo_output(d, valid), m, 0.0, PARAMS)
    assert out.fallback.startswith("solo estéreo")
    assert np.array_equal(out.valid, valid)
    b3 = combine("B3", stereo_output(d, valid), m, 0.0, PARAMS)
    assert not b3.valid.any() and b3.fallback is not None


def test_combine_respects_physical_disparity_with_doffs():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False
    # Con doffs muy negativo, d + doffs <= 0 en todo el mapa: nada es físico.
    out = combine("B4", stereo_output(d, valid), m, -100.0, PARAMS)
    assert not out.source[~valid].any()


def test_combine_validates_inputs():
    d, m = scene()
    with pytest.raises(InvalidInput):
        combine("B5", stereo_output(d, np.ones(d.shape, bool)), m, 0.0, PARAMS)
    with pytest.raises(InvalidInput):
        check_relative(np.full((H, W), np.nan, np.float32), (H, W))


# --- Estimador con monocular falso ---------------------------------------------------------


class FakeMono:
    """Monocular falso: devuelve (d - t) / s para la disparidad conocida del par sintético."""

    model_id = "fake-mono"

    def __init__(self, d_true: float) -> None:
        self.d_true = d_true

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        # Una rampa suave para que haya variación de m; la escena real es frontal (d fija),
        # así que el ajuste se rechaza por falta de variación: es lo que un plano frontal da.
        ramp = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)
        return ((self.d_true - T_TRUE) / S_TRUE + ramp).astype(np.float32)


def test_hybrid_estimator_contract_and_fallback_status():
    rng = np.random.default_rng(0)
    left = rng.integers(0, 256, (60, 100, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left, 2, 0), 2, 1)
    right = np.roll(left, -6, axis=1)
    geom = StereoGeometry(width=200, height=120, fx=400.0, fy=400.0, cx=100, cy=60,
                          baseline_m=0.06)  # fmt: skip
    frame = StereoFrame(
        session_id="s", frame_id=1, left=left, right=right, received_ns=1,
        rectified=True, calibration_id="c", geometry_id=geom.geometry_id,
    )  # fmt: skip
    mono = FakeMono(6.0)
    assert isinstance(mono, MonoModel)
    matcher = HybridMatcher("B4", "B1", SgbmParams(num_disparities=32), mono, PARAMS)
    est = HybridEstimator(matcher, geom)
    result, detail = est.predict_detailed(frame)
    check_depth_geometry(result, geom)
    # Plano frontal: el estéreo mide 6 px en todos lados, sin variación para fijar la escala.
    assert not detail.alignment.accepted
    assert result.status is DepthStatus.PARTIAL and result.reason.startswith("solo estéreo")
    assert result.model_id == "b4-sgbm-b1-fake-mono"
    assert result.alignment_quality is None  # el diagnóstico vive en detail.alignment


def test_mono_config_defaults_match_default_yaml():
    from visionx.shared.config import load_config

    cfg = MonoConfig.from_config(load_config()["depth"]["mono"])
    assert cfg == MonoConfig()  # pesos fijados en un solo lugar coherente
    assert AlignParams.from_config(load_config()["depth"]["alineacion"])
