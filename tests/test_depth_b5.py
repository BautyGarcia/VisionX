"""Tests de B5 (T12): recta global de Huber, corrección afín local y el control fijo B5.

Sintéticos, sin pesos ni GPU: monocular falso y escenas armadas a mano. Los tests de B4 de
test_depth_hybrid.py no se tocan; acá se comprueba además que B4 no cambió con B5.
"""

import math
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest

from visionx.depth.align import (
    AlignParams,
    LocalAffineParams,
    _LowResGaussian,
    align,
    anchor_support,
    local_affine,
)
from visionx.depth.hybrid import (
    SOURCE_MONO,
    SOURCE_STEREO,
    B5Params,
    HybridEstimator,
    HybridMatcher,
    combine,
)
from visionx.depth.stereo import SgbmParams, StereoOutput
from visionx.shared.contracts import DepthStatus, StereoFrame, check_depth_geometry
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

S_TRUE, T_TRUE = 4.0, 10.0  # d = 4·m + 10
PARAMS = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5)
HUBER = replace(PARAMS, method="huber")


def plane(h: int = 60, w: int = 80):
    """Plano inclinado: d de 10 a 40 px hacia abajo; m exacto (d = 4·m + 10)."""
    rows = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    d = (10 + 30 * rows).astype(np.float32)
    return d, ((d - T_TRUE) / S_TRUE).astype(np.float32)


def stereo_output(d: np.ndarray, valid: np.ndarray) -> StereoOutput:
    return StereoOutput(np.where(valid, d, np.nan).astype(np.float32), valid, {})


def accepted_line(s: float = S_TRUE, t: float = T_TRUE):
    d, m = plane()
    a = align(m, d, np.ones(d.shape, bool), PARAMS)
    assert a.accepted
    return replace(a, scale=s, offset=t)


# --- Huber ----------------------------------------------------------------------------------


def test_huber_recovers_scale_and_offset_despite_outliers():
    d, m = plane()
    rng = np.random.default_rng(1)
    noisy = d + rng.normal(0, 0.3, d.shape).astype(np.float32)  # ruido de medida chico
    bad = rng.random(d.shape) < 0.2  # 20 % de anclas equivocadas (reflejos, patrón repetido)
    noisy[bad] += rng.uniform(-15, 15, bad.sum()).astype(np.float32)
    a = align(m, noisy, np.ones(d.shape, bool), HUBER)
    assert a.accepted, a.reason
    assert a.scale == pytest.approx(S_TRUE, rel=0.01)
    assert a.offset == pytest.approx(T_TRUE, abs=0.2)
    # Mínimos cuadrados comunes con los mismos datos se corren bastante más (por qué Huber).
    a_ls = np.linalg.lstsq(np.stack([m.ravel(), np.ones(m.size)], 1), noisy.ravel(), rcond=None)
    assert abs(a.offset - T_TRUE) < abs(a_ls[0][1] - T_TRUE) or abs(a_ls[0][1] - T_TRUE) < 0.2


def test_huber_uses_the_same_acceptance_rules_and_is_deterministic():
    d, m = plane()
    assert align(m, d + 0.3, np.ones(d.shape, bool), HUBER) == align(
        m, d + 0.3, np.ones(d.shape, bool), HUBER
    )
    assert not align(-m, d, np.ones(d.shape, bool), HUBER).accepted  # pendiente negativa
    flat = np.full(d.shape, 20.0, np.float32)
    assert "variación" in align(m, flat, np.ones(d.shape, bool), HUBER).reason
    with pytest.raises(InvalidInput):
        AlignParams(method="minimos")


# --- corrección afín local -------------------------------------------------------------------


def varying_scale_scene(h: int = 90, w: int = 240):
    """La escala monocular -> disparidad cambia suave con la columna (3,4 a 4,6): ninguna
    recta global la describe, pero en cada zona sí hay una recta local."""
    rows = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    d = (10 + 30 * rows).astype(np.float32)
    scale = np.linspace(3.4, 4.6, w, dtype=np.float32)[None, :]
    m = ((d - T_TRUE) / scale).astype(np.float32)
    holes = np.zeros(d.shape, bool)
    for x0 in (40, 110, 180):
        holes[:, x0 : x0 + 14] = True  # huecos angostos: hay anclas a los dos lados
    return d, m, holes


def test_local_affine_recovers_a_smoothly_varying_scale():
    d, m, holes = varying_scale_scene()
    loose = replace(PARAMS, inlier_threshold_px=8.0, min_holdout_inlier_ratio=0.0)
    line = align(m, d, ~holes, loose)
    assert line.accepted
    out = local_affine(m, d, ~holes, line, holes, LocalAffineParams(sigma_px=20.0))
    assert np.array_equal(out.valid, holes)  # solo corrige en los huecos
    assert np.isnan(out.disparity_px[~holes]).all()
    err_global = np.abs(line.apply(m) - d)[holes].mean()
    err_local = np.abs(out.disparity_px - d)[holes].mean()
    assert err_global > 0.5 and err_local < err_global / 4, (err_global, err_local)
    assert out.stats["fraccion_global"] == 0.0


def test_local_affine_does_not_hurt_an_exact_global_line():
    d, m = plane(90, 160)
    holes = np.zeros(d.shape, bool)
    holes[:, 60:90] = True
    out = local_affine(m, d, ~holes, accepted_line(), holes, LocalAffineParams(sigma_px=25.0))
    assert np.abs(out.disparity_px - d)[holes].max() < 1e-3


def test_anchors_next_to_holes_are_excluded():
    d, m = plane(90, 160)
    holes = np.zeros(d.shape, bool)
    holes[:, 70:90] = True
    dist = cv2.distanceTransform((~holes).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    support = anchor_support(~holes, holes, LocalAffineParams(sigma_px=20.0, exclude_px=6.0))
    assert not support.used[(dist > 0) & (dist < 6)].any()
    assert support.used[dist >= 6].all() and not support.used[holes].any()
    # Columna 80: el hueco termina en la 89; la primera ancla usada es la 95 (a 6 px del hueco).
    assert support.distance_px[:, 80].min() == pytest.approx(15.0)
    # Estéreo malo pegado al hueco (+20 px a menos de 4 px): sin exclusión arrastra la
    # corrección; con exclusión de 6 px, no.
    bad = d.copy()
    bad[(dist > 0) & (dist < 4)] += 20.0
    line = accepted_line()
    plain = local_affine(m, bad, ~holes, line, holes, LocalAffineParams(sigma_px=20.0))
    excluded = local_affine(m, bad, ~holes, line, holes,
                            LocalAffineParams(sigma_px=20.0, exclude_px=6.0))  # fmt: skip
    assert np.abs(plain.disparity_px - d)[holes].mean() > 2.0
    assert np.abs(excluded.disparity_px - d)[holes].max() < 1e-3


def test_without_nearby_anchors_the_global_line_stays():
    d, m = plane(60, 400)
    anchors = np.zeros(d.shape, bool)
    anchors[:, :60] = True  # anclas solo a la izquierda
    holes = ~anchors
    biased = d + 3.0  # el estéreo dice 3 px más: la corrección local lo copiaría cerca
    p = LocalAffineParams(sigma_px=15.0, min_density=1e-3)
    out = local_affine(m, biased, anchors, accepted_line(), holes, p)
    far = np.zeros(d.shape, bool)
    far[:, 300:] = True  # a 16 σ de la última ancla
    assert (out.support.density[far] < 1e-3).all()
    assert np.abs(out.disparity_px - d)[far].max() < 1e-4  # recta global, sin corrección
    assert np.abs(out.disparity_px - biased)[:, 60:66].max() < 0.2  # cerca sí corrige
    assert 0 < out.stats["fraccion_global"] < 1
    assert support_inf(anchor_support(np.zeros(d.shape, bool), holes, p))


def support_inf(support) -> bool:
    """Sin ninguna ancla: distancia infinita y densidad 0 (F3 lo satura en 1 y 0)."""
    return np.isinf(support.distance_px).all() and not support.density.any()


def test_low_resolution_blur_matches_a_full_resolution_gaussian():
    rng = np.random.default_rng(0)
    mask = (rng.random((200, 300)) < 0.3).astype(np.float64)
    mask[:, 150:] = 0.0  # una mitad sin anclas: el borde de la densidad tiene que estar bien
    sigma = 40.0
    blur = _LowResGaussian(mask.shape, sigma, 8.0)
    approx = blur.up(blur.down(mask))
    exact = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    assert np.abs(approx - exact).max() < 0.02  # densidad en [0, 0,3]: error < 7 %


def test_local_params_are_validated():
    with pytest.raises(InvalidInput):
        LocalAffineParams(sigma_px=0.0)
    with pytest.raises(InvalidInput):
        LocalAffineParams(max_slope_ratio=0.5)
    with pytest.raises(InvalidInput):
        LocalAffineParams(min_density=2.0)
    assert LocalAffineParams.from_config({"sigma_px": 90, "otra": 1}).sigma_px == 90.0


# --- B5 en hybrid.combine ------------------------------------------------------------------


B5 = B5Params(HUBER, LocalAffineParams(sigma_px=20.0))


def test_b5_keeps_stereo_and_corrects_holes():
    d, m, holes = varying_scale_scene()
    b5p = replace(B5, align=replace(HUBER, inlier_threshold_px=8.0, min_holdout_inlier_ratio=0.0))
    stereo = stereo_output(d, ~holes)
    out = combine("B5", stereo, m, 0.0, b5p.align, local=b5p.local)
    assert out.variant == "B5" and out.fallback is None and out.local is not None
    assert np.array_equal(out.disparity_px[~holes], d[~holes])  # estéreo intacto
    assert (out.source[~holes] == SOURCE_STEREO).all() and (out.source[holes] == SOURCE_MONO).all()
    b4 = combine("B4", stereo, m, 0.0, b5p.align)  # misma recta, sin corrección local
    err4 = np.abs(b4.disparity_px - d)[holes].mean()
    err5 = np.abs(out.disparity_px - d)[holes].mean()
    assert err5 < err4 / 4
    with pytest.raises(InvalidInput, match="corrección local"):
        combine("B5", stereo, m, 0.0, b5p.align)


def test_b5_falls_back_like_b4_and_invalidates_non_physical():
    d, m = plane()
    valid = np.zeros(d.shape, bool)
    valid[:15, :20] = True  # anclas en una sola celda: la recta se rechaza
    out = combine("B5", stereo_output(d, valid), m, 0.0, B5.align, local=B5.local)
    assert out.fallback.startswith("solo estéreo") and np.array_equal(out.valid, valid)
    assert out.local.support.used.any()  # el apoyo de anclas existe igual (F3 lo usa)
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False
    neg = combine("B5", stereo_output(d, valid), m, -100.0, B5.align, local=B5.local)
    assert not neg.source[~valid].any()  # d + doffs <= 0: se invalida, no se recorta


def test_b4_is_unchanged_by_b5():
    """B4 sigue siendo estéreo + la recta RANSAC de T05, bit a bit; `method` por defecto es
    "ransac" y no entra en la clave de la caché de Fusion (las cachés viejas siguen valiendo)."""
    from visionx.depth.fusion.cache import CacheSpec

    d, m = plane()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False
    stereo = stereo_output(d + 0.25, valid)
    out = combine("B4", stereo, m, 0.0, PARAMS)
    assert PARAMS.method == "ransac" and out.local is None
    a = align(m, stereo.disparity_px, valid, PARAMS)
    assert a == align(m, stereo.disparity_px, valid, replace(PARAMS, method="ransac"))
    expected = np.where(valid, stereo.disparity_px, a.apply(m))
    assert np.array_equal(out.disparity_px, expected)

    spec = CacheSpec("x", SgbmParams(), PARAMS, {"model_id": "m"})
    material = spec.key_material({"l.png": "0"}, 64, 0.0)
    assert "method" not in material["alineacion"]
    assert material["alineacion"] == {k: x for k, x in asdict(PARAMS).items() if k != "method"}
    huber = replace(spec, align=HUBER)
    assert huber.key_for({"l.png": "0"}, 64, 0.0) != spec.key_for({"l.png": "0"}, 64, 0.0)


def test_b5_params_inherit_alignment_from_config():
    from visionx.shared.config import load_config

    depth = load_config()["depth"]
    b5 = B5Params.from_config(depth)
    assert b5.align == replace(AlignParams.from_config(depth["alineacion"]),
                               method=depth["b5"]["method"])  # fmt: skip
    custom = {**depth, "b5": {"method": "huber", "sigma_px": 90,
                              "alineacion": {"inlier_threshold_px": 2.0}}}  # fmt: skip
    b5 = B5Params.from_config(custom)
    assert b5.align.method == "huber" and b5.align.inlier_threshold_px == 2.0
    assert b5.align.grid_cols == depth["alineacion"]["grid_cols"] and b5.local.sigma_px == 90.0
    with pytest.raises(InvalidInput, match="b5"):
        B5Params.from_config({k: x for k, x in depth.items() if k != "b5"})


class FakeMono:
    model_id = "fake-mono"

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        return np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)


def test_b5_estimator_meets_the_contract():
    rng = np.random.default_rng(0)
    left = np.repeat(np.repeat(rng.integers(0, 256, (60, 100, 3), dtype=np.uint8), 2, 0), 2, 1)
    right = np.roll(left, -6, axis=1)
    geom = StereoGeometry(width=200, height=120, fx=400.0, fy=400.0, cx=100, cy=60,
                          baseline_m=0.06)  # fmt: skip
    frame = StereoFrame(session_id="s", frame_id=1, left=left, right=right, received_ns=1,
                        rectified=True, calibration_id="c",
                        geometry_id=geom.geometry_id)  # fmt: skip
    with pytest.raises(InvalidInput):
        HybridMatcher("B5", "B1", SgbmParams(num_disparities=32), FakeMono(), HUBER)
    matcher = HybridMatcher("B5", "B1", SgbmParams(num_disparities=32), FakeMono(), B5.align,
                            B5.local)  # fmt: skip
    result, detail = HybridEstimator(matcher, geom).predict_detailed(frame)
    check_depth_geometry(result, geom)
    assert result.model_id == "b5-sgbm-b1-fake-mono"
    assert not detail.alignment.accepted  # plano frontal: sin variación para la escala
    assert result.status is DepthStatus.PARTIAL and result.reason.startswith("solo estéreo")
    assert math.isnan(detail.stats["fraccion_global"])


# --- herramienta de ajuste ----------------------------------------------------------------


def test_b5_tuning_grid_and_metrics_from_the_cache(tmp_path):
    from fusion_synthetic import NUM_DISPARITIES, make_refs, make_spec
    from fusion_synthetic import FakeMono as CacheMono

    from visionx.depth.fusion import b5_tuning
    from visionx.depth.fusion.cache import build_cache

    cands = b5_tuning.candidates(B5)
    assert len(cands) == 1 + 2 + 24 and cands[0].name == "B4"
    assert len({c.name for c in cands}) == len(cands)
    refs = make_refs(tmp_path, 2, gt_bias_px=1.0)
    spec = make_spec(tmp_path / "cache")
    build_cache(refs, lambda s: NUM_DISPARITIES, spec, CacheMono, tmp_path, log=lambda _: None)
    small = b5_tuning.candidates(B5, {"method": ["huber"], "sigma_px": [10.0, 20.0]})
    rows = b5_tuning.run(refs, lambda s: NUM_DISPARITIES, spec, tmp_path, small, log=lambda _: None)
    assert len(rows) == 2 * len(small) == 2 * 4
    table = b5_tuning.summarize(rows, small)
    best = b5_tuning.ranking(table)
    assert [r["tipo"] for r in best] == ["B5", "B5"]
    assert best[0]["acierto2"] >= best[1]["acierto2"]
    b4 = next(r for r in table if r["candidato"] == "B4")
    assert 0 < b4["cobertura"] <= 1 and b4["epe"] > 0.5  # GT con +1 px de sesgo
