"""Tests de T05 (estéreo clásico y métricas) sin cámara ni datos: pares sintéticos."""

import math

import numpy as np
import pytest

from visionx.depth.__main__ import main as depth_main
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.stereo import (
    SgbmEstimator,
    SgbmMatcher,
    SgbmParams,
    lr_consistency,
    num_disparities_for,
    round_up_16,
)
from visionx.shared.contracts import DepthStatus, StereoFrame, check_depth_geometry
from visionx.shared.errors import IdentityMismatch, InvalidInput
from visionx.shared.geometry import StereoGeometry

D_TRUE = 6  # px
PARAMS = SgbmParams(num_disparities=32, block_size=5)


def textured_pair(h: int = 120, w: int = 200, d: int = D_TRUE, seed: int = 0):
    """Textura aleatoria; el ojo derecho ve cada punto d px más a la izquierda."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (h // 2, w // 2, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left, 2, 0), 2, 1)  # manchas de 2 px: subpíxel interpolable
    right = np.roll(left, -d, axis=1)  # right[x - d] = left[x]
    return left, right


def geometry(w: int = 200, h: int = 120, doffs: float = 0.0) -> StereoGeometry:
    return StereoGeometry(width=w, height=h, fx=400.0, fy=400.0, cx=w / 2, cy=h / 2,
                          baseline_m=0.06, doffs=doffs)  # fmt: skip


def frame(left, right, geom: StereoGeometry, rectified: bool = True) -> StereoFrame:
    return StereoFrame(session_id="s", frame_id=3, left=left, right=right, received_ns=1,
                       rectified=rectified, calibration_id="calib-test" if rectified else None,
                       geometry_id=geom.geometry_id if rectified else None)  # fmt: skip


# --- Núcleo SGBM -------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["B0", "B1", "B2"])
def test_recovers_known_disparity_in_pixels(variant):
    left, right = textured_pair()
    out = SgbmMatcher(variant, PARAMS).compute(left, right)
    assert out.disparity_px.dtype == np.float32
    good = out.disparity_px[out.valid]
    # Si no se dividiera por 16, saldría ~96 en vez de ~6.
    assert np.median(good) == pytest.approx(D_TRUE, abs=0.25)
    assert out.stats["cobertura"] > 0.5
    # Contrato interno: NaN fuera de la máscara, finito adentro.
    assert np.isnan(out.disparity_px[~out.valid]).all()
    assert np.isfinite(good).all()


def test_left_band_without_search_range_is_invalid_not_zero():
    # Las primeras num_disparities columnas no tienen dónde buscar en el ojo derecho:
    # SGBM las marca con (min_disparity - 1) * 16, que no puede llegar como medición.
    left, right = textured_pair()
    out = SgbmMatcher("B0", PARAMS).compute(left, right)
    assert not out.valid[:, : PARAMS.num_disparities - 1].any()


def test_negative_raw_disparity_kept_when_doffs_makes_it_physical():
    # min_disparity negativo: SGBM puede devolver d < 0, válida si d + doffs > 0.
    left, right = textured_pair(d=-3)
    params = SgbmParams(num_disparities=32, min_disparity=-8)
    out = SgbmMatcher("B0", params).compute(left, right, doffs=10.0)
    assert np.median(out.disparity_px[out.valid]) == pytest.approx(-3, abs=0.25)
    without = SgbmMatcher("B0", params).compute(left, right, doffs=0.0)
    assert not without.valid.any()  # d + 0 <= 0: sin sentido físico


def test_lr_check_samples_right_map_at_x_minus_d():
    h, w = 4, 20
    dl = np.full((h, w), 5.0, np.float32)
    vl = np.ones((h, w), bool)
    # Mapa derecho que solo coincide si se lee en x - d: dR(x') = 5 donde x' = x - 5.
    cols = np.arange(w)
    dr = np.where(cols[None, :] < w - 5, 5.0, 50.0).astype(np.float32).repeat(h, 0)
    vr = np.ones((h, w), bool)
    ok = lr_consistency(dl, vl, dr, vr, max_diff_px=1.0)
    assert not ok[:, :5].any()  # x - d cae fuera de la imagen
    assert ok[:, 5:].all()  # x - 5 <= w - 6 < w - 5: todos leen 5
    # Leer en x + d (el error típico de convención) daría inconsistentes las últimas 10
    # columnas: el test distingue las dos convenciones.
    wrong = dr[:, np.clip(cols + 5, 0, w - 1)]
    assert not (np.abs(dl - wrong) <= 1.0)[:, -10:].any()


def test_lr_check_removes_inconsistent_and_occluded_pixels():
    left, right = textured_pair()
    # Un parche que cambia solo en el ojo derecho: coincidencias falsas en esa zona.
    right = right.copy()
    right[40:80, 100:140] = np.random.default_rng(9).integers(0, 256, (40, 40, 3), np.uint8)
    b0 = SgbmMatcher("B0", PARAMS).compute(left, right)
    b1 = SgbmMatcher("B1", PARAMS).compute(left, right)
    assert b1.stats["lr_descartada"] > 0
    assert b1.valid.sum() < b0.valid.sum()
    assert not (b1.valid & ~b0.valid).any()  # B1 solo filtra, no inventa


def test_params_validate():
    with pytest.raises(InvalidInput):
        SgbmParams(num_disparities=40)  # no múltiplo de 16
    with pytest.raises(InvalidInput):
        SgbmParams(block_size=4)
    with pytest.raises(InvalidInput):
        SgbmParams(p1_factor=32, p2_factor=8)
    with pytest.raises(InvalidInput):
        SgbmMatcher("B9", PARAMS)
    p = SgbmParams.from_config({"block_size": 7, "otra_clave": 1}, num_disparities=48)
    assert (p.block_size, p.num_disparities) == (7, 48)


def test_search_range_from_geometry():
    # fx = 400 px, B = 0,06 m: a 0,5 m, d = 48 px -> 49 valores (0..48) -> 64.
    assert num_disparities_for(geometry(), 0.5) == 64
    assert num_disparities_for(geometry(), 1.0) == 32  # d = 24 -> 25 valores
    assert round_up_16(265) == 272
    with pytest.raises(InvalidInput):
        num_disparities_for(geometry(), 0.0)


# --- Estimador (DepthEstimator) ----------------------------------------------------------


def test_estimator_result_matches_geometry_and_contract():
    left, right = textured_pair()
    geom = geometry(doffs=0.5)
    est = SgbmEstimator("B1", PARAMS, geom)
    res = est.predict(frame(left, right, geom))
    assert (res.session_id, res.frame_id, res.model_id) == ("s", 3, "sgbm-b1")
    assert res.status is DepthStatus.VALID
    check_depth_geometry(res, geom)  # depth_m coincide con fx*B/(d+doffs)
    z = 400.0 * 0.06 / (D_TRUE + 0.5)
    assert np.nanmedian(res.depth_m) == pytest.approx(z, rel=0.05)


def test_estimator_rejects_unrectified_or_other_geometry():
    left, right = textured_pair()
    geom = geometry()
    est = SgbmEstimator("B0", PARAMS, geom)
    with pytest.raises(InvalidInput):
        est.predict(frame(left, right, geom, rectified=False))
    other = geometry(doffs=1.0)
    with pytest.raises(IdentityMismatch):
        est.predict(frame(left, right, other))


def test_estimator_compute_size_works_in_resized_geometry():
    left, right = textured_pair(h=240, w=400, d=12)
    geom = geometry(w=400, h=240)
    est = SgbmEstimator("B0", PARAMS, geom, compute_size=(200, 120))
    res = est.predict(frame(left, right, geom))
    assert res.size == (200, 120)
    assert res.geometry_id == geom.resized(200, 120).geometry_id
    # Mitad de ancho: la disparidad en px se reduce a la mitad; la profundidad no cambia.
    assert np.nanmedian(res.disparity_px) == pytest.approx(6, abs=0.3)


def test_estimator_reports_partial_and_unmeasurable():
    left, right = textured_pair()
    geom = geometry()
    partial = SgbmEstimator("B1", PARAMS, geom, min_coverage=0.99).predict(frame(left, right, geom))
    assert partial.status is DepthStatus.PARTIAL and "cobertura" in partial.reason
    flat = np.full_like(left, 128)  # sin textura: SGBM no encuentra nada confiable
    none = SgbmEstimator("B1", PARAMS, geom).predict(frame(flat, flat, geom))
    assert none.status is DepthStatus.UNMEASURABLE
    assert not none.valid_mask.any()


# --- Métricas ----------------------------------------------------------------------------


def test_metrics_epe_bad2_and_coverage():
    gt = np.full((2, 4), 10.0, np.float32)
    gt_valid = np.ones((2, 4), bool)
    pred = np.array([[10, 11, 13, np.nan], [10, 10, 10, np.nan]], np.float32)
    pred_valid = np.isfinite(pred)
    m = disparity_metrics(pred, pred_valid, gt, gt_valid)
    assert m["cobertura"] == pytest.approx(6 / 8)
    assert m["epe"] == pytest.approx((0 + 1 + 3) / 6)
    assert m["bad2"] == pytest.approx(100 / 6)  # solo el error de 3 px supera 2
    assert m["bad1"] == pytest.approx(100 / 6)  # 1 px no supera 1


def test_metrics_failure_is_nan_error_and_zero_coverage():
    gt = np.full((2, 2), 5.0, np.float32)
    nothing = np.full((2, 2), np.nan, np.float32)
    m = disparity_metrics(nothing, np.zeros((2, 2), bool), gt, np.ones((2, 2), bool))
    assert m["cobertura"] == 0.0
    assert math.isnan(m["epe"])  # nunca EPE 0 por no responder


def test_metrics_in_meters_with_doffs():
    geom = StereoGeometry(width=2, height=1, fx=100.0, fy=100.0, cx=1, cy=0,
                          baseline_m=0.06, doffs=10.0)  # fmt: skip
    gt = np.array([[-4.0, -4.0]], np.float32)  # d + doffs = 6 -> Z = 1 m
    pred = np.array([[-4.0, -2.0]], np.float32)  # Z = 1 m y 0,75 m
    m = disparity_metrics(pred, np.ones((1, 2), bool), gt, np.ones((1, 2), bool), geometry=geom)
    assert m["absrel"] == pytest.approx(0.125)
    assert m["sesgo_z_m"] == pytest.approx(-0.125)


def test_aggregate_scene_mean_does_not_double_count_variants():
    rows = [
        {"escena": "A", "n": 100.0, "n_gt": 100.0, "epe": 1.0, "cobertura": 1.0},
        {"escena": "A", "n": 100.0, "n_gt": 100.0, "epe": 3.0, "cobertura": 1.0},
        {"escena": "B", "n": 50.0, "n_gt": 100.0, "epe": 10.0, "cobertura": 0.5},
        {"escena": "C", "n": 0.0, "n_gt": 100.0, "epe": math.nan, "cobertura": 0.0},
    ]
    s = aggregate(rows)
    assert s["por_escena"]["epe"] == pytest.approx((2.0 + 10.0) / 2)  # C no tiene error
    assert s["por_escena"]["cobertura"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert s["por_escena"]["n_escenas"] == 3
    assert s["por_pixel"]["epe"] == pytest.approx((100 + 300 + 500) / 250)
    assert s["por_pixel"]["cobertura"] == pytest.approx(250 / 400)


# --- CLI ---------------------------------------------------------------------------------


def test_cli_refuses_test_group_without_freeze_flag(tmp_path, capsys):
    code = depth_main(["evaluar", "B1", "middeval3", "--grupo", "test",
                       "--datos", str(tmp_path), "--salida", str(tmp_path)])  # fmt: skip
    assert code == 2
    assert "--test-congelado" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())  # no escribió resultados
