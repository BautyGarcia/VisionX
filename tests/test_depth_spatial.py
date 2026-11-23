"""Tests del ubicador por objeto de A (`depth/spatial/`, T09), sin cámara, GPU ni pesos.

Escenas sintéticas armadas a mano: un mapa de Z conocido se pasa a disparidad con la
geometría (d = fx·B/Z - doffs) y se le pone una máscara con la forma que interesa probar
(objeto compacto, calado, fino, cortado en un borde...). Así cada resultado se puede predecir.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visionx.depth.spatial import LocatorParams, ObjectLocator, build_locator
from visionx.depth.spatial.surface import (
    interior_mask,
    layer_edges,
    mask_to_depth_grid,
    nearest_layer,
)
from visionx.detection.convert import build_detection
from visionx.shared.config import load_config
from visionx.shared.contracts import (
    DepthResult,
    DepthStatus,
    DistancePolicy,
    StereoFrame,
    UncertaintyKind,
)
from visionx.shared.errors import IdentityMismatch, InvalidInput
from visionx.shared.geometry import StereoGeometry, resize_mask
from visionx.shared.interfaces import SpatialEstimator

W, H = 640, 360
G = StereoGeometry(width=W, height=H, fx=600.0, fy=600.0, cx=319.5, cy=179.5, baseline_m=0.06)
SPEAK = DistancePolicy.SPEAK_METERS
# Sin erosión ni mínimo de píxeles: para aislar cada regla en su test.
PLAIN = LocatorParams(min_valid_px=0)


def make_frame(geometry: StereoGeometry = G, frame_id: int = 0) -> StereoFrame:
    return StereoFrame(
        session_id="s",
        frame_id=frame_id,
        left=np.zeros((geometry.height, geometry.width, 3), np.uint8),
        right=np.zeros((geometry.height, geometry.width, 3), np.uint8),
        received_ns=frame_id,
        rectified=True,
        calibration_id="test",
        geometry_id=geometry.geometry_id,
    )


def make_depth(
    frame: StereoFrame,
    z: np.ndarray,
    geometry: StereoGeometry = G,
    valid: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> DepthResult:
    """DepthResult de un mapa de Z (NaN = sin medida) en `geometry`."""
    ok = np.isfinite(z) if valid is None else valid & np.isfinite(z)
    d, ok = geometry.depth_to_disparity(np.where(ok, z, np.nan).astype(np.float32), ok)
    depth, ok = geometry.disparity_to_depth(d, ok)
    status = DepthStatus.VALID if ok.any() else DepthStatus.UNMEASURABLE
    return DepthResult(
        session_id=frame.session_id,
        frame_id=frame.frame_id,
        model_id="test-depth",
        geometry_id=geometry.geometry_id,
        disparity_px=d,
        valid_mask=ok,
        status=status,
        reason=None if ok.any() else "sin píxeles",
        inference_ms=1.0,
        depth_m=depth,
        uncertainty=None if b is None else np.where(ok, b, np.nan).astype(np.float32),
        uncertainty_kind=UncertaintyKind.NONE if b is None else UncertaintyKind.LAPLACE_B,
        doffs=geometry.doffs,
    )


def make_det(frame: StereoFrame, masks: list[np.ndarray] | None, boxes=None, labels=None):
    """Detection con cajas = extensión de cada máscara (o `boxes` si se dan)."""
    if boxes is None:
        boxes = []
        for m in masks:
            rows, cols = np.nonzero(m)
            boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
    n = len(boxes)
    return build_detection(
        frame,
        model_id="test-det",
        boxes_xyxy=np.array(boxes, np.float32),
        class_names=labels or ["cup"] * n,
        scores=np.full(n, 0.9, np.float32),
        inference_ms=1.0,
        masks=None if masks is None else np.stack(masks),
    )


def rect(y0: int, y1: int, x0: int, x1: int, shape=(H, W)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def locate1(z, mask, params=PLAIN, **kw):
    frame = make_frame()
    obj, diag = ObjectLocator(G, params).locate_detailed(
        frame, make_depth(frame, z, **kw), make_det(frame, [mask])
    )[0]
    return obj, diag


# --- superficie ------------------------------------------------------------------------


def test_compact_object_on_far_background_is_not_contaminated():
    # vaso a 1 m; la máscara se pasa 3 px y toma fondo a 3 m: el frente no se mueve
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 1.0
    obj, _ = locate1(z, rect(97, 203, 297, 343))
    assert obj.policy is SPEAK and obj.z_m == pytest.approx(1.0, abs=0.01)


def test_adaptive_erosion_removes_a_near_rim_that_would_win_the_nearest_layer():
    # objeto a 1,5 m; el borde de la máscara toma un apoyo más cercano (1,2 m) que suma > 25 %
    # de los píxeles: sin erosión gana esa capa; con el interior adaptativo, el objeto
    z = np.full((H, W), 1.2, np.float32)
    z[110:190, 290:350] = 1.5
    mask = rect(100, 200, 280, 360)
    plain, _ = locate1(z, mask)
    assert plain.z_m == pytest.approx(1.2, abs=0.02)
    eroded, diag = locate1(z, mask, LocatorParams(min_valid_px=0, erode_frac=0.5, erode_max_px=15))
    assert eroded.z_m == pytest.approx(1.5, abs=0.02)
    assert diag.erosion_px > 10 and 0 < diag.coverage < 1


def test_see_through_object_reports_its_front_not_the_background():
    # silla calada: 40 % de la máscara es la silla (0,6 m), el resto el fondo a través (2,5 m)
    z = np.full((H, W), 2.5, np.float32)
    mask = rect(100, 300, 200, 300)
    frame_rows = np.zeros((H, W), bool)
    frame_rows[100:300:5, 200:300] = True  # travesaños
    frame_rows[100:300, 200:300:5] = True  # patas y varillas
    z[frame_rows & mask] = 0.6
    obj, diag = locate1(z, mask)
    assert obj.policy is SPEAK and obj.z_m == pytest.approx(0.6, abs=0.02)
    assert diag.layer_px < diag.valid_px  # el fondo quedó en otra capa


def test_thin_object_survives_the_erosion():
    # cable de 3 px de ancho a 0,8 m sobre fondo a 2 m, con erosión fuerte: no se borra
    z = np.full((H, W), 2.0, np.float32)
    z[50:300, 200:203] = 0.8
    mask = rect(50, 300, 200, 203)
    strong = LocatorParams(min_valid_px=0, erode_frac=0.5, erode_max_px=6, min_interior_px=50)
    obj, diag = locate1(z, mask, strong)
    assert obj.policy is SPEAK and obj.z_m == pytest.approx(0.8, abs=0.01)
    assert diag.interior_px > 0.3 * diag.mask_px  # queda al menos la línea central
    # 2 px de ancho (r = 1): no se toca
    z[50:300, 200:203] = 2.0
    z[50:300, 200:202] = 0.8
    obj, diag = locate1(z, rect(50, 300, 200, 202), strong)
    assert obj.z_m == pytest.approx(0.8, abs=0.01) and diag.interior_px == diag.mask_px


def test_interior_mask_never_goes_below_the_minimum():
    mask = rect(0, 10, 0, 10, (20, 20))
    inner, k = interior_mask(mask, 0.9, 100, min_px=50)  # dejaría ~4 px: se descarta
    assert inner.sum() == mask.sum() and k == 0.0
    inner, k = interior_mask(mask, 0.9, 100, min_px=1)
    assert 0 < inner.sum() < mask.sum() and k > 0
    # contra el borde de la imagen no se erosiona: lo de afuera no es fondo
    inner, _ = interior_mask(mask, 0.5, 100, 1, image_edges=(True, True, False, False))
    assert inner[0, 0] and not inner[9, 9]


def test_nearest_layer_in_disparity_ignores_sparse_noise_in_front():
    rng = np.random.default_rng(0)
    obj = rng.normal(24.0, 0.2, 950)  # d efectiva de un objeto (~1,5 m con fx·B = 36)
    noise = rng.normal(40.0, 0.3, 50)  # 5 % de píxeles sueltos más cerca
    layer = nearest_layer(np.concatenate([noise, obj]))
    assert layer is not None and layer[50:].mean() > 0.95 and not layer[:50].any()
    assert nearest_layer(np.array([], np.float64)) is None


def test_layer_edges_relative_with_a_floor_in_pixels():
    plain = layer_edges(2.0, 50.0, 0.03, 0.0)
    assert np.allclose(np.diff(plain) / plain[:-1], 0.03)  # logarítmico: 3 % en d = 3 % en Z
    floored = layer_edges(2.0, 50.0, 0.03, 0.5)
    steps = np.diff(floored)
    assert steps.min() == pytest.approx(0.5) and steps.max() > 0.5  # piso lejos, 3 % cerca
    with pytest.raises(InvalidInput):
        layer_edges(1.0, 2.0, 0.0, 0.0)


def test_diffuse_layer_gives_direction_only():
    z = np.full((H, W), 5.0, np.float32)
    z[100:200, 300:400] = np.linspace(1.0, 3.0, 100, dtype=np.float32)  # rampa sin separación
    obj, diag = locate1(z, rect(100, 200, 300, 400))
    assert obj.policy is DistancePolicy.DIRECTION_ONLY and diag.reason == "capa-difusa"
    assert obj.z_m is not None  # se informa, pero no se dice


# --- bordes ---------------------------------------------------------------------------


def test_left_and_right_borders_ask_to_reframe():
    z = np.full((H, W), 1.0, np.float32)
    for mask, reason in ((rect(100, 200, 0, 50), "borde-izquierdo"),
                         (rect(100, 200, W - 50, W), "borde-derecho")):  # fmt: skip
        obj, diag = locate1(z, mask)
        assert obj.policy is DistancePolicy.ASK_REFRAME and diag.reason == reason


def test_top_border_does_not_ask_to_reframe():
    z = np.full((H, W), 1.0, np.float32)
    obj, diag = locate1(z, rect(0, 100, 300, 340))
    assert obj.policy is SPEAK and "arriba" in diag.cut_edges


def test_bottom_border_reframes_only_when_enabled_and_the_front_continues_outside():
    # objeto que se acerca hacia abajo (lo más cercano, abajo, queda cortado)
    z = np.full((H, W), 3.0, np.float32)
    z[260:360, 300:340] = np.linspace(1.4, 1.0, 100, dtype=np.float32)[:, None]
    mask = rect(260, 360, 300, 340)
    obj, _ = locate1(z, mask)
    assert obj.policy is SPEAK  # por defecto, abajo no reencuadra
    on = LocatorParams(min_valid_px=0, reframe_bottom=True)
    obj, diag = locate1(z, mask, on)
    assert obj.policy is DistancePolicy.ASK_REFRAME and diag.reason == "borde-inferior"
    # la parte cortada es la más lejana: el frente está en cuadro y se mide
    z[260:360, 300:340] = np.linspace(1.0, 1.4, 100, dtype=np.float32)[:, None]
    obj, _ = locate1(z, mask, on)
    assert obj.policy is SPEAK and obj.z_m == pytest.approx(1.08, abs=0.03)


# --- profundidad parcial, inexistente y sin máscara ------------------------------------


def test_partial_depth_gives_direction_only():
    z = np.full((H, W), 1.0, np.float32)
    valid = np.ones((H, W), bool)
    valid[100:200, 300:370] = False  # 70 % de la máscara sin medida
    obj, diag = locate1(z, rect(100, 200, 300, 400), valid=valid)
    assert obj.policy is DistancePolicy.DIRECTION_ONLY and diag.reason == "poca-fraccion-valida"
    assert obj.valid_fraction == pytest.approx(0.3)


def test_too_few_valid_pixels_give_direction_only():
    z = np.full((H, W), 1.0, np.float32)
    obj, diag = locate1(z, rect(100, 104, 300, 305), LocatorParams(min_valid_px=30))
    assert obj.policy is DistancePolicy.DIRECTION_ONLY and diag.reason == "pocos-pixeles"


def test_no_valid_pixel_in_the_object_abstains():
    z = np.full((H, W), 1.0, np.float32)
    z[100:200, 300:400] = np.nan
    obj, diag = locate1(z, rect(100, 200, 300, 400))
    assert obj.policy is DistancePolicy.ABSTAIN and obj.z_m is None
    assert diag.reason == "sin-pixeles-validos" and diag.direction_source == "mascara"


def test_unmeasurable_depth_abstains_but_keeps_direction():
    frame = make_frame()
    depth = make_depth(frame, np.full((H, W), np.nan, np.float32))
    assert depth.status is DepthStatus.UNMEASURABLE
    mask = rect(100, 200, 500, 540)
    (obj,) = ObjectLocator(G, PLAIN).locate(frame, depth, make_det(frame, [mask]))
    assert obj.policy is DistancePolicy.ABSTAIN and obj.z_m is None and obj.angle_h_deg > 0


def test_without_masks_direction_from_box_and_abstain():
    frame = make_frame()
    box = [400.0, 50.0, 480.0, 150.0]
    depth = make_depth(frame, np.full((H, W), 1.0, np.float32))
    (pair,) = ObjectLocator(G, PLAIN).locate_detailed(frame, depth, make_det(frame, None, [box]))
    obj, diag = pair
    assert obj.policy is DistancePolicy.ABSTAIN and obj.z_m is None
    assert diag.reason == "sin-mascara" and diag.direction_source == "caja"
    assert (obj.angle_h_deg, obj.angle_v_deg) == pytest.approx(G.pixel_to_angles(440.0, 100.0))


def test_mask_that_vanishes_at_depth_resolution_abstains():
    frame = make_frame()
    small = G.resized(W // 4, H // 4)
    depth = make_depth(frame, np.full((H // 4, W // 4), 1.0, np.float32), small)
    mask = np.zeros((H, W), bool)
    mask[101, 301] = True  # un píxel que el vecino más cercano no muestrea
    det = make_det(frame, [mask])
    (pair,) = ObjectLocator(G, PLAIN, small).locate_detailed(frame, depth, det)
    assert pair[0].policy is DistancePolicy.ABSTAIN and pair[1].reason == "mascara-vacia"


# --- geometría: reescalado, doffs, ángulos e identidad ---------------------------------


@pytest.mark.parametrize("size", [(320, 180), (427, 240), (640, 360), (213, 120)])
def test_mask_to_depth_grid_matches_nearest_exact_resize(size):
    rng = np.random.default_rng(1)
    mask = np.zeros((H, W), bool)
    mask[57:211, 133:401] = rng.random((154, 268)) > 0.3
    crop, (c0, r0) = mask_to_depth_grid(mask, size)
    full = resize_mask(mask, *size)
    rebuilt = np.zeros_like(full)
    rebuilt[r0 : r0 + crop.shape[0], c0 : c0 + crop.shape[1]] = crop
    assert np.array_equal(rebuilt, full)


def test_depth_in_a_resized_geometry():
    frame = make_frame()
    half = G.resized(W // 2, H // 2)
    z = np.full((H // 2, W // 2), 2.0, np.float32)
    z[50:100, 150:170] = 1.2
    depth = make_depth(frame, z, half)
    det = make_det(frame, [rect(100, 200, 300, 340)])
    with pytest.raises(IdentityMismatch):  # sin declarar la geometría reducida
        ObjectLocator(G, PLAIN).locate(frame, depth, det)
    (obj,) = ObjectLocator(G, PLAIN, half).locate(frame, depth, det)
    assert obj.z_m == pytest.approx(1.2, abs=0.01) and obj.valid_fraction == 1.0
    # el centroide de la capa en la grilla chica da el mismo ángulo que en la grande
    assert obj.angle_h_deg == pytest.approx(G.pixel_to_angles(319.5, 149.5)[0], abs=0.1)


def test_nonzero_doffs_uses_effective_disparity():
    g = StereoGeometry(width=W, height=H, fx=100.0, fy=100.0, cx=319.5, cy=179.5,
                       baseline_m=0.06, doffs=10.0)  # fmt: skip
    frame = make_frame(g)
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 1.0  # d + doffs = 6 px -> d cruda = -4 px (TECNICO)
    depth = make_depth(frame, z, g)
    (pair,) = ObjectLocator(g, PLAIN).locate_detailed(
        frame, depth, make_det(frame, [rect(100, 200, 300, 340)])
    )
    obj, diag = pair
    assert obj.z_m == pytest.approx(1.0, abs=1e-4) and diag.d_front_px == pytest.approx(-4.0)


def test_angle_signs_and_direction_from_the_layer_centroid():
    # objeto arriba a la derecha: h > 0 (derecha), v > 0 (arriba)
    z = np.full((H, W), 3.0, np.float32)
    z[40:80, 500:560] = 1.0
    obj, _ = locate1(z, rect(40, 80, 500, 560))
    assert obj.angle_h_deg > 0 and obj.angle_v_deg > 0
    # máscara en L: la capa cercana es solo el brazo izquierdo; la dirección sale de ahí y no
    # del centro de la caja
    z = np.full((H, W), 3.0, np.float32)
    mask = rect(100, 200, 100, 120) | rect(180, 200, 100, 300)
    z[100:200, 100:120] = 1.0
    z[180:200, 120:300] = 2.0
    obj, diag = locate1(z, mask)
    assert diag.direction_source == "capa"
    expected_h = G.pixel_to_angles(109.5, 149.5)[0]
    assert obj.angle_h_deg == pytest.approx(expected_h, abs=0.2)
    assert abs(obj.angle_h_deg - diag.box_angle_h_deg) > 5


def test_frames_must_match():
    frame, other = make_frame(frame_id=0), make_frame(frame_id=1)
    depth = make_depth(other, np.full((H, W), 1.0, np.float32))
    with pytest.raises(IdentityMismatch):
        ObjectLocator(G).locate(frame, depth, make_det(frame, [rect(10, 20, 300, 310)]))


# --- protocolo, contrato, diagnóstico y config -----------------------------------------


def test_satisfies_the_spatial_estimator_protocol():
    assert isinstance(ObjectLocator(G), SpatialEstimator)


def test_never_speaks_meters_without_z_and_abstain_has_no_z():
    rng = np.random.default_rng(2)
    loc = ObjectLocator(G, LocatorParams(erode_frac=0.3, erode_max_px=4, reframe_bottom=True))
    for i in range(30):
        frame = make_frame(frame_id=i)
        z = rng.uniform(0.3, 6.0, (H, W)).astype(np.float32)
        z[rng.random((H, W)) < rng.uniform(0, 1)] = np.nan
        masks = []
        for _ in range(4):
            y0, x0 = rng.integers(0, H - 5), rng.integers(0, W - 5)
            hh, ww = rng.integers(1, 120), rng.integers(1, 200)
            m = rect(y0, y0 + hh, x0, x0 + ww) & (rng.random((H, W)) < 0.8)
            m[y0, x0] = True
            masks.append(m)
        for obj in loc.locate(frame, make_depth(frame, z), make_det(frame, masks)):
            assert obj.policy is not SPEAK or obj.z_m is not None
            assert obj.policy is not DistancePolicy.ABSTAIN or obj.z_m is None


def test_laplace_b_is_only_a_diagnostic():
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 1.0
    b = np.full((H, W), 0.7, np.float32)
    plain, diag_plain = locate1(z, rect(100, 200, 300, 340))
    with_b, diag_b = locate1(z, rect(100, 200, 300, 340), b=b)
    assert diag_plain.b_median_px is None and diag_b.b_median_px == pytest.approx(0.7)
    assert (with_b.z_m, with_b.policy, with_b.z_spread_m) == (
        plain.z_m,
        plain.policy,
        plain.z_spread_m,
    )


def test_source_map_gives_the_stereo_fraction():
    frame = make_frame()
    z = np.full((H, W), 1.0, np.float32)
    source = np.ones((H, W), np.uint8)
    source[:, 320:] = 2  # mitad derecha rellenada con monocular
    det = make_det(frame, [rect(100, 200, 300, 340)])
    ((_, diag),) = ObjectLocator(G, PLAIN).locate_detailed(
        frame, make_depth(frame, z), det, source=source
    )
    assert diag.stereo_fraction == pytest.approx(0.5)


def test_build_locator_reads_the_spatial_section():
    cfg = load_config()
    loc = build_locator(cfg, G)
    assert loc.params == LocatorParams.from_config(cfg["spatial"])
    assert loc.depth_geometry is G
    half = G.resized(W // 2, H // 2)
    assert build_locator(cfg, G, half).depth_geometry is half
    with pytest.raises(InvalidInput):
        LocatorParams.from_config({"erosion": 3})
    with pytest.raises(InvalidInput):
        LocatorParams(front_percentile=120)


def test_front_percentile_is_the_same_in_depth_and_in_disparity():
    # Z = fx·B/d es monótona: el percentil p de Z es el (100 - p) de d. No hay que elegir.
    rng = np.random.default_rng(3)
    z = rng.uniform(0.9, 1.1, 2000)
    d = G.fx * G.baseline_m / z
    p20_z = np.percentile(z, 20, method="lower")
    from_d = G.fx * G.baseline_m / np.percentile(d, 80, method="higher")
    assert math.isclose(p20_z, from_d, rel_tol=1e-12)


def test_b_factor_fills_the_90_percent_interval_around_the_front():
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 2.0
    b = np.full((H, W), 0.5, np.float32)
    obj, diag = locate1(
        z, rect(100, 200, 300, 340), LocatorParams(min_valid_px=0, b_factor=0.4), b=b
    )
    fb = G.fx * G.baseline_m
    d = fb / obj.z_m  # disparidad efectiva del frente
    t = math.log(10.0) * 0.5 * 0.4
    assert obj.z_bajo_m == pytest.approx(fb / (d + t))
    assert obj.z_alto_m == pytest.approx(fb / (d - t))
    assert obj.z_bajo_m < obj.z_m < obj.z_alto_m
    assert diag.b_median_px == pytest.approx(0.5)  # el diagnóstico sigue sin el factor


def test_without_b_factor_or_without_b_there_is_no_interval():
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 1.0
    b = np.full((H, W), 0.5, np.float32)
    off, _ = locate1(z, rect(100, 200, 300, 340), b=b)  # b_factor 0: como antes
    no_b, _ = locate1(z, rect(100, 200, 300, 340), LocatorParams(min_valid_px=0, b_factor=0.4))
    assert (off.z_bajo_m, off.z_alto_m) == (None, None)
    assert (no_b.z_bajo_m, no_b.z_alto_m) == (None, None)
    with pytest.raises(InvalidInput):
        LocatorParams(b_factor=-1.0)


def test_interval_without_upper_bound_when_b_crosses_zero_disparity():
    z = np.full((H, W), 3.0, np.float32)
    z[100:200, 300:340] = 2.5  # d ≈ 14,4 px
    b = np.full((H, W), 20.0, np.float32)
    obj, _ = locate1(z, rect(100, 200, 300, 340), LocatorParams(min_valid_px=0, b_factor=1.0), b=b)
    assert obj.z_bajo_m is not None and obj.z_bajo_m < obj.z_m
    assert obj.z_alto_m is None


def test_abstained_object_has_no_interval():
    z = np.full((H, W), np.nan, np.float32)
    b = np.full((H, W), 0.5, np.float32)
    obj, _ = locate1(z, rect(100, 200, 300, 340), LocatorParams(min_valid_px=0, b_factor=0.4), b=b)
    assert obj.z_m is None and (obj.z_bajo_m, obj.z_alto_m) == (None, None)
