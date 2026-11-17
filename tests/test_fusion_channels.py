"""Tests de los canales de Fusion (T12) y del error L/R de stereo.py: sintéticos, sin torch."""

from dataclasses import replace

import numpy as np
import pytest

from visionx.depth.align import AlignParams, LocalAffineParams
from visionx.depth.fusion.channels import (
    CHANNELS_VERSION,
    F0_CHANNELS,
    F1_CHANNELS,
    F3_CHANNELS,
    ChannelParams,
    build_inputs,
    check_channels,
    describe_channels,
    fusion_bases,
)
from visionx.depth.hybrid import B5Params, combine
from visionx.depth.stereo import SgbmMatcher, SgbmParams, StereoOutput, lr_consistency, lr_error
from visionx.shared.errors import InvalidInput

H, W = 60, 80
S_TRUE, T_TRUE = 4.0, 10.0  # monocular exacto: d = 4·m + 10
ALIGN = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5)
IDX = {name: i for i, name in enumerate(F1_CHANNELS)}


def scene():
    """Plano inclinado: la disparidad crece hacia abajo, de 10 a 40 px. m exacto."""
    rows = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
    d = (10 + 30 * rows).astype(np.float32)
    m = ((d - T_TRUE) / S_TRUE).astype(np.float32)
    return d, m


def images(seed: int = 0):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return left, right


def stereo_output(d, valid, lr=None) -> StereoOutput:
    disp = np.where(valid, d, np.nan).astype(np.float32)
    if lr is None:  # error L/R chico donde hay estéreo, sin dato afuera
        lr = np.where(valid, 0.5, np.nan).astype(np.float32)
    return StereoOutput(disp, valid, {}, lr)


def inputs_for(stereo, m, left=None, right=None, variant="F1", doffs=0.0, params=None):
    if left is None:
        left, right = images()
    base = combine("B4", stereo, m, doffs, ALIGN)
    return build_inputs(left, right, stereo, base, m, doffs, variant, params), base


# --- lr_error en stereo.py ------------------------------------------------------------------


def random_lr_maps(seed: int = 0):
    rng = np.random.default_rng(seed)
    dl = rng.uniform(0, 12, (20, 30)).astype(np.float32)
    dr = rng.uniform(0, 12, (20, 30)).astype(np.float32)
    vl = rng.random(dl.shape) < 0.8
    vr = rng.random(dr.shape) < 0.8
    dl[~vl] = np.nan  # como sale de SGBM: NaN fuera de la máscara
    dr[~vr] = np.nan
    return dl, vl, dr, vr


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0, 3.0])
def test_lr_error_matches_lr_consistency(threshold):
    dl, vl, dr, vr = random_lr_maps()
    err = lr_error(dl, vl, dr, vr)
    assert err.dtype == np.float32 and err.shape == dl.shape
    consistent = lr_consistency(dl, vl, dr, vr, threshold)
    computable = np.isfinite(err)
    # Consistente ⇔ calculable y error <= umbral.
    assert np.array_equal(consistent, computable & (np.where(computable, err, np.inf) <= threshold))


def test_lr_error_is_nan_where_not_computable():
    # Una fila de 6 px, a mano.
    dl = np.array([[0.0, 2.0, 5.0, 1.0, np.nan, 1.0]], np.float32)
    vl = np.array([[True, True, True, True, False, True]])
    dr = np.array([[0.0, 1.5, 7.0, 1.0, 1.0, np.nan]], np.float32)
    vr = np.array([[True, True, True, True, True, False]])
    err = lr_error(dl, vl, dr, vr)
    # x=0: d=0 → dR(0)=0 → 0. x=1: d=2 → x−d=−1 fuera. x=2: d=5 → fuera.
    # x=3: d=1 → dR(2)=7 → 6. x=4: dL inválida. x=5: d=1 → dR(4)=1 → 0.
    expected = np.array([[0.0, np.nan, np.nan, 6.0, np.nan, 0.0]], np.float32)
    np.testing.assert_array_equal(err, expected)
    # dR inválida en el punto de llegada: x=1 con d=-4 iría a x=5, que no es válido en dR.
    dl2 = np.array([[0.0, -4.0, 0.0, 0.0, 0.0, 0.0]], np.float32)
    assert np.isnan(lr_error(dl2, np.ones_like(vl), dr, vr)[0, 1])


def textured_pair(h: int = 120, w: int = 200, d: int = 6):
    rng = np.random.default_rng(0)
    left = rng.integers(0, 256, (h // 2, w // 2, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left, 2, 0), 2, 1)
    return left, np.roll(left, -d, axis=1)


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_sgbm_b1_b2_fill_lr_error(variant):
    left, right = textured_pair()
    params = SgbmParams(num_disparities=32)
    out = SgbmMatcher(variant, params).compute(left, right)
    err = out.lr_error_px
    assert err is not None and err.shape == left.shape[:2] and err.dtype == np.float32
    computable = np.isfinite(err)
    assert computable.mean() > 0.5
    assert (err[computable] >= 0).all()
    # La banda izquierda sin rango de búsqueda no tiene SGBM crudo: sin dato.
    assert not computable[:, : params.num_disparities - 1].any()
    if variant == "B1":  # todo lo que B1 dejó pasó el chequeo L/R
        assert (err[out.valid] <= params.lr_max_diff_px).all()


def test_sgbm_b0_has_no_lr_error():
    left, right = textured_pair()
    out = SgbmMatcher("B0", SgbmParams(num_disparities=32)).compute(left, right)
    assert out.lr_error_px is None


# --- Orden y forma de los canales -----------------------------------------------------------


def test_channel_order_count_and_names():
    assert len(F0_CHANNELS) == 10 and len(F1_CHANNELS) == 11
    assert F1_CHANNELS[:10] == F0_CHANNELS and F1_CHANNELS[10] == "valid_mono"
    assert F0_CHANNELS == ("rgb_r", "rgb_g", "rgb_b", "disp_estereo", "valid_estereo",
                           "error_lr", "error_foto", "gradiente", "disp_mono",
                           "dif_estereo_mono")  # fmt: skip
    assert CHANNELS_VERSION == "fusion-canales-2"  # 2: agregó F3 sin tocar F0/F1
    assert F3_CHANNELS[:11] == F1_CHANNELS and len(F3_CHANNELS) == 14
    d, m = scene()
    stereo = stereo_output(d, np.ones(d.shape, bool))
    for variant, names in (("F0", F0_CHANNELS), ("F1", F1_CHANNELS)):
        x, _ = inputs_for(stereo, m, variant=variant)
        assert x.names == names and x.variant == variant
        assert x.channels.shape == (len(names), H, W) and x.channels.dtype == np.float32
        assert x.base_norm.shape == (H, W) and x.base_norm.dtype == np.float32
        assert x.base_valid.dtype == np.bool_
        assert [row["nombre"] for row in describe_channels(variant)] == list(names)
    assert all(row["unidad"] and row["tratamiento"] for row in describe_channels("F1"))


def test_rgb_channels_are_rgb_centered():
    d, m = scene()
    left, right = images()
    x, _ = inputs_for(stereo_output(d, np.ones(d.shape, bool)), m, left, right)
    # BGR público -> RGB: el canal 1 (rojo) es el último de BGR.
    np.testing.assert_allclose(x.channels[IDX["rgb_r"]], left[..., 2] / 255.0 - 0.5, atol=1e-6)
    np.testing.assert_allclose(x.channels[IDX["rgb_b"]], left[..., 0] / 255.0 - 0.5, atol=1e-6)


def test_gradient_is_zero_on_flat_image_and_capped():
    d, m = scene()
    flat = np.full((H, W, 3), 128, np.uint8)
    x, _ = inputs_for(stereo_output(d, np.ones(d.shape, bool)), m, flat, flat)
    assert not x.channels[IDX["gradiente"]].any()
    # Escalón vertical de gris 0 a g: Sobel 3x3 da 4·g/255 en las columnas del borde.
    for level, expected in ((51, 4 * 0.2 * 0.25), (255, 1.0)):  # 4·0,2·0,25 y 4·1·0,25
        step = np.zeros((H, W, 3), np.uint8)
        step[:, W // 2 :] = level
        x, _ = inputs_for(stereo_output(d, np.ones(d.shape, bool)), m, step, step)
        g = x.channels[IDX["gradiente"]]
        assert g[:, W // 2 - 1 : W // 2 + 1] == pytest.approx(expected, abs=1e-6)
        assert not g[:, : W // 2 - 1].any() and not g[:, W // 2 + 1 :].any()
    # Esquina de un cuadrante blanco: gx = gy = 3 → |∇| ≈ 4,24 → × 0,25 = 1,06 → tope 1.
    corner = np.zeros((H, W, 3), np.uint8)
    corner[: H // 2, : W // 2] = 255
    x, _ = inputs_for(stereo_output(d, np.ones(d.shape, bool)), m, corner, corner)
    g = x.channels[IDX["gradiente"]]
    assert g.max() == 1.0 and g[H // 2 - 1, W // 2 - 1] == 1.0


# --- Finitud, unidades y recortes -----------------------------------------------------------


def test_channels_are_finite_with_nan_outside_masks():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False  # franja sin estéreo; ahí la disparidad es NaN
    lr = np.where(valid, 0.3, np.nan).astype(np.float32)
    lr[::7, ::5] = np.nan  # L/R sin dato aun dentro del estéreo (p. ej. WLS rellenó)
    stereo = stereo_output(d, valid, lr)
    assert np.isnan(stereo.disparity_px).any() and np.isnan(stereo.lr_error_px).any()
    for variant in ("F0", "F1"):
        x, base = inputs_for(stereo, m, variant=variant)
        assert np.isfinite(x.channels).all() and np.isfinite(x.base_norm).all()
    cap = ChannelParams().lr_cap
    assert (x.channels[IDX["error_lr"]][np.isnan(lr)] == cap).all()
    expected = min(0.3 / W * ChannelParams().error_gain, cap)
    assert x.channels[IDX["error_lr"]][np.isfinite(lr)] == pytest.approx(expected)


def test_disparity_units_are_divided_by_reference_width():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    valid[:, 30:50] = False
    stereo = stereo_output(d + 0.25, valid)
    x, base = inputs_for(stereo, m)
    assert x.ref_width == W
    ch4 = x.channels[IDX["disp_estereo"]]
    np.testing.assert_allclose(ch4[valid], stereo.disparity_px[valid] / W, rtol=1e-6)
    assert not ch4[~valid].any()
    assert x.channels[IDX["valid_estereo"]][valid].all()
    np.testing.assert_allclose(x.base_norm[base.valid], base.disparity_px[base.valid] / W,
                               rtol=1e-6)  # fmt: skip
    assert not x.base_norm[~base.valid].any()
    np.testing.assert_array_equal(x.base_valid, base.valid)


def test_crop_keeps_reference_width():
    # Se arma con el frame entero y se recorta después: un parche conserva d/W_ref del frame,
    # no d/W_parche. Volver a px exige multiplicar por ref_width, no por el ancho del parche.
    d, m = scene()
    stereo = stereo_output(d, np.ones(d.shape, bool))
    x, _ = inputs_for(stereo, m)
    c0, c1 = 20, 52  # parche de 32 px de ancho
    patch = x.channels[:, :, c0:c1]
    patch_base = x.base_norm[:, c0:c1]
    assert x.ref_width == W != patch.shape[2]
    np.testing.assert_allclose(patch[IDX["disp_estereo"]] * x.ref_width, d[:, c0:c1], rtol=1e-6)
    np.testing.assert_allclose(patch_base * x.ref_width, d[:, c0:c1], rtol=1e-6)
    # Renormalizar por el ancho del parche daría otra escala: eso es lo que NO se hace.
    assert not np.allclose(patch[IDX["disp_estereo"]], d[:, c0:c1] / patch.shape[2])


# --- Error fotométrico --------------------------------------------------------------------


def test_photometric_error_zero_with_correct_disparity():
    d_true = 6
    left, _ = images()
    right = np.roll(left, -d_true, axis=1)  # right[x − d] = left[x]
    _, m = scene()
    d = np.full((H, W), float(d_true), np.float32)
    stereo = stereo_output(d, np.ones(d.shape, bool))
    x, base = inputs_for(stereo, m, left, right)
    assert not base.alignment.accepted  # plano frontal: B4 = estéreo
    photo = x.channels[IDX["error_foto"]]
    assert np.abs(photo).max() < 1e-6
    # Disparidad equivocada (4 px en vez de 6): la correspondencia no se parece.
    wrong = stereo_output(np.full((H, W), 4.0, np.float32), np.ones(d.shape, bool))
    x_wrong, _ = inputs_for(wrong, m, left, right)
    photo_wrong = x_wrong.channels[IDX["error_foto"]]
    assert photo_wrong[:, 10:].mean() > 0.1
    assert (photo_wrong >= 0).all() and (photo_wrong <= 1).all()


def test_photometric_error_zero_outside_image_and_outside_base():
    left, right = images()
    _, m = scene()
    d = np.full((H, W), 6.0, np.float32)
    valid = np.ones(d.shape, bool)
    valid[:, 40:50] = False
    x, base = inputs_for(stereo_output(d, valid), m, left, right)
    photo = x.channels[IDX["error_foto"]]
    assert not photo[:, :6].any()  # x − 6 < 0: fuera de la imagen derecha
    assert not photo[~base.valid].any()
    assert photo[:, 6:40].min() > 0  # imágenes independientes: hay error donde se compara


# --- Monocular y diferencia ---------------------------------------------------------------


def test_stereo_minus_mono_sign_and_clipping():
    d, m = scene()
    stereo_d = d.copy()
    stereo_d[5:10, 5:10] += 0.5  # estéreo un poco por encima del monocular
    stereo_d[40:45, 60:65] -= 5.0  # estéreo muy por debajo: satura en −1
    stereo_d[20:25, 30:35] += 5.0  # muy por encima: satura en +1
    valid = np.ones(d.shape, bool)
    valid[:, 70:] = False  # franja solo monocular
    x, base = inputs_for(stereo_output(stereo_d, valid), m)
    assert base.alignment.accepted, base.alignment.reason
    diff = x.channels[IDX["dif_estereo_mono"]]
    gain = ChannelParams().error_gain
    assert diff[5:10, 5:10] == pytest.approx(0.5 / W * gain, abs=0.02)  # +0,2: estéreo − mono
    assert (diff[40:45, 60:65] == -1.0).all()
    assert (diff[20:25, 30:35] == 1.0).all()
    assert np.abs(diff).max() <= 1.0
    assert not diff[:, 70:].any()  # solo donde existen las dos fuentes
    mono = x.channels[IDX["disp_mono"]]
    np.testing.assert_allclose(mono * W, d, atol=0.05)  # monocular alineado en d/W_ref
    assert x.channels[IDX["valid_mono"]].all()


def test_mono_respects_physical_disparity():
    d, m = scene()
    valid = np.ones(d.shape, bool)
    # Con doffs = −25, el monocular alineado de las filas de arriba (d < 25) no es físico.
    stereo = stereo_output(d, valid & (d > 25.5))
    x, base = inputs_for(stereo, m, doffs=-25.0)
    assert base.alignment.accepted, base.alignment.reason
    vm = x.channels[IDX["valid_mono"]].astype(bool)
    assert vm[d > 25.5].all() and not vm[d < 24.5].any()
    assert not x.channels[IDX["disp_mono"]][~vm].any()


def test_rejected_alignment_zeroes_mono_channels():
    d, m = scene()
    valid = np.zeros(d.shape, bool)
    valid[:15, :20] = True  # anclas en una sola celda: alineación rechazada
    stereo = stereo_output(d, valid)
    x, base = inputs_for(stereo, m)
    assert not base.alignment.accepted
    for name in ("disp_mono", "dif_estereo_mono", "valid_mono"):  # canales 9, 10 y 11
        assert not x.channels[IDX[name]].any(), name
    np.testing.assert_array_equal(x.base_valid, stereo.valid)


def test_both_sources_failing_gives_finite_channels():
    left, right = images()
    _, m = scene()
    nothing = np.zeros((H, W), bool)
    stereo = StereoOutput(np.full((H, W), np.nan, np.float32), nothing, {},
                          np.full((H, W), np.nan, np.float32))  # fmt: skip
    x, base = inputs_for(stereo, m, left, right)
    assert not base.alignment.accepted
    assert np.isfinite(x.channels).all()
    assert not x.base_valid.any() and not x.base_norm.any()
    assert (x.channels[IDX["error_lr"]] == ChannelParams().lr_cap).all()
    for name in ("disp_estereo", "valid_estereo", "error_foto", "disp_mono",
                 "dif_estereo_mono", "valid_mono"):  # fmt: skip
        assert not x.channels[IDX[name]].any(), name


# --- Validaciones -------------------------------------------------------------------------


def test_build_inputs_rejects_missing_lr_error_and_bad_shapes():
    d, m = scene()
    left, right = images()
    valid = np.ones(d.shape, bool)
    good = stereo_output(d, valid)
    base = combine("B4", good, m, 0.0, ALIGN)
    b0_like = StereoOutput(good.disparity_px, valid, {})  # sin lr_error_px, como B0
    with pytest.raises(InvalidInput, match="lr_error_px"):
        build_inputs(left, right, b0_like, base, m)
    with pytest.raises(InvalidInput):  # derecha de otro tamaño
        build_inputs(left, right[:, :-1], good, base, m)
    with pytest.raises(InvalidInput):  # monocular de otro tamaño
        build_inputs(left, right, good, base, m[:-1])
    small = stereo_output(d[:, :-4], valid[:, :-4])
    with pytest.raises(InvalidInput):  # estéreo de otro tamaño que el par
        build_inputs(left, right, small, base, m)
    bad_lr = StereoOutput(good.disparity_px, valid, {}, good.lr_error_px[:-1])
    with pytest.raises(InvalidInput):
        build_inputs(left, right, bad_lr, base, m)
    with pytest.raises(InvalidInput):  # variante desconocida
        build_inputs(left, right, good, base, m, variant="F2")
    with pytest.raises(InvalidInput):  # imagen float en vez de uint8
        build_inputs(left.astype(np.float32), right, good, base, m)
    b3 = combine("B3", good, m, 0.0, ALIGN)
    with pytest.raises(InvalidInput, match="B4"):  # la base tiene que ser B4 de este estéreo
        build_inputs(left, right, good, b3, m)


def test_channel_params_config_roundtrip_and_validation():
    p = ChannelParams.from_config({"error_gain": 16.0, "lr_cap": 2.0, "otra_clave": 1})
    assert p == ChannelParams(error_gain=16.0, lr_cap=2.0)
    assert ChannelParams.from_config(p.to_dict()) == p
    with pytest.raises(InvalidInput):
        ChannelParams(error_gain=0.0)
    with pytest.raises(InvalidInput):
        ChannelParams(lr_cap=float("nan"))
    d, m = scene()
    lr = np.full((H, W), 1.0, np.float32)  # 1 px de error L/R
    x, _ = inputs_for(stereo_output(d, np.ones(d.shape, bool), lr), m, params=p)
    assert x.channels[IDX["error_lr"]] == pytest.approx(1.0 / W * 16.0)


# --- F3 y compatibilidad de versiones ---------------------------------------------------------

# Umbrales como los de producción (4 px, reservadas 0,3): el monocular deformado no es una
# recta exacta y con 1 px la alineación se rechazaría.
ALIGN4 = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5, inlier_threshold_px=4.0,
                     min_holdout_inlier_ratio=0.3)  # fmt: skip
B5 = B5Params(replace(ALIGN4, method="huber"), LocalAffineParams(sigma_px=15.0))
I3 = {name: i for i, name in enumerate(F3_CHANNELS)}


def warped_scene():
    """Plano inclinado con un monocular que NO es una recta de la disparidad (escala que cambia
    con la columna): la recta global se equivoca en el hueco y la corrección local lo nota."""
    d, _ = scene()
    cols = np.linspace(0.9, 1.1, W, dtype=np.float32)[None, :]
    m = ((d - T_TRUE) / (S_TRUE * cols)).astype(np.float32)
    valid = np.ones(d.shape, bool)
    valid[:, 50:62] = False  # hueco vertical
    return d, m, valid


def test_f3_channels_are_finite_bounded_and_zero_outside_holes():
    d, m, valid = warped_scene()
    stereo = stereo_output(d, valid)
    left, right = images()
    b4, b5 = fusion_bases(stereo, m, 0.0, ALIGN4, B5, True, True)
    x = build_inputs(left, right, stereo, b4, m, 0.0, "F3", b5=b5)
    assert x.names == F3_CHANNELS and x.channels.shape == (14, H, W)
    assert np.isfinite(x.channels).all()
    corr, dist, dens = (x.channels[I3[n]] for n in ("corr_local", "dist_ancla", "densidad_anclas"))
    assert (corr[valid] == 0).all() and np.abs(corr).max() <= 1.0
    assert b4.alignment.accepted and b5.alignment.accepted
    assert np.abs(corr[~valid]).max() > 0  # en el hueco B5 corrige algo
    expected = (b5.disparity_px - b4.disparity_px)[~valid] / W * ChannelParams().corr_gain
    assert np.allclose(corr[~valid], np.clip(expected, -1, 1), atol=1e-6)
    assert (dist[valid] == 0).all() and (dist[~valid] > 0).all() and dist.max() <= 1.0
    assert dist[:, 56].min() == pytest.approx(6 / W * 4.0, rel=1e-4)  # centro: 6 px al ancla
    assert 0 <= dens.min() and dens.max() <= 1.0 and dens[:, 56].max() < dens[:, 10].max()
    # Los 11 primeros canales son los de F1 con la misma base (F3 no cambia F0/F1).
    f1 = build_inputs(left, right, stereo, b4, m, 0.0, "F1")
    assert np.array_equal(x.channels[:11], f1.channels)


def test_f3_with_b5_base_and_b5_computed_inside():
    d, m, valid = warped_scene()
    stereo = stereo_output(d, valid)
    left, right = images()
    b4, b5 = fusion_bases(stereo, m, 0.0, ALIGN4, B5, True, True)
    on_b5 = build_inputs(left, right, stereo, b5, m, 0.0, "F3", b4=b4)
    inside = build_inputs(left, right, stereo, b4, m, 0.0, "F3", b5_params=B5)
    assert np.array_equal(on_b5.channels[I3["corr_local"]], inside.channels[I3["corr_local"]])
    assert np.array_equal(on_b5.base_norm * W, np.where(b5.valid, b5.disparity_px, 0) / W * W)
    with pytest.raises(InvalidInput, match="B5"):
        build_inputs(left, right, stereo, b4, m, 0.0, "F3")  # sin B5
    with pytest.raises(InvalidInput, match="B4"):
        build_inputs(left, right, stereo, b5, m, 0.0, "F3")  # base B5 sin B4


def test_check_channels_accepts_old_f0_f1_and_rejects_the_rest():
    check_channels("fusion-canales-1", "F1", F1_CHANNELS)
    check_channels("fusion-canales-1", "F0", F0_CHANNELS)
    check_channels(CHANNELS_VERSION, "F3", F3_CHANNELS)
    with pytest.raises(InvalidInput, match="F3"):
        check_channels("fusion-canales-1", "F3")  # F3 no existía en la versión 1
    with pytest.raises(InvalidInput, match="canales"):
        check_channels("fusion-canales-0", "F1")
    with pytest.raises(InvalidInput, match="orden"):
        check_channels(CHANNELS_VERSION, "F1", F0_CHANNELS)
    assert [c["nombre"] for c in describe_channels("F3")] == list(F3_CHANNELS)
