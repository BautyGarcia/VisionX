"""CREStereo adaptable (T12): corrección geométrica, generador de perturbaciones y
estimador clásico de alineación. Todo con datos sintéticos; sin red, cámara ni pesos."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from visionx.depth.crestereo_adapt.classic import (
    ClassicParams,
    DecisionParams,
    estimate_alignment,
    fit_correction,
    fit_significant,
    significant_parameters,
)
from visionx.depth.crestereo_adapt.correction import (
    Correction,
    apply_homography,
    compose_maps,
    displacement_field,
    homography,
    warp_image,
)
from visionx.depth.crestereo_adapt.perturb import (
    CameraModel,
    CameraRanges,
    PerturbationRanges,
    mismatch_field,
    mismatch_maps,
    parametric_error,
    perturb_right,
    perturb_right_camera,
    restore_right,
    sample_error,
    sample_sibling,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

# Gran angular parecido a la ELP reducida: 640 px de ancho y fx = 200 px (~116° horizontales).
GEOM = StereoGeometry(
    width=640, height=360, fx=200.0, fy=200.0, cx=320.0, cy=180.0, baseline_m=0.06
)


def texture(seed: int = 0, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """Imagen con detalle a varias escalas: ruido suavizado, para que haya puntos reconocibles."""
    rng = np.random.default_rng(seed)
    width, height = size
    image = np.zeros((height, width), dtype=np.float32)
    for sigma in (1.5, 4.0, 10.0):
        noise = rng.standard_normal((height, width)).astype(np.float32)
        layer = cv2.GaussianBlur(noise, (0, 0), sigma)
        image += layer / layer.std()
    image = (image - image.min()) / (image.max() - image.min())
    return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def interior(margin: int = 40) -> tuple[slice, slice]:
    return slice(margin, GEOM.height - margin), slice(margin, GEOM.width - margin)


# --- corrección ---------------------------------------------------------------------------


def test_sin_correccion_la_homografia_es_la_identidad() -> None:
    assert Correction().is_identity
    assert np.allclose(homography(Correction(), GEOM), np.eye(3))
    assert not Correction(yaw_deg=0.1).is_identity


def test_vector_ida_y_vuelta() -> None:
    c = Correction(roll_deg=0.4, pitch_deg=-0.2, yaw_deg=0.1, log_focal=0.003)
    back = Correction.from_vector(c.as_vector())
    assert np.allclose(back.as_vector(), c.as_vector())


@pytest.mark.parametrize(
    ("correction", "expected_dy"),
    [
        # En (x̂, ŷ) = (1, 0.5): ver las fórmulas de giros chicos en correction.py.
        (Correction(pitch_deg=0.1), lambda a: -200 * a * (1 + 0.5**2)),
        (Correction(roll_deg=0.1), lambda a: 200 * a * 1.0),
        (Correction(yaw_deg=0.1), lambda a: 200 * a * 1.0 * 0.5),
    ],
)
def test_corrimiento_vertical_de_cada_giro(correction, expected_dy) -> None:
    point = np.array([[GEOM.cx + 200.0, GEOM.cy + 100.0]])
    moved = apply_homography(homography(correction, GEOM), point)
    assert moved[0, 1] - point[0, 1] == pytest.approx(expected_dy(math.radians(0.1)), abs=0.01)


def test_el_yaw_corre_la_disparidad_y_casi_no_las_filas_en_el_centro() -> None:
    center = np.array([[GEOM.cx, GEOM.cy]])
    moved = apply_homography(homography(Correction(yaw_deg=0.1), GEOM), center)
    assert moved[0, 0] - center[0, 0] == pytest.approx(200 * math.radians(0.1), abs=0.01)
    assert abs(moved[0, 1] - center[0, 1]) < 1e-9


def test_la_focal_agranda_desde_el_centro() -> None:
    point = np.array([[GEOM.cx + 100.0, GEOM.cy - 50.0]])
    moved = apply_homography(homography(Correction(log_focal=math.log(1.01)), GEOM), point)
    assert np.allclose(moved - point, [[1.0, -0.5]], atol=1e-9)


def test_componer_con_los_mapas_equivale_a_mover_la_imagen() -> None:
    image = texture(1)
    u, v = np.meshgrid(
        np.arange(GEOM.width, dtype=np.float32), np.arange(GEOM.height, dtype=np.float32)
    )
    g = homography(Correction(roll_deg=0.8, pitch_deg=0.5, log_focal=0.004), GEOM)
    map_x, map_y = compose_maps(u, v, g)  # mapas identidad: la "foto cruda" ya es la base
    via_maps = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    direct = warp_image(image, g)
    rows, cols = interior()
    diff = np.abs(via_maps[rows, cols].astype(np.int16) - direct[rows, cols].astype(np.int16))
    assert diff.mean() < 0.5  # niveles de gris sobre 255


# --- generador de perturbaciones ----------------------------------------------------------


def test_sin_perturbacion_el_campo_es_cero_y_la_imagen_no_cambia() -> None:
    right = texture(2)
    out = perturb_right(right, Correction(), GEOM)
    assert np.abs(out.field).max() < 1e-3
    assert out.valid.all()
    assert np.array_equal(out.right, right)


def test_ida_y_vuelta_del_generador() -> None:
    error = Correction(roll_deg=1.2, pitch_deg=-0.7, yaw_deg=0.3, log_focal=0.008)
    # En coordenadas: desalinear y arreglar deja cada píxel donde estaba (mucho menos de 0,1 px).
    do = np.linalg.inv(homography(error, GEOM))
    undo = homography(error, GEOM)
    field = displacement_field(undo @ do, GEOM.size)
    assert np.abs(field).max() < 1e-3
    # En la imagen: quedan dos interpolaciones, que suavizan pero no corren nada.
    right = texture(3)
    out = perturb_right(right, error, GEOM)
    restored = restore_right(out.right, error, GEOM)
    rows, cols = interior()
    diff = np.abs(restored[rows, cols].astype(np.int16) - right[rows, cols].astype(np.int16))
    assert diff.mean() < 3.0  # niveles de gris sobre 255
    # y la perturbada sí es distinta: el error mueve varios píxeles
    assert np.abs(out.field).max() > 3.0


def test_el_campo_dice_donde_quedo_cada_pixel() -> None:
    error = Correction(roll_deg=0.6, pitch_deg=0.4)
    out = perturb_right(texture(4), error, GEOM)
    do = np.linalg.inv(homography(error, GEOM))
    moved = apply_homography(do, np.array([[400.0, 250.0]]))
    assert np.allclose(out.field[250, 400], moved[0] - [400.0, 250.0], atol=1e-3)


def test_semillas_reproducibles() -> None:
    ranges = PerturbationRanges()
    a = sample_error(np.random.default_rng(7), ranges)
    b = sample_error(np.random.default_rng(7), ranges)
    c = sample_error(np.random.default_rng(8), ranges)
    assert a == b and a != c
    assert abs(a.roll_deg) <= ranges.roll_deg and abs(a.pitch_deg) <= ranges.pitch_deg
    assert abs(a.yaw_deg) <= ranges.yaw_deg and abs(a.log_focal) <= ranges.log_focal
    with pytest.raises(InvalidInput):
        PerturbationRanges(roll_deg=-1.0)


# --- estimador clásico --------------------------------------------------------------------


def synthetic_matches(
    error: Correction, n: int = 600, noise_px: float = 0.15, outliers: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Puntos 3D al azar vistos por los dos ojos; el derecho, desalineado con `error`."""
    rng = np.random.default_rng(seed)
    left = np.column_stack(
        [rng.uniform(20, GEOM.width - 20, n), rng.uniform(20, GEOM.height - 20, n)]
    )
    depth = rng.uniform(0.5, 6.0, n)
    disparity = GEOM.fx * GEOM.baseline_m / depth
    right_clean = left - np.column_stack([disparity, np.zeros(n)])
    right = apply_homography(np.linalg.inv(homography(error, GEOM)), right_clean)
    left = left + rng.normal(0, noise_px, left.shape)
    right = right + rng.normal(0, noise_px, right.shape)
    bad = rng.random(n) < outliers
    right[bad] += rng.uniform(-40, 40, (int(bad.sum()), 2))
    return left, right


def test_el_ajuste_recupera_la_correccion() -> None:
    error = Correction(roll_deg=0.8, pitch_deg=-0.5, yaw_deg=0.3, log_focal=0.006)
    est = fit_correction(*synthetic_matches(error), GEOM)
    assert est.ok, est.reason
    assert est.correction.roll_deg == pytest.approx(error.roll_deg, abs=0.02)
    assert est.correction.pitch_deg == pytest.approx(error.pitch_deg, abs=0.02)
    assert est.correction.log_focal == pytest.approx(error.log_focal, abs=1e-3)
    # El yaw se recupera dentro de su propia incertidumbre, que es la mayor de los tres giros.
    assert abs(est.correction.yaw_deg - error.yaw_deg) < 4 * est.sigma["yaw"]
    assert est.sigma["yaw"] > est.sigma["roll"] and est.sigma["yaw"] > est.sigma["pitch"]
    assert est.residual_before_px > 1.0 and est.residual_after_px < 0.3


def test_el_ajuste_aguanta_pares_equivocados() -> None:
    error = Correction(roll_deg=-0.6, pitch_deg=0.4)
    est = fit_correction(*synthetic_matches(error, outliers=0.25, seed=1), GEOM)
    assert est.ok, est.reason
    assert est.correction.roll_deg == pytest.approx(error.roll_deg, abs=0.03)
    assert est.correction.pitch_deg == pytest.approx(error.pitch_deg, abs=0.03)
    assert est.inliers < est.matches


def test_par_alineado_da_correccion_casi_nula() -> None:
    est = fit_correction(*synthetic_matches(Correction(), seed=2), GEOM)
    assert est.ok
    assert abs(est.correction.roll_deg) < 0.02 and abs(est.correction.pitch_deg) < 0.02


def test_sin_yaw_ni_focal_esos_parametros_quedan_en_cero() -> None:
    params = ClassicParams(estimate_yaw=False, estimate_focal=False)
    est = fit_correction(*synthetic_matches(Correction(roll_deg=0.5)), GEOM, params)
    assert est.correction.yaw_deg == 0.0 and est.correction.log_focal == 0.0
    assert est.correction.roll_deg == pytest.approx(0.5, abs=0.02)


def test_pocos_pares_no_se_confia() -> None:
    left, right = synthetic_matches(Correction(roll_deg=0.5), n=12)
    est = fit_correction(left, right, GEOM)
    assert not est.ok and "pares" in est.reason
    assert est.correction.is_identity
    with pytest.raises(InvalidInput):
        fit_correction(left, right[:-1], GEOM)


def test_se_puede_elegir_que_parametros_ajustar() -> None:
    left, right = synthetic_matches(Correction(roll_deg=0.5))
    only_roll = fit_correction(left, right, GEOM, parameters=("roll",))
    assert only_roll.correction.pitch_deg == 0.0 and only_roll.correction.yaw_deg == 0.0
    assert only_roll.correction.roll_deg == pytest.approx(0.5, abs=0.03)
    nothing = fit_correction(left, right, GEOM, parameters=())
    assert nothing.ok and nothing.correction.is_identity
    assert nothing.residual_after_px == nothing.residual_before_px
    with pytest.raises(InvalidInput):
        fit_correction(left, right, GEOM, parameters=("zoom",))


def test_la_regla_no_corrige_un_par_alineado() -> None:
    """Con el par alineado, el ajuste libre da valores chicos y ruidosos: no se aplican."""
    left, right = synthetic_matches(Correction(), seed=4)
    free = fit_correction(left, right, GEOM)
    assert not free.correction.is_identity  # el ajuste libre siempre encuentra "algo"
    ruled, applied = fit_significant(left, right, GEOM)
    assert applied == () and ruled.correction.is_identity and ruled.ok
    assert ruled.residual_after_px == pytest.approx(free.residual_before_px, abs=0.05)


def test_la_regla_aplica_solo_lo_que_supera_el_umbral() -> None:
    error = Correction(roll_deg=0.6, pitch_deg=-0.4)
    left, right = synthetic_matches(error, seed=5)
    ruled, applied = fit_significant(left, right, GEOM)
    assert set(applied) == {"roll", "pitch"}
    assert ruled.correction.yaw_deg == 0.0 and ruled.correction.log_focal == 0.0
    assert ruled.correction.roll_deg == pytest.approx(0.6, abs=0.03)
    assert ruled.correction.pitch_deg == pytest.approx(-0.4, abs=0.03)
    assert ruled.residual_after_px < 0.5


def test_el_umbral_usa_el_piso_cuando_la_incertidumbre_calculada_es_optimista() -> None:
    left, right = synthetic_matches(Correction(yaw_deg=0.06), n=4000, noise_px=0.05, seed=6)
    estimate = fit_correction(left, right, GEOM)
    assert estimate.sigma["yaw"] < 0.02  # el ajuste se cree muy preciso
    # 0,06° no llega a dos veces el piso de 0,04° medido en sesiones sanas: no se aplica...
    assert "yaw" not in significant_parameters(estimate)
    # ...y sí se aplicaría si alguien confiara en la incertidumbre calculada.
    assert "yaw" in significant_parameters(estimate, DecisionParams(floor_yaw_deg=0.0))


def test_de_punta_a_punta_con_imagenes() -> None:
    """Plano de frente con textura: la derecha es la izquierda corrida y después desalineada."""
    left = texture(5)
    shift = np.float64([[1, 0, -12.0], [0, 1, 0]])  # 12 px de disparidad
    right_clean = cv2.warpAffine(
        left, shift, GEOM.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    error = Correction(roll_deg=0.7, pitch_deg=-0.4)
    perturbed = perturb_right(right_clean, error, GEOM).right
    est = estimate_alignment(left, perturbed, GEOM, ClassicParams(estimate_yaw=False))
    assert est.ok, est.reason
    assert est.correction.roll_deg == pytest.approx(error.roll_deg, abs=0.05)
    assert est.correction.pitch_deg == pytest.approx(error.pitch_deg, abs=0.05)
    assert est.residual_after_px < est.residual_before_px


# --- medición de robustez -----------------------------------------------------------------


def test_cambio_de_profundidad_contra_la_referencia() -> None:
    from visionx.depth.crestereo_adapt import robustness

    reference = np.full((4, 5), 2.0, dtype=np.float32)
    reference[0] = 10.0  # fuera del rango donde se anuncian metros: no cuenta
    depth = reference * 1.10  # todo 10 % más lejos
    depth[1, 0] = 2.0  # un píxel exacto
    valid = np.ones((4, 5), dtype=bool)
    valid[3] = False  # una fila sin dato en la corrida perturbada
    stats = robustness.depth_change(depth, valid, reference, valid.copy() | True, (0.5, 3.0))
    assert stats["mediana_pct"] == pytest.approx(10.0, abs=1e-3)
    assert stats["mas_de_5_pct"] == pytest.approx(90.0)  # 9 de los 10 píxeles comparados
    assert stats["cobertura"] == pytest.approx(10 / 15)
    empty = robustness.depth_change(depth, ~valid & False, reference, valid, (0.5, 3.0))
    assert empty["cobertura"] == 0.0 and math.isnan(empty["mediana_pct"])


def test_barridos_de_a_un_parametro() -> None:
    from visionx.depth.crestereo_adapt import robustness

    assert robustness.error_for("roll_deg", 0.5) == Correction(roll_deg=0.5)
    assert robustness.error_for("log_focal", 0.01) == Correction(log_focal=0.01)
    with pytest.raises(KeyError):
        robustness.error_for("otro", 1.0)
    rows = [{"a": 1.0, "b": float("nan")}, {"a": 3.0, "b": float("nan")}]
    out = robustness.summarize(rows)
    assert out["a"] == 2.0 and math.isnan(out["b"])


def test_la_homografia_gira_alrededor_del_centro_del_ojo_derecho() -> None:
    """Con los centros corridos (doffs), el punto que no se mueve en filas es cx + doffs."""
    import dataclasses

    shifted = dataclasses.replace(GEOM, doffs=30.0)
    g = homography(Correction(roll_deg=0.5), shifted)
    center = np.array([[shifted.cx + 30.0, shifted.cy]])
    assert np.allclose(apply_homography(g, center), center, atol=1e-9)
    assert not np.allclose(apply_homography(g, [[shifted.cx, shifted.cy]]), [[GEOM.cx, GEOM.cy]])


# --- generador, etapa 2: cámara real contra nominal ---------------------------------------


def elp_like() -> CameraModel:
    """Lente con distorsión fuerte, de forma parecida a la ELP, para la geometría de prueba."""
    k = np.array([[205.0, 0.0, 318.0], [0.0, 205.0, 182.0], [0.0, 0.0, 1.0]])
    d = np.array([0.03, 1.7, 0.0003, 0.0, 0.09, 0.08, 1.6, 0.2])  # modelo racional de 8
    return CameraModel(k, d, np.eye(3))


def test_camaras_iguales_no_mueven_nada() -> None:
    camera = elp_like()
    assert np.abs(mismatch_field(camera, camera, GEOM)).max() < 1e-3
    right = texture(6)
    out = perturb_right_camera(right, camera, camera, GEOM)
    rows, cols = interior()
    diff = np.abs(out.right[rows, cols].astype(np.int16) - right[rows, cols].astype(np.int16))
    assert diff.max() <= 1  # la única diferencia es el redondeo de interpolar en el mismo lugar
    assert out.error.is_identity


@pytest.mark.parametrize(
    "error",
    [Correction(roll_deg=0.7), Correction(pitch_deg=-0.5), Correction(yaw_deg=0.3)],
)
def test_sin_lente_un_giro_da_lo_mismo_que_la_etapa_1(error) -> None:
    nominal = CameraModel.ideal(GEOM)
    rotation, _ = cv2.Rodrigues(error.as_vector()[:3])  # pitch, yaw, roll = x, y, z
    true = CameraModel(nominal.K, nominal.D, rotation)
    stage_2 = mismatch_field(true, nominal, GEOM)
    stage_1 = perturb_right(texture(7), error, GEOM).field
    assert np.abs(stage_2 - stage_1).max() < 1e-2
    back = parametric_error(true, nominal)
    assert np.allclose(back.as_vector(), error.as_vector(), atol=1e-9)


def test_sin_lente_otra_focal_da_lo_mismo_que_la_etapa_1() -> None:
    nominal = CameraModel.ideal(GEOM)
    k = nominal.K.copy()
    k[0, 0] *= 1.01
    k[1, 1] *= 1.01  # la lente real agranda 1 % más de lo que cree la calibración
    true = CameraModel(k, nominal.D, nominal.R)
    error = parametric_error(true, nominal)
    assert error.log_focal == pytest.approx(-math.log(1.01))  # arreglar es achicar
    stage_1 = perturb_right(texture(7), error, GEOM).field
    assert np.abs(mismatch_field(true, nominal, GEOM) - stage_1).max() < 1e-2


def test_campo_y_mapas_de_la_etapa_2_son_inversos() -> None:
    nominal = elp_like()
    true = sample_sibling(np.random.default_rng(3), nominal, CameraRanges())
    field = mismatch_field(true, nominal, GEOM)
    map_x, map_y = mismatch_maps(true, nominal, GEOM)
    # Un píxel limpio q termina en q + campo(q); el mapa, evaluado ahí, tiene que devolver q.
    u, v = np.meshgrid(np.arange(GEOM.width, dtype=np.float32),
                       np.arange(GEOM.height, dtype=np.float32))  # fmt: skip
    back_x = cv2.remap(map_x, u + field[:, :, 0], v + field[:, :, 1], cv2.INTER_LINEAR)
    back_y = cv2.remap(map_y, u + field[:, :, 0], v + field[:, :, 1], cv2.INTER_LINEAR)
    rows, cols = interior(60)
    assert np.abs(back_x - u)[rows, cols].max() < 0.1  # px
    assert np.abs(back_y - v)[rows, cols].max() < 0.1


def test_otra_distorsion_deja_un_resto_que_no_es_giro_ni_aumento() -> None:
    nominal = elp_like()
    d = nominal.D.copy()
    d[1] *= 1.05  # 5 % más del segundo coeficiente radial
    true = CameraModel(nominal.K, d, nominal.R)
    out = perturb_right_camera(texture(8), true, nominal, GEOM)
    assert out.error.is_identity  # no hay giro ni cambio de focal...
    vertical = np.abs(out.field[:, :, 1])
    assert vertical.max() > 0.5  # ...pero las filas se corren, sobre todo lejos del centro
    assert vertical[GEOM.height // 2, GEOM.width // 2] < 0.05


def test_hermanas_reproducibles_y_dentro_del_rango() -> None:
    nominal = elp_like()
    ranges = CameraRanges()
    a = sample_sibling(np.random.default_rng(11), nominal, ranges)
    b = sample_sibling(np.random.default_rng(11), nominal, ranges)
    c = sample_sibling(np.random.default_rng(12), nominal, ranges)
    assert np.array_equal(a.K, b.K) and np.array_equal(a.D, b.D) and np.array_equal(a.R, b.R)
    assert not np.array_equal(a.K, c.K)
    error = parametric_error(a, nominal)
    assert abs(error.roll_deg) <= ranges.roll_deg + 1e-6
    assert abs(error.log_focal) <= ranges.focal + 1e-9
    assert abs(a.K[0, 2] - nominal.K[0, 2]) <= ranges.center_px
    assert np.all(np.abs(a.D - nominal.D) <= ranges.distortion * np.abs(nominal.D) + 1e-12)
    with pytest.raises(InvalidInput):
        CameraRanges(focal=-0.1)
