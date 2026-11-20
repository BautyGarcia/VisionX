"""Red de alineación (T12): cuentas geométricas, datos sintéticos y la red en CPU.

Sin CREStereo, sin pesos, sin datasets ni cámara. La red se prueba con imágenes chicas.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from visionx.depth.crestereo_adapt.align_data import (
    SampleSpec,
    build_item,
    coarse_field_and_maps,
    virtual_camera,
)
from visionx.depth.crestereo_adapt.alignnet import (
    CALIB_SIZE,
    calibration_vector,
    grid_centers,
    jacobian_maps,
    ray_maps,
)
from visionx.depth.crestereo_adapt.lens import LensCorrection, pull_maps
from visionx.depth.crestereo_adapt.perturb import (
    CameraModel,
    CameraRanges,
    mismatch_field,
    mismatch_maps,
    sample_sibling,
)
from visionx.shared.geometry import StereoGeometry

GEOM = StereoGeometry(width=160, height=96, fx=120.0, fy=120.0, cx=80.0, cy=48.0, baseline_m=0.06)


def texture(seed: int, size=(160, 96)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = size
    image = np.zeros((height, width), dtype=np.float32)
    for sigma in (1.0, 3.0, 8.0):
        layer = cv2.GaussianBlur(
            rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigma
        )
        image += layer / layer.std()
    image = (image - image.min()) / (image.max() - image.min())
    return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def camera() -> CameraModel:
    k = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]])
    return CameraModel(k, np.array([-0.2, 0.05, 0.0, 0.0, 0.0]), np.eye(3))


# --- cuentas ------------------------------------------------------------------------------


def test_grilla_y_rayos() -> None:
    centers = grid_centers(GEOM.size)
    assert centers.shape == (12, 20, 2)
    assert np.allclose(centers[0, 0], [3.5, 3.5]) and np.allclose(centers[0, 1], [11.5, 3.5])
    rays = ray_maps(GEOM)
    assert rays.shape == (3, 12, 20)
    assert np.allclose(np.linalg.norm(rays, axis=0), 1.0)


def test_el_jacobiano_predice_el_campo_de_una_correccion_chica() -> None:
    """Campo de un `LensCorrection` chico ≈ −J·θ en los centros de la grilla."""
    theta = np.array(
        [math.radians(0.3), math.radians(0.1), math.radians(-0.4), 0.004, 0.003, -0.001]
    )
    jac = jacobian_maps(GEOM)
    assert jac.shape == (12, 12, 20)
    linear = -np.stack(
        [(jac[0::2] * theta[:, None, None]).sum(0), (jac[1::2] * theta[:, None, None]).sum(0)]
    )
    map_x, map_y = pull_maps(LensCorrection.from_vector(theta), GEOM)
    centers = grid_centers(GEOM.size)
    exact = np.stack(
        [
            cv2.remap(
                map_x,
                centers[..., 0].astype(np.float32),
                centers[..., 1].astype(np.float32),
                cv2.INTER_LINEAR,
            ),
            cv2.remap(
                map_y,
                centers[..., 0].astype(np.float32),
                centers[..., 1].astype(np.float32),
                cv2.INTER_LINEAR,
            ),
        ]
    ) - centers.transpose(2, 0, 1)
    assert np.abs(exact).max() > 0.3  # la corrección mueve algo
    assert np.abs(linear - exact).max() < 0.02  # px


def test_vector_de_calibracion() -> None:
    known = calibration_vector(GEOM, np.array([0.1, -0.2]), known=True)
    assert known.shape == (CALIB_SIZE,) and known[12] == 1.0
    assert known[0] == pytest.approx(0.75) and known[4] == pytest.approx(0.1)
    assert not calibration_vector(GEOM, None, known=False).any()


# --- datos --------------------------------------------------------------------------------


def test_campo_en_grilla_gruesa_coincide_con_el_exacto() -> None:
    nominal = camera()
    true = sample_sibling(np.random.default_rng(1), nominal, CameraRanges())
    field, map_x, map_y = coarse_field_and_maps(true, nominal, GEOM)
    exact_field = mismatch_field(true, nominal, GEOM)
    exact_x, exact_y = mismatch_maps(true, nominal, GEOM)
    assert np.abs(exact_field).max() > 1.0
    assert np.abs(field - exact_field).max() < 0.05
    assert np.abs(map_x - exact_x).max() < 0.05 and np.abs(map_y - exact_y).max() < 0.05


def test_muestra_limpia_y_desalineada() -> None:
    left, right = texture(2), texture(3)
    rng = np.random.default_rng(0)
    clean = build_item(left, right, camera(), GEOM, SampleSpec(0.0, True), rng)
    assert (
        not clean["target"].any()
        and np.array_equal(clean["right_u8"], right)
        and clean["mask"].all()
    )
    assert clean["left"].shape == (3, 96, 160) and clean["left"].max() <= 1.0
    assert clean["jacobian"].shape == (12, 12, 20) and clean["calib"][12] == 1.0
    moved = build_item(left, right, camera(), GEOM, SampleSpec(1.0, False), rng)
    assert np.abs(moved["target"]).max() > 1.0 and not moved["calib"].any()
    assert 0.5 < moved["mask"].mean() < 1.0  # el borde que se pierde no cuenta
    # El campo deshace la desalineación: tomar la perturbada donde dice el campo da la limpia.
    u, v = np.meshgrid(np.arange(160, dtype=np.float32), np.arange(96, dtype=np.float32))
    fixed = cv2.remap(
        moved["right_u8"], u + moved["target"][0], v + moved["target"][1], cv2.INTER_LINEAR
    )
    inner = (slice(12, -12), slice(12, -12))
    assert np.abs(fixed[inner].astype(int) - right[inner].astype(int)).mean() < 4.0


def test_camara_virtual_razonable() -> None:
    nominal, geometry = virtual_camera(np.random.default_rng(5), (544, 432))
    assert 0.55 * 544 <= geometry.fx <= 1.1 * 544 and geometry.fx == nominal.K[0, 0]
    assert abs(geometry.cx - 272) <= 0.03 * 544 + 1e-9


# --- la red -------------------------------------------------------------------------------


def test_la_red_mide_un_corrimiento_de_filas_sin_entrenar() -> None:
    """Con el derecho igual al izquierdo corrido 8 px hacia arriba, la correlación ya lo ve."""
    torch = pytest.importorskip("torch")
    from visionx.depth.crestereo_adapt.alignnet import AlignConfig, build_model

    torch.manual_seed(0)
    model = build_model(AlignConfig(width=32, head_width=32)).eval()
    # Recortes de una imagen más grande: lo que el izquierdo ve en (x+16, y+8), el derecho lo ve
    # en (x, y). Es decir, 16 px de disparidad y el derecho 8 px más arriba. Sin bordes que den
    # la vuelta.
    big = texture(4, (224, 160))
    left = np.ascontiguousarray(big[:128, :192])
    right = np.ascontiguousarray(big[8:136, 16:208])
    geometry = StereoGeometry(
        width=192, height=128, fx=150.0, fy=150.0, cx=96.0, cy=64.0, baseline_m=0.06
    )
    item = build_item(
        left,
        right,
        CameraModel.ideal(geometry),
        geometry,
        SampleSpec(0.0, True),
        np.random.default_rng(0),
    )
    from visionx.depth.crestereo_adapt.align_data import to_batch

    batch = to_batch([item])
    with torch.no_grad():
        out = model(*(batch[k] for k in ("left", "right", "jacobian", "rays", "calib")))
    dy = out["dy"][0, 0, 3:-3, 3:-5].numpy()  # sin los bordes, donde la búsqueda se sale
    # Sin entrenar, el promedio suave sobre las filas candidatas tira hacia cero: se controla la
    # dirección y que esté más cerca de −8 que de 0. El entrenamiento afina el valor.
    assert -9.0 < np.median(dy) < -4.0
    # Ese corrimiento uniforme es un pitch: el ajuste lo convierte en un campo negativo en filas.
    assert -9.0 < np.median(out["field"][0, 1, 3:-3, 3:-5].numpy()) < -4.0
    assert out["theta"].shape == (1, 6) and torch.isfinite(out["field"]).all()


def test_la_red_aprende_con_pocos_ejemplos() -> None:
    torch = pytest.importorskip("torch")
    from visionx.depth.crestereo_adapt.align_data import to_batch
    from visionx.depth.crestereo_adapt.align_train import loss_terms
    from visionx.depth.crestereo_adapt.alignnet import AlignConfig, build_model

    torch.manual_seed(0)
    model = build_model(AlignConfig(width=32, head_width=32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    items = [
        build_item(
            texture(10 + k),
            np.roll(texture(10 + k), -6, axis=1),
            camera(),
            GEOM,
            SampleSpec(0.8, True),
            np.random.default_rng(k),
        )
        for k in range(2)
    ]
    batch = to_batch(items)
    inputs = [batch[k] for k in ("left", "right", "jacobian", "rays", "calib")]
    first = None
    for _ in range(40):
        loss, parts = loss_terms(model(*inputs), batch, 0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = parts["campo"] if first is None else first
    assert parts["campo"] < 0.7 * first


def test_corregir_a_resolucion_de_captura_escala_el_campo() -> None:
    """Un campo constante de (1, −2) px a la resolución de la red es (0, −4) px al doble.

    Las filas se escalan; el corrimiento constante de columnas se descarta (lo arregla la
    distancia conocida).
    """
    torch = pytest.importorskip("torch")
    from visionx.depth.crestereo_adapt.align_train import correct_right

    class Constant(torch.nn.Module):
        def forward(self, left, right, jacobian, rays, calib):
            field = torch.zeros(jacobian.shape[0], 2, *jacobian.shape[-2:])
            field[:, 0], field[:, 1] = 1.0, -2.0
            return {"field": field}

    capture = StereoGeometry(
        width=320, height=192, fx=240.0, fy=240.0, cx=160.0, cy=96.0, baseline_m=0.06
    )
    right = texture(7, (320, 192))
    out = correct_right(Constant(), "cpu", texture(8, (320, 192)), right, capture, None, (160, 96))
    # El píxel (x, y) de la salida se toma de (x, y − 4) de la entrada.
    assert np.array_equal(out[40:150, 20:300], right[36:146, 20:300])


def test_sin_corrimiento_constante_de_columnas() -> None:
    """Se va la mediana de las columnas; lo que varía en la imagen y las filas quedan."""
    from visionx.depth.crestereo_adapt.align_train import without_column_shift

    ramp = np.tile(np.linspace(-1.0, 1.0, 9, dtype=np.float32), (4, 1))
    field = np.stack([ramp + 3.0, np.full((4, 9), -2.0, dtype=np.float32)])
    out = without_column_shift(field)
    assert np.allclose(out[0], ramp)
    assert np.allclose(out[1], -2.0)
    assert np.allclose(field[0], ramp + 3.0)  # no toca la entrada


def test_campo_de_varios_pares_es_la_mediana_por_pixel() -> None:
    """Un par malo no arrastra el campo de la cámara (mediana, no promedio)."""
    from visionx.depth.crestereo_adapt.align_train import pooled_field

    good = np.full((2, 3, 4), 0.5, dtype=np.float32)
    bad = np.full((2, 3, 4), 40.0, dtype=np.float32)
    out = pooled_field([good, good * 1.2, bad])
    assert np.allclose(out, 0.6)


def test_campo_encima_de_otra_correccion_lee_la_imagen_una_vez() -> None:
    """Componer los mapas da lo mismo que corregir en dos pasos (con enteros, exacto)."""
    from visionx.depth.crestereo_adapt.align_train import apply_field

    right = texture(3, (160, 96))
    height, width = right.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    base = (u + 3.0, v - 2.0)  # primera corrección: corre 3 px en x y −2 en y
    field = np.stack([np.full_like(u, -1.0), np.full_like(v, 5.0)])
    two_steps = cv2.remap(
        cv2.remap(right, *base, cv2.INTER_LINEAR, borderValue=0),
        u + field[0],
        v + field[1],
        cv2.INTER_LINEAR,
        borderValue=0,
    )
    once = apply_field(right, field, base)
    assert np.array_equal(once[10:80, 10:150], two_steps[10:80, 10:150])
    assert np.array_equal(once[10:80, 10:150], right[13:83, 12:152])
