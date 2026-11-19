"""CREStereo adaptable (T12): U-Net de incertidumbre. Métricas, canales, caché y un
entrenamiento mínimo con datos sintéticos. Sin CREStereo, sin pesos y sin datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from visionx.depth.crestereo_adapt import uncertainty_metrics as um
from visionx.depth.crestereo_adapt.correction import Correction
from visionx.depth.crestereo_adapt.uncertainty import CHANNELS, build_channels, warp_right
from visionx.depth.crestereo_adapt.uncertainty_data import (
    CacheSpec,
    assumed_geometry,
    frame_arrays,
    load_entry,
    perturbed_right,
    save_entry,
    variant_error,
)
from visionx.shared.errors import InvalidInput

# --- métricas -----------------------------------------------------------------------------


def test_sparsification_perfecta_y_al_azar() -> None:
    rng = np.random.default_rng(0)
    error = rng.exponential(1.0, size=(60, 80))
    mask = np.ones_like(error, dtype=bool)
    # Incertidumbre igual al error: descarta exactamente los peores. AUSE = 0.
    assert um.ause(error, np.abs(error), mask) == pytest.approx(0.0, abs=1e-12)
    curve = um.sparsification_curve(error, np.abs(error), mask)
    assert curve[0] == pytest.approx(1.0) and np.all(np.diff(curve) <= 1e-12)
    # Incertidumbre sin relación con el error: la curva queda plana y el AUSE es grande.
    random = um.ause(error, rng.random(error.shape), mask)
    assert random > 0.3
    # Incertidumbre al revés: peor que al azar.
    assert um.ause(error, -np.abs(error), mask) > random


def test_sparsification_respeta_la_mascara_y_los_no_finitos() -> None:
    error = np.array([[1.0, 2.0, 100.0, np.nan]])
    uncertainty = np.array([[1.0, 2.0, 0.0, 5.0]])
    mask = np.array([[True, True, False, True]])
    # El 100 está fuera de la máscara y el NaN no cuenta: quedan dos píxeles bien ordenados.
    assert um.ause(error, uncertainty, mask) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(InvalidInput):
        um.ause(error, uncertainty[:, :2], mask)


def test_cobertura_de_intervalos_de_laplace() -> None:
    rng = np.random.default_rng(1)
    b = np.full(200_000, 0.7)
    error = rng.laplace(0.0, 0.7, size=b.size)
    mask = np.ones(b.size, dtype=bool)
    assert um.interval_coverage(error, b, mask, 0.5) == pytest.approx(0.5, abs=0.01)
    assert um.interval_coverage(error, b, mask, 0.9) == pytest.approx(0.9, abs=0.01)
    # b a la mitad: confía de más y los intervalos se cumplen menos de lo prometido.
    assert um.interval_coverage(error, b / 2, mask, 0.9) < 0.75
    with pytest.raises(InvalidInput):
        um.interval_coverage(error, b, mask, 1.0)


def test_la_nll_es_minima_con_la_b_verdadera() -> None:
    rng = np.random.default_rng(2)
    error = rng.laplace(0.0, 1.5, size=100_000)
    mask = np.ones(error.size, dtype=bool)
    nll = {s: um.laplace_nll(error, np.full(error.size, s), mask) for s in (0.5, 1.5, 4.0)}
    assert nll[1.5] < nll[0.5] and nll[1.5] < nll[4.0]
    assert math.isnan(um.laplace_nll(error, np.ones(error.size), ~mask))


# --- canales ------------------------------------------------------------------------------


def shifted_pair(disparity: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Izquierda con textura y derecha = izquierda corrida `disparity` columnas."""
    rng = np.random.default_rng(3)
    left = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    shift = int(disparity)
    right[:, : left.shape[1] - shift] = left[:, shift:]
    return left, right, np.full(left.shape[:2], disparity, dtype=np.float32)


def test_warp_trae_la_derecha_a_la_grilla_izquierda() -> None:
    left, right, disparity = shifted_pair()
    warped, inside = warp_right(right, disparity)
    assert not inside[:, :6].any() and inside[:, 6:].all()  # x - d < 0 cae fuera
    assert np.array_equal(warped[:, 6:], left[:, 6:])


def test_canales_formas_rangos_y_significado() -> None:
    left, right, disparity = shifted_pair()
    channels = build_channels(left, right, disparity)
    assert channels.shape == (len(CHANNELS), 48, 64) and channels.dtype == np.float32
    assert channels.min() >= 0.0 and channels.max() <= 1.0
    photometric, visible, edge = channels[3], channels[4], channels[5]
    assert photometric[:, 6:].max() < 1e-6  # disparidad correcta: las dos imágenes coinciden
    assert visible[:, 6:].min() == 1.0 and visible[:, :6].max() == 0.0
    assert edge.max() == 0.0  # disparidad constante: sin bordes
    # Con una disparidad equivocada, el error fotométrico aparece.
    wrong = build_channels(left, right, disparity + 3.0)
    assert wrong[3][:, 12:].mean() > 0.1
    # Un salto de profundidad se marca como borde.
    step = disparity.copy()
    step[:, 32:] += 16.0
    assert build_channels(left, right, step)[5][:, 31:33].max() == 1.0
    with pytest.raises(InvalidInput):
        build_channels(left, right, disparity[:-1])
    # Subproductos de CREStereo: sin ellos van en cero; con ellos, entre 0 y 1.
    assert channels[6].max() == 0.0 and channels[7].max() == 0.0
    extra = build_channels(left, right, disparity, half=disparity - 2.0, vertical=-disparity)
    assert np.allclose(extra[6], 0.5)  # 2 px de desacuerdo sobre una escala de 4
    assert np.allclose(extra[7], 1.0)  # 6 px de flujo vertical: satura
    with pytest.raises(InvalidInput):
        build_channels(left, right, disparity, half=disparity[:-1])


# --- variantes y caché --------------------------------------------------------------------


def test_variantes_reproducibles_y_la_cero_es_limpia() -> None:
    assert variant_error("000123", 0).is_identity
    a, b = variant_error("000123", 1, seed=0), variant_error("000123", 1, seed=0)
    assert a == b and not a.is_identity
    assert variant_error("000123", 1, seed=1) != a
    assert variant_error("000124", 1, seed=0) != a
    errors = [variant_error(f"{i:06d}", 1) for i in range(300)]
    rolls = np.abs([e.roll_deg for e in errors])
    assert rolls.max() <= 1.0 and 0.2 < np.mean(rolls > 0.15) < 0.45  # ~1/3 son grandes


def test_derecha_desalineada() -> None:
    _, right, _ = shifted_pair()
    assert perturbed_right(right, Correction()) is right
    moved = perturbed_right(right, Correction(pitch_deg=1.0))
    assert moved.shape == right.shape and not np.array_equal(moved, right)
    g = assumed_geometry(1080, 860)
    assert g.fx == 1080.0 and g.cx == 540.0 and g.cy == 430.0


def test_cache_ida_y_vuelta_y_clave(tmp_path) -> None:
    spec = CacheSpec(tmp_path, "abcd1234", "i4p2", seed=0)
    path = spec.path("instereo2k", "000123", 1)
    assert path == tmp_path / "instereo2k" / "000123-v1.npz"
    disparity = np.linspace(0.0, 300.0, 48 * 64, dtype=np.float32).reshape(48, 64)
    disparity[0, 0] = np.nan
    save_entry(
        path,
        spec.key("000123", 1),
        disparity,
        right_disparity=disparity * 0.5,
        half=disparity * 0.9,
        vertical=-disparity * 0.01,
    )
    back = load_entry(path, spec.key("000123", 1))
    assert np.abs(back["half"] - disparity * 0.9)[1:].max() <= 0.5 / 64 + 1e-6
    assert np.abs(back["vertical"] + disparity * 0.01)[1:].max() <= 0.5 / 64 + 1e-6
    assert back["vertical"].min() < 0  # conserva el signo
    assert np.nanmax(np.abs(back["disparity"] - np.nan_to_num(disparity))) <= 0.5 / 64 + 1e-6
    assert back["disparity"][0, 0] == 0.0  # sin dato se guarda como 0
    assert np.abs(back["right_disparity"] - disparity * 0.5)[1:].max() <= 0.5 / 64 + 1e-6
    other = CacheSpec(tmp_path, "abcd1234", "i10p2", seed=0)
    with pytest.raises(InvalidInput, match="otros parámetros"):
        load_entry(path, other.key("000123", 1))
    with pytest.raises(InvalidInput, match="falta el caché"):
        load_entry(spec.path("instereo2k", "999999", 0), spec.key("999999", 0))


@dataclass
class FakeSample:
    left: np.ndarray
    right: np.ndarray
    disparity: np.ndarray
    valid: np.ndarray


def test_arreglos_de_un_cuadro() -> None:
    left, right, disparity = shifted_pair()
    valid = np.ones(disparity.shape, dtype=bool)
    valid[:4] = False
    predicted = disparity + 1.0
    predicted[5, 5] = np.nan
    out = frame_arrays(FakeSample(left, right, disparity, valid), predicted, Correction())
    assert out["channels"].shape == (8, 48, 64)
    assert out["base"].shape == (1, 48, 64) and out["gt"].shape == (1, 48, 64)
    assert out["ref_width"] == 64.0
    assert out["base"][0, 10, 10] == pytest.approx(7.0 / 64)  # d/W_ref
    assert out["gt"][0, 10, 10] == pytest.approx(6.0 / 64)
    assert not out["mask"][0, :4].any() and not out["mask"][0, 5, 5]  # sin GT o sin predicción
    assert out["mask"][0, 10, 10]


# --- entrenamiento mínimo -----------------------------------------------------------------


def test_la_red_aprende_que_el_error_fotometrico_avisa() -> None:
    """Sintético: el error depende de un canal. La red tiene que aprender a dar más b ahí."""
    torch = pytest.importorskip("torch")
    from visionx.depth.crestereo_adapt.uncertainty import model_config
    from visionx.depth.fusion.loss import laplace_nll
    from visionx.depth.fusion.model import FusionNet

    torch.manual_seed(0)
    model = FusionNet(model_config((8, 16)))
    for p in model.delta_head.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    width = 64.0
    generator = torch.Generator().manual_seed(0)

    def batch() -> tuple:
        channels = torch.rand(8, len(CHANNELS), 32, 32, generator=generator)
        noisy = channels[:, 3:4] > 0.5  # "fotométrico" alto
        scale = torch.where(noisy, torch.tensor(4.0), torch.tensor(0.25))  # px de error
        error = (torch.rand(8, 1, 32, 32, generator=generator) - 0.5) * 2 * scale
        base = torch.full((8, 1, 32, 32), 10.0 / width)
        return channels, base, base + error / width, noisy

    for _ in range(150):
        channels, base, gt, _ = batch()
        out = model(channels, base)
        loss = laplace_nll(base, out["b"], gt, torch.ones_like(gt, dtype=torch.bool), width)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        channels, base, _, noisy = batch()
        out = model(channels, base)
    b_px = out["b"] * width
    assert float(b_px[noisy].mean()) > 3 * float(b_px[~noisy].mean())
    assert torch.equal(out["disparity"], base)  # la corrección quedó en cero
