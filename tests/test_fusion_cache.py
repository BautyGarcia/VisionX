"""Tests de la caché de ingredientes de Fusion (T12): sintéticos, monocular falso, sin torch."""

import math

import cv2
import numpy as np
import pytest
from fusion_synthetic import ALIGN, NUM_DISPARITIES, FakeMono, make_refs, make_spec

from visionx.depth.align import AlignParams, align
from visionx.depth.fusion import cache
from visionx.depth.fusion.channels import build_inputs
from visionx.depth.hybrid import HybridMatcher, combine
from visionx.depth.mono import MonoConfig
from visionx.shared.errors import InvalidInput


def ndisp(_sample) -> int:
    return NUM_DISPARITIES


def never_called():
    raise AssertionError("no debería cargar el monocular: todo estaba en caché")


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def test_roundtrip_is_bit_exact(tmp_path):
    ref = make_refs(tmp_path, 1)[0]
    sample = ref.load()
    spec = make_spec(tmp_path / "cache")
    mono = FakeMono()
    ing = cache.compute_ingredients(sample, NUM_DISPARITIES, spec, mono, tmp_path)
    path = cache.save(ing, spec.path_for(sample.dataset, sample.sample_id))
    assert path == tmp_path / "cache" / "sint" / "s000.npz"
    assert not list(path.parent.glob(".tmp-*"))  # escritura atómica: sin temporales
    back = cache.read(sample, NUM_DISPARITIES, spec, tmp_path)

    # B4 desde la caché == B4 calculado de cero por el camino de T05, bit a bit.
    direct = HybridMatcher("B4", "B2", spec.sgbm_for(NUM_DISPARITIES), mono, ALIGN)
    expected = direct.compute(sample.left, sample.right, 0.0)
    base = back.base(0.0, ALIGN)
    assert base.alignment.accepted
    assert _same(base.disparity_px, expected.disparity_px)
    assert _same(base.valid, expected.valid) and _same(base.source, expected.source)
    assert base.alignment == expected.alignment
    assert base.stats == {k: expected.stats[k] for k in base.stats}

    # El estéreo vuelve igual, incluido el error L/R guardado en float16 (múltiplos de 1/16).
    stereo = back.stereo()
    assert _same(stereo.disparity_px, ing.disparity_px) and _same(stereo.valid, ing.valid)
    assert _same(stereo.lr_error_px, ing.lr_error_px)
    assert _same(back.mono_relative, ing.mono_relative)

    # Y los canales de Fusion salen idénticos.
    direct_base = combine("B4", ing.stereo(), ing.mono_relative, 0.0, ALIGN)
    a = build_inputs(sample.left, sample.right, ing.stereo(), direct_base, ing.mono_relative)
    b = build_inputs(sample.left, sample.right, stereo, base, back.mono_relative)
    assert _same(a.channels, b.channels) and _same(a.base_norm, b.base_norm)


def test_rejected_alignment_roundtrips_with_nan(tmp_path):
    ref = make_refs(tmp_path, 1)[0]
    sample = ref.load()
    strict = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=10**6)
    spec = cache.CacheSpec(tmp_path / "c", make_spec(tmp_path).sgbm, strict, {"model_id": "falso"})
    ing = cache.compute_ingredients(sample, NUM_DISPARITIES, spec, FakeMono(), tmp_path)
    assert not ing.alignment.accepted and math.isnan(ing.alignment.scale)
    cache.save(ing, spec.path_for("sint", "s000"))
    back = cache.read(sample, NUM_DISPARITIES, spec, tmp_path)
    assert back.alignment.reason == ing.alignment.reason
    assert math.isnan(back.alignment.scale) and back.alignment.n_anchors == ing.alignment.n_anchors
    assert back.base(0.0, strict).fallback is not None  # B4 solo estéreo, declarado


def test_build_cache_reuses_and_does_not_load_mono(tmp_path):
    refs = make_refs(tmp_path, 3)
    spec = make_spec(tmp_path / "cache")
    mono = FakeMono()
    first = cache.build_cache(refs, ndisp, spec, lambda: mono, tmp_path, log=lambda _: None)
    assert (first["calculadas"], first["reusadas"]) == (3, 0) and mono.calls == 3
    assert first["alineacion_aceptada_pct"] == 100.0 and first["mb_por_muestra"] > 0
    second = cache.build_cache(refs, ndisp, spec, never_called, tmp_path, log=lambda _: None)
    assert (second["calculadas"], second["reusadas"]) == (0, 3)


@pytest.mark.parametrize("change", ["sgbm", "alineacion", "archivo", "monocular", "rango"])
def test_key_changes_invalidate(tmp_path, change):
    ref = make_refs(tmp_path, 1)[0]
    spec = make_spec(tmp_path / "cache")
    cache.build_cache([ref], ndisp, spec, FakeMono, tmp_path, log=lambda _: None)
    sample = ref.load()
    cache.read(sample, NUM_DISPARITIES, spec, tmp_path)  # clave vigente: se lee
    rng_ndisp = NUM_DISPARITIES
    if change == "sgbm":
        spec = make_spec(tmp_path / "cache", block_size=7)
    elif change == "alineacion":
        spec = cache.CacheSpec(spec.root, spec.sgbm, AlignParams(seed=1), spec.mono)
    elif change == "monocular":
        spec = cache.CacheSpec(spec.root, spec.sgbm, spec.align, {"model_id": "otro"})
    elif change == "rango":
        rng_ndisp = 48
    else:  # cambia un archivo de la muestra: mismo nombre, otro contenido
        right = tmp_path / "sint" / "s000" / "right.png"
        img = cv2.imread(str(right))
        img[0, 0] = 255 - img[0, 0]
        cv2.imwrite(str(right), img)
        sample = ref.load()
    # Modo solo-lectura (entrenamiento): clave vieja => error claro, no datos viejos.
    with pytest.raises(InvalidInput, match="clave distinta"):
        cache.read(sample, rng_ndisp, spec, tmp_path)
    if change == "monocular":
        return  # FakeMono es "falso": el modo construir también lo rechaza (ver abajo)
    # Modo construir: recalcula y después la lectura vuelve a funcionar.
    ing, reused, _ = cache.get_or_compute(sample, rng_ndisp, spec, FakeMono, tmp_path)
    assert not reused and ing is not None
    cache.read(sample, rng_ndisp, spec, tmp_path)


def test_missing_cache_and_wrong_mono_are_rejected(tmp_path):
    ref = make_refs(tmp_path, 1)[0]
    sample = ref.load()
    spec = make_spec(tmp_path / "cache")
    with pytest.raises(InvalidInput, match="no hay caché"):
        cache.read(sample, NUM_DISPARITIES, spec, tmp_path)
    other = cache.CacheSpec(spec.root, spec.sgbm, spec.align, {"model_id": "otro"})
    with pytest.raises(InvalidInput, match="monocular"):
        cache.compute_ingredients(sample, NUM_DISPARITIES, other, FakeMono(), tmp_path)


def test_spec_from_config_and_key_material(tmp_path):
    from visionx.shared.config import load_config

    depth = load_config()["depth"]
    spec = cache.CacheSpec.from_config(depth)
    assert spec.stereo_variant == depth["b4"]["estereo"] == "B2"
    assert spec.mono == cache.mono_identity(MonoConfig.from_config(depth["mono"]))
    assert spec.root.as_posix().endswith("data/cache/fusion")  # as_posix: en Windows str usa "\\"
    files = {"a/left.png": "0" * 64, "a/right.png": "1" * 64}
    material = spec.key_material(files, 272, 0.0)
    assert material["sgbm"]["num_disparities"] == 272  # el rango del par entra en la clave
    assert set(material) == {"formato", "archivos", "sgbm", "estereo", "alineacion", "doffs",
                             "mono"}  # fmt: skip
    assert spec.key_for(files, 272, 0.0) != spec.key_for(files, 272, 1.5)  # doffs también
    with pytest.raises(InvalidInput):
        cache.CacheSpec(tmp_path, spec.sgbm, spec.align, spec.mono, stereo_variant="B0")
    with pytest.raises(InvalidInput):
        spec.key_for({}, 272, 0.0)


def test_alignment_from_dict_inverts_to_dict():
    ref_d = np.linspace(10, 40, 60 * 80, dtype=np.float32).reshape(60, 80)
    m = ((ref_d - 10) / 4).astype(np.float32)
    a = align(m, ref_d, np.ones(ref_d.shape, bool), ALIGN)
    assert cache.alignment_from_dict(a.to_dict()) == a


def test_library_versions_are_recorded_and_mismatch_only_warns(tmp_path, monkeypatch):
    import dataclasses
    import warnings

    monkeypatch.setattr(cache, "_WARNED", set())
    ref = make_refs(tmp_path, 1)[0]
    sample = ref.load()
    spec = make_spec(tmp_path / "cache")
    ing = cache.compute_ingredients(sample, NUM_DISPARITIES, spec, FakeMono(), tmp_path)
    assert ing.meta["versiones"] == cache.library_versions()
    assert set(ing.meta["versiones"]) == {"opencv", "numpy", "torch", "transformers"}
    path = spec.path_for("sint", "s000")
    cache.save(ing, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # mismas versiones: ni un aviso
        cache.read(sample, NUM_DISPARITIES, spec, tmp_path)

    # Otra versión de OpenCV: se lee igual (la clave no cambia), con un aviso claro.
    versions = {**ing.meta["versiones"], "opencv": "3.0.0"}
    old = dataclasses.replace(ing, meta={**ing.meta, "versiones": versions})
    cache.save(old, path)
    with pytest.warns(UserWarning, match="opencv: caché 3.0.0"):
        back = cache.read(sample, NUM_DISPARITIES, spec, tmp_path)
    assert back.key == ing.key

    # Caché escrita antes del registro de versiones (sin la clave): se lee sin aviso.
    meta = {k: x for k, x in ing.meta.items() if k != "versiones"}
    cache.save(dataclasses.replace(ing, meta=meta), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache.read(sample, NUM_DISPARITIES, spec, tmp_path)
