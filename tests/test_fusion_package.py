"""Tests del paquete y del estimador de Fusion (T12): muestras sintéticas, caché en tmp_path,
monocular falso, CPU. Sin datos reales, GPU ni pesos reales."""

import csv
import re
import shutil
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from fusion_synthetic import (  # noqa: E402
    ALIGN,
    NUM_DISPARITIES,
    SGBM,
    FakeMono,
    H,
    W,
    make_refs,
)

from visionx.depth.fusion.__main__ import main as fusion_cli  # noqa: E402
from visionx.depth.fusion.cache import CacheSpec, build_cache, mono_identity  # noqa: E402
from visionx.depth.fusion.channels import ChannelParams  # noqa: E402
from visionx.depth.fusion.data import FusionDataset  # noqa: E402
from visionx.depth.fusion.estimator import (  # noqa: E402
    FusionEstimator,
    FusionMatcher,
    load_estimator,
)
from visionx.depth.fusion.model import FusionNet, ModelConfig  # noqa: E402
from visionx.depth.fusion.package import (  # noqa: E402
    CARD_FILE,
    META_FILE,
    NOT_MEASURED,
    WEIGHTS_FILE,
    config_base_differences,
    empaquetar,
    load_package,
)
from visionx.depth.fusion.train import (  # noqa: E402
    RunInputs,
    TrainConfig,
    load_checkpoint,
    run_training,
)
from visionx.depth.hybrid import B5Params, HybridMatcher  # noqa: E402
from visionx.depth.mono import MonoConfig, sha256_file  # noqa: E402
from visionx.depth.stereo import SgbmMatcher, SgbmParams  # noqa: E402
from visionx.shared.contracts import (  # noqa: E402
    DepthStatus,
    StereoFrame,
    UncertaintyKind,
    check_depth_geometry,
)
from visionx.shared.errors import InvalidInput  # noqa: E402
from visionx.shared.geometry import StereoGeometry  # noqa: E402
from visionx.shared.interfaces import DepthEstimator  # noqa: E402

# El monocular falso se presenta con la identidad del de la config: así la base registrada en
# el run (sección `depth.mono` por defecto) coincide con la que usa el estimador.
MONO_ID = mono_identity(MonoConfig())["model_id"]
MODEL = ModelConfig(widths=(8, 16), norm_groups=4)
NORMS = ("group", "batch", "none")
# Geometría sintética: fx·B / 0,3 m = 20 px -> rango SGBM 32 px, el mismo de la caché.
GEOM = StereoGeometry(width=W, height=H, fx=100.0, fy=100.0, cx=W / 2, cy=H / 2, baseline_m=0.06)


class LikeConfigMono(FakeMono):
    """Monocular exacto de la escena sintética, con la identidad del de la config."""

    model_id = MONO_ID


class InvertedMono(LikeConfigMono):
    """Más cerca = valor más chico: ninguna recta s > 0 lo alinea (la alineación se rechaza)."""

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        return -super().predict(image_bgr)


B5_SECTION = {"method": "huber", "sigma_px": 24.0, "slope_lambda_px2": 100.0}


def depth_record() -> dict:
    """Secciones `depth` con las que se calculó la base (lo que escribe el CLI en config.yaml)."""
    return {
        "sgbm": asdict(SGBM),
        "alineacion": asdict(ALIGN),
        "mono": asdict(MonoConfig()),
        "b4": {"estereo": "B2"},
        "b5": dict(B5_SECTION),
        "rango_busqueda": {"sint": NUM_DISPARITIES},
    }


def b5_params() -> B5Params:
    return B5Params.from_config(depth_record())


def app_config(package, variant: str = "F1") -> dict:
    """Config completa como la de load_config, con la base del entrenamiento."""
    return {
        "depth": {
            "compute": [W, H],
            "range_m": [0.3, 3.0],
            "min_coverage": 0.0,
            **depth_record(),
            "fusion": {"variante": variant, "paquete": str(package)},
        }
    }


def edit_checkpoint(run_dir, change) -> None:
    path = run_dir / "mejor.pt"
    ckpt = torch.load(path, weights_only=True)
    change(ckpt["model"])
    torch.save(ckpt, path)


@pytest.fixture(scope="module", params=NORMS)
def world(tmp_path_factory, request):
    """Caché sintética, un run real de 1 época y variantes de su checkpoint empaquetadas.

    Parametrizado por la normalización: paquete, paridad y estimador valen para las tres
    (con BatchNorm, la red del paquete tiene que estar en eval())."""
    norm = request.param
    model = replace(MODEL, norm=norm)
    root = tmp_path_factory.mktemp(f"fusion_pkg_{norm}")
    train_refs = make_refs(root, 2, gt_bias_px=2.0)
    val_refs = make_refs(root, 1, start=10, gt_bias_px=2.0)
    spec = CacheSpec(root / "cache", SGBM, ALIGN, mono_identity(MonoConfig()), "B2")
    build_cache(train_refs + val_refs, lambda s: NUM_DISPARITIES, spec, LikeConfigMono, root,
                log=lambda _: None)  # fmt: skip
    inp = RunInputs(train_refs, val_refs, NUM_DISPARITIES, root, spec, ChannelParams(),
                    config_record={"depth": depth_record()}, commit="abc1234")  # fmt: skip
    cfg = TrainConfig(model=model, crop=(32, 48), crops_per_image=2, batch_images=2, epochs=1,
                      lr=3e-3, warmup_steps=0, device="cpu", num_workers=0, seed=3)  # fmt: skip
    runs = root / "runs"
    run_training(cfg, runs / "real", inp, log=lambda _: None)

    def variant(name, change):
        shutil.copytree(runs / "real", runs / name)
        edit_checkpoint(runs / name, change)
        return empaquetar(runs / name, root / "paquetes", name)

    def fresh(state):  # red recién creada: cabeza delta en cero
        state.update(FusionNet(replace(model, in_channels=11)).state_dict())

    def perturbed(state):  # cabeza delta al azar con semilla fija: corrección de ~1 px
        g = torch.Generator().manual_seed(7)
        w = state["delta_head.weight"]
        state["delta_head.weight"] = torch.randn(w.shape, generator=g) * 0.5
        state["delta_head.bias"] = torch.full_like(state["delta_head.bias"], 0.3)

    def shift_down(state):  # resta 12 px en todo el frame (W = 128, residual_scale = 1/128)
        state["delta_head.weight"] = torch.zeros_like(state["delta_head.weight"])
        state["delta_head.bias"] = torch.full_like(state["delta_head.bias"], -12.0)

    return SimpleNamespace(
        root=root, spec=spec, val_refs=val_refs, runs=runs, norm=norm,
        real=empaquetar(runs / "real", root / "paquetes", "real"),
        fresh=variant("fresca", fresh),
        perturbed=variant("perturbada", perturbed),
        shifted=variant("corrida", shift_down),
    )  # fmt: skip


def matcher(package_dir, mono=None, sgbm=None, stereo="B2", **kwargs) -> FusionMatcher:
    sgbm = sgbm or SGBM
    kwargs.setdefault("b5_params", b5_params())
    return FusionMatcher(SgbmMatcher(stereo, replace(sgbm, num_disparities=NUM_DISPARITIES)),
                         mono or LikeConfigMono(), ALIGN, load_package(package_dir, "cpu"),
                         **kwargs)  # fmt: skip


def frame_of(left, right) -> StereoFrame:
    return StereoFrame(session_id="s", frame_id=1, left=left, right=right, received_ns=1,
                       rectified=True, calibration_id="c",
                       geometry_id=GEOM.geometry_id)  # fmt: skip


# --- paquete --------------------------------------------------------------------------------


def test_package_has_weights_metadata_and_card_from_the_run(world):
    pkg_dir = world.real
    assert {p.name for p in pkg_dir.iterdir()} == {WEIGHTS_FILE, META_FILE, CARD_FILE}
    state = torch.load(pkg_dir / WEIGHTS_FILE, weights_only=True)
    _, ckpt = load_checkpoint(world.runs / "real" / "mejor.pt")
    assert set(state) == set(ckpt["model"])  # state_dict y nada más
    meta = yaml.safe_load((pkg_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["modelo"]["sha256"] == sha256_file(pkg_dir / WEIGHTS_FILE)
    assert meta["canales"]["version"] == "fusion-canales-2" and meta["canales"]["variante"] == "F1"
    assert meta["base"]["b4"] == {"estereo": "B2"} and meta["base"]["monocular_id"] == MONO_ID
    # Base B4 con F1: depth.b5 no se registra (cambiarla no tiene por qué invalidar el paquete).
    assert meta["base"]["metodo"] == "B4" and "b5" not in meta["base"]
    assert meta["origen"]["commit"] == "abc1234" and meta["origen"]["seed"] == 3
    assert meta["datos"]["n_train"] == 2 and meta["datos"]["n_val"] == 1

    # Las métricas de val son las de curvas.csv, sin redondear ni inventar.
    with (world.runs / "real" / "curvas.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert meta["metricas_val"]["epe_fusion_px"] == float(row["val_epe_fusion"])
    assert meta["metricas_val"]["epe_base_px"] == float(row["val_epe_base"])

    pkg = load_package(pkg_dir, "cpu")
    assert pkg.model.config.norm == world.norm == meta["modelo"]["config"]["norm"]
    assert re.fullmatch(r"fusion-f1-[0-9a-f]{8}", pkg.model_id)
    assert pkg.model_id.endswith(meta["modelo"]["sha256"][:8])
    assert not pkg.model.training
    for name, tensor in pkg.model.state_dict().items():
        assert torch.equal(tensor, ckpt["model"][name]), name  # la misma red que el .pt

    card = (pkg_dir / CARD_FILE).read_text(encoding="utf-8")
    assert pkg.model_id in card and f"{float(row['val_epe_base']):.3f}" in card
    assert f"Test público (MiddEval3, ETH3D): {NOT_MEASURED}" in card


def test_empaquetar_refuses_overwrite_and_runs_without_base(world, tmp_path):
    with pytest.raises(InvalidInput, match="ya existe"):
        empaquetar(world.runs / "real", world.root / "paquetes", "real")
    with pytest.raises(InvalidInput, match="nombre"):
        empaquetar(world.runs / "real", tmp_path, "../afuera")
    run = tmp_path / "sin-base"
    shutil.copytree(world.runs / "real", run)
    record = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    del record["depth"]
    (run / "config.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")
    with pytest.raises(InvalidInput, match="base"):
        empaquetar(run, tmp_path / "paquetes", "x")
    assert not (tmp_path / "paquetes" / "x").exists()


def test_altered_weights_are_rejected(world, tmp_path):
    pkg = shutil.copytree(world.real, tmp_path / "alterado")
    data = bytearray((pkg / WEIGHTS_FILE).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (pkg / WEIGHTS_FILE).write_bytes(bytes(data))
    with pytest.raises(InvalidInput, match="sha256"):
        load_package(pkg, "cpu")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("canales", "version", "fusion-canales-0", "canales"),
        ("canales", "nombres", ["rgb_r"], "orden de canales"),
        (None, "formato", "fusion-paquete-0", "formato"),
        ("unidades", "disparidad", "px", "unidad"),
    ],
)
def test_package_with_other_channels_or_format_is_rejected(
    world, tmp_path, section, key, value, message
):
    pkg = shutil.copytree(world.real, tmp_path / "otro")
    meta = yaml.safe_load((pkg / META_FILE).read_text(encoding="utf-8"))
    (meta[section] if section else meta)[key] = value
    (pkg / META_FILE).write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    with pytest.raises(InvalidInput, match=message):
        load_package(pkg, "cpu")


# --- matcher: B4, paridad y validez ---------------------------------------------------------


def test_fresh_network_reproduces_b4(world):
    sample = world.val_refs[0].load()
    out = matcher(world.fresh).compute(sample.left, sample.right, 0.0)
    b4 = HybridMatcher("B4", "B2", replace(SGBM, num_disparities=NUM_DISPARITIES),
                       LikeConfigMono(), ALIGN).compute(sample.left, sample.right, 0.0)  # fmt: skip
    assert out.base.alignment.accepted and out.fallback is None
    assert np.array_equal(out.valid, b4.valid) and out.valid.any()
    assert np.abs(out.disparity_px - b4.disparity_px)[out.valid].max() <= 1e-4
    assert np.isnan(out.disparity_px[~out.valid]).all()
    assert (out.b_px[out.valid] > 0).all()
    for key in ("ms_estereo", "ms_mono", "ms_alineacion", "ms_canales", "ms_red"):
        assert out.stats[key] >= 0, key


def test_offline_and_application_paths_give_the_same_disparity(world):
    """Camino de entrenamiento (caché + FusionDataset de evaluación + red) contra el de la app
    (FusionMatcher calculando todo) sobre el mismo par, con una red que sí corrige."""
    pkg = load_package(world.perturbed, "cpu")
    ref = world.val_refs[0]
    ds = FusionDataset([ref], NUM_DISPARITIES, world.root, world.spec, pkg.variant,
                       pkg.channel_params, crop=None)  # fmt: skip
    item = ds[0]
    with torch.no_grad():
        offline = pkg.model(item["channels"], item["base"])
    width = float(item["ref_width"][0])
    d_off = offline["disparity"][0, 0].numpy() * width
    b_off = offline["b"][0, 0].numpy() * width

    sample = ref.load()
    out = matcher(world.perturbed).compute(sample.left, sample.right, 0.0)
    assert out.valid.any()
    assert np.abs(d_off - out.disparity_px)[out.valid].max() <= 1e-4
    assert np.abs(b_off - out.b_px)[out.valid].max() <= 1e-4
    # No es trivial: la red perturbada mueve la disparidad respecto de B4.
    moved = np.abs(out.disparity_px - out.base.disparity_px)[out.valid]
    assert np.median(moved) > 0.1  # ~0,27-0,6 px según la normalización


def test_residual_below_physical_limit_invalidates_instead_of_clipping(world):
    sample = world.val_refs[0].load()
    out = matcher(world.shifted).compute(sample.left, sample.right, 0.0)
    base = out.base
    assert (out.disparity_px[out.valid] > 0).all()
    gone = base.valid & ~out.valid  # la base valía y el residual la llevó a d <= 0
    assert gone.any() and np.isnan(out.disparity_px[gone]).all() and np.isnan(out.b_px[gone]).all()
    assert (base.disparity_px[gone] <= 12.0 + 1e-3).all()
    assert out.valid[base.valid & (base.disparity_px > 12.5)].all()
    assert out.stats["invalidada_por_residual"] > 0


def test_other_base_is_rejected_unless_allowed(world):
    with pytest.raises(InvalidInput, match="sgbm.block_size"):
        matcher(world.real, sgbm=replace(SGBM, block_size=7))
    with pytest.raises(InvalidInput, match="b4.estereo"):
        matcher(world.real, stereo="B1")

    class OtherMono(LikeConfigMono):
        model_id = "otro-mono"

    with pytest.raises(InvalidInput, match="mono"):
        matcher(world.real, mono=OtherMono())
    with pytest.warns(UserWarning, match="base distinta"):
        m = matcher(world.real, sgbm=replace(SGBM, block_size=7), allow_other_base=True)
    assert any("block_size" in d for d in m.base_differences)
    # El rango de búsqueda no es parte de la base: sale del dataset o de la geometría.
    trained = load_package(world.real, "cpu").base
    cfg = app_config(world.real)["depth"]
    assert config_base_differences(trained, cfg) == []
    cfg["sgbm"] = {**cfg["sgbm"], "num_disparities": 256}
    assert config_base_differences(trained, cfg) == []
    with pytest.raises(InvalidInput, match="B0"):
        matcher(world.real, stereo="B0")


# --- estimador de la app --------------------------------------------------------------------


def test_estimator_from_config_meets_the_contract(world):
    est = load_estimator(app_config(world.perturbed), GEOM, mono=LikeConfigMono(), device="cpu")
    assert isinstance(est, FusionEstimator) and isinstance(est, DepthEstimator)
    sample = world.val_refs[0].load()
    result, out = est.predict_detailed(frame_of(sample.left, sample.right))
    check_depth_geometry(result, est.geometry)
    assert result.model_id == load_package(world.perturbed, "cpu").model_id
    assert result.status is DepthStatus.VALID and result.reason is None
    assert result.uncertainty_kind is UncertaintyKind.LAPLACE_B
    mask = result.valid_mask
    assert mask.any() and np.array_equal(mask, out.valid)
    assert (result.uncertainty[mask] > 0).all() and np.isnan(result.uncertainty[~mask]).all()
    assert np.array_equal(result.uncertainty[mask], out.b_px[mask])  # b en px de cómputo
    # Misma salida que el matcher armado a mano con la base del entrenamiento.
    direct = matcher(world.perturbed).compute(sample.left, sample.right, 0.0)
    assert np.array_equal(mask, direct.valid)
    assert np.abs(result.disparity_px - direct.disparity_px)[mask].max() <= 1e-4
    est.close()


def test_estimator_partial_when_mono_does_not_align(world):
    est = load_estimator(app_config(world.perturbed), GEOM, mono=InvertedMono(), device="cpu")
    sample = world.val_refs[0].load()
    result, out = est.predict_detailed(frame_of(sample.left, sample.right))
    check_depth_geometry(result, est.geometry)
    assert not out.base.alignment.accepted
    assert result.status is DepthStatus.PARTIAL and result.reason.startswith("solo estéreo")
    assert result.valid_mask.any() and out.stats["fraccion_mono"] == 0.0
    assert (result.uncertainty[result.valid_mask] > 0).all()


def test_estimator_unmeasurable_when_both_sources_fail(world):
    est = load_estimator(app_config(world.perturbed), GEOM, mono=InvertedMono(), device="cpu")
    flat = np.full((H, W, 3), 128, np.uint8)  # sin textura: SGBM no mide y no hay anclas
    result, out = est.predict_detailed(frame_of(flat, flat))
    check_depth_geometry(result, est.geometry)
    assert result.status is DepthStatus.UNMEASURABLE and result.reason
    assert not result.valid_mask.any() and not out.valid.any()
    assert result.uncertainty_kind is UncertaintyKind.LAPLACE_B


def test_load_estimator_needs_a_package_and_the_same_base(world):
    cfg = app_config("")
    with pytest.raises(InvalidInput, match="paquete"):
        load_estimator(cfg, GEOM, mono=LikeConfigMono(), device="cpu")
    cfg = app_config(world.real)
    cfg["depth"]["alineacion"] = {**cfg["depth"]["alineacion"], "inlier_threshold_px": 9.0}
    with pytest.raises(InvalidInput, match="alineacion.inlier_threshold_px"):
        load_estimator(cfg, GEOM, mono=LikeConfigMono(), device="cpu")


# --- CLI ------------------------------------------------------------------------------------


def test_cli_packages_a_run(world, tmp_path, capsys):
    code = fusion_cli(["empaquetar", "real", "--runs", str(world.runs),
                       "--salida", str(tmp_path), "--nombre", "cli"])  # fmt: skip
    assert code == 0
    pkg = load_package(tmp_path / "cli", "cpu")
    assert pkg.model_id in capsys.readouterr().out
    assert fusion_cli(["empaquetar", "no-existe", "--runs", str(world.runs)]) == 2


def test_build_result_without_uncertainty_is_unchanged():
    from visionx.depth.stereo import SgbmEstimator

    rng = np.random.default_rng(0)
    left = np.repeat(np.repeat(rng.integers(0, 256, (H // 2, W // 2, 3), np.uint8), 2, 0), 2, 1)
    right = np.roll(left, -10, axis=1)
    result = SgbmEstimator("B1", SgbmParams(num_disparities=32), GEOM).predict(
        frame_of(left, right)
    )
    assert result.uncertainty is None and result.uncertainty_kind is UncertaintyKind.NONE


def test_base_differences_compare_full_mono_identity(world):
    """No alcanza con `model_id` (8 caracteres de la revisión): repo, revisión completa y sha
    de los pesos también son la base."""
    from visionx.depth.fusion.package import base_differences, mono_identity_of

    trained = load_package(world.real, "cpu").base
    cfg = app_config(world.real)["depth"]
    other_weights = {**cfg["mono"], "weights_sha256": "f" * 64}
    diffs = config_base_differences(trained, {**cfg, "mono": other_weights})
    assert diffs == [f"mono.weights_sha256: entrenado {MonoConfig().weights_sha256!r}, "
                     f"en uso {'f' * 64!r}"]  # fmt: skip
    same_short = {**cfg["mono"], "revision": MonoConfig().revision[:8] + "0" * 32}
    diffs = config_base_differences(trained, {**cfg, "mono": same_short})
    assert [d.split(":")[0] for d in diffs] == ["mono.revision"]  # mismo model_id, otra revisión

    class WithConfig(LikeConfigMono):  # como DepthAnythingV2: trae su MonoConfig
        config = MonoConfig(weights_sha256="0" * 64)

    identity = mono_identity_of(WithConfig())
    assert identity["weights_sha256"] == "0" * 64 and identity["model_id"] == MONO_ID
    sgbm = replace(SGBM, num_disparities=NUM_DISPARITIES)
    diffs = base_differences(trained, "B2", sgbm, ALIGN, identity)
    assert any(d.startswith("mono.weights_sha256") for d in diffs)
    assert base_differences(trained, "B2", sgbm, ALIGN, mono_identity_of(LikeConfigMono())) == []


# --- paquetes de la versión 1 de canales ------------------------------------------------------


def test_package_with_channels_v1_and_f1_still_loads(world, tmp_path):
    """Un paquete de antes de F3 ("fusion-canales-1", sin base registrada): F1 no cambió de
    definición, así que carga y corre sobre B4; F3 con la versión 1 no existe."""
    pkg = shutil.copytree(world.perturbed, tmp_path / "v1")
    meta = yaml.safe_load((pkg / META_FILE).read_text(encoding="utf-8"))
    meta["canales"]["version"] = "fusion-canales-1"
    del meta["base"]["metodo"]
    (pkg / META_FILE).write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    old = load_package(pkg, "cpu")
    assert old.base_variant == "B4" and not old.uses_b5
    sample = world.val_refs[0].load()
    a = matcher(pkg).compute(sample.left, sample.right, 0.0)
    b = matcher(world.perturbed).compute(sample.left, sample.right, 0.0)
    assert np.array_equal(a.valid, b.valid) and np.array_equal(
        a.disparity_px[a.valid], b.disparity_px[b.valid]
    )
    meta["canales"]["variante"] = "F3"
    (pkg / META_FILE).write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    with pytest.raises(InvalidInput, match="F3"):
        load_package(pkg, "cpu")


# --- Fusion v2: F3, base B5 y bloques dilatados ---------------------------------------------


@pytest.fixture(scope="module")
def world_v2(tmp_path_factory):
    """Run de 1 época con F3 sobre base B5 y un bloque dilatado, empaquetado y con la cabeza
    delta perturbada (para que la red sí corrija)."""
    root = tmp_path_factory.mktemp("fusion_v2")
    train_refs = make_refs(root, 2, gt_bias_px=2.0)
    val_refs = make_refs(root, 1, start=10, gt_bias_px=2.0)
    spec = CacheSpec(root / "cache", SGBM, ALIGN, mono_identity(MonoConfig()), "B2")
    build_cache(train_refs + val_refs, lambda s: NUM_DISPARITIES, spec, LikeConfigMono, root,
                log=lambda _: None)  # fmt: skip
    inp = RunInputs(train_refs, val_refs, NUM_DISPARITIES, root, spec, ChannelParams(),
                    b5_params=b5_params(), config_record={"depth": depth_record()},
                    commit="abc1234")  # fmt: skip
    model = ModelConfig(widths=(8, 16), norm="batch", bottleneck_dilations=(2,))
    cfg = TrainConfig(variant="F3", base="B5", model=model, crop=(32, 48), crops_per_image=2,
                      batch_images=2, epochs=1, lr=3e-3, warmup_steps=0, device="cpu",
                      num_workers=0, seed=3)  # fmt: skip
    runs = root / "runs"
    run_training(cfg, runs / "v2", inp, log=lambda _: None)

    def perturbed(state):
        g = torch.Generator().manual_seed(7)
        w = state["delta_head.weight"]
        state["delta_head.weight"] = torch.randn(w.shape, generator=g) * 0.5
        state["delta_head.bias"] = torch.full_like(state["delta_head.bias"], 0.3)

    edit_checkpoint(runs / "v2", perturbed)
    return SimpleNamespace(root=root, spec=spec, val_refs=val_refs,
                           pkg=empaquetar(runs / "v2", root / "paquetes", "v2"))  # fmt: skip


def test_v2_package_records_b5_and_dilations(world_v2):
    pkg = load_package(world_v2.pkg, "cpu")
    assert pkg.variant == "F3" and pkg.base_variant == "B5" and pkg.uses_b5
    assert pkg.base["b5"] == B5_SECTION
    assert pkg.model.config.bottleneck_dilations == (2,) and pkg.model.config.in_channels == 14
    card = (world_v2.pkg / CARD_FILE).read_text(encoding="utf-8")
    assert "Base B5" in card and "soporte común de los tres" in card


def test_v2_offline_and_application_paths_give_the_same_disparity(world_v2):
    """Paridad con F3 y base B5: caché + FusionDataset contra el FusionMatcher de la app."""
    pkg = load_package(world_v2.pkg, "cpu")
    ref = world_v2.val_refs[0]
    ds = FusionDataset([ref], NUM_DISPARITIES, world_v2.root, world_v2.spec, pkg.variant,
                       pkg.channel_params, crop=None, base="B5",
                       b5_params=b5_params())  # fmt: skip
    item = ds[0]
    with torch.no_grad():
        offline = pkg.model(item["channels"], item["base"])
    width = float(item["ref_width"][0])
    d_off = offline["disparity"][0, 0].numpy() * width
    b_off = offline["b"][0, 0].numpy() * width

    sample = ref.load()
    out = matcher(world_v2.pkg).compute(sample.left, sample.right, 0.0)
    assert out.base.variant == "B5" and out.valid.any()
    assert np.abs(d_off - out.disparity_px)[out.valid].max() <= 1e-4
    assert np.abs(b_off - out.b_px)[out.valid].max() <= 1e-4
    moved = np.abs(out.disparity_px - out.base.disparity_px)[out.valid]
    assert np.median(moved) > 0.1  # la red perturbada sí corrige


def test_v2_needs_the_same_b5_as_training(world_v2):
    other = B5Params.from_config({**depth_record(), "b5": {**B5_SECTION, "sigma_px": 48.0}})
    with pytest.raises(InvalidInput, match="b5.sigma_px"):
        matcher(world_v2.pkg, b5_params=other)
    with pytest.raises(InvalidInput, match="depth.b5"):
        matcher(world_v2.pkg, b5_params=None)
    cfg = app_config(world_v2.pkg, "F3")
    del cfg["depth"]["b5"]
    with pytest.raises(InvalidInput, match="b5"):
        load_estimator(cfg, GEOM, mono=LikeConfigMono(), device="cpu")
    cfg = app_config(world_v2.pkg, "F3")
    est = load_estimator(cfg, GEOM, mono=LikeConfigMono(), device="cpu")
    sample = world_v2.val_refs[0].load()
    result, out = est.predict_detailed(frame_of(sample.left, sample.right))
    check_depth_geometry(result, est.geometry)
    assert result.status is DepthStatus.VALID and out.base.variant == "B5"
    cfg["depth"]["b5"] = {**B5_SECTION, "exclude_px": 8.0}
    assert config_base_differences(load_package(world_v2.pkg, "cpu").base, cfg["depth"]) == [
        "b5.exclude_px: entrenado 0.0, en uso 8.0"
    ]


def test_b4_package_is_not_tied_to_depth_b5(world):
    cfg = app_config(world.real)["depth"]
    cfg["b5"] = {**B5_SECTION, "sigma_px": 999.0}
    assert config_base_differences(load_package(world.real, "cpu").base, cfg) == []
