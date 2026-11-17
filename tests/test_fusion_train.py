"""Tests de datos y entrenamiento de Fusion (T12): muestras sintéticas, caché en tmp_path,
monocular falso, CPU. Sin datos reales, GPU ni pesos."""

import csv
import json
import pickle
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fusion_synthetic import (  # noqa: E402
    ALIGN,
    NUM_DISPARITIES,
    FakeMono,
    make_refs,
    make_spec,
    write_instereo2k,
)

from visionx.depth.align import LocalAffineParams  # noqa: E402
from visionx.depth.fusion import train as train_mod  # noqa: E402
from visionx.depth.fusion.cache import build_cache  # noqa: E402
from visionx.depth.fusion.channels import F1_CHANNELS, F3_CHANNELS, ChannelParams  # noqa: E402
from visionx.depth.fusion.data import EpochSampler, FusionDataset, collate  # noqa: E402
from visionx.depth.fusion.model import ModelConfig  # noqa: E402
from visionx.depth.fusion.train import (  # noqa: E402
    NonFiniteLoss,
    RunInputs,
    TrainConfig,
    load_checkpoint,
    overfit,
    run_training,
)
from visionx.depth.hybrid import B5Params, combine  # noqa: E402

W_FULL = 128
CROP = (32, 48)
B5 = B5Params(replace(ALIGN, method="huber"), LocalAffineParams(sigma_px=24.0))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """4 muestras de train y 2 de val con GT sesgada +2 px, caché ya construida."""
    root = tmp_path_factory.mktemp("fusion")
    train_refs = make_refs(root, 4, gt_bias_px=2.0)
    val_refs = make_refs(root, 2, start=10, gt_bias_px=2.0)
    spec = make_spec(root / "cache")
    build_cache(train_refs + val_refs, lambda s: NUM_DISPARITIES, spec, FakeMono, root,
                log=lambda _: None)  # fmt: skip
    return root, train_refs, val_refs, spec


def dataset(world, refs=None, **kwargs) -> FusionDataset:
    root, train_refs, _, spec = world
    kwargs.setdefault("crop", CROP)
    kwargs.setdefault("crops_per_image", 2)
    return FusionDataset(refs or train_refs, NUM_DISPARITIES, root, spec, **kwargs)


def small_config(**changes) -> TrainConfig:
    base = dict(model=ModelConfig(widths=(8, 16), norm_groups=4), crop=CROP, crops_per_image=2,
                batch_images=2, epochs=2, lr=3e-3, warmup_steps=0, device="cpu",
                num_workers=0, seed=3)  # fmt: skip
    base.update(changes)
    return TrainConfig(**base)


def inputs(world) -> RunInputs:
    root, train_refs, val_refs, spec = world
    return RunInputs(train_refs, val_refs, NUM_DISPARITIES, root, spec, ChannelParams())


# --- dataset --------------------------------------------------------------------------------


def test_items_have_shapes_masks_and_full_frame_width(world):
    ds = dataset(world)
    item = ds[(1, 0)]
    k, c, (h, w) = 2, len(F1_CHANNELS), CROP
    assert item["channels"].shape == (k, c, h, w) and item["channels"].dtype == torch.float32
    for name in ("base", "gt", "mask"):
        assert item[name].shape == (k, 1, h, w)
    assert item["mask"].dtype == torch.bool
    assert torch.isfinite(item["channels"]).all() and torch.isfinite(item["gt"]).all()
    # W_ref es el ancho del frame completo, no el del recorte.
    assert item["ref_width"].tolist() == [float(W_FULL)] * k
    assert (item["gt"][~item["mask"]] == 0).all() and (item["base"][~item["mask"]] >= 0).all()
    assert item["sample_id"] == ["s000", "s000"]

    # Frame completo (evaluación): mismas cuentas que full_frame, sin recortar.
    full = dataset(world, crop=None, crops_per_image=1)
    frame = full.full_frame(0)
    item = full[0]
    assert item["channels"].shape == (1, c, 72, W_FULL)
    assert np.array_equal(item["channels"][0].numpy(), frame["channels"])
    mask = frame["mask"][0]
    gt_px = item["gt"][0, 0].numpy() * W_FULL
    sample = world[1][0].load()
    assert np.allclose(gt_px[mask], sample.disparity[mask], atol=1e-4)  # GT / W_ref
    assert not (mask & ~sample.valid).any()  # máscara = GT válida ∩ base válida
    batch = collate([full[0], full[1]])
    assert batch["channels"].shape[0] == 2 and batch["sample_id"] == ["s000", "s001"]


def test_crops_are_reproducible_by_seed_epoch_and_index(world):
    a, b = dataset(world, seed=5), dataset(world, seed=5)
    first = a[(2, 1)]
    assert torch.equal(first["channels"], b[(2, 1)]["channels"])  # otro objeto, mismo recorte
    assert torch.equal(first["channels"], a[(2, 1)]["channels"])  # pedirlo dos veces
    assert not torch.equal(first["channels"], a[(3, 1)]["channels"])  # otra época, otro
    # Copiado a un worker (pickle), da lo mismo.
    clone = pickle.loads(pickle.dumps(a))
    assert torch.equal(first["gt"], clone[(2, 1)]["gt"])
    sampler = EpochSampler(4, seed=5)
    sampler.set_epoch(2)
    order = list(sampler)
    assert sorted(i for _, i in order) == [0, 1, 2, 3] and {e for e, _ in order} == {2}
    assert order == list(sampler)


def test_crop_retries_when_mask_is_small(world):
    root, _, _, spec = world
    # GT solo en la mitad derecha: un recorte al azar suele tener poca máscara.
    refs = make_refs(root, 1, start=20, gt_valid_from_x=W_FULL // 2)
    build_cache(refs, lambda s: NUM_DISPARITIES, spec, FakeMono, root, log=lambda _: None)
    ds = FusionDataset(refs, NUM_DISPARITIES, root, spec, crop=CROP, crops_per_image=4,
                       min_valid_fraction=0.9, seed=0)  # fmt: skip
    item = ds[(0, 0)]
    fraction = item["mask"].float().mean(dim=(1, 2, 3))
    assert ds.crop_retries > 0
    assert (fraction >= 0.9).all()
    lax = FusionDataset(refs, NUM_DISPARITIES, root, spec, crop=CROP, crops_per_image=4,
                        min_valid_fraction=0.0, seed=0)  # fmt: skip
    lax[(0, 0)]
    assert lax.crop_retries == 0


def test_dataset_rejects_bad_crop(world):
    with pytest.raises(Exception, match="más grande"):
        dataset(world, crop=(100, 48))[(0, 0)]
    with pytest.raises(Exception, match="un solo ítem"):
        dataset(world, crop=None, crops_per_image=2)


# --- entrenamiento --------------------------------------------------------------------------


def test_training_writes_all_artifacts(world, tmp_path):
    run = tmp_path / "run"
    cfg = small_config(epochs=1)
    result = run_training(cfg, run, inputs(world), log=lambda _: None)
    for name in ("config.yaml", "meta.json", "curvas.csv", "ultimo.pt", "mejor.pt"):
        assert (run / name).is_file(), name
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["estado"] == "terminado" and meta["n_train"] == 4 and meta["n_val"] == 2
    assert meta["parametros"] > 0 and meta["recortes_por_paso"] == 4
    with (run / "curvas.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]["val_epe_base"]) > 1.0  # B4 hereda el sesgo de 2 px
    # B4 en los mismos recortes de train: hereda el mismo sesgo (así se ve si la red corrige).
    assert float(rows[0]["epe_train_base_px"]) > 1.0
    assert float(rows[0]["val_cobertura_fusion"]) == pytest.approx(1.0)
    ckpt = torch.load(run / "ultimo.pt", weights_only=True)
    assert set(ckpt) == {"format", "model", "model_config", "variant", "base",
                         "channels_version", "channel_names", "channel_params", "optimizer",
                         "scheduler", "epoch", "step", "best", "seed", "rng"}  # fmt: skip
    assert ckpt["base"] == "B4"
    assert ckpt["format"] == 1 and ckpt["epoch"] == 1 and ckpt["step"] == 2
    assert ckpt["channel_names"] == list(F1_CHANNELS)
    assert result["best"]["epoch"] == 1
    model, best = load_checkpoint(run / "mejor.pt")
    assert not model.training
    assert best["best"]["epe"] == pytest.approx(float(rows[0]["val_epe_fusion"]), rel=1e-6)
    with pytest.raises(Exception, match="ya existe|exist"):
        run_training(cfg, run, inputs(world), log=lambda _: None)  # no pisa un run


def test_resume_matches_uninterrupted_run(world, tmp_path):
    cfg = small_config(epochs=2)
    run_training(cfg, tmp_path / "seguido", inputs(world), log=lambda _: None)
    cut = tmp_path / "cortado"
    run_training(cfg, cut, inputs(world), stop_after_epoch=1, log=lambda _: None)
    assert json.loads((cut / "meta.json").read_text())["estado"].startswith("cortado")
    run_training(cfg, cut, inputs(world), resume=True, log=lambda _: None)

    def curves(path):
        with (path / "curvas.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    a, b = curves(tmp_path / "seguido"), curves(cut)
    assert [r["epoca"] for r in b] == ["1", "2"]
    for ra, rb in zip(a, b, strict=True):
        for key in ("paso", "lr", "perdida_train", "val_epe_fusion", "val_nll_px"):
            assert float(ra[key]) == pytest.approx(float(rb[key]), rel=1e-5, abs=1e-7), key
    wa = torch.load(tmp_path / "seguido" / "ultimo.pt", weights_only=True)["model"]
    wb = torch.load(cut / "ultimo.pt", weights_only=True)["model"]
    for name in wa:
        assert torch.allclose(wa[name], wb[name], rtol=1e-5, atol=1e-7), name


def test_non_finite_loss_stops_with_clear_error(world, tmp_path, monkeypatch):
    def broken(out, *args, **kwargs):
        return out["disparity"].sum() * float("nan"), {}

    monkeypatch.setattr(train_mod, "combined_loss", broken)
    with pytest.raises(NonFiniteLoss, match="no finita"):
        run_training(small_config(epochs=1), tmp_path / "nan", inputs(world), log=lambda _: None)


def test_train_config_from_yaml_and_variant_sets_channels(tmp_path):
    from visionx.shared.config import ROOT

    cfg = TrainConfig.from_yaml(ROOT / "configs" / "fusion-entrenamiento.yaml")
    # Los valores de la receta pueden cambiar; lo que se comprueba es que el YAML se lee entero.
    assert cfg.model.in_channels == 11 and cfg.lr > 0
    assert cfg.crops_per_step == cfg.batch_images * cfg.crops_per_image
    assert TrainConfig(variant="F0").model.in_channels == 10
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(Exception, match="desconocidas"):
        TrainConfig.from_dict({"lr_maximo": 1.0})
    v2 = TrainConfig.from_yaml(ROOT / "configs" / "fusion-v2.yaml")
    assert v2.variant == "F3" and v2.base in ("B4", "B5") and v2.model.in_channels == 14
    assert v2.model.bottleneck_dilations == (2, 4, 8) and len(v2.model.widths) == 5
    assert TrainConfig.from_dict(v2.to_dict()) == v2


def test_overfit_reduces_epe_at_least_80_percent(world):
    root, train_refs, _, spec = world
    cfg = small_config(model=ModelConfig(widths=(16, 32), norm_groups=4), crops_per_image=1)
    inp = inputs(world)
    result = overfit(cfg, train_refs, inp, steps=150, lr=3e-3, log=lambda _: None)
    assert result["recortes"] == 4 * 3  # 4 imágenes x 3 recortes (OverfitConfig)
    assert result["epe_base_px"] > 1.5  # el sesgo de 2 px de la GT
    assert result["reduccion_pct"] >= 80.0, result
    assert result["cumple_m5"]


def test_non_finite_gradient_stops_before_updating(world, tmp_path, monkeypatch):
    # Pérdida finita pero gradiente infinito: cortar sin llamar a optimizer.step().
    monkeypatch.setattr(train_mod, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("inf")))

    def no_step(*args, **kwargs):
        raise AssertionError("optimizer.step() no debería llamarse con gradiente no finito")

    monkeypatch.setattr(torch.optim.AdamW, "step", no_step)
    with pytest.raises(NonFiniteLoss, match="gradiente no finito"):
        run_training(small_config(epochs=1), tmp_path / "inf", inputs(world), log=lambda _: None)


def _eval_rows(world, change=None):
    from visionx.depth.fusion.model import FusionNet
    from visionx.depth.fusion.train import evaluate, make_loader

    root, _, val_refs, spec = world
    torch.manual_seed(0)
    net = FusionNet(ModelConfig(widths=(8, 16), norm_groups=4, norm="batch"))
    if change:
        change(net)
    ds = FusionDataset(val_refs, NUM_DISPARITIES, root, spec, crop=None)
    return evaluate(net, make_loader(ds, 1, 0), torch.device("cpu"))


def test_evaluate_fresh_network_equals_b4_in_pixels(world):
    for row in _eval_rows(world):
        assert row["epe_fusion"] == row["epe_base"]  # delta 0 y base en px: idénticos
        assert row["cobertura_fusion"] == row["cobertura_base"] > 0.5
        assert row["invalidada_por_residual"] == 0.0
        # Sin B5 en el dataset, el soporte común de los tres queda vacío: NaN, no inventado.
        assert np.isnan(row["epe_comun_b5"])


def test_evaluate_reports_real_fusion_coverage(world):
    def shift_down(net):  # resta 12 px en todo el frame (W = 128, residual_scale = 1/128)
        torch.nn.init.zeros_(net.delta_head.weight)
        torch.nn.init.constant_(net.delta_head.bias, -12.0)

    rows = _eval_rows(world, shift_down)
    for row in rows:
        # La GT va de 10 a 21 px: restar 12 lleva parte de la base a d <= 0 (doffs = 0).
        assert 0 < row["cobertura_fusion"] < row["cobertura_base"]
        assert row["invalidada_por_residual"] > 0
        assert row["acierto2_gt_fusion"] < row["acierto2_gt_base"]
        assert row["epe_fusion"] > row["epe_base"]  # mismo soporte común, Fusion peor
        assert row["n"] < row["n_gt"]


def test_catalog_dataset_survives_pickle_for_spawn_workers(tmp_path):
    """Los SampleRef del catálogo traen `load` lambda: sin `catalog`, no se pueden copiar a
    un worker con spawn; con `catalog`, viajan los ids y el worker reindexa."""
    from visionx.depth.datasets.catalog import CATALOG

    for frame in (1, 2):
        write_instereo2k(tmp_path, frame, seed=frame, gt_bias_px=2.0)
    info = CATALOG["instereo2k"]
    refs = info.index(tmp_path / "instereo2k")
    assert [r.sample_id for r in refs] == ["000001", "000002"]
    spec = make_spec(tmp_path / "cache")
    build_cache(refs, lambda s: NUM_DISPARITIES, spec, FakeMono, tmp_path, log=lambda _: None)

    plain = FusionDataset(refs, NUM_DISPARITIES, tmp_path, spec, crop=None)
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(plain)
    ds = FusionDataset(refs, NUM_DISPARITIES, tmp_path, spec, crop=CROP, crops_per_image=2,
                       seed=4, catalog="instereo2k")  # fmt: skip
    data = pickle.dumps(ds)
    assert b"<lambda>" not in data
    clone = pickle.loads(data)
    assert [r.sample_id for r in clone.refs] == ["000001", "000002"]
    a, b = ds[(1, 1)], clone[(1, 1)]
    for name in ("channels", "base", "gt", "mask"):
        assert torch.equal(a[name], b[name]), name
    assert a["sample_id"] == b["sample_id"] == ["000002", "000002"]


# --- base B5 y canales F3 --------------------------------------------------------------------


def test_b5_base_and_f3_channels_come_from_the_same_cache(world):
    """B5 se arma al vuelo de los ingredientes (la caché no cambia) y es la misma que
    `combine("B5", ...)` directo; F3 agrega 3 canales calculados en el frame completo."""
    from visionx.depth.fusion import cache

    root, train_refs, _, spec = world
    ds = dataset(world, variant="F3", base="B5", b5_params=B5, crop=None, crops_per_image=1)
    frame = ds.full_frame(0)
    assert frame["channels"].shape[0] == len(F3_CHANNELS) == 14
    assert np.isfinite(frame["channels"]).all()
    sample = train_refs[0].load()
    ing = cache.read(sample, NUM_DISPARITIES, spec, root)
    direct = combine("B5", ing.stereo(), ing.mono_relative, 0.0, B5.align, local=B5.local)
    base_px = np.where(direct.valid, direct.disparity_px, 0.0)
    assert np.array_equal(frame["base_px"][0], base_px)
    assert np.array_equal(frame["b5_px"][0], base_px)  # la base es B5
    b4 = ing.base(0.0, spec.align)
    assert np.array_equal(frame["b4_px"][0], np.where(b4.valid, b4.disparity_px, 0.0))
    # Recortar no recalcula: el recorte es una ventana del frame completo (canales F3 incluidos).
    crops = dataset(world, variant="F3", base="B5", b5_params=B5)[(1, 0)]
    assert crops["channels"].shape[1] == 14


def test_b5_or_f3_without_b5_params_is_rejected(world, tmp_path):
    with pytest.raises(Exception, match="b5"):
        dataset(world, variant="F3")
    with pytest.raises(Exception, match="b5"):
        dataset(world, base="B5")
    with pytest.raises(Exception, match="b5"):
        run_training(small_config(variant="F3"), tmp_path / "x", inputs(world),
                     log=lambda _: None)  # fmt: skip
    with pytest.raises(Exception, match="base"):
        TrainConfig(base="B3")


def test_training_with_b5_base_and_f3_reports_both_controls(world, tmp_path):
    inp = inputs(world)
    inp.b5_params = B5
    cfg = small_config(
        epochs=1,
        variant="F3",
        base="B5",
        model=ModelConfig(widths=(8, 16), norm="batch", bottleneck_dilations=(2,)),
    )
    run_training(cfg, tmp_path / "v2", inp, log=lambda _: None)
    with (tmp_path / "v2" / "curvas.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    for key in ("val_epe_comun_b4", "val_epe_comun_b5", "val_epe_comun_fusion", "val_epe_base"):
        assert np.isfinite(float(row[key])), key
    model, ckpt = load_checkpoint(tmp_path / "v2" / "mejor.pt")
    assert ckpt["base"] == "B5" and ckpt["variant"] == "F3"
    assert model.config.in_channels == 14 and model.config.bottleneck_dilations == (2,)


def test_old_curve_columns_are_renamed_when_resuming(tmp_path):
    path = tmp_path / "curvas.csv"
    path.write_text("epoca,val_epe_b4,epe_train_b4_px\n1,2.5,1.5\n2,2.4,1.4\n", encoding="utf-8")
    rows = train_mod._read_curves(path, 1)
    assert rows == [{"epoca": "1", "val_epe_base": "2.5", "epe_train_base_px": "1.5"}]
