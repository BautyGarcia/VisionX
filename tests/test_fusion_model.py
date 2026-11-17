"""Tests de T12: red Fusion y pérdidas. Sin pesos, GPU ni datos reales (tensores sintéticos)."""

import pytest

torch = pytest.importorskip("torch")

from visionx.depth.fusion.loss import combined_loss, epe_px, l1_loss, laplace_nll  # noqa: E402
from visionx.depth.fusion.model import (  # noqa: E402
    NORMS,
    FusionNet,
    ModelConfig,
    count_parameters,
    receptive_field,
)
from visionx.shared.errors import InvalidInput  # noqa: E402

W_REF = 100.0  # ancho de referencia chico: 1 px = 0,01 en unidades d/W_ref


def small_net(seed: int = 0, norm: str = "group") -> FusionNet:
    torch.manual_seed(seed)
    return FusionNet(ModelConfig(norm=norm))


def random_inputs(n=1, h=32, w=40, seed=1):
    gen = torch.Generator().manual_seed(seed)
    channels = torch.rand(n, 11, h, w, generator=gen) - 0.5
    base = torch.rand(n, 1, h, w, generator=gen) * 0.2
    return channels, base


# --- configuración y tamaño ---


def test_parameter_count_under_budget():
    assert count_parameters(FusionNet(ModelConfig())) <= 2_500_000


def test_config_roundtrip():
    cfg = ModelConfig(in_channels=10, widths=(16, 32, 64), norm_groups=4, b_init=0.3)
    data = cfg.to_dict()
    assert isinstance(data["widths"], list)  # serializable
    assert ModelConfig.from_dict(data) == cfg
    assert ModelConfig.from_dict({**data, "campo_extra": 1}) == cfg


def test_config_invalid():
    with pytest.raises(InvalidInput):
        ModelConfig(widths=(30, 64))  # no múltiplo de norm_groups
    with pytest.raises(InvalidInput):
        ModelConfig(b_init=0.0)
    with pytest.raises(InvalidInput):
        ModelConfig(widths=(32,))


def test_forward_rejects_bad_shapes():
    net = small_net()
    channels, base = random_inputs()
    with pytest.raises(InvalidInput):
        net(channels[:, :5], base)
    with pytest.raises(InvalidInput):
        net(channels, base[:, :, :-1])


# --- criterios M5 ---


@pytest.mark.parametrize("norm", NORMS)
def test_initial_residual_is_zero(norm):
    net = small_net(norm=norm).eval()
    channels, base = random_inputs()
    with torch.no_grad():
        out = net(channels, base)
    # |disparity - base| * W_ref <= 1e-5 px
    assert float((out["disparity"] - base).abs().max()) * W_REF <= 1e-5
    assert float(out["delta"].abs().max()) == 0.0


@pytest.mark.parametrize("norm", NORMS)
def test_initial_b_matches_config(norm):
    cfg = ModelConfig()
    net = small_net(norm=norm).eval()
    channels, base = random_inputs()
    with torch.no_grad():
        b = net(channels, base)["b"]
    expected = cfg.residual_scale * cfg.b_init
    assert float(b.mean()) == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("norm", NORMS)
def test_no_nan_when_both_sources_fail(norm):
    net = small_net(norm=norm).eval()
    channels = torch.zeros(2, 11, 32, 40)
    base = torch.zeros(2, 1, 32, 40)
    with torch.no_grad():
        out = net(channels, base)
    for name, tensor in out.items():
        assert torch.isfinite(tensor).all(), name
    assert (out["b"] > 0).all()


@pytest.mark.parametrize("norm", NORMS)
def test_b_positive_after_random_weights(norm):
    net = small_net(norm=norm)
    for p in net.parameters():  # perturbación grande para forzar valores extremos
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.5)
    channels, base = random_inputs()
    out = net(channels * 10, base)
    assert (out["b"] > 0).all() and torch.isfinite(out["b"]).all()


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("hw", [(67, 93), (33, 33), (16, 16), (9, 50)])
def test_non_multiple_sizes(hw, norm):
    net = small_net(norm=norm).eval()
    channels, base = random_inputs(n=2, h=hw[0], w=hw[1])
    with torch.no_grad():
        out = net(channels, base)
    for tensor in out.values():
        assert tuple(tensor.shape) == (2, 1, hw[0], hw[1])


def test_config_without_norm_is_group_and_norm_is_validated():
    # Checkpoints de antes de la opción: sin la clave "norm" => GroupNorm (lo que tenían).
    old = {k: val for k, val in ModelConfig().to_dict().items() if k != "norm"}
    assert ModelConfig.from_dict(old).norm == "group"
    assert ModelConfig.from_dict(ModelConfig(norm="batch").to_dict()).norm == "batch"
    with pytest.raises(InvalidInput):
        ModelConfig(norm="layer")
    ModelConfig(widths=(30, 60), norm="batch")  # norm_groups solo importa con GroupNorm
    group = FusionNet(ModelConfig(norm="group")).state_dict()
    old_net = FusionNet(ModelConfig.from_dict(old))
    old_net.load_state_dict(group)  # pesos viejos cargan en la red reconstruida


def _crop_vs_full(norm: str, dilations: tuple[int, ...] = ()) -> float:
    """Diferencia máxima, lejos del borde del recorte, entre la salida de la red sobre un
    recorte y sobre el frame completo (mismos píxeles), con la red en eval()."""
    torch.manual_seed(0)
    cfg = ModelConfig(widths=(8, 16, 32), norm_groups=4, norm=norm, bottleneck_dilations=dilations)
    net = FusionNet(cfg)
    for p in net.delta_head.parameters():  # cabeza delta no nula: que la salida dependa de algo
        torch.nn.init.normal_(p, std=0.1)
    if norm == "batch":  # medias móviles distintas de las iniciales (0 y 1)
        net.train()
        with torch.no_grad():
            for i in range(3):
                net(*random_inputs(n=4, h=64, w=64, seed=10 + i))
    net.eval()
    # Centro del recorte: a más de medio campo receptivo de sus bordes (56 px sin dilaciones,
    # `receptive_field`), así el relleno del borde no llega. Con dilaciones el recorte crece.
    m = receptive_field(cfg) // 2 + 4
    h, w = 2 * m + 16, 2 * m + 48
    h, w = h + (-h) % 4, w + (-w) % 4  # múltiplos de 4: sin relleno replicado en el recorte
    y, x = 32, 48  # esquina múltiplo de 4 (2 reducciones x2): misma grilla
    gen = torch.Generator().manual_seed(3)
    big_h, big_w = y + h + 32, x + w + 80
    channels = torch.rand(1, 11, big_h, big_w, generator=gen) - 0.5
    channels[..., :, x + w + 20 :] += 4.0  # una zona muy distinta, fuera del recorte
    base = torch.rand(1, 1, big_h, big_w, generator=gen) * 0.2
    with torch.no_grad():
        full = net(channels, base)["disparity"][..., y : y + h, x : x + w]
        crop = net(channels[..., y : y + h, x : x + w], base[..., y : y + h, x : x + w])
    diff = (full - crop["disparity"])[..., m : h - m, m : w - m].abs().max()
    return float(diff) * W_REF


def test_batchnorm_in_eval_is_local_and_groupnorm_is_not():
    # Con BatchNorm en eval() (y sin normalización) la salida en un píxel depende solo de su
    # vecindad: recorte (entrenamiento) y frame completo (validación, app) coinciden lejos del
    # borde. Con GroupNorm no: normaliza con las estadísticas de toda la entrada, y la zona
    # distinta que el recorte no ve cambia la salida en el centro.
    assert _crop_vs_full("batch") <= 1e-4
    assert _crop_vs_full("none") <= 1e-4
    assert _crop_vs_full("group") > 1e-2
    # Los bloques dilatados del nivel bajo (v2) siguen siendo locales en eval().
    assert _crop_vs_full("batch", (2, 4)) <= 1e-4


@pytest.mark.parametrize("norm", NORMS)
def test_fresh_network_reproduces_b4_in_real_pixels(norm):
    """M5 en px reales (W = 1080, d hasta 272): la base se suma en px, sin pasar por d/W."""
    from visionx.depth.fusion.channels import disparity_from_residual

    width = 1080
    gen = torch.Generator().manual_seed(5)
    base_px = (torch.rand(24, width, generator=gen) * 272).numpy().astype("float32")
    base_valid = base_px > 5.0
    base_norm = torch.from_numpy(base_px / width)[None, None]
    channels = torch.rand(1, 11, 24, width, generator=gen) - 0.5
    net = small_net(norm=norm).eval()
    with torch.no_grad():
        delta = net(channels, base_norm)["delta"][0, 0].numpy()
    fusion = disparity_from_residual(base_px, base_valid, delta, width)
    diff = abs(fusion - base_px)[base_valid]
    assert diff.max() <= 1e-5
    # Por qué: la ida y vuelta px -> d/W -> px en float32 sí pierde más que eso.
    roundtrip = (base_px / width).astype("float32") * width
    assert abs(roundtrip - base_px)[base_valid].max() > 1e-5


# --- pérdidas ---


def loss_case():
    gen = torch.Generator().manual_seed(3)
    n, h, w = 2, 8, 10
    gt = torch.rand(n, 1, h, w, generator=gen) * 0.2
    disparity = (gt + 0.02 * torch.randn(n, 1, h, w, generator=gen)).requires_grad_()
    b = (torch.rand(n, 1, h, w, generator=gen) * 0.05 + 0.01).requires_grad_()
    mask = torch.rand(n, 1, h, w, generator=gen) > 0.4
    return disparity, b, gt, mask


def test_gradient_zero_outside_mask():
    disparity, b, gt, mask = loss_case()
    gt = torch.where(mask, gt, torch.full_like(gt, float("nan")))  # basura fuera de la máscara
    for fn in (
        lambda: laplace_nll(disparity, b, gt, mask, W_REF),
        lambda: l1_loss(disparity, gt, mask, W_REF),
    ):
        disparity.grad = None
        b.grad = None
        value = fn()
        assert torch.isfinite(value)
        value.backward()
        assert torch.isfinite(disparity.grad).all()
        assert float(disparity.grad[~mask].abs().max()) == 0.0
        if b.grad is not None:
            assert float(b.grad[~mask].abs().max()) == 0.0
        assert float(disparity.grad[mask].abs().max()) > 0.0


def test_nll_minimal_when_b_equals_error():
    # Laplace: para un |error| fijo, la NLL es mínima en b = |error|.
    n, h, w = 1, 6, 6
    gt = torch.full((n, 1, h, w), 0.1)
    disparity = gt + 0.03
    mask = torch.ones(n, 1, h, w, dtype=torch.bool)
    values = {
        scale: float(laplace_nll(disparity, torch.full_like(gt, 0.03 * scale), gt, mask, W_REF))
        for scale in (0.5, 0.8, 1.0, 1.25, 2.0)
    }
    assert min(values, key=values.get) == 1.0


def test_nll_value_in_pixels():
    gt = torch.zeros(1, 1, 2, 2)
    disparity = torch.full_like(gt, 0.04)  # 4 px con W_REF = 100
    b = torch.full_like(gt, 0.02)  # 2 px
    mask = torch.ones_like(gt, dtype=torch.bool)
    value = float(laplace_nll(disparity, b, gt, mask, W_REF))
    assert value == pytest.approx(4 / 2 + torch.log(torch.tensor(4.0)).item(), abs=1e-5)


def test_ref_width_per_sample():
    gt = torch.zeros(2, 1, 3, 3)
    disparity = torch.full_like(gt, 0.01)
    mask = torch.ones_like(gt, dtype=torch.bool)
    value = l1_loss(disparity, gt, mask, torch.tensor([100.0, 300.0]))
    assert float(value) == pytest.approx((1.0 + 3.0) / 2)
    with pytest.raises(InvalidInput):
        l1_loss(disparity, gt, mask, torch.tensor([1.0, 2.0, 3.0]))


def test_empty_mask_gives_zero_with_graph():
    disparity, b, gt, _ = loss_case()
    mask = torch.zeros_like(gt, dtype=torch.bool)
    for value in (laplace_nll(disparity, b, gt, mask, W_REF), l1_loss(disparity, gt, mask, W_REF)):
        assert value.item() == 0.0
        value.backward()  # el grafo es válido
    assert float(epe_px(disparity, gt, mask, W_REF)) == 0.0


def test_nonfinite_outside_mask_does_not_contaminate():
    disparity, b, gt, mask = loss_case()
    clean = laplace_nll(disparity, b, gt, mask, W_REF)
    dirty_gt = torch.where(mask, gt, torch.full_like(gt, float("inf")))
    dirty_gt[~mask] = float("nan")
    dirty = laplace_nll(disparity, b, dirty_gt, mask, W_REF)
    assert dirty.item() == pytest.approx(clean.item())  # .item(): sin aviso por requires_grad
    assert torch.isfinite(epe_px(disparity, dirty_gt, mask, W_REF))


def test_combined_loss_and_epe():
    disparity, b, gt, mask = loss_case()
    out = {"disparity": disparity, "b": b}
    nll, parts = combined_loss(out, gt, mask, W_REF)
    assert set(parts) == {"nll", "total"}
    assert nll.item() == pytest.approx(laplace_nll(disparity, b, gt, mask, W_REF).item())
    both, parts = combined_loss(out, gt, mask, W_REF, kind="nll", l1_weight=0.5)
    assert both.item() == pytest.approx(
        nll.item() + 0.5 * l1_loss(disparity, gt, mask, W_REF).item()
    )
    only_l1, parts = combined_loss(out, gt, mask, W_REF, kind="l1")
    assert only_l1.item() == pytest.approx(epe_px(disparity, gt, mask, W_REF).item())
    assert not epe_px(disparity, gt, mask, W_REF).requires_grad
    with pytest.raises(InvalidInput):
        combined_loss(out, gt, mask, W_REF, kind="otra")


# --- mini-overfit ---


def test_overfit_corrects_known_bias():
    """La red aprende a quitar un sesgo que depende de un canal (criterio M5: EPE baja >= 80 %)."""
    torch.manual_seed(0)
    n, h, w = 2, 48, 64
    gen = torch.Generator().manual_seed(5)
    channels = torch.zeros(n, 11, h, w)
    # Canal 6 ("error_foto"): parches de 0 / 1 que marcan dónde la base está sesgada.
    patches = (torch.rand(n, 1, 6, 8, generator=gen) > 0.5).float()
    channels[:, 6:7] = torch.nn.functional.interpolate(patches, size=(h, w), mode="nearest")
    channels[:, :3] = torch.rand(n, 3, h, w, generator=gen) - 0.5  # ruido sin información
    gt = 0.1 + 0.05 * torch.rand(n, 1, 1, w, generator=gen).expand(n, 1, h, w)
    bias = 0.04  # 4 px con W_REF = 100
    base = gt + bias * channels[:, 6:7]
    mask = torch.ones(n, 1, h, w, dtype=torch.bool)

    net = FusionNet(ModelConfig())
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    epe_before = float(epe_px(base, gt, mask, W_REF))
    for _ in range(150):
        optimizer.zero_grad()
        loss, _ = combined_loss(net(channels, base), gt, mask, W_REF)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        out = net(channels, base)
    epe_after = float(epe_px(out["disparity"], gt, mask, W_REF))
    print(f"EPE base {epe_before:.3f} px -> {epe_after:.3f} px")
    assert epe_after <= 0.2 * epe_before, (epe_before, epe_after)
    assert torch.isfinite(out["b"]).all() and (out["b"] > 0).all()


# --- contexto: bloques dilatados (v2) -------------------------------------------------------


def test_receptive_field_matches_the_layer_count():
    # Cuenta a mano de la red de v1 (4 niveles): 124 px; la de v2 pasa de mil.
    assert receptive_field(ModelConfig()) == 124
    assert receptive_field(ModelConfig(widths=(8, 16, 32))) == 56
    v2 = ModelConfig(widths=(32, 64, 128, 192, 256), bottleneck_dilations=(2, 4, 8))
    assert receptive_field(v2) == 1156
    # Cota empírica: ningún píxel fuera del campo teórico influye en la salida (gradiente).
    # El max-pool solo pasa gradiente por el máximo de cada ventana, así que lo medido puede
    # quedar hasta 1 px por reducción (en px del nivel) por debajo del teórico, nunca encima.
    cfg = ModelConfig(in_channels=2, widths=(4, 8, 16), norm="none", bottleneck_dilations=(2,))
    torch.manual_seed(0)
    net = FusionNet(cfg).eval()
    x = torch.zeros(1, 2, 192, 192, requires_grad=True)
    net(x, torch.zeros(1, 1, 192, 192))["b"][0, 0, 96, 96].backward()
    rows = (x.grad[0].abs().sum(0) > 0).any(1).nonzero()
    measured = int(rows.max() - rows.min() + 1)
    assert receptive_field(cfg) - 3 <= measured <= receptive_field(cfg)


@pytest.mark.parametrize("norm", NORMS)
def test_dilated_bottleneck_keeps_zero_initial_residual(norm):
    torch.manual_seed(0)
    net = FusionNet(ModelConfig(norm=norm, bottleneck_dilations=(2, 4, 8))).eval()
    channels, base = random_inputs(h=48, w=72)
    with torch.no_grad():
        out = net(channels, base)
    assert float((out["disparity"] - base).abs().max()) * W_REF <= 1e-5
    assert torch.isfinite(out["b"]).all() and (out["b"] > 0).all()


def test_dilations_roundtrip_and_old_checkpoints_load():
    cfg = ModelConfig(bottleneck_dilations=(2, 4))
    data = cfg.to_dict()
    assert data["bottleneck_dilations"] == [2, 4]  # YAML/JSON sin tuplas
    assert ModelConfig.from_dict(data) == cfg
    old = {k: val for k, val in ModelConfig().to_dict().items() if k != "bottleneck_dilations"}
    assert ModelConfig.from_dict(old).bottleneck_dilations == ()
    FusionNet(ModelConfig.from_dict(old)).load_state_dict(FusionNet(ModelConfig()).state_dict())
    with pytest.raises(InvalidInput):
        ModelConfig(bottleneck_dilations=(0,))
    more = count_parameters(FusionNet(cfg)) - count_parameters(FusionNet(ModelConfig()))
    assert more == 2 * 2 * (192 * 192 * 9 + 2 * 192)  # 2 bloques x 2 conv 3x3 + 2 normas
