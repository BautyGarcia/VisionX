"""CNN Fusion (T12): U-Net residual compacta que corrige B4 y estima incertidumbre.

Unidad: todo en **d/W_ref** (disparidad dividida por el ancho en px del frame preparado
completo, no el del recorte). La red no predice la disparidad desde cero: parte de la base B4
y suma un residual `delta`, así que una red recién creada reproduce B4 exactamente (la cabeza
delta arranca en cero). Además estima `b`, la escala de una distribución de Laplace sobre el
error (`loss.laplace_nll`); b siempre es positiva.

La red no decide validez: quien la usa aplica `base_valid` (donde B4 no respondió, la base
vale 0 y la salida no debe leerse).

**Normalización y localidad** (`ModelConfig.norm`). Se entrena con recortes y se aplica sobre el
frame completo, así que importa si la salida en un píxel depende solo de su vecindad (el campo
receptivo) o de toda la imagen:
- "group" (GroupNorm): normaliza con media y varianza **de cada imagen de entrada**. La salida
  en un píxel depende de toda la imagen: el mismo punto da otra salida en un recorte que en el
  frame completo. No es local, ni en entrenamiento ni en `eval()`.
- "batch" (BatchNorm): en `train()` usa las estadísticas del lote; en `eval()`, las medias
  móviles aprendidas, que son constantes. En `eval()` la red es local: recorte y frame completo
  dan lo mismo lejos del borde. Exige `model.train()` al entrenar y `model.eval()` al validar y
  aplicar (el paquete y el estimador cargan la red en `eval()`).
- "none": sin normalización (convoluciones con sesgo). Local siempre.
Checkpoints sin la clave `norm` son de antes de esta opción: "group".

**Contexto** (`ModelConfig.bottleneck_dilations`, v2). Una U-Net de 4 niveles ve ~124 px
(`receptive_field`); las anclas útiles para corregir un hueco están a cientos de px. Cada
entrada de la tupla agrega, en el nivel más bajo, un bloque residual con dos convoluciones 3x3
dilatadas (los taps separados `dilación` celdas): con dilaciones (2, 4, 8) a 1/16 de
resolución el campo receptivo pasa de cientos a más de mil px sin agregar reducciones. Las
convoluciones dilatadas siguen siendo locales (con BatchNorm en eval()). Checkpoints sin la
clave: tupla vacía, la red de antes.

torch es del extra `ml`; se importa acá, al nivel del módulo, porque todo el archivo es de
torch. El resto de VisionX no importa este módulo salvo que se lo pida (ver `visionx.depth.mono`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from visionx.shared.errors import InvalidInput

NORMS = ("group", "batch", "none")


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparámetros de la arquitectura; viajan dentro del checkpoint."""

    in_channels: int = 11  # canales de channels.py (F1); F0 usa 10
    widths: tuple[int, ...] = (32, 64, 128, 192)  # canales por nivel; len-1 reducciones x2
    norm_groups: int = 8  # grupos de GroupNorm (solo con norm="group")
    residual_scale: float = 1 / 128  # 1 unidad cruda de red = W_ref/128 px
    b_floor: float = 1e-3  # piso de b en unidades de red; evita b = 0
    b_init: float = 0.25  # b inicial en unidades de red (~2 px con W_ref ~ 1080)
    norm: str = "group"  # group | batch | none (ver docstring del módulo: localidad)
    bottleneck_dilations: tuple[int, ...] = ()  # bloques residuales dilatados en el nivel bajo

    def __post_init__(self) -> None:
        if any(int(dil) < 1 for dil in self.bottleneck_dilations):
            raise InvalidInput(f"dilaciones >= 1, vino {self.bottleneck_dilations}")
        if self.norm not in NORMS:
            raise InvalidInput(f"norm {self.norm!r}: se esperaba una de {NORMS}")
        if self.in_channels < 1:
            raise InvalidInput(f"in_channels debe ser >= 1, vino {self.in_channels}")
        if len(self.widths) < 2 or any(w < 1 for w in self.widths):
            raise InvalidInput(f"widths necesita al menos 2 niveles positivos: {self.widths}")
        if self.norm == "group" and any(w % self.norm_groups for w in self.widths):
            raise InvalidInput(
                f"cada ancho de {self.widths} debe ser múltiplo de norm_groups={self.norm_groups}"
            )
        if not (math.isfinite(self.residual_scale) and self.residual_scale > 0):
            raise InvalidInput(f"residual_scale debe ser > 0, vino {self.residual_scale}")
        if not (math.isfinite(self.b_floor) and self.b_floor >= 0):
            raise InvalidInput(f"b_floor debe ser >= 0, vino {self.b_floor}")
        if not (math.isfinite(self.b_init) and self.b_init > self.b_floor):
            raise InvalidInput(f"b_init debe ser > b_floor, vino {self.b_init}")

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["widths"] = list(self.widths)  # JSON/YAML no tienen tuplas
        data["bottleneck_dilations"] = list(self.bottleneck_dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        names = set(cls.__dataclass_fields__)
        kwargs = {k: val for k, val in data.items() if k in names}
        if "widths" in kwargs:
            kwargs["widths"] = tuple(int(w) for w in kwargs["widths"])
        if "bottleneck_dilations" in kwargs:
            kwargs["bottleneck_dilations"] = tuple(int(x) for x in kwargs["bottleneck_dilations"])
        return cls(**kwargs)


def receptive_field(config: ModelConfig) -> int:
    """Campo receptivo teórico de `FusionNet`, en px de entrada (camino más largo: bajar hasta
    el último nivel, los bloques dilatados y volver a subir).

    Cuenta estándar capa por capa: con `jump` = distancia en px de entrada entre celdas
    vecinas del mapa actual, una capa de núcleo efectivo k suma (k − 1)·jump. Conv 3x3 con
    dilación r: k = 2r + 1. Max-pool 2x2 paso 2: suma 1·jump y duplica jump. Interpolación
    bilineal x2: cada salida mezcla 2 celdas vecinas del mapa grueso, suma 1·jump grueso.
    Es el alcance máximo; el peso efectivo cae mucho antes del borde (campo receptivo efectivo).
    """
    rf, jump = 1, 1

    def conv(dilation: int = 1) -> None:
        nonlocal rf
        rf += 2 * dilation * jump  # núcleo 3x3: k − 1 = 2·dilación

    levels = len(config.widths)
    conv(), conv()  # nivel 0
    for _ in range(1, levels):
        rf += jump  # max-pool 2x2
        jump *= 2
        conv(), conv()
    for dilation in config.bottleneck_dilations:
        conv(dilation), conv(dilation)
    for _ in range(1, levels):
        rf += jump  # bilineal x2 (taps del mapa grueso)
        jump //= 2
        conv()  # reduce
        conv(), conv()  # bloque del nivel
    return rf


def count_parameters(model: nn.Module) -> int:
    """Cantidad de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _norm_layer(kind: str, channels: int, groups: int) -> nn.Module:
    if kind == "group":
        return nn.GroupNorm(groups, channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class _DoubleConv(nn.Module):
    """Dos veces conv3x3 + normalización + SiLU (un bloque por nivel)."""

    def __init__(self, c_in: int, c_out: int, groups: int, norm: str = "group") -> None:
        super().__init__()
        # Con normalización el sesgo de la conv sobra (lo absorbe la norma); sin ella, no.
        bias = norm == "none"
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=bias),
            _norm_layer(norm, c_out, groups),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=bias),
            _norm_layer(norm, c_out, groups),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _DilatedResidual(nn.Module):
    """x + bloque(x), con dos conv 3x3 dilatadas: más alcance sin reducir resolución.

    Relleno = dilación: la salida tiene el tamaño de la entrada. La suma residual hace que el
    bloque recién creado no rompa lo que ya pasa por la red.
    """

    def __init__(self, channels: int, dilation: int, groups: int, norm: str) -> None:
        super().__init__()
        bias = norm == "none"
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=bias),
            _norm_layer(norm, channels, groups),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=bias),
            _norm_layer(norm, channels, groups),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class _Up(nn.Module):
    """Subida: interpolación bilineal al tamaño del skip + conv, concatenación y bloque."""

    def __init__(self, c_in: int, c_skip: int, c_out: int, groups: int, norm: str) -> None:
        super().__init__()
        # Interpolar + conv en vez de ConvTranspose: evita el patrón de tablero de ajedrez.
        self.reduce = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.block = _DoubleConv(c_out + c_skip, c_out, groups, norm)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([self.reduce(x), skip], dim=1))


class FusionNet(nn.Module):
    """U-Net residual: (canales, base) -> {disparity, delta, b}, todo en d/W_ref."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        w = cfg.widths
        g = cfg.norm_groups
        # Entrada: los canales de channels.py más la base B4 (1 canal). La base no es un canal
        # de preprocesamiento (mezcla estéreo y monocular), pero la red necesita verla para
        # saber qué está corrigiendo.
        self.encoders = nn.ModuleList(
            [_DoubleConv(cfg.in_channels + 1, w[0], g, cfg.norm)]
            + [_DoubleConv(w[i], w[i + 1], g, cfg.norm) for i in range(len(w) - 1)]
        )
        self.bottleneck = nn.Sequential(
            *[_DilatedResidual(w[-1], int(dil), g, cfg.norm) for dil in cfg.bottleneck_dilations]
        )
        self.ups = nn.ModuleList(
            [_Up(w[i + 1], w[i], w[i], g, cfg.norm) for i in reversed(range(len(w) - 1))]
        )
        self.delta_head = nn.Conv2d(w[0], 1, 1)
        self.b_head = nn.Conv2d(w[0], 1, 1)
        # Delta arranca en cero: la salida inicial es exactamente la base.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        # b inicial: softplus(bias) + b_floor = b_init  ->  bias = softplus^-1(b_init - b_floor).
        nn.init.normal_(self.b_head.weight, std=1e-3)
        nn.init.constant_(self.b_head.bias, math.log(math.expm1(cfg.b_init - cfg.b_floor)))

    def forward(self, channels: Tensor, base_norm: Tensor) -> dict[str, Tensor]:
        """channels N x C x H x W; base_norm N x 1 x H x W (d/W_ref, 0 sin base)."""
        cfg = self.config
        if channels.ndim != 4 or channels.shape[1] != cfg.in_channels:
            raise InvalidInput(
                f"channels debe ser N x {cfg.in_channels} x H x W, vino {tuple(channels.shape)}"
            )
        if base_norm.ndim != 4 or base_norm.shape[1] != 1:
            raise InvalidInput(f"base_norm debe ser N x 1 x H x W, vino {tuple(base_norm.shape)}")
        if base_norm.shape[0] != channels.shape[0] or base_norm.shape[-2:] != channels.shape[-2:]:
            raise InvalidInput("channels y base_norm deben tener el mismo N, H y W")

        height, width = channels.shape[-2:]
        # Con L-1 reducciones x2 el tamaño tiene que ser múltiplo de 2^(L-1).
        step = 2 ** (len(cfg.widths) - 1)
        pad_h, pad_w = (-height) % step, (-width) % step
        x = torch.cat([channels, base_norm], dim=1)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips: list[Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.max_pool2d(skips[-1], 2))
            skips.append(x)
        x = self.bottleneck(skips[-1])  # identidad sin dilaciones (Sequential vacío)
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        raw_delta = self.delta_head(x)[..., :height, :width]
        raw_b = self.b_head(x)[..., :height, :width]
        delta = cfg.residual_scale * raw_delta
        b = cfg.residual_scale * (F.softplus(raw_b) + cfg.b_floor)
        return {"disparity": base_norm + delta, "delta": delta, "b": b}
