"""Red que estima la cámara: focal, centro y distorsión, cada uno con su b de Laplace.

- Un codificador ResNet-18 (preentrenado en ImageNet) que comparten los dos ojos. A la imagen se
  le suman dos canales con la coordenada de cada píxel (x, y relativos al centro de la imagen):
  sin ellos, la red no sabe dónde está cada cosa y no puede ubicar el centro ni la distorsión.
- Los rasgos de la última capa se reducen a 64 canales y se promedian en una grilla de 3×4
  zonas (no en un solo punto): la curvatura de las rectas cambia de una zona a otra.
- Se promedian los dos ojos (y, más adelante, varios cuadros) y una cabeza da la media y log b
  de las cinco etiquetas en unidades de `lente.ESCALA`.
"""

from __future__ import annotations

import torch
from torch import nn

from visionx.depth.camara_net.lente import ESCALA

N_PARAMS = 5
LOG_B_MIN, LOG_B_MAX = -4.0, 4.0


def _coordenadas(x: torch.Tensor) -> torch.Tensor:
    """Dos canales: x en [-1, 1] y y en [-alto/ancho, alto/ancho] (misma escala que x)."""
    b, _, h, w = x.shape
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype) * (h / w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy]).expand(b, 2, h, w)


class RedCamara(nn.Module):
    def __init__(self, preentrenada: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if preentrenada else None)
        conv = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        with torch.no_grad():  # los 3 canales de color arrancan con los pesos de ImageNet
            conv.weight.zero_()
            conv.weight[:, :3] = base.conv1.weight
        self.tronco = nn.Sequential(conv, base.bn1, base.relu, base.maxpool, base.layer1,
                                    base.layer2, base.layer3, base.layer4)  # fmt: skip
        self.reducir = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU(inplace=True))
        self.zonas = nn.AdaptiveAvgPool2d((3, 4))
        self.cabeza = nn.Sequential(
            nn.Linear(64 * 12 + 1, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, 2 * N_PARAMS),
        )  # fmt: skip
        self.register_buffer("escala", torch.from_numpy(ESCALA.copy()))
        # Arranque: medias en 0 (se suman al centro de la distribución de entrenamiento, que se
        # fija con `fijar_centro`) y b chico pero no mínimo.
        self.register_buffer("centro", torch.zeros(N_PARAMS))
        nn.init.zeros_(self.cabeza[-1].weight)
        nn.init.zeros_(self.cabeza[-1].bias)

    def fijar_centro(self, y_medio: torch.Tensor) -> None:
        self.centro.copy_(y_medio)

    def rasgos(self, ojos: torch.Tensor) -> torch.Tensor:
        """ojos (B, V, 3, H, W) → rasgos (B, 768) promediados sobre las V imágenes."""
        b, v = ojos.shape[:2]
        x = ojos.flatten(0, 1)
        x = torch.cat([x, _coordenadas(x)], 1)
        f = self.reducir(self.tronco(x))
        h, w = f.shape[-2:]
        if f.device.type == "mps" and (h % 3 or w % 4):
            # MPS no sabe hacer el promedio adaptativo con zonas desparejas (p. ej. 1280×800 da
            # 13×20 rasgos); en la CPU el resultado es el mismo.
            f = self.zonas(f.cpu()).to(f.device)
        else:
            f = self.zonas(f)
        return f.flatten(1).view(b, v, -1).mean(1)

    def forward(self, ojos: torch.Tensor, aspecto: torch.Tensor) -> dict[str, torch.Tensor]:
        """Media (en unidades de etiqueta) y log b (en unidades de `ESCALA`) de cada parámetro."""
        f = self.rasgos(ojos)
        out = self.cabeza(torch.cat([f, aspecto[:, None].to(f.dtype)], 1))
        mu = self.centro + out[:, :N_PARAMS] * self.escala * 10
        log_b = out[:, N_PARAMS:].clamp(LOG_B_MIN, LOG_B_MAX)
        return {"mu": mu, "log_b": log_b}


def b_en_unidades(log_b: torch.Tensor, escala: torch.Tensor) -> torch.Tensor:
    """b de Laplace en unidades de cada etiqueta."""
    return torch.exp(log_b) * escala
