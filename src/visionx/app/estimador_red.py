"""La red de entrada de Bauty (`depth.camara_net`, #90) para la pantalla «Cámara nueva».

Cumple `camara_nueva.EstimadorCamara`: mira los pares que juntó el alta (los dos ojos, tal
cual, a 640 px de ancho como en su evaluación), combina los cuadros pesando por 1/b² y
devuelve campo, centro, lente y la duda de log f. La duda se agranda por un factor medido:
1,79 lleva val al 90 %, y fuera de val (gt-video) hace falta el doble (BITACORA 22/11).

La distorsión de la red solo se usa si es clara frente a su duda: la red casi no la aprende y
una k1 inventada torcería una cámara que no la tiene.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from visionx.app.camara_nueva import EstimacionCamara

FACTOR_B = 1.79 * 2
MAX_PARES = 8


class EstimadorRed:
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        from visionx.depth.camara_net.entrenar import cargar

        self.modelo, _, self.device = cargar(checkpoint, device)
        self.origen = f"red de entrada ({Path(checkpoint).parent.name})"

    def _entrada(self, izq: np.ndarray, der: np.ndarray):
        import torch

        from visionx.depth.camara_net.datos import ANCHO_RED, DESVIO, MEDIA, alto_red

        h, w = izq.shape[:2]
        tam = (ANCHO_RED, alto_red(w, h))
        ojos = []
        for img in (izq, der):
            x = cv2.resize(img, tam, interpolation=cv2.INTER_AREA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            ojos.append(((x - MEDIA) / DESVIO).transpose(2, 0, 1))
        t = torch.from_numpy(np.stack(ojos)[None]).to(self.device)
        return t, torch.tensor([tam[1] / tam[0]], dtype=torch.float32, device=self.device)

    def estimar(self, pares: Sequence[tuple[np.ndarray, np.ndarray]]) -> EstimacionCamara:
        import torch

        from visionx.depth.camara_net.entrenar import combinar
        from visionx.depth.camara_net.modelo import b_en_unidades

        paso = max(1, len(pares) // MAX_PARES)
        mus, bs = [], []
        with torch.no_grad():
            for izq, der in list(pares)[::paso][:MAX_PARES]:
                out = self.modelo(*self._entrada(izq, der))
                mus.append(out["mu"][0].cpu().numpy())
                bs.append(b_en_unidades(out["log_b"], self.modelo.escala)[0].cpu().numpy())
        mu, b = combinar(np.stack(mus), np.stack(bs))
        s = math.sqrt(2) * FACTOR_B * b  # σ de cada etiqueta (Laplace: σ = √2·b)
        k1 = float(mu[3]) if abs(mu[3]) > 1.645 * s[3] else 0.0
        k2 = float(mu[4]) if k1 and abs(mu[4]) > 1.645 * s[4] else 0.0
        campo = math.degrees(2 * math.atan(0.5 / math.exp(float(mu[0]))))
        return EstimacionCamara(
            campo_horizontal_deg=campo, centro_rel=(float(mu[1]), float(mu[2])), k1=k1, k2=k2,
            confianza=float(max(0.0, min(1.0, 1 - s[0] / 0.2))), origen=self.origen,
            sigma_log_f=float(s[0]),
        )  # fmt: skip


def buscar_checkpoint() -> Path | None:
    """El checkpoint de la red de entrada si está bajado (v2 primero)."""
    from visionx.shared.config import ROOT

    for v in ("e2-v2", "e2-v1"):
        p = ROOT / "outputs" / "camara_net" / v / "mejor.pt"
        if p.exists():
            return p
    return None
