"""U-Net de incertidumbre: cuánto creerle a cada píxel de la disparidad de CREStereo.

Es la segunda red propia (TECNICO, «CREStereo adaptable»). CREStereo devuelve una disparidad
por píxel pero no dice cuán segura es. Esta red mira esa disparidad junto con las imágenes y
predice **b**, la escala de una distribución de Laplace: el error esperado, en píxeles.

No se le pide adivinar: se le dan señales que se pueden medir y que suelen acompañar al error.

    azul, verde, rojo   la imagen izquierda: textura, brillos, zonas lisas
    fotometrico         diferencia entre la izquierda y la derecha traída con la disparidad
                        predicha; si la disparidad es correcta, las dos deberían coincidir
    visible             1 donde ese punto cae dentro de la imagen derecha; afuera no hay con
                        qué comparar
    borde               cuánto cambia la disparidad entre vecinos: los saltos de profundidad
                        son donde más se erra
    pasadas             cuánto difieren las dos pasadas de CREStereo (media resolución y
                        completa): donde la red cambió de opinión, está menos segura
    vertical            el canal vertical del flujo de CREStereo. Se entrenó hacia cero, así
                        que si se aparta de cero las filas no están coincidiendo

Los dos últimos salen gratis de la corrida de CREStereo (`matcher.last_extras`). Medidos solos
sobre InStereo2K val, «pasadas» ordena los errores casi tan bien como la primera versión de
esta red, que no los tenía (BITACORA 2026-11-19). Si una corrida no tuvo dos pasadas, ese canal
va en cero: la red se entrenó con dos pasadas y está pensada para el modo preciso.

La red es la misma U-Net de Fusion (`fusion/model.py`), con estos canales en lugar de los
suyos y la disparidad de CREStereo como base. Su salida de corrección arranca en cero y por
defecto no se entrena: acá importa b. La corrección queda como ablación.

Unidades, como en Fusion: adentro de la red todo va dividido por el ancho de la imagen
(d/W_ref), así la misma red sirve a otra resolución. Afuera, b sale en píxeles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.shared.errors import EstimatorError, InvalidInput

CHANNELS = ("azul", "verde", "rojo", "fotometrico", "visible", "borde", "pasadas", "vertical")
CHANNELS_VERSION = "crestereo-inc-2"
EDGE_SCALE_PX = 8.0  # un salto de 8 px entre vecinos ya cuenta como borde pleno
PASSES_SCALE_PX = 4.0  # 4 px de diferencia entre pasadas ya cuenta como desacuerdo pleno
VERTICAL_SCALE_PX = 2.0  # 2 px de flujo vertical ya cuenta como desalineación plena
DEFAULT_WIDTHS = (24, 48, 96, 144)


def warp_right(right: np.ndarray, disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trae la imagen derecha a la grilla de la izquierda usando la disparidad.

    El punto que la izquierda ve en la columna x, la derecha lo ve en x − d. Devuelve la
    derecha remuestreada y una máscara de dónde ese lugar cae dentro de la imagen.
    """
    height, width = disparity.shape
    if right.shape[:2] != (height, width):
        raise InvalidInput("right y disparity deben tener el mismo alto y ancho")
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    d = np.nan_to_num(disparity.astype(np.float32), nan=0.0)
    source = cols - d
    warped = cv2.remap(
        right, source, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    inside = np.isfinite(disparity) & (source >= 0) & (source <= width - 1)
    return warped, inside


def build_channels(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    half: np.ndarray | None = None,
    vertical: np.ndarray | None = None,
) -> np.ndarray:
    """Canales de entrada de la red: (8, alto, ancho) float32, todos entre 0 y 1.

    `half` es la disparidad de la pasada a media resolución, ya en px del tamaño final, y
    `vertical` el canal vertical del flujo. Sin ellos, sus canales van en cero.
    """
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise InvalidInput("left y right deben ser H x W x 3 del mismo tamaño")
    if disparity.shape != left.shape[:2]:
        raise InvalidInput("disparity debe tener el alto y ancho de las imágenes")
    image = left.astype(np.float32) / 255.0
    warped, inside = warp_right(right, disparity)
    photometric = np.abs(image - warped.astype(np.float32) / 255.0).mean(axis=2)
    photometric[~inside] = 0.0
    d = np.nan_to_num(disparity.astype(np.float32), nan=0.0)
    grad_x = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0  # /8: Sobel 3x3 a px por píxel
    grad_y = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    edge = np.clip(np.hypot(grad_x, grad_y) / EDGE_SCALE_PX, 0.0, 1.0)
    zeros = np.zeros_like(d)
    passes = zeros
    if half is not None:
        if half.shape != disparity.shape:
            raise InvalidInput("half debe tener la forma de disparity")
        passes = np.clip(np.abs(d - np.nan_to_num(half, nan=0.0)) / PASSES_SCALE_PX, 0.0, 1.0)
    rows = zeros
    if vertical is not None:
        if vertical.shape != disparity.shape:
            raise InvalidInput("vertical debe tener la forma de disparity")
        rows = np.clip(np.abs(np.nan_to_num(vertical, nan=0.0)) / VERTICAL_SCALE_PX, 0.0, 1.0)
    return np.concatenate(
        [
            image.transpose(2, 0, 1),
            photometric[None],
            inside[None].astype(np.float32),
            edge[None],
            passes[None],
            rows[None],
        ]
    ).astype(np.float32)


def model_config(widths: tuple[int, ...] = DEFAULT_WIDTHS) -> Any:
    """Config de la U-Net de Fusion para estos canales."""
    from visionx.depth.fusion.model import ModelConfig

    return ModelConfig(in_channels=len(CHANNELS), widths=tuple(widths), norm="group")


class UncertaintyHead:
    """Red entrenada lista para usar: (izquierda, derecha, disparidad) -> b en píxeles."""

    def __init__(self, model: Any, device: str, meta: dict[str, Any] | None = None) -> None:
        self.model = model
        self.device = device
        self.meta = meta or {}

    @classmethod
    def load(cls, checkpoint: str | Path, device: str = "auto") -> UncertaintyHead:
        import torch

        from visionx.depth.fusion.model import FusionNet
        from visionx.depth.mono import pick_device

        path = Path(checkpoint)
        if not path.exists():
            raise EstimatorError(f"no existe el checkpoint de incertidumbre {path}")
        # weights_only=True: tensores y tipos básicos, sin ejecutar código guardado.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("canales") != list(CHANNELS) or state.get("version") != CHANNELS_VERSION:
            raise EstimatorError(f"{path} se entrenó con otros canales")
        device = pick_device(device)
        model = FusionNet(model_config(tuple(state["widths"])))
        model.load_state_dict(state["model"])
        model.requires_grad_(False)
        model.to(device).eval()
        return cls(model, device, {k: v for k, v in state.items() if k != "model"})

    def predict(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        half: np.ndarray | None = None,
        vertical: np.ndarray | None = None,
    ) -> np.ndarray:
        """b por píxel, en px de la imagen que entra. NaN donde la disparidad no es finita."""
        import torch

        width = left.shape[1]
        arrays = build_channels(left, right, disparity, half, vertical)
        channels = torch.from_numpy(arrays)[None].to(self.device)
        base = np.nan_to_num(disparity.astype(np.float32), nan=0.0) / width
        base_t = torch.from_numpy(base)[None, None].to(self.device)
        with torch.inference_mode():
            b = self.model(channels, base_t)["b"][0, 0].float().cpu().numpy() * width
        b = b.astype(np.float32)
        b[~np.isfinite(disparity)] = np.nan
        return b
