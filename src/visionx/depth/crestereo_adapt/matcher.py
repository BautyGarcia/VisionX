"""Correr CREStereo sobre un par rectificado y devolver la disparidad.

La red recibe dos imágenes float en 0-255 con forma (1, 3, alto, ancho) y devuelve un "flujo"
de dos canales a la resolución de entrada. El canal 0 es la **disparidad**, positiva, en
píxeles de la imagen que entró. El canal 1 (vertical) se entrenó con objetivo cero y no se usa.

Tres decisiones de este archivo:

- **Relleno.** La red reduce la imagen 16 veces: alto y ancho tienen que ser múltiplos de 16.
  Se rellena abajo y a la derecha repitiendo el borde, y después se recorta. Así no se deforma
  la imagen ni cambia la escala de la disparidad.
- **Dos pasadas** (`passes=2`), como el script de prueba del port: primero a media resolución
  y después a resolución completa, partiendo de ese resultado. Es más preciso y más lento.
- **Arranque en caliente** (`warm_start`): en video, la disparidad de un cuadro se parece a la
  del anterior. Si se le pasa a la red como punto de partida, saltea sus etapas gruesas y corre
  solo la fina. Cada `warm_reset_every` cuadros se arranca en frío, para no arrastrar errores.

El modelo se recibe ya cargado (`port.load_model`), así los tests usan uno falso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visionx.shared.errors import InvalidInput


@dataclass(frozen=True)
class CreStereoParams:
    """Cómo se corre la red. No cambia los pesos."""

    iters: int = 4  # iteraciones de refinamiento de la etapa fina
    passes: int = 1  # 1: una pasada | 2: media resolución y después completa
    warm_start: bool = False  # partir de la disparidad del cuadro anterior
    warm_reset_every: int = 30  # cada cuántos cuadros se arranca en frío (0: nunca)
    bgr_input: bool = True  # el port y el original leen con OpenCV: BGR

    def __post_init__(self) -> None:
        if self.iters < 1:
            raise InvalidInput("iters >= 1")
        if self.passes not in (1, 2):
            raise InvalidInput("passes es 1 o 2")
        if self.warm_reset_every < 0:
            raise InvalidInput("warm_reset_every >= 0")

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> CreStereoParams:
        return cls(
            iters=int(section.get("iters", 4)),
            passes=int(section.get("passes", 1)),
            warm_start=bool(section.get("warm_start", False)),
            warm_reset_every=int(section.get("warm_reset_every", 30)),
        )

    @property
    def multiple(self) -> int:
        """Múltiplo al que se rellena: 16, o 32 si además se corre a media resolución."""
        return 16 * self.passes

    @property
    def tag(self) -> str:
        """Texto corto y estable para identificar esta forma de correr la red."""
        warm = f"w{self.warm_reset_every}" if self.warm_start else ""
        return f"i{self.iters}p{self.passes}{warm}"


def padded_size(height: int, width: int, multiple: int) -> tuple[int, int]:
    """Alto y ancho llevados hacia arriba al múltiplo pedido."""
    return -(-height // multiple) * multiple, -(-width // multiple) * multiple


class CreStereoMatcher:
    """Par rectificado BGR uint8 -> disparidad float32 en píxeles de ese par."""

    def __init__(self, model: Any, device: str, params: CreStereoParams) -> None:
        self.model = model
        self.device = device
        self.params = params
        self._previous: Any = None  # flujo del cuadro anterior, en el dispositivo
        self._since_reset = 0
        self.last_stats: dict[str, float] = {}
        # Subproductos de la última corrida, gratis, que sirven para estimar incertidumbre:
        #   "vertical"  el canal vertical del flujo (la red lo entrenó hacia cero)
        #   "half"      la disparidad de la pasada a media resolución, llevada al tamaño final
        #               (None si esa corrida no tuvo dos pasadas)
        self.last_extras: dict[str, np.ndarray | None] = {}

    def reset(self) -> None:
        """Olvida el cuadro anterior: el próximo par arranca en frío."""
        self._previous = None
        self._since_reset = 0

    def _to_tensor(self, image: np.ndarray, height: int, width: int) -> Any:
        import torch
        import torch.nn.functional as F

        if not self.params.bgr_input:
            image = image[:, :, ::-1]
        # torch.tensor copia: los frames del contrato son de solo lectura.
        tensor = torch.tensor(np.ascontiguousarray(image)).to(self.device)
        tensor = tensor.permute(2, 0, 1)[None].float()  # (1, 3, alto, ancho), 0-255
        pad_h, pad_w = height - image.shape[0], width - image.shape[1]
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Disparidad del ojo izquierdo, del mismo tamaño que el par."""
        import torch
        import torch.nn.functional as F

        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise InvalidInput("left y right deben ser H x W x 3 del mismo tamaño")
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise InvalidInput("left y right deben ser uint8")
        p = self.params
        h, w = left.shape[:2]
        ph, pw = padded_size(h, w, p.multiple)
        left_t, right_t = self._to_tensor(left, ph, pw), self._to_tensor(right, ph, pw)

        previous = self._previous
        warm = (
            p.warm_start
            and previous is not None
            and tuple(previous.shape[-2:]) == (ph, pw)
            and (p.warm_reset_every == 0 or self._since_reset < p.warm_reset_every)
        )
        with torch.inference_mode():
            if warm:
                flow = self.model(left_t, right_t, iters=p.iters, flow_init=previous)
                self._since_reset += 1
            else:
                init = None
                if p.passes == 2:
                    half = (ph // 2, pw // 2)
                    left_h = F.interpolate(left_t, size=half, mode="bilinear", align_corners=True)
                    right_h = F.interpolate(right_t, size=half, mode="bilinear", align_corners=True)
                    init = self.model(left_h, right_h, iters=p.iters, flow_init=None)
                flow = self.model(left_t, right_t, iters=p.iters, flow_init=init)
                self._since_reset = 0
            self._previous = flow if p.warm_start else None
            # Pasar a NumPy obliga a la GPU a terminar: el tiempo medido afuera es real.
            disparity = flow[0, 0, :h, :w].float().cpu().numpy()
            vertical = flow[0, 1, :h, :w].float().cpu().numpy()
            half = None
            if not warm and init is not None:
                # La pasada chica mide en sus propios píxeles: al duplicar el tamaño, la
                # disparidad también se duplica.
                up = F.interpolate(init[:, :1], size=(ph, pw), mode="bilinear", align_corners=True)
                half = (2.0 * up)[0, 0, :h, :w].float().cpu().numpy()
        self.last_extras = {"vertical": vertical, "half": half}
        self.last_stats = {"caliente": float(warm), "alto_red": float(ph), "ancho_red": float(pw)}
        return np.ascontiguousarray(disparity, dtype=np.float32)
