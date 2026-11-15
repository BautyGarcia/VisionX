"""Detector falso y determinista (T06, fake público de T01).

Sirve para desarrollar y probar lo que consume detecciones (T07, T09, T11) sin pesos ni torch.
Cumple la interfaz Detector igual que el real. Sus objetos son un guion fijo: **no** es una
predicción y no debe usarse como evidencia de detección.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from visionx.detection.convert import build_detection
from visionx.shared.contracts import Detection, StereoFrame
from visionx.shared.errors import EstimatorError


@dataclass(frozen=True, slots=True)
class FakeObject:
    """Un objeto del guion. La caja está en fracciones de la imagen (0 a 1) para que el mismo
    guion sirva a cualquier resolución; la máscara es la elipse inscripta en la caja."""

    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 en fracción del ancho/alto
    score: float = 0.9
    track_id: int | None = None


Script = Mapping[int, Iterable[FakeObject]] | Callable[[int], Iterable[FakeObject]]


class FakeDetector:
    """Devuelve, para cada frame_id, los objetos que indica el guion.

    script: diccionario frame_id -> objetos (los frames ausentes no tienen objetos) o una función
    de frame_id. fail_on: frames en los que simula una caída del modelo con EstimatorError.
    """

    def __init__(
        self,
        script: Script | None = None,
        *,
        model_id: str = "fake-detector",
        fail_on: Iterable[int] = (),
        inference_ms: float = 1.0,
        with_masks: bool = True,
    ) -> None:
        self.model_id = model_id
        self._script = script or {}
        self._fail_on = frozenset(fail_on)
        self._inference_ms = inference_ms
        self._with_masks = with_masks
        self.calls = 0
        self.closed = False

    def _objects(self, frame_id: int) -> list[FakeObject]:
        if callable(self._script):
            return list(self._script(frame_id))
        return list(self._script.get(frame_id, ()))

    def predict(self, frame: StereoFrame) -> Detection:
        if self.closed:
            raise EstimatorError("FakeDetector cerrado")
        self.calls += 1
        if frame.frame_id in self._fail_on:
            raise EstimatorError(f"falla simulada en el frame {frame.frame_id}")
        objects = self._objects(frame.frame_id)
        w, h = frame.size
        scale = np.array([w, h, w, h], np.float32)
        boxes = np.array([o.box for o in objects], np.float32).reshape(-1, 4) * scale
        masks = None
        if self._with_masks:
            masks = np.zeros((len(boxes), h, w), np.bool_)
            for i, box in enumerate(boxes):
                masks[i] = _ellipse(box, w, h)
        tracks = tuple(o.track_id for o in objects)
        return build_detection(
            frame,
            model_id=self.model_id,
            boxes_xyxy=boxes,
            class_names=tuple(o.label for o in objects),
            scores=np.array([o.score for o in objects], np.float32),
            inference_ms=self._inference_ms,
            masks=masks,
            track_ids=tracks if any(t is not None for t in tracks) else None,
        )

    def close(self) -> None:
        self.closed = True


def _ellipse(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Máscara bool H x W con la elipse inscripta en la caja: más parecida a un objeto real
    que el rectángulo, así una prueba no confunde "dentro de la caja" con "dentro del objeto"."""
    x1, y1, x2, y2 = (float(c) for c in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = max((x2 - x1) / 2, 0.5), max((y2 - y1) / 2, 0.5)
    ys, xs = np.ogrid[:h, :w]
    # Centros de píxel en +0,5: un píxel pertenece si su centro cae dentro de la elipse.
    return ((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / ry) ** 2 <= 1.0
