"""Dibujo de detecciones para revisar a ojo (diagnóstico, no para el usuario final)."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from visionx.shared.contracts import Detection

# Paleta fija en BGR: el mismo track conserva su color entre frames y se ve si cambia de ID.
_PALETTE = np.array(
    [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245),
        (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (190, 190, 250),
    ],
    dtype=np.uint8,
)  # fmt: skip


def color_for(index: int, track_id: int | None) -> tuple[int, int, int]:
    key = track_id if track_id is not None else index
    return tuple(int(c) for c in _PALETTE[key % len(_PALETTE)])  # type: ignore[return-value]


def draw_detection(
    image: np.ndarray,
    det: Detection,
    name: Callable[[str], str] = str,
    alpha: float = 0.45,
) -> np.ndarray:
    """Copia de `image` con máscaras semitransparentes, cajas y "nombre score #track"."""
    if image.shape[1::-1] != det.image_size:
        raise ValueError(f"imagen {image.shape[1::-1]} y detección {det.image_size} no coinciden")
    view = image.copy()
    for i in range(len(det)):
        track = det.track_ids[i] if det.track_ids is not None else None
        color = color_for(i, track)
        if det.masks is not None:
            m = det.masks[i]
            view[m] = (view[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = (int(round(float(c))) for c in det.boxes[i])
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = f"{name(det.classes[i])} {det.scores[i]:.2f}"
        if track is not None:
            label += f" #{track}"
        cv2.putText(view, label, (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(
        view,
        f"{det.model_id}  frame {det.frame_id}  {det.inference_ms:.0f} ms",
        (10, view.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return view
