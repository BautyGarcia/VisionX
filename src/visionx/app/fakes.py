"""Sustitutos para ejecutar la aplicación sin cámara ni profundidad real (T07).

**No son mediciones.** Sirven para probar el armado (colas, unión por frame, cancelación,
dashboard) sin la ELP ni un estimador de profundidad real. Cada salida lleva un model_id que
empieza con "fake-" para que ningún resultado de estos sustitutos pase por una medición. La
ubicación por objeto provisoria (con profundidad real) está en app/spatial.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import (
    DepthResult,
    DepthStatus,
    Detection,
    StereoFrame,
)
from visionx.shared.geometry import StereoGeometry


def synthetic_geometry(width: int = 640, height: int = 360) -> StereoGeometry:
    """Geometría parecida a la ELP escalada (fx ≈ 0,95·ancho, baseline 6 cm)."""
    return StereoGeometry(
        width=width,
        height=height,
        fx=0.95 * width,
        fy=0.95 * width,
        cx=width / 2,
        cy=height / 2,
        baseline_m=0.06,
    )


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    """Un rectángulo que se mueve de izquierda a derecha a una distancia fija."""

    label: str
    z_m: float
    size: tuple[float, float] = (0.18, 0.3)  # ancho, alto en fracción de la imagen
    period_s: float = 8.0  # tiempo de un cruce completo
    color: tuple[int, int, int] = (40, 160, 230)

    def box(self, t_s: float, width: int, height: int) -> tuple[int, int, int, int]:
        phase = (t_s / self.period_s) % 1.0
        cx = width * (0.15 + 0.7 * (0.5 - 0.5 * math.cos(2 * math.pi * phase)))
        w, h = self.size[0] * width, self.size[1] * height
        cy = height * 0.55
        return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


class SyntheticStereoSource:
    """FrameSource de pares rectificados sintéticos, con el objeto corrido en el ojo derecho
    según su disparidad. Con `fps` entrega a ese ritmo (como una cámara); con None, tan rápido
    como se lea. `n_frames` None = sin fin."""

    def __init__(
        self,
        objects: tuple[SyntheticObject, ...] = (SyntheticObject("cup", 1.2),),
        geometry: StereoGeometry | None = None,
        *,
        fps: float | None = 30.0,
        n_frames: int | None = None,
        session_id: str = "sintetica",
        clock: Clock | None = None,
    ) -> None:
        self.objects = objects
        self.geometry = geometry or synthetic_geometry()
        self.fps = fps
        self.n_frames = n_frames
        self.session_id = session_id
        self.clock = clock or MonotonicClock()
        self._next = 0
        self._t0: float | None = None
        self.closed = False

    def object_boxes(
        self, frame_id: int
    ) -> list[tuple[SyntheticObject, tuple[int, int, int, int]]]:
        t = frame_id / (self.fps or 30.0)
        g = self.geometry
        return [(o, o.box(t, g.width, g.height)) for o in self.objects]

    def read(self) -> StereoFrame | None:
        if self.closed or (self.n_frames is not None and self._next >= self.n_frames):
            return None
        if self.fps:
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            wait = self._t0 + self._next / self.fps - now
            if wait > 0:
                time.sleep(wait)
        g = self.geometry
        left = np.full((g.height, g.width, 3), 70, np.uint8)
        left[::24, :] = 90  # textura de fondo
        right = left.copy()
        for obj, (x1, y1, x2, y2) in self.object_boxes(self._next):
            d = g.disparity_at(obj.z_m) or 0.0
            cv2.rectangle(left, (x1, y1), (x2, y2), obj.color, -1)
            shift = int(round(d))
            cv2.rectangle(right, (x1 - shift, y1), (x2 - shift, y2), obj.color, -1)
        frame = StereoFrame(
            session_id=self.session_id,
            frame_id=self._next,
            left=left,
            right=right,
            received_ns=self.clock.now_ns(),
            rectified=True,
            calibration_id="fake-calibracion",
            geometry_id=g.geometry_id,
            source_id="sintetica",
        )
        self._next += 1
        return frame

    def close(self) -> None:
        self.closed = True


class ScriptedDetector:
    """Detector falso que "ve" exactamente los objetos de una SyntheticStereoSource."""

    def __init__(self, source: SyntheticStereoSource, delay_ms: float = 0.0) -> None:
        self.source = source
        self.delay_ms = delay_ms
        self.model_id = "fake-detector-sintetico"

    def predict(self, frame: StereoFrame) -> Detection:
        from visionx.detection.convert import build_detection

        t0 = time.perf_counter()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        w, h = frame.size
        objs = self.source.object_boxes(frame.frame_id)
        masks = np.zeros((len(objs), h, w), np.bool_)
        for i, (_, (x1, y1, x2, y2)) in enumerate(objs):
            masks[i, max(y1, 0) : y2, max(x1, 0) : x2] = True
        return build_detection(
            frame,
            model_id=self.model_id,
            boxes_xyxy=np.array([b for _, b in objs], np.float32).reshape(-1, 4),
            class_names=[o.label for o, _ in objs],
            scores=np.full(len(objs), 0.9, np.float32),
            inference_ms=(time.perf_counter() - t0) * 1000,
            masks=masks,
            track_ids=list(range(1, len(objs) + 1)),
        )

    def close(self) -> None:
        pass


class FakeDepthEstimator:
    """Profundidad falsa: fondo a `background_m` y, si se le da la fuente sintética, cada
    objeto a su distancia. `delay_ms` simula un estimador lento (sobrecarga)."""

    def __init__(
        self,
        geometry: StereoGeometry,
        *,
        source: SyntheticStereoSource | None = None,
        background_m: float = 3.0,
        delay_ms: float = 0.0,
    ) -> None:
        self.geometry = geometry
        self.source = source
        self.background_m = background_m
        self.delay_ms = delay_ms
        self.model_id = "fake-profundidad"

    def predict(self, frame: StereoFrame) -> DepthResult:
        t0 = time.perf_counter()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        g = self.geometry
        z = np.full((g.height, g.width), self.background_m, np.float32)
        if self.source is not None:
            for obj, (x1, y1, x2, y2) in self.source.object_boxes(frame.frame_id):
                z[max(y1, 0) : y2, max(x1, 0) : x2] = obj.z_m
        d, valid = g.depth_to_disparity(z)
        depth, _ = g.disparity_to_depth(d, valid)
        return DepthResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            model_id=self.model_id,
            geometry_id=g.geometry_id,
            disparity_px=np.nan_to_num(d, nan=0.0).astype(np.float32),
            valid_mask=valid,
            status=DepthStatus.VALID,
            inference_ms=(time.perf_counter() - t0) * 1000,
            depth_m=np.nan_to_num(depth, nan=1.0).astype(np.float32),
            doffs=g.doffs,
        )

    def close(self) -> None:
        pass
