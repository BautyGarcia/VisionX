"""Fuente de pares desde una Intel RealSense (D435, D435i, D455) con `librealsense`.

La RealSense no entrega una imagen side by side como la ELP: tiene dos cámaras infrarrojas que
llegan como **dos transmisiones** (`infrared` 1 y 2). Esas imágenes ya vienen **rectificadas
con la calibración de fábrica**, así que no hace falta ChArUco ni alineación: la focal, el
centro y la separación se leen de la cámara y el perfil queda «preciso».

- Ven en escala de grises (infrarrojo); se pasan a BGR de 3 canales para el resto de la app.
- El proyector de puntos (`emisor`) agrega textura en paredes lisas, pero los puntos también
  los ven el detector y Claude. Por defecto apagado; probar las dos cosas en la facultad.
- La cámara de color es otro sensor, en otro lugar: no se usa para el par estéreo.

Instalar la librería (no es dependencia fija del proyecto):

    Windows / Linux:      uv pip install pyrealsense2
    macOS (Apple, 15+):   uv pip install pyrealsense2-macosx   (puede pedir sudo para el USB)
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from visionx.camera.usb import new_session_id
from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import StereoFrame
from visionx.shared.errors import SourceError
from visionx.shared.geometry import StereoGeometry

INSTALAR = (
    "falta librealsense: en Windows o Linux `uv pip install pyrealsense2`; "
    "en una Mac con chip Apple `uv pip install pyrealsense2-macosx`"
)


def importar_rs() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SourceError(INSTALAR) from exc
    return rs


class RealSenseSource:
    """FrameSource en vivo: los dos ojos infrarrojos, ya rectificados por la cámara."""

    def __init__(
        self,
        ancho: int = 1280,
        alto: int = 720,
        fps: int = 30,
        emisor: bool = False,
        *,
        session_id: str | None = None,
        clock: Clock | None = None,
        rs_module: Any = None,
        timeout_ms: int = 5000,
        calentamiento: int = 5,
    ) -> None:
        rs = rs_module or importar_rs()
        self._rs = rs
        self._clock = clock or MonotonicClock()
        self._timeout_ms = timeout_ms
        self._next_id = 0
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        for ojo in (1, 2):
            cfg.enable_stream(rs.stream.infrared, ojo, ancho, alto, rs.format.y8, fps)
        try:
            perfil = self.pipeline.start(cfg)
        except RuntimeError as exc:
            msg = f"no se pudo abrir la RealSense ({ancho}x{alto} a {fps}): {exc}"
            raise SourceError(msg) from exc
        dev = perfil.get_device()
        self.nombre = str(dev.get_info(rs.camera_info.name))
        self.serie = str(dev.get_info(rs.camera_info.serial_number))
        sensor = dev.first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if emisor else 0.0)
        s1 = perfil.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        s2 = perfil.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intr = s1.get_intrinsics()
        ext = s2.get_extrinsics_to(s1)
        self.geometry = StereoGeometry(
            width=int(intr.width), height=int(intr.height), fx=float(intr.fx),
            fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy),
            baseline_m=abs(float(ext.translation[0])),
        )  # fmt: skip
        self.calibration_id = f"realsense-{self.serie}"
        self.session_id = session_id or new_session_id("realsense")
        for _ in range(calentamiento):
            self._leer_crudo()

    def _leer_crudo(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            frames = self.pipeline.wait_for_frames(self._timeout_ms)
        except RuntimeError as exc:
            raise SourceError(f"la RealSense dejó de entregar cuadros: {exc}") from exc
        izq = np.asanyarray(frames.get_infrared_frame(1).get_data())
        der = np.asanyarray(frames.get_infrared_frame(2).get_data())
        return cv2.cvtColor(izq, cv2.COLOR_GRAY2BGR), cv2.cvtColor(der, cv2.COLOR_GRAY2BGR)

    def read(self) -> StereoFrame:
        izq, der = self._leer_crudo()
        frame = StereoFrame(
            session_id=self.session_id, frame_id=self._next_id, left=izq, right=der,
            received_ns=self._clock.now_ns(), rectified=True,
            calibration_id=self.calibration_id, geometry_id=self.geometry.geometry_id,
            source_id="realsense",
        )  # fmt: skip
        self._next_id += 1
        return frame

    def perfil(self):
        """El perfil de esta unidad, con la calibración de fábrica (nivel «precisa»)."""
        from visionx.camera.perfiles import PerfilCamara

        g = self.geometry
        return PerfilCamara(
            id=f"realsense-{self.serie}".lower(), nombre=self.nombre, fabricante="Intel",
            formato="realsense", usb=(self.nombre,), resolucion=(g.width, g.height),
            campo_horizontal_deg=math.degrees(2 * math.atan(g.width / (2 * g.fx))),
            separacion_m=g.baseline_m, nivel="precisa",
            fuente=f"calibración de fábrica (librealsense), serie {self.serie}",
            centro_rel=(g.cx / g.width - 0.5, g.cy / g.height - 0.5),
        )  # fmt: skip

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
