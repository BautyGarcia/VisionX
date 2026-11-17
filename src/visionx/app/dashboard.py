"""Dashboard de diagnóstico (T07). Para nosotros y la demo; el usuario no necesita mirarlo.

Arma una imagen con: el ojo izquierdo con detecciones, el mapa de profundidad del **mismo
frame** con una escala de color fija (la misma para todos los métodos: una escala que se
reajusta en cada imagen parece precisa aunque no lo sea), una vista de arriba de la nube de
puntos de ese frame (sin acumular: no es un mapa) y el estado de cada módulo con edad,
latencia, FPS y unidades.

Solo dibuja resultados ya calculados: nunca corre un modelo para mostrar algo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from visionx.app.runtime import Joint, ModuleStatus
from visionx.detection.draw import draw_detection
from visionx.shared.geometry import StereoGeometry

PANEL_W = 480  # ancho de cada panel de imagen, px
TEXT = (235, 235, 235)
WARN = (60, 170, 255)
BAD = (70, 70, 230)


@dataclass(frozen=True, slots=True)
class DashboardState:
    joint: Joint | None
    statuses: tuple[ModuleStatus, ...]
    now_ns: int
    range_m: tuple[float, float] = (0.5, 3.0)
    dropped: dict[str, int] | None = None
    queues: dict[str, int] | None = None
    command: str = "-"
    audio: str = "-"
    max_age_ms: float = 500.0  # más viejo que esto se marca en naranja
    geometry: StereoGeometry | None = None  # la de la profundidad; sin ella no hay vista de arriba
    measurement: dict[str, Any] | None = None


def depth_colormap(
    depth_m: np.ndarray, valid: np.ndarray, range_m: tuple[float, float]
) -> np.ndarray:
    """Profundidad en metros a color con escala fija `range_m`; sin dato = negro."""
    near, far = range_m
    norm = np.clip((np.nan_to_num(depth_m, nan=far) - near) / (far - near), 0, 1)
    # Cerca = cálido: invertir para que TURBO ponga rojo en lo cercano.
    colored = cv2.applyColorMap(((1 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def top_view(
    depth_m: np.ndarray,
    valid: np.ndarray,
    fx: float,
    cx: float,
    range_m: tuple[float, float],
    size: tuple[int, int] = (PANEL_W, 270),
    stride: int = 4,
) -> np.ndarray:
    """Nube de puntos del frame vista desde arriba: X lateral (m) contra Z (m).

    X = (u - cx) · Z / fx. Se submuestrea con `stride` porque es diagnóstico, no medición.
    """
    w, h = size
    view = np.zeros((h, w, 3), np.uint8)
    far = range_m[1] * 1.2
    for z_mark in np.arange(0.5, far, 0.5):  # líneas cada 50 cm
        y = int(h - 1 - z_mark / far * (h - 10))
        cv2.line(view, (0, y), (w, y), (50, 50, 50), 1)
        cv2.putText(view, f"{z_mark:.1f} m", (4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT, 1)
    z = depth_m[::stride, ::stride]
    ok = valid[::stride, ::stride] & np.isfinite(z) & (z > 0) & (z < far)
    if ok.any():
        rows, cols = np.nonzero(ok)
        zz = z[rows, cols]
        xx = (cols * stride - cx) * zz / fx
        px = (w / 2 + xx / far * (w / 2)).astype(int)
        py = (h - 1 - zz / far * (h - 10)).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        view[py[inside], px[inside]] = (200, 200, 120)
    cv2.circle(view, (w // 2, h - 3), 4, (0, 255, 0), -1)  # la cámara
    return view


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


def _status_lines(state: DashboardState) -> list[tuple[str, tuple[int, int, int]]]:
    lines = []

    def num(value: float | None, fmt: str) -> str:
        return format(value, fmt) if value is not None else "-".rjust(len(format(0.0, fmt)))

    for s in state.statuses:
        age = s.age_ms(state.now_ns)
        color = TEXT
        if s.errors:
            color = BAD
        elif age is not None and age > state.max_age_ms:
            color = WARN
        frame = str(s.last_key.frame_id) if s.last_key else "-"
        lines.append(
            (
                f"{s.name:<12} {s.model_id[:28]:<28} frame {frame:>6}"
                f"  edad {num(age, '7.0f')} ms  lat {num(s.last_latency_ms, '6.0f')} ms"
                f"  {num(s.fps(), '5.1f')} fps  err {s.errors}",
                color,
            )
        )
    if state.dropped is not None:
        lines.append((f"salteados (modelo ocupado): {state.dropped}", TEXT))
    if state.queues is not None:
        lines.append((f"colas: {state.queues}", TEXT))
    lines.append((f"orden: {state.command}", TEXT))
    lines.append((f"audio: {state.audio}", TEXT))
    if state.measurement:
        m = state.measurement
        lines.append(
            (f"Metros: {m.get('model_id')} | {m.get('latencia_ms')} ms | {m.get('motivo')}", TEXT)
        )
    joint = state.joint
    if joint is not None:
        for obj in joint.objects[:4]:
            z = f"{obj.z_m:.2f} m" if obj.z_m is not None else "sin Z"
            lines.append(
                (
                    f"  {obj.label} #{obj.track_id}: {obj.angle_h_deg:+.0f} grados, {z}, "
                    f"validez {obj.valid_fraction:.0%}, {obj.policy.value}",
                    TEXT,
                )
            )
    return lines


def render(state: DashboardState) -> np.ndarray:
    """Imagen BGR del dashboard."""
    joint = state.joint
    if joint is None:
        left = np.zeros((270, PANEL_W, 3), np.uint8)
        cv2.putText(
            left, "esperando resultados...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2
        )
        depth_panel = np.zeros_like(left)
        cloud = np.zeros_like(left)
    else:
        left = _fit(draw_detection(joint.frame.left, joint.detection), PANEL_W)
        depth = joint.depth
        if depth is not None and depth.depth_m is not None:
            depth_panel = _fit(
                depth_colormap(depth.depth_m, depth.valid_mask, state.range_m), PANEL_W
            )
            label = (
                f"{depth.model_id}  {state.range_m[0]}-{state.range_m[1]} m  {depth.status.value}"
            )
            cv2.putText(depth_panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1)
            g = state.geometry
            if g is not None and g.size == depth.size:
                cloud = top_view(depth.depth_m, depth.valid_mask, g.fx, g.cx, state.range_m)
            else:  # sin la geometría de ese mapa, X lateral no se puede calcular
                cloud = np.zeros((270, PANEL_W, 3), np.uint8)
                cv2.putText(
                    cloud, "sin geometria", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2
                )
        else:
            depth_panel = np.zeros_like(left)
            cv2.putText(
                depth_panel, "sin profundidad", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2
            )
            cloud = np.zeros((270, PANEL_W, 3), np.uint8)
    h = max(left.shape[0], depth_panel.shape[0], cloud.shape[0])
    row = np.hstack([_pad(left, h), _pad(depth_panel, h), _pad(cloud, h)])
    lines = _status_lines(state)
    text = np.zeros((22 * len(lines) + 12, row.shape[1], 3), np.uint8)
    for i, (line, color) in enumerate(lines):
        cv2.putText(text, line, (10, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return np.vstack([row, text])


def _pad(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    out = np.zeros((height, image.shape[1], 3), np.uint8)
    out[: image.shape[0]] = image
    return out
