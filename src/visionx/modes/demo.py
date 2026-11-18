"""Demo en video de buscar/distancia sobre una grabación (T09): imagen, guía, subtítulos y voz.

Usa la simulación de `simulate.py` (tiempo grabado, latencias emuladas) y dibuja cada frame:

    +--------------------------------------+-----------+
    | ojo izquierdo rectificado            | profund.  |
    |  gris: detecciones; amarillo: clase   | (unión    |
    |  pedida; verde + flecha: objetivo     |  medida)  |
    |                                      | estado    |
    +--------------------------------------+-----------+
    | orden actual y subtítulos de lo que suena            |
    +--------------------------------------------------+

El audio imita a `AudioOutput`: las frases de una misma prioridad suenan en cola, y cancelar
(o una orden nueva, que cancela la anterior) corta lo que sonaba y descarta lo pendiente de ese
pedido. Los subtítulos aparecen cuando la frase EMPIEZA a sonar, no cuando se pide. La voz es
Piper (extra `voz`); los sonidos, los de `voice.output`. Con ffmpeg se junta todo en un MP4
H.264 con audio; sin ffmpeg queda el video mudo y el WAV al lado.

Es material para mirar cómo trabaja el sistema, no una medición: las latencias son emuladas.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visionx.app.runtime import View
from visionx.modes.common import ModeSettings, candidates, direction
from visionx.modes.simulate import RecordingSink
from visionx.shared.geometry import StereoGeometry
from visionx.voice.output import SOUND_PREFIX, render_sound

W, H = 1280, 720
CAM_W, CAM_H = 960, 540
SUB_H = H - CAM_H
SIDE_W = W - CAM_W
GRAY, YELLOW, GREEN = (150, 150, 150), (0, 215, 255), (60, 220, 60)


# --- audio ---------------------------------------------------------------------------------


@dataclass(slots=True)
class Segment:
    request_id: int
    text: str
    start_s: float
    end_s: float
    samples: np.ndarray


@dataclass(slots=True)
class AudioSchedule:
    """Cuándo suena cada cosa, imitando la cola y la cancelación de AudioOutput."""

    synthesize: Callable[[str], np.ndarray] | None  # None: sin voz (duración estimada)
    sample_rate: int = 22050
    segments: list[Segment] = field(default_factory=list)
    _n_spoken: int = 0
    _n_cancel: int = 0

    def update(self, sink: RecordingSink) -> None:
        """Incorpora lo nuevo del sink: primero cancelaciones y después pedidos, en orden de
        tiempo (el controlador cancela lo viejo antes de decir lo nuevo en el mismo instante)."""
        events: list[tuple[float, int, object]] = []
        for c in sink.cancelled[self._n_cancel :]:
            events.append((c[0], 0, c))
        for s in sink.spoken[self._n_spoken :]:
            events.append((s.t_s, 1, s))
        self._n_cancel, self._n_spoken = len(sink.cancelled), len(sink.spoken)
        for t, kind, item in sorted(events, key=lambda e: (e[0], e[1])):
            if kind == 0:
                self._cancel(t, item[1])  # type: ignore[index]
            else:
                self._add(t, item.request_id, item.text)  # type: ignore[attr-defined]

    def _cancel(self, t: float, request_id: int | None) -> None:
        kept = []
        for seg in self.segments:
            hit = request_id is None or seg.request_id == request_id
            if hit and seg.start_s >= t:
                continue  # pendiente: se descarta
            if hit and seg.start_s < t < seg.end_s:
                seg.end_s = t  # sonando: se corta
            kept.append(seg)
        self.segments = kept

    def _add(self, t: float, request_id: int, text: str) -> None:
        samples = self._render(text)
        start = max([t] + [s.end_s for s in self.segments])
        dur = len(samples) / self.sample_rate
        self.segments.append(Segment(request_id, text, start, start + dur, samples))

    def _render(self, text: str) -> np.ndarray:
        if text.startswith(SOUND_PREFIX):
            return render_sound(text.removeprefix(SOUND_PREFIX), self.sample_rate)
        if self.synthesize is None:
            # Sin voz: silencio con la duración aproximada de la frase (para los subtítulos).
            return np.zeros(int(self.sample_rate * (0.4 + 0.065 * len(text))), np.float32)
        return np.asarray(self.synthesize(text), np.float32)

    def playing(self, t: float) -> list[Segment]:
        return [s for s in self.segments if s.start_s <= t < s.end_s]

    def mix(self, duration_s: float) -> np.ndarray:
        out = np.zeros(int(duration_s * self.sample_rate) + 1, np.float32)
        for s in self.segments:
            a = int(s.start_s * self.sample_rate)
            n = min(len(s.samples), int((s.end_s - s.start_s) * self.sample_rate), len(out) - a)
            if n > 0:
                out[a : a + n] += s.samples[:n]
        return np.clip(out, -1.0, 1.0)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())


# --- dibujo --------------------------------------------------------------------------------


class Painter:
    """Texto con tildes (Pillow + DejaVu de matplotlib); el resto con OpenCV."""

    def __init__(self) -> None:
        from matplotlib import font_manager
        from PIL import ImageFont

        path = font_manager.findfont("DejaVu Sans")
        self._font = {size: ImageFont.truetype(path, size) for size in (18, 22, 30)}

    def text(
        self, img: np.ndarray, lines: list[tuple[str, tuple[int, int], int, tuple]]
    ) -> np.ndarray:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, xy, size, bgr in lines:
            draw.text(xy, text, font=self._font[size], fill=(bgr[2], bgr[1], bgr[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# Escala del panel de profundidad: logarítmica entre estos metros. Con una escala lineal que
# terminaba en el borde del rango anunciado (3 m), todo lo de más de 3 m (en el clip de
# distancias, más de la mitad del centro: el pasillo a 3–6 m) salía de un mismo azul.
PANEL_RANGE_M = (0.4, 8.0)
PANEL_TICKS_M = (0.5, 1, 2, 3, 5, 8)
LEGEND_H = 34


def _depth_colors(z: np.ndarray) -> np.ndarray:
    """Metros -> BGR (rojo cerca, azul lejos), escala logarítmica en PANEL_RANGE_M."""
    near, far = PANEL_RANGE_M
    z = np.clip(np.nan_to_num(z, nan=far), near, far)
    norm = np.log(z / near) / np.log(far / near)
    return cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)


def depth_legend(width: int, range_m: tuple[float, float]) -> np.ndarray:
    """Barra de colores con marcas en metros; la marca del límite anunciado va en blanco."""
    legend = np.zeros((LEGEND_H, width, 3), np.uint8)
    near, far = PANEL_RANGE_M
    xs = np.linspace(0, 1, width)
    bar = _depth_colors(near * (far / near) ** xs).reshape(1, width, 3)
    legend[2:12] = np.repeat(bar, 10, axis=0)
    for tick in PANEL_TICKS_M:
        x = int(round((width - 1) * np.log(tick / near) / np.log(far / near)))
        color = (255, 255, 255) if tick == range_m[1] else (170, 170, 170)
        cv2.line(legend, (x, 12), (x, 16), color, 1)
        label = f"{tick:g}".replace(".", ",") + (" m" if tick == PANEL_TICKS_M[-1] else "")
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        x0 = min(max(0, x - size // 2), width - size)
        cv2.putText(legend, label, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return legend


def depth_panel(view: View, range_m: tuple[float, float]) -> np.ndarray:
    """Profundidad de la unión medida (negro: sin dato) y su escala debajo."""
    map_h = SIDE_W * 9 // 16
    panel = np.zeros((map_h + LEGEND_H, SIDE_W, 3), np.uint8)
    panel[map_h:] = depth_legend(SIDE_W, range_m)
    joint = view.measured
    if joint is None or joint.depth is None or joint.depth.depth_m is None:
        return panel
    color = _depth_colors(joint.depth.depth_m)
    color[~joint.depth.valid_mask] = 0
    panel[:map_h] = cv2.resize(color, (SIDE_W, map_h), interpolation=cv2.INTER_AREA)
    return panel


@dataclass(slots=True)
class DemoState:
    """Lo que la demo necesita saber del controlador para dibujar."""

    order: str = "-"
    mode: str = "-"
    state: str = "-"
    label: str | None = None
    target_angle: tuple[float, float] | None = None  # (h, v) del objetivo guiado


def render_frame(
    frame,
    view: View,
    st: DemoState,
    geometry: StereoGeometry,
    spoken: Callable[[str], str],
    t_s: float,
    info: list[str],
    settings: ModeSettings,
    painter: Painter,
) -> np.ndarray:
    """Imagen base (sin subtítulos) de un frame."""
    range_m = settings.range_m
    canvas = np.zeros((H, W, 3), np.uint8)
    cam = cv2.resize(frame.left, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    sx, sy = CAM_W / frame.left.shape[1], CAM_H / frame.left.shape[0]
    labels: list[tuple[str, tuple[int, int], int, tuple]] = []
    seen = view.seen
    target_xy = None
    if seen is not None and seen.frame.frame_id <= frame.frame_id:
        det = seen.detection
        cands = candidates(seen, st.label, geometry) if st.label else []
        chosen = None
        if st.target_angle is not None and cands:
            best = min(cands, key=lambda c: abs(c.angle_h_deg - st.target_angle[0]))
            chosen = best.index if abs(best.angle_h_deg - st.target_angle[0]) < 3 else None
        for i, cls in enumerate(det.classes):
            x1, y1, x2, y2 = (
                int(v * s) for v, s in zip(det.boxes[i], (sx, sy, sx, sy), strict=True)
            )
            color, thick = (
                (GREEN, 4) if i == chosen else ((YELLOW, 2) if cls == st.label else (GRAY, 1))
            )
            cv2.rectangle(cam, (x1, y1), (x2, y2), color, thick)
            labels.append((f"{spoken(cls)} {float(det.scores[i]):.2f}", (x1 + 3, max(0, y1 - 22)),
                           18, color))  # fmt: skip
            if i == chosen:
                target_xy = ((x1 + x2) // 2, (y1 + y2) // 2)
    if target_xy is not None:
        cv2.arrowedLine(cam, (CAM_W // 2, CAM_H // 2), target_xy, GREEN, 3, tipLength=0.05)
    cv2.drawMarker(cam, (CAM_W // 2, CAM_H // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    # Las etiquetas se dibujan sobre la imagen de la cámara: así no invaden el panel lateral.
    canvas[:CAM_H, :CAM_W] = painter.text(cam, labels)
    dp = depth_panel(view, range_m)
    canvas[: dp.shape[0], CAM_W:] = dp
    side = [f"t = {t_s:5.1f} s", f"modo: {st.mode}", f"estado: {st.state}"]
    if st.target_angle is not None:
        h, v = st.target_angle
        side.append(f"objetivo: {direction(h, settings, v)}")
        side.append(f"  h {h:+.0f}°  v {v:+.0f}°")
    side += info
    lines = [
        (s, (CAM_W + 10, dp.shape[0] + 10 + 26 * k), 18, (230, 230, 230))
        for k, s in enumerate(side)
    ]
    lines.append((f"Orden: {st.order}", (16, CAM_H + 12), 22, YELLOW))
    return painter.text(canvas, lines)


def add_subtitles(
    base: np.ndarray, schedule: AudioSchedule, t: float, painter: Painter
) -> np.ndarray:
    playing = [s for s in schedule.playing(t) if not s.text.startswith(SOUND_PREFIX)]
    beeping = any(s.text.startswith(SOUND_PREFIX) for s in schedule.playing(t))
    recent = [s for s in schedule.segments if s.end_s <= t and t - s.end_s < 2.0
              and not s.text.startswith(SOUND_PREFIX)]  # fmt: skip
    lines = []
    if playing:
        lines.append((playing[-1].text, (16, CAM_H + 60), 30, (255, 255, 255)))
    elif recent:  # lo último dicho queda un rato en gris
        lines.append((recent[-1].text, (16, CAM_H + 60), 30, (140, 140, 140)))
    img = painter.text(base.copy(), lines) if lines else base.copy()
    if beeping:
        cv2.circle(img, (W - 40, CAM_H + 40), 16, GREEN, -1)
    return img


def mux(video: Path, audio: Path, out: Path) -> bool:
    """Junta video y audio con ffmpeg (H.264 + AAC). False si no hay ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return True
