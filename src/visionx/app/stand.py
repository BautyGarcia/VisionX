"""Pantalla del stand: VisionX en vivo, para mostrarle al público qué ve, mide y dice.

A diferencia del dashboard de diagnóstico (`dashboard.py`), esta vista cuenta la interacción:

- **Cámara** grande con cada objeto rotulado con su distancia y el color de su certeza
  (verde precisa, ámbar aproximada, rojo no segura) y la flecha al objetivo de buscar.
- **Profundidad** (CREStereo) e **incertidumbre** (U-Net: cuánto puede errar cada píxel, en %
  de la distancia, con los mismos umbrales que deciden qué se dice).
- **Radar** desde arriba: la escena en planta, los objetos con su distancia y su margen.
- **Conversación** en subtítulos: lo que dijo la persona y lo que respondió VisionX.
- **Tiempos** de la última orden (voz a texto, Claude, primera palabra), FPS de cada módulo y
  el **cuerpo** (estatura, altura e inclinación de la cámara, si se ve el piso).

Solo dibuja resultados ya calculados (como el dashboard): nunca corre un modelo.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.app.dashboard import depth_colormap
from visionx.modes.common import ModeSettings, measure
from visionx.shared.contracts import UncertaintyKind

W, H = 1920, 1080
BG = (24, 20, 18)  # BGR
PANEL = (40, 34, 30)
INK = (240, 238, 235)
MUTED = (160, 150, 145)
ACCENT = (255, 190, 60)  # celeste VisionX (BGR)
YOU = (255, 210, 120)
GREEN = (90, 210, 90)
AMBER = (40, 180, 255)
RED = (70, 70, 235)
LEVEL_COLOR = {"precisa": GREEN, "aproximada": AMBER, "no-segura": RED, "sin-medida": MUTED}
ACKS = {"Leo.", "Describo.", "Mido la distancia."}

# --- texto con tildes ------------------------------------------------------------------------

_FONTS = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}  # fmt: skip


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _FONTS[bold]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Texts:
    """Junta los textos de un cuadro y los dibuja de una vez con Pillow (tildes, ñ, ¿)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, tuple[int, int], int, tuple[int, int, int], bool]] = []

    def add(self, text: str, xy: tuple[int, int], size: int = 22,
            color: tuple[int, int, int] = INK, bold: bool = False) -> None:  # fmt: skip
        self.items.append((text, xy, size, color, bold))

    def draw(self, canvas: np.ndarray) -> np.ndarray:
        if not self.items:
            return canvas
        if _font(20) is None:  # sin Pillow: OpenCV, sin tildes
            import unicodedata

            for text, (x, y), size, color, bold in self.items:
                plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
                cv2.putText(canvas, plain, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 34,
                            color, 2 if bold else 1, cv2.LINE_AA)  # fmt: skip
            return canvas
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas[:, :, ::-1])
        d = ImageDraw.Draw(img)
        for text, xy, size, color, bold in self.items:
            d.text(xy, text, font=_font(size, bold), fill=color[::-1])
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def text_width(text: str, size: int, bold: bool = False) -> int:
    f = _font(size, bold)
    if f is None:
        return int(len(text) * size * 0.55)
    return int(f.getlength(text))


def wrap(text: str, size: int, width: int, bold: bool = False) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_width(trial, size, bold) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# --- lo que pasó: conversación y tiempos -----------------------------------------------------


@dataclass
class Turn:
    who: str  # "vos" | "visionx"
    text: str
    t_ns: int
    request_id: int | None = None


def is_ack(text: str) -> bool:
    """La confirmación de la orden («Busco taza.», «Leo.»): no es la respuesta."""
    return text in ACKS or (text.startswith("Busco ") and text.endswith(".") and len(text) < 40)


@dataclass
class Timing:
    stt_ms: float | None = None
    claude_ms: float | None = None
    first_word_ms: float | None = None
    order: str = ""


class LiveLog:
    """Escucha los eventos de la app (los mismos de la traza) y arma conversación y tiempos."""

    def __init__(self, max_turns: int = 6) -> None:
        self._lock = threading.Lock()
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.timing = Timing()
        self._released_ns: int | None = None
        self._request: int | None = None
        self._stt_ms: float | None = None

    def feed(self, kind: str, data: dict) -> None:
        now = time.monotonic_ns()
        with self._lock:
            if kind == "boton-soltado":
                self._released_ns, self._stt_ms = now, None
            elif kind == "stt":
                self._stt_ms = data.get("ms")
                if data.get("texto"):
                    self.turns.append(Turn("vos", str(data["texto"]), now))
            elif kind == "orden" and data.get("request_id") is not None:
                if self._released_ns is None:  # orden por teclado o guion
                    self.turns.append(Turn("vos", str(data.get("texto", "")), now))
                self._request = data["request_id"]
                self.timing = Timing(stt_ms=self._stt_ms, order=str(data.get("texto", "")))
                self._start_ns = self._released_ns or now
                self._released_ns = None
            elif kind == "asistente-respuesta" and data.get("request_id") == self._request:
                if data.get("primer_texto_s") is not None:
                    self.timing.claude_ms = data["primer_texto_s"] * 1000
            elif kind == "audio":
                text = str(data.get("texto", ""))
                started = data.get("started_ns")
                rid = data.get("request_id")
                if not started or text.startswith("sonido:") or is_ack(text):
                    return
                if rid == self._request and self.timing.first_word_ms is None:
                    self.timing.first_word_ms = (
                        started - getattr(self, "_start_ns", started)
                    ) / 1e6
                last = self.turns[-1] if self.turns else None
                if last is not None and last.who == "visionx" and last.request_id == rid:
                    if not last.text.endswith(text):  # la guía repite: no se agrega
                        last.text = f"{last.text} {text}"  # tramos de una misma respuesta
                    last.t_ns = now
                else:
                    self.turns.append(Turn("visionx", text, now, rid))

    def snapshot(self) -> tuple[list[Turn], Timing]:
        with self._lock:
            return [Turn(t.who, t.text, t.t_ns, t.request_id) for t in self.turns], Timing(
                **vars(self.timing)
            )


# --- estado de un cuadro ---------------------------------------------------------------------


@dataclass
class StandState:
    joint: Any
    statuses: tuple
    geometry: Any  # de la profundidad (radar)
    frame_geometry: Any  # de la imagen (flecha y rótulos)
    range_m: tuple[float, float]
    settings: ModeSettings = field(default_factory=ModeSettings)
    mode: str = "Esperando una orden"
    recording: bool = False
    turns: list[Turn] = field(default_factory=list)
    timing: Timing = field(default_factory=Timing)
    body: Any = None
    target_angle: tuple[float, float] | None = None
    target_label: str | None = None
    spoken: Any = None  # nombre en castellano de una clase


def short_m(z: float, s: ModeSettings, approx: bool = False) -> str:
    step = s.approx_round_m if approx else s.round_m
    r = round(round(z / step) * step, 2)
    return f"{r:g} m".replace(".", ",")


def level_of(obj: Any, s: ModeSettings) -> tuple[str, str]:
    """(nivel, rótulo corto de distancia) de un objeto ubicado, con la regla de la voz."""
    m = measure(obj, s)
    if m.ok and m.z_m is not None:
        approx = m.approximate or m.z_m > s.approx_from_m
        level = "aproximada" if m.approximate else "precisa"
        return level, ("~" if approx else "") + short_m(m.z_m, s, approx)
    if m.reason == "no-segura":
        return "no-segura", "¿?"
    if m.reason in ("muy-cerca", "muy-lejos") and m.z_m is not None:
        near, far = s.range_m
        return (
            "precisa",
            f"<{short_m(near, s)}" if m.reason == "muy-cerca" else f">{short_m(far, s)}",
        )
    return "sin-medida", ""


def _overlaps(r, placed) -> bool:
    x1, y1, x2, y2 = r
    return any(x1 < b[2] and b[0] < x2 and y1 < b[3] and b[1] < y2 for b in placed)


def model_name(model_id: str) -> str:
    m = model_id.lower()
    if m.startswith("crestereo"):
        return "CREStereo"
    if m.startswith("sgbm"):
        return "SGBM " + model_id.split("-")[-1].upper()
    return model_id.split("-")[0]


# --- paneles -------------------------------------------------------------------------------


def _panel(canvas, x, y, w, h, title, texts: Texts):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    if title:
        texts.add(title, (x + 14, y + 8), 20, MUTED, True)


def _place(canvas, img, x, y, w, h) -> tuple[float, float, int, int]:
    """Pega img escalada dentro de (x, y, w, h) sin deformar; devuelve escala y origen."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return s, s, ox, oy


def _camera(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, None, texts)
    joint = st.joint
    if joint is None:
        texts.add("Esperando la cámara…", (x + 30, y + h // 2 - 20), 34, MUTED)
        return
    img = joint.frame.left
    sx, sy, ox, oy = _place(canvas, img, x, y, w, h)
    det = joint.detection
    objs = {o.detection_index: o for o in (joint.objects or ())}
    if det is not None:
        items = []
        for i in range(len(det)):
            if float(det.scores[i]) < 0.35:
                continue
            x1, y1, x2, y2 = (float(c) for c in det.boxes[i])
            obj = objs.get(i)
            level, dist = level_of(obj, st.settings) if obj is not None else ("sin-medida", "")
            target = st.target_label is not None and det.classes[i] == st.target_label
            prio = (target, level != "sin-medida", (x2 - x1) * (y2 - y1))
            items.append((prio, i, (x1, y1, x2, y2), level, dist, target))
        items.sort(key=lambda it: it[0], reverse=True)
        placed: list[tuple[int, int, int, int]] = []
        for n, (_, i, (x1, y1, x2, y2), level, dist, target) in enumerate(items):
            p1 = (int(ox + x1 * sx), int(oy + y1 * sy))
            p2 = (int(ox + x2 * sx), int(oy + y2 * sy))
            color = LEVEL_COLOR[level]
            cv2.rectangle(canvas, p1, p2, color, 5 if target else 2, cv2.LINE_AA)
            if n >= 8:
                continue  # caja sí, rótulo no: con muchos objetos se tapaban
            name = st.spoken(det.classes[i]) if st.spoken else det.classes[i]
            label = f"{name} · {dist}" if dist else name
            size = 26 if target else 21
            tw, th = text_width(label, size, True) + 16, size + 12
            for ly in (p1[1] - th - 2, p1[1] + 2, p2[1] - th - 2):  # arriba, adentro, abajo
                rect = (p1[0], max(y + 2, ly), p1[0] + tw, max(y + 2, ly) + th)
                if not _overlaps(rect, placed):
                    break
            else:
                continue
            placed.append(rect)
            cv2.rectangle(canvas, rect[:2], rect[2:], color, -1)
            texts.add(label, (rect[0] + 8, rect[1] + 3), size, (20, 20, 20), True)
    if st.target_angle is not None and st.frame_geometry is not None:
        g = st.frame_geometry
        hd, vd = st.target_angle
        u = g.cx + g.fx * math.tan(math.radians(hd))
        v = g.cy - g.fy * math.tan(math.radians(vd))
        tip = (
            int(ox + min(max(u, 0), g.width - 1) * sx),
            int(oy + min(max(v, 0), g.height - 1) * sy),
        )
        center = (int(ox + g.width * sx / 2), int(oy + g.height * sy / 2))
        cv2.arrowedLine(canvas, center, tip, GREEN, 6, cv2.LINE_AA, tipLength=0.05)
        cv2.circle(canvas, tip, 22, GREEN, 4, cv2.LINE_AA)


def uncertainty_colormap(depth, s: ModeSettings) -> np.ndarray | None:
    """Medio ancho relativo del 90 % (ln10·b/(d+doffs)): verde ≤ precisa, rojo ≥ aproximada."""
    if depth is None or depth.uncertainty is None:
        return None
    if depth.uncertainty_kind is not UncertaintyKind.LAPLACE_B:
        return None
    d = depth.disparity_px.astype(np.float32) + depth.doffs
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = math.log(10.0) * depth.uncertainty / np.where(d > 0, d, np.nan)
    r = np.clip(np.nan_to_num(rel / s.approx_rel, nan=1.0), 0, 1)
    # verde (0) -> ámbar (0,4, umbral de precisa) -> rojo (1)
    g = np.where(r < 0.4, 1.0, np.clip((1 - r) / 0.6, 0, 1))
    red = np.where(r < 0.4, r / 0.4, 1.0)
    img = np.stack([np.full_like(r, 0.25) * 255 * (1 - r), g * 220, red * 235], -1).astype(np.uint8)
    img[~depth.valid_mask] = (30, 30, 30)
    return img


def _depth_panels(canvas, st: StandState, texts: Texts, x, y, w, h):
    half = (w - 16) // 2
    depth = st.joint.depth if st.joint is not None else None
    name = model_name(depth.model_id) if depth is not None else "—"
    for k, title in enumerate((f"Profundidad ({name})", "Incertidumbre (U-Net)")):
        px = x + k * (half + 16)
        _panel(canvas, px, y, half, h, title, texts)
        if k == 0:
            img = (
                None
                if depth is None or depth.depth_m is None
                else depth_colormap(depth.depth_m, depth.valid_mask, st.range_m)
            )
            note = (
                f"{st.range_m[0]:g}–{st.range_m[1]:g} m" if img is not None else "sin profundidad"
            )
        else:
            img = uncertainty_colormap(depth, st.settings)
            note = (f"verde ≤ {st.settings.precise_rel:.0%} · rojo ≥ {st.settings.approx_rel:.0%}"
                    if img is not None else "solo en el modo preciso / con la U-Net")  # fmt: skip
        if img is not None:
            _place(canvas, img, px + 8, y + 40, half - 16, h - 76)
        texts.add(note, (px + 14, y + h - 32), 18, MUTED)


def _radar(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Radar: la escena vista desde arriba", texts)
    top, bottom = y + 44, y + h - 20
    far = max(st.range_m[1] * 1.3, 3.5)
    cxp = x + w // 2
    scale = (bottom - top) / far
    for r in np.arange(1.0, far + 0.01, 1.0):  # anillos cada metro
        rr = int(r * scale)
        cv2.ellipse(canvas, (cxp, bottom), (rr, rr), 0, 180, 360, (70, 62, 58), 1, cv2.LINE_AA)
        texts.add(f"{r:g} m", (cxp + 6, bottom - rr - 4), 15, MUTED)
    g = st.geometry
    if g is not None:  # campo de visión
        half = math.atan2(g.width / 2, g.fx)
        for sgn in (-1, 1):
            end = (
                int(cxp + sgn * math.sin(half) * far * scale),
                int(bottom - math.cos(half) * far * scale),
            )
            cv2.line(canvas, (cxp, bottom), end, (80, 72, 66), 1, cv2.LINE_AA)
    joint = st.joint
    depth = joint.depth if joint is not None else None
    if depth is not None and depth.depth_m is not None and g is not None and g.size == depth.size:
        z = depth.depth_m[::6, ::6]
        ok = depth.valid_mask[::6, ::6] & np.isfinite(z) & (z > 0) & (z < far)
        rows, cols = np.nonzero(ok)
        zz = z[rows, cols]
        xx = (cols * 6 - g.cx) * zz / g.fx
        px = (cxp + xx * scale).astype(int)
        py = (bottom - zz * scale).astype(int)
        inside = (px > x) & (px < x + w) & (py > top) & (py < bottom)
        canvas[py[inside], px[inside]] = (120, 110, 95)
    objs = [o for o in ((joint.objects if joint is not None else ()) or ()) if o.z_m is not None]
    objs.sort(key=lambda o: (st.target_label is not None and o.label == st.target_label,
                             -o.z_m), reverse=True)  # fmt: skip
    placed: list[tuple[int, int, int, int]] = []
    for n, obj in enumerate(objs):
        level, dist = level_of(obj, st.settings)
        color = LEVEL_COLOR[level]
        xm = obj.z_m * math.tan(math.radians(obj.angle_h_deg))
        p = (int(cxp + xm * scale), int(bottom - obj.z_m * scale))
        if not (x < p[0] < x + w and top < p[1] < bottom):
            continue
        if obj.z_bajo_m is not None:  # margen del 90 % sobre la línea de vista
            hi = obj.z_alto_m if obj.z_alto_m is not None else far
            ang = math.tan(math.radians(obj.angle_h_deg))
            q1 = (int(cxp + obj.z_bajo_m * ang * scale), int(bottom - obj.z_bajo_m * scale))
            q2 = (int(cxp + min(hi, far) * ang * scale), int(bottom - min(hi, far) * scale))
            cv2.line(canvas, q1, q2, color, 4, cv2.LINE_AA)
        target = st.target_label is not None and obj.label == st.target_label
        cv2.circle(canvas, p, 13 if target else 9, color, -1, cv2.LINE_AA)
        if target:
            cv2.circle(canvas, p, 22, GREEN, 3, cv2.LINE_AA)
        if n >= 7:
            continue
        name = st.spoken(obj.label) if st.spoken else obj.label
        label = f"{name} {dist}".strip()
        tw = text_width(label, 19, True)
        rect = (p[0] + 14, p[1] - 16, p[0] + 18 + tw, p[1] + 10)
        if _overlaps(rect, placed):
            continue
        placed.append(rect)
        texts.add(label, (p[0] + 16, p[1] - 14), 19, INK, True)
    cv2.circle(canvas, (cxp, bottom), 12, ACCENT, -1, cv2.LINE_AA)
    texts.add("vos", (cxp - 16, bottom - 40), 18, ACCENT, True)


def _conversation(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Conversación", texts)
    lines: list[tuple[str, tuple[int, int, int], bool]] = []
    for turn in st.turns[-4:]:
        who = "Vos" if turn.who == "vos" else "VisionX"
        color = YOU if turn.who == "vos" else INK
        for i, line in enumerate(wrap(f"{who}: {turn.text}", 28, w - 40, bold=True)):
            lines.append((line, color, i == 0))
    lines = lines[-8:]
    cy = y + 44
    for line, color, first in lines:
        texts.add(line, (x + 20, cy), 28, color, first)
        cy += 38
    if st.recording:
        cv2.circle(canvas, (x + w - 40, y + 24), 12, RED, -1, cv2.LINE_AA)
        texts.add("Escuchando…", (x + w - 190, y + 10), 22, RED, True)


def _metrics(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Tiempos de la última orden", texts)
    t = st.timing
    bars = [("Voz a texto (Whisper)", t.stt_ms), ("Claude, primer texto", t.claude_ms),
            ("Primera palabra", t.first_word_ms)]  # fmt: skip
    by = y + 46
    for label, ms in bars:
        texts.add(label, (x + 20, by), 19, INK)
        frac = 0.0 if ms is None else min(ms / 4000, 1.0)
        bx, bw = x + 260, w - 380
        cv2.rectangle(canvas, (bx, by + 6), (bx + bw, by + 24), (60, 54, 50), -1)
        color = (
            GREEN
            if ms is not None and ms < 1500
            else AMBER
            if ms is not None and ms < 3000
            else RED
        )
        if ms is not None:
            cv2.rectangle(canvas, (bx, by + 6), (bx + int(bw * frac), by + 24), color, -1)
        texts.add(
            "—" if ms is None else f"{ms / 1000:.1f} s".replace(".", ","),
            (bx + bw + 14, by),
            20,
            INK,
            True,
        )
        by += 38
    fps = []
    for s in st.statuses:
        f = s.fps()
        fps.append(f"{s.name} {f:.0f}" if f else f"{s.name} —")
    texts.add("FPS: " + "  ·  ".join(fps), (x + 20, by + 8), 19, MUTED)
    b = st.body
    if b is not None:
        cam = f"cámara a {b.camera_m:.2f} m".replace(".", ",")
        pitch = f"inclinada {b.pitch_deg:+.0f}°" if b.floor_seen else "piso no visto"
        known = f"{b.height_m:.2f} m".replace(".", ",") if b.height_known else "sin decir (1,70 m)"
        texts.add(f"Cuerpo: estatura {known} · {cam} · {pitch}", (x + 20, by + 42), 19, MUTED)


def render_stand(st: StandState) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    texts = Texts()
    texts.add("VisionX", (28, 14), 44, ACCENT, True)
    texts.add("asistente por voz para personas ciegas · estéreo + IA", (230, 32), 20, MUTED)
    chip = st.mode
    cw = text_width(chip, 26, True) + 40
    cv2.rectangle(canvas, (W - cw - 28, 18), (W - 28, 66), (70, 58, 40), -1)
    texts.add(chip, (W - cw - 8, 26), 26, INK, True)
    _camera(canvas, st, texts, 20, 84, 1080, 608)
    _depth_panels(canvas, st, texts, 1120, 84, 780, 250)
    _radar(canvas, st, texts, 1120, 346, 780, 346)
    _conversation(canvas, st, texts, 20, 706, 1080, 354)
    _metrics(canvas, st, texts, 1120, 706, 780, 354)
    return texts.draw(canvas)


def mode_label(controller: Any, spoken: Any, recording: bool) -> tuple[str, str | None]:
    """(texto del chip, clase del objetivo) según la orden activa."""
    if recording:
        return "Escuchando…", None
    active = getattr(controller, "active", None)
    if active is None:
        return "Esperando una orden", None
    names = {"search": "Buscar", "distance": "Distancia", "read": "Leer", "describe": "Describir"}
    name = names.get(active.mode.value, active.mode.value)
    target = active.target if active.mode.value in ("search", "distance") else None
    if target:
        return f"{name}: {spoken(target)}", target
    return name, None


if __name__ == "__main__":  # vista previa sin cámara: python -m visionx.app.stand salida.png
    print("usar desde `python -m visionx.app ejecutar` (vista stand)", file=sys.stderr)
