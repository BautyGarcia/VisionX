"""Pantalla del stand: VisionX en vivo, para mostrarle al público qué ve, mide y dice.

A diferencia del dashboard de diagnóstico (`dashboard.py`), esta vista cuenta la interacción:

- **Cámara** grande con la máscara de cada objeto (como YOLO-seg en vivo, en tonos suaves) y un
  rótulo chico con su distancia en el color de su certeza (verde precisa, ámbar aproximada,
  coral no segura); el objetivo de buscar, resaltado y con la flecha.
- **Profundidad** (CREStereo) e **incertidumbre** (U-Net: cuánto puede errar cada píxel, en %
  de la distancia, con los mismos umbrales que deciden qué se dice).
- **Radar** desde arriba: la escena en planta, los objetos con su distancia y su margen.
- **Conversación** en burbujas: lo que dijo la persona y lo que respondió VisionX.
- **Cuánto tarda**: la última orden como barra (voz a texto → Claude → voz) con el total, los
  FPS de cada módulo y el **cuerpo** (estatura, altura e inclinación de la cámara).

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
# Paleta (BGR): azul noche, un solo acento celeste y colores de certeza apagados.
BG = (22, 17, 14)
PANEL = (34, 27, 22)
BORDER = (52, 43, 36)
RING = (62, 54, 48)
INK = (243, 237, 230)
MUTED = (165, 155, 146)
FAINT = (120, 110, 102)
ACCENT = (240, 201, 76)  # celeste
YOU = (240, 201, 76)
YOU_BG = (92, 70, 38)  # burbuja de la persona
VX_BG = (48, 40, 34)  # burbuja de VisionX
SEG = [(240, 201, 76), (220, 140, 170), (130, 200, 120)]  # voz a texto, Claude, voz
LEVEL_COLOR = {"precisa": (140, 200, 90), "aproximada": (60, 180, 235),
               "no-segura": (110, 105, 240), "sin-medida": (165, 155, 146)}  # fmt: skip
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
    camara: tuple[str, str] | None = None  # (texto, nivel) del perfil de `--camara`
    aceleracion: str | None = None  # «CREStereo en Radeon … · YOLO en CPU»


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


# --- dibujo: sistema visual ------------------------------------------------------------------


def _rounded(canvas, x1, y1, x2, y2, color, r=12, fill=True, thickness=1):
    """Rectángulo con esquinas redondeadas (relleno o borde)."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if fill:
        cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    for (cx, cy), a0 in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                         ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):  # fmt: skip
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, a0, a0 + 90, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)


def _blend_rect(canvas, x1, y1, x2, y2, color, alpha, r=8):
    """Rectángulo redondeado semitransparente (chips sobre la imagen)."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = canvas[y1:y2, x1:x2]
    layer = roi.copy()
    _rounded(layer, 0, 0, x2 - x1 - 1, y2 - y1 - 1, color, r)
    mask = np.any(layer != roi, axis=2)
    roi[mask] = (roi[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


SOMBRA_PAD = 12  # margen alrededor del panel que ocupa la sombra
PANEL_ALTO = tuple(min(255, c + 8) for c in PANEL)  # arriba del degradé
PANEL_BAJO = tuple(max(0, c - 4) for c in PANEL)  # abajo
BRILLO = tuple(min(255, c + 22) for c in BORDER)  # filo de arriba


@lru_cache(maxsize=96)
def _panel_bitmap(w: int, h: int, r: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """El fondo de un panel de w × h ya dibujado: sombra suave, degradé vertical, borde y un filo
    de luz arriba. Se calcula una vez por tamaño (la pantalla en vivo no paga el desenfoque)."""
    pad = SOMBRA_PAD
    alto, ancho = h + 2 * pad, w + 2 * pad
    sombra = np.zeros((alto, ancho), np.uint8)
    _rounded(sombra, pad + 2, pad + 6, pad + w - 2, pad + h + 4, 255, r)
    sombra = cv2.GaussianBlur(sombra, (0, 0), 7)
    a = (sombra.astype(np.float32) / 255.0 * 0.55)[:, :, None]
    img = (np.array(BG, np.float32) * (1 - a) + np.array((6, 5, 4), np.float32) * a).astype(
        np.uint8
    )
    forma = np.zeros((alto, ancho), np.uint8)
    _rounded(forma, pad, pad, pad + w, pad + h, 255, r)
    k = np.linspace(0.0, 1.0, alto, dtype=np.float32)[:, None, None]
    grad = np.array(PANEL_ALTO, np.float32) * (1 - k) + np.array(PANEL_BAJO, np.float32) * k
    grad = np.broadcast_to(grad, (alto, ancho, 3)).astype(np.uint8)
    dentro = forma > 0
    img[dentro] = grad[dentro]
    _rounded(img, pad, pad, pad + w, pad + h, BORDER, r, fill=False)
    cv2.line(img, (pad + r, pad + 1), (pad + w - r, pad + 1), BRILLO, 1, cv2.LINE_AA)
    mascara = dentro | (sombra > 2)
    mascara[pad - 1 : pad + h + 2, pad - 1 : pad + w + 2] |= True  # el borde antialiasado
    return img, mascara.astype(np.uint8) * 255


def _fondo_panel(canvas, x, y, w, h) -> None:
    img, mascara = _panel_bitmap(int(w), int(h))
    pad = SOMBRA_PAD
    x0, y0 = x - pad, y - pad
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + img.shape[1]), min(canvas.shape[0], y0 + img.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = img[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    m = mascara[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    roi = canvas[cy0:cy1, cx0:cx1]
    cv2.copyTo(sub, m, roi)  # copia nativa con máscara: ~1 ms por pantalla


def _panel(canvas, x, y, w, h, title, texts: Texts, note: str | None = None):
    _fondo_panel(canvas, x, y, w, h)
    if title:
        texts.add(title.upper(), (x + 18, y + 14), 13, MUTED, True)
    if note:
        texts.add(note, (x + w - 18 - text_width(note, 13), y + 14), 13, FAINT)


def _place(canvas, img, x, y, w, h) -> tuple[float, float, int, int]:
    """Pega img escalada dentro de (x, y, w, h) sin deformar; devuelve escala y origen."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return s, s, ox, oy


# Tonos suaves por clase (BGR): la máscara se distingue sin gritar.
SOFT = [
    (235, 175, 110), (150, 200, 120), (120, 170, 245), (210, 140, 200), (110, 205, 215),
    (160, 150, 240), (130, 210, 170), (215, 190, 140), (120, 140, 230), (200, 200, 120),
]  # fmt: skip


def _soft(label: str) -> tuple[int, int, int]:
    return SOFT[sum(map(ord, label)) % len(SOFT)]


def _camera(canvas, st: StandState, texts: Texts, x, y, w, h):
    _rounded(canvas, x, y, x + w, y + h, (14, 11, 9), 16)
    joint = st.joint
    if joint is None:
        texts.add("Esperando la cámara…", (x + w // 2 - 120, y + h // 2 - 14), 24, MUTED)
        return
    img = joint.frame.left.copy()
    det = joint.detection
    objs = {o.detection_index: o for o in (joint.objects or ())}
    items = []
    if det is not None:
        for i in range(len(det)):
            if float(det.scores[i]) < 0.35:
                continue
            target = st.target_label is not None and det.classes[i] == st.target_label
            color = (120, 235, 120) if target else _soft(det.classes[i])
            if det.masks is not None:  # la máscara pintada, como la ve YOLO-seg en vivo
                m = det.masks[i]
                a = 0.42 if target else 0.30
                img[m] = (img[m] * (1 - a) + np.array(color) * a).astype(np.uint8)
            obj = objs.get(i)
            level, dist = level_of(obj, st.settings) if obj is not None else ("sin-medida", "")
            z = obj.z_m if obj is not None and obj.z_m is not None else 99.0
            items.append((target, level != "sin-medida", -z, i, color, level, dist))
    sx, sy, ox, oy = _place(canvas, img, x, y, w, h)
    if det is not None and det.masks is not None:  # contornos finos a la escala del panel
        nw, nh = int(det.image_size[0] * sx), int(det.image_size[1] * sy)
        for target, _, _, i, color, _, _ in items:
            small = cv2.resize(
                det.masks[i].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
            )
            cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c + np.array([ox, oy]) for c in cnts]
            cv2.drawContours(canvas, cnts, -1, color, 3 if target else 1, cv2.LINE_AA)
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
        cv2.circle(canvas, center, 6, ACCENT, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, center, tip, ACCENT, 3, cv2.LINE_AA, tipLength=0.04)
        cv2.circle(canvas, tip, 18, ACCENT, 2, cv2.LINE_AA)
    items.sort(reverse=True)  # objetivo, medidos, más cercanos
    placed: list[tuple[int, int, int, int]] = []
    for target, _, _, i, color, level, dist in items[:6]:
        x1, y1, x2, _ = (float(c) for c in det.boxes[i])
        name = st.spoken(det.classes[i]) if st.spoken else det.classes[i]
        size = 17 if target else 15
        wn, wd = text_width(name, size, True), text_width(dist, size, True) if dist else 0
        cw, ch = 26 + wn + (10 + wd if dist else 0) + 12, size + 14
        cx = int(ox + (x1 + x2) / 2 * sx - cw / 2)
        cx = min(max(cx, x + 6), x + w - cw - 6)
        for cy in (int(oy + y1 * sy) - ch - 6, int(oy + y1 * sy) + 6):
            cy = max(y + 6, cy)
            rect = (cx, cy, cx + cw, cy + ch)
            if not _overlaps(rect, placed):
                break
        else:
            continue
        placed.append(rect)
        _blend_rect(canvas, *rect, (16, 13, 11), 0.78, r=ch // 2)
        cv2.circle(canvas, (cx + 14, cy + ch // 2), 5, color, -1, cv2.LINE_AA)
        texts.add(name, (cx + 26, cy + 5), size, INK, True)
        if dist:
            texts.add(dist, (cx + 26 + wn + 10, cy + 5), size, LEVEL_COLOR[level], True)


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
    lo, mid, hi = (
        np.array(LEVEL_COLOR["precisa"]),
        np.array(LEVEL_COLOR["aproximada"]),
        np.array(LEVEL_COLOR["no-segura"]),
    )
    k = s.precise_rel / s.approx_rel  # dónde termina «precisa» en la escala
    t1 = np.clip(r / k, 0, 1)[..., None]
    t2 = np.clip((r - k) / (1 - k), 0, 1)[..., None]
    img = np.where(r[..., None] <= k, lo * (1 - t1) + mid * t1, mid * (1 - t2) + hi * t2)
    img = (img * 0.85).astype(np.uint8)
    img[~depth.valid_mask] = (24, 20, 17)
    return img


def _depth_panels(canvas, st: StandState, texts: Texts, x, y, w, h):
    half = (w - 14) // 2
    depth = st.joint.depth if st.joint is not None else None
    name = model_name(depth.model_id) if depth is not None else "—"
    for k in range(2):
        px = x + k * (half + 14)
        if k == 0:
            img = (
                None
                if depth is None or depth.depth_m is None
                else depth_colormap(depth.depth_m, depth.valid_mask, st.range_m)
            )
            _panel(canvas, px, y, half, h, "Profundidad", texts, name)
            note = (
                f"cerca {st.range_m[0]:g} m · lejos {st.range_m[1]:g} m" if img is not None else ""
            )
        else:
            img = uncertainty_colormap(depth, st.settings)
            _panel(canvas, px, y, half, h, "Incertidumbre", texts, "U-Net")
            note = "cuánto puede errar cada punto" if img is not None else ""
        if img is not None:
            _place(canvas, img, px + 12, y + 40, half - 24, h - 74)
            texts.add(note, (px + 18, y + h - 28), 13, FAINT)
        else:
            msg = "sin profundidad" if k == 0 else "se activa con la U-Net de incertidumbre"
            texts.add(msg, (px + half // 2 - text_width(msg, 14) // 2, y + h // 2 - 8), 14, FAINT)


def _radar(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Vista desde arriba", texts, "cada anillo, 1 m")
    top, bottom = y + 40, y + h - 22
    far = max(st.range_m[1] * 1.3, 3.5)
    cxp = x + w // 2
    scale = (bottom - top) / far
    for r in np.arange(1.0, far + 0.01, 1.0):
        rr = int(r * scale)
        cv2.ellipse(canvas, (cxp, bottom), (rr, rr), 0, 200, 340, RING, 1, cv2.LINE_AA)
    g = st.geometry
    if g is not None:  # campo de visión
        half = math.atan2(g.width / 2, g.fx)
        pts = [(cxp, bottom)] + [
            (int(cxp + math.sin(a) * far * scale), int(bottom - math.cos(a) * far * scale))
                                 for a in np.linspace(-half, half, 24)]  # fmt: skip
        layer = canvas.copy()
        cv2.fillPoly(layer, [np.array(pts, np.int32)], (40, 34, 28))
        cv2.addWeighted(layer, 0.5, canvas, 0.5, 0, canvas)
    joint = st.joint
    depth = joint.depth if joint is not None else None
    if depth is not None and depth.depth_m is not None and g is not None and g.size == depth.size:
        z = depth.depth_m[::5, ::5]
        ok = depth.valid_mask[::5, ::5] & np.isfinite(z) & (z > 0) & (z < far)
        rows, cols = np.nonzero(ok)
        zz = z[rows, cols]
        xx = (cols * 5 - g.cx) * zz / g.fx
        px = (cxp + xx * scale).astype(int)
        py = (bottom - zz * scale).astype(int)
        inside = (px > x + 4) & (px < x + w - 4) & (py > top) & (py < bottom)
        canvas[py[inside], px[inside]] = (95, 88, 80)
    objs = [o for o in ((joint.objects if joint is not None else ()) or ()) if o.z_m is not None]
    objs.sort(key=lambda o: (st.target_label is not None and o.label == st.target_label, -o.z_m),
              reverse=True)  # fmt: skip
    placed: list[tuple[int, int, int, int]] = []
    for n, obj in enumerate(objs):
        level, dist = level_of(obj, st.settings)
        target = st.target_label is not None and obj.label == st.target_label
        color = (120, 235, 120) if target else _soft(obj.label)
        ang = math.tan(math.radians(obj.angle_h_deg))
        p = (int(cxp + obj.z_m * ang * scale), int(bottom - obj.z_m * scale))
        if not (x < p[0] < x + w and top < p[1] < bottom):
            continue
        if obj.z_bajo_m is not None:  # margen del 90 % sobre la línea de vista
            hi = min(obj.z_alto_m if obj.z_alto_m is not None else far, far)
            q1 = (int(cxp + obj.z_bajo_m * ang * scale), int(bottom - obj.z_bajo_m * scale))
            q2 = (int(cxp + hi * ang * scale), int(bottom - hi * scale))
            cv2.line(canvas, q1, q2, LEVEL_COLOR[level], 3, cv2.LINE_AA)
        cv2.circle(canvas, p, 8 if target else 6, color, -1, cv2.LINE_AA)
        if target:
            cv2.circle(canvas, p, 15, ACCENT, 2, cv2.LINE_AA)
        if n >= 6:
            continue
        name = st.spoken(obj.label) if st.spoken else obj.label
        wn = text_width(name, 13, True)
        rect = (p[0] + 12, p[1] - 10, p[0] + 18 + wn + text_width(dist, 13, True), p[1] + 8)
        if _overlaps(rect, placed):
            continue
        placed.append(rect)
        texts.add(name, (p[0] + 12, p[1] - 9), 13, INK, True)
        texts.add(dist, (p[0] + 16 + wn, p[1] - 9), 13, LEVEL_COLOR[level], True)
    cv2.circle(canvas, (cxp, bottom), 9, ACCENT, -1, cv2.LINE_AA)
    texts.add("vos", (cxp + 14, bottom - 9), 13, ACCENT, True)


def _conversation(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Conversación", texts)
    size, line_h, max_w = 19, 27, int(w * 0.68)
    bubbles = []
    for turn in st.turns[-6:]:
        mine = turn.who == "vos"
        lines = wrap(turn.text, size, max_w - 36)[-4:]
        bubbles.append((mine, lines))
    # De abajo hacia arriba: entran las últimas que caben.
    cy = y + h - 16
    drawn = []
    for mine, lines in reversed(bubbles):
        bh = 18 + line_h * len(lines)
        if cy - bh < y + 40:
            break
        cy -= bh
        drawn.append((mine, lines, cy, bh))
        cy -= 10
    for mine, lines, by, bh in drawn:
        bw = max(text_width(line, size) for line in lines) + 36
        bx = x + w - 18 - bw if mine else x + 18
        _rounded(canvas, bx, by, bx + bw, by + bh, YOU_BG if mine else VX_BG, 14)
        for k, line in enumerate(lines):
            texts.add(line, (bx + 18, by + 8 + k * line_h), size, INK)
    if st.recording:
        cv2.circle(canvas, (x + w - 30, y + 22), 7, LEVEL_COLOR["no-segura"], -1, cv2.LINE_AA)
        texts.add("escuchando", (x + w - 120, y + 13), 14, LEVEL_COLOR["no-segura"], True)


def _metrics(canvas, st: StandState, texts: Texts, x, y, w, h):
    _panel(canvas, x, y, w, h, "Cuánto tarda en responder", texts, "última orden")
    t = st.timing
    stt = t.stt_ms or 0.0
    claude = t.claude_ms or 0.0
    total = t.first_word_ms
    if total is not None and stt:
        total = total + stt if total < stt else total  # sin botón no hay STT
    segs = [("Voz a texto", stt, SEG[0]), ("Claude", claude, SEG[1])]
    rest = (total - stt - claude) if total is not None else 0.0
    segs.append(("Voz", max(rest, 0.0), SEG[2]))
    bx, by, bw, bh = x + 18, y + 72, w - 36, 22
    _rounded(canvas, bx, by, bx + bw, by + bh, (44, 38, 33), bh // 2)
    full = max(sum(s[1] for s in segs), 1.0)
    scale_ms = max(full, 4000.0)
    cx = bx
    for _name, ms, color in segs:
        sw = int(bw * ms / scale_ms)
        if sw > 2:
            _rounded(canvas, cx, by, cx + sw, by + bh, color, min(bh // 2, sw // 2))
        cx += sw
    big = "—" if total is None else f"{total / 1000:.1f} s".replace(".", ",")
    texts.add(big, (x + w - 18 - text_width(big, 30, True), y + 30), 30, INK, True)
    texts.add("desde que soltás el botón hasta la primera palabra", (x + 18, y + 40), 13, FAINT)
    lx = bx
    for name, ms, color in segs:
        cv2.circle(canvas, (lx + 6, by + bh + 22), 5, color, -1, cv2.LINE_AA)
        label = f"{name} {ms / 1000:.1f} s".replace(".", ",") if ms else f"{name} —"
        texts.add(label, (lx + 16, by + bh + 13), 14, MUTED)
        lx += text_width(label, 14) + 40
    fy = by + bh + 58
    b = st.body
    if b is not None:
        est = f"{b.height_m:.2f} m".replace(".", ",") if b.height_known else "1,70 m (sin decir)"
        cam = f"cámara a {b.camera_m:.2f} m".replace(".", ",")
        inc = f", inclinada {b.pitch_deg:+.0f}°" if b.floor_seen else ""
        texts.add(f"Estatura {est} · {cam}{inc}", (x + 18, fy + 4), 13, FAINT)


STREAM = {"captura": "Cámara", "detector": "YOLO", "profundidad": None, "precisa": "Preciso"}


def _streams(canvas, st: StandState, texts: Texts, x: int, y: int) -> None:
    """Un chip por flujo: Cámara, YOLO, la profundidad (CREStereo) y el preciso, con FPS y ms."""
    depth = st.joint.depth if st.joint is not None else None
    now = time.monotonic_ns()
    cx = x
    for s in st.statuses:
        name = STREAM.get(s.name, s.name)
        if name is None:
            name = model_name(depth.model_id) if depth is not None else "Profundidad"
        f = s.fps()
        parts = [f"{f:.0f} fps" if f else "— fps"]
        if s.name != "captura" and s.last_latency_ms is not None:
            ms = s.last_latency_ms
            parts.append(f"{ms / 1000:.1f} s".replace(".", ",") if ms >= 1000 else f"{ms:.0f} ms")
        age = s.age_ms(now)
        health = (LEVEL_COLOR["no-segura"] if s.errors else
                  LEVEL_COLOR["aproximada"] if age is not None and age > 800 else
                  LEVEL_COLOR["precisa"])  # fmt: skip
        label, value = name, " · ".join(parts)
        cw = 30 + text_width(label, 14, True) + 8 + text_width(value, 14) + 14
        _rounded(canvas, cx, y, cx + cw, y + 32, (40, 33, 27), 16)
        cv2.circle(canvas, (cx + 15, y + 16), 5, health, -1, cv2.LINE_AA)
        texts.add(label, (cx + 27, y + 7), 14, INK, True)
        texts.add(value, (cx + 27 + text_width(label, 14, True) + 8, y + 7), 14, MUTED)
        cx += cw + 8


def render_stand(st: StandState) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    texts = Texts()
    texts.add("VisionX", (32, 22), 30, INK, True)
    texts.add(
        "asistente por voz con visión estéreo",
        (32 + text_width("VisionX", 30, True) + 16, 34),
        15,
        MUTED,
    )
    if st.camara is not None:
        texto, nivel = st.camara
        color = LEVEL_COLOR.get(nivel, LEVEL_COLOR["no-segura"])
        cv2.circle(canvas, (38, 66), 5, color, -1, cv2.LINE_AA)
        texts.add(texto, (50, 57), 14, MUTED)
    chip = st.mode
    cw = text_width(chip, 16, True) + 44
    color = LEVEL_COLOR["no-segura"] if st.recording else ACCENT
    _rounded(canvas, W - cw - 32, 22, W - 32, 58, (48, 40, 30), 18)
    cv2.circle(canvas, (W - cw - 32 + 18, 40), 6, color, -1, cv2.LINE_AA)
    texts.add(chip, (W - cw - 32 + 32, 30), 16, INK, True)
    _streams(canvas, st, texts, 470, 24)
    if st.aceleracion:
        texts.add(st.aceleracion, (474, 60), 13, FAINT)
    _camera(canvas, st, texts, 24, 80, 1160, 653)
    _depth_panels(canvas, st, texts, 1204, 80, 692, 236)
    _radar(canvas, st, texts, 1204, 330, 692, 403)
    _conversation(canvas, st, texts, 24, 750, 1160, 306)
    _metrics(canvas, st, texts, 1204, 750, 692, 306)
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
