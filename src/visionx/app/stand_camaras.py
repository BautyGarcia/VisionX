"""Pestaña «Cámaras» del stand: ¿VisionX funciona con otra cámara estéreo?

Muestra, para unas pocas escenas de validación, la misma vista con siete cámaras distintas
(campo de 60° a 130°, ojos separados de 3 a 20 cm, formato 16:9 o 4:3): lo que ve el ojo
izquierdo, la profundidad real del render, la que calcula CREStereo con la calibración de esa
cámara y el error por píxel. Pasa sola de una cámara a la siguiente.

Todo viene precalculado por `python -m visionx.gtsynth camaras` (gtsynth/camaras.py): la
pantalla solo lee imágenes, no corre ningún modelo. Tecla `k` en la app; o sola con
`python -m visionx.gtsynth camaras-ver`.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from visionx.app.stand import (
    ACCENT,
    BG,
    FAINT,
    INK,
    LEVEL_COLOR,
    MUTED,
    PANEL,
    H,
    Texts,
    W,
    _panel,
    _place,
    _rounded,
    text_width,
    wrap,
)

CADA_S = 6.0  # segundos por cámara (fotos)


def carpeta_por_defecto() -> Path:
    from visionx.shared.config import ROOT

    video = ROOT / "data" / "stand-video"
    return video if video.is_dir() and any(video.glob("*.mp4")) else ROOT / "data" / "stand-camaras"


@lru_cache(maxsize=4)
def cargar(carpeta: str) -> list[list[dict]]:
    """Escenas -> cámaras (en el orden de CAMARAS), con sus imágenes ya leídas."""
    from visionx.gtsynth.camaras import CAMARAS

    escenas = []
    for d in sorted(p for p in Path(carpeta).iterdir() if p.is_dir()):
        camaras = []
        for var, _ in CAMARAS:
            c = d / var
            if not (c / "datos.json").is_file():
                continue
            datos = json.loads((c / "datos.json").read_text(encoding="utf-8"))
            for n in ("izq", "real", "estimada", "error"):
                datos[n] = cv2.imread(str(c / f"{n}.jpg"))
            camaras.append(datos)
        if camaras:
            escenas.append(camaras)
    return escenas


def _es(clase: str) -> str:
    """Clase del render en castellano (lamp -> lámpara)."""
    from visionx.detection.catalog import VISIONX_CLASSES, ClassCatalog
    from visionx.gtsynth.deteccion import REPLICA_A_VISIONX

    return ClassCatalog(VISIONX_CLASSES).spoken(REPLICA_A_VISIONX.get(clase, clase))


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f} %".replace(".", ",")


def _color_error(x: float | None) -> tuple[int, int, int]:
    if x is None:
        return LEVEL_COLOR["sin-medida"]
    a = abs(x)
    return LEVEL_COLOR["precisa" if a < 3 else "aproximada" if a < 10 else "no-segura"]


class Videos:
    """Los MP4 de `gtsynth video` en orden (por escena, las cámaras de la expo), a 10 cuadros/s.

    Solo las de `CAMARAS_VIDEO`; las genéricas (60°, 130°, 20 cm) quedan para evaluar. Si en la
    carpeta no hay ninguna de la expo, se reproducen todas.
    """

    def __init__(self, carpeta: Path) -> None:
        from visionx.gtsynth.video import CAMARAS_VIDEO, FPS

        orden = {var: k for k, (var, _) in enumerate(CAMARAS_VIDEO)}

        def clave(p: Path):
            escena, _, var = p.stem.partition("-video-")
            return escena, orden.get("video-" + var, 99)

        todas = sorted(carpeta.glob("*.mp4"), key=clave)
        self.paths = [p for p in todas if clave(p)[1] < 99] or todas
        self.fps = FPS
        self._k = -1
        self._cap = None
        self._t0 = 0.0
        self._leidos = 0
        self._ultimo = None

    def _siguiente(self, t: float) -> None:
        if self._cap is not None:
            self._cap.release()
        self._k = (self._k + 1) % len(self.paths)
        self._cap = cv2.VideoCapture(str(self.paths[self._k]))
        self._t0, self._leidos = t, 0

    def cuadro(self, t: float) -> np.ndarray:
        if self._cap is None:
            self._siguiente(t)
        quiero = int((t - self._t0) * self.fps) + 1
        while self._leidos < quiero:
            ok, img = self._cap.read()
            if not ok:  # terminó: el próximo video
                self._siguiente(t)
                quiero = 1
                continue
            self._leidos += 1
            self._ultimo = img
        if self._ultimo is None or self._ultimo.shape[:2] != (H, W):
            return np.full((H, W, 3), BG, np.uint8)
        return self._ultimo


_VIDEOS: dict[str, Videos] = {}


def render_camaras(carpeta: str | Path, ahora_s: float | None = None) -> np.ndarray:
    """Videos si la carpeta los tiene (`gtsynth video`); si no, las fotos (`gtsynth camaras`)."""
    carpeta = Path(carpeta)
    if carpeta.is_dir() and any(carpeta.glob("*.mp4")):
        if str(carpeta) not in _VIDEOS:
            _VIDEOS[str(carpeta)] = Videos(carpeta)
        return _VIDEOS[str(carpeta)].cuadro(time.monotonic() if ahora_s is None else ahora_s)
    return _fotos(carpeta, ahora_s)


def _fotos(carpeta: str | Path, ahora_s: float | None = None) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    texts = Texts()
    escenas = cargar(str(carpeta))
    texts.add("VisionX", (32, 22), 30, INK, True)
    texts.add("¿Funciona con otra cámara?", (32 + text_width("VisionX", 30, True) + 16, 30), 20,
              ACCENT, True)  # fmt: skip
    if not escenas:
        texts.add(f"No hay demos en {carpeta}: python -m visionx.gtsynth camaras ...",
                  (32, 120), 20, MUTED)  # fmt: skip
        return texts.draw(canvas)
    t = time.monotonic() if ahora_s is None else ahora_s
    n_cam = len(escenas[0])
    paso = int(t // CADA_S)
    camaras = escenas[(paso // n_cam) % len(escenas)]
    k = paso % len(camaras)
    c = camaras[k]

    # Cuatro paneles 2×2: lo que ve, la verdad, lo que calcula, el error.
    x0, y0, pw, ph, gap = 24, 80, 700, 430, 14
    titulos = [
        ("izq", "Lo que ve el ojo izquierdo", None),
        ("real", "Distancia real", "render: la verdad"),
        ("estimada", "Distancia que calcula VisionX", "CREStereo"),
        ("error", "Error por píxel", "verde < 3 % · ámbar < 10 %"),
    ]
    for i, (clave, titulo, nota) in enumerate(titulos):
        x = x0 + (i % 2) * (pw + gap)
        y = y0 + (i // 2) * (ph + gap)
        _panel(canvas, x, y, pw, ph, titulo, texts, nota)
        _place(canvas, c[clave], x + 12, y + 40, pw - 24, ph - 52)

    # Columna derecha: la cámara de ahora y la lista de todas.
    cx = x0 + 2 * pw + gap + 20
    cw = W - cx - 24
    _panel(canvas, cx, y0, cw, 300, "Esta cámara", texts)
    texts.add(c["nombre"].split(" · ")[0], (cx + 18, y0 + 44), 26, INK, True)
    texts.add(f"campo {c['hfov_deg']:.0f}°  ·  ojos a {c['base_cm']:.0f} cm  ·  "
              f"{c['ancho']}×{c['alto']}", (cx + 18, y0 + 84), 16, MUTED)  # fmt: skip
    texts.add("Error mediano de la distancia", (cx + 18, y0 + 126), 15, MUTED)
    texts.add(_pct(c["error_mediano_pct"]), (cx + 18, y0 + 148), 40,
              _color_error(c["error_mediano_pct"]), True)  # fmt: skip
    obj = c.get("objeto")
    if obj:
        real = f"{obj['z_real_m']:.2f}".replace(".", ",")
        est = f"{obj['z_estimada_m']:.2f}".replace(".", ",")
        texts.add(f"Objeto apuntado: {_es(obj['clase'])}", (cx + 18, y0 + 210), 15, MUTED)
        texts.add(f"real {real} m  ·  VisionX {est} m", (cx + 18, y0 + 234), 18, INK, True)
        texts.add(_pct(obj["error_pct"]), (cx + cw - 18 - text_width(_pct(obj["error_pct"]), 18,
                  True), y0 + 234), 18, _color_error(obj["error_pct"]), True)  # fmt: skip
    else:
        texts.add("El objeto apuntado queda fuera del campo de esta cámara.",
                  (cx + 18, y0 + 222), 15, FAINT)  # fmt: skip

    ly = y0 + 316
    _panel(canvas, cx, ly, cw, 2 * ph + gap - 316, "Las siete cámaras, misma escena", texts)
    for j, cam in enumerate(camaras):
        yy = ly + 44 + j * 52
        if j == k:
            _rounded(canvas, cx + 10, yy - 6, cx + cw - 10, yy + 40, (60, 48, 36), 10)
        nombre = cam["nombre"].split(" · ")
        texts.add(nombre[0], (cx + 22, yy), 16, INK if j == k else MUTED, j == k)
        texts.add(" · ".join(nombre[1:]), (cx + 22, yy + 20), 13, FAINT)
        e = _pct(cam["error_mediano_pct"])
        texts.add(e, (cx + cw - 24 - text_width(e, 18, True), yy + 6), 18,
                  _color_error(cam["error_mediano_pct"]), True)  # fmt: skip

    # Pie: cómo leerlo, sin sobrevender.
    pie = ("Mismo modelo de profundidad para todas: solo cambia la calibración que se le pasa. "
           "Es un render con la calibración exacta; con una cámara real el error es mayor y "
           "depende de calibrarla bien (lo que corrige la red de alineación).")  # fmt: skip
    for i, linea in enumerate(wrap(pie, 15, W - 48)):
        texts.add(linea, (24, y0 + 2 * ph + gap + 18 + i * 22), 15, FAINT)
    barra = int((t % CADA_S) / CADA_S * (W - 48))
    cv2.rectangle(canvas, (24, H - 12), (W - 24, H - 8), PANEL, -1)
    cv2.rectangle(canvas, (24, H - 12), (24 + barra, H - 8), FAINT, -1)
    return texts.draw(canvas)


def mostrar(carpeta: str | Path) -> None:
    """Ventana sola (segundo monitor del stand): q, Esc o la X salen; f pantalla completa."""
    cv2.namedWindow("VisionX - Cámaras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionX - Cámaras", 1600, 900)
    while True:
        cv2.imshow("VisionX - Cámaras", render_camaras(carpeta))
        key = cv2.waitKey(30) & 0xFF
        # q, Esc o la X de la ventana (sin esto, cerrarla con la X la volvía a abrir).
        cerrada = cv2.getWindowProperty("VisionX - Cámaras", cv2.WND_PROP_VISIBLE) < 1
        if key in (ord("q"), 27) or cerrada:
            break
        if key == ord("f"):
            full = cv2.getWindowProperty("VisionX - Cámaras", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("VisionX - Cámaras", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN
                                  else cv2.WINDOW_FULLSCREEN)  # fmt: skip
    cv2.destroyAllWindows()
