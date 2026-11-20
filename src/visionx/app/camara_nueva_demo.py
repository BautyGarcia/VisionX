"""Pantalla «Cámara nueva» del dashboard y comando para correrla.

    python -m visionx.app camara-nueva --fuente sim:video-zed-mini [--escala 6.3|saltear] [--voz]
    python -m visionx.app camara-nueva --fuente usb [--dispositivo "Nombre"] [--voz]
    python -m visionx.app camara-nueva --fuente realsense [--voz]

Con `sim:<variante>` usa un recorrido de `data/gt-video` como si fuera una cámara desconocida
(el ojo derecho de «otra unidad», como en los videos del stand): sirve para mostrar el flujo
en la expo sin la cámara. La escala se puede decir por voz o dejar fija con `--escala` (cm).

Teclas: q / Esc = salir, r = empezar de nuevo, s = saltear la escala.
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from visionx.app.camara_nueva import ORDEN, TITULOS, AltaCamara, Paso
from visionx.app.stand import (
    ACCENT,
    BG,
    FAINT,
    INK,
    LEVEL_COLOR,
    MUTED,
    Texts,
    _panel,
    _place,
    text_width,
    wrap,
)
from visionx.camera.perfiles import cargar_perfiles
from visionx.shared.config import ROOT

W, H = 1920, 1080
OK = (140, 200, 90)
NIVEL_COLOR = {**LEVEL_COLOR, "sin-estimar": LEVEL_COLOR["no-segura"]}
NIVEL_TEXTO = {"precisa": "precisa", "aproximada": "aproximada", "sin-estimar": "sin escala"}


def puntos_filas(izq: np.ndarray, filas) -> np.ndarray:
    """El ojo izquierdo apagado con cada punto emparejado: verde si los dos ojos lo ven a la
    misma altura, ámbar si se corre un poco y rojo si se corre más de 1,5 píxeles."""
    img = (izq.astype(np.float32) * 0.45).astype(np.uint8)
    r = max(3, izq.shape[1] // 260)
    for (x, y), dy in zip(filas.pts_izq, filas.dy, strict=False):
        a = abs(float(dy))
        color = OK if a < 0.5 else (60, 180, 235) if a < 1.5 else (110, 105, 240)
        cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
    return img


def _icono(canvas: np.ndarray, cx: int, cy: int, estado: str, color) -> None:
    if estado == "hecho":
        cv2.circle(canvas, (cx, cy), 13, color, -1, cv2.LINE_AA)
        pts = np.array([[cx - 6, cy], [cx - 1, cy + 5], [cx + 7, cy - 5]], np.int32)
        cv2.polylines(canvas, [pts], False, BG, 3, cv2.LINE_AA)
    elif estado == "actual":
        cv2.circle(canvas, (cx, cy), 13, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 7, color, -1, cv2.LINE_AA)
    elif estado == "salteado":
        cv2.line(canvas, (cx - 9, cy), (cx + 9, cy), color, 2, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (cx, cy), 13, color, 2, cv2.LINE_AA)


def render_alta(alta: AltaCamara, izq: np.ndarray | None) -> np.ndarray:
    """La pantalla completa (1920×1080) para el estado actual del alta."""
    e = alta.estado
    canvas = np.full((H, W, 3), BG, np.uint8)
    t = Texts()
    t.add("VisionX", (32, 22), 30, INK, True)
    t.add("Cámara nueva", (32 + text_width("VisionX", 30, True) + 16, 30), 20, ACCENT)
    if e.dispositivo:
        nota = f"Dispositivo: {e.dispositivo}"
        t.add(nota, (W - 32 - text_width(nota, 18), 32), 18, MUTED)

    # Pasos
    x, y, w, h = 32, 84, 520, 640
    _panel(canvas, x, y, w, h, "Pasos", t)
    actual = ORDEN.index(e.paso)
    yy = y + 56
    for k, paso in enumerate(ORDEN):
        if e.conocida and paso in (Paso.MIRAR, Paso.ESTIMAR, Paso.ALINEAR, Paso.ESCALA):
            estado, color = "salteado", FAINT
        elif k < actual or e.paso is Paso.LISTO:
            estado, color = "hecho", OK
        elif k == actual:
            estado, color = "actual", ACCENT
        else:
            estado, color = "falta", FAINT
        _icono(canvas, x + 38, yy + 16, estado, color)
        t.add(TITULOS[paso], (x + 64, yy + 2), 24, INK if k <= actual else MUTED, k == actual)
        detalle = _detalle(alta, paso)
        if detalle:
            t.add(detalle, (x + 64, yy + 32), 16, MUTED)
        yy += 78

    # Imagen en vivo y antes / después
    px, pw = 584, W - 584 - 32
    _panel(canvas, px, 84, pw, 420, "Lo que ve la cámara", t)
    if izq is not None:
        _place(canvas, izq, px + 16, 84 + 44, pw - 32, 420 - 60)
    muestra = getattr(alta, "muestra", None)
    hw = (pw - 16) // 2
    for k, titulo in enumerate(("Antes de alinear", "Después de alinear")):
        bx = px + k * (hw + 16)
        nota = None
        filas = e.filas_antes if k == 0 else e.filas_despues
        if filas is not None and filas.mediana_px is not None:
            nota = f"desfase de filas {filas.mediana_px:.1f} px".replace(".", ",")
        _panel(canvas, bx, 520, hw, 300, titulo, t, nota)
        if muestra is not None and filas is not None:
            img = puntos_filas(muestra[0], filas)
            _place(canvas, img, bx + 16, 520 + 44, hw - 32, 300 - 60)
        else:
            t.add("Aparece al alinear los dos ojos", (bx + 24, 520 + 140), 18, FAINT)
    leyenda = "Cada punto es un detalle que se ve en los dos ojos: verde si está a la misma "
    leyenda += "altura, rojo si no."
    t.add(leyenda, (px + 4, 822), 15, FAINT)

    # Perfil
    _panel(canvas, 32, 740, 520, 308, "Perfil de la cámara", t)
    p = e.perfil
    if p is not None:
        filas = [
            ("Nombre", p.nombre),
            ("Campo horizontal", f"{p.campo_horizontal_deg:.0f}°"),
            ("Separación entre lentes", f"{100 * p.separacion_m:.1f} cm".replace(".", ",")),
            ("Escala", {"separacion": "separación dicha", "distancia": "objeto a distancia",
                        "perfil": "del perfil", "salteada": "sin escala"}.get(e.escala, "—")),
        ]  # fmt: skip
        for k, (a, b) in enumerate(filas):
            t.add(a, (56, 796 + k * 44), 18, MUTED)
            t.add(b, (300, 794 + k * 44), 20, INK)
        nivel = e.nivel if e.paso is Paso.LISTO else "sin-estimar"
        etiqueta = NIVEL_TEXTO.get(nivel, nivel) if e.paso is Paso.LISTO else "en curso"
        t.add("Distancias:", (56, 990), 20, MUTED)
        t.add(etiqueta, (180, 988), 22, NIVEL_COLOR.get(nivel, MUTED) if e.paso is Paso.LISTO
              else ACCENT, True)  # fmt: skip

    # Lo que dice la voz
    _panel(canvas, px, 836, pw, 212, "VisionX dice", t)
    lineas: list[str] = []
    for m in e.mensajes[-3:]:
        lineas += wrap(m, 21, pw - 48)
    for k, linea in enumerate(lineas[-6:]):
        t.add(linea, (px + 24, 836 + 46 + k * 27), 21, INK)
    return t.draw(canvas)


def _detalle(alta: AltaCamara, paso: Paso) -> str:
    e = alta.estado
    if paso is Paso.RECONOCER and e.dispositivo:
        return "Conocida: perfil cargado" if e.conocida else "No está en los perfiles"
    if paso is Paso.MIRAR and not e.conocida:
        return f"{e.pares} de {e.pares_necesarios} vistas distintas"
    if paso is Paso.ESTIMAR and e.estimacion is not None:
        est = e.estimacion
        if est.confianza <= 0:
            return "Sin red entrenada: valor promedio (90°)"
        return f"{est.campo_horizontal_deg:.0f}° · confianza {est.confianza:.0%}"
    if paso is Paso.ALINEAR and e.filas_despues is not None:
        aplica = ", ".join(e.aplica) if e.aplica else "nada que corregir"
        return f"Corrige: {aplica}"
    if paso is Paso.ESCALA and e.paso is Paso.ESCALA:
        return "Decí los centímetros entre lentes o «saltear»"
    return ""


# --- fuentes ---------------------------------------------------------------------------------


class FuenteSimulada:
    """Un recorrido de data/gt-video como cámara desconocida (ojo derecho de otra unidad)."""

    def __init__(self, variante: str, escena: str = "hotel_0", raiz: Path | None = None) -> None:
        from visionx.gtsynth import vistas
        from visionx.gtsynth.video import _camara_nominal, _hermana

        raiz = raiz or ROOT / "data" / "gt-video"
        self.vs = sorted((v for v in vistas(raiz, variante) if v.escena == escena),
                         key=lambda v: v.meta["indice"])  # fmt: skip
        if not self.vs:
            raise FileNotFoundError(f"no hay recorrido {variante}/{escena} en {raiz}")
        v0 = self.vs[0]
        self._nominal = _camara_nominal(v0.geometria)
        self._real = _hermana(zlib.crc32(f"{v0.escena}/{v0.variante}".encode()), self._nominal)
        self.nombre = "Cámara simulada"
        r = v0.recortar(v0.izq())
        self.resolucion = (r.shape[1], r.shape[0])
        self._k = 0

    def leer(self) -> tuple[np.ndarray, np.ndarray]:
        from visionx.depth.crestereo_adapt.perturb import perturb_right_camera

        v = self.vs[self._k % len(self.vs)]
        self._k += 1
        der = perturb_right_camera(v.der(), self._real, self._nominal, v.geometria).right
        return v.recortar(v.izq()), v.recortar(der)


class FuenteUSB:
    """La cámara USB de la config (side by side), sin rectificar: el alta la calibra."""

    def __init__(self, cfg: dict, dispositivo: str | None) -> None:
        from visionx.camera.perfiles import listar_dispositivos
        from visionx.camera.usb import CameraSettings, UsbStereoSource

        self.src = UsbStereoSource(CameraSettings.from_config(cfg))
        nombres = listar_dispositivos()
        self.nombre = dispositivo or (nombres[0] if nombres else "Cámara USB")
        f = self.src.read()
        self.resolucion = f.size
        self._primero = f

    def leer(self) -> tuple[np.ndarray, np.ndarray]:
        f, self._primero = self._primero or self.src.read(), None
        return f.left, f.right


class FuenteRealSense:
    """La RealSense: se reconoce por su nombre y trae la calibración de fábrica."""

    def __init__(self) -> None:
        from visionx.camera.realsense import RealSenseSource

        self.src = RealSenseSource()
        self.nombre = self.src.nombre
        self.resolucion = (self.src.geometry.width, self.src.geometry.height)

    def leer(self) -> tuple[np.ndarray, np.ndarray]:
        f = self.src.read()
        return f.left, f.right


# --- comando ---------------------------------------------------------------------------------


def cmd_camara_nueva(args: argparse.Namespace) -> int:
    from visionx.shared.config import load_config

    cfg = load_config(args.config) if args.config else load_config()
    if args.fuente.startswith("sim:"):
        fuente = FuenteSimulada(
            args.fuente[4:], args.escena, Path(args.raiz) if args.raiz else None
        )
    elif args.fuente == "usb":
        fuente = FuenteUSB(cfg, args.dispositivo)
    elif args.fuente == "realsense":
        fuente = FuenteRealSense()
    else:
        raise SystemExit("--fuente: sim:<variante de data/gt-video>, usb o realsense")
    decir = _voz(cfg) if args.voz else print

    def nueva() -> AltaCamara:
        alta = AltaCamara(cargar_perfiles(), pares_necesarios=args.pares)
        formato = "realsense" if isinstance(fuente, FuenteRealSense) else "sbs"
        for f in alta.conectar(args.dispositivo or fuente.nombre, formato, fuente.resolucion):
            decir(f)
        return alta

    alta = nueva()
    ventana = not args.sin_ventana
    if ventana:
        cv2.namedWindow("VisionX · Cámara nueva", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VisionX · Cámara nueva", 1600, 900)
    t_escala = None
    while True:
        izq, der = fuente.leer()
        for f in alta.agregar_par(izq, der):
            decir(f)
        if alta.estado.paso is Paso.ESCALA and args.escala is not None:
            t_escala = t_escala or time.monotonic()
            if time.monotonic() - t_escala > 4.0:  # dejar ver el antes/después
                frases = (alta.saltear_escala() if args.escala == "saltear"
                          else alta.separacion(float(args.escala) / 100.0))  # fmt: skip
                for f in frases:
                    decir(f)
        img = render_alta(alta, izq)
        if args.captura and alta.estado.paso is Paso.LISTO:
            cv2.imwrite(args.captura, img)
        if not ventana:
            if alta.estado.paso is Paso.LISTO:
                break
            continue
        cv2.imshow("VisionX · Cámara nueva", img)
        key = cv2.waitKey(100) & 0xFF
        visible = cv2.getWindowProperty("VisionX · Cámara nueva", cv2.WND_PROP_VISIBLE) >= 1
        if key in (ord("q"), 27) or not visible:
            break
        if key == ord("r"):
            alta, t_escala = nueva(), None
        if key == ord("s"):
            for f in alta.saltear_escala():
                decir(f)
    if ventana:
        cv2.destroyAllWindows()
    if alta.estado.paso is Paso.LISTO and not alta.estado.conocida and args.guardar:
        from visionx.camera.perfiles import guardar_local

        print("perfil guardado en", guardar_local(alta.estado.perfil))
    return 0


def _voz(cfg: dict):
    """Habla con el sintetizador de la app (Edge / Piper), una frase detrás de otra."""
    from visionx.shared.contracts import AudioEvent, Priority
    from visionx.voice.engines import VoiceSettings, build_synth
    from visionx.voice.output import AudioOutput, SoundDevicePlayer

    s = VoiceSettings.from_config(cfg)
    out = AudioOutput(build_synth(s), SoundDevicePlayer(s.output_device, s.block_ms,
                                                        s.output_latency_s))  # fmt: skip
    n = iter(range(1, 10**9))

    def decir(texto: str) -> None:
        print(texto)
        out.submit(AudioEvent(next(n), Priority.SHORT, texto, created_ns=out.clock.now_ns(),
                              cancelable=False))  # fmt: skip

    return decir


def register(sub) -> None:
    p = sub.add_parser("camara-nueva", help="dar de alta una cámara estéreo (plug and play)")
    p.add_argument("--fuente", default="sim:video-zed-mini",
                   help="sim:<variante de data/gt-video>, usb o realsense")  # fmt: skip
    p.add_argument("--escena", default="hotel_0", help="escena del recorrido simulado")
    p.add_argument("--raiz", default=None, help="carpeta gt-video (por defecto data/gt-video)")
    p.add_argument("--dispositivo", default=None, help="nombre a reconocer (por defecto, el real)")
    p.add_argument("--pares", type=int, default=8, help="vistas distintas que junta")
    p.add_argument("--escala", default=None, help="centímetros entre lentes, o «saltear»")
    p.add_argument("--voz", action="store_true", help="decir los pasos en voz alta")
    p.add_argument("--sin-ventana", action="store_true")
    p.add_argument("--captura", help="PNG de la pantalla al terminar")
    p.add_argument(
        "--guardar", action="store_true", help="guardar el perfil en configs/camaras/locales"
    )
    p.add_argument("--config")
    p.set_defaults(fn=cmd_camara_nueva)
