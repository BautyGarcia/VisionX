"""Pantalla «Cámara nueva» del dashboard y comando para correrla.

    python -m visionx.app camara-nueva --fuente sim:video-zed-mini [--escala 6.3|saltear] [--voz]
    python -m visionx.app camara-nueva --fuente usb [--dispositivo "Nombre"] [--voz]
    python -m visionx.app camara-nueva --fuente realsense [--voz]

Con `sim:<variante>` usa un recorrido de `data/gt-video` como si fuera una cámara desconocida
(el ojo derecho de «otra unidad», como en los videos del stand): sirve para mostrar el flujo
en la expo sin la cámara. La escala se puede decir por voz o dejar fija con `--escala` (cm).

Teclas: q / Esc = salir, r = empezar de nuevo, s = saltear la escala. Con `--voz`, espacio =
hablar (apretar, decir «seis centímetros» o «saltear», apretar de nuevo).
"""

from __future__ import annotations

import argparse
import threading
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


def _ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000:.1f} s".replace(".", ",") if ms >= 1000 else f"{ms:.0f} ms"


def _metricas(canvas, t, alta: AltaCamara, x: int, y: int, w: int, h: int, fps: float | None):
    """Rendimiento, alineación y cámara en números (lo que pide el stand)."""
    from visionx.gtsynth.mosaico import PARAMETROS

    e = alta.estado
    _panel(canvas, x, y, w, h, "Métricas", t)
    a, d = e.filas_antes, e.filas_despues
    puntos = "—" if a is None else f"{a.puntos} → {d.puntos if d else '—'}"
    desfase = "—"
    if a is not None and a.mediana_px is not None and d is not None and d.mediana_px is not None:
        desfase = f"{a.mediana_px:.1f} → {d.mediana_px:.1f} px".replace(".", ",")
    corr: list[str] = []
    for p in e.aplica:
        nombre = PARAMETROS.get(p, p)
        if nombre not in corr:
            corr.append(nombre)
    res = f"{e.perfil.resolucion[0]}×{e.perfil.resolucion[1]}" if e.perfil else "—"
    filas = [
        ("Cámara", f"{res} · {fps:.1f} fps".replace(".", ",") if fps else res),
        ("Puntos emparejados", puntos),
        ("Desfase de filas", desfase),
        ("Corrige", " · ".join(corr) if corr else "—"),
        ("Tiempos", f"estimar {_ms(e.tiempos_ms.get('estimar'))} · "
                    f"alinear {_ms(e.tiempos_ms.get('alinear'))}"),
    ]  # fmt: skip
    for k, (etiqueta, valor) in enumerate(filas):
        t.add(etiqueta, (x + 22, y + 46 + k * 31), 15, MUTED)
        t.add(valor, (x + 190, y + 45 + k * 31), 16, INK)


def render_alta(alta: AltaCamara, izq: np.ndarray | None, fps: float | None = None) -> np.ndarray:
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
        salteables = (Paso.MIRAR, Paso.ESTIMAR, Paso.MEDIR, Paso.ALINEAR, Paso.ESCALA)
        if e.conocida and paso in salteables:
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
        yy += 70

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

    # Lo que dice la voz y las métricas
    vw = int(pw * 0.55)
    _panel(canvas, px, 836, vw, 212, "VisionX dice", t)
    lineas: list[str] = []
    for m in e.mensajes[-3:]:
        lineas += wrap(m, 19, vw - 48)
    for k, linea in enumerate(lineas[-6:]):
        t.add(linea, (px + 24, 836 + 46 + k * 26), 19, INK)
    _metricas(canvas, t, alta, px + vw + 16, 836, pw - vw - 16, 212, fps)
    return t.draw(canvas)


def _detalle(alta: AltaCamara, paso: Paso) -> str:
    e = alta.estado
    if paso is Paso.RECONOCER and e.dispositivo:
        return "Conocida: perfil cargado" if e.conocida else "No está en los perfiles"
    if paso is Paso.MIRAR and not e.conocida:
        if e.cuadros_mov_necesarios:
            pct = min(100, round(100 * e.cuadros_mov / e.cuadros_mov_necesarios))
            return f"{pct} % del giro · {e.pares} de {e.pares_necesarios} vistas"
        return f"{e.pares} de {e.pares_necesarios} vistas distintas"
    if paso is Paso.MEDIR:
        if e.cuadros_mov_necesarios == 0:
            return "Sin autocalibración"
        if e.sigma_focal is not None:
            m = e.medida
            fuente = "red + movimiento" if m is not None and m.ok else "solo la red"
            return f"Focal ±{100 * 1.645 * e.sigma_focal:.0f} % · {fuente}".replace(".", ",")
    if paso is Paso.ESTIMAR and e.estimacion is not None:
        est = e.estimacion
        if est.confianza <= 0:
            return "Sin red entrenada: valor promedio (90°)"
        if est.sigma_log_f:
            margen = f"{100 * 1.645 * est.sigma_log_f:.0f}"
            return f"Red: {est.campo_horizontal_deg:.0f}° ± {margen} %"
        return f"{est.campo_horizontal_deg:.0f}° · confianza {est.confianza:.0%}"
    if paso is Paso.ALINEAR and e.filas_despues is not None:
        from visionx.gtsynth.mosaico import PARAMETROS

        nombres = list(dict.fromkeys(PARAMETROS.get(p, p) for p in e.aplica))
        aplica = " · ".join(nombres) if nombres else "nada que corregir"
        return f"Corrige: {aplica}"
    if paso is Paso.ESCALA and e.paso is Paso.ESCALA:
        return "Centímetros entre lentes, «está a N metros» o «saltear»"
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
    oido = _oido(cfg) if args.voz else None  # (micrófono, voz a texto) para responder la escala
    oidos: list[str] = []  # textos reconocidos, los consume el lazo principal
    grabando = False

    estimador = _estimador(args.red)
    autocalibrar = None
    if not args.sin_movimiento:
        from visionx.app.autocalibracion import focal_por_movimiento

        autocalibrar = focal_por_movimiento
    # Los recorridos simulados ya vienen espaciados: cada cuadro cuenta.
    intervalo = 0.0 if isinstance(fuente, FuenteSimulada) else 0.1

    def nueva() -> AltaCamara:
        alta = AltaCamara(
            cargar_perfiles(), estimador=estimador, pares_necesarios=args.pares,
            autocalibrar=autocalibrar, cuadros_movimiento=args.cuadros_movimiento,
            intervalo_movimiento_s=intervalo,
        )  # fmt: skip
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
    t_prev: float | None = None
    fps: float | None = None
    while True:
        izq, der = fuente.leer()
        for f in alta.agregar_par(izq, der):
            decir(f)
        if alta.estado.paso is Paso.ESCALA and args.escala is not None:
            t_escala = t_escala or time.monotonic()
            if time.monotonic() - t_escala > 4.0:  # dejar ver el antes/después
                try:  # «6» = centímetros entre lentes; si no, como si se dijera
                    frases = alta.separacion(float(args.escala.replace(",", ".")) / 100.0)
                except ValueError:
                    frases = alta.responder(args.escala)
                for f in frases:
                    decir(f)
        ahora = time.monotonic()
        if t_prev is not None:
            fps = (
                1 / max(ahora - t_prev, 1e-3)
                if fps is None
                else 0.9 * fps + 0.1 / max(ahora - t_prev, 1e-3)
            )
        t_prev = ahora
        img = render_alta(alta, izq, fps)
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
        if key == ord(" ") and oido is not None and alta.estado.paso is Paso.ESCALA:
            mic, stt = oido
            if not grabando:
                mic.start()
                grabando = True
            else:
                audio = mic.stop()
                grabando = False

                def reconocer(audio=audio, stt=stt) -> None:
                    oidos.append(stt.transcribe(audio).text)

                threading.Thread(target=reconocer, daemon=True).start()
        while oidos:
            texto = oidos.pop(0)
            print(f"  [oído] {texto}")
            for f in alta.responder(texto):
                decir(f)
    if ventana:
        cv2.destroyAllWindows()
    salida = getattr(decir, "salida", None)
    if salida is not None:  # que termine de hablar antes de salir
        salida.wait_idle(60.0)
        salida.close()
    if alta.estado.paso is Paso.LISTO and not alta.estado.conocida and args.guardar:
        from visionx.camera.perfiles import guardar_local

        perfil = alta.estado.perfil
        print("perfil guardado en", guardar_local(perfil))
        print(f"para usarla: python -m visionx.app ejecutar --fuente usb --camara {perfil.id}")
    return 0


def _estimador(red: str | None):
    """La red de entrada si hay checkpoint (y su código); si no, el valor por defecto."""
    if red == "ninguna":
        return None
    from visionx.app.estimador_red import EstimadorRed, buscar_checkpoint

    ruta = Path(red) if red else buscar_checkpoint()
    if ruta is None or not ruta.exists():
        print("  sin checkpoint de la red de entrada: uso el valor por defecto (90°)")
        return None
    try:
        return EstimadorRed(ruta)
    except ImportError as exc:  # falta `depth.camara_net` (#90) o torch
        print(f"  no pude cargar la red de entrada ({exc}): uso el valor por defecto")
        return None


def _oido(cfg: dict):
    """Micrófono con botón (espacio) y voz a texto de la app, para responder la escala."""
    from visionx.voice.engines import PushToTalk, VoiceSettings, build_transcriber

    s = VoiceSettings.from_config(cfg)
    return PushToTalk(s.input_device, s.max_record_s), build_transcriber(s)


def _voz(cfg: dict):
    """Habla con el sintetizador de la app (Edge / Piper), una frase detrás de otra."""
    from visionx.shared.contracts import AudioEvent, Priority
    from visionx.voice.engines import VoiceSettings, build_synth
    from visionx.voice.output import AudioOutput, SoundDevicePlayer

    s = VoiceSettings.from_config(cfg)

    def anotar(r) -> None:  # cómo salió cada frase (para probar sin escuchar)
        if r.outcome is not None and r.ended_ns:
            dur = (r.ended_ns - (r.started_ns or r.ended_ns)) / 1e9
            extra = (f" ({r.error})" if r.error else "") + (" con cortes" if r.cortes else "")
            print(f"  [voz] {r.outcome.name.lower()} {dur:.1f} s{extra}")

    player = SoundDevicePlayer(s.output_device, s.block_ms, s.output_latency_s)
    out = AudioOutput(build_synth(s), player, on_record=anotar)
    n = iter(range(1, 10**9))

    def decir(texto: str) -> None:
        print(texto)
        out.submit(AudioEvent(next(n), Priority.SHORT, texto, created_ns=out.clock.now_ns(),
                              cancelable=False))  # fmt: skip

    decir.salida = out  # type: ignore[attr-defined]
    return decir


def register(sub) -> None:
    p = sub.add_parser("camara-nueva", help="dar de alta una cámara estéreo (plug and play)")
    p.add_argument("--fuente", default="sim:video-zed-mini",
                   help="sim:<variante de data/gt-video>, usb o realsense")  # fmt: skip
    p.add_argument("--escena", default="hotel_0", help="escena del recorrido simulado")
    p.add_argument("--raiz", default=None, help="carpeta gt-video (por defecto data/gt-video)")
    p.add_argument("--dispositivo", default=None, help="nombre a reconocer (por defecto, el real)")
    p.add_argument("--pares", type=int, default=8, help="vistas distintas que junta")
    p.add_argument("--escala", default=None,
                   help="centímetros entre lentes, «está a 2 metros» o «saltear»")  # fmt: skip
    p.add_argument("--red", default=None,
                   help="checkpoint de la red de entrada (por defecto outputs/camara_net/e2-v2; "
                   "«ninguna» = valor por defecto)")  # fmt: skip
    p.add_argument("--sin-movimiento", action="store_true",
                   help="no medir la lente con el movimiento (solo la red)")  # fmt: skip
    p.add_argument("--cuadros-movimiento", type=int, default=60,
                   help="cuadros del giro para medir la lente")  # fmt: skip
    p.add_argument("--voz", action="store_true", help="decir los pasos en voz alta")
    p.add_argument("--sin-ventana", action="store_true")
    p.add_argument("--captura", help="PNG de la pantalla al terminar")
    p.add_argument(
        "--guardar", action="store_true", help="guardar el perfil en configs/camaras/locales"
    )
    p.add_argument("--config")
    p.set_defaults(fn=cmd_camara_nueva)
