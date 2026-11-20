"""Panel de audio de la ventana: elegir micrófono y salida mientras la app corre (tecla «a»).

Hoy los dispositivos se fijan por nombre en `audio.input_device` / `audio.output_device` de la
config. Este panel deja cambiarlos en vivo, sin reiniciar, y comprueba que funcionen:

- una barra con el nivel del micrófono que está bajo el cursor, que avisa «sin señal» si llega
  silencio digital exacto (lo que entrega macOS cuando la terminal no tiene permiso de
  micrófono: no da error, da ceros);
- un bip por la salida recién elegida.

Lo elegido vale solo para esta sesión: al reiniciar manda otra vez la config.

Se divide en partes que se prueban sin audio: `listar` (dispositivos), `PanelAudio` (estado y
teclas, sin dibujar), `MedidorNivel` (el micrófono abierto para la barra) y `dibujar_panel`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Códigos de las flechas según el sistema (`cv2.waitKeyEx`): macOS, Windows y Linux (GTK).
# Las flechas no tienen un código común; por eso también sirven «w» y «s».
_ARRIBA = {63232, 2490368, 65362, ord("w")}
_ABAJO = {63233, 2621440, 65364, ord("s")}
# Medio segundo de ceros exactos seguidos = el sistema no nos da el micrófono.
SIN_SENAL_S = 0.5
PISO_DB = -60.0  # la barra empieza acá (silencio de una habitación ~ -55 a -45 dBFS)


def tecla(codigo: int) -> str | None:
    """Nombre de una tecla de `cv2.waitKeyEx`: flechas, tab, enter, esc o el carácter."""
    if codigo < 0:
        return None
    if codigo in _ARRIBA:
        return "arriba"
    if codigo in _ABAJO:
        return "abajo"
    base = codigo & 0xFF
    if base == 9:
        return "tab"
    if base in (10, 13):
        return "enter"
    if base == 27:
        return "esc"
    return chr(base) if 32 <= base < 127 else None


@dataclass(frozen=True)
class Dispositivo:
    indice: int  # índice de PortAudio en esta sesión
    nombre: str


def listar(sd: Any = None) -> tuple[list[Dispositivo], list[Dispositivo]]:
    """Micrófonos y salidas del sistema de audio por defecto (Core Audio en la Mac).

    En Windows cada dispositivo aparece una vez por API (MME, DirectSound, WASAPI...): se
    muestran solo los de la API del dispositivo por defecto, para no repetir nombres.
    """
    if sd is None:
        import sounddevice as sd
    todos = sd.query_devices()
    try:
        api = sd.query_devices(kind="input")["hostapi"]
    except Exception:  # sin entrada por defecto
        api = None
    entradas, salidas = [], []
    for k, d in enumerate(todos):
        if api is not None and d.get("hostapi") != api:
            continue
        if d.get("max_input_channels", 0) > 0:
            entradas.append(Dispositivo(k, str(d["name"])))
        if d.get("max_output_channels", 0) > 0:
            salidas.append(Dispositivo(k, str(d["name"])))
    return entradas, salidas


def posicion(lista: list[Dispositivo], configurado: int | str | None, defecto: int | None) -> int:
    """Lugar en `lista` del dispositivo configurado (índice o parte del nombre), o del de
    defecto del sistema; 0 si no aparece."""
    for k, d in enumerate(lista):
        if isinstance(configurado, int) and d.indice == configurado:
            return k
        if isinstance(configurado, str) and configurado.lower() in d.nombre.lower():
            return k
    for k, d in enumerate(lista):
        if d.indice == defecto:
            return k
    return 0


@dataclass
class PanelAudio:
    """Estado del panel y respuesta a las teclas. No toca audio: devuelve qué aplicar."""

    entradas: list[Dispositivo]
    salidas: list[Dispositivo]
    mic: int = 0  # posición del micrófono en uso dentro de `entradas`
    salida: int = 0  # posición de la salida en uso dentro de `salidas`
    abierto: bool = False
    columna: str = "mic"  # "mic" o "salida"
    cursor: dict[str, int] = field(default_factory=lambda: {"mic": 0, "salida": 0})
    error: str | None = None

    def abrir(self) -> None:
        self.abierto, self.columna, self.error = True, "mic", None
        self.cursor = {"mic": self.mic, "salida": self.salida}

    def cerrar(self) -> None:
        self.abierto = False

    def bajo_cursor(self) -> Dispositivo | None:
        """El micrófono marcado por el cursor: es el que mide la barra de nivel."""
        if not self.entradas:
            return None
        return self.entradas[self.cursor["mic"]]

    def manejar(self, nombre: str | None) -> tuple[str, Dispositivo | None] | None:
        """Una tecla con el panel abierto. Devuelve la acción para la app, si hay una:
        («mic», d) / («salida», d) para usar ese dispositivo, («cerrar», None) o
        («medir», d) cuando el cursor cambió de micrófono."""
        if nombre in ("a", "esc"):
            self.cerrar()
            return ("cerrar", None)
        if nombre == "tab":
            self.columna = "salida" if self.columna == "mic" else "mic"
            return None
        lista = self.entradas if self.columna == "mic" else self.salidas
        if not lista:
            return None
        if nombre in ("arriba", "abajo"):
            paso = -1 if nombre == "arriba" else 1
            self.cursor[self.columna] = (self.cursor[self.columna] + paso) % len(lista)
            self.error = None
            return ("medir", self.bajo_cursor()) if self.columna == "mic" else None
        if nombre == "enter":
            return (self.columna, lista[self.cursor[self.columna]])
        return None

    def usar(self, columna: str, dispositivo: Dispositivo) -> None:
        """La app aplicó el cambio: queda marcado como en uso."""
        lista = self.entradas if columna == "mic" else self.salidas
        setattr(self, "mic" if columna == "mic" else "salida", lista.index(dispositivo))
        self.error = None


class MedidorNivel:
    """Abre un micrófono solo para la barra de nivel. El callback guarda el nivel y nada más."""

    def __init__(self, sd: Any = None) -> None:
        if sd is None:
            import sounddevice as sd
        self._sd = sd
        self._stream = None
        self._lock = threading.Lock()
        self._rms = 0.0
        self._ultimo_sonido = time.monotonic()
        self._abierto_en = time.monotonic()

    def _callback(self, indata, frames, time_info, status) -> None:  # hilo de PortAudio
        block = indata[:, 0]
        self._rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        if np.any(block != 0):
            self._ultimo_sonido = time.monotonic()

    def abrir(self, dispositivo: Dispositivo) -> None:
        """Cierra el anterior y abre este. Lanza el error de PortAudio si no abre."""
        self.cerrar()
        stream = self._sd.InputStream(
            channels=1, dtype="float32", device=dispositivo.indice, callback=self._callback
        )
        stream.start()
        with self._lock:
            self._stream = stream
            self._rms = 0.0
            self._abierto_en = self._ultimo_sonido = time.monotonic()

    def cerrar(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def nivel(self) -> tuple[float | None, bool]:
        """(nivel en dBFS o None si está cerrado, sin_senal)."""
        if self._stream is None:
            return None, False
        db = 20.0 * float(np.log10(max(self._rms, 1e-9)))
        ahora = time.monotonic()
        sin_senal = ahora - self._abierto_en > SIN_SENAL_S and (
            ahora - self._ultimo_sonido > SIN_SENAL_S
        )
        return db, sin_senal


def dibujar_panel(
    canvas: np.ndarray, panel: PanelAudio, nivel_db: float | None, sin_senal: bool
) -> np.ndarray:
    """Dibuja el panel centrado sobre `canvas` (stand o diagnóstico) y devuelve la imagen."""
    from visionx.app.stand import ACCENT, BORDER, FAINT, INK, LEVEL_COLOR, MUTED, PANEL, Texts
    from visionx.app.stand import _rounded as rounded

    alto_img, ancho_img = canvas.shape[:2]
    s = min(ancho_img / 1920, alto_img / 1080)  # el diagnóstico es más chico que el stand
    fila = int(34 * s)
    # Encabezado 56 + dos títulos 36 + barra de nivel 58 + error y ayuda 100 (px a 1920x1080).
    filas = max(len(panel.entradas), 1) + max(len(panel.salidas), 1)
    w, h = int(760 * s), int((56 + 2 * 36 + 58 + 100) * s) + fila * filas
    x, y = (ancho_img - w) // 2, (alto_img - h) // 2
    # Fondo oscurecido para que el panel se lea sobre la cámara.
    canvas[:] = (canvas * 0.45).astype(np.uint8)
    rounded(canvas, x, y, x + w, y + h, PANEL, 16)
    rounded(canvas, x, y, x + w, y + h, BORDER, 16, fill=False)
    t = Texts()
    t.add("AUDIO", (x + int(24 * s), y + int(18 * s)), int(15 * s), MUTED, True)
    rojo = LEVEL_COLOR["no-segura"]
    cy = y + int(56 * s)
    for columna, titulo, lista, en_uso in (
        ("mic", "Micrófono", panel.entradas, panel.mic),
        ("salida", "Salida", panel.salidas, panel.salida),
    ):
        activa = panel.columna == columna
        t.add(titulo, (x + int(24 * s), cy), int(20 * s), ACCENT if activa else INK, True)
        cy += int(36 * s)
        if not lista:
            t.add("(ninguno disponible)", (x + int(40 * s), cy), int(17 * s), FAINT)
            cy += fila
        for k, d in enumerate(lista):
            if activa and k == panel.cursor[columna]:
                rounded(canvas, x + int(16 * s), cy - int(4 * s), x + w - int(16 * s),
                        cy + fila - int(6 * s), BORDER, 8)  # fmt: skip
            marca = "●" if k == en_uso else " "
            t.add(f"{marca}  {d.nombre}", (x + int(32 * s), cy), int(18 * s),
                  INK if k == en_uso else MUTED, k == en_uso)  # fmt: skip
            cy += fila
        if columna == "mic":
            # Barra de nivel del micrófono bajo el cursor.
            bx, bw, bh = x + int(32 * s), w - int(64 * s), int(12 * s)
            rounded(canvas, bx, cy, bx + bw, cy + bh, BORDER, 6)
            if nivel_db is not None:
                frac = min(max((nivel_db - PISO_DB) / -PISO_DB, 0.0), 1.0)
                if frac > 0:
                    rounded(canvas, bx, cy, bx + int(bw * frac), cy + bh,
                            rojo if sin_senal else LEVEL_COLOR["precisa"], 6)  # fmt: skip
            nota = (
                "sin señal: revisá el permiso de micrófono de la terminal"
                if sin_senal
                else "nivel del micrófono marcado"
                if nivel_db is not None
                else "micrófono cerrado"
            )
            t.add(nota, (bx, cy + int(18 * s)), int(15 * s), rojo if sin_senal else FAINT)
            cy += int(58 * s)
    if panel.error:
        t.add(panel.error, (x + int(24 * s), y + h - int(64 * s)), int(15 * s), rojo)
    ayuda = "flechas o w/s: mover   Tab: micrófono/salida   Enter: usar   a o Esc: cerrar"
    t.add(ayuda, (x + int(24 * s), y + h - int(36 * s)), int(14 * s), FAINT)
    return t.draw(canvas)
