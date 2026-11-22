"""Alineación en vivo: alinear los ojos al arrancar y volver a alinear si la cámara se golpea.

Una cámara dada de alta con «Cámara nueva» tiene su perfil (focal, centro, lente, separación),
pero los dos ojos de esa unidad no son perfectamente paralelos: el clásico lo corrige con pares
de la escena. `FuenteAlineada` envuelve a la fuente rectificada y:

1. **Al arrancar** junta `pares` distintos, corre el clásico (en un hilo, para no frenar la
   cámara) y desde ahí corrige el ojo derecho de cada cuadro.
2. **En uso** mide cada `cada_s` segundos el desfase vertical entre ojos (mediana de |Δy| de los
   puntos emparejados). Si `seguidas` mediciones pasan `umbral_px` (la cámara se golpeó, se
   aflojó el soporte, se calentó), vuelve a juntar pares y a alinear, sola.

`estado` dice qué está pasando («alineando», «alineada», «realineando») para la pantalla y la
voz; `historial` guarda cada medición para la BITACORA.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np


class FuenteAlineada:
    def __init__(
        self,
        fuente: Any,
        pares: int = 8,
        cada_s: float = 5.0,
        umbral_px: float = 1.0,
        seguidas: int = 2,
        cambio_min: float = 6.0,
        separacion_min_s: float = 0.3,
        en_hilo: bool = True,
        reloj: Callable[[], float] = time.monotonic,
        alinear: Callable[..., tuple[Any, tuple]] | None = None,
        medir: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    ) -> None:
        self._fuente = fuente
        self.geometry = fuente.geometry
        self.calibration = getattr(fuente, "calibration", None)
        self.pares_necesarios = pares
        self.cada_s = cada_s
        self.umbral_px = umbral_px
        self.seguidas = seguidas
        self.cambio_min = cambio_min
        self.separacion_min_s = separacion_min_s
        self.en_hilo = en_hilo
        self._reloj = reloj
        self._alinear = alinear or _alinear_clasico
        self._medir = medir or _medir_filas
        self._lock = threading.Lock()
        self._mapas: Any = None
        self._pares: list[tuple[np.ndarray, np.ndarray]] = []
        self._ultima_mini: np.ndarray | None = None
        self._ultimo_t = -1e9
        self._ultima_medicion_t = -1e9
        self._altas = 0
        self._trabajando = False
        self.estado = "alineando"
        self.realineaciones = 0
        self.ultimo_desfase_px: float | None = None
        self.historial: list[dict] = []

    @property
    def original_frame(self):  # lo usa el OCR (RectifiedSource)
        return getattr(self._fuente, "original_frame", None)

    # --- lectura -------------------------------------------------------------------------

    def read(self):
        frame = self._fuente.read()
        if frame is None:
            return None
        t = self._reloj()
        with self._lock:
            mapas, juntando = self._mapas, self.estado in ("alineando", "realineando")
        if juntando and not self._trabajando:
            self._juntar(frame.left, frame.right, t)
        der = frame.right if mapas is None else cv2.remap(frame.right, mapas[0], mapas[1],
                                                          cv2.INTER_LINEAR)  # fmt: skip
        if (not juntando and not self._trabajando
                and t - self._ultima_medicion_t >= self.cada_s):  # fmt: skip
            self._ultima_medicion_t = t
            self._lanzar(self._vigilar, frame.left, der, t)
        return frame if mapas is None else dataclasses.replace(frame, right=der)

    def close(self) -> None:
        self._fuente.close()

    # --- internos ------------------------------------------------------------------------

    def _juntar(self, izq: np.ndarray, der: np.ndarray, t: float) -> None:
        if t - self._ultimo_t < self.separacion_min_s:
            return
        mini = cv2.resize(cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        if self._ultima_mini is not None and float(np.mean(np.abs(mini - self._ultima_mini))) < (
            self.cambio_min
        ):
            return
        self._pares.append((izq.copy(), der.copy()))
        self._ultimo_t, self._ultima_mini = t, mini
        if len(self._pares) >= self.pares_necesarios:
            pares, self._pares = self._pares, []
            self._lanzar(self._calcular, pares, t)

    def _lanzar(self, fn, *args) -> None:
        self._trabajando = True
        if self.en_hilo:
            threading.Thread(target=fn, args=args, daemon=True).start()
        else:
            fn(*args)

    def _calcular(self, pares, t: float) -> None:
        try:
            mapas, aplica = self._alinear(pares, self.geometry)
            antes = self._medir(*pares[len(pares) // 2])
            der = pares[len(pares) // 2][1]
            if mapas is not None:
                der = cv2.remap(der, mapas[0], mapas[1], cv2.INTER_LINEAR)
            despues = self._medir(pares[len(pares) // 2][0], der)
            with self._lock:
                self._mapas = mapas
                self.estado = "alineada"
                self._altas += 1
                self.ultimo_desfase_px = despues.mediana_px
            self.historial.append({"t": round(t, 2), "evento": "alineada", "aplica": list(aplica),
                                   "antes_px": antes.mediana_px,
                                   "despues_px": despues.mediana_px})  # fmt: skip
        finally:
            self._trabajando = False

    def _vigilar(self, izq: np.ndarray, der: np.ndarray, t: float) -> None:
        try:
            f = self._medir(izq, der)
            self.ultimo_desfase_px = f.mediana_px
            self.historial.append({"t": round(t, 2), "evento": "medida", "px": f.mediana_px})
            if f.mediana_px is None:  # sin puntos (pared lisa, oscuridad): no se decide nada
                return
            ultimas = [h["px"] for h in self.historial if h["evento"] == "medida"]
            ultimas = [p for p in ultimas[-self.seguidas :] if p is not None]
            if len(ultimas) == self.seguidas and min(ultimas) > self.umbral_px:
                with self._lock:
                    self.estado = "realineando"
                    self.realineaciones += 1
                    self._ultima_mini = None
                self.historial.append({"t": round(t, 2), "evento": "desalineada",
                                       "px": f.mediana_px})  # fmt: skip
        finally:
            self._trabajando = False


def _alinear_clasico(pares, geometria) -> tuple[Any, tuple]:
    from visionx.depth.crestereo_adapt.align_apply import Alineador

    return Alineador("clasico", geometria, np.zeros(5))._clasico(pares)


def _medir_filas(izq: np.ndarray, der: np.ndarray):
    from visionx.app.camara_nueva import medir_filas

    return medir_filas(izq, der)
