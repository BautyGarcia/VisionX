"""Pantalla «Cámara nueva»: dar de alta una cámara estéreo desconocida (plug and play).

El flujo de docs/PLUG-AND-PLAY.md, sección 3, como una máquina de pasos que no depende del
hardware (la alimenta la cámara en vivo, un replay o un recorrido simulado):

    CONECTAR  -> RECONOCER -> (conocida) ---------------------------------------> LISTO
                           -> (nueva)   MIRAR -> ESTIMAR -> MEDIR -> ALINEAR -> ESCALA -> LISTO

- RECONOCER busca el nombre del dispositivo en los perfiles (`camera/perfiles.py`).
- MIRAR junta pares distintos entre sí (la persona gira despacio la cámara) y, si hay
  autocalibración, una secuencia de cuadros del ojo izquierdo.
- ESTIMAR pregunta al estimador de cámara (la red de entrada de Bauty) el campo, el centro y
  la lente. Sin red, queda un valor por defecto marcado «sin estimar».
- MEDIR (con `autocalibrar`) mide la focal con el movimiento (`app.autocalibracion`) partiendo
  de la red y combina las dos; si no alcanza, pide girar un poco más (una vez).
- ALINEAR rectifica con esa estimación, corre el clásico con todos los pares juntos y mide
  cuánto se desalinean las filas antes y después.
- ESCALA: la separación entre lentes (dicha o de la hoja de datos) o un objeto a distancia
  conocida («está a dos metros»: con la focal medida, su disparidad da la separación); se
  puede saltear y queda «aproximada».

Cada paso devuelve las frases que dice la voz; `estado()` da lo que dibuja el dashboard.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

import cv2
import numpy as np

from visionx.camera.perfiles import PerfilCamara, calibracion_nominal, reconocer

CAMPO_POR_DEFECTO = 90.0  # sin estimador: un valor medio de las cámaras estéreo comunes
SEPARACION_POR_DEFECTO = 0.06
SIGMA_SIN_RED = 0.3  # duda de log f del valor por defecto (≈ ±30 %)
SIGMA_LENTE_OK = 0.03  # con ±3 % de focal o menos, la lente cuenta como medida


class Paso(StrEnum):
    CONECTAR = "conectar"
    RECONOCER = "reconocer"
    MIRAR = "mirar"
    ESTIMAR = "estimar"
    MEDIR = "medir"
    ALINEAR = "alinear"
    ESCALA = "escala"
    LISTO = "listo"


ORDEN = tuple(Paso)
TITULOS = {
    Paso.CONECTAR: "Conectar",
    Paso.RECONOCER: "Reconocer la cámara",
    Paso.MIRAR: "Girar la cámara",
    Paso.ESTIMAR: "Estimar la cámara",
    Paso.MEDIR: "Medir con el movimiento",
    Paso.ALINEAR: "Alinear los dos ojos",
    Paso.ESCALA: "Escala",
    Paso.LISTO: "Lista",
}


@dataclass(frozen=True, slots=True)
class EstimacionCamara:
    """Lo que devuelve el estimador de cámara para un ojo (los dos se suponen iguales)."""

    campo_horizontal_deg: float
    centro_rel: tuple[float, float] = (0.0, 0.0)
    k1: float = 0.0
    k2: float = 0.0
    confianza: float = 0.0  # 0 = valor por defecto; 1 = estimación segura
    origen: str = "por defecto"
    sigma_log_f: float | None = None  # duda de log f (≈ error relativo de la focal); None = ?


class EstimadorCamara(Protocol):
    """La red de entrada (PLUG-AND-PLAY, sección 4) cumple este contrato."""

    def estimar(self, pares: Sequence[tuple[np.ndarray, np.ndarray]]) -> EstimacionCamara: ...


class EstimadorPorDefecto:
    """Sin red entrenada: campo de 90°, centro en el medio, sin lente; confianza 0."""

    def estimar(self, pares: Sequence[tuple[np.ndarray, np.ndarray]]) -> EstimacionCamara:
        return EstimacionCamara(CAMPO_POR_DEFECTO)


@dataclass
class Filas:
    """Desalineación vertical entre ojos medida con puntos emparejados (píxeles)."""

    mediana_px: float | None
    puntos: int
    dy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pts_izq: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def medir_filas(izq: np.ndarray, der: np.ndarray, max_ancho: int = 960) -> Filas:
    """Mediana de |Δy| de los puntos SIFT emparejados (a lo sumo `max_ancho` de ancho)."""
    from visionx.depth.crestereo_adapt.classic import ClassicParams, match_points

    esc = min(1.0, max_ancho / izq.shape[1])
    if esc < 1.0:
        tam = (round(izq.shape[1] * esc), round(izq.shape[0] * esc))
        izq = cv2.resize(izq, tam, interpolation=cv2.INTER_AREA)
        der = cv2.resize(der, tam, interpolation=cv2.INTER_AREA)
    pl, pr = match_points(izq, der, ClassicParams())
    if len(pl) < 8:
        return Filas(None, len(pl))
    dy = (pr[:, 1] - pl[:, 1]) / esc
    return Filas(float(np.median(np.abs(dy))), len(pl), dy, pl / esc)


_NUMEROS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}  # fmt: skip
_SALTEAR = ("saltear", "saltea", "salteá", "salta", "no se", "no lo se", "ni idea", "omitir")


def interpretar_escala(texto: str) -> float | str | None:
    """Lo que dijo la persona sobre la separación de los lentes: metros, «saltear» o None.

    «seis centímetros», «6,3 cm», «doce», «12 centímetros y medio», «saltear», «no sé».
    Sin unidad se entiende centímetros (es lo que se pregunta).
    """
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9,. ]+", " ", t)
    if any(s in t for s in _SALTEAR):
        return "saltear"
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(mm|milimetros?|cm|centimetros?|m\b|metros?)?", t)
    if m:
        valor = float(m.group(1).replace(",", "."))
        unidad = m.group(2) or "cm"
    else:
        palabras = [w for w in t.split() if w in _NUMEROS]
        if not palabras:
            return None
        valor = float(_NUMEROS[palabras[0]])
        unidad = "mm" if "milimetro" in t else "m" if re.search(r"\bmetros?\b", t) else "cm"
    if "y medio" in t:
        valor += 0.5
    if unidad.startswith("m") and not unidad.startswith(("mm", "mil")):
        metros = valor
    elif unidad.startswith(("mm", "mil")):
        metros = valor / 1000
    else:
        metros = valor / 100
    return metros if 0.005 <= metros <= 1.0 else None


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return (t or "camara")[:40]


@dataclass
class EstadoAlta:
    """Lo que el dashboard necesita para dibujar la pantalla."""

    paso: Paso
    dispositivo: str = ""
    perfil: PerfilCamara | None = None
    conocida: bool = False
    pares: int = 0
    pares_necesarios: int = 8
    cuadros_mov: int = 0  # cuadros de la secuencia para medir con el movimiento
    cuadros_mov_necesarios: int = 0  # 0 = sin autocalibración
    estimacion: EstimacionCamara | None = None
    medida: Any = None  # app.autocalibracion.FocalMedida
    sigma_focal: float | None = None  # duda final de log f (red + movimiento)
    aplica: tuple[str, ...] = ()
    filas_antes: Filas | None = None
    filas_despues: Filas | None = None
    escala: str = "-"  # separacion | distancia | perfil | salteada
    nivel: str = "sin-estimar"
    mensajes: list[str] = field(default_factory=list)
    tiempos_ms: dict[str, float] = field(default_factory=dict)  # por paso, para la pantalla


class AltaCamara:
    """Lleva el flujo de una cámara. Cada método devuelve las frases para la voz."""

    def __init__(
        self,
        perfiles: list[PerfilCamara],
        estimador: EstimadorCamara | None = None,
        pares_necesarios: int = 8,
        separacion_min_s: float = 0.4,
        cambio_min: float = 6.0,
        alinear: Callable[[list[tuple[np.ndarray, np.ndarray]], Any], tuple[Any, tuple]]
        | None = None,
        reloj: Callable[[], float] = time.monotonic,
        autocalibrar: Callable[..., Any] | None = None,
        cuadros_movimiento: int = 60,
        intervalo_movimiento_s: float = 0.1,
    ) -> None:
        self.perfiles = perfiles
        self._autocalibrar = autocalibrar
        self.intervalo_movimiento_s = intervalo_movimiento_s
        self._secuencia: list[np.ndarray] = []
        self._q_secuencia = 1.0
        self._t_secuencia = -1e9
        self._reintentos = 0
        self._calib: Any = None
        self._maps_rect: Any = None
        self._ultimo: tuple[np.ndarray, np.ndarray] | None = None
        self.estimador = estimador or EstimadorPorDefecto()
        self.separacion_min_s = separacion_min_s
        self.cambio_min = cambio_min
        self._alinear = alinear or _alinear_clasico
        self._reloj = reloj
        self._pares: list[tuple[np.ndarray, np.ndarray]] = []
        self._ultimo_t = -1e9
        self._ultima_mini: np.ndarray | None = None
        self.mapas: Any = None  # mapas del clásico para el ojo derecho (None si no aplicó nada)
        # Par de ejemplo para la pantalla: (izquierdo, derecho antes, derecho después).
        self.muestra: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.estado = EstadoAlta(
            Paso.CONECTAR, pares_necesarios=pares_necesarios,
            cuadros_mov_necesarios=cuadros_movimiento if autocalibrar else 0,
        )  # fmt: skip

    # --- pasos -----------------------------------------------------------------------------

    def conectar(self, dispositivo: str, formato: str, resolucion: tuple[int, int]) -> list[str]:
        """La cámara se abrió: reconocerla o arrancar el alta."""
        e = self.estado
        e.dispositivo = dispositivo
        perfil = reconocer(dispositivo, self.perfiles)
        if perfil is not None:
            e.perfil, e.conocida, e.escala, e.nivel = perfil, True, "perfil", perfil.nivel
            e.paso = Paso.LISTO
            return self._decir(f"Reconocí la cámara: {perfil.nombre}. Lista para usar.")
        e.perfil = PerfilCamara(
            id=_slug(dispositivo), nombre=dispositivo or "Cámara nueva", formato=formato,
            resolucion=tuple(resolucion), campo_horizontal_deg=CAMPO_POR_DEFECTO,
            separacion_m=SEPARACION_POR_DEFECTO, nivel="sin-estimar",
            fuente="pantalla Cámara nueva",
        )  # fmt: skip
        e.paso = Paso.MIRAR
        if self._autocalibrar is not None:
            return self._decir(
                "Cámara nueva. Girá la cámara despacio hacia un lado y hacia el otro, "
                "como mirando toda la habitación, unos quince segundos."
            )
        return self._decir("Cámara nueva. Mirá alrededor despacio unos segundos.")

    def agregar_par(self, izq: np.ndarray, der: np.ndarray) -> list[str]:
        """Un par de la cámara. En MIRAR guarda los que son distintos de los anteriores."""
        e = self.estado
        if e.paso is Paso.ESCALA:
            self._ultimo = (izq, der)  # para medir un objeto a distancia conocida
        if e.paso is not Paso.MIRAR:
            return []
        t = self._reloj()
        if e.cuadros_mov_necesarios and e.cuadros_mov < e.cuadros_mov_necesarios:
            if t - self._t_secuencia >= self.intervalo_movimiento_s:
                self._q_secuencia = min(1.0, 960 / izq.shape[1])
                q = self._q_secuencia
                chica = izq if q >= 1 else cv2.resize(
                    izq, (round(izq.shape[1] * q), round(izq.shape[0] * q)),
                    interpolation=cv2.INTER_AREA)  # fmt: skip
                self._secuencia.append(chica)
                self._t_secuencia = t
                e.cuadros_mov = len(self._secuencia)
        if e.pares >= e.pares_necesarios:  # ya están los pares: falta la secuencia
            return self._estimar_y_alinear() if e.cuadros_mov >= e.cuadros_mov_necesarios else []
        mini = cv2.resize(cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        if t - self._ultimo_t < self.separacion_min_s:
            return []
        if self._ultima_mini is not None and float(np.mean(np.abs(mini - self._ultima_mini))) < (
            self.cambio_min
        ):
            return []
        self._pares.append((izq, der))
        self._ultimo_t, self._ultima_mini = t, mini
        e.pares = len(self._pares)
        if e.pares < e.pares_necesarios or e.cuadros_mov < e.cuadros_mov_necesarios:
            return []
        return self._estimar_y_alinear()

    def separacion(self, metros: float) -> list[str]:
        """La persona dijo la separación entre lentes (o se tomó de la hoja de datos)."""
        if self.estado.paso is not Paso.ESCALA:
            return []
        if not 0.005 <= metros <= 1.0:
            return self._decir("Esa separación no parece correcta. Decila en centímetros.")
        e = self.estado
        e.perfil = replace(e.perfil, separacion_m=float(metros))
        return self._terminar("separacion")

    def objeto_a(self, metros: float) -> list[str]:
        """«Está a dos metros»: el objeto del centro, a esa distancia, da la separación.

        Con la focal ya medida (`f`), la disparidad `d` del objeto a `Z` metros da `B = Z·d/f`.
        La disparidad sale de un SGBM sobre el par rectificado y alineado, en el centro.
        """
        if self.estado.paso is not Paso.ESCALA:
            return []
        if not 0.3 <= metros <= 6.0:
            return self._decir("Decime una distancia entre 30 centímetros y 6 metros.")
        par = self._ultimo or (self._pares[-1] if self._pares else None)
        if par is None or self._calib is None:
            return self._decir("Todavía no tengo imagen. Probá de nuevo en un momento.")
        izq, der = self._calib.rectify(par[0], par[1], self._maps_rect)
        if self.mapas is not None:
            der = cv2.remap(der, self.mapas[0], self.mapas[1], cv2.INTER_LINEAR)
        d = disparidad_central(izq, der)
        if d is None:
            return self._decir("No pude medir ese objeto. Probá con otro con más detalle, "
                               "en el centro de la imagen.")  # fmt: skip
        fx = float(self._calib.geometry().fx)
        b = metros * d / fx
        if not 0.005 <= b <= 1.0:
            return self._decir("Esa medida no me cierra. Probá con otro objeto, más cerca.")
        e = self.estado
        e.perfil = replace(e.perfil, separacion_m=float(b))
        self._decir(f"Con ese objeto, los lentes están a unos {_coma(100 * b)} centímetros.")
        return [self.estado.mensajes[-1], *self._terminar("distancia")]

    def distancia_conocida(self, ok: bool) -> list[str]:
        """Resultado de `calibrar_distancia` con un objeto a distancia conocida."""
        if self.estado.paso is not Paso.ESCALA:
            return []
        if not ok:
            return self._decir("No pude usar ese objeto. Probá con otro, más cerca del centro.")
        return self._terminar("distancia")

    def responder(self, texto: str) -> list[str]:
        """La respuesta por voz a la pregunta de la escala («seis centímetros», «saltear»)."""
        if self.estado.paso is not Paso.ESCALA:
            return []
        if re.search(r"\bmetros?\b|\bmetro y medio\b|\bmedio metro\b",
                     unicodedata.normalize("NFKD", texto.lower())):  # fmt: skip
            from visionx.app.calibrate import parse_meters

            z = parse_meters(texto)
            if z is not None and z >= 0.3:  # «está a dos metros»: un objeto
                return self.objeto_a(z)
        r = interpretar_escala(texto)
        if r == "saltear":
            return self.saltear_escala()
        if r is None:
            return self._decir("No entendí. Decí los centímetros, por ejemplo «seis», o «saltear».")
        return self.separacion(float(r))

    def saltear_escala(self) -> list[str]:
        if self.estado.paso is not Paso.ESCALA:
            return []
        return self._terminar("salteada")

    # --- internos --------------------------------------------------------------------------

    def _estimar_y_alinear(self) -> list[str]:
        e = self.estado
        e.paso = Paso.ESTIMAR
        t0 = time.perf_counter()
        est = self.estimador.estimar(self._pares)
        e.tiempos_ms["estimar"] = (time.perf_counter() - t0) * 1000
        e.estimacion = est
        e.perfil = replace(
            e.perfil, campo_horizontal_deg=est.campo_horizontal_deg, centro_rel=est.centro_rel,
            k1=est.k1, k2=est.k2,
        )  # fmt: skip
        frase_medida = None
        if self._autocalibrar is not None:
            frase_medida = self._medir(est)
            if frase_medida is None:  # pidió girar más: vuelve a MIRAR
                return [self.estado.mensajes[-1]]
        e.paso = Paso.ALINEAR
        t0 = time.perf_counter()
        calib = calibracion_nominal(e.perfil)
        maps = calib.rectify_maps()
        self._calib, self._maps_rect = calib, maps
        rect = [calib.rectify(i, d, maps) for i, d in self._pares]
        medio = len(rect) // 2
        e.filas_antes = medir_filas(*rect[medio])
        self.mapas, e.aplica = self._alinear(rect, calib.geometry())
        der = rect[medio][1]
        if self.mapas is not None:
            der = cv2.remap(der, self.mapas[0], self.mapas[1], cv2.INTER_LINEAR)
        e.filas_despues = medir_filas(rect[medio][0], der)
        self.muestra = (rect[medio][0], rect[medio][1], der)
        e.tiempos_ms["alinear"] = (time.perf_counter() - t0) * 1000
        e.paso = Paso.ESCALA
        frases = [frase_medida or self._frase_estimacion(est), self._frase_filas()]
        frases.append(
            "¿Cuántos centímetros hay entre los dos lentes? Si no sabés, apuntá a algo y decime "
            "a cuántos metros está, o decí «saltear»."
        )
        return self._decir(*frases)

    def _medir(self, est: EstimacionCamara) -> str | None:
        """Mide la focal con la secuencia y la combina con la red. None = pidió girar más."""
        from visionx.app.autocalibracion import combinar

        e = self.estado
        e.paso = Paso.MEDIR
        t0 = time.perf_counter()
        w, h = e.perfil.resolucion
        q = self._q_secuencia
        f0 = e.perfil.fx(w)
        cx = ((0.5 + e.perfil.centro_rel[0]) * w - 0.5) * q
        cy = ((0.5 + e.perfil.centro_rel[1]) * h - 0.5) * q
        m = self._autocalibrar(self._secuencia, cx, cy, f0 * q)
        e.medida = m
        e.tiempos_ms["medir"] = (time.perf_counter() - t0) * 1000
        s_red = est.sigma_log_f if est.sigma_log_f else SIGMA_SIN_RED
        if m.ok:
            m = type(m)(m.fx / q, m.sigma_log, m.pares, m.motivo)
        lf, s = combinar(math.log(f0), s_red, m)
        if s > SIGMA_LENTE_OK and self._reintentos < 1:
            self._reintentos += 1
            self._secuencia, e.cuadros_mov, e.paso = [], 0, Paso.MIRAR
            self._t_secuencia = -1e9
            self._decir("No alcancé a medir bien la lente. Girá la cámara un poco más, "
                        "despacio, hacia los dos costados.")  # fmt: skip
            return None
        e.sigma_focal = s
        campo = math.degrees(2 * math.atan(w / (2 * math.exp(lf))))
        e.perfil = replace(e.perfil, campo_horizontal_deg=float(campo))
        if not m.ok:
            return (f"No pude medir la lente con el movimiento; uso la estimación: unos "
                    f"{round(campo)} grados de ancho.")  # fmt: skip
        margen = max(1, round(100 * 1.645 * s))
        return (f"Medí la lente con el movimiento: ve unos {round(campo)} grados de ancho, "
                f"con un margen del {margen} por ciento.")  # fmt: skip

    def _frase_estimacion(self, est: EstimacionCamara) -> str:
        if est.confianza <= 0:
            return "Todavía no sé estimar la lente: uso un valor promedio."
        return f"La cámara ve unos {round(est.campo_horizontal_deg)} grados de ancho."

    def _frase_filas(self) -> str:
        a, d = self.estado.filas_antes, self.estado.filas_despues
        if a is None or a.mediana_px is None or d is None or d.mediana_px is None:
            return (
                "No encontré suficientes detalles para alinear los ojos; "
                "probá mirando algo con más textura."
            )
        return (
            f"Alineé los ojos: el desfase bajó de {_coma(a.mediana_px)} "
            f"a {_coma(d.mediana_px)} píxeles."
        )

    def _terminar(self, escala: str) -> list[str]:
        e = self.estado
        e.escala = escala
        est = e.estimacion
        if e.sigma_focal is not None:  # medida con el movimiento
            lente_ok = e.sigma_focal <= SIGMA_LENTE_OK
        else:
            lente_ok = est is not None and est.confianza >= 0.5
        despues = e.filas_despues.mediana_px if e.filas_despues is not None else None
        filas_ok = despues is not None and despues < 0.75  # 0,0 px también es «bien»
        if escala in ("separacion", "distancia") and lente_ok and filas_ok:
            e.nivel = "precisa"
        elif escala in ("separacion", "distancia") or lente_ok:
            e.nivel = "aproximada"
        else:
            e.nivel = "sin-estimar"
        e.perfil = replace(
            e.perfil, nivel=e.nivel,
            extra={"escala": escala, "alineacion": list(e.aplica),
                   "filas_antes_px": e.filas_antes and e.filas_antes.mediana_px,
                   "filas_despues_px": e.filas_despues and e.filas_despues.mediana_px,
                   "estimador": est.origen if est else "-",
                   "sigma_focal": e.sigma_focal,
                   "movimiento": None if e.medida is None else
                   {"ok": e.medida.ok, "pares": e.medida.pares, "motivo": e.medida.motivo}},
        )  # fmt: skip
        e.paso = Paso.LISTO
        texto = {
            "precisa": "Listo. Las distancias van a ser precisas.",
            "aproximada": "Listo. Las distancias van a ser aproximadas.",
            "sin-estimar": "Listo, pero sin escala: voy a dar direcciones y no metros.",
        }[e.nivel]
        return self._decir(texto)

    def _decir(self, *frases: str) -> list[str]:
        self.estado.mensajes.extend(frases)
        return list(frases)


def disparidad_central(izq: np.ndarray, der: np.ndarray, caja: float = 0.2,
                       max_ancho: int = 960) -> float | None:  # fmt: skip
    """Mediana de la disparidad (px de la imagen dada) en el centro, con SGBM. None si no hay."""
    q = min(1.0, max_ancho / izq.shape[1])
    if q < 1:
        tam = (round(izq.shape[1] * q), round(izq.shape[0] * q))
        izq = cv2.resize(izq, tam, interpolation=cv2.INTER_AREA)
        der = cv2.resize(der, tam, interpolation=cv2.INTER_AREA)
    gi = izq if izq.ndim == 2 else cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY)
    gd = der if der.ndim == 2 else cv2.cvtColor(der, cv2.COLOR_BGR2GRAY)
    nd = max(16, min(192, gi.shape[1] // 4 // 16 * 16))  # el centro tiene que quedar a la vista
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2)  # fmt: skip
    disp = sgbm.compute(gi, gd).astype(np.float32) / 16.0
    h, w = disp.shape
    y0, y1 = int(h * (0.5 - caja / 2)), int(h * (0.5 + caja / 2))
    x0, x1 = int(w * (0.5 - caja / 2)), int(w * (0.5 + caja / 2))
    c = disp[y0:y1, x0:x1]
    c = c[c > 0.5]
    if c.size < 200:
        return None
    return float(np.median(c)) / q


def _coma(x: float) -> str:
    """Un decimal con coma, como se dice en castellano."""
    return f"{x:.1f}".replace(".", ",")


def _alinear_clasico(pares, geometria) -> tuple[Any, tuple]:
    """Clásico con lente y regla, con los puntos de todos los pares (el de la app)."""
    from visionx.depth.crestereo_adapt.align_apply import Alineador

    return Alineador("clasico", geometria, np.zeros(5))._clasico(pares)
