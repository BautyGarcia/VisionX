"""Pantalla «Cámara nueva»: dar de alta una cámara estéreo desconocida (plug and play).

El flujo de docs/PLUG-AND-PLAY.md, sección 3, como una máquina de pasos que no depende del
hardware (la alimenta la cámara en vivo, un replay o un recorrido simulado):

    CONECTAR  -> RECONOCER -> (conocida) ------------------------------> LISTO
                           -> (nueva)   MIRAR -> ESTIMAR -> ALINEAR -> ESCALA -> LISTO

- RECONOCER busca el nombre del dispositivo en los perfiles (`camera/perfiles.py`).
- MIRAR junta pares distintos entre sí (la persona mira alrededor).
- ESTIMAR pregunta al estimador de cámara (la red de entrada de Bauty) el campo, el centro y
  la lente. Sin red, queda un valor por defecto marcado «sin estimar».
- ALINEAR rectifica con esa estimación, corre el clásico con todos los pares juntos y mide
  cuánto se desalinean las filas antes y después.
- ESCALA: la separación entre lentes (dicha o de la hoja de datos) o un objeto a distancia
  conocida (`calibrar_distancia`, ya en la app); se puede saltear y queda «aproximada».

Cada paso devuelve las frases que dice la voz; `estado()` da lo que dibuja el dashboard.
"""

from __future__ import annotations

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


class Paso(StrEnum):
    CONECTAR = "conectar"
    RECONOCER = "reconocer"
    MIRAR = "mirar"
    ESTIMAR = "estimar"
    ALINEAR = "alinear"
    ESCALA = "escala"
    LISTO = "listo"


ORDEN = tuple(Paso)
TITULOS = {
    Paso.CONECTAR: "Conectar",
    Paso.RECONOCER: "Reconocer la cámara",
    Paso.MIRAR: "Mirar alrededor",
    Paso.ESTIMAR: "Estimar la cámara",
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
    estimacion: EstimacionCamara | None = None
    aplica: tuple[str, ...] = ()
    filas_antes: Filas | None = None
    filas_despues: Filas | None = None
    escala: str = "-"  # separacion | distancia | perfil | salteada
    nivel: str = "sin-estimar"
    mensajes: list[str] = field(default_factory=list)


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
    ) -> None:
        self.perfiles = perfiles
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
        self.estado = EstadoAlta(Paso.CONECTAR, pares_necesarios=pares_necesarios)

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
        return self._decir(
            "Es una cámara que no conozco. Vamos a calibrarla. "
            "Mirá alrededor despacio durante unos segundos."
        )

    def agregar_par(self, izq: np.ndarray, der: np.ndarray) -> list[str]:
        """Un par de la cámara. En MIRAR guarda los que son distintos de los anteriores."""
        e = self.estado
        if e.paso is not Paso.MIRAR:
            return []
        t = self._reloj()
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
        if e.pares < e.pares_necesarios:
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

    def distancia_conocida(self, ok: bool) -> list[str]:
        """Resultado de `calibrar_distancia` con un objeto a distancia conocida."""
        if self.estado.paso is not Paso.ESCALA:
            return []
        if not ok:
            return self._decir("No pude usar ese objeto. Probá con otro, más cerca del centro.")
        return self._terminar("distancia")

    def saltear_escala(self) -> list[str]:
        if self.estado.paso is not Paso.ESCALA:
            return []
        return self._terminar("salteada")

    # --- internos --------------------------------------------------------------------------

    def _estimar_y_alinear(self) -> list[str]:
        e = self.estado
        e.paso = Paso.ESTIMAR
        est = self.estimador.estimar(self._pares)
        e.estimacion = est
        e.perfil = replace(
            e.perfil, campo_horizontal_deg=est.campo_horizontal_deg, centro_rel=est.centro_rel,
            k1=est.k1, k2=est.k2,
        )  # fmt: skip
        e.paso = Paso.ALINEAR
        calib = calibracion_nominal(e.perfil)
        maps = calib.rectify_maps()
        rect = [calib.rectify(i, d, maps) for i, d in self._pares]
        medio = len(rect) // 2
        e.filas_antes = medir_filas(*rect[medio])
        self.mapas, e.aplica = self._alinear(rect, calib.geometry())
        der = rect[medio][1]
        if self.mapas is not None:
            der = cv2.remap(der, self.mapas[0], self.mapas[1], cv2.INTER_LINEAR)
        e.filas_despues = medir_filas(rect[medio][0], der)
        self.muestra = (rect[medio][0], rect[medio][1], der)
        e.paso = Paso.ESCALA
        frases = [self._frase_estimacion(est), self._frase_filas()]
        frases.append(
            "Para medir en metros necesito la escala. Decime cuántos centímetros hay entre los "
            "dos lentes, o apuntá a un objeto y decí a qué distancia está. Si no sabés, decí "
            "«saltear»."
        )
        return self._decir(*frases)

    def _frase_estimacion(self, est: EstimacionCamara) -> str:
        if est.confianza <= 0:
            return "Todavía no puedo estimar la lente: uso un valor promedio."
        return f"La cámara ve unos {round(est.campo_horizontal_deg)} grados de ancho."

    def _frase_filas(self) -> str:
        a, d = self.estado.filas_antes, self.estado.filas_despues
        if a is None or a.mediana_px is None or d is None or d.mediana_px is None:
            return (
                "No encontré suficientes detalles para alinear los ojos; "
                "probá mirando algo con más textura."
            )
        return (
            f"Alineé los dos ojos: el desfase entre filas pasó de {_coma(a.mediana_px)} "
            f"a {_coma(d.mediana_px)} píxeles."
        )

    def _terminar(self, escala: str) -> list[str]:
        e = self.estado
        e.escala = escala
        est = e.estimacion
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
                   "estimador": est.origen if est else "-"},
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


def _coma(x: float) -> str:
    """Un decimal con coma, como se dice en castellano."""
    return f"{x:.1f}".replace(".", ",")


def _alinear_clasico(pares, geometria) -> tuple[Any, tuple]:
    """Clásico con lente y regla, con los puntos de todos los pares (el de la app)."""
    from visionx.depth.crestereo_adapt.align_apply import Alineador

    return Alineador("clasico", geometria, np.zeros(5))._clasico(pares)
