"""«Calibrá, está a 2 metros» en la app: el objeto corrige el corrimiento y, con dos distancias,
también la escala.

La distancia sale de `Z = f·B / D`, con `D` la disparidad efectiva. Con una cámara estimada (o
otra unidad) quedan dos errores: la escala de `f·B` y un corrimiento de columnas de la unidad,
`D_medida = escala · D_real + corrimiento`. Un objeto a distancia conocida da una ecuación: corrige
el corrimiento (la escala ya la fijaron la focal medida y la separación). El mismo objeto (u otro)
a **otra distancia** da la segunda, y se corrigen las dos (`gtsynth.evaluar.ajustar_ambas`).

`ProfundidadCorregida` envuelve a un estimador de profundidad (continuo y preciso comparten la
misma `CorreccionDistancia`, guardada en anchos de imagen para servir a las dos resoluciones) y
le agrega `calibrar_distancia`, el método opcional que llama `app.calibrate`.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

MIN_PIXELES = 50


@dataclass
class CorreccionDistancia:
    """Escala y corrimiento (en anchos de imagen) medidos con objetos a distancia conocida."""

    escala: float = 1.0
    corrimiento_rel: float = 0.0
    observaciones: list[tuple[float, float]] = field(default_factory=list)  # (medida, real)/ancho

    @property
    def activa(self) -> bool:
        return self.escala != 1.0 or self.corrimiento_rel != 0.0

    modo: str = ""  # "" | "corrimiento" | "ambas"

    sigma_escala: float | None = None  # duda de la focal (movimiento); None = sin previo

    def agregar(self, medida_rel: float, real_rel: float, sigma_rel: float | None = None) -> str:
        """Suma una observación y vuelve a ajustar. Devuelve el modo: corrimiento o ambas.

        Con `sigma_escala` (focal medida con el movimiento) la escala tiene un previo cerca de 1:
        un objeto ruidoso no la arruina (`gtsynth.evaluar.ajustar_ambas`).
        """
        from visionx.gtsynth.evaluar import ajustar_ambas

        self.observaciones.append((float(medida_rel), float(real_rel)))
        ms, rs = zip(*self.observaciones, strict=True)
        ajuste = None
        if len(self.observaciones) >= 2:
            ajuste = ajustar_ambas(list(ms), list(rs), sigma_rel, self.sigma_escala)
            if self.sigma_escala is not None and ajuste is not None:
                rango = max(rs) / max(min(rs), 1e-9)
                if rango < 1.15:  # misma distancia: con el previo es casi solo corrimiento
                    ajuste = None
        if ajuste is not None:
            self.escala, self.corrimiento_rel = ajuste
            self.modo = "ambas"
        else:  # todas a la misma distancia: solo el corrimiento
            self.escala = 1.0
            self.corrimiento_rel = float(np.median(np.subtract(ms, rs)))
            self.modo = "corrimiento"
        return self.modo

    def a_dict(self) -> dict:
        return {"escala": round(self.escala, 5), "corrimiento_rel": round(self.corrimiento_rel, 6),
                "modo": self.modo, "observaciones": [list(map(float, o))
                                                     for o in self.observaciones]}  # fmt: skip

    @classmethod
    def de_dict(cls, d: dict | None) -> CorreccionDistancia:
        if not d:
            return cls()
        obs = [tuple(o) for o in d.get("observaciones", [])]
        return cls(float(d.get("escala", 1.0)), float(d.get("corrimiento_rel", 0.0)), obs,
                   str(d.get("modo", "")))  # fmt: skip


class ProfundidadCorregida:
    """Un estimador de profundidad con la corrección por distancia conocida aplicada."""

    def __init__(
        self, estimador: Any, correccion: CorreccionDistancia,
        al_cambiar: Callable[[CorreccionDistancia], None] | None = None,
    ) -> None:  # fmt: skip
        self._est = estimador
        self.correccion = correccion
        self._al_cambiar = al_cambiar
        self._crudos: OrderedDict[int, Any] = OrderedDict()

    def __getattr__(self, nombre: str) -> Any:  # geometry, model_id, close, etc.
        return getattr(self._est, nombre)

    def _fb(self) -> float:
        g = self._est.geometry
        return float(g.fx * g.baseline_m)

    def predict(self, frame):
        r = self._est.predict(frame)
        self._crudos[frame.frame_id] = r
        while len(self._crudos) > 8:
            self._crudos.popitem(last=False)
        return self.corregir(r)

    def corregir(self, r):
        c = self.correccion
        if not c.activa or r is None or getattr(r, "disparity_px", None) is None:
            return r
        from visionx.gtsynth.evaluar import con_corrimiento, con_escala

        fb = self._fb()
        out = r
        if c.corrimiento_rel:
            out = con_corrimiento(out, c.corrimiento_rel * r.disparity_px.shape[1], fb)
        if c.escala != 1.0:
            out = con_escala(out, c.escala, fb)
        return out

    def calibrar_distancia(self, frame, mask: np.ndarray | None, z_m: float) -> bool:
        """El objeto de `mask` (o el centro de la imagen) está a `z_m` metros."""
        r = self._crudos.get(frame.frame_id)
        if r is None or getattr(r, "disparity_px", None) is None or z_m <= 0:
            return False
        h, w = r.disparity_px.shape
        if mask is None:  # sin máscara: el centro de la imagen
            m = np.zeros((h, w), bool)
            m[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)] = True
        else:
            m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        d = r.disparity_px + r.doffs
        ok = m & r.valid_mask & np.isfinite(d) & (d > 0)
        if ok.sum() < MIN_PIXELES:
            return False
        medida = float(np.median(d[ok]))
        real = self._fb() / float(z_m)
        from visionx.gtsynth.evaluar import SIGMA_D_PX

        self.correccion.agregar(medida / w, real / w, SIGMA_D_PX / w)
        if self._al_cambiar is not None:
            self._al_cambiar(self.correccion)
        return True

    @property
    def sugerencia(self) -> str | None:
        """Qué decir después de calibrar: con una sola distancia, pedir otra para afinar."""
        if self.correccion.modo == "corrimiento":
            return "Para afinar más, alejate o acercate un paso y decime de nuevo la distancia."
        if self.correccion.modo == "ambas":
            return "Con dos distancias distintas corregí la escala y el corrimiento."
        return None


def envolver(depth: Any, precisa: Any, perfil: Any = None, guardar: bool = True):
    """Envuelve el estimador continuo y el del modo preciso con la misma corrección.

    Con un perfil de «Cámara nueva», la corrección se lee de `extra.correccion_distancia` y se
    guarda ahí cada vez que cambia (la próxima vez arranca corregida).
    """
    extra = dict(getattr(perfil, "extra", None) or {})
    corr = CorreccionDistancia.de_dict(extra.get("correccion_distancia"))
    if extra.get("sigma_focal"):  # la focal se midió con el movimiento: previo para la escala
        corr.sigma_escala = max(float(extra["sigma_focal"]), 0.01)

    def al_cambiar(c: CorreccionDistancia) -> None:
        if perfil is None or not guardar or not extra.get("escala"):
            return
        from visionx.camera.perfiles import guardar_local

        nuevo = dataclasses.replace(perfil, extra={**perfil.extra,
                                                   "correccion_distancia": c.a_dict()})  # fmt: skip
        guardar_local(nuevo)

    d = ProfundidadCorregida(depth, corr, al_cambiar) if depth is not None else None
    if precisa is not None and getattr(precisa, "estimator", None) is not None:
        precisa.estimator = ProfundidadCorregida(precisa.estimator, corr)
    return d, precisa
