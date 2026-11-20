"""Alinear el ojo derecho de varios pares de una misma cámara antes de CREStereo.

Junta en un solo lugar las correcciones que T12 comparó en `hermanas`, para usarlas fuera de
ese experimento (por ejemplo, en `gtsynth evaluar`):

    ninguna       el par tal como llegó (rectificado con la calibración de la ELP)
    clasico       clásico con términos de lente y regla (`lens.py`), con los puntos de todos
                  los pares juntos
    red           red de alineación, un solo campo por cámara (mediana de los campos de cada par)
    clasico+red   la red mira los pares ya corregidos por el clásico y su campo se compone con
                  el de él (la imagen se remuestrea una sola vez)

Todos los pares que recibe `corregir` tienen que ser de **la misma cámara**: la desalineación es
una propiedad de la unidad, no del cuadro. Ninguno corrige el corrimiento constante de columnas;
eso lo hace la distancia conocida.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

MODOS = ("ninguna", "clasico", "red", "clasico+red")


@dataclass
class Alineacion:
    """Lo que hizo el alineador con un grupo de pares (para dejarlo en el CSV)."""

    modo: str
    aplica: tuple[str, ...] = ()  # parámetros del clásico que pasaron la regla
    filas_red_px: float | None = None  # mediana |filas| del campo de la red, px de captura
    extra: dict[str, Any] = field(default_factory=dict)


class Alineador:
    """Corrige el ojo derecho de los pares de una cámara, con el modo elegido."""

    def __init__(
        self,
        modo: str,
        geometria: StereoGeometry,
        distorsion: np.ndarray | None = None,
        red: str | None = None,
        device: str = "auto",
    ) -> None:
        if modo not in MODOS:
            raise InvalidInput(f"modo de alineación desconocido: {modo!r}; use {MODOS}")
        if "red" in modo and not red:
            raise InvalidInput(f"--alinear {modo} necesita el checkpoint de la red (--red)")
        self.modo = modo
        self.geometria = geometria
        self.distorsion = distorsion
        self.red = None
        if "red" in modo:
            from visionx.depth.crestereo_adapt.align_train import load_model

            self.red = load_model(red, device)

    def _clasico(self, pares: Sequence[tuple[np.ndarray, np.ndarray]]):
        """Mapas absolutos del clásico con lente para esta cámara, o None si no aplica nada."""
        from visionx.depth.crestereo_adapt.classic import ClassicParams, match_points
        from visionx.depth.crestereo_adapt.lens import fit_lens, pull_maps

        # Como en `hermanas`: los puntos se buscan a media resolución (más rápido, mismo ajuste).
        chica = self.geometria.resized(self.geometria.width // 2, self.geometria.height // 2)
        size = (chica.width, chica.height)
        params = ClassicParams()
        puntos = [
            match_points(
                cv2.resize(izq, size, interpolation=cv2.INTER_AREA),
                cv2.resize(der, size, interpolation=cv2.INTER_AREA),
                params,
            )
            for izq, der in pares
        ]
        est = fit_lens(
            np.vstack([p[0] for p in puntos]), np.vstack([p[1] for p in puntos]), chica, params
        )
        if not (est.ok and est.applied):
            return None, ()
        return pull_maps(est.lens, self.geometria), tuple(est.applied)

    def corregir(
        self, pares: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[list[np.ndarray], Alineacion]:
        """Los derechos corregidos (mismo orden que `pares`) y qué se hizo."""
        from visionx.depth.crestereo_adapt.align_data import ELP_SIZE
        from visionx.depth.crestereo_adapt.align_train import (
            apply_field,
            pooled_field,
            right_field,
        )

        if not pares:
            return [], Alineacion(self.modo)
        info = Alineacion(self.modo)
        if self.modo == "ninguna":
            return [der for _, der in pares], info
        mapas = None
        if "clasico" in self.modo:
            mapas, info.aplica = self._clasico(pares)
        if self.red is None:
            if mapas is None:
                return [der for _, der in pares], info
            return [cv2.remap(der, mapas[0], mapas[1], cv2.INTER_LINEAR) for _, der in pares], info
        # La red mira lo que dejó el clásico (o el par tal cual si el clásico no aplicó nada).
        vistos = [
            der if mapas is None else cv2.remap(der, mapas[0], mapas[1], cv2.INTER_LINEAR)
            for _, der in pares
        ]
        campos = [
            right_field(
                self.red[0], self.red[1], izq, der, self.geometria, self.distorsion, ELP_SIZE
            )
            for (izq, _), der in zip(pares, vistos, strict=True)
        ]
        campo = pooled_field(campos)
        info.filas_red_px = round(float(np.median(np.abs(campo[1]))), 3)
        return [apply_field(der, campo, mapas) for _, der in pares], info
