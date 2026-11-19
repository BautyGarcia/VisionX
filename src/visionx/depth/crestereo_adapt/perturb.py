"""Generador de perturbaciones, etapa 1: desalinear un par de forma conocida.

Sirve para dos cosas (TECNICO, «CREStereo adaptable»):
- medir cuánto aguanta CREStereo solo cuando la calibración no describe bien a la cámara;
- entrenar y evaluar la alineación: como la desalineación la inventamos nosotros, conocemos
  la respuesta correcta sin necesitar disparidad de referencia.

Etapa 1 = cámara ideal, sin distorsión, y solo el ojo derecho. En ese caso, un giro o un cambio
de aumento del ojo derecho mueve la imagen con una homografía exacta (`correction.py`). La
etapa 2 (distorsión y los dos ojos, sobre la foto cruda) todavía no está escrita.

Convención: un `Correction` siempre es **lo que hay que aplicar para arreglar** la imagen. El
generador recibe ese error y produce la imagen que lo necesita:

    derecha_perturbada = mover(derecha_limpia, G(error)⁻¹)
    mover(derecha_perturbada, G(error)) recupera la limpia

El izquierdo no se toca: la disparidad de referencia, si existe, sigue valiendo en su grilla.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionx.depth.crestereo_adapt.correction import (
    Correction,
    displacement_field,
    homography,
    warp_image,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True)
class PerturbationRanges:
    """Amplitud máxima de cada error; se sortea uniforme en ±ese valor.

    Los valores por defecto son los rangos iniciales de la nota del 18/11 (roll ±1,5°,
    pitch ±1°, focal ±1 %) más yaw, que la nota excluía. Se ajustan a lo que se mida.
    """

    roll_deg: float = 1.5
    pitch_deg: float = 1.0
    yaw_deg: float = 0.5
    log_focal: float = 0.01

    def __post_init__(self) -> None:
        if min(self.roll_deg, self.pitch_deg, self.yaw_deg, self.log_focal) < 0:
            raise InvalidInput("los rangos de perturbación no pueden ser negativos")


def sample_error(rng: np.random.Generator, ranges: PerturbationRanges) -> Correction:
    """Sortea un error. La misma semilla da siempre el mismo error."""
    roll, pitch, yaw, log_focal = rng.uniform(-1.0, 1.0, size=4)
    return Correction(
        roll_deg=float(roll) * ranges.roll_deg,
        pitch_deg=float(pitch) * ranges.pitch_deg,
        yaw_deg=float(yaw) * ranges.yaw_deg,
        log_focal=float(log_focal) * ranges.log_focal,
    )


@dataclass(frozen=True)
class PerturbedPair:
    right: np.ndarray  # la derecha desalineada, como la vería el sistema
    error: Correction  # lo que hay que aplicarle para arreglarla (la respuesta correcta)
    field: np.ndarray  # (alto, ancho, 2): cuánto se movió cada píxel de la derecha limpia, en px
    valid: np.ndarray  # (alto, ancho) bool: píxeles de la perturbada que tienen dato


def perturb_right(right: np.ndarray, error: Correction, geometry: StereoGeometry) -> PerturbedPair:
    """Desalinea la imagen derecha con un error conocido."""
    geometry.require_size(right, "right")
    undo = homography(error, geometry)  # arregla: perturbada -> limpia
    do = np.linalg.inv(undo)  # desalinea: limpia -> perturbada
    perturbed = warp_image(right, do)
    # Dónde hay dato: se mueve una imagen toda en 1 y se mira qué quedó cubierto.
    coverage = warp_image(np.full(right.shape[:2], 255, dtype=np.uint8), do)
    return PerturbedPair(
        right=perturbed,
        error=error,
        field=displacement_field(do, geometry.size),
        valid=coverage == 255,
    )


def restore_right(perturbed: np.ndarray, error: Correction, geometry: StereoGeometry) -> np.ndarray:
    """Aplica la corrección a una derecha desalineada (lo que hace la alineación)."""
    geometry.require_size(perturbed, "right")
    return warp_image(perturbed, homography(error, geometry))
