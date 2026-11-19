"""Generador de perturbaciones, etapa 1: desalinear un par de forma conocida.

Sirve para dos cosas (TECNICO, «CREStereo adaptable»):
- medir cuánto aguanta CREStereo solo cuando la calibración no describe bien a la cámara;
- entrenar y evaluar la alineación: como la desalineación la inventamos nosotros, conocemos
  la respuesta correcta sin necesitar disparidad de referencia.

Etapa 1 = cámara ideal, sin distorsión, y solo el ojo derecho. En ese caso, un giro o un cambio
de aumento del ojo derecho mueve la imagen con una homografía exacta (`correction.py`).

Etapa 2 (segunda mitad del archivo) = el ojo derecho con su lente: la cámara real y la que
describe la calibración difieren también en centro óptico y distorsión, y la imagen ya no se
mueve con una homografía sino con un campo que cambia de un lugar a otro. Sigue siendo solo el
ojo derecho; perturbar también el izquierdo queda pendiente.

Convención: un `Correction` siempre es **lo que hay que aplicar para arreglar** la imagen. El
generador recibe ese error y produce la imagen que lo necesita:

    derecha_perturbada = mover(derecha_limpia, G(error)⁻¹)
    mover(derecha_perturbada, G(error)) recupera la limpia

El izquierdo no se toca: la disparidad de referencia, si existe, sigue valiendo en su grilla.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
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
    # Lo que hay que aplicarle para arreglarla. En la etapa 2 es solo la parte de giro y
    # aumento: el resto del arreglo no se describe con cuatro números y está en `field`.
    error: Correction
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


# --- Etapa 2: la cámara real contra la que describe la calibración ---------------------------
#
# Un ojo se describe con tres cosas (convención de OpenCV):
#   K  matriz de la lente: focal y centro óptico, en píxeles de la foto cruda
#   D  coeficientes de distorsión (k1, k2, p1, p2, k3, k4, k5, k6, ...)
#   R  giro que lleva la cámara cruda al marco rectificado
#
# Hay dos cámaras en juego:
#   real      la que sacó la foto
#   nominal   la que el sistema cree que sacó la foto (su calibración)
#
# El sistema rectifica con la nominal. Si coinciden, un punto del mundo cae donde debe. Si no,
# cae corrido. Para un píxel q de la derecha limpia (bien rectificada):
#
#   1. q -> dónde lo vio la cámara real en la foto cruda       (K, D, R de la real)
#   2. ese punto crudo -> dónde lo pone la rectificación base  (K, D, R de la nominal)
#
# La diferencia entre el final y q es el campo de desalineación. Las dos cuentas son las de
# OpenCV (`projectPoints` hacia la foto cruda, `undistortPoints` desde ella): exactas, sin
# aproximar giros chicos. La foto cruda nunca se arma: se compone todo en un solo mapa y la
# imagen se interpola una sola vez.


@dataclass(frozen=True)
class CameraModel:
    """Un ojo: lente (K, D) y giro al marco rectificado (R)."""

    K: np.ndarray
    D: np.ndarray
    R: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "K", np.asarray(self.K, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "D", np.asarray(self.D, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64).reshape(3, 3))

    @classmethod
    def ideal(cls, geometry: StereoGeometry) -> CameraModel:
        """Cámara sin distorsión que ya mira en el marco rectificado."""
        return cls(_rectified_matrix(geometry), np.zeros(5), np.eye(3))


def _rectified_matrix(geometry: StereoGeometry) -> np.ndarray:
    """K del ojo derecho ya rectificado (la misma de `correction.camera_matrix`)."""
    return np.array(
        [
            [geometry.fx, 0.0, geometry.cx + geometry.doffs],
            [0.0, geometry.fy, geometry.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _to_raw(points: np.ndarray, camera: CameraModel, rect_k: np.ndarray) -> np.ndarray:
    """Píxeles rectificados -> píxeles de la foto cruda de `camera`."""
    hom = np.column_stack([points, np.ones(len(points))])
    rays = (camera.R.T @ np.linalg.inv(rect_k) @ hom.T).T  # dirección vista desde la cámara
    # projectPoints aplica la distorsión y K. Se le pasan los rayos como puntos 3D sin girar.
    raw, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.D)
    return raw.reshape(-1, 2)


def _to_rectified(points: np.ndarray, camera: CameraModel, rect_k: np.ndarray) -> np.ndarray:
    """Píxeles de la foto cruda de `camera` -> píxeles rectificados."""
    # Sacar la distorsión es iterativo: se piden más vueltas que las 5 por defecto para que el
    # error quede muy por debajo de la centésima de píxel también en los bordes.
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    out = cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64),
        camera.K,
        camera.D,
        R=camera.R,
        P=rect_k,
        criteria=criteria,
    )
    return out.reshape(-1, 2)


def _grid(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return np.column_stack([u.ravel(), v.ravel()])


def mismatch_field(true: CameraModel, nominal: CameraModel, geometry: StereoGeometry) -> np.ndarray:
    """Dónde queda cada píxel de la derecha limpia después de rectificar con la nominal.

    Devuelve (alto, ancho, 2) con (dx, dy) en px. Si las dos cámaras coinciden, es cero.
    """
    rect_k = _rectified_matrix(geometry)
    clean = _grid(geometry.size)
    moved = _to_rectified(_to_raw(clean, true, rect_k), nominal, rect_k)
    return (moved - clean).reshape(geometry.height, geometry.width, 2).astype(np.float32)


def mismatch_maps(
    true: CameraModel, nominal: CameraModel, geometry: StereoGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Mapas para `cv2.remap`: de qué lugar de la derecha limpia sale cada píxel perturbado."""
    rect_k = _rectified_matrix(geometry)
    base = _grid(geometry.size)
    source = _to_rectified(_to_raw(base, nominal, rect_k), true, rect_k)
    source = source.reshape(geometry.height, geometry.width, 2).astype(np.float32)
    return np.ascontiguousarray(source[:, :, 0]), np.ascontiguousarray(source[:, :, 1])


def parametric_error(true: CameraModel, nominal: CameraModel) -> Correction:
    """La parte del arreglo que sí son cuatro números: giro y aumento.

    Con la convención de este paquete (un `Correction` es lo que arregla), el giro que arregla
    es el de la real respecto de la nominal, y el aumento es focal nominal / focal real.
    """
    rotation, _ = cv2.Rodrigues(true.R @ nominal.R.T)
    pitch, yaw, roll = (float(x) for x in rotation.ravel())
    focal_true = math.sqrt(true.K[0, 0] * true.K[1, 1])
    focal_nominal = math.sqrt(nominal.K[0, 0] * nominal.K[1, 1])
    return Correction(
        roll_deg=math.degrees(roll),
        pitch_deg=math.degrees(pitch),
        yaw_deg=math.degrees(yaw),
        log_focal=math.log(focal_nominal / focal_true),
    )


def perturb_right_camera(
    right: np.ndarray, true: CameraModel, nominal: CameraModel, geometry: StereoGeometry
) -> PerturbedPair:
    """Desalinea la derecha como lo haría rectificar con una calibración que no es la real."""
    geometry.require_size(right, "right")
    map_x, map_y = mismatch_maps(true, nominal, geometry)
    perturbed = cv2.remap(
        right, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    inside = (
        (map_x >= 0) & (map_x <= geometry.width - 1) & (map_y >= 0) & (map_y <= geometry.height - 1)
    )
    return PerturbedPair(
        right=perturbed,
        error=parametric_error(true, nominal),
        field=mismatch_field(true, nominal, geometry),
        valid=inside,
    )


@dataclass(frozen=True)
class CameraRanges:
    """Cuánto puede diferir la cámara real de la nominal. Se sortea uniforme en ±cada valor.

    Pensado para «hermanas»: otra unidad del mismo modelo, o la misma después de un golpe.
    Son rangos iniciales sin medir: la variación real entre unidades ELP no se conoce.
    """

    roll_deg: float = 1.0
    pitch_deg: float = 1.0
    yaw_deg: float = 0.5
    focal: float = 0.02  # fracción: ±2 %
    center_px: float = 8.0  # corrimiento del centro óptico, en px de la foto cruda
    distortion: float = 0.05  # fracción por coeficiente: ±5 %

    def __post_init__(self) -> None:
        values = (self.roll_deg, self.pitch_deg, self.yaw_deg, self.focal, self.center_px,
                  self.distortion)  # fmt: skip
        if min(values) < 0:
            raise InvalidInput("los rangos de la cámara no pueden ser negativos")


def sample_sibling(
    rng: np.random.Generator, nominal: CameraModel, ranges: CameraRanges
) -> CameraModel:
    """Sortea una cámara real cerca de la nominal. La misma semilla da la misma cámara."""
    u = rng.uniform(-1.0, 1.0, size=6)
    k = nominal.K.copy()
    k[0, 0] *= math.exp(u[0] * ranges.focal)
    k[1, 1] *= math.exp(u[0] * ranges.focal)  # el mismo cambio en x e y: es la lente
    k[0, 2] += u[1] * ranges.center_px
    k[1, 2] += u[2] * ranges.center_px
    d = nominal.D * (1.0 + rng.uniform(-1.0, 1.0, size=nominal.D.shape) * ranges.distortion)
    angles = np.radians([u[3] * ranges.pitch_deg, u[4] * ranges.yaw_deg, u[5] * ranges.roll_deg])
    extra, _ = cv2.Rodrigues(angles)
    return CameraModel(k, d, extra @ nominal.R)
