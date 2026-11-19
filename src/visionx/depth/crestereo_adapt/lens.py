"""Corrección con términos de lente: giro, aumento y un resto radial.

El experimento de cámaras hermanas (BITACORA 2026-11-19) mostró que, para otra unidad del mismo
modelo, corregir solo giro y aumento deja la mayor parte del daño: también cambian el centro
óptico y la distorsión. Este módulo extiende el estimador clásico con lo que de eso se puede
recuperar mirando filas.

**Resto radial.** Una diferencia de distorsión mueve cada punto a lo largo de la línea que lo
une con el centro de la imagen, más cuanto más lejos está. Se modela con dos coeficientes:

    punto corregido = centro + (punto − centro) · (1 + a2·r² + a4·r⁴)

`r` es la distancia al centro medida en focales (r = 1 a 45° del eje). El término sin `r` ya
está: es el aumento (`log_focal`). No depende del modelo de distorsión de la cámara: es un
ajuste encima de la rectificación nominal.

Lo importante es que el movimiento es radial: tiene una parte en filas, que se ve, y una en
columnas, que no se ve pero queda determinada por la otra. Estimando los coeficientes con las
filas se corrigen también las columnas, que son las que cambian la profundidad.

**Lo que no entra: el corrimiento horizontal del centro óptico.** Mueve todas las columnas
por igual, igual que el yaw, y no deja huella en las filas. Eso se arregla con una distancia
conocida. El corrimiento vertical del centro tampoco se agrega: se confunde con el pitch.

Solo el ojo derecho, como el resto del paquete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionx.depth.crestereo_adapt.classic import (
    ClassicParams,
    DecisionParams,
    epipolar_inliers,
    robust_fit,
)
from visionx.depth.crestereo_adapt.correction import (
    Correction,
    apply_homography,
    camera_matrix,
    homography,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

LENS_PARAMETERS = ("pitch", "yaw", "roll", "log_focal", "radial2", "radial4")


@dataclass(frozen=True)
class LensCorrection:
    """Giro y aumento (`correction`) más el resto radial. Todo en cero = no tocar."""

    correction: Correction = field(default_factory=Correction)
    radial2: float = 0.0  # coeficiente de r²
    radial4: float = 0.0  # coeficiente de r⁴

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.correction.as_vector(), [self.radial2, self.radial4]])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> LensCorrection:
        return cls(Correction.from_vector(vector[:4]), float(vector[4]), float(vector[5]))

    @property
    def is_identity(self) -> bool:
        return not np.any(self.as_vector())

    def value(self, name: str) -> float:
        """Valor de un parámetro en las unidades en que se informa (grados para los giros)."""
        if name in ("radial2", "radial4"):
            return float(getattr(self, name))
        if name == "log_focal":
            return self.correction.log_focal
        return float(getattr(self.correction, f"{name}_deg"))


def _radial(points: np.ndarray, lens: LensCorrection, k: np.ndarray) -> np.ndarray:
    """Aplica el resto radial alrededor del centro del ojo derecho."""
    x = (points[:, 0] - k[0, 2]) / k[0, 0]
    y = (points[:, 1] - k[1, 2]) / k[1, 1]
    r2 = x * x + y * y
    scale = 1.0 + lens.radial2 * r2 + lens.radial4 * r2 * r2
    return np.column_stack([k[0, 2] + k[0, 0] * x * scale, k[1, 2] + k[1, 1] * y * scale])


def correct_points(
    lens: LensCorrection, geometry: StereoGeometry, points: np.ndarray
) -> np.ndarray:
    """Dónde debería estar cada punto de la derecha base: primero el radial, después el giro."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return apply_homography(
        homography(lens.correction, geometry), _radial(pts, lens, camera_matrix(geometry))
    )


def pull_maps(lens: LensCorrection, geometry: StereoGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Mapas para `cv2.remap`: de qué lugar de la derecha base sale cada píxel corregido.

    Es la cuenta de `correct_points` al revés: se deshace el giro (inversa exacta) y después el
    radial, que no tiene inversa cerrada y se resuelve con unas vueltas de Newton sobre r.
    """
    k = camera_matrix(geometry)
    width, height = geometry.size
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    grid = np.column_stack([u.ravel(), v.ravel()])
    before_rotation = apply_homography(np.linalg.inv(homography(lens.correction, geometry)), grid)
    x = (before_rotation[:, 0] - k[0, 2]) / k[0, 0]
    y = (before_rotation[:, 1] - k[1, 2]) / k[1, 1]
    target = np.hypot(x, y)  # radio ya corregido; se busca el radio r de origen
    r = target.copy()
    for _ in range(8):
        value = r * (1.0 + lens.radial2 * r**2 + lens.radial4 * r**4) - target
        slope = 1.0 + 3.0 * lens.radial2 * r**2 + 5.0 * lens.radial4 * r**4
        r = r - value / np.where(np.abs(slope) < 1e-9, 1.0, slope)
    scale = np.where(target > 1e-12, r / np.maximum(target, 1e-12), 1.0)
    map_x = (k[0, 2] + k[0, 0] * x * scale).reshape(height, width).astype(np.float32)
    map_y = (k[1, 2] + k[1, 1] * y * scale).reshape(height, width).astype(np.float32)
    return map_x, map_y


def warp_lens(image: np.ndarray, lens: LensCorrection, geometry: StereoGeometry) -> np.ndarray:
    """Aplica la corrección a la imagen derecha base, en una sola interpolación."""
    geometry.require_size(image, "right")
    map_x, map_y = pull_maps(lens, geometry)
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


@dataclass(frozen=True)
class LensDecision(DecisionParams):
    """Pisos del resto radial, medidos como los demás: en sesiones sanas de la ELP, donde la
    respuesta correcta es cero, el ajuste devolvió radial2 entre +0,0007 y +0,0012 y radial4
    entre −0,0001 y −0,0004 (BITACORA 2026-11-19)."""

    floor_radial2: float = 1.2e-3
    floor_radial4: float = 4e-4

    def floor(self, name: str) -> float:
        if name == "radial2":
            return self.floor_radial2
        if name == "radial4":
            return self.floor_radial4
        return super().floor(name)


@dataclass(frozen=True)
class LensEstimate:
    lens: LensCorrection  # lo que se aplica (los parámetros que no pasan la regla van en cero)
    fitted: LensCorrection  # el ajuste completo, antes de la regla
    sigma: dict[str, float]  # desvío por parámetro: grados para los giros, sin unidades el resto
    applied: tuple[str, ...]
    inliers: int
    residual_before_px: float
    residual_after_px: float
    ok: bool
    reason: str | None = None


def fit_lens(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    geometry: StereoGeometry,
    params: ClassicParams | None = None,
    decision: LensDecision | None = None,
) -> LensEstimate:
    """Ajusta giro, aumento y resto radial a puntos emparejados, con la regla de decisión."""
    p = params or ClassicParams()
    d = decision or LensDecision()
    pts_l = np.asarray(pts_l, dtype=np.float64).reshape(-1, 2)
    pts_r = np.asarray(pts_r, dtype=np.float64).reshape(-1, 2)
    if pts_l.shape != pts_r.shape:
        raise InvalidInput("pts_l y pts_r deben tener la misma cantidad de puntos")
    nan = float("nan")
    unknown = dict.fromkeys(LENS_PARAMETERS, nan)
    if len(pts_l) < p.min_inliers:
        reason = f"solo {len(pts_l)} pares emparejados (mínimo {p.min_inliers})"
        zero = LensCorrection()
        return LensEstimate(zero, zero, unknown, (), 0, nan, nan, False, reason)
    before = float(np.median(np.abs(pts_l[:, 1] - pts_r[:, 1])))
    pts_l, pts_r = epipolar_inliers(pts_l, pts_r, p)

    def residual(vector: np.ndarray) -> np.ndarray:
        moved = correct_points(LensCorrection.from_vector(vector), geometry, pts_r)
        return pts_l[:, 1] - moved[:, 1]

    vector, deviation, inlier = robust_fit(residual, len(LENS_PARAMETERS), list(range(6)), p)
    fitted = LensCorrection.from_vector(vector)
    sigma = {}
    for index, name in enumerate(LENS_PARAMETERS):
        sigma[name] = math.degrees(deviation[index]) if index < 3 else float(deviation[index])
    # Regla: cada parámetro se aplica solo si supera k veces su incertidumbre (con piso).
    applied = tuple(
        name
        for name in LENS_PARAMETERS
        if abs(fitted.value(name)) > d.k * max(sigma[name], d.floor(name))
    )
    kept = np.array(
        [vector[i] if name in applied else 0.0 for i, name in enumerate(LENS_PARAMETERS)]
    )
    lens = LensCorrection.from_vector(kept)
    after = float(np.median(np.abs(residual(kept))))
    inliers = int(inlier.sum())
    reason = None
    if inliers < p.min_inliers:
        reason = f"solo {inliers} pares coherentes (mínimo {p.min_inliers})"
    elif after > max(p.max_residual_px, before):
        reason = f"quedan {after:.2f} px de diferencia de filas"
    return LensEstimate(
        lens, fitted, sigma, applied, inliers, before, after, reason is None, reason
    )
