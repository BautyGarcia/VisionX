"""Estimador clásico de la corrección entre ojos, con puntos emparejados (sin red).

Es el comparador obligatorio de la red de alineación y su respaldo (TECNICO, «CREStereo
adaptable»). Si la red propia no lo iguala en precisión o en tiempo, queda este.

Idea: en un par bien rectificado, un punto de la escena cae en la **misma fila** de las dos
imágenes. Si quedó un giro o un aumento distinto en el ojo derecho, aparece una diferencia de
filas que depende de dónde está el punto (`correction.py` explica cuánto mueve cada parámetro).
Se buscan puntos reconocibles en las dos imágenes, se mide esa diferencia y se ajustan los
parámetros que mejor la anulan.

Pasos:
1. **SIFT** encuentra puntos reconocibles y los empareja (se queda con los pares donde el
   mejor candidato es claramente mejor que el segundo, y que coinciden en los dos sentidos).
2. **RANSAC** sobre la geometría epipolar descarta emparejamientos equivocados groseros.
3. **Cuadrados mínimos robustos**: se busca la corrección que deja la menor diferencia de
   filas. Los residuos grandes pesan menos (Huber), para que unos pocos pares malos no manden.

Además de la corrección se devuelve su **incertidumbre** (desvío estándar por parámetro), que
sale de la curvatura del ajuste. Para el yaw suele ser grande: casi no deja huella vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from visionx.depth.crestereo_adapt.correction import (
    PARAMETERS,
    Correction,
    apply_homography,
    homography,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True)
class ClassicParams:
    max_features: int = 4000  # puntos SIFT por imagen
    ratio: float = 0.75  # el mejor candidato debe ganarle al segundo por este factor
    ransac_px: float = 3.0  # tolerancia de la geometría epipolar al descartar pares
    huber_px: float = 1.0  # residuos mayores pesan menos
    iterations: int = 30
    estimate_yaw: bool = True
    estimate_focal: bool = True
    min_inliers: int = 40  # con menos pares no se confía en el ajuste
    max_residual_px: float = 1.0  # mediana de |diferencia de filas| aceptable al final


@dataclass(frozen=True)
class AlignmentEstimate:
    """Resultado del ajuste. `ok=False` significa «no confiar»: `reason` dice por qué."""

    correction: Correction
    sigma: dict[str, float]  # desvío estándar: ángulos en grados, log_focal sin unidades
    matches: int  # pares emparejados antes de descartar
    inliers: int  # pares usados en el ajuste
    residual_before_px: float  # mediana de |diferencia de filas| sin corregir
    residual_after_px: float  # la misma mediana después de corregir
    ok: bool
    reason: str | None = None


def match_points(
    left: np.ndarray, right: np.ndarray, params: ClassicParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Puntos emparejados entre los dos ojos: dos arreglos (N, 2) en píxeles."""
    p = params or ClassicParams()
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    sift = cv2.SIFT_create(nfeatures=p.max_features)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    empty = np.zeros((0, 2), dtype=np.float64)
    if des_l is None or des_r is None or len(kp_l) < 2 or len(kp_r) < 2:
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(des_l, des_r, k=2)
    backward = matcher.knnMatch(des_r, des_l, k=1)
    best_back = {m[0].queryIdx: m[0].trainIdx for m in backward if m}
    pts_l, pts_r = [], []
    for pair in forward:
        if len(pair) < 2:
            continue
        first, second = pair
        # Coincidencia en los dos sentidos y con margen sobre el segundo candidato.
        if first.distance < p.ratio * second.distance and best_back.get(first.trainIdx) == (
            first.queryIdx
        ):
            pts_l.append(kp_l[first.queryIdx].pt)
            pts_r.append(kp_r[first.trainIdx].pt)
    if not pts_l:
        return empty, empty
    return np.asarray(pts_l, dtype=np.float64), np.asarray(pts_r, dtype=np.float64)


def _residuals(
    vector: np.ndarray, pts_l: np.ndarray, pts_r: np.ndarray, geometry: StereoGeometry
) -> np.ndarray:
    """Diferencia de filas (px) que queda después de aplicar la corrección `vector`."""
    moved = apply_homography(homography(Correction.from_vector(vector), geometry), pts_r)
    return pts_l[:, 1] - moved[:, 1]


def _jacobian(
    vector: np.ndarray,
    free: list[int],
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    geometry: StereoGeometry,
) -> np.ndarray:
    """Cuánto cambia cada residuo al mover cada parámetro libre (diferencias centradas)."""
    step = 1e-6
    columns = []
    for index in free:
        delta = np.zeros(4)
        delta[index] = step
        plus = _residuals(vector + delta, pts_l, pts_r, geometry)
        minus = _residuals(vector - delta, pts_l, pts_r, geometry)
        columns.append((plus - minus) / (2 * step))
    return np.column_stack(columns)


def fit_correction(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    geometry: StereoGeometry,
    params: ClassicParams | None = None,
) -> AlignmentEstimate:
    """Ajusta la corrección del ojo derecho a pares de puntos ya emparejados."""
    p = params or ClassicParams()
    pts_l = np.asarray(pts_l, dtype=np.float64).reshape(-1, 2)
    pts_r = np.asarray(pts_r, dtype=np.float64).reshape(-1, 2)
    if pts_l.shape != pts_r.shape:
        raise InvalidInput("pts_l y pts_r deben tener la misma cantidad de puntos")
    matches = len(pts_l)
    nan = float("nan")
    sigma_unknown = {name: nan for name in PARAMETERS}

    def failed(reason: str, before: float = nan) -> AlignmentEstimate:
        return AlignmentEstimate(
            Correction(), sigma_unknown, matches, 0, before, nan, False, reason
        )

    if matches < p.min_inliers:
        return failed(f"solo {matches} pares emparejados (mínimo {p.min_inliers})")
    before = float(np.median(np.abs(pts_l[:, 1] - pts_r[:, 1])))

    # RANSAC epipolar: descarta pares equivocados sin suponer que el par ya está alineado.
    _, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, p.ransac_px, 0.999)
    if mask is not None and int(mask.sum()) >= p.min_inliers:
        keep = mask.ravel().astype(bool)
        pts_l, pts_r = pts_l[keep], pts_r[keep]

    # Parámetros libres, en el orden de PARAMETERS: pitch, yaw, roll, log_focal.
    free = [0, 2]
    if p.estimate_yaw:
        free.insert(1, 1)
    if p.estimate_focal:
        free.append(3)
    vector = np.zeros(4)
    weights = np.ones(len(pts_l))
    for _ in range(p.iterations):
        r = _residuals(vector, pts_l, pts_r, geometry)
        # Huber: peso 1 para residuos chicos y decreciente para los grandes. La escala sale de
        # la dispersión robusta de los propios residuos, con piso en `huber_px`.
        spread = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        k = max(p.huber_px, 1.345 * spread)
        weights = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
        jac = _jacobian(vector, free, pts_l, pts_r, geometry)
        normal = jac.T @ (weights[:, None] * jac)
        # Amortiguación mínima: evita dividir por cero si un parámetro casi no se observa.
        normal += 1e-12 * np.trace(normal) * np.eye(len(free))
        step = np.linalg.solve(normal, -jac.T @ (weights * r))
        vector[free] += step
        if float(np.max(np.abs(step))) < 1e-9:
            break

    r = _residuals(vector, pts_l, pts_r, geometry)
    spread = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    inlier = np.abs(r) <= max(3.0 * spread, p.huber_px)
    inliers = int(inlier.sum())
    after = float(np.median(np.abs(r)))
    # Incertidumbre: varianza de los residuos por la inversa de la curvatura del ajuste.
    jac = _jacobian(vector, free, pts_l[inlier], pts_r[inlier], geometry)
    dof = max(inliers - len(free), 1)
    variance = float(np.sum(r[inlier] ** 2)) / dof
    covariance = variance * np.linalg.pinv(jac.T @ jac)
    sigma = dict.fromkeys(PARAMETERS, 0.0)
    for position, index in enumerate(free):
        value = math.sqrt(max(float(covariance[position, position]), 0.0))
        sigma[PARAMETERS[index]] = value if index == 3 else math.degrees(value)

    correction = Correction.from_vector(vector)
    reason = None
    if inliers < p.min_inliers:
        reason = f"solo {inliers} pares coherentes (mínimo {p.min_inliers})"
    elif after > p.max_residual_px:
        reason = f"quedan {after:.2f} px de diferencia de filas (máximo {p.max_residual_px})"
    return AlignmentEstimate(
        correction, sigma, matches, inliers, before, after, reason is None, reason
    )


def estimate_alignment(
    left: np.ndarray,
    right: np.ndarray,
    geometry: StereoGeometry,
    params: ClassicParams | None = None,
) -> AlignmentEstimate:
    """Corrección del ojo derecho a partir de un par rectificado BGR del tamaño de `geometry`."""
    geometry.require_size(left, "left")
    geometry.require_size(right, "right")
    pts_l, pts_r = match_points(left, right, params)
    return fit_correction(pts_l, pts_r, geometry, params)
