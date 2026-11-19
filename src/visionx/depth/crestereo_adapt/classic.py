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

import dataclasses
import math
from collections.abc import Callable
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


def epipolar_inliers(
    pts_l: np.ndarray, pts_r: np.ndarray, params: ClassicParams
) -> tuple[np.ndarray, np.ndarray]:
    """Descarta emparejamientos equivocados groseros con RANSAC sobre la geometría epipolar.

    No supone que el par ya esté alineado. Sortea subconjuntos al azar: se fija la semilla para
    que el mismo par dé siempre lo mismo.
    """
    cv2.setRNGSeed(0)
    _, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, params.ransac_px, 0.999)
    if mask is not None and int(mask.sum()) >= params.min_inliers:
        keep = mask.ravel().astype(bool)
        return pts_l[keep], pts_r[keep]
    return pts_l, pts_r


def robust_fit(
    residual: Callable[[np.ndarray], np.ndarray],
    size: int,
    free: list[int],
    params: ClassicParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuadrados mínimos robustos de un vector de `size` parámetros; solo se mueven los `free`.

    `residual(vector)` devuelve un residuo por punto. Devuelve (vector ajustado, desvío
    estándar por parámetro con 0 en los fijos, máscara de puntos coherentes).
    """

    def jacobian(vector: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        # Cuánto cambia cada residuo al mover cada parámetro libre (diferencias centradas).
        step = 1e-6
        columns = []
        for index in free:
            delta = np.zeros(size)
            delta[index] = step
            column = (residual(vector + delta) - residual(vector - delta)) / (2 * step)
            columns.append(column if rows is None else column[rows])
        return np.column_stack(columns)

    vector = np.zeros(size)
    for _ in range(params.iterations):
        r = residual(vector)
        # Huber: peso 1 para residuos chicos y decreciente para los grandes. La escala sale de
        # la dispersión robusta de los propios residuos, con piso en `huber_px`.
        spread = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        k = max(params.huber_px, 1.345 * spread)
        weights = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
        jac = jacobian(vector)
        normal = jac.T @ (weights[:, None] * jac)
        # Amortiguación mínima: evita dividir por cero si un parámetro casi no se observa.
        normal += 1e-12 * np.trace(normal) * np.eye(len(free))
        step = np.linalg.solve(normal, -jac.T @ (weights * r))
        vector[free] += step
        if float(np.max(np.abs(step))) < 1e-9:
            break

    r = residual(vector)
    spread = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    inlier = np.abs(r) <= max(3.0 * spread, params.huber_px)
    # Incertidumbre: varianza de los residuos por la inversa de la curvatura del ajuste.
    jac = jacobian(vector, inlier)
    dof = max(int(inlier.sum()) - len(free), 1)
    variance = float(np.sum(r[inlier] ** 2)) / dof
    covariance = variance * np.linalg.pinv(jac.T @ jac)
    sigma = np.zeros(size)
    for position, index in enumerate(free):
        sigma[index] = math.sqrt(max(float(covariance[position, position]), 0.0))
    return vector, sigma, inlier


def fit_correction(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    geometry: StereoGeometry,
    params: ClassicParams | None = None,
    parameters: tuple[str, ...] | None = None,
) -> AlignmentEstimate:
    """Ajusta la corrección del ojo derecho a pares de puntos ya emparejados.

    `parameters` elige a mano qué se ajusta (nombres de `PARAMETERS`); lo demás queda en cero.
    Sin él, se ajustan pitch y roll, y yaw y focal según `params`.
    """
    p = params or ClassicParams()
    if parameters is not None:
        unknown = set(parameters) - set(PARAMETERS)
        if unknown:
            raise InvalidInput(f"parámetros desconocidos: {sorted(unknown)}")
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

    pts_l, pts_r = epipolar_inliers(pts_l, pts_r, p)

    # Parámetros libres, en el orden de PARAMETERS: pitch, yaw, roll, log_focal.
    if parameters is not None:
        free = [i for i, name in enumerate(PARAMETERS) if name in parameters]
    else:
        free = [0, 2]
        if p.estimate_yaw:
            free.insert(1, 1)
        if p.estimate_focal:
            free.append(3)
    if not free:  # nada que ajustar: la corrección es "no tocar"
        zero = dict.fromkeys(PARAMETERS, 0.0)
        return AlignmentEstimate(Correction(), zero, matches, len(pts_l), before, before, True)
    vector, deviation, inlier = robust_fit(
        lambda v: _residuals(v, pts_l, pts_r, geometry), 4, free, p
    )
    inliers = int(inlier.sum())
    after = float(np.median(np.abs(_residuals(vector, pts_l, pts_r, geometry))))
    sigma = dict.fromkeys(PARAMETERS, 0.0)
    for index in free:  # los ángulos se informan en grados; log_focal no tiene unidades
        value = float(deviation[index])
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


@dataclass(frozen=True)
class DecisionParams:
    """Cuándo vale la pena aplicar cada parámetro de la corrección.

    Corregir con un valor ruidoso empeora lo que había: en la curva de robustez del 18/11, con
    errores de 0,05° el clásico dejaba la profundidad peor que sin tocar. La regla es aplicar
    un parámetro solo si su valor supera `k` veces su incertidumbre.

    La incertidumbre que calcula el ajuste es optimista: supone que los residuos son ruido
    puro y no ve errores sistemáticos (distorsión mal modelada, puntos mal ubicados). Por eso
    se le pone un **piso** por parámetro, medido corriendo el estimador sobre sesiones sanas
    de la ELP, donde la respuesta correcta es cero (BITACORA 2026-11-18): roll hasta 0,021°,
    pitch hasta 0,009°, yaw hasta 0,036°, focal hasta 0,053 %.
    """

    k: float = 2.0
    floor_roll_deg: float = 0.02
    floor_pitch_deg: float = 0.01
    floor_yaw_deg: float = 0.04
    floor_log_focal: float = 0.0005

    def floor(self, name: str) -> float:
        return {
            "roll": self.floor_roll_deg,
            "pitch": self.floor_pitch_deg,
            "yaw": self.floor_yaw_deg,
            "log_focal": self.floor_log_focal,
        }[name]


def _value(correction: Correction, name: str) -> float:
    return correction.log_focal if name == "log_focal" else getattr(correction, f"{name}_deg")


def significant_parameters(
    estimate: AlignmentEstimate, decision: DecisionParams | None = None
) -> tuple[str, ...]:
    """Parámetros cuyo valor supera `k` veces su incertidumbre (con piso)."""
    d = decision or DecisionParams()
    keep = []
    for name in PARAMETERS:
        threshold = d.k * max(float(estimate.sigma[name]), d.floor(name))
        if abs(_value(estimate.correction, name)) > threshold:
            keep.append(name)
    return tuple(keep)


def fit_significant(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    geometry: StereoGeometry,
    params: ClassicParams | None = None,
    decision: DecisionParams | None = None,
) -> tuple[AlignmentEstimate, tuple[str, ...]]:
    """Ajuste con la regla de decisión: devuelve la corrección a aplicar y qué parámetros usa.

    Se ajusta todo junto y después se anulan los parámetros que no superan su umbral. Los que
    quedan conservan el valor del ajuste completo. No se vuelve a ajustar solo con ellos: los
    parámetros se influyen entre sí, y al sacar uno del ajuste los demás absorben su efecto y
    cambian (se probó con los pares del 09/11: el yaw solo dejaba de superar el umbral). Si
    ninguno lo supera, la respuesta es no corregir.
    """
    estimate = fit_correction(pts_l, pts_r, geometry, params)
    if not estimate.ok:
        return estimate, ()
    chosen = significant_parameters(estimate, decision)
    kept = {
        name: _value(estimate.correction, name) if name in chosen else 0.0 for name in PARAMETERS
    }
    correction = Correction(
        roll_deg=kept["roll"],
        pitch_deg=kept["pitch"],
        yaw_deg=kept["yaw"],
        log_focal=kept["log_focal"],
    )
    # Residuo que queda con la corrección que de verdad se aplica.
    remaining = _residuals(
        correction.as_vector(),
        np.asarray(pts_l, dtype=np.float64).reshape(-1, 2),
        np.asarray(pts_r, dtype=np.float64).reshape(-1, 2),
        geometry,
    )
    applied = dataclasses.replace(
        estimate, correction=correction, residual_after_px=float(np.median(np.abs(remaining)))
    )
    return applied, chosen
