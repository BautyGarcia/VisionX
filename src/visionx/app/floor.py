"""El piso en la profundidad: altura e inclinación reales de la cámara en el pecho.

Una cámara en un arnés se inclina con la postura: 10° cambian «pecho» por «cintura». En la
parte de abajo de la imagen casi siempre hay piso; se ajusta un plano con RANSAC a esos puntos
3D y se acepta si es horizontal (normal a ±35° de la vertical de la cámara), está entre 0,6 y
2 m por debajo y lo sostienen al menos 150 puntos y el 8 % de la mitad de abajo (en clips
reales el piso es el 5–15 % de esa zona: hay muebles), con 5 cm de tolerancia. La altura de la
cámara es la distancia al plano; la inclinación, el ángulo de su normal.

`FloorTracker` usa la mediana de los últimos ajustes y solo la aplica si son estables
(dispersión < 12 cm y < 8°); si no, el cuerpo vuelve a la proporción (cámara al 72 % de la
estatura, nivelada). Barato: unos 2000 puntos y 80 intentos de RANSAC por cuadro nuevo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FloorFit:
    camera_height_m: float
    pitch_deg: float  # + la cámara mira hacia arriba, − hacia abajo
    inlier_fraction: float


def fit_floor(
    depth_m: np.ndarray,
    valid: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    rng: np.random.Generator | None = None,
    max_points: int = 2000,
    iterations: int = 80,
    tol_m: float = 0.05,
    min_inliers: float = 0.08,
    min_points: int = 150,
) -> FloorFit | None:
    """Plano del piso en la mitad de abajo de la imagen, o None si no hay uno creíble."""
    rng = rng or np.random.default_rng(0)
    h, w = depth_m.shape
    rows = np.arange(h // 2, h)
    sub = depth_m[h // 2 :]
    ok = valid[h // 2 :] & np.isfinite(sub) & (sub > 0.2) & (sub < 6.0)
    vs, us = np.nonzero(ok)
    if len(vs) < 200:
        return None
    if len(vs) > max_points:
        pick = rng.choice(len(vs), max_points, replace=False)
        vs, us = vs[pick], us[pick]
    z = sub[vs, us].astype(np.float64)
    v = rows[vs].astype(np.float64)
    # Marco de la cámara con Y hacia ARRIBA: x derecha, y arriba, z adelante.
    pts = np.column_stack([(us - cx) * z / fx, -(v - cy) * z / fy, z])
    candidates = []  # (inliers, d, n, máscara) de cada plano horizontal plausible
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[1] < 0:
            n = -n  # normal hacia arriba
        if n[1] < math.cos(math.radians(35)):
            continue  # no es horizontal
        d = -float(n @ a)  # plano n·p + d = 0; la cámara está a +d sobre él
        if not 0.6 <= d <= 2.0:
            continue
        inliers = np.abs(pts @ n + d) < tol_m
        candidates.append((int(inliers.sum()), d, n, inliers))
    if not candidates:
        return None
    top_n, _, n0, _ = max(candidates, key=lambda c: c[0])
    need = max(min_points, int(min_inliers * len(pts)))
    if top_n < need:
        return None
    # El plano con más apoyo da la ORIENTACIÓN (confiable); entre los planos paralelos a él
    # (±5°) con buen apoyo, el piso es el MÁS BAJO: en vistas apuntadas a objetos una mesa o
    # una cama puede tener más puntos que el piso. Los reflejos de un piso brillante dan planos
    # inclinados (dev-1114-distancias: −27° con la cámara nivelada) y el paralelismo los saca.
    cos5 = math.cos(math.radians(5))
    parallel = [c for c in candidates if c[0] >= max(need, int(0.3 * top_n)) and c[2] @ n0 > cos5]
    best_n, d, n, inliers = max(parallel, key=lambda c: c[1])
    # Reajuste por mínimos cuadrados con los inliers (plano por SVD).
    p = pts[inliers]
    centroid = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
    n = vt[-1] if vt[-1][1] > 0 else -vt[-1]
    d = -float(n @ centroid)
    # Con la cámara mirando θ hacia abajo, la normal del piso en la cámara es (0, cos θ, −sin θ):
    # atan2(n_z, n_y) = −θ, negativa al mirar abajo.
    pitch = math.degrees(math.atan2(n[2], n[1]))
    return FloorFit(camera_height_m=d, pitch_deg=pitch, inlier_fraction=best_n / len(pts))


class FloorTracker:
    """Mediana de los últimos ajustes del piso, aplicada solo si son estables; actualiza
    `modes.body` con la altura y la inclinación de la cámara."""

    def __init__(
        self, window: int = 7, keep_s: float = 5.0, max_spread_m: float = 0.12,
        max_spread_deg: float = 8.0, min_fits: int = 3,
    ) -> None:  # fmt: skip
        from collections import deque

        self.keep_s = keep_s
        self.max_spread_m, self.max_spread_deg, self.min_fits = (
            max_spread_m,
            max_spread_deg,
            min_fits,
        )
        self._recent: deque[tuple[float, float]] = deque(maxlen=window)
        self.height: float | None = None
        self.pitch: float = 0.0
        self.stable = False
        self.last_seen_s: float | None = None
        self.fits = 0

    def update(self, depth_m, valid, geometry, now_s: float) -> FloorFit | None:
        from visionx.modes import body

        fit = fit_floor(depth_m, valid, geometry.fx, geometry.fy, geometry.cx, geometry.cy)
        if fit is not None:
            self.fits += 1
            self._recent.append((fit.camera_height_m, fit.pitch_deg))
            hs = np.array([h for h, _ in self._recent])
            ps = np.array([q for _, q in self._recent])
            self.height, self.pitch = float(np.median(hs)), float(np.median(ps))
            spread_h = float(np.subtract(*np.percentile(hs, [75, 25])))
            spread_p = float(np.subtract(*np.percentile(ps, [75, 25])))
            self.stable = (
                len(self._recent) >= self.min_fits
                and spread_h < self.max_spread_m
                and spread_p < self.max_spread_deg
            )
            self.last_seen_s = now_s
            if self.stable:
                body.set_camera(self.height, self.pitch, True)
            else:
                body.set_camera(None, 0.0, False)
        elif self.last_seen_s is not None and now_s - self.last_seen_s > self.keep_s:
            self.height, self.last_seen_s, self.stable = None, None, False
            self._recent.clear()
            body.set_camera(None, 0.0, False)
        return fit
