"""Profundidad estéreo clásica (T05): SGBM y sus controles B0, B1 y B2.

- **B0:** SGBM de OpenCV con sus opciones base. Solo se descartan los píxeles que SGBM marca
  como inválidos y los que no cumplen d + doffs > 0.
- **B1:** B0 + filtros de fiabilidad: unicidad (el mejor costo tiene que ganarle claramente al
  segundo), manchas chicas aisladas y consistencia izquierda/derecha (L/R).
- **B2:** B1 + filtro WLS de `cv2.ximgproc`, que rellena y suaviza guiado por la imagen
  izquierda. Su máscara sale del mapa de confianza del propio filtro.

Convenciones (las mismas de shared/geometry.py):
- Par rectificado, BGR uint8. Disparidad d = x_izquierda - x_derecha, en píxeles float32.
- SGBM devuelve enteros multiplicados por 16 (4 bits de subpíxel): se dividen acá, una sola
  vez, antes de cualquier métrica o geometría.
- El matcher del ojo derecho (`createRightMatcher`) devuelve disparidades NEGATIVAS. Se
  convierten a positivas apenas salen, para que los dos mapas hablen la misma convención.
- Inválido: SGBM escribe (min_disparity - 1) * 16. Una disparidad 0 o negativa NO es inválida
  por sí misma: lo físico es d + doffs > 0, y eso decide la geometría, no SGBM.

Los valores de los parámetros en configs/default.yaml son provisorios hasta el ajuste con la
validación de InStereo2K (T05, PR 3). El test público no se usa para elegirlos.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields, replace
from typing import Any

import cv2
import numpy as np

from visionx.shared import _validate as v
from visionx.shared.contracts import DepthResult, DepthStatus, StereoFrame
from visionx.shared.errors import EstimatorError, IdentityMismatch, InvalidInput
from visionx.shared.geometry import StereoGeometry

VARIANTS = ("B0", "B1", "B2")
SUBPIXEL = 16  # SGBM entrega la disparidad en dieciseisavos de píxel


@dataclass(frozen=True)
class SgbmParams:
    """Parámetros de SGBM y de los filtros de B1/B2. Unidades en cada comentario."""

    num_disparities: int = 96  # ancho del rango de búsqueda, px; múltiplo de 16
    min_disparity: int = 0  # primera disparidad buscada, px
    block_size: int = 5  # lado de la ventana de comparación, px; impar
    p1_factor: int = 8  # P1 = p1_factor * canales * block_size²: castigo a saltos de 1 px
    p2_factor: int = 32  # P2 = p2_factor * canales * block_size²: castigo a saltos mayores
    mode: str = "sgbm_3way"  # "sgbm", "sgbm_3way" o "hh": cuántas direcciones recorre
    # --- B1 ---
    uniqueness_ratio: int = 10  # %: el mejor costo le gana al segundo por este margen
    speckle_window: int = 100  # px: manchas conexas más chicas que esto se descartan
    speckle_range: int = 2  # px: variación máxima de d dentro de una mancha
    lr_max_diff_px: float = 1.0  # px: diferencia máxima ida y vuelta entre los dos ojos
    # --- B2 ---
    wls_lambda: float = 8000.0  # fuerza del suavizado (sin unidad)
    wls_sigma: float = 1.5  # sensibilidad a bordes de color de la imagen guía
    wls_min_confidence: float = 128.0  # 0-255: confianza mínima del WLS para dar valor

    def __post_init__(self) -> None:
        if v.non_negative_int("num_disparities", self.num_disparities) == 0:
            raise InvalidInput("num_disparities tiene que ser > 0")
        if self.num_disparities % SUBPIXEL:
            raise InvalidInput(f"num_disparities={self.num_disparities} no es múltiplo de 16")
        if type(self.min_disparity) is not int:
            raise InvalidInput("min_disparity tiene que ser entero")
        if v.non_negative_int("block_size", self.block_size) % 2 == 0:
            raise InvalidInput("block_size tiene que ser impar")
        if self.mode not in _MODES:
            raise InvalidInput(f"mode {self.mode!r}: se esperaba uno de {sorted(_MODES)}")
        for name in ("p1_factor", "p2_factor", "uniqueness_ratio", "speckle_window",
                     "speckle_range"):  # fmt: skip
            v.non_negative_int(name, getattr(self, name))
        if self.p2_factor <= self.p1_factor:
            raise InvalidInput("P2 tiene que ser mayor que P1 (exigencia de SGBM)")
        v.finite("lr_max_diff_px", self.lr_max_diff_px, low=0.0)
        v.finite("wls_lambda", self.wls_lambda, low=0.0)
        v.finite("wls_sigma", self.wls_sigma, low=0.0)
        v.finite("wls_min_confidence", self.wls_min_confidence, low=0.0)

    @classmethod
    def from_config(cls, section: dict[str, Any], **overrides: Any) -> SgbmParams:
        """Toma de la sección `depth.sgbm` solo las claves que son parámetros."""
        names = {f.name for f in fields(cls)}
        values = {k: val for k, val in section.items() if k in names}
        values.update(overrides)
        return cls(**values)


_MODES = {
    "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
    "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "hh": cv2.STEREO_SGBM_MODE_HH,
}


def round_up_16(value: float) -> int:
    """Redondea hacia arriba al múltiplo de 16 que exige SGBM (mínimo 16)."""
    return max(SUBPIXEL, SUBPIXEL * math.ceil(value / SUBPIXEL))


def num_disparities_for(
    geometry: StereoGeometry, min_depth_m: float, min_disparity: int = 0
) -> int:
    """Rango de búsqueda para ver objetos desde `min_depth_m`.

    Un objeto a Z metros tiene d = fx * B / Z - doffs. Más cerca, más disparidad: el rango
    tiene que cubrir la del punto más cercano que queremos medir. Un valor fijo (p. ej. 96)
    sirve para una resolución y deja ciegos los objetos cercanos en otra.
    """
    if v.finite("min_depth_m", min_depth_m) <= 0:
        raise InvalidInput("min_depth_m tiene que ser > 0")
    d_max = geometry.disparity_at(min_depth_m)
    if d_max is None:
        raise InvalidInput(f"no hay disparidad física a {min_depth_m} m con esta geometría")
    return round_up_16(d_max - min_disparity + 1)


@dataclass(frozen=True)
class StereoOutput:
    """Disparidad de un par: el núcleo que comparten el estimador y la evaluación offline."""

    disparity_px: np.ndarray  # H x W float32; NaN fuera de valid
    valid: np.ndarray  # H x W bool
    stats: dict[str, float]  # fracciones descartadas por cada filtro, para el diagnóstico


class SgbmMatcher:
    """Calcula B0, B1 o B2 sobre un par rectificado. Sin estado entre pares."""

    def __init__(self, variant: str, params: SgbmParams) -> None:
        if variant not in VARIANTS:
            raise InvalidInput(f"variante {variant!r}: se esperaba una de {VARIANTS}")
        self.variant = variant
        self.params = params
        if variant == "B0":
            # B0 = SGBM pelado: sin filtros de fiabilidad, para medir cuánto aportan en B1.
            params = replace(params, uniqueness_ratio=0, speckle_window=0, speckle_range=0)
        self._left = self._create(params)
        # El matcher derecho comparte los parámetros del izquierdo; lo necesitan L/R y WLS.
        self._right = cv2.ximgproc.createRightMatcher(self._left) if variant != "B0" else None
        self._wls = None
        if variant == "B2":
            self._wls = cv2.ximgproc.createDisparityWLSFilter(self._left)
            self._wls.setLambda(params.wls_lambda)
            self._wls.setSigmaColor(params.wls_sigma)

    @staticmethod
    def _create(p: SgbmParams) -> cv2.StereoSGBM:
        channels = 3  # trabajamos en BGR; ETH3D llega como gris replicado a 3 canales
        area = p.block_size * p.block_size
        return cv2.StereoSGBM_create(
            minDisparity=p.min_disparity,
            numDisparities=p.num_disparities,
            blockSize=p.block_size,
            P1=p.p1_factor * channels * area,
            P2=p.p2_factor * channels * area,
            disp12MaxDiff=-1,  # L/R propio y explícito en B1; no el atajo interno de OpenCV
            uniquenessRatio=p.uniqueness_ratio,
            speckleWindowSize=p.speckle_window,
            speckleRange=p.speckle_range,
            mode=_MODES[p.mode],
        )

    def compute(self, left: np.ndarray, right: np.ndarray, doffs: float = 0.0) -> StereoOutput:
        v.array("left", left, np.uint8, 3, (None, None, 3))
        v.array("right", right, np.uint8, 3, left.shape)
        v.finite("doffs", doffs)
        raw_left = self._left.compute(left, right)
        disp_left, valid = _to_pixels(raw_left, self._left.getMinDisparity())
        total = valid.size
        stats = {"sgbm_valida": float(valid.sum()) / total}

        if self.variant == "B0":
            return _finish(disp_left, valid, doffs, stats)

        raw_right = self._right.compute(right, left)
        # El matcher derecho busca con disparidades negativas: d_derecha = -raw/16 >= 0.
        disp_right, valid_right = _to_pixels(raw_right, self._right.getMinDisparity())
        disp_right = -disp_right

        if self.variant == "B1":
            consistent = lr_consistency(
                disp_left, valid, disp_right, valid_right, self.params.lr_max_diff_px
            )
            stats["lr_descartada"] = float((valid & ~consistent).sum()) / total
            return _finish(disp_left, valid & consistent, doffs, stats)

        # B2: WLS recibe los mapas crudos (x16, convención de OpenCV) y la imagen guía.
        filtered = self._wls.filter(raw_left, left, disparity_map_right=raw_right)
        confidence = self._wls.getConfidenceMap()  # 0-255, mismo tamaño que la imagen
        disp, ok = _to_pixels(filtered, self._left.getMinDisparity())
        ok &= confidence >= self.params.wls_min_confidence
        stats["wls_confianza_baja"] = (
            float((~(confidence >= self.params.wls_min_confidence)).sum()) / total
        )
        return _finish(disp, ok, doffs, stats)


def _to_pixels(raw: np.ndarray, min_disparity: int) -> tuple[np.ndarray, np.ndarray]:
    """int16 x16 -> float32 en píxeles, más la máscara de lo que SGBM no marcó inválido."""
    valid = raw.astype(np.int32) >= min_disparity * SUBPIXEL
    disp = raw.astype(np.float32) / SUBPIXEL
    disp[~valid] = np.nan
    return disp, valid


def lr_consistency(
    disp_left: np.ndarray,
    valid_left: np.ndarray,
    disp_right: np.ndarray,
    valid_right: np.ndarray,
    max_diff_px: float,
) -> np.ndarray:
    """Chequeo de ida y vuelta entre los dos ojos.

    Un punto en x del ojo izquierdo con disparidad dL está en x - dL del derecho. Si la
    medida es buena, el ojo derecho ve allí la misma disparidad (dR, en convención positiva):
    |dL(x) - dR(x - dL)| <= max_diff_px. Falla típicamente en oclusiones (zonas que un ojo
    ve y el otro no) y en coincidencias equivocadas.
    """
    h, w = disp_left.shape
    cols = np.arange(w, dtype=np.float32)[None, :]
    target = np.rint(cols - np.where(valid_left, disp_left, 0.0)).astype(np.int64)
    inside = (target >= 0) & (target < w)
    target = np.clip(target, 0, w - 1)
    rows = np.arange(h)[:, None]
    back = disp_right[rows, target]
    back_ok = valid_right[rows, target]
    with np.errstate(invalid="ignore"):
        close = np.abs(disp_left - back) <= max_diff_px
    return valid_left & inside & back_ok & close


def _finish(
    disp: np.ndarray, valid: np.ndarray, doffs: float, stats: dict[str, float]
) -> StereoOutput:
    # Lo físico es d + doffs > 0 (geometry.py): con doffs > 0, d puede ser 0 o negativa.
    with np.errstate(invalid="ignore"):
        valid = valid & np.isfinite(disp) & ((disp.astype(np.float64) + doffs) > 0)
    disp = disp.astype(np.float32, copy=True)
    disp[~valid] = np.nan
    stats["cobertura"] = float(valid.sum()) / valid.size
    return StereoOutput(disp, valid, stats)


def compute_geometry(
    geometry: StereoGeometry, compute_size: tuple[int, int] | None
) -> StereoGeometry:
    """Geometría en la que se calcula: la de captura o su versión reescalada."""
    width, height = compute_size or geometry.size
    return geometry if (width, height) == geometry.size else geometry.resized(width, height)


def prepare_pair(
    frame: StereoFrame, capture: StereoGeometry, target: StereoGeometry, model_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Controla que el par sea el esperado y lo lleva a la resolución de cómputo."""
    if not frame.rectified:
        raise InvalidInput(f"frame {frame.frame_id}: {model_id} necesita el par rectificado")
    if frame.geometry_id != capture.geometry_id:
        raise IdentityMismatch(
            f"frame en geometría {frame.geometry_id!r}, {model_id} espera {capture.geometry_id!r}"
        )
    capture.require_size(frame.left, "left")
    left, right = frame.left, frame.right
    if target is not capture:
        # INTER_AREA promedia los píxeles que se juntan: reduce sin aliasing. Se aplica
        # a las imágenes, no a mapas de disparidad (esos van por resize_disparity).
        left = cv2.resize(left, target.size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, target.size, interpolation=cv2.INTER_AREA)
    return left, right


def build_result(
    frame: StereoFrame,
    model_id: str,
    geometry: StereoGeometry,
    disparity_px: np.ndarray,
    valid: np.ndarray,
    elapsed_ms: float,
    min_coverage: float,
    reason_if_partial: str | None = None,
) -> DepthResult:
    """Arma el DepthResult: metros desde la geometría, NaN afuera y status con su motivo.

    `reason_if_partial` declara un fallback (p. ej. B4 sin monocular): el resultado sale
    PARTIAL aunque la cobertura alcance, para que quien lo consume lo sepa.
    """
    depth, depth_ok = geometry.disparity_to_depth(disparity_px, valid)
    valid = valid & depth_ok
    coverage = float(valid.mean())
    if not valid.any():
        status = DepthStatus.UNMEASURABLE
        reason = reason_if_partial or f"{model_id} sin píxeles válidos"
    elif reason_if_partial is not None:
        status, reason = DepthStatus.PARTIAL, reason_if_partial
    elif coverage < min_coverage:
        status, reason = DepthStatus.PARTIAL, f"cobertura {coverage:.0%} < {min_coverage:.0%}"
    else:
        status, reason = DepthStatus.VALID, None
    disparity = disparity_px.astype(np.float32, copy=True)
    disparity[~valid] = np.nan
    depth[~valid] = np.nan
    return DepthResult(
        session_id=frame.session_id,
        frame_id=frame.frame_id,
        model_id=model_id,
        geometry_id=geometry.geometry_id,
        disparity_px=disparity,
        valid_mask=valid,
        status=status,
        reason=reason,
        inference_ms=elapsed_ms,
        depth_m=depth,
        doffs=geometry.doffs,
    )


class SgbmEstimator:
    """DepthEstimator de la aplicación: StereoFrame rectificado -> DepthResult.

    Si `compute_size` es menor que la geometría de captura, reduce las dos imágenes con
    INTER_AREA y trabaja en la geometría reescalada (geometry.resized): la disparidad sale
    en píxeles de ESA resolución y el DepthResult lleva su geometry_id.
    """

    def __init__(
        self,
        variant: str,
        params: SgbmParams,
        geometry: StereoGeometry,
        compute_size: tuple[int, int] | None = None,
        min_coverage: float = 0.0,
    ) -> None:
        self.capture_geometry = geometry
        self.geometry = compute_geometry(geometry, compute_size)
        self.matcher = SgbmMatcher(variant, params)
        self.min_coverage = v.finite("min_coverage", min_coverage, low=0.0)
        self.model_id = f"sgbm-{variant.lower()}"

    def predict(self, frame: StereoFrame) -> DepthResult:
        start = time.perf_counter()
        left, right = prepare_pair(frame, self.capture_geometry, self.geometry, self.model_id)
        try:
            out = self.matcher.compute(left, right, self.geometry.doffs)
        except cv2.error as exc:  # fallo del algoritmo, no "no se pudo medir"
            raise EstimatorError(
                f"{self.model_id}: OpenCV falló en frame {frame.frame_id}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return build_result(frame, self.model_id, self.geometry, out.disparity_px, out.valid,
                            elapsed_ms, self.min_coverage)  # fmt: skip

    def close(self) -> None:
        """Sin recursos externos que liberar."""
