"""Controles que combinan estéreo y monocular (T05): B3 y B4.

- **B3:** el mapa monocular llevado a la escala estéreo con las anclas (align.py). No es un
  monocular métrico independiente: su escala sale del estéreo de ese mismo par. Si la
  alineación no se acepta, B3 no responde.
- **B4:** regla fija y escrita a mano. Estéreo donde es fiable, monocular alineado en el
  resto, y nada donde fallan los dos. Si la alineación no se acepta, queda solo el estéreo
  y el resultado se declara PARTIAL con el motivo.

B4 es el control clave del trabajo: ya combina las dos fuentes. Si Fusion mejora SGBM pero
no B4, no se demostró que aprender la combinación agregue algo.

Las anclas de la alineación son los píxeles válidos de la variante estéreo elegida en la
config (`depth.b4.estereo`, B1 de inicio: B2 rellena con WLS y esos valores rellenados no son
medidas independientes). Cuál variante queda se congela con validación (T05 PR 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionx.depth.align import Alignment, AlignParams, align
from visionx.depth.mono import MonoModel, check_relative
from visionx.depth.stereo import (
    SgbmMatcher,
    SgbmParams,
    StereoOutput,
    build_result,
    compute_geometry,
    prepare_pair,
)
from visionx.shared import _validate as v
from visionx.shared.contracts import DepthResult, StereoFrame
from visionx.shared.errors import EstimatorError, InvalidInput
from visionx.shared.geometry import StereoGeometry

HYBRID_VARIANTS = ("B3", "B4")
SOURCE_NONE, SOURCE_STEREO, SOURCE_MONO = 0, 1, 2  # valores del mapa `source`


@dataclass(frozen=True)
class HybridOutput:
    """Disparidad combinada y todo lo necesario para diagnosticarla."""

    disparity_px: np.ndarray  # H x W float32, NaN fuera de valid
    valid: np.ndarray  # H x W bool
    source: np.ndarray  # H x W uint8: 0 nada, 1 estéreo, 2 monocular alineado
    alignment: Alignment
    fallback: str | None  # motivo si B3/B4 no pudieron usar el monocular
    stats: dict[str, float] = field(default_factory=dict)


def combine(
    variant: str,
    stereo: StereoOutput,
    mono_relative: np.ndarray,
    doffs: float,
    params: AlignParams,
) -> HybridOutput:
    """B3 o B4 a partir de una salida estéreo y un mapa monocular del mismo par."""
    if variant not in HYBRID_VARIANTS:
        raise InvalidInput(f"variante {variant!r}: se esperaba una de {HYBRID_VARIANTS}")
    m = check_relative(mono_relative, stereo.disparity_px.shape)
    alignment = align(m, stereo.disparity_px, stereo.valid, params)

    shape = m.shape
    disparity = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, bool)
    source = np.zeros(shape, np.uint8)
    fallback = None

    if alignment.accepted:
        aligned = alignment.apply(m)
        # Igual que en el estéreo: lo físico es d + doffs > 0.
        aligned_ok = np.isfinite(aligned) & ((aligned.astype(np.float64) + doffs) > 0)
    else:
        aligned, aligned_ok = None, np.zeros(shape, bool)
        fallback = f"monocular sin escala fiable: {alignment.reason}"

    if variant == "B4":
        use_stereo = stereo.valid
        disparity[use_stereo] = stereo.disparity_px[use_stereo]
        source[use_stereo] = SOURCE_STEREO
        use_mono = aligned_ok & ~use_stereo
        if fallback is not None:
            fallback = f"solo estéreo; {fallback}"
    else:  # B3: todo monocular alineado
        use_mono = aligned_ok
    if aligned is not None:
        disparity[use_mono] = aligned[use_mono]
        source[use_mono] = SOURCE_MONO
    valid = source != SOURCE_NONE

    total = valid.size
    stats = {
        "cobertura": float(valid.sum()) / total,
        "fraccion_estereo": float((source == SOURCE_STEREO).sum()) / total,
        "fraccion_mono": float((source == SOURCE_MONO).sum()) / total,
    }
    return HybridOutput(disparity, valid, source, alignment, fallback, stats)


class HybridMatcher:
    """Núcleo offline de B3/B4: SGBM + monocular + alineación sobre un par ya preparado."""

    def __init__(
        self,
        variant: str,
        stereo_variant: str,
        sgbm_params: SgbmParams,
        mono: MonoModel,
        align_params: AlignParams,
    ) -> None:
        if variant not in HYBRID_VARIANTS:
            raise InvalidInput(f"variante {variant!r}: se esperaba una de {HYBRID_VARIANTS}")
        self.variant = variant
        self.stereo_variant = stereo_variant
        self.stereo = SgbmMatcher(stereo_variant, sgbm_params)
        self.mono = mono
        self.align_params = align_params

    def compute(self, left: np.ndarray, right: np.ndarray, doffs: float = 0.0) -> HybridOutput:
        t0 = time.perf_counter()
        stereo = self.stereo.compute(left, right, doffs)
        t1 = time.perf_counter()
        relative = self.mono.predict(left)  # el monocular mira el ojo izquierdo, como la GT
        t2 = time.perf_counter()
        out = combine(self.variant, stereo, relative, doffs, self.align_params)
        t3 = time.perf_counter()
        out.stats.update(ms_estereo=(t1 - t0) * 1e3, ms_mono=(t2 - t1) * 1e3,
                         ms_alineacion=(t3 - t2) * 1e3)  # fmt: skip
        return out


class HybridEstimator:
    """DepthEstimator de la app para B3/B4. `predict_detailed` agrega el diagnóstico."""

    def __init__(
        self,
        matcher: HybridMatcher,
        geometry: StereoGeometry,
        compute_size: tuple[int, int] | None = None,
        min_coverage: float = 0.0,
    ) -> None:
        self.matcher = matcher
        self.capture_geometry = geometry
        self.geometry = compute_geometry(geometry, compute_size)
        self.min_coverage = v.finite("min_coverage", min_coverage, low=0.0)
        variant, stereo = matcher.variant.lower(), matcher.stereo_variant.lower()
        self.model_id = f"{variant}-sgbm-{stereo}-{matcher.mono.model_id}"

    def predict_detailed(self, frame: StereoFrame) -> tuple[DepthResult, HybridOutput]:
        start = time.perf_counter()
        left, right = prepare_pair(frame, self.capture_geometry, self.geometry, self.model_id)
        try:
            out = self.matcher.compute(left, right, self.geometry.doffs)
        except cv2.error as exc:
            raise EstimatorError(
                f"{self.model_id}: OpenCV falló en frame {frame.frame_id}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # alignment_quality del contrato queda en None a propósito: el diagnóstico tiene
        # varios números y vive en HybridOutput.alignment (TECNICO, "Alineación robusta").
        result = build_result(frame, self.model_id, self.geometry, out.disparity_px, out.valid,
                              elapsed_ms, self.min_coverage, out.fallback)  # fmt: skip
        return result, out

    def predict(self, frame: StereoFrame) -> DepthResult:
        return self.predict_detailed(frame)[0]

    def close(self) -> None:
        """El modelo monocular vive mientras viva el estimador; no hay archivos abiertos."""
