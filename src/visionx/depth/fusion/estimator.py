"""Fusion en la aplicación (T12): par rectificado -> DepthResult con b de Laplace.

Mismo camino que el entrenamiento, en vivo: SGBM (B2) -> monocular -> alineación -> base B4
o B5 (`channels.fusion_bases`, la misma función que usa el entrenamiento) -> canales
(`channels.build_inputs`, el mismo preprocesamiento público) -> red del paquete. Lo único
distinto del entrenamiento es de dónde salen los ingredientes caros: acá se calculan; al
entrenar se leen de la caché. Un test comprueba que los dos caminos dan la misma disparidad
(paridad offline/aplicación).

- `FusionMatcher`: núcleo sobre un par ya preparado (como `HybridMatcher`).
- `FusionEstimator`: `DepthEstimator` de la app (como `HybridEstimator`), con
  `uncertainty` = b en px a la resolución de cómputo y `UncertaintyKind.LAPLACE_B`.
- `load_estimator`: arma todo desde la config (`depth.sgbm`, `depth.alineacion`, `depth.mono`,
  `depth.b4`, `depth.b5` si el paquete la usa, `depth.fusion.paquete`), así la app cambia B4
  por Fusion solo por configuración.

Validez: un píxel vale si la base respondió y, después de sumar el residual, `d + doffs > 0`. Si no,
se invalida; no se recorta a un valor físico (eso inventaría una medida).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from visionx.depth.align import AlignParams
from visionx.depth.fusion.cache import FUSION_STEREO_VARIANTS
from visionx.depth.fusion.channels import (
    FusionInputs,
    build_inputs,
    disparity_from_residual,
    fusion_bases,
    fusion_validity,
)
from visionx.depth.fusion.package import (
    FusionPackage,
    base_differences,
    config_base_differences,
    load_package,
    mono_identity_of,
)
from visionx.depth.hybrid import SOURCE_MONO, SOURCE_STEREO, B5Params, HybridOutput
from visionx.depth.mono import MonoModel, check_relative
from visionx.depth.stereo import (
    SgbmMatcher,
    SgbmParams,
    build_result,
    compute_geometry,
    num_disparities_for,
    prepare_pair,
)
from visionx.shared import _validate as v
from visionx.shared.config import ROOT
from visionx.shared.contracts import DepthResult, StereoFrame, UncertaintyKind
from visionx.shared.errors import EstimatorError, InvalidInput
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True)
class FusionOutput:
    """Disparidad corregida, su incertidumbre y la base (B4 o B5) para diagnosticar."""

    disparity_px: np.ndarray  # H x W float32, NaN fuera de valid
    valid: np.ndarray  # H x W bool: base válida Y d + doffs > 0 tras el residual
    b_px: np.ndarray  # H x W float32: b de Laplace en px (> 0 dentro de valid, NaN afuera)
    base: HybridOutput  # la base con la que se corrió (su alineación y su fallback)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def fallback(self) -> str | None:
        """Motivo si la base no pudo usar el monocular (Fusion corrió sobre solo estéreo)."""
        return self.base.fallback


class FusionMatcher:
    """Núcleo de Fusion sobre un par rectificado ya preparado (tamaño de cómputo)."""

    def __init__(
        self,
        stereo: SgbmMatcher,
        mono: MonoModel,
        align_params: AlignParams,
        package: FusionPackage,
        allow_other_base: bool = False,
        b5_params: B5Params | None = None,
    ) -> None:
        """`align_params`: la alineación de B4 (`depth.alineacion`). `b5_params`: `depth.b5`;
        obligatorio si el paquete usa B5 (base B5 o canales F3)."""
        if stereo.variant not in FUSION_STEREO_VARIANTS:
            raise InvalidInput(
                f"estéreo {stereo.variant!r}: Fusion necesita {FUSION_STEREO_VARIANTS} "
                "(B0 no calcula el error L/R)"
            )
        self.stereo = stereo
        self.mono = mono
        self.align_params = align_params
        self.package = package
        self.variant = package.variant
        self.base_variant = package.base_variant
        if package.uses_b5 and b5_params is None:
            raise InvalidInput(f"{package.model_id}: usa B5 ({self.variant}, base "
                               f"{self.base_variant}) y la config no tiene depth.b5")  # fmt: skip
        self.b5_params = b5_params
        self.channel_params = package.channel_params
        self.device = torch.device(package.device)
        # La base es parte del modelo: con otra base, el residual aprendido no corresponde.
        self.base_differences = base_differences(
            package.base,
            stereo.variant,
            stereo.params,
            align_params,
            mono_identity_of(mono),
            b5_params,
            package.uses_b5,
        )
        if self.base_differences and not allow_other_base:
            raise InvalidInput(
                f"{package.model_id}: la base en uso difiere de la del entrenamiento "
                f"({'; '.join(self.base_differences)}); usar allow_other_base solo a propósito"
            )
        if self.base_differences:
            warnings.warn(f"{package.model_id}: base distinta de la del entrenamiento: "
                          f"{'; '.join(self.base_differences)}", stacklevel=2)  # fmt: skip

    def infer(self, inputs: FusionInputs) -> tuple[np.ndarray, np.ndarray]:
        """Red sobre el frame completo: (residual delta, b), los dos H x W en d/W_ref.

        La red está en `eval()` (lo garantiza `load_package`): con BatchNorm usa sus medias
        móviles y la salida es local; en train() dependería del frame.
        """
        model = self.package.model
        if model.training:
            raise InvalidInput(f"{self.package.model_id}: la red tiene que estar en eval()")
        with torch.inference_mode():
            channels = torch.from_numpy(inputs.channels)[None].to(self.device)
            base = torch.from_numpy(inputs.base_norm)[None, None].to(self.device)
            out = model(channels, base)
            # .cpu() espera a que la GPU termine: el tiempo de la red queda bien medido.
            delta = out["delta"][0, 0].float().cpu().numpy()
            b = out["b"][0, 0].float().cpu().numpy()
        return delta, b

    def compute(self, left: np.ndarray, right: np.ndarray, doffs: float = 0.0) -> FusionOutput:
        doffs = v.finite("doffs", doffs)
        t0 = time.perf_counter()
        stereo = self.stereo.compute(left, right, doffs)
        t1 = time.perf_counter()
        relative = check_relative(self.mono.predict(left), stereo.disparity_px.shape)
        t2 = time.perf_counter()
        need_b4 = self.base_variant == "B4" or self.variant == "F3"
        b4, b5 = fusion_bases(stereo, relative, doffs, self.align_params, self.b5_params,
                              need_b4, self.package.uses_b5)  # fmt: skip
        base = b4 if self.base_variant == "B4" else b5
        t3 = time.perf_counter()
        inputs = build_inputs(left, right, stereo, base, relative, doffs, self.variant,
                              self.channel_params, b4=b4, b5=b5)  # fmt: skip
        t4 = time.perf_counter()
        delta_norm, b_norm = self.infer(inputs)
        t5 = time.perf_counter()

        # Base en px originales + residual · W_ref (la misma cuenta que train.evaluate): la base
        # no pasa por d/W_ref, así una red con delta 0 reproduce la base exactamente.
        width = float(inputs.ref_width)
        disparity = disparity_from_residual(base.disparity_px, inputs.base_valid, delta_norm,
                                            width)  # fmt: skip
        b_px = (b_norm * np.float32(width)).astype(np.float32)
        valid = fusion_validity(disparity, b_px, inputs.base_valid, doffs)
        n_base = int(inputs.base_valid.sum())
        delta_px = np.abs(delta_norm)[valid] * width
        stats = {
            "cobertura": float(valid.mean()),
            "cobertura_base": base.stats["cobertura"],
            "fraccion_estereo": float((valid & (base.source == SOURCE_STEREO)).mean()),
            "fraccion_mono": float((valid & (base.source == SOURCE_MONO)).mean()),
            # Píxeles con base que el residual llevó a d + doffs <= 0 (quedan inválidos).
            "invalidada_por_residual": float((inputs.base_valid & ~valid).sum()) / max(1, n_base),
            "delta_abs_mediana_px": float(np.median(delta_px)) if valid.any() else 0.0,
            "b_mediana_px": float(np.median(b_px[valid])) if valid.any() else 0.0,
            "alin_aceptada": float(base.alignment.accepted),
            "ms_estereo": (t1 - t0) * 1e3,
            "ms_mono": (t2 - t1) * 1e3,
            "ms_alineacion": (t3 - t2) * 1e3,  # recta(s) global(es) y, con B5, la corrección local
            "ms_canales": (t4 - t3) * 1e3,
            "ms_red": (t5 - t4) * 1e3,
        }  # fmt: skip
        return FusionOutput(
            disparity_px=np.where(valid, disparity, np.nan).astype(np.float32),
            valid=valid,
            b_px=np.where(valid, b_px, np.nan).astype(np.float32),
            base=base,
            stats=stats,
        )


class FusionEstimator:
    """DepthEstimator de la app para Fusion. `predict_detailed` agrega el diagnóstico."""

    def __init__(
        self,
        matcher: FusionMatcher,
        geometry: StereoGeometry,
        compute_size: tuple[int, int] | None = None,
        min_coverage: float = 0.0,
    ) -> None:
        self.matcher = matcher
        self.capture_geometry = geometry
        self.geometry = compute_geometry(geometry, compute_size)
        self.min_coverage = v.finite("min_coverage", min_coverage, low=0.0)
        self.model_id = matcher.package.model_id

    def predict_detailed(self, frame: StereoFrame) -> tuple[DepthResult, FusionOutput]:
        start = time.perf_counter()
        left, right = prepare_pair(frame, self.capture_geometry, self.geometry, self.model_id)
        try:
            out = self.matcher.compute(left, right, self.geometry.doffs)
        except cv2.error as exc:
            raise EstimatorError(
                f"{self.model_id}: OpenCV falló en frame {frame.frame_id}"
            ) from exc
        except RuntimeError as exc:  # fallos de torch (memoria, backend) son del algoritmo
            if isinstance(exc, EstimatorError):
                raise
            raise EstimatorError(
                f"{self.model_id}: la red falló en frame {frame.frame_id}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # b viaja en px de la resolución de cómputo, como la disparidad (b_norm · W_ref).
        result = build_result(frame, self.model_id, self.geometry, out.disparity_px, out.valid,
                              elapsed_ms, self.min_coverage, out.fallback, uncertainty=out.b_px,
                              uncertainty_kind=UncertaintyKind.LAPLACE_B)  # fmt: skip
        return result, out

    def predict(self, frame: StereoFrame) -> DepthResult:
        return self.predict_detailed(frame)[0]

    def close(self) -> None:
        """La red y el monocular viven mientras viva el estimador; no hay archivos abiertos."""


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path  # relativa al repo, no al cwd


def load_estimator(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    compute_size: tuple[int, int] | None = None,
    *,
    package: str | Path | None = None,
    mono: MonoModel | None = None,
    device: str | None = None,
    allow_other_base: bool = False,
) -> FusionEstimator:
    """Arma un `FusionEstimator` desde la config completa (la de `load_config`).

    - Paquete: `package` o `depth.fusion.paquete` (relativo al repo). Vacío = error claro.
    - `compute_size`: (ancho, alto); None = `depth.compute`. El rango de SGBM sale de la
      geometría de cómputo para ver desde `depth.range_m[0]`, como en T05 (CLI de latencia).
    - Estéreo `depth.b4.estereo`, SGBM `depth.sgbm`, alineación `depth.alineacion`,
      `depth.b5` (si el paquete usa B5), monocular `depth.mono` (o `mono`, p. ej. un falso en
      tests), cobertura mínima `depth.min_coverage`.
    - `device`: de la red; None = `depth.fusion.dispositivo` si existe, si no "auto".
    - Si la base de la config difiere de la del entrenamiento: error, salvo `allow_other_base`.
    - Variante y canales salen del paquete (son parte del modelo); si `depth.fusion.variante`
      dice otra cosa, se avisa.
    """
    depth = cfg["depth"]
    fusion = depth.get("fusion") or {}
    pkg_path = package or fusion.get("paquete")
    if not pkg_path:
        raise InvalidInput("depth.fusion.paquete está vacío: indicar la carpeta de un paquete")
    pkg = load_package(_resolve(pkg_path), device or fusion.get("dispositivo", "auto"))
    if not allow_other_base:
        # Antes de cargar Depth Anything: si la base no corresponde, fallar rápido.
        diffs = config_base_differences(pkg.base, depth)
        if diffs:
            raise InvalidInput(f"{pkg.model_id}: la config difiere de la base del entrenamiento "
                               f"({'; '.join(diffs)})")  # fmt: skip
    if fusion.get("variante") and fusion["variante"] != pkg.variant:
        warnings.warn(f"depth.fusion.variante={fusion['variante']!r} pero el paquete es "
                      f"{pkg.variant}: manda el paquete", stacklevel=2)  # fmt: skip

    if compute_size is None and depth.get("compute"):
        compute_size = (int(depth["compute"][0]), int(depth["compute"][1]))
    geom = compute_geometry(geometry, compute_size)
    ndisp = num_disparities_for(geom, float(depth["range_m"][0]))
    stereo = SgbmMatcher(depth["b4"]["estereo"],
                         SgbmParams.from_config(depth["sgbm"], num_disparities=ndisp))  # fmt: skip
    if mono is None:
        from visionx.depth.mono import DepthAnythingV2, MonoConfig

        mono = DepthAnythingV2(MonoConfig.from_config(depth["mono"]))
    b5 = None
    if pkg.uses_b5:
        if "b5" not in depth:
            raise InvalidInput(f"{pkg.model_id}: usa B5 y la config no tiene depth.b5")
        b5 = B5Params.from_config(depth)
    matcher = FusionMatcher(stereo, mono, AlignParams.from_config(depth["alineacion"]), pkg,
                            allow_other_base, b5)  # fmt: skip
    return FusionEstimator(matcher, geometry, compute_size, float(depth.get("min_coverage", 0.0)))
