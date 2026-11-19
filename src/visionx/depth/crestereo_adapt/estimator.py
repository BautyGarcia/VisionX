"""CREStereo con el contrato `DepthEstimator` que ya usa la app.

Etapa 1 del diseño: el estimador recibe el par **ya rectificado** (el que hoy entrega
`RectifiedSource`) y devuelve disparidad y metros. Todavía no corrige la alineación ni estima
incertidumbre: eso lo agregan las redes propias. Sirve para medir latencia y precisión de
CREStereo solo, que es lo que decide la compuerta de T12.

La disparidad sale a la resolución de cómputo, igual que en SGBM y Fusion: los metros se
calculan con la geometría reescalada a esa resolución (`compute_geometry`).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from visionx.depth.crestereo_adapt.matcher import CreStereoMatcher, CreStereoParams
from visionx.depth.crestereo_adapt.port import PortConfig, load_model
from visionx.depth.stereo import build_result, compute_geometry, prepare_pair
from visionx.shared.contracts import DepthResult, StereoFrame
from visionx.shared.errors import EstimatorError, InvalidInput
from visionx.shared.geometry import StereoGeometry

MODES = ("continuo", "preciso")


class CreStereoEstimator:
    """`DepthEstimator` sobre CREStereo congelado."""

    def __init__(
        self,
        matcher: CreStereoMatcher,
        geometry: StereoGeometry,
        compute_size: tuple[int, int] | None = None,
        min_coverage: float = 0.0,
        weights_id: str = "local",
    ) -> None:
        self.matcher = matcher
        self.capture_geometry = geometry
        self.geometry = compute_geometry(geometry, compute_size)
        self.min_coverage = min_coverage
        width, height = self.geometry.size
        # Estable: mismos pesos, misma resolución y misma forma de correr dan el mismo id.
        self.model_id = f"crestereo-{weights_id}-{width}x{height}-{matcher.params.tag}"

    def predict(self, frame: StereoFrame) -> DepthResult:
        left, right = prepare_pair(frame, self.capture_geometry, self.geometry, self.model_id)
        start = time.perf_counter()
        try:
            disparity = self.matcher.compute(left, right)
        except RuntimeError as exc:  # errores de PyTorch o del dispositivo
            message = f"{self.model_id} falló en el frame {frame.frame_id}: {exc}"
            raise EstimatorError(message) from exc
        elapsed_ms = (time.perf_counter() - start) * 1e3
        # Lo físico es d + doffs > 0 (geometry.py). La red no marca huecos: todo píxel con
        # disparidad finita y positiva es válido. Cuánto creerle lo dirá la incertidumbre.
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(disparity) & ((disparity + self.geometry.doffs) > 0)
        return build_result(
            frame, self.model_id, self.geometry, disparity, valid, elapsed_ms, self.min_coverage
        )

    def close(self) -> None:
        """Suelta el cuadro anterior del arranque en caliente. El modelo sigue cargado."""
        self.matcher.reset()


def load_estimator(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    mode: str = "continuo",
    *,
    model: Any = None,
    device: str | None = None,
    params: CreStereoParams | None = None,
    compute_size: tuple[int, int] | None = None,
) -> CreStereoEstimator:
    """Arma el estimador desde la config (`depth.crestereo`).

    `mode` elige la sección `continuo` (cuadro a cuadro) o `preciso` (a pedido). `model` y
    `device` permiten compartir una red ya cargada entre los dos modos; `params` y
    `compute_size` pisan lo de la config (los usa la medición de latencia).
    """
    if mode not in MODES:
        raise InvalidInput(f"modo {mode!r}: se esperaba uno de {MODES}")
    depth_cfg = cfg["depth"]
    section = depth_cfg["crestereo"]
    port = PortConfig.from_config(section)
    if model is None:
        model, device = load_model(port)
    elif device is None:
        raise InvalidInput("con `model` hay que pasar también `device`")
    mode_cfg = section[mode]
    params = params or CreStereoParams.from_config(mode_cfg)
    size = compute_size or tuple(int(x) for x in mode_cfg["compute"])
    return CreStereoEstimator(
        CreStereoMatcher(model, device, params),
        geometry,
        size,
        float(depth_cfg.get("min_coverage", 0.0)),
        weights_id=port.weights_sha256[:8],
    )
