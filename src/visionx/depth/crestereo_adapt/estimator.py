"""CREStereo con el contrato `DepthEstimator` que ya usa la app.

Etapa 1 del diseño: el estimador recibe el par **ya rectificado** (el que hoy entrega
`RectifiedSource`) y devuelve disparidad y metros. Todavía no corrige la alineación.

Si se le da una U-Net de incertidumbre (`uncertainty.py`), agrega b por píxel al resultado, con
el tipo `LAPLACE_B` que el contrato ya tiene. Sin ella, el resultado sale sin incertidumbre.

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
from visionx.shared.config import ROOT
from visionx.shared.contracts import DepthResult, StereoFrame, UncertaintyKind
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
        uncertainty: Any = None,
    ) -> None:
        self.matcher = matcher
        self.uncertainty = uncertainty  # UncertaintyHead o None
        self.capture_geometry = geometry
        self.geometry = compute_geometry(geometry, compute_size)
        self.min_coverage = min_coverage
        width, height = self.geometry.size
        # Estable: mismos pesos, misma resolución y misma forma de correr dan el mismo id.
        self.model_id = f"crestereo-{weights_id}-{width}x{height}-{matcher.params.tag}"

    def predict(self, frame: StereoFrame) -> DepthResult:
        left, right = prepare_pair(frame, self.capture_geometry, self.geometry, self.model_id)
        start = time.perf_counter()
        b = None
        try:
            disparity = self.matcher.compute(left, right)
            if self.uncertainty is not None:
                extras = self.matcher.last_extras
                b = self.uncertainty.predict(
                    left, right, disparity, extras.get("half"), extras.get("vertical")
                )
        except RuntimeError as exc:  # errores de PyTorch o del dispositivo
            message = f"{self.model_id} falló en el frame {frame.frame_id}: {exc}"
            raise EstimatorError(message) from exc
        elapsed_ms = (time.perf_counter() - start) * 1e3
        # Lo físico es d + doffs > 0 (geometry.py). La red no marca huecos: todo píxel con
        # disparidad finita y positiva es válido. Cuánto creerle lo dice la incertidumbre.
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(disparity) & ((disparity + self.geometry.doffs) > 0)
        if b is None:
            return build_result(
                frame, self.model_id, self.geometry, disparity, valid, elapsed_ms, self.min_coverage
            )
        return build_result(
            frame,
            self.model_id,
            self.geometry,
            disparity,
            valid & np.isfinite(b),
            elapsed_ms,
            self.min_coverage,
            uncertainty=b,
            uncertainty_kind=UncertaintyKind.LAPLACE_B,
        )

    def close(self) -> None:
        """Suelta el cuadro anterior del arranque en caliente. El modelo sigue cargado."""
        self.matcher.reset()


def incertidumbre_de(section: dict[str, Any], mode: str) -> str:
    """Checkpoint de la U-Net para `mode`: el de su sección si lo trae, si no el general."""
    mode_cfg = section.get(mode) or {}
    if "incertidumbre" in mode_cfg:
        return str(mode_cfg["incertidumbre"] or "")
    return str(section.get("incertidumbre") or "")


def load_estimator(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    mode: str = "continuo",
    *,
    model: Any = None,
    device: str | None = None,
    params: CreStereoParams | None = None,
    compute_size: tuple[int, int] | None = None,
    uncertainty: Any = None,
) -> CreStereoEstimator:
    """Arma el estimador desde la config (`depth.crestereo`).

    Si `depth.crestereo.incertidumbre` apunta a un checkpoint, se carga la U-Net de
    incertidumbre y el resultado trae b. La sección del modo puede pisarla con su propia
    `incertidumbre` (p. ej. solo en `preciso`, donde está validada; "" la apaga en ese modo).
    `uncertainty` permite pasar una ya cargada.

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
    checkpoint = incertidumbre_de(section, mode)
    if uncertainty is None and checkpoint:
        from visionx.depth.crestereo_adapt.uncertainty import UncertaintyHead

        uncertainty = UncertaintyHead.load(ROOT / checkpoint, device)
    return CreStereoEstimator(
        CreStereoMatcher(model, device, params),
        geometry,
        size,
        float(depth_cfg.get("min_coverage", 0.0)),
        weights_id=port.weights_sha256[:8],
        uncertainty=uncertainty,
    )
