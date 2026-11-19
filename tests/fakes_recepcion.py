"""Entregas falsas importables por configuración; nunca se usan como mediciones reales."""

from dataclasses import replace

import numpy as np

from visionx.shared.contracts import DepthResult, DepthStatus, UncertaintyKind

INSTANCES = []


class ExampleEstimator:
    def __init__(self, cfg, geometry):
        self.cfg = cfg
        self._geometry = geometry.resized(*cfg["size"]) if cfg.get("size") else geometry
        if cfg.get("declare_geometry", True):
            self.geometry = self._geometry
        self.model_id = "fake-recepcion"
        self.closed = False
        self.calls = 0
        INSTANCES.append(self)

    def predict(self, frame):
        self.calls += 1
        g = self._geometry
        z = np.full(g.size[::-1], self.cfg.get("z_m", 1.0), np.float32)
        d, valid = g.depth_to_disparity(z)
        uncertainty = np.full(z.shape, 0.2, np.float32) if self.cfg.get("uncertainty") else None
        if self.cfg.get("outside_nan"):
            valid[0] = False
            d[0] = z[0] = np.nan
            if uncertainty is not None:
                uncertainty[0] = np.nan
        result = DepthResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            model_id=self.model_id,
            geometry_id=g.geometry_id,
            disparity_px=d,
            valid_mask=valid,
            status=DepthStatus.VALID,
            inference_ms=np.float32(0.1),
            depth_m=None if self.cfg.get("disparity_only") else z,
            uncertainty=uncertainty,
            uncertainty_kind=UncertaintyKind.LAPLACE_B
            if uncertainty is not None
            else UncertaintyKind.NONE,
            doffs=g.doffs,
        )
        broken = self.cfg.get("broken")
        if broken in {"nan", "inf", "dtype", "units", "meters_as_disparity"}:
            # Productor malicioso que elude el constructor: la recepción debe revalidar.
            values = d.copy()
            if broken in {"nan", "inf"}:
                values[valid] = np.nan if broken == "nan" else np.inf
            elif broken == "dtype":
                values = values.astype(np.float64)
            else:
                values[valid] = z[valid]  # metros escritos en un campo en píxeles
            object.__setattr__(result, "disparity_px", values)
            if broken == "meters_as_disparity":
                object.__setattr__(result, "depth_m", None)
        elif broken == "frame":
            result = replace(result, frame_id=frame.frame_id + 1)
        elif broken == "model" and self.calls > 1:
            result = replace(result, model_id="fake-otro-modelo")
        elif broken == "reason":
            object.__setattr__(result, "status", DepthStatus.PARTIAL)
        elif broken == "uncertainty":
            object.__setattr__(result, "uncertainty", np.full(z.shape, np.nan, np.float32))
            object.__setattr__(result, "uncertainty_kind", UncertaintyKind.LAPLACE_B)
        return result

    def close(self):
        self.closed = True
        if self.cfg.get("broken") == "close":
            raise RuntimeError("fallo de cierre deliberado")


def build_example(cfg, geometry):
    """Firma de entrega de ejemplo: (cfg, geometry) -> DepthEstimator."""
    return ExampleEstimator(cfg, geometry)


def build_broken(cfg, geometry):
    """Por defecto NaN; cfg permite resolución sin geometry y otros fallos de recepción."""
    return ExampleEstimator({"broken": "nan", **cfg}, geometry)
