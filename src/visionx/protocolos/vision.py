"""Adaptadores offline. Único punto de construcción por nombre: build_estimator."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from .common import number, sha


def build_estimator(name: str, cfg: dict, geometry):
    """Unificar con el registro de app cuando esté disponible; no importar CLI privados."""
    from visionx.depth.stereo import SgbmEstimator, SgbmParams

    if name not in ("B0", "B1", "B2", "B3", "B4"):
        raise ValueError(f"método desconocido: {name}")
    d = cfg["depth"]
    size = tuple(d["compute"]) if d.get("compute") else None
    params = SgbmParams.from_config(d.get("sgbm") or {})
    coverage = float(d.get("min_coverage", 0))
    if name in ("B0", "B1", "B2"):
        return SgbmEstimator(name, params, geometry, size, coverage)
    from visionx.depth.align import AlignParams
    from visionx.depth.hybrid import HybridEstimator, HybridMatcher
    from visionx.depth.mono import DepthAnythingV2, MonoConfig

    mono = DepthAnythingV2(MonoConfig.from_config(d["mono"]))
    matcher = HybridMatcher(
        name, d["b4"]["estereo"], params, mono, AlignParams.from_config(d["alineacion"])
    )
    return HybridEstimator(matcher, geometry, size, coverage)


def sample_entries(clip: Path, t_s: float, window_s: float, samples: int):
    from visionx.camera.replay import load_recording

    number(t_s, "t_s")
    number(window_s, "ventana")
    if samples < 1:
        raise ValueError("frames debe ser >=1")
    manifest, entries, truncated = load_recording(clip)
    if truncated or not entries or len(entries) != manifest.get("pares"):
        raise ValueError(f"{clip}: índice vacío o incompleto")
    origin = entries[0]["received_ns"]
    end_s = (entries[-1]["received_ns"] - origin) / 1e9
    if t_s > end_s:
        raise ValueError(f"t_s={t_s} fuera del clip de {end_s:.3f} s")
    nearest = min(entries, key=lambda e: abs((e["received_ns"] - origin) / 1e9 - t_s))
    candidates = [
        e for e in entries if abs((e["received_ns"] - origin) / 1e9 - t_s) <= window_s / 2
    ]
    if not candidates:
        candidates = [nearest]
    if samples == 1:
        chosen = [min(candidates, key=lambda e: abs((e["received_ns"] - origin) / 1e9 - t_s))]
    else:
        indices = (
            np.linspace(0, len(candidates) - 1, min(samples, len(candidates))).round().astype(int)
        )
        chosen = [candidates[i] for i in sorted(set(indices))]
    return manifest, chosen


class OfflineVision:
    def __init__(self, cfg: dict, *, detection=True):
        from visionx.calibration import StereoCalibration
        from visionx.detection import DetectorSettings, YoloDetector
        from visionx.shared.config import ROOT

        self.cfg = cfg
        path = Path(cfg["camera"]["calibration"])
        path = path if path.is_absolute() else ROOT / path
        self.calibration = StereoCalibration.load(path)
        self.geometry = self.calibration.geometry()
        self.maps = self.calibration.rectify_maps()
        settings = dataclasses.replace(DetectorSettings.from_config(cfg), track=False)
        self.detector = YoloDetector(settings) if detection else None
        self.inputs = [path] + ([settings.weights_path] if detection else [])
        self.estimators = {}

    def frames(self, clip: Path, t_s: float, window_s: float, samples: int, *, rectified=True):
        from visionx.shared.contracts import StereoFrame

        manifest, entries = sample_entries(clip, t_s, window_s, samples)
        self.inputs.extend([clip / "manifest.yaml", clip / "frames.jsonl"])
        for e in entries:
            images = []
            for eye in ("left", "right"):
                p = clip / e[eye]
                if sha(p) != e[f"sha256_{eye}"]:
                    raise ValueError(f"hash incorrecto: {p}")
                self.inputs.append(p)
                image = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"imagen ilegible: {p}")
                images.append(image)
            if tuple(images[0].shape[1::-1]) != tuple(self.calibration.image_size):
                raise ValueError("tamaño de imagen incompatible con calibración")
            already = bool(manifest.get("rectificado"))
            if already and manifest.get("geometry_id") != self.geometry.geometry_id:
                raise ValueError("clip rectificado con otra geometría")
            if rectified and not already:
                images = self.calibration.rectify(*images, self.maps)
            rect = rectified or already
            yield StereoFrame(
                session_id=manifest["session_id"],
                frame_id=int(e["frame_id"]),
                left=images[0],
                right=images[1],
                received_ns=int(e["received_ns"]),
                replay_original_ns=int(e["received_ns"]),
                rectified=rect,
                calibration_id=self.calibration.calibration_id if rect else None,
                geometry_id=self.geometry.geometry_id if rect else None,
                source_id="protocolo-replay",
            )

    def observe(self, frame, methods):
        from visionx.app.spatial import ProvisionalSpatialEstimator

        detection = self.detector.predict(frame)
        results = {}
        for name in methods:
            if name not in self.estimators:
                self.estimators[name] = build_estimator(name, self.cfg, self.geometry)
            estimator = self.estimators[name]
            depth = estimator.predict(frame)
            # Mismos parámetros y misma política que build_spatial de la app.
            spatial = ProvisionalSpatialEstimator(
                self.geometry,
                depth_geometry=estimator.geometry,
                **(self.cfg.get("spatial_provisional") or {}),
            )
            results[name] = (depth, spatial.locate(frame, depth, detection))
        return detection, results

    def close(self):
        if self.detector is not None:
            self.detector.close()
        for estimator in self.estimators.values():
            estimator.close()
