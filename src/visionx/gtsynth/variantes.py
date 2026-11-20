"""Variantes de cámara sobre las vistas `elp` (decisión (a) de T12).

- `elp`: el par tal cual se renderizó, con la geometría de la ELP.
- `hermana-<f>`: otra unidad del mismo modelo. El ojo derecho se desalinea como lo haría
  rectificar con la calibración de la ELP una cámara real sorteada cerca de ella
  (`sample_sibling` con los rangos de `CameraRanges` × f; generador de etapa 2 de A).
- `distinta`: renderizada aparte (otra focal y otra separación); se mide suponiendo la
  geometría de la ELP, como pasaría al enchufar otra cámara sin calibrarla.

Las hermanas se arman al vuelo con una semilla por vista: misma vista, misma cámara.
"""

from __future__ import annotations

import dataclasses
import zlib
from functools import lru_cache

import numpy as np

from visionx.gtsynth.dataset import Vista

CALIBRACION = "configs/calib/elp-2026-11-11-corregida.yaml"


@lru_cache(maxsize=1)
def _nominal(calibracion: str = CALIBRACION):
    from visionx.calibration.storage import StereoCalibration
    from visionx.depth.crestereo_adapt.perturb import CameraModel
    from visionx.shared.config import ROOT

    cal = StereoCalibration.load(ROOT / calibracion)
    return CameraModel(cal.right.K, cal.right.D, cal.rect.R2)


def es_hermana(variante: str) -> bool:
    return variante.startswith("hermana-")


def fraccion(variante: str) -> float:
    return float(variante.removeprefix("hermana-"))


def derecha_hermana(vista: Vista, variante: str) -> np.ndarray:
    """El ojo derecho de la vista visto por una hermana de la ELP (semilla = la vista)."""
    from visionx.depth.crestereo_adapt.perturb import (
        CameraRanges,
        perturb_right_camera,
        sample_sibling,
    )

    f = fraccion(variante)
    base = CameraRanges()
    rangos = dataclasses.replace(
        base,
        **{k: getattr(base, k) * f for k in ("roll_deg", "pitch_deg", "yaw_deg", "focal",
                                              "center_px", "distortion")},
    )  # fmt: skip
    semilla = zlib.crc32(f"{vista.escena}/{vista.meta['indice']}".encode())
    nominal = _nominal()
    real = sample_sibling(np.random.default_rng(semilla), nominal, rangos)
    return perturb_right_camera(vista.der(), real, nominal, vista.geometria).right
