"""Muestras sintéticas y monocular falso para los tests de Fusion (caché, datos, entrenamiento).

No es un archivo de tests: lo importan test_fusion_cache.py y test_fusion_train.py.

Escena: textura al azar con disparidad que crece hacia abajo (8 a 19 px), como en
test_depth_tuning.py. Las imágenes se escriben como PNG en `<raíz>/sint/<id>/` porque la clave
de la caché hashea los archivos. La GT puede llevar un sesgo constante (`gt_bias_px`): la
"verdad" difiere de lo que muestran las imágenes y B4 queda con un error que Fusion puede
aprender a corregir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visionx.depth.align import AlignParams
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.fusion.cache import CacheSpec
from visionx.depth.stereo import SgbmParams

H, W = 72, 128
NUM_DISPARITIES = 32  # regla de rango fija para todas las muestras
DATASET = "sint"
ALIGN = AlignParams(grid_cols=4, grid_rows=3, min_anchors_per_cell=5)
SGBM = SgbmParams(block_size=5)
MONO_ID = {"model_id": "falso"}


def disparity_rows() -> np.ndarray:
    return 8 + (np.arange(H) * 12 // H)


class FakeMono:
    """Monocular exacto: m = (d - 10) / 4 para la escena sintética (no mira la imagen)."""

    model_id = "falso"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        self.calls += 1
        d = np.repeat(disparity_rows()[:, None], W, 1).astype(np.float32)
        return ((d - 10.0) / 4.0).astype(np.float32)


def synthetic_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Par rectificado: la derecha es la izquierda corrida d px por fila."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (H // 2, W // 2, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left, 2, 0), 2, 1)
    d_rows = disparity_rows()
    right = np.stack([np.roll(left[y], -int(d_rows[y]), axis=0) for y in range(H)])
    return left, right


def write_images(root: Path, sample_id: str, seed: int) -> None:
    left, right = synthetic_pair(seed)
    folder = root / DATASET / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "left.png"), left)
    cv2.imwrite(str(folder / "right.png"), right)


def write_instereo2k(data_root: Path, frame: int, seed: int, gt_bias_px: float = 0.0) -> None:
    """La misma escena con la estructura real de InStereo2K (para el catálogo):
    `<data_root>/instereo2k/train/part1/<cuadro>/{left,right,left_disp}.png`, GT x100."""
    base = data_root / "instereo2k"
    folder = base / "train" / "part1" / f"{frame:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    (base / "test").mkdir(exist_ok=True)
    left, right = synthetic_pair(seed)
    d_rows = disparity_rows()
    d = np.repeat(d_rows[:, None], W, 1).astype(np.float64) + gt_bias_px
    valid = np.arange(W)[None, :] >= d_rows[:, None]
    cv2.imwrite(str(folder / "left.png"), left)
    cv2.imwrite(str(folder / "right.png"), right)
    cv2.imwrite(
        str(folder / "left_disp.png"), np.where(valid, np.rint(d * 100), 0).astype(np.uint16)
    )


@dataclass(frozen=True)
class SynthRef:
    """Cumple la interfaz de SampleRef que usa FusionDataset (sample_id, scene_id, load)."""

    root: Path
    sample_id: str
    scene_id: str
    gt_bias_px: float = 0.0
    gt_valid_from_x: int = 0  # GT solo desde esta columna (para probar el reintento de recorte)

    @property
    def files(self) -> tuple[str, str]:
        return (f"{self.sample_id}/left.png", f"{self.sample_id}/right.png")

    def load(self) -> StereoSample:
        folder = self.root / DATASET / self.sample_id
        left = cv2.imread(str(folder / "left.png"), cv2.IMREAD_COLOR)
        right = cv2.imread(str(folder / "right.png"), cv2.IMREAD_COLOR)
        d_rows = disparity_rows()
        disparity = np.repeat(d_rows[:, None], W, 1).astype(np.float32) + self.gt_bias_px
        cols = np.arange(W)[None, :]
        valid = (cols >= d_rows[:, None]) & (cols >= self.gt_valid_from_x)
        disparity[~valid] = np.nan
        return StereoSample(DATASET, self.sample_id, self.scene_id, left, right, disparity,
                            valid, files=self.files)  # fmt: skip


def make_refs(root: Path, n: int, start: int = 0, **kwargs) -> list[SynthRef]:
    refs = []
    for i in range(start, start + n):
        sample_id = f"s{i:03d}"
        write_images(root, sample_id, seed=i)
        refs.append(SynthRef(root, sample_id, f"escena{i % 2}", **kwargs))
    return refs


def make_spec(cache_root: Path, **sgbm_changes) -> CacheSpec:
    from dataclasses import replace

    return CacheSpec(cache_root, replace(SGBM, **sgbm_changes), ALIGN, MONO_ID, "B2")
