"""Datos para entrenar la red de alineación: pares desalineados de forma conocida.

No hace falta disparidad de referencia. Se toma un par bien rectificado, se inventa una cámara
«real» un poco distinta de la que describe la calibración (giro, aumento, centro y distorsión
del ojo derecho; generador de etapa 2, `perturb.py`) y se desalinea la imagen derecha como lo
haría esa diferencia. La respuesta correcta es el campo que deshace esa desalineación.

Dos fuentes, con papeles distintos:

    instereo2k   pares de otra cámara. Para cada muestra se sortea una cámara virtual (focal,
                 centro y distorsión): así la red ve muchas cámaras y el condicionamiento por
                 calibración tiene algo que aprender.
    elp          clips de desarrollo de nuestra cámara, rectificados con su calibración. La
                 cámara nominal es la ELP y las «reales» son hermanas: otras unidades posibles.
                 A veces se oculta la calibración: «sin dato» significa «cámara tipo ELP».

Riesgo declarado: en los clips de la ELP, el error que haya en nuestra propia calibración queda
como parte de la imagen «limpia» y la red lo aprende como normal.

El campo se calcula en una grilla de 4 px y se interpola: es suave y así cuesta 16 veces menos.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.depth.crestereo_adapt.alignnet import (
    calibration_vector,
    jacobian_maps,
    ray_maps,
)
from visionx.depth.crestereo_adapt.perturb import (
    CameraModel,
    CameraRanges,
    _rectified_matrix,
    _to_raw,
    _to_rectified,
    sample_sibling,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

INSTEREO_SIZE = (544, 432)  # 1080x860 a la mitad, llevado a múltiplos de 8
ELP_SIZE = (960, 536)  # 1920x1080 a la mitad, llevado a múltiplos de 8
COARSE = 4  # paso de la grilla donde se calcula el campo exacto


def _rng(*parts: Any) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def coarse_field_and_maps(
    true: CameraModel, nominal: CameraModel, geometry: StereoGeometry, stride: int = COARSE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lo mismo que `mismatch_field` y `mismatch_maps`, calculado en una grilla y interpolado.

    Devuelve (campo (H, W, 2), map_x, map_y). El campo dice, para cada píxel de la derecha
    limpia, cuánto se corrió en la perturbada; los mapas, para cada píxel de la perturbada, de
    dónde sale en la limpia.
    """
    width, height = geometry.size
    nx, ny = math.ceil((width - 1) / stride) + 1, math.ceil((height - 1) / stride) + 1
    xs, ys = np.linspace(0, width - 1, nx), np.linspace(0, height - 1, ny)
    gu, gv = np.meshgrid(xs, ys)
    nodes = np.column_stack([gu.ravel(), gv.ravel()])
    rect_k = _rectified_matrix(geometry)
    moved = _to_rectified(_to_raw(nodes, true, rect_k), nominal, rect_k)
    source = _to_rectified(_to_raw(nodes, nominal, rect_k), true, rect_k)
    coarse_field = (moved - nodes).reshape(ny, nx, 2).astype(np.float32)
    coarse_source = source.reshape(ny, nx, 2).astype(np.float32)
    # Interpolación bilineal exacta sobre los nodos: posición de cada píxel en la grilla.
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    at_x, at_y = u * (nx - 1) / (width - 1), v * (ny - 1) / (height - 1)

    def up(array: np.ndarray) -> np.ndarray:
        return cv2.remap(array, at_x, at_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return up(coarse_field), up(coarse_source[..., 0]), up(coarse_source[..., 1])


@dataclass(frozen=True)
class SampleSpec:
    """Cómo se desalinea una muestra. Todo sale de una semilla: la muestra se puede repetir."""

    strength: float  # fracción de los rangos de `CameraRanges` (0 = sin desalinear)
    calibration_known: bool
    gain_left: float = 1.0
    gain_right: float = 1.0


def sample_spec(rng: np.random.Generator, elp: bool) -> SampleSpec:
    clean = rng.random() < 0.15  # algunos pares sin tocar: la red tiene que saber no hacer nada
    strength = 0.0 if clean else float(rng.random() ** 1.5)  # más casos chicos que grandes
    known = (not elp) or rng.random() >= 0.3
    gains = rng.uniform(0.85, 1.15, size=2)
    return SampleSpec(strength, bool(known), float(gains[0]), float(gains[1]))


def virtual_camera(rng: np.random.Generator, size: tuple[int, int]) -> tuple[CameraModel, Any]:
    """Una cámara nominal inventada para pares sin calibración: focal, centro y distorsión."""
    width, height = size
    focal = width * float(rng.uniform(0.55, 1.1))
    cx = width / 2.0 + float(rng.uniform(-0.03, 0.03)) * width
    cy = height / 2.0 + float(rng.uniform(-0.03, 0.03)) * height
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    if rng.random() < 0.5:  # modelo simple de 5 coeficientes
        d = np.array([rng.uniform(-0.3, 0.15), rng.uniform(-0.1, 0.1), 0.0, 0.0, 0.0])
    else:  # modelo racional de 8, con la forma de la ELP y amplitud sorteada
        elp = np.array([0.03, 1.7, 0.0003, 0.0, 0.09, 0.08, 1.6, 0.2])
        d = elp * rng.uniform(0.3, 1.2)
    geometry = StereoGeometry(
        width=width, height=height, fx=focal, fy=focal, cx=cx, cy=cy, baseline_m=0.1
    )
    return CameraModel(k, d, np.eye(3)), geometry


def build_item(
    left: np.ndarray,
    right: np.ndarray,
    nominal: CameraModel,
    geometry: StereoGeometry,
    spec: SampleSpec,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Una muestra lista para la red, en NumPy. `left` y `right` ya al tamaño de `geometry`."""
    geometry.require_size(left, "left")
    ranges = CameraRanges()
    if spec.strength > 0:
        scaled = CameraRanges(
            **{k: getattr(ranges, k) * spec.strength for k in ranges.__dataclass_fields__}
        )
        true = sample_sibling(rng, nominal, scaled)
        target, map_x, map_y = coarse_field_and_maps(true, nominal, geometry)
        perturbed = cv2.remap(right, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    else:
        target = np.zeros((geometry.height, geometry.width, 2), dtype=np.float32)
        perturbed = right
    width, height = geometry.size
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    lands_x, lands_y = u + target[..., 0], v + target[..., 1]
    # Solo cuenta donde el píxel limpio cae dentro de la imagen perturbada.
    mask = (lands_x >= 0) & (lands_x <= width - 1) & (lands_y >= 0) & (lands_y <= height - 1)

    def as_input(image: np.ndarray, gain: float) -> np.ndarray:
        scaled = np.clip(image.astype(np.float32) * gain, 0, 255) / 255.0
        return scaled.transpose(2, 0, 1)

    return {
        "left": as_input(left, spec.gain_left),
        "right": as_input(perturbed, spec.gain_right),
        "jacobian": jacobian_maps(geometry),
        "rays": ray_maps(geometry),
        "calib": calibration_vector(geometry, nominal.D, spec.calibration_known),
        "target": target.transpose(2, 0, 1).copy(),
        "mask": mask,
        "left_u8": left,
        "right_u8": perturbed,
        "strength": spec.strength,
    }


# --- fuentes ------------------------------------------------------------------------------


@dataclass
class InStereoSource:
    """Pares de InStereo2K con una cámara virtual sorteada por muestra."""

    refs: Sequence[Any]
    name: str = "instereo2k"

    def __len__(self) -> int:
        return len(self.refs)

    def item(self, index: int, seed: Any) -> dict[str, Any]:
        rng = _rng(self.name, self.refs[index].sample_id, seed)
        sample = self.refs[index].load()
        left = cv2.resize(sample.left, INSTEREO_SIZE, interpolation=cv2.INTER_AREA)
        right = cv2.resize(sample.right, INSTEREO_SIZE, interpolation=cv2.INTER_AREA)
        nominal, geometry = virtual_camera(rng, INSTEREO_SIZE)
        return build_item(left, right, nominal, geometry, sample_spec(rng, elp=False), rng)


@dataclass
class ElpSource:
    """Cuadros de clips de desarrollo de la ELP, rectificados con su calibración."""

    calibration: Any  # StereoCalibration
    frames: list[tuple[Path, str, str]]  # (carpeta, archivo izquierdo, archivo derecho)
    name: str = "elp"
    _maps: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        cal = self.calibration
        self.geometry = cal.geometry(disparity_offset=False).resized(*ELP_SIZE)
        k = cal.right.K.astype(np.float64).copy()
        k[0] *= ELP_SIZE[0] / cal.image_size[0]
        k[1] *= ELP_SIZE[1] / cal.image_size[1]
        self.nominal = CameraModel(k, cal.right.D, cal.rect.R2)

    def __len__(self) -> int:
        return len(self.frames)

    def item(self, index: int, seed: Any) -> dict[str, Any]:
        folder, left_name, right_name = self.frames[index]
        rng = _rng(self.name, folder.name, left_name, seed)
        if self._maps is None:
            self._maps = self.calibration.rectify_maps()
        left = cv2.imread(str(folder / left_name))
        right = cv2.imread(str(folder / right_name))
        if left is None or right is None:
            raise InvalidInput(f"no se pudo leer {folder / left_name}")
        left, right = self.calibration.rectify(left, right, self._maps)
        left = cv2.resize(left, ELP_SIZE, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, ELP_SIZE, interpolation=cv2.INTER_AREA)
        spec = sample_spec(rng, elp=True)
        return build_item(left, right, self.nominal, self.geometry, spec, rng)


def elp_frames(folders: Sequence[str | Path], every: int) -> list[tuple[Path, str, str]]:
    """Uno de cada `every` cuadros de cada clip (los vecinos son casi iguales)."""
    from visionx.camera.replay import load_recording

    out = []
    for folder in folders:
        folder = Path(folder)
        _, entries, _ = load_recording(folder)
        if not entries:
            raise InvalidInput(f"{folder} no tiene pares")
        out.extend((folder, e["left"], e["right"]) for e in entries[::every])
    return out


def to_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Junta muestras del mismo tamaño en tensores de PyTorch."""
    import torch

    batch = {
        name: torch.from_numpy(np.stack([item[name] for item in items]))
        for name in ("left", "right", "jacobian", "rays", "calib", "target", "mask")
    }
    batch["strength"] = [item["strength"] for item in items]
    return batch


def with_known(item: dict[str, Any], known: bool) -> dict[str, Any]:
    """Copia de una muestra con la calibración visible u oculta (para las pruebas)."""
    out = dict(item)
    if not known:
        out["calib"] = np.zeros_like(item["calib"])
    return out
