"""Datos para entrenar la U-Net de incertidumbre: salidas de CREStereo guardadas en disco.

Entrenar la red de incertidumbre no necesita correr CREStereo en cada paso: su salida sobre un
par no cambia (está congelado). Se calcula una vez por muestra y se guarda. Después el
entrenamiento solo lee archivos.

Cada muestra tiene **variantes**:

    variante 0   el par tal cual, bien rectificado
    variante 1+  el mismo par con el ojo derecho desalineado por un error sorteado

Las variantes desalineadas enseñan a la red que, cuando las filas no coinciden, hay que confiar
menos. La referencia sigue siendo la disparidad real del par limpio: para quien usa el sistema,
el corrimiento que produce una mala alineación es un error como cualquier otro.

El error de cada variante sale de una semilla y del nombre de la muestra: se puede regenerar
la imagen desalineada sin guardarla. Lo que se guarda es la disparidad de CREStereo.

InStereo2K no trae calibración. Para armar el giro hace falta una focal: se supone igual al
ancho de la imagen (unos 53° de campo horizontal). Solo define cuántos píxeles mueve cada giro.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionx.depth.crestereo_adapt.correction import Correction, homography, warp_image
from visionx.depth.crestereo_adapt.uncertainty import build_channels
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

CACHE_FORMAT = "crestereo-inc-cache-2"
SUBPIXEL = 64  # la disparidad se guarda como entero: 1/64 px de resolución, hasta 1023 px


def assumed_geometry(width: int, height: int) -> StereoGeometry:
    """Cámara supuesta para un par sin calibración: focal = ancho, centro en el medio."""
    return StereoGeometry(
        width=width,
        height=height,
        fx=float(width),
        fy=float(width),
        cx=width / 2.0,
        cy=height / 2.0,
        baseline_m=0.1,
    )


def variant_error(sample_id: str, variant: int, seed: int = 0) -> Correction:
    """Error de alineación de una variante. La 0 es el par limpio.

    De las desalineadas, dos de cada tres son **chicas** (lo que puede quedar después de una
    corrección: décimas de grado) y una es **grande** (cámara sin corregir).
    """
    if variant == 0:
        return Correction()
    digest = hashlib.sha256(f"{sample_id}|{variant}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    large = rng.random() < 1.0 / 3.0
    angle, yaw, focal = (1.0, 0.5, 0.02) if large else (0.15, 0.10, 0.003)
    roll, pitch, yaw_u, focal_u = rng.uniform(-1.0, 1.0, size=4)
    return Correction(
        roll_deg=float(roll) * angle,
        pitch_deg=float(pitch) * angle,
        yaw_deg=float(yaw_u) * yaw,
        log_focal=float(focal_u) * focal,
    )


def perturbed_right(right: np.ndarray, error: Correction) -> np.ndarray:
    """La derecha desalineada con `error` (lo que hay que aplicarle para arreglarla)."""
    if error.is_identity:
        return right
    height, width = right.shape[:2]
    undo = homography(error, assumed_geometry(width, height))
    return warp_image(right, np.linalg.inv(undo))


@dataclass(frozen=True)
class CacheSpec:
    """Dónde va el caché y con qué se calculó. Cambiar algo de esto cambia la clave."""

    root: Path
    weights_id: str  # primeros caracteres del SHA-256 de los pesos de CREStereo
    run_tag: str  # cómo se corrió la red, p. ej. "i4p2"
    seed: int = 0

    def key(self, sample_id: str, variant: int) -> str:
        return f"{CACHE_FORMAT}|{self.weights_id}|{self.run_tag}|{self.seed}|{sample_id}|{variant}"

    def path(self, dataset: str, sample_id: str, variant: int) -> Path:
        safe = sample_id.replace("/", "_").replace("\\", "_")
        return self.root / dataset / f"{safe}-v{variant}.npz"


def save_entry(
    path: Path,
    key: str,
    disparity: np.ndarray,
    right_disparity: np.ndarray | None = None,
    half: np.ndarray | None = None,
    vertical: np.ndarray | None = None,
) -> None:
    """Guarda la disparidad de CREStereo y sus subproductos: la pasada a media resolución, el
    canal vertical y, si se calculó, la disparidad vista desde el ojo derecho."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def pack(d: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.nan_to_num(d, nan=0.0), 0.0, 65535.0 / SUBPIXEL)
        return np.round(clipped * SUBPIXEL).astype(np.uint16)

    arrays = {"key": np.array(key), "disparity": pack(disparity)}
    if right_disparity is not None:
        arrays["right_disparity"] = pack(right_disparity)
    if half is not None:
        arrays["half"] = pack(half)
    if vertical is not None:  # con signo: entero de 16 bits, ±511 px
        limit = 32767.0 / SUBPIXEL
        clipped = np.clip(np.nan_to_num(vertical, nan=0.0), -limit, limit)
        arrays["vertical"] = np.round(clipped * SUBPIXEL).astype(np.int16)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_entry(path: Path, key: str) -> dict[str, np.ndarray]:
    """Lee una entrada y controla que sea la esperada."""
    if not path.exists():
        raise InvalidInput(f"falta el caché {path}: corré `inc-cache`")
    with np.load(path, allow_pickle=False) as data:
        if str(data["key"]) != key:
            raise InvalidInput(f"{path} se calculó con otros parámetros: rehacer el caché")
        out = {
            name: data[name].astype(np.float32) / SUBPIXEL
            for name in ("disparity", "right_disparity", "half", "vertical")
            if name in data
        }
    return out


def frame_arrays(
    sample: Any,
    disparity: np.ndarray,
    error: Correction,
    half: np.ndarray | None = None,
    vertical: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Todo lo que la red y la pérdida necesitan de un cuadro completo, en NumPy.

    `base` y `gt` van divididos por el ancho (d/W_ref), como en Fusion. La máscara es donde hay
    referencia y la red dio un valor.
    """
    width = sample.left.shape[1]
    right = perturbed_right(sample.right, error)
    channels = build_channels(sample.left, right, disparity, half, vertical)
    gt = np.where(sample.valid, sample.disparity, 0.0).astype(np.float32)
    mask = sample.valid & np.isfinite(disparity)
    return {
        "channels": channels,
        "base": (np.nan_to_num(disparity, nan=0.0) / width)[None].astype(np.float32),
        "gt": (gt / width)[None],
        "mask": mask[None],
        "ref_width": np.float32(width),
    }


class UncertaintyDataset:
    """Recortes al azar de cuadros con su disparidad de CREStereo en caché.

    Sigue el protocolo de `torch.utils.data.Dataset` (`__len__` y `__getitem__`) sin heredar de
    él, para que el módulo se pueda importar sin PyTorch. La clave es `(época, índice)`: la
    variante y los recortes salen de esa pareja y de la semilla, así una corrida se repite.
    """

    def __init__(
        self,
        refs: Sequence[Any],
        dataset: str,
        spec: CacheSpec,
        variants: int = 2,
        crop: tuple[int, int] | None = (320, 448),
        crops_per_image: int = 4,
        min_valid_fraction: float = 0.1,
    ) -> None:
        if variants < 1:
            raise InvalidInput("variants >= 1")
        self.refs = list(refs)
        self.dataset = dataset
        self.spec = spec
        self.variants = variants
        self.crop = crop
        self.crops_per_image = crops_per_image
        self.min_valid_fraction = min_valid_fraction

    def __len__(self) -> int:
        return len(self.refs) if self.crop else len(self.refs) * self.variants

    def frame(self, index: int, variant: int) -> dict[str, np.ndarray]:
        ref = self.refs[index]
        entry = load_entry(
            self.spec.path(self.dataset, ref.sample_id, variant),
            self.spec.key(ref.sample_id, variant),
        )
        error = variant_error(ref.sample_id, variant, self.spec.seed)
        out = frame_arrays(
            ref.load(), entry["disparity"], error, entry.get("half"), entry.get("vertical")
        )
        out["right_disparity"] = entry.get("right_disparity")
        out["variant"] = variant
        out["sample_id"] = ref.sample_id
        return out

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        import torch

        epoch, index = key if isinstance(key, tuple) else (0, key)
        if self.crop is None:  # evaluación: cuadro completo, todas las variantes en orden
            arrays = self.frame(index // self.variants, index % self.variants)
            tensors = {
                k: torch.from_numpy(np.ascontiguousarray(arrays[k]))[None]
                for k in ("channels", "base", "gt", "mask")
            }
            tensors["ref_width"] = torch.tensor([float(arrays["ref_width"])])
            tensors["variant"] = arrays["variant"]
            tensors["sample_id"] = arrays["sample_id"]
            tensors["right_disparity"] = arrays["right_disparity"]
            return tensors
        rng = np.random.default_rng([self.spec.seed, epoch, index])
        arrays = self.frame(index, int(rng.integers(self.variants)))
        crop_h, crop_w = self.crop
        height, width = arrays["mask"].shape[-2:]
        if crop_h > height or crop_w > width:
            raise InvalidInput(f"recorte {self.crop} más grande que la imagen {height}x{width}")
        windows = []
        for _ in range(self.crops_per_image):
            for _try in range(20):  # se evita un recorte casi sin referencia
                top = int(rng.integers(height - crop_h + 1))
                left = int(rng.integers(width - crop_w + 1))
                window = arrays["mask"][0, top : top + crop_h, left : left + crop_w]
                if window.mean() >= self.min_valid_fraction:
                    break
            windows.append((top, left))
        tensors = {}
        for name in ("channels", "base", "gt", "mask"):
            stack = np.stack(
                [arrays[name][:, t : t + crop_h, lf : lf + crop_w] for t, lf in windows]
            )
            tensors[name] = torch.from_numpy(np.ascontiguousarray(stack))
        tensors["ref_width"] = torch.full((len(windows),), float(arrays["ref_width"]))
        return tensors


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Junta los recortes de varias imágenes en un solo lote."""
    import torch

    return {
        name: torch.cat([item[name] for item in items])
        for name in ("channels", "base", "gt", "mask", "ref_width")
    }
