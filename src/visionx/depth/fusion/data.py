"""Datos de entrenamiento y evaluación de Fusion (T12): caché -> base -> canales -> recortes.

Cada ítem carga la muestra (`ref.load()`), lee los ingredientes de la caché en modo
solo-lectura, arma la base (B4 con la alineación guardada, o B5 al vuelo desde los mismos
ingredientes: `channels.fusion_bases`) y los canales con `channels.build_inputs` sobre el
**frame completo** (W_ref = ancho del frame entero) y recién después recorta. Recortar antes
cambiaría W_ref, el error fotométrico en los bordes, la alineación y los canales de F3 (que
miran anclas a cientos de px); recortar después no cambia nada de eso.

**Qué se pide de cada `ref`** (la interfaz de `datasets.catalog.SampleRef` que se usa):
- `sample_id: str` y `scene_id: str`;
- `load() -> StereoSample` con `dataset`, `files` (izquierda y derecha primero, relativas a
  `<data_root>/<dataset>/`), imágenes, GT y, si hay, geometría.
Sirve cualquier objeto con eso (los tests usan muestras sintéticas en una carpeta temporal).

**Tensores de cada ítem** (K = recortes por imagen; 1 en evaluación):
- `channels` K x C x h x w float32; `base` K x 1 x h x w (base / W_ref, 0 sin base);
- `gt` K x 1 x h x w (GT / W_ref, 0 fuera de la máscara); `mask` K x 1 x h x w bool
  (GT válida Y base válida); `ref_width` K float32; `n_gt` K: píxeles con GT del recorte;
- `sample_id`, `scene_id`: listas de K textos.
Solo en evaluación (frame completo), para medir en px originales con la regla de la app:
- `base_px` (base en px, 0 sin base), `base_valid` bool, `source` uint8 (fuente de la base:
  0 nada, 1 estéreo, 2 monocular), `gt_px` (GT en px, 0 sin GT), `gt_valid` bool,
  `b4_px`/`b4_valid` y `b5_px`/`b5_valid` (los dos controles, para compararlos con Fusion en
  el mismo soporte; B5 todo False si no hay `b5_params`): K x 1 x h x w; `doffs` K float32.
`collate` concatena los ítems a lo largo del primer eje: un lote de B imágenes con K
recortes da N = B·K recortes.

**Aleatoriedad reproducible.** El recorte del ítem i en la época e sale de un generador
sembrado con (semilla, e, i): no depende del orden ni del proceso que lo pida. Por eso el
muestreador (`EpochSampler`) entrega pares (época, índice) y no hace falta reiniciar los
workers para cambiar de época.

**Sin aumentos en v1.** Un espejo horizontal rompe la geometría estéreo (la disparidad
cambiaría de signo y de ojo) y un jitter de color aplicado a la izquierda desincroniza el
canal fotométrico, que compara izquierda con derecha. Quedan como límite registrado.

**Workers en macOS.** El DataLoader arranca los workers con *spawn*: el dataset se copia por
pickle. Los `SampleRef` del catálogo llevan un `load` lambda que no se puede copiar; con
`catalog=<nombre>` el dataset viaja con los ids y cada worker reindexa el catálogo una vez.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from visionx.depth.fusion import cache
from visionx.depth.fusion.cache import CacheSpec
from visionx.depth.fusion.channels import (
    BASES,
    ChannelParams,
    build_inputs,
    fusion_bases,
    uses_b5,
)
from visionx.depth.hybrid import B5Params
from visionx.shared.errors import InvalidInput


def _search_range(sample: Any, rule: Any, data_root: Path) -> int:
    # Import diferido: el CLI de T05 importa medio paquete de profundidad.
    from visionx.depth.__main__ import search_range

    return search_range(sample, rule, data_root)


class FusionDataset(Dataset):
    """Recortes (entrenamiento) o frames completos (evaluación) listos para la red."""

    def __init__(
        self,
        refs: Sequence[Any],
        rule: Any,
        data_root: str | Path,
        cache_spec: CacheSpec,
        variant: str = "F1",
        channel_params: ChannelParams | None = None,
        crop: tuple[int, int] | None = None,
        crops_per_image: int = 1,
        min_valid_fraction: float = 0.1,
        seed: int = 0,
        max_crop_tries: int = 20,
        catalog: str | None = None,
        base: str = "B4",
        b5_params: B5Params | None = None,
    ) -> None:
        if not refs:
            raise InvalidInput("FusionDataset sin muestras")
        if base not in BASES:
            raise InvalidInput(f"base {base!r}: se esperaba una de {BASES}")
        if uses_b5(variant, base) and b5_params is None:
            raise InvalidInput(f"variante {variant} con base {base} necesita depth.b5")
        if crop is not None:
            crop = (int(crop[0]), int(crop[1]))
            if min(crop) < 1:
                raise InvalidInput(f"recorte {crop}: alto y ancho tienen que ser > 0")
        elif crops_per_image != 1:
            raise InvalidInput("sin recorte (frame completo) hay un solo ítem por imagen")
        if crops_per_image < 1 or max_crop_tries < 1:
            raise InvalidInput("crops_per_image y max_crop_tries tienen que ser >= 1")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise InvalidInput("min_valid_fraction tiene que estar en [0, 1]")
        self.refs = list(refs)
        self.rule = rule
        self.data_root = Path(data_root)
        self.cache_spec = cache_spec
        self.variant = variant
        self.channel_params = channel_params or ChannelParams()
        self.crop = crop
        self.crops_per_image = crops_per_image
        self.min_valid_fraction = float(min_valid_fraction)
        self.seed = int(seed)
        self.max_crop_tries = int(max_crop_tries)
        self.catalog = catalog
        self.base = base
        self.b5_params = b5_params
        self.crop_retries = 0  # recortes reintentados por poca máscara (en este proceso)

    # --- copia a los workers (spawn) -------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.catalog is not None:
            # Los SampleRef del catálogo no se pueden copiar (load es lambda): viajan los ids.
            state["refs"] = None
            state["_ref_ids"] = [r.sample_id for r in self.refs]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        ids = state.pop("_ref_ids", None)
        self.__dict__.update(state)
        if ids is not None:
            from visionx.depth.datasets.catalog import CATALOG

            info = CATALOG[self.catalog]
            by_id = {r.sample_id: r for r in info.index(self.data_root / info.name)}
            self.refs = [by_id[i] for i in ids]

    # --- ítems -----------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.refs)

    def full_frame(self, index: int) -> dict[str, Any]:
        """Canales, base, GT y máscara del frame completo, en numpy (sin recortar)."""
        ref = self.refs[index]
        sample = ref.load()
        ndisp = _search_range(sample, self.rule, self.data_root)
        doffs = cache.sample_doffs(sample)
        ing = cache.read(sample, ndisp, self.cache_spec, self.data_root)
        stereo = ing.stereo()
        # Evaluación (frame completo): B4 y B5 siempre que se pueda, para el soporte común.
        evaluating = self.crop is None
        need_b4 = self.base == "B4" or self.variant == "F3" or evaluating
        need_b5 = uses_b5(self.variant, self.base) or (evaluating and self.b5_params is not None)
        b4, b5 = fusion_bases(stereo, ing.mono_relative, doffs, self.cache_spec.align,
                              self.b5_params, need_b4, need_b5,
                              b4_alignment=ing.alignment)  # fmt: skip
        base = b4 if self.base == "B4" else b5
        inputs = build_inputs(sample.left, sample.right, stereo, base, ing.mono_relative, doffs,
                              self.variant, self.channel_params, b4=b4, b5=b5)  # fmt: skip
        width = inputs.ref_width
        shape = inputs.base_valid.shape
        controls = {}
        for name, out in (("b4", b4), ("b5", b5)):
            ok = out.valid if out is not None else np.zeros(shape, bool)
            d = np.where(ok, out.disparity_px, 0.0) if out is not None else np.zeros(shape)
            controls[f"{name}_px"] = d.astype(np.float32)[None]
            controls[f"{name}_valid"] = ok[None]
        # GT en d/W_ref; 0 fuera de la máscara (np.where: el NaN de afuera no entra).
        gt = np.where(sample.valid, sample.disparity, 0.0).astype(np.float32) / np.float32(width)
        return {
            "channels": inputs.channels,
            "base": inputs.base_norm[None],
            "gt": gt[None],
            "mask": (sample.valid & inputs.base_valid)[None],
            "gt_valid": sample.valid[None],
            "gt_px": np.where(sample.valid, sample.disparity, 0.0).astype(np.float32)[None],
            "base_px": np.where(base.valid, base.disparity_px, 0.0).astype(np.float32)[None],
            "base_valid": inputs.base_valid[None],
            "source": base.source[None],
            "doffs": float(doffs),
            "ref_width": float(width),
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "alignment_accepted": bool(base.alignment.accepted),
            **controls,
        }

    def _crop_windows(self, mask: np.ndarray, epoch: int, index: int) -> list[tuple[int, int]]:
        """Esquinas (y, x) de los recortes; reintenta si la máscara cubre poco."""
        h, w = mask.shape[-2:]
        ch, cw = self.crop
        if ch > h or cw > w:
            raise InvalidInput(f"recorte {ch}x{cw} más grande que la imagen {h}x{w}")
        rng = np.random.default_rng([self.seed, epoch, index])
        # Tabla de sumas acumuladas: fracción de máscara de cualquier ventana en O(1).
        integral = np.pad(mask[0].astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        area = ch * cw
        windows = []
        for _ in range(self.crops_per_image):
            best, best_fraction = None, -1.0
            for _ in range(self.max_crop_tries):
                y = int(rng.integers(0, h - ch + 1))
                x = int(rng.integers(0, w - cw + 1))
                inside = (integral[y + ch, x + cw] - integral[y, x + cw] - integral[y + ch, x]
                          + integral[y, x])  # fmt: skip
                fraction = inside / area
                if fraction > best_fraction:
                    best, best_fraction = (y, x), fraction
                if fraction >= self.min_valid_fraction:
                    break
                self.crop_retries += 1
            # Si ningún intento alcanza, queda el mejor visto (la pérdida promedia sobre la
            # máscara; un recorte casi vacío pesa poco pero no rompe nada).
            windows.append(best)
        return windows

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        epoch, index = (0, key) if isinstance(key, int) else (int(key[0]), int(key[1]))
        frame = self.full_frame(index)
        if self.crop is None:
            windows = [(0, 0)]
            ch, cw = frame["mask"].shape[-2:]
        else:
            windows = self._crop_windows(frame["mask"], epoch, index)
            ch, cw = self.crop

        def cut(a: np.ndarray) -> np.ndarray:
            return np.stack([a[:, y : y + ch, x : x + cw] for y, x in windows])

        k = len(windows)
        extra: dict[str, Any] = {}
        if self.crop is None:  # evaluación: px originales y máscaras para la regla de la app
            extra = {
                name: torch.from_numpy(np.ascontiguousarray(cut(frame[name])))
                for name in (
                    "base_px",
                    "base_valid",
                    "source",
                    "gt_px",
                    "gt_valid",
                    "b4_px",
                    "b4_valid",
                    "b5_px",
                    "b5_valid",
                )  # fmt: skip
            }
            extra["doffs"] = torch.full((k,), frame["doffs"], dtype=torch.float32)
        return {
            "channels": torch.from_numpy(np.ascontiguousarray(cut(frame["channels"]))),
            "base": torch.from_numpy(np.ascontiguousarray(cut(frame["base"]))),
            "gt": torch.from_numpy(np.ascontiguousarray(cut(frame["gt"]))),
            "mask": torch.from_numpy(np.ascontiguousarray(cut(frame["mask"]))),
            "n_gt": torch.tensor(
                [float(frame["gt_valid"][:, y : y + ch, x : x + cw].sum()) for y, x in windows]
            ),
            "ref_width": torch.full((k,), frame["ref_width"], dtype=torch.float32),
            "sample_id": [frame["sample_id"]] * k,
            "scene_id": [frame["scene_id"]] * k,
            "alignment_accepted": [frame["alignment_accepted"]] * k,
            **extra,
        }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aplana los recortes: B ítems de K recortes -> N = B·K a lo largo del primer eje."""
    out: dict[str, Any] = {}
    for name in items[0]:
        values = [it[name] for it in items]
        if isinstance(values[0], torch.Tensor):
            out[name] = torch.cat(values, dim=0)
        else:
            out[name] = [x for v in values for x in v]
    return out


class EpochSampler(Sampler):
    """Entrega pares (época, índice) en un orden barajado con (semilla, época).

    El orden y los recortes quedan fijados por la época, no por el estado de un generador:
    reanudar en la época e reproduce exactamente lo que habría pasado sin cortar.
    """

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch, 7919]).permutation(self.n)
        else:
            order = np.arange(self.n)
        return iter([(self.epoch, int(i)) for i in order])

    def __len__(self) -> int:
        return self.n
