"""Caché de ingredientes de Fusion (T12): lo caro de cada muestra, calculado una vez.

Entrenar Fusion necesita, por muestra, el estéreo B2, el mapa monocular y la alineación. SGBM
a ~1080x860 con 272 disparidades tarda del orden de un segundo y Depth Anything necesita la
GPU: repetirlo en cada época sería tirar horas. Se guardan esos **ingredientes** (no los
canales): con ellos, `hybrid.combine("B4", ..., alignment=...)` y `channels.build_inputs`
reconstruyen base y canales en milisegundos, sin RANSAC ni redes. Guardar ingredientes y no
canales tiene otra ventaja: si cambia `ChannelParams` no hay que recalcular la caché.

Formato: un `.npz` comprimido por muestra en `<raíz>/<dataset>/<sample_id saneado>.npz`:
- `disparity_px` float32 (0 fuera de `valid`; al leer vuelve a NaN), `valid` bool;
- `lr_error_px` float16 con NaN donde no se pudo calcular. Es exacto donde importa: SGBM
  entrega múltiplos de 1/16 px y float16 representa exactamente esos valores hasta 128 px; el
  canal `error_lr` satura mucho antes (W_ref / ganancia ≈ 34 px con W_ref = 1080);
- `mono_relative` float32: el mapa relativo del monocular, sin alinear;
- `key` y `meta` (JSON): la clave y lo necesario para diagnosticar (alineación, tiempos y
  versiones de OpenCV, numpy, torch y transformers con que se calculó).

**Clave.** sha256 de un JSON canónico con todo lo que cambia los ingredientes: versión de
este formato, sha256 de las imágenes izquierda y derecha, `SgbmParams` completos (incluye el
rango del par), variante estéreo, `AlignParams`, `doffs` e identidad del monocular (repo,
revisión y sha de pesos). Si la clave guardada no coincide con la esperada, el modo
construir recalcula y el modo lectura (el del entrenamiento) falla con `InvalidInput`.
La caché es reutilizar un cálculo, no una verdad permanente.

**Versiones de bibliotecas.** OpenCV (SGBM, WLS) y torch/transformers (Depth Anything) pueden
cambiar los números entre versiones. No entran en la clave (la caché ya construida seguiría
valiendo con las mismas versiones y no conviene invalidarla por un cambio de formato): se
guardan en `meta` y, al leer, si difieren de las instaladas se emite un aviso (`UserWarning`),
no un error. Archivos escritos antes de este registro no traen versiones y no avisan.

Solo numpy (más los módulos de profundidad): no importa torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.depth.align import Alignment, AlignParams, align
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.hybrid import HybridOutput, combine
from visionx.depth.mono import MonoConfig, MonoModel, check_relative, sha256_file
from visionx.depth.stereo import SgbmMatcher, SgbmParams, StereoOutput
from visionx.shared.errors import InvalidInput

CACHE_FORMAT = "fusion-cache-1"  # cambiarlo invalida todas las cachés escritas antes
FUSION_STEREO_VARIANTS = ("B1", "B2")  # las que calculan los dos ojos (error L/R)


def mono_identity(config: MonoConfig) -> dict[str, str]:
    """Qué pesos monoculares produjeron el mapa. El `model_id` es el de `DepthAnythingV2`.

    No incluye el dispositivo: MPS y CPU dan mapas casi iguales y la caché se arma una vez.
    """
    return {
        "model_id": f"da-v2-small@{config.revision[:8]}",
        "repo": config.repo,
        "revision": config.revision,
        "weights_sha256": config.weights_sha256,
    }


@dataclass(frozen=True)
class CacheSpec:
    """Dónde vive la caché y con qué parámetros se calculan los ingredientes."""

    root: Path
    sgbm: SgbmParams  # base; `num_disparities` se reemplaza por el rango de cada par
    align: AlignParams
    mono: dict[str, str]  # identidad del monocular (ver mono_identity)
    stereo_variant: str = "B2"

    def __post_init__(self) -> None:
        if self.stereo_variant not in FUSION_STEREO_VARIANTS:
            raise InvalidInput(
                f"estéreo {self.stereo_variant!r}: Fusion necesita {FUSION_STEREO_VARIANTS} "
                "(B0 no calcula el error L/R)"
            )
        if not self.mono.get("model_id"):
            raise InvalidInput("CacheSpec.mono necesita al menos 'model_id'")

    @classmethod
    def from_config(cls, depth_cfg: dict[str, Any], root: str | Path | None = None) -> CacheSpec:
        """Arma la especificación desde la sección `depth` de la config."""
        fusion = depth_cfg.get("fusion", {})
        if root is None:
            root = fusion.get("cache_root", "data/cache/fusion")
        return cls(
            root=Path(root),
            sgbm=SgbmParams.from_config(depth_cfg["sgbm"]),
            align=AlignParams.from_config(depth_cfg["alineacion"]),
            mono=mono_identity(MonoConfig.from_config(depth_cfg["mono"])),
            stereo_variant=depth_cfg["b4"]["estereo"],
        )

    def sgbm_for(self, num_disparities: int) -> SgbmParams:
        return replace(self.sgbm, num_disparities=int(num_disparities))

    def key_material(
        self, file_hashes: dict[str, str], num_disparities: int, doffs: float
    ) -> dict[str, Any]:
        """Todo lo que, si cambia, cambia los ingredientes."""
        if not file_hashes:
            raise InvalidInput("la clave de caché necesita el hash de las imágenes")
        alignment = asdict(self.align)
        # `method` llegó con B5 (T12). Con "ransac" (lo único que existía antes) se omite: así
        # las claves de las cachés ya construidas siguen valiendo. Otro método sí cambia la clave.
        if alignment.get("method") == "ransac":
            del alignment["method"]
        return {
            "formato": CACHE_FORMAT,
            "archivos": dict(sorted(file_hashes.items())),
            "sgbm": asdict(self.sgbm_for(num_disparities)),
            "estereo": self.stereo_variant,
            "alineacion": alignment,
            "doffs": float(doffs),
            "mono": dict(sorted(self.mono.items())),
        }

    def key_for(self, file_hashes: dict[str, str], num_disparities: int, doffs: float) -> str:
        """sha256 del JSON canónico (claves ordenadas, sin espacios) de `key_material`."""
        material = self.key_material(file_hashes, num_disparities, doffs)
        text = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def path_for(self, dataset: str, sample_id: str) -> Path:
        """`<raíz>/<dataset>/<sample_id saneado>.npz`. El id saneado solo ubica el archivo:
        si dos ids chocaran, la clave (con el hash de las imágenes) lo detecta."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
        return self.root / dataset / f"{safe}.npz"


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    """Versiones instaladas de lo que calcula los ingredientes. Sin importar torch."""
    out = {"opencv": cv2.__version__, "numpy": np.__version__}
    for package in ("torch", "transformers"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = "no instalado"
    return out


_WARNED: set[tuple[str, ...]] = set()


def check_versions(meta: dict[str, Any], where: str = "") -> list[str]:
    """Compara las versiones guardadas en `meta` con las instaladas; avisa una vez por
    combinación de diferencias (no por archivo). Devuelve las diferencias."""
    stored = meta.get("versiones")
    if not stored:
        return []
    current = library_versions()
    diffs = [f"{k}: caché {v}, instalada {current.get(k, '?')}"
             for k, v in sorted(stored.items()) if current.get(k) != v]  # fmt: skip
    if diffs and tuple(diffs) not in _WARNED:
        _WARNED.add(tuple(diffs))
        warnings.warn(
            f"caché de Fusion calculada con otras versiones ({'; '.join(diffs)}), p. ej. "
            f"{where}: se usa igual; si los números importan, reconstruirla",
            UserWarning,
            stacklevel=2,
        )
    return diffs


def sample_doffs(sample: StereoSample) -> float:
    return sample.geometry.doffs if sample.geometry is not None else 0.0


def sample_file_hashes(sample: StereoSample, data_root: str | Path) -> dict[str, str]:
    """sha256 de las imágenes izquierda y derecha de la muestra.

    Convención de los adaptadores de `datasets/`: `files[0]` es la izquierda y `files[1]` la
    derecha, relativas a `<data_root>/<dataset>/`. La GT no entra: no cambia los ingredientes.
    """
    if len(sample.files) < 2:
        raise InvalidInput(f"{sample.sample_id}: la caché necesita las rutas de las imágenes")
    base = Path(data_root) / sample.dataset
    return {rel: sha256_file(base / rel) for rel in sample.files[:2]}


# Campos float de Alignment: en JSON un NaN viaja como null y al leer vuelve a NaN.
_ALIGN_FLOATS = {f.name for f in fields(Alignment) if f.type in ("float", float)}


def alignment_from_dict(data: dict[str, Any]) -> Alignment:
    """Inversa de `Alignment.to_dict`: None -> NaN en los campos numéricos."""
    values = {}
    for f in fields(Alignment):
        val = data[f.name]
        if f.name in _ALIGN_FLOATS:
            val = math.nan if val is None else float(val)
        values[f.name] = val
    return Alignment(**values)


@dataclass(frozen=True)
class Ingredients:
    """Lo caro de una muestra. Con esto se reconstruyen B4 y los canales sin redes ni SGBM."""

    key: str
    disparity_px: np.ndarray  # H x W float32, NaN fuera de valid (como StereoOutput)
    valid: np.ndarray  # H x W bool
    lr_error_px: np.ndarray  # H x W float32, NaN donde no se pudo calcular
    mono_relative: np.ndarray  # H x W float32, mapa relativo sin alinear
    alignment: Alignment  # la recta que usó B4 (aceptada o no)
    meta: dict[str, Any] = field(default_factory=dict)

    def stereo(self) -> StereoOutput:
        """La salida estéreo tal como la devolvió SgbmMatcher (con copias, no vistas)."""
        stats = dict(self.meta.get("stereo_stats", {}))
        return StereoOutput(self.disparity_px.copy(), self.valid.copy(), stats,
                            self.lr_error_px.copy())  # fmt: skip

    def base(self, doffs: float, align_params: AlignParams) -> HybridOutput:
        """B4 reusando la alineación guardada (no se repite RANSAC)."""
        return combine("B4", self.stereo(), self.mono_relative, doffs, align_params,
                       alignment=self.alignment)  # fmt: skip


def compute_ingredients(
    sample: StereoSample,
    num_disparities: int,
    spec: CacheSpec,
    mono: MonoModel,
    data_root: str | Path,
) -> Ingredients:
    """Calcula estéreo, monocular y alineación de una muestra con los parámetros de `spec`.

    Repite exactamente lo que hacen `HybridMatcher.compute` y `combine("B4", ...)`: mismo
    matcher, el monocular sobre el ojo izquierdo y `align` con las anclas del estéreo.
    """
    if mono.model_id != spec.mono["model_id"]:
        raise InvalidInput(
            f"monocular {mono.model_id!r} distinto del de la caché {spec.mono['model_id']!r}"
        )
    doffs = sample_doffs(sample)
    key = spec.key_for(sample_file_hashes(sample, data_root), num_disparities, doffs)
    matcher = SgbmMatcher(spec.stereo_variant, spec.sgbm_for(num_disparities))
    t0 = time.perf_counter()
    stereo = matcher.compute(sample.left, sample.right, doffs)
    t1 = time.perf_counter()
    relative = check_relative(mono.predict(sample.left), stereo.disparity_px.shape)
    t2 = time.perf_counter()
    alignment = align(relative, stereo.disparity_px, stereo.valid, spec.align)
    t3 = time.perf_counter()
    if stereo.lr_error_px is None:
        raise InvalidInput(f"estéreo {spec.stereo_variant} sin error L/R")
    meta = {
        "formato": CACHE_FORMAT,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "num_disparities": int(num_disparities),
        "doffs": float(doffs),
        "monocular": mono.model_id,
        "dispositivo_mono": str(getattr(mono, "device", "")),
        "stereo_stats": {k: float(x) for k, x in stereo.stats.items()},
        "ms_estereo": (t1 - t0) * 1e3,
        "ms_mono": (t2 - t1) * 1e3,
        "ms_alineacion": (t3 - t2) * 1e3,
        "versiones": library_versions(),
    }
    return Ingredients(key, stereo.disparity_px, stereo.valid, stereo.lr_error_px, relative,
                       alignment, meta)  # fmt: skip


def save(ingredients: Ingredients, path: str | Path) -> Path:
    """Escritura atómica: se escribe un temporal en la misma carpeta y se renombra.

    Un corte a mitad de camino deja, a lo sumo, un temporal huérfano; nunca un .npz roto con
    nombre válido.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ing = ingredients
    meta = dict(ing.meta, alignment=ing.alignment.to_dict())
    arrays = {
        "key": np.array(ing.key),
        "meta": np.array(json.dumps(meta, sort_keys=True, ensure_ascii=True)),
        # 0 fuera de valid: comprime mejor que NaN y la máscara ya dice dónde no hay dato.
        "disparity_px": np.where(ing.valid, ing.disparity_px, 0.0).astype(np.float32),
        "valid": ing.valid.astype(bool),
        "lr_error_px": ing.lr_error_px.astype(np.float16),
        "mono_relative": ing.mono_relative.astype(np.float32),
    }
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def stored_key(path: str | Path) -> str | None:
    """Clave guardada en un archivo de caché, o None si no existe o no se puede leer.

    `np.load` de un .npz es perezoso: leer `key` no descomprime los mapas.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["key"])
    except (OSError, ValueError, KeyError):
        return None


def load(path: str | Path) -> Ingredients:
    """Lee un archivo de caché (sin verificar la clave; para eso, `read`)."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            key = str(data["key"])
            meta = json.loads(str(data["meta"]))
            valid = data["valid"].astype(bool)
            disparity = np.where(valid, data["disparity_px"], np.nan).astype(np.float32)
            lr = data["lr_error_px"].astype(np.float32)
            mono = data["mono_relative"].astype(np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise InvalidInput(f"{path}: archivo de caché ilegible ({exc})") from exc
    alignment = alignment_from_dict(meta.pop("alignment"))
    return Ingredients(key, disparity, valid, lr, mono, alignment, meta)


def read(
    sample: StereoSample, num_disparities: int, spec: CacheSpec, data_root: str | Path
) -> Ingredients:
    """Modo solo-lectura (entrenamiento): la clave guardada tiene que ser la esperada."""
    path = spec.path_for(sample.dataset, sample.sample_id)
    expected = spec.key_for(sample_file_hashes(sample, data_root), num_disparities,
                            sample_doffs(sample))  # fmt: skip
    found = stored_key(path)
    if found is None:
        raise InvalidInput(
            f"{sample.sample_id}: no hay caché en {path}; armarla con "
            "`python -m visionx.depth.fusion cache`"
        )
    if found != expected:
        raise InvalidInput(
            f"{sample.sample_id}: la caché en {path} se calculó con otras imágenes o parámetros "
            "(clave distinta); reconstruirla con `python -m visionx.depth.fusion cache`"
        )
    ing = load(path)
    check_versions(ing.meta, str(path))
    if ing.disparity_px.shape != sample.left.shape[:2]:
        raise InvalidInput(f"{sample.sample_id}: la caché no tiene el tamaño de la muestra")
    return ing


def get_or_compute(
    sample: StereoSample,
    num_disparities: int,
    spec: CacheSpec,
    mono: Callable[[], MonoModel],
    data_root: str | Path,
) -> tuple[Ingredients | None, bool, str]:
    """Modo construir: reusa si la clave coincide; si no, recalcula y guarda.

    `mono` es una función que devuelve el modelo: solo se llama si hay algo que calcular, así
    una segunda corrida con todo en caché no carga Depth Anything.
    Devuelve (ingredientes recién calculados o None si se reusó, reusado, ruta).
    """
    path = spec.path_for(sample.dataset, sample.sample_id)
    expected = spec.key_for(sample_file_hashes(sample, data_root), num_disparities,
                            sample_doffs(sample))  # fmt: skip
    if stored_key(path) == expected:
        return None, True, str(path)
    ing = compute_ingredients(sample, num_disparities, spec, mono(), data_root)
    save(ing, path)
    return ing, False, str(path)


def _meta(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


def build_cache(
    refs: Sequence[Any],
    num_disparities: Callable[[StereoSample], int],
    spec: CacheSpec,
    mono: Callable[[], MonoModel],
    data_root: str | Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Arma (o completa) la caché de una lista de muestras y devuelve un resumen.

    `refs`: objetos con `sample_id` y `load() -> StereoSample` (p. ej. `SampleRef`).
    `num_disparities`: rango de SGBM de cada muestra (en el CLI, `search_range` con la regla
    de la config).
    """
    start = time.perf_counter()
    computed = reused = accepted = 0
    compute_s: list[float] = []
    total_bytes = 0
    for i, ref in enumerate(refs, 1):
        t0 = time.perf_counter()
        sample = ref.load()
        ing, was_reused, path = get_or_compute(sample, num_disparities(sample), spec, mono,
                                               data_root)  # fmt: skip
        elapsed = time.perf_counter() - t0
        if was_reused:
            reused += 1
            meta = _meta(path)
            check_versions(meta, path)
            ok = bool(meta["alignment"]["accepted"])
        else:
            computed += 1
            compute_s.append(elapsed)
            ok = ing.alignment.accepted
        accepted += ok
        size = Path(path).stat().st_size
        total_bytes += size
        state = "reusada" if was_reused else f"calculada en {elapsed:.1f} s"
        log(f"  [{i}/{len(refs)}] {sample.sample_id}: {state}, {size / 1e6:.1f} MB, "
            f"alineación {'aceptada' if ok else 'rechazada'}")  # fmt: skip
    n = len(refs)
    return {
        "muestras": n,
        "calculadas": computed,
        "reusadas": reused,
        "segundos": time.perf_counter() - start,
        "s_por_muestra_calculada": float(np.mean(compute_s)) if compute_s else math.nan,
        "alineacion_aceptada_pct": 100.0 * accepted / n if n else math.nan,
        "mb_total": total_bytes / 1e6,
        "mb_por_muestra": total_bytes / 1e6 / n if n else math.nan,
    }
