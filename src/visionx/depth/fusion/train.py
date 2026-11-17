"""Entrenamiento de Fusion (T12): recortes de InStereo2K train, validación en frames completos.

Qué hace una corrida (`run_training`):
1. Arma la red (`FusionNet`) con los canales de la variante y la siembra con `seed`.
2. Cada época recorre train en recortes (`FusionDataset`, ver data.py) con AdamW, warmup
   lineal corto y coseno; corta con un error claro si la pérdida sale NaN/Inf.
3. Valida sobre **frames completos** de val. Soporte = GT válida ∩ base válida (B4 o B5,
   `TrainConfig.base`): base y Fusion se miden en los mismos píxeles (Fusion no agrega ni
   quita cobertura). Además se informan B4, B5 y Fusion en el soporte común de los tres.
   Métricas por muestra con `metrics.disparity_metrics`, en px reales.
4. Elige checkpoint por **EPE media por muestra en val** (val tiene 2 tramos y uno con solo
   2 pares: la media por escena se informa, no elige; mismo criterio que T05).

Salida en `outputs/fusion/<run>/`: `config.yaml`, `meta.json`, `curvas.csv`, `ultimo.pt`
(para reanudar) y `mejor.pt` (el elegido). El formato del checkpoint está en `save_checkpoint`.

`overfit` es la prueba M5 de "la red puede aprender": pocos recortes fijos, muchos pasos, y
la EPE tiene que bajar ≥ 80 % respecto de la base sobre esos mismos recortes.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from visionx.depth.fusion.cache import CACHE_FORMAT, CacheSpec
from visionx.depth.fusion.channels import (
    BASES,
    CHANNELS_VERSION,
    VARIANTS,
    ChannelParams,
    check_channels,
    disparity_from_residual,
    fusion_validity,
    uses_b5,
)
from visionx.depth.fusion.data import EpochSampler, FusionDataset, collate
from visionx.depth.fusion.loss import combined_loss, epe_px, laplace_nll
from visionx.depth.fusion.model import FusionNet, ModelConfig, count_parameters
from visionx.depth.hybrid import B5Params
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.mono import pick_device
from visionx.shared.errors import EstimatorError, InvalidInput

CHECKPOINT_FORMAT = 1
TRAIN_DATASET = "instereo2k"  # el único con train/val (TECNICO T04)


class NonFiniteLoss(EstimatorError):
    """La pérdida dio NaN o infinito: se corta en vez de seguir con pesos rotos."""


@dataclass(frozen=True)
class OverfitConfig:
    """Prueba M5: pocos recortes fijos de train, muchos pasos."""

    images: int = 4
    crops_per_image: int = 3  # 4 x 3 = 12 recortes (M5 pide 8-16)
    steps: int = 400
    lr: float = 1.0e-3
    target_reduction_pct: float = 80.0


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparámetros de entrenamiento (configs/fusion-entrenamiento.yaml)."""

    variant: str = "F1"
    base: str = "B4"  # la base que corrige la red: B4 | B5 (B5 se arma al vuelo de la caché)
    model: ModelConfig = field(default_factory=ModelConfig)  # in_channels sale de la variante
    loss: str = "nll"  # nll (Laplace, con b) | l1 (ablación sin incertidumbre)
    l1_weight: float = 0.0  # término L1 extra sumado a la NLL
    crop: tuple[int, int] = (320, 448)  # alto, ancho del recorte, px
    crops_per_image: int = 4  # recortes por imagen decodificada
    batch_images: int = 4  # imágenes por paso => batch_images * crops_per_image recortes
    min_valid_fraction: float = 0.1  # fracción mínima de máscara de un recorte (si no, reintenta)
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 100  # tope; nunca más de 1/5 de los pasos totales
    min_lr_ratio: float = 0.02  # el coseno baja hasta lr * min_lr_ratio
    grad_clip: float = 1.0  # norma máxima del gradiente
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"  # auto | mps | cuda | cpu
    val_every: int = 1  # validar cada N épocas (y siempre en la última)
    limit_train: int = 0  # 0 = todas; N = N repartidas (pilotos)
    limit_val: int = 0
    overfit: OverfitConfig = field(default_factory=OverfitConfig)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise InvalidInput(f"variante {self.variant!r}: se esperaba una de {sorted(VARIANTS)}")
        if self.base not in BASES:
            raise InvalidInput(f"base {self.base!r}: se esperaba una de {BASES}")
        if self.model.in_channels != len(VARIANTS[self.variant]):
            # La variante manda: se corrige sin preguntar (el YAML no repite in_channels).
            fixed = replace(self.model, in_channels=len(VARIANTS[self.variant]))
            object.__setattr__(self, "model", fixed)
        if self.loss not in ("nll", "l1"):
            raise InvalidInput(f"loss {self.loss!r}: se esperaba 'nll' o 'l1'")
        for name in ("crops_per_image", "batch_images", "epochs", "val_every"):
            if int(getattr(self, name)) < 1:
                raise InvalidInput(f"{name} tiene que ser >= 1")
        if len(self.crop) != 2 or min(self.crop) < 1:
            raise InvalidInput(f"crop {self.crop}: se esperaba [alto, ancho] positivos")
        for name in ("lr", "grad_clip"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise InvalidInput(f"{name} tiene que ser > 0")
        if self.num_workers < 0 or self.limit_train < 0 or self.limit_val < 0:
            raise InvalidInput("num_workers y los límites tienen que ser >= 0")

    @property
    def crops_per_step(self) -> int:
        return self.batch_images * self.crops_per_image

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["model"] = self.model.to_dict()
        data["crop"] = list(self.crop)
        data["overfit"] = asdict(self.overfit)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise InvalidInput(f"claves desconocidas en la config de Fusion: {sorted(unknown)}")
        kwargs = dict(data)
        if "model" in kwargs:
            kwargs["model"] = ModelConfig.from_dict(kwargs["model"] or {})
        if "crop" in kwargs:
            kwargs["crop"] = tuple(int(x) for x in kwargs["crop"])
        if "overfit" in kwargs:
            kwargs["overfit"] = OverfitConfig(**(kwargs["overfit"] or {}))
        for name in ("lr", "weight_decay", "min_lr_ratio", "grad_clip", "l1_weight",
                     "min_valid_fraction"):  # fmt: skip
            if name in kwargs:
                kwargs[name] = float(kwargs[name])  # YAML lee 1e-3 (sin punto) como texto
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise InvalidInput(f"{path}: se esperaba un mapping YAML")
        return cls.from_dict(data)


# --- piezas ------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Semillas de torch y numpy. Los recortes no dependen de esto (ver data.py)."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def build_model(cfg: TrainConfig) -> FusionNet:
    return FusionNet(cfg.model)


def lr_factor(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Multiplicador del lr en el paso `step` (0 = primero): warmup lineal y coseno."""
    warmup = min(cfg.warmup_steps, total_steps // 5)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig, total_steps: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, cfg)
    )
    return optimizer, scheduler


def make_loader(dataset: FusionDataset, batch: int, workers: int, sampler=None) -> DataLoader:
    """DataLoader portable: sin pin_memory (MPS no lo usa) y workers persistentes.

    Con workers, el contexto es spawn (el de macOS): no copia el estado de MPS del proceso
    principal, que con fork puede colgarse.
    """
    kwargs: dict[str, Any] = {}
    if workers > 0:
        kwargs = {"persistent_workers": True, "multiprocessing_context": "spawn",
                  "prefetch_factor": 2}  # fmt: skip
    return DataLoader(dataset, batch_size=batch, sampler=sampler, shuffle=False,
                      num_workers=workers, collate_fn=collate, pin_memory=False,
                      drop_last=False, **kwargs)  # fmt: skip


_TENSORS = ("channels", "base", "gt", "mask", "ref_width")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (val.to(device) if k in _TENSORS else val) for k, val in batch.items()}


def mps_memory_mb() -> tuple[float, float]:
    """(memoria asignada por tensores, memoria del driver) en MB; NaN fuera de MPS."""
    if not torch.backends.mps.is_available():
        return math.nan, math.nan
    return (torch.mps.current_allocated_memory() / 2**20,
            torch.mps.driver_allocated_memory() / 2**20)  # fmt: skip


def check_finite(loss: torch.Tensor, where: str) -> float:
    value = float(loss.detach())
    if not math.isfinite(value):
        raise NonFiniteLoss(
            f"pérdida no finita ({value}) en {where}: se corta. Revisar lr, entradas y b"
        )
    return value


def clip_and_check(model: torch.nn.Module, max_norm: float, where: str) -> float:
    """Recorta el gradiente y verifica su norma ANTES de `optimizer.step()`.

    Una pérdida finita puede dar un gradiente infinito (p. ej. b muy chica en la NLL). Si la
    norma no es finita se corta sin actualizar: un paso con NaN arruinaría todos los pesos.
    """
    norm = float(clip_grad_norm_(model.parameters(), max_norm))
    if not math.isfinite(norm):
        raise NonFiniteLoss(f"gradiente no finito (norma {norm}) en {where}: se corta sin "
                            "actualizar los pesos")  # fmt: skip
    return norm


# --- evaluación -----------------------------------------------------------------------------


def _disparity(batch: dict[str, Any], name: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(disparidad en px con NaN fuera de la máscara, máscara) de `<name>_px` del ítem i."""
    ok = batch[f"{name}_valid"][i, 0].numpy()
    return np.where(ok, batch[f"{name}_px"][i, 0].numpy(), np.nan).astype(np.float32), ok


@torch.no_grad()
def evaluate(model: FusionNet, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    """Métricas por muestra de la base, Fusion, B4 y B5 sobre frames completos, en px originales.

    Fusion se arma como en la app: d = d_base (px) + delta · W_ref (`disparity_from_residual`)
    y vale donde la base vale, d + doffs > 0 y b > 0 (`fusion_validity`).
    - `*_fusion` y `*_base` (EPE, bad-2, acierto-2): soporte = GT válida ∩ base válida ∩
      Fusion válida; los dos medidos en los mismos píxeles (elige con `epe_fusion`).
    - `cobertura_*` y `acierto2_gt_*`: cada método con su propia máscara sobre toda la GT,
      como T05 (no responder resta).
    - `*_comun_{fusion,b4,b5}`: soporte común de los tres = GT ∩ Fusion ∩ B4 ∩ B5. Es la
      comparación justa contra los dos controles fijos; NaN si no se armó B5.
    """
    model.eval()
    rows = []
    for batch in loader:
        b = to_device(batch, device)
        out = model(b["channels"], b["base"])
        nll = laplace_nll(out["disparity"], out["b"], b["gt"], b["mask"], b["ref_width"])
        for i in range(b["channels"].shape[0]):
            w = float(batch["ref_width"][i])
            doffs = float(batch["doffs"][i])
            gt_valid = batch["gt_valid"][i, 0].numpy()
            gt = batch["gt_px"][i, 0].numpy()

            base, base_valid = _disparity(batch, "base", i)
            b4, b4_valid = _disparity(batch, "b4", i)
            b5, b5_valid = _disparity(batch, "b5", i)
            delta = out["delta"][i, 0].float().cpu().numpy()
            b_px = out["b"][i, 0].float().cpu().numpy() * np.float32(w)
            fusion = disparity_from_residual(base, base_valid, delta, w)
            fusion_ok = fusion_validity(fusion, b_px, base_valid, doffs)
            common = base_valid & fusion_ok
            m_f = disparity_metrics(fusion, fusion_ok, gt, gt_valid, support=common)
            m_b = disparity_metrics(base, base_valid, gt, gt_valid, support=common)
            g_f = disparity_metrics(fusion, fusion_ok, gt, gt_valid)
            g_b = disparity_metrics(base, base_valid, gt, gt_valid)
            all3 = common & b4_valid & b5_valid
            c_f = disparity_metrics(fusion, fusion_ok, gt, gt_valid, support=all3)
            c_4 = disparity_metrics(b4, b4_valid, gt, gt_valid, support=all3)
            c_5 = disparity_metrics(b5, b5_valid, gt, gt_valid, support=all3)
            in_common = common & gt_valid
            rows.append({
                "muestra": batch["sample_id"][i],
                "escena": batch["scene_id"][i],
                "n": m_f["n"],
                "n_gt": g_f["n_gt"],
                "epe_fusion": m_f["epe"],
                "epe_base": m_b["epe"],
                "bad2_fusion": m_f["bad2"],
                "bad2_base": m_b["bad2"],
                "acierto2_fusion": m_f["acierto2"],
                "acierto2_base": m_b["acierto2"],
                "cobertura_fusion": g_f["cobertura"],
                "cobertura_base": g_b["cobertura"],
                "acierto2_gt_fusion": g_f["acierto2"],
                "acierto2_gt_base": g_b["acierto2"],
                **{f"{k}_comun_{name}": m[k] for name, m in (("fusion", c_f), ("b4", c_4),
                                                              ("b5", c_5))
                   for k in ("epe", "bad2", "acierto2")},
                "fraccion_comun": c_f["n_gt"] / g_f["n_gt"] if g_f["n_gt"] else math.nan,
                "nll_px": float(nll) if b["channels"].shape[0] == 1 else math.nan,
                "b_mediana_px": float(np.median(b_px[in_common])) if in_common.any()
                else math.nan,
                "invalidada_por_residual": float((base_valid & ~fusion_ok).sum())
                / max(1, int(base_valid.sum())),
                "alin_aceptada": float(batch["alignment_accepted"][i]),
            })  # fmt: skip
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Media por muestra (la que elige) y media por escena (se informa)."""
    return {
        "por_muestra": aggregate(rows, key="muestra")["por_escena"],
        "por_escena": aggregate(rows, key="escena")["por_escena"],
    }


# --- checkpoints ----------------------------------------------------------------------------


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Escritura atómica de un dict de torch (tmp + rename)."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def checkpoint_state(
    model: FusionNet,
    cfg: TrainConfig,
    channel_params: ChannelParams,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    best: dict[str, float],
) -> dict[str, Any]:
    """Formato 1 del checkpoint (lo leen `evaluar`, `reanudar` y el empaquetado)."""
    return {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "model_config": cfg.model.to_dict(),
        "variant": cfg.variant,
        "base": cfg.base,
        "channels_version": CHANNELS_VERSION,
        "channel_names": list(VARIANTS[cfg.variant]),
        "channel_params": channel_params.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),  # épocas completas
        "step": int(step),  # pasos de optimizador hechos
        "best": {"epe": float(best["epe"]), "epoch": int(best["epoch"])},
        "seed": int(cfg.seed),
        "rng": {"torch": torch.get_rng_state()},
    }


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Lee un checkpoint de formato 1 y devuelve (red en eval, dict). Rechaza una versión de
    canales incompatible o un orden de canales distinto del de la variante. Sin clave "base"
    (checkpoints de antes de B5) la base es B4."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise InvalidInput(f"{path}: formato de checkpoint {ckpt.get('format')!r} desconocido")
    try:
        check_channels(ckpt["channels_version"], ckpt["variant"], ckpt["channel_names"])
    except InvalidInput as exc:
        raise InvalidInput(f"{path}: {exc}") from exc
    ckpt.setdefault("base", "B4")  # checkpoints de antes de la base configurable
    model = FusionNet(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


# --- corrida --------------------------------------------------------------------------------

CURVE_FIELDS = (
    "epoca", "paso", "lr", "perdida_train", "epe_train_px", "epe_train_base_px",
    "val_epe_fusion", "val_epe_base", "val_cobertura_fusion", "val_cobertura_base",
    "val_bad2_fusion", "val_bad2_base", "val_acierto2_fusion", "val_acierto2_base",
    "val_nll_px", "val_b_mediana_px", "val_epe_fusion_escena", "val_epe_base_escena",
    "val_epe_comun_fusion", "val_epe_comun_b4", "val_epe_comun_b5",
    "val_acierto2_comun_fusion", "val_acierto2_comun_b4", "val_acierto2_comun_b5",
    "segundos_epoca", "segundos_val", "segundos_datos", "recortes_por_s",
    "mps_asignada_pico_mb", "mps_driver_pico_mb",
)  # fmt: skip
# Columnas de curvas.csv de antes de la base configurable: la base era siempre B4.
OLD_CURVE_NAMES = {
    "epe_train_b4_px": "epe_train_base_px", "val_epe_b4": "val_epe_base",
    "val_cobertura_b4": "val_cobertura_base", "val_bad2_b4": "val_bad2_base",
    "val_acierto2_b4": "val_acierto2_base", "val_epe_b4_escena": "val_epe_base_escena",
}  # fmt: skip


def manifest_hash(refs: Sequence[Any]) -> str:
    """sha256 de la lista de muestras (ids, escenas y rutas), no de su contenido: barato y
    alcanza para ver si dos corridas usaron las mismas muestras."""
    items = [[r.sample_id, r.scene_id, list(getattr(r, "files", ()))] for r in refs]
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def _read_curves(path: Path, up_to_epoch: int) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoca"]) <= up_to_epoch]
    # Al reanudar un run viejo, sus columnas pasan a los nombres nuevos (mismos valores).
    return [{OLD_CURVE_NAMES.get(k, k): val for k, val in r.items()} for r in rows]


def _write_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _fmt(value: Any) -> Any:
    return f"{value:.9g}" if isinstance(value, float) else value


@dataclass
class RunInputs:
    """Lo que una corrida necesita además de la config de entrenamiento."""

    train_refs: Sequence[Any]
    val_refs: Sequence[Any]
    rule: Any  # regla de rango de SGBM (int o "calib")
    data_root: Path
    cache_spec: CacheSpec
    channel_params: ChannelParams
    catalog: str | None = None  # nombre del catálogo (para workers con spawn)
    b5_params: B5Params | None = None  # depth.b5: base B5, canales F3 y el control B5 en val
    config_record: dict[str, Any] = field(default_factory=dict)  # secciones de config usadas
    commit: str = "desconocido"


def _datasets(cfg: TrainConfig, inp: RunInputs) -> tuple[FusionDataset, FusionDataset]:
    common = dict(rule=inp.rule, data_root=inp.data_root, cache_spec=inp.cache_spec,
                  variant=cfg.variant, channel_params=inp.channel_params, seed=cfg.seed,
                  catalog=inp.catalog, base=cfg.base, b5_params=inp.b5_params)  # fmt: skip
    train_ds = FusionDataset(inp.train_refs, crop=cfg.crop, crops_per_image=cfg.crops_per_image,
                             min_valid_fraction=cfg.min_valid_fraction, **common)  # fmt: skip
    val_ds = FusionDataset(inp.val_refs, crop=None, **common)
    return train_ds, val_ds


def run_training(
    cfg: TrainConfig,
    run_dir: str | Path,
    inp: RunInputs,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Entrena (o reanuda desde `ultimo.pt`) y deja los artefactos en `run_dir`.

    `stop_after_epoch`: corta después de esa época (sin cambiar el plan de lr, que depende de
    `cfg.epochs`); lo usa el test de reanudación.
    """
    run_dir = Path(run_dir)
    if uses_b5(cfg.variant, cfg.base) and inp.b5_params is None:
        raise InvalidInput(f"{cfg.variant} con base {cfg.base} necesita depth.b5 en la config")
    device = torch.device(pick_device(cfg.device))
    seed_everything(cfg.seed)
    train_ds, val_ds = _datasets(cfg, inp)
    sampler = EpochSampler(len(train_ds), cfg.seed, shuffle=True)
    train_loader = make_loader(train_ds, cfg.batch_images, cfg.num_workers, sampler)
    val_loader = make_loader(val_ds, 1, cfg.num_workers)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs

    model = build_model(cfg).to(device)
    optimizer, scheduler = make_optimizer(model, cfg, total_steps)
    start_epoch, step = 0, 0
    best = {"epe": math.inf, "epoch": 0}
    curves_path = run_dir / "curvas.csv"
    meta_path = run_dir / "meta.json"

    if resume:
        ckpt = torch.load(run_dir / "ultimo.pt", map_location=device, weights_only=True)
        if (ckpt.get("format") != CHECKPOINT_FORMAT or ckpt["variant"] != cfg.variant
                or ckpt.get("base", "B4") != cfg.base):  # fmt: skip
            raise InvalidInput(f"{run_dir}/ultimo.pt no corresponde a esta config")
        check_channels(ckpt["channels_version"], ckpt["variant"], ckpt["channel_names"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        start_epoch, step = int(ckpt["epoch"]), int(ckpt["step"])
        best = dict(ckpt["best"])
        curves = _read_curves(curves_path, start_epoch)  # descarta filas posteriores al corte
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("reanudaciones", []).append(
            {"desde_epoca": start_epoch, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
        log(f"reanudando {run_dir.name} desde la época {start_epoch} (paso {step})")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        curves = []
        record = {
            "entrenamiento": cfg.to_dict(),
            **inp.config_record,
            "canales": {
                "version": CHANNELS_VERSION,
                "variante": cfg.variant,
                "nombres": list(VARIANTS[cfg.variant]),
                "params": inp.channel_params.to_dict(),
            },
            "base": cfg.base,
            "cache": {
                "formato": CACHE_FORMAT,
                "raiz": str(inp.cache_spec.root),
                "estereo": inp.cache_spec.stereo_variant,
            },  # fmt: skip
        }
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)
        meta = {
            "commit": inp.commit,
            "torch": torch.__version__,
            "dispositivo": str(device),
            "equipo": {
                "sistema": platform.platform(),
                "procesador": platform.processor(),
                "python": platform.python_version(),
            },  # fmt: skip
            "seed": cfg.seed,
            "parametros": count_parameters(model),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "recortes_por_paso": cfg.crops_per_step,
            "pasos_por_epoca": steps_per_epoch,
            "manifiesto_train_sha256": manifest_hash(inp.train_refs),
            "manifiesto_val_sha256": manifest_hash(inp.val_refs),
            "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    meta["estado"] = "en curso"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    last_epoch = cfg.epochs if stop_after_epoch is None else min(cfg.epochs, stop_after_epoch)
    log(f"{len(train_ds)} imágenes de train ({cfg.crops_per_step} recortes por paso, "
        f"{steps_per_epoch} pasos por época), {len(val_ds)} de val, en {device}, "
        f"{count_parameters(model):,} parámetros")  # fmt: skip
    for epoch in range(start_epoch + 1, last_epoch + 1):
        model.train()
        sampler.set_epoch(epoch)
        t_epoch = time.perf_counter()
        data_s = 0.0
        losses, epes, epes_base = [], [], []
        peak = [0.0, 0.0]
        t_wait = time.perf_counter()
        n_crops = 0
        for batch in train_loader:
            data_s += time.perf_counter() - t_wait
            b = to_device(batch, device)
            out = model(b["channels"], b["base"])
            loss, _ = combined_loss(out, b["gt"], b["mask"], b["ref_width"], cfg.loss,
                                    cfg.l1_weight)  # fmt: skip
            value = check_finite(loss, f"época {epoch}, paso {step + 1}")
            if device.type == "mps":
                # Después del forward (check_finite ya sincronizó): las activaciones guardadas
                # para el backward siguen vivas; es el pico de memoria del paso.
                alloc, driver = mps_memory_mb()
                peak = [max(peak[0], alloc), max(peak[1], driver)]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_and_check(model, cfg.grad_clip, f"época {epoch}, paso {step + 1}")
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(value)
            epes.append(float(epe_px(out["disparity"], b["gt"], b["mask"], b["ref_width"])))
            # La base en los mismos recortes y máscara: sin esto no se ve si la red corrige.
            epes_base.append(float(epe_px(b["base"], b["gt"], b["mask"], b["ref_width"])))
            n_crops += b["channels"].shape[0]
            t_wait = time.perf_counter()
        train_s = time.perf_counter() - t_epoch

        row: dict[str, Any] = {
            "epoca": epoch,
            "paso": step,
            "lr": optimizer.param_groups[0]["lr"],
            "perdida_train": float(np.mean(losses)),
            "epe_train_px": float(np.mean(epes)),
            "epe_train_base_px": float(np.mean(epes_base)),
            "segundos_datos": data_s,
            "recortes_por_s": n_crops / train_s if train_s > 0 else math.nan,
            "mps_asignada_pico_mb": peak[0] if device.type == "mps" else "",
            "mps_driver_pico_mb": peak[1] if device.type == "mps" else "",
        }
        validated = epoch % cfg.val_every == 0 or epoch == cfg.epochs
        val_s = 0.0
        if validated:
            t_val = time.perf_counter()
            summary = summarize(evaluate(model, val_loader, device))
            val_s = time.perf_counter() - t_val
            s, sc = summary["por_muestra"], summary["por_escena"]
            row.update(val_epe_fusion=s["epe_fusion"], val_epe_base=s["epe_base"],
                       val_bad2_fusion=s["bad2_fusion"], val_bad2_base=s["bad2_base"],
                       val_acierto2_fusion=s["acierto2_fusion"],
                       val_acierto2_base=s["acierto2_base"], val_nll_px=s["nll_px"],
                       val_b_mediana_px=s["b_mediana_px"],
                       val_cobertura_fusion=s["cobertura_fusion"],
                       val_cobertura_base=s["cobertura_base"],
                       val_epe_fusion_escena=sc["epe_fusion"],
                       val_epe_base_escena=sc["epe_base"],
                       **{f"val_{k}_comun_{m}": s[f"{k}_comun_{m}"] for k in ("epe", "acierto2")
                          for m in ("fusion", "b4", "b5")})  # fmt: skip
        row["segundos_epoca"] = time.perf_counter() - t_epoch
        row["segundos_val"] = val_s
        if device.type == "mps":
            # La validación usa frames completos de tamaños distintos y el allocator de MPS
            # retiene bloques de cada tamaño: en v2 la memoria del driver creció de 15 a 23,6 GB
            # en 20 épocas y el sistema entró en swap (BITACORA 2026-11-16). Devolver lo que
            # no está en uso al final de cada época no cambia resultados, solo la memoria.
            gc.collect()
            torch.mps.empty_cache()
        curves.append({k: _fmt(row.get(k, "")) for k in CURVE_FIELDS})
        _write_curves(curves_path, curves)

        improved = validated and row["val_epe_fusion"] < best["epe"]
        if improved:
            best = {"epe": float(row["val_epe_fusion"]), "epoch": epoch}
        state = checkpoint_state(model, cfg, inp.channel_params, optimizer, scheduler, epoch,
                                 step, best)  # fmt: skip
        save_checkpoint(run_dir / "ultimo.pt", state)
        if improved:
            save_checkpoint(run_dir / "mejor.pt", state)

        msg = (f"época {epoch}/{cfg.epochs}: pérdida {row['perdida_train']:.4f}, "
               f"EPE train {row['epe_train_px']:.3f} px ({cfg.base} "
               f"{row['epe_train_base_px']:.3f}), "
               f"{row['segundos_epoca']:.0f} s "
               f"({row['segundos_datos']:.0f} s esperando datos)")  # fmt: skip
        if validated:
            msg += (f"; val EPE Fusion {row['val_epe_fusion']:.3f} px vs {cfg.base} "
                    f"{row['val_epe_base']:.3f} px{' (mejor)' if improved else ''}"
                    f"; soporte común B4 {row['val_epe_comun_b4']:.3f}, B5 "
                    f"{row['val_epe_comun_b5']:.3f}, Fusion "
                    f"{row['val_epe_comun_fusion']:.3f} px")  # fmt: skip
        log(msg)

    finished = last_epoch == cfg.epochs
    meta.update(estado="terminado" if finished else f"cortado tras la época {last_epoch}",
                mejor=best, pasos=step, fin=time.strftime("%Y-%m-%d %H:%M:%S"))  # fmt: skip
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"run_dir": str(run_dir), "best": best, "epochs_done": last_epoch, "steps": step}


# --- overfit (M5) ---------------------------------------------------------------------------


def overfit(
    cfg: TrainConfig,
    refs: Sequence[Any],
    inp: RunInputs,
    steps: int | None = None,
    lr: float | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Aprende de memoria unos pocos recortes fijos y mide cuánto baja la EPE respecto de la base.

    Sin aleatoriedad de datos: los recortes se eligen una vez (época 0) y se repiten en todos
    los pasos. Adam sin weight decay ni scheduler. EPE por píxel sobre todos los recortes.
    """
    oc = cfg.overfit
    steps = int(steps if steps is not None else oc.steps)
    lr = float(lr if lr is not None else oc.lr)
    device = torch.device(pick_device(cfg.device))
    seed_everything(cfg.seed)
    chosen = list(refs)[: oc.images]
    ds = FusionDataset(chosen, inp.rule, inp.data_root, inp.cache_spec, cfg.variant,
                       inp.channel_params, crop=cfg.crop, crops_per_image=oc.crops_per_image,
                       min_valid_fraction=cfg.min_valid_fraction, seed=cfg.seed, base=cfg.base,
                       b5_params=inp.b5_params)  # fmt: skip
    t0 = time.perf_counter()
    batch = to_device(collate([ds[(0, i)] for i in range(len(ds))]), device)
    data_s = time.perf_counter() - t0
    gt, mask, width = batch["gt"], batch["mask"], batch["ref_width"]
    epe_base = float(epe_px(batch["base"], gt, mask, width))

    model = build_model(cfg).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t1 = time.perf_counter()
    history = []
    for i in range(1, steps + 1):
        out = model(batch["channels"], batch["base"])
        loss, _ = combined_loss(out, gt, mask, width, cfg.loss, cfg.l1_weight)
        value = check_finite(loss, f"overfit, paso {i}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_and_check(model, cfg.grad_clip, f"overfit, paso {i}")
        optimizer.step()
        if i == 1 or i % max(1, steps // 10) == 0 or i == steps:
            epe = float(epe_px(out["disparity"], gt, mask, width))
            history.append({"paso": i, "perdida": value, "epe_px": epe})
            log(f"  paso {i}/{steps}: pérdida {value:.4f}, EPE {epe:.3f} px "
                f"({cfg.base} {epe_base:.3f})")  # fmt: skip
    train_s = time.perf_counter() - t1
    model.eval()
    with torch.no_grad():
        out = model(batch["channels"], batch["base"])
        epe_fusion = float(epe_px(out["disparity"], gt, mask, width))
    reduction = 100.0 * (1.0 - epe_fusion / epe_base) if epe_base > 0 else math.nan
    return {
        "imagenes": len(chosen),
        "recortes": int(gt.shape[0]),
        "recorte": list(cfg.crop),
        "pasos": steps,
        "lr": lr,
        "perdida": cfg.loss,
        "variante": cfg.variant,
        "base": cfg.base,
        "parametros": count_parameters(model),
        "dispositivo": str(device),
        "epe_base_px": epe_base,
        "epe_fusion_px": epe_fusion,
        "reduccion_pct": reduction,
        "cumple_m5": bool(reduction >= oc.target_reduction_pct),
        "segundos_datos": data_s,
        "segundos_entrenamiento": train_s,
        "historia": history,
    }
