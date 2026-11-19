"""Caché, entrenamiento y evaluación de la U-Net de incertidumbre.

Tres pasos, cada uno con su comando en la CLI del paquete:

    inc-cache      corre CREStereo una vez por muestra y variante, y guarda la disparidad
    inc-entrenar   entrena la red con esos archivos (CREStereo ya no interviene)
    inc-evaluar    mide la red entrenada contra dos referencias simples

CREStereo nunca recibe gradiente: su salida es un dato de entrada. Por defecto tampoco se
entrena la corrección de disparidad de la U-Net: la pérdida compara la disparidad de CREStereo
tal cual contra la referencia, y lo único que la red puede ajustar es b. Con `train_delta` se
habilita la corrección, como ablación.

La pérdida es la de Fusion (`fusion/loss.py`): la verosimilitud negativa de una Laplace,
|error|/b + log(2b). Si b es chico donde hay error, el primer término castiga; si es grande
donde no hace falta, castiga el segundo. El mínimo está en b = error esperado.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionx.depth.crestereo_adapt import uncertainty_metrics as um
from visionx.depth.crestereo_adapt.uncertainty import (
    CHANNELS,
    CHANNELS_VERSION,
    DEFAULT_WIDTHS,
    model_config,
)
from visionx.depth.crestereo_adapt.uncertainty_data import (
    CacheSpec,
    UncertaintyDataset,
    collate,
    perturbed_right,
    save_entry,
    variant_error,
)
from visionx.shared.errors import InvalidInput

CURVE_FIELDS = (
    "epoca",
    "perdida_train",
    "lr",
    "nll_val",
    "ause_b_val",
    "cobertura90_val",
    "segundos",
)


@dataclass(frozen=True)
class TrainParams:
    epochs: int = 12
    batch_images: int = 4
    crops_per_image: int = 4
    crop: tuple[int, int] = (320, 448)  # alto, ancho
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 100
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    seed: int = 0
    widths: tuple[int, ...] = DEFAULT_WIDTHS
    variants: int = 2
    train_delta: bool = False  # ablación: dejar que la red también corrija la disparidad
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_images < 1 or self.crops_per_image < 1:
            raise InvalidInput("epochs, batch_images y crops_per_image deben ser >= 1")


# --- caché --------------------------------------------------------------------------------


def right_referenced_disparity(matcher: Any, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Disparidad vista desde el ojo derecho, para el chequeo izquierda-derecha.

    La red solo sabe calcular la disparidad del ojo izquierdo. Espejando las dos imágenes e
    intercambiándolas, la derecha pasa a ocupar el lugar de la izquierda; se corre la red y se
    vuelve a espejar el resultado. Cuesta una pasada más: solo se usa para evaluar.
    """
    flipped = matcher.compute(
        np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1])
    )
    return np.ascontiguousarray(flipped[:, ::-1])


def build_cache(
    refs: Sequence[Any],
    dataset: str,
    spec: CacheSpec,
    matcher: Any,
    variants: int,
    with_right: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Corre CREStereo sobre cada muestra y variante que todavía no esté guardada."""
    done = skipped = 0
    start = time.perf_counter()
    for number, ref in enumerate(refs, start=1):
        sample = None
        for variant in range(variants):
            path = spec.path(dataset, ref.sample_id, variant)
            key = spec.key(ref.sample_id, variant)
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    usable = str(data["key"]) == key and (
                        not with_right or "right_disparity" in data
                    )
                if usable:
                    skipped += 1
                    continue
            sample = sample or ref.load()
            right = perturbed_right(sample.right, variant_error(ref.sample_id, variant, spec.seed))
            matcher.reset()  # cada par es otra escena: sin arranque en caliente
            disparity = matcher.compute(sample.left, right)
            extras = dict(matcher.last_extras)  # antes de que otra corrida los pise
            right_disparity = None
            if with_right:
                matcher.reset()
                right_disparity = right_referenced_disparity(matcher, sample.left, right)
            save_entry(
                path, key, disparity, right_disparity, extras.get("half"), extras.get("vertical")
            )
            done += 1
        if number % 50 == 0 or number == len(refs):
            elapsed = time.perf_counter() - start
            log(f"  {number}/{len(refs)} muestras, {done} calculadas, {elapsed / 60:.1f} min")
    return {"calculadas": done, "ya_estaban": skipped}


# --- evaluación ---------------------------------------------------------------------------


def evaluate(
    model: Any, dataset: UncertaintyDataset, device: str, train_delta: bool = False
) -> dict[str, dict[str, float]]:
    """Mide la red sobre cuadros completos, separando pares limpios y desalineados.

    Compara b contra dos incertidumbres que no necesitan entrenar nada:
    - el error fotométrico (el mismo canal que ve la red);
    - el chequeo izquierda-derecha, si el caché trae la disparidad del ojo derecho.
    """
    import torch

    from visionx.depth.stereo import lr_error

    model.eval()
    groups: dict[str, list[dict[str, float]]] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            out = model(item["channels"].to(device), item["base"].to(device))
            width = float(item["ref_width"][0])
            base = item["base"][0, 0].numpy() * width
            pred = out["disparity"][0, 0].cpu().numpy() * width if train_delta else base
            b = out["b"][0, 0].cpu().numpy() * width
            gt = item["gt"][0, 0].numpy() * width
            mask = item["mask"][0, 0].numpy()
            error = pred - gt
            row = {
                "epe_px": float(np.abs(error[mask]).mean()) if mask.any() else float("nan"),
                "epe_base_px": float(np.abs((base - gt)[mask]).mean()) if mask.any() else math.nan,
                "nll": um.laplace_nll(error, b, mask),
                "b_mediana_px": float(np.median(b[mask])) if mask.any() else float("nan"),
                "ause_b": um.ause(error, b, mask),
                "ause_fotometrico": um.ause(error, item["channels"][0, 3].numpy(), mask),
                "cobertura_50": um.interval_coverage(error, b, mask, 0.5),
                "cobertura_90": um.interval_coverage(error, b, mask, 0.9),
            }
            right = item["right_disparity"]
            if right is not None:
                both = np.ones_like(mask)
                lr = lr_error(base.astype(np.float32), both, right, both)
                # Donde el chequeo no se puede calcular (el punto cae fuera de la derecha) se
                # toma como lo más incierto: es lo que haría un sistema que descarta por L/R.
                lr = np.where(np.isfinite(lr), lr, np.nanmax(lr) if np.isfinite(lr).any() else 0.0)
                row["ause_izq_der"] = um.ause(error, lr, mask)
            name = "limpio" if item["variant"] == 0 else "desalineado"
            groups.setdefault(name, []).append(row)
            groups.setdefault("todos", []).append(row)
    summary = {}
    for name, rows in groups.items():
        keys = sorted({k for r in rows for k in r})
        summary[name] = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys} | {
            "cuadros": float(len(rows))
        }
    return summary


# --- entrenamiento ------------------------------------------------------------------------


def _lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Subida lineal y después coseno hasta `min_ratio` de la tasa inicial."""
    warmup = min(warmup, max(total // 5, 1))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_state(model: Any, params: TrainParams, spec: CacheSpec, extra: dict) -> dict:
    """Lo que se guarda: pesos y lo necesario para saber con qué se entrenó. Solo tipos simples,
    para poder leerlo con `weights_only=True`."""
    return {
        "model": model.state_dict(),
        "widths": list(params.widths),
        "canales": list(CHANNELS),
        "version": CHANNELS_VERSION,
        "train_delta": params.train_delta,
        "cache": {"weights_id": spec.weights_id, "run_tag": spec.run_tag, "seed": spec.seed},
        **extra,
    }


def train(
    params: TrainParams,
    train_refs: Sequence[Any],
    val_refs: Sequence[Any],
    dataset: str,
    spec: CacheSpec,
    run_dir: str | Path,
    log: Callable[[str], None] = print,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Entrena y deja en `run_dir`: mejor.pt, ultimo.pt, curvas.csv y meta.json."""
    import torch

    from visionx.depth.fusion.loss import laplace_nll
    from visionx.depth.fusion.model import FusionNet, count_parameters
    from visionx.depth.mono import pick_device

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    device = pick_device(params.device)
    model = FusionNet(model_config(params.widths)).to(device)
    if not params.train_delta:
        for p in model.delta_head.parameters():
            p.requires_grad_(False)  # la corrección queda en cero: solo se aprende b
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    train_set = UncertaintyDataset(
        train_refs, dataset, spec, params.variants, params.crop, params.crops_per_image
    )
    val_set = UncertaintyDataset(val_refs, dataset, spec, params.variants, crop=None)
    steps_per_epoch = math.ceil(len(train_set) / params.batch_images)
    total_steps = steps_per_epoch * params.epochs
    meta: dict[str, Any] = {
        "parametros": asdict(params),
        "dataset": dataset,
        "n_train": len(train_refs),
        "n_val": len(val_refs),
        "canales": list(CHANNELS),
        "parametros_red": count_parameters(model),
        "dispositivo": device,
        "torch": str(torch.__version__),
        "cache": {"weights_id": spec.weights_id, "run_tag": spec.run_tag, "seed": spec.seed},
        "estado": "en curso",
        **(extra_meta or {}),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    log(
        f"red: {meta['parametros_red'] / 1e6:.2f} M de parámetros en {device}; "
        f"{len(train_refs)} muestras de entrenamiento, {len(val_refs)} de validación"
    )

    curves: list[dict[str, Any]] = []
    best = math.inf
    step = 0
    with ThreadPoolExecutor(max_workers=params.batch_images) as pool:
        for epoch in range(params.epochs):
            started = time.perf_counter()
            model.train()
            order = np.random.default_rng([params.seed, epoch, 7919]).permutation(len(train_set))
            losses = []
            for start in range(0, len(order), params.batch_images):
                indices = order[start : start + params.batch_images]
                items = list(pool.map(lambda i, e=epoch: train_set[(e, int(i))], indices))
                batch = {k: v.to(device) for k, v in collate(items).items()}
                for group in optimizer.param_groups:
                    group["lr"] = params.lr * _lr_factor(
                        step, total_steps, params.warmup_steps, params.min_lr_ratio
                    )
                out = model(batch["channels"], batch["base"])
                disparity = out["disparity"] if params.train_delta else batch["base"]
                loss = laplace_nll(
                    disparity, out["b"], batch["gt"], batch["mask"], batch["ref_width"]
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"pérdida no finita en la época {epoch}, paso {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
                optimizer.step()
                losses.append(float(loss))
                step += 1
            val = evaluate(model, val_set, device, params.train_delta)
            score = val["todos"]["nll"]
            row = {
                "epoca": epoch,
                "perdida_train": round(float(np.mean(losses)), 5),
                "lr": optimizer.param_groups[0]["lr"],
                "nll_val": round(score, 5),
                "ause_b_val": round(val["todos"]["ause_b"], 5),
                "cobertura90_val": round(val["todos"]["cobertura_90"], 4),
                "segundos": round(time.perf_counter() - started, 1),
            }
            curves.append(row)
            with (run_dir / "curvas.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                writer.writeheader()
                writer.writerows(curves)
            state = checkpoint_state(model, params, spec, {"epoch": epoch, "nll_val": score})
            torch.save(state, run_dir / "ultimo.pt")
            improved = score < best
            if improved:
                best = score
                torch.save(state, run_dir / "mejor.pt")
            log(
                f"época {epoch:2d}: pérdida {row['perdida_train']:.4f} | val NLL {score:.4f}, "
                f"AUSE {row['ause_b_val']:.4f}, cobertura 90 % {row['cobertura90_val']:.3f} "
                f"({row['segundos']:.0f} s){' *' if improved else ''}"
            )
    meta["estado"] = "terminado"
    meta["mejor_nll_val"] = best
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return meta
