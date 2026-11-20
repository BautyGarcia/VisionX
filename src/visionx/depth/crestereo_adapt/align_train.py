"""Entrenar y evaluar la red de alineación.

**Pérdida.** Se compara el campo que predice la red con el exacto, píxel por píxel:
- en filas, el error tal cual;
- en columnas, el error **menos su mediana en esa imagen**. El corrimiento constante de columnas
  no se ve en las filas y lo arregla la distancia conocida: no se le pide a la red.
Además, una pérdida auxiliar enseña a la cabeza a medir filas y a decir cuánto confía (la misma
verosimilitud de Laplace de la U-Net de incertidumbre).

**Evaluación.** Mismas medidas para tres sistemas sobre las mismas muestras: sin corregir, el
estimador clásico con términos de lente y regla (`lens.py`), y la red. Se informa la mediana
del error del campo en px, separando filas y columnas, y por nivel de desalineación.
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

from visionx.depth.crestereo_adapt.align_data import to_batch
from visionx.depth.crestereo_adapt.alignnet import AlignConfig, build_model, upsample_field
from visionx.shared.errors import InvalidInput

BUCKETS = (
    ("limpio", 0.0, 0.0),
    ("chico", 1e-9, 0.33),
    ("medio", 0.33, 0.66),
    ("grande", 0.66, 1.0),
)


@dataclass(frozen=True)
class AlignTrainParams:
    epochs: int = 15
    batch: int = 4
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 100
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    aux_weight: float = 0.1
    seed: int = 0
    device: str = "auto"
    workers: int = 4

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch < 1:
            raise InvalidInput("epochs y batch deben ser >= 1")


def field_errors(pred, target, mask):
    """Error por muestra: (filas, columnas sin su mediana), cada uno (N, H, W), en px."""
    import torch

    error = pred - target
    rows = error[:, 1].abs()
    cols = []
    for k in range(error.shape[0]):
        inside = error[k, 0][mask[k]]
        center = inside.median() if inside.numel() else error.new_zeros(())
        cols.append((error[k, 0] - center).abs())
    return rows, torch.stack(cols)


def loss_terms(out: dict, batch: dict, aux_weight: float) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    target, mask = batch["target"], batch["mask"]
    pred = upsample_field(out["field"], (target.shape[-1], target.shape[-2]))
    rows, cols = field_errors(pred, target, mask)
    m = mask.float()
    count = m.sum().clamp_min(1.0)
    main = ((rows + cols) * m).sum() / count
    # Auxiliar: las filas que mide la cabeza contra las exactas, promediadas por celda.
    cell = F.avg_pool2d(m[:, None], 8)
    t_rows = F.avg_pool2d(target[:, 1:2] * m[:, None], 8) / cell.clamp_min(1e-6)
    use = (cell > 0.5).float()
    nll = (out["dy"] - t_rows).abs() / out["b"] + out["b"].log()
    aux = (nll * use).sum() / use.sum().clamp_min(1.0)
    total = main + aux_weight * aux
    return total, {"campo": float(main.detach()), "aux": float(aux.detach())}


def _lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    warmup = min(warmup, max(total // 5, 1))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def predict_field(model: Any, item: dict[str, Any], device: str) -> np.ndarray:
    """Campo de la red para una muestra, a resolución completa: (2, H, W) en px."""
    import torch

    batch = to_batch([item])
    with torch.inference_mode():
        out = model(*(batch[k].to(device) for k in ("left", "right", "jacobian", "rays", "calib")))
        height, width = item["target"].shape[-2:]
        return upsample_field(out["field"], (width, height))[0].float().cpu().numpy()


def classic_field(item: dict[str, Any], geometry: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    """Campo del estimador clásico con lente y regla, en el mismo formato que la red."""
    from visionx.depth.crestereo_adapt.classic import match_points
    from visionx.depth.crestereo_adapt.lens import fit_lens, pull_maps

    estimate = fit_lens(*match_points(item["left_u8"], item["right_u8"]), geometry)
    height, width = item["target"].shape[-2:]
    if not (estimate.ok and estimate.applied):
        return np.zeros((2, height, width), dtype=np.float32), ()
    map_x, map_y = pull_maps(estimate.lens, geometry)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.stack([map_x - u, map_y - v]), estimate.applied


def _medians(field: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    error = field - target
    rows = np.abs(error[1][mask])
    cols = error[0][mask]
    return float(np.median(rows)), float(np.median(np.abs(cols - np.median(cols))))


def evaluate(
    model: Any,
    sources: Sequence[Any],
    device: str,
    seed: Any = "val",
    with_classic: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    """Error del campo de cada sistema, por fuente y por nivel de desalineación."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for source in sources:
        count = len(source) if not limit else min(limit, len(source))
        for index in np.linspace(0, len(source) - 1, count).round().astype(int):
            item = source.item(int(index), seed)
            geometry = getattr(source, "geometry", None)
            if geometry is None:
                from visionx.depth.crestereo_adapt.align_data import _rng as make_rng
                from visionx.depth.crestereo_adapt.align_data import virtual_camera

                ref = source.refs[int(index)]
                _, geometry = virtual_camera(
                    make_rng(source.name, ref.sample_id, seed),
                    (item["target"].shape[-1], item["target"].shape[-2]),
                )
            target, mask = item["target"], item["mask"]
            row = {"fuente": source.name, "fuerza": item["strength"]}
            zero = np.zeros_like(target)
            row["sin_filas"], row["sin_columnas"] = _medians(zero, target, mask)
            row["red_filas"], row["red_columnas"] = _medians(
                predict_field(model, item, device), target, mask
            )
            if with_classic:
                classic, applied = classic_field(item, geometry)
                row["clasico_filas"], row["clasico_columnas"] = _medians(classic, target, mask)
                row["clasico_aplica"] = ",".join(applied)
            rows.append(row)
    summary: dict[str, Any] = {}
    for source_name in sorted({r["fuente"] for r in rows}):
        for name, low, high in BUCKETS:
            chosen = [r for r in rows if r["fuente"] == source_name and low <= r["fuerza"] <= high]
            if not chosen:
                continue
            keys = [k for k in chosen[0] if k.endswith(("_filas", "_columnas"))]
            summary[f"{source_name}/{name}"] = {
                "muestras": len(chosen),
                **{k: round(float(np.mean([r[k] for r in chosen])), 3) for k in keys},
            }
    return {"resumen": summary, "filas": rows}


def train(
    params: AlignTrainParams,
    train_sources: Sequence[Any],
    val_sources: Sequence[Any],
    run_dir: str | Path,
    config: AlignConfig | None = None,
    log: Callable[[str], None] = print,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Entrena y deja en `run_dir`: mejor.pt, ultimo.pt, curvas.csv y meta.json."""
    import torch

    from visionx.depth.mono import pick_device

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(params.seed)
    device = pick_device(params.device)
    config = config or AlignConfig()
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    sizes = [len(s) for s in train_sources]
    steps_per_epoch = max(sum(sizes) // params.batch, 1)
    total = steps_per_epoch * params.epochs
    parameters = sum(p.numel() for p in model.parameters())
    meta: dict[str, Any] = {
        "parametros": asdict(params),
        "config": asdict(config),
        "fuentes_train": {s.name: len(s) for s in train_sources},
        "fuentes_val": {s.name: len(s) for s in val_sources},
        "parametros_red": parameters,
        "dispositivo": device,
        "torch": str(torch.__version__),
        "estado": "en curso",
        **(extra_meta or {}),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    log(
        f"red: {parameters / 1e6:.2f} M de parámetros en {device}; "
        f"{dict(zip([s.name for s in train_sources], sizes, strict=True))}"
    )
    curves: list[dict[str, Any]] = []
    best = math.inf
    step = 0
    with ThreadPoolExecutor(max_workers=params.workers) as pool:
        for epoch in range(params.epochs):
            started = time.perf_counter()
            model.train()
            rng = np.random.default_rng([params.seed, epoch])
            # Lotes de una sola fuente (los tamaños de imagen difieren), mezclados entre sí.
            plan = []
            for k, source in enumerate(train_sources):
                order = rng.permutation(len(source))
                plan += [
                    (k, order[i : i + params.batch])
                    for i in range(0, len(order) - params.batch + 1, params.batch)
                ]
            plan = [plan[i] for i in rng.permutation(len(plan))]
            terms: list[dict[str, float]] = []
            for k, indices in plan:
                source = train_sources[k]
                items = list(pool.map(lambda i, s=source, e=epoch: s.item(int(i), e), indices))
                batch = to_batch(items)
                inputs = [
                    batch[n].to(device) for n in ("left", "right", "jacobian", "rays", "calib")
                ]
                batch["target"], batch["mask"] = (
                    batch["target"].to(device),
                    batch["mask"].to(device),
                )
                for group in optimizer.param_groups:
                    group["lr"] = params.lr * _lr_factor(
                        step, total, params.warmup_steps, params.min_lr_ratio
                    )
                loss, parts = loss_terms(model(*inputs), batch, params.aux_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"pérdida no finita en la época {epoch}, paso {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
                optimizer.step()
                terms.append(parts)
                step += 1
            val = evaluate(model, val_sources, device, with_classic=False)["resumen"]
            score = float(np.mean([v["red_filas"] + v["red_columnas"] for v in val.values()]))
            row = {
                "epoca": epoch,
                "campo_train": round(float(np.mean([t["campo"] for t in terms])), 4),
                "aux_train": round(float(np.mean([t["aux"] for t in terms])), 4),
                "error_val_px": round(score, 4),
                "segundos": round(time.perf_counter() - started, 1),
            }
            curves.append(row)
            with (run_dir / "curvas.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(curves)
            state = {
                "model": model.state_dict(),
                "config": asdict(config),
                "epoch": epoch,
                "error_val_px": score,
            }
            torch.save(state, run_dir / "ultimo.pt")
            improved = score < best
            if improved:
                best = score
                torch.save(state, run_dir / "mejor.pt")
            log(
                f"época {epoch:2d}: campo {row['campo_train']:.3f} px, aux {row['aux_train']:.3f}"
                f" | val {score:.3f} px ({row['segundos']:.0f} s){' *' if improved else ''}"
            )
    meta["estado"] = "terminado"
    meta["mejor_error_val_px"] = best
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return meta


def load_model(checkpoint: str | Path, device: str = "auto") -> tuple[Any, str]:
    import torch

    from visionx.depth.mono import pick_device

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = pick_device(device)
    model = build_model(AlignConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), device


def without_column_shift(field: np.ndarray) -> np.ndarray:
    """El campo (2, H, W) sin su corrimiento constante de columnas (la mediana de `field[0]`).

    La pérdida no mira ese corrimiento (no se ve en las filas; lo arregla la distancia
    conocida), así que la red lo deja en cualquier valor y distinto en cada par. Aplicarlo
    corre la disparidad entera: con la v2, 0,7–1,3 px a 960 de ancho, ~3–10 % de profundidad a
    1,5 m, y la distancia conocida medida en un par no lo cancela en los otros.
    """
    out = field.copy()
    out[0] -= np.median(out[0])
    return out


def right_field(
    model: Any,
    device: str,
    left: np.ndarray,
    right: np.ndarray,
    geometry: Any,
    distortion: np.ndarray | None,
    size: tuple[int, int],
    calibration_known: bool = True,
) -> np.ndarray:
    """Campo de la red para un par, llevado a la resolución de captura: (2, H, W) en px.

    La red trabaja a `size` (más chico): se reduce el par, se predice el campo, se le saca el
    corrimiento constante de columnas y se lo agranda escalando también los px. Separado de
    `apply_field` para poder combinar campos de varios pares antes de aplicarlos.
    """
    import cv2

    from visionx.depth.crestereo_adapt.alignnet import (
        calibration_vector,
        jacobian_maps,
        ray_maps,
    )

    small = geometry.resized(*size)
    item = {
        "left": cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .transpose(2, 0, 1)
        / 255.0,
        "right": cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .transpose(2, 0, 1)
        / 255.0,
        "jacobian": jacobian_maps(small),
        "rays": ray_maps(small),
        "calib": calibration_vector(small, distortion, calibration_known),
        "target": np.zeros((2, size[1], size[0]), dtype=np.float32),
        "mask": np.ones((size[1], size[0]), dtype=bool),
        "strength": 0.0,
    }
    field = without_column_shift(predict_field(model, item, device))
    height, width = right.shape[:2]
    scale_x, scale_y = width / size[0], height / size[1]
    fx = cv2.resize(field[0], (width, height), interpolation=cv2.INTER_LINEAR) * scale_x
    fy = cv2.resize(field[1], (width, height), interpolation=cv2.INTER_LINEAR) * scale_y
    return np.stack([fx, fy]).astype(np.float32)


def pooled_field(fields: Sequence[np.ndarray]) -> np.ndarray:
    """Un solo campo para varios pares de la misma cámara: la mediana por píxel.

    La desalineación de una cámara es la misma en todos sus pares; lo que cambia de par en par
    es el ruido de la red. La mediana (y no el promedio) aguanta un par malo, como la regla del
    clásico aguanta puntos malos.
    """
    if not fields:
        raise InvalidInput("hace falta al menos un campo")
    return np.median(np.stack(fields), axis=0).astype(np.float32)


def apply_field(
    right: np.ndarray,
    field: np.ndarray,
    base_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Remuestrea el derecho con el campo (2, H, W): el píxel (u, v) toma (u+fx, v+fy).

    Con `base_maps` (mapas absolutos de una corrección anterior, como los del clásico con
    lente) el campo se aplica sobre la imagen ya corregida sin remuestrear dos veces: se
    componen los mapas y la imagen original se lee una sola vez.
    """
    import cv2

    height, width = right.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = u + field[0], v + field[1]
    if base_maps is not None:
        # corregida(p) = original(base(p)); la red pide corregida(p + f) = original(base(p + f)).
        edge = cv2.BORDER_REPLICATE
        map_x, map_y = (
            cv2.remap(base_maps[0], map_x, map_y, cv2.INTER_LINEAR, borderMode=edge),
            cv2.remap(base_maps[1], map_x, map_y, cv2.INTER_LINEAR, borderMode=edge),
        )
    return cv2.remap(right, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def correct_right(
    model: Any,
    device: str,
    left: np.ndarray,
    right: np.ndarray,
    geometry: Any,
    distortion: np.ndarray | None,
    size: tuple[int, int],
    calibration_known: bool = True,
) -> np.ndarray:
    """Corrige el ojo derecho de un par a resolución de captura con la red (un solo remuestreo)."""
    field = right_field(model, device, left, right, geometry, distortion, size, calibration_known)
    return apply_field(right, field)
