"""Fusion (T12): caché de ingredientes, prueba de sobreajuste, entrenamiento y evaluación.

    uv run --extra ml python -m visionx.depth.fusion cache                  # InStereo2K train+val
    uv run --extra ml python -m visionx.depth.fusion cache --limite-train 12 --limite-val 6
    uv run --extra ml python -m visionx.depth.fusion overfit --limite-train 12
    uv run --extra ml python -m visionx.depth.fusion entrenar --run f1-base
    uv run --extra ml python -m visionx.depth.fusion entrenar --config configs/fusion-v2.yaml
    uv run --extra ml python -m visionx.depth.fusion entrenar --reanudar f1-base
    uv run --extra ml python -m visionx.depth.fusion evaluar outputs/fusion/f1-base/mejor.pt
    uv run --extra ml python -m visionx.depth.fusion empaquetar f1-base --nombre f1-v1
    uv run --extra ml python -m visionx.depth.fusion evaluar outputs/fusion/paquetes/f1-v1
    uv run python -m visionx.depth.fusion ajustar-b5      # B5 con val (T05), desde la caché

Todo se entrena y se elige con InStereo2K (train para aprender, val para elegir el
checkpoint). **El test público no elige nada:** `evaluar` (y `cache`) sobre el grupo test se
rechazan sin `--test-congelado`, igual que en `python -m visionx.depth`.

Rutas: la caché va a `depth.fusion.cache_root` (por defecto data/cache/fusion), los runs a
outputs/fusion/<run>/ y los paquetes (pesos + metadatos + model card, lo que usa la app) a
outputs/fusion/paquetes/<nombre>/; todo fuera de Git. `--config` es la config de entrenamiento
(configs/fusion-entrenamiento.yaml o configs/fusion-v2.yaml); `--config-base` es un override
explícito de configs/default.yaml (SGBM, alineación, B5, monocular, canales), como `--config`
en T05.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visionx.depth.__main__ import TEST_GROUPS, _git_commit, _jsonable, search_range, select_refs
from visionx.depth.datasets.catalog import CATALOG
from visionx.shared.config import ROOT, load_config
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError

DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "fusion-entrenamiento.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "fusion"


def _depth_cfg(override: str | None) -> dict[str, Any]:
    return load_config(override)["depth"]


def _cache_spec(depth_cfg: dict[str, Any], root: str | None = None):
    from visionx.depth.fusion.cache import CacheSpec

    spec = CacheSpec.from_config(depth_cfg, root)
    if not spec.root.is_absolute():
        spec = replace(spec, root=ROOT / spec.root)  # relativa al repo, no al cwd
    return spec


def _config_record(depth_cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Secciones de `depth` con las que se calculan base y canales (van a config.yaml)."""
    return {
        "sgbm": depth_cfg["sgbm"],
        "alineacion": depth_cfg["alineacion"],
        "mono": depth_cfg["mono"],
        "b4": depth_cfg["b4"],
        **({"b5": depth_cfg["b5"]} if "b5" in depth_cfg else {}),
        "rango_busqueda": {dataset: depth_cfg["evaluacion"]["rango_busqueda"][dataset]},
        "fusion": depth_cfg.get("fusion", {}),
    }


def _spread(items: list[Any], n: int) -> list[Any]:
    if n <= 0 or n >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def _check_test(group: str, allowed: bool) -> bool:
    if group in TEST_GROUPS and not allowed:
        print("El grupo test no se usa para entrenar ni elegir. Si el checkpoint ya está "
              "elegido con val y registrado en BITACORA, repetir con --test-congelado.",
              file=sys.stderr)  # fmt: skip
        return False
    return True


# --- cache ----------------------------------------------------------------------------------


def _cache(args: argparse.Namespace) -> int:
    from visionx.depth.fusion.cache import build_cache
    from visionx.depth.mono import DepthAnythingV2, MonoConfig

    depth_cfg = _depth_cfg(args.config_base)
    if args.dataset not in CATALOG:
        raise InvalidInput(f"dataset {args.dataset!r}: se esperaba uno de {sorted(CATALOG)}")
    info = CATALOG[args.dataset]
    for group in args.grupo:
        if not _check_test(group, args.test_congelado):
            return 2
    spec = _cache_spec(depth_cfg, args.cache_root)
    data_root = Path(args.datos)
    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    holder: dict[str, Any] = {}

    def mono():  # se carga solo si hay algo que calcular
        if "m" not in holder:
            model = DepthAnythingV2(MonoConfig.from_config(depth_cfg["mono"]))
            model.predict(np.zeros((64, 64, 3), np.uint8))  # calentamiento: el primero compila
            print(f"monocular: {model.model_id} en {model.device}")
            holder["m"] = model
        return holder["m"]

    limits = {"train": args.limite_train, "val": args.limite_val}
    print(f"caché en {spec.root} (estéreo {spec.stereo_variant}, monocular "
          f"{spec.mono['model_id']})")  # fmt: skip
    totals = []
    for group in args.grupo:
        refs = select_refs(info, data_root, group, limits.get(group) or args.limite)
        print(f"{info.name}/{group}: {len(refs)} muestras")
        summary = build_cache(refs, lambda s: search_range(s, rule, data_root), spec, mono,
                              data_root)  # fmt: skip
        totals.append((group, summary))
    for group, s in totals:
        per = s["s_por_muestra_calculada"]
        per_text = f", {per:.2f} s por muestra calculada" if s["calculadas"] else ""
        print(f"{info.name}/{group}: {s['calculadas']} calculadas, {s['reusadas']} reusadas, "
              f"{s['segundos']:.1f} s{per_text}; alineación aceptada "
              f"{s['alineacion_aceptada_pct']:.0f} %, {s['mb_total']:.0f} MB "
              f"({s['mb_por_muestra']:.1f} MB por muestra)")  # fmt: skip
    return 0


# --- entrenamiento --------------------------------------------------------------------------


def _run_inputs(record: dict[str, Any], data_root: Path, train_n: int, val_n: int):
    """Muestras, caché y canales de una corrida a partir de las secciones `depth` registradas
    (las de `_config_record`, recién leídas de la config o de un run a reanudar)."""
    from visionx.depth.fusion.channels import ChannelParams
    from visionx.depth.fusion.train import TRAIN_DATASET, RunInputs
    from visionx.depth.hybrid import B5Params

    info = CATALOG[TRAIN_DATASET]
    return RunInputs(
        train_refs=select_refs(info, data_root, "train", train_n),
        val_refs=select_refs(info, data_root, "val", val_n),
        rule=record["rango_busqueda"][info.name],
        data_root=data_root,
        cache_spec=_cache_spec(record),
        channel_params=ChannelParams.from_config(record.get("fusion", {}).get("canales", {})),
        catalog=info.name,
        # Runs de antes de B5 no registran depth.b5: sin B5 (base B4, F0/F1, val sin B5).
        b5_params=B5Params.from_config(record) if "b5" in record else None,
        config_record={"depth": record},
        commit=_git_commit(),
    )


def _train_config(args: argparse.Namespace):
    from visionx.depth.fusion.train import TrainConfig

    cfg = TrainConfig.from_yaml(args.config)
    over: dict[str, Any] = {}
    if getattr(args, "epocas", None):
        over["epochs"] = args.epocas
    if getattr(args, "limite_train", None) is not None:
        over["limit_train"] = args.limite_train
    if getattr(args, "limite_val", None) is not None:
        over["limit_val"] = args.limite_val
    if args.dispositivo:
        over["device"] = args.dispositivo
    if getattr(args, "workers", None) is not None:
        over["num_workers"] = args.workers
    if getattr(args, "semilla", None) is not None:
        over["seed"] = args.semilla  # otra semilla del mismo modelo (TECNICO: tres si se puede)
    # Variantes de receta sin editar el YAML (quedan registradas en el config.yaml del run).
    if getattr(args, "norm", None):
        over["model"] = {**cfg.model.to_dict(), "norm": args.norm}
    if getattr(args, "perdida", None):
        over["loss"] = args.perdida
    if getattr(args, "peso_l1", None) is not None:
        over["l1_weight"] = args.peso_l1
    return TrainConfig.from_dict({**cfg.to_dict(), **over})


def _overfit(args: argparse.Namespace) -> int:
    from visionx.depth.fusion.train import overfit

    cfg = _train_config(args)
    oc = cfg.overfit
    oc = replace(oc, images=args.imagenes or oc.images,
                 crops_per_image=args.recortes_por_imagen or oc.crops_per_image)  # fmt: skip
    cfg = replace(cfg, overfit=oc)
    record = _config_record(_depth_cfg(args.config_base), "instereo2k")
    inp = _run_inputs(record, Path(args.datos), cfg.limit_train, 1)
    refs = _spread(list(inp.train_refs), oc.images)
    print(f"overfit: {len(refs)} imágenes de train x {oc.crops_per_image} recortes "
          f"{cfg.crop[0]}x{cfg.crop[1]}, pérdida {cfg.loss}")  # fmt: skip
    result = overfit(cfg, refs, inp, steps=args.pasos, lr=args.lr)
    print(f"\n{cfg.variant}, {result['parametros']:,} parámetros: EPE {cfg.base} "
          f"{result['epe_base_px']:.3f} px -> Fusion {result['epe_fusion_px']:.3f} px "
          f"(reducción {result['reduccion_pct']:.1f} %, M5 pide >= {oc.target_reduction_pct:g} %"
          f": {'cumple' if result['cumple_m5'] else 'NO cumple'}); {result['pasos']} pasos en "
          f"{result['segundos_entrenamiento']:.0f} s en {result['dispositivo']}")  # fmt: skip
    if args.salida:
        out = Path(args.salida)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{datetime.now():%Y%m%d-%H%M%S}-overfit.yaml"
        report = {"commit": _git_commit(), "comando": " ".join(sys.argv), **result}
        path.write_text(yaml.safe_dump(_jsonable(report), allow_unicode=True, sort_keys=False),
                        encoding="utf-8")  # fmt: skip
        print(f"resultado en {path}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from visionx.depth.fusion.train import TrainConfig, run_training

    out_root = Path(args.salida)
    if args.reanudar:
        if args.norm or args.perdida or args.peso_l1 is not None:
            raise InvalidInput(
                "al reanudar manda la receta del run: sin --norm, --perdida, --peso-l1"
            )
        run_dir = Path(args.reanudar)
        if not run_dir.is_dir():
            run_dir = out_root / args.reanudar
        if not (run_dir / "ultimo.pt").is_file():
            raise InvalidInput(f"{run_dir}: no hay ultimo.pt para reanudar")
        with (run_dir / "config.yaml").open(encoding="utf-8") as f:
            record = yaml.safe_load(f)
        cfg = TrainConfig.from_dict(record["entrenamiento"])
        # Solo se puede cambiar dónde y cómo corre, no qué se entrena.
        if args.dispositivo:
            cfg = replace(cfg, device=args.dispositivo)
        if args.workers is not None:
            cfg = replace(cfg, num_workers=args.workers)
        inp = _run_inputs(record["depth"], Path(args.datos), cfg.limit_train, cfg.limit_val)
        run_training(cfg, run_dir, inp, resume=True)
    else:
        cfg = _train_config(args)
        depth_cfg = _depth_cfg(args.config_base)
        if cfg.variant != depth_cfg.get("fusion", {}).get("variante", cfg.variant):
            print(f"aviso: variante {cfg.variant} de la config de entrenamiento distinta de "
                  "depth.fusion.variante", file=sys.stderr)  # fmt: skip
        name = args.run or f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.variant.lower()}"
        run_dir = out_root / name
        if run_dir.exists():
            raise InvalidInput(f"{run_dir} ya existe: elegir otro --run o usar --reanudar")
        record = _config_record(depth_cfg, "instereo2k")
        inp = _run_inputs(record, Path(args.datos), cfg.limit_train, cfg.limit_val)
        run_training(cfg, run_dir, inp)
    print(f"resultados en {run_dir}")
    return 0


# --- evaluar --------------------------------------------------------------------------------


def _evaluate(args: argparse.Namespace) -> int:
    import torch

    from visionx.depth.fusion.channels import ChannelParams, uses_b5
    from visionx.depth.fusion.data import FusionDataset
    from visionx.depth.fusion.package import load_package
    from visionx.depth.fusion.train import evaluate, load_checkpoint, make_loader, summarize
    from visionx.depth.hybrid import B5Params
    from visionx.depth.mono import pick_device

    if not _check_test(args.grupo, args.test_congelado):
        return 2
    if args.dataset not in CATALOG:
        raise InvalidInput(f"dataset {args.dataset!r}: se esperaba uno de {sorted(CATALOG)}")
    info = CATALOG[args.dataset]
    ckpt_path = Path(args.checkpoint)
    depth_cfg = _depth_cfg(args.config_base)
    device = torch.device(pick_device(args.dispositivo or "auto"))
    if ckpt_path.is_dir():  # paquete: la misma red que usa la app
        pkg = load_package(ckpt_path, str(device))
        _check_base(pkg.base, depth_cfg, args.permitir_otra_base)
        model = pkg.model
        ckpt = {"variant": pkg.variant, "channel_params": pkg.channel_params.to_dict(),
                "epoch": pkg.meta["checkpoint"]["epoca"], "base": pkg.base_variant}  # fmt: skip
        tag = f"paquete-{ckpt_path.name}"
    elif ckpt_path.suffix == ".pt":
        model, ckpt = load_checkpoint(ckpt_path, device)
        _check_same_base(ckpt_path, depth_cfg, args.permitir_otra_base,
                         uses_b5(ckpt["variant"], ckpt["base"]))  # fmt: skip
        tag = f"{ckpt_path.parent.name}-{ckpt_path.stem}"
    else:
        raise InvalidInput("evaluar recibe un checkpoint .pt de un run o la carpeta de un paquete")
    data_root = Path(args.datos)
    refs = select_refs(info, data_root, args.grupo, args.limite)
    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    # B5 de la config en uso (ya comparada con la del entrenamiento si el modelo la usa); sin
    # depth.b5, la evaluación no informa B5.
    b5 = B5Params.from_config(depth_cfg) if "b5" in depth_cfg else None
    ds = FusionDataset(refs, rule, data_root, _cache_spec(depth_cfg), ckpt["variant"],
                       ChannelParams(**ckpt["channel_params"]), crop=None,
                       catalog=info.name, base=ckpt["base"], b5_params=b5)  # fmt: skip
    print(f"{ckpt_path} (variante {ckpt['variant']}, base {ckpt['base']}, época {ckpt['epoch']}) "
          f"{info.name}/{args.grupo}: {len(ds)} muestras en {device}")  # fmt: skip
    start = time.perf_counter()
    rows = evaluate(model, make_loader(ds, 1, args.workers), device)
    seconds = time.perf_counter() - start
    summary = summarize(rows)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / "evaluaciones" / f"{stamp}-{tag}-{info.name}-{args.grupo}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "muestras.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint": str(ckpt_path),
        "variante": ckpt["variant"],
        "base": ckpt["base"],
        "epoca": ckpt["epoch"],
        "dataset": info.name,
        "grupo": args.grupo,
        "test_congelado": bool(args.test_congelado),
        "soporte": "*_fusion/*_base: GT ∩ base válida ∩ Fusion válida (d + doffs > 0, b > 0); "
        "*_comun_*: GT ∩ Fusion ∩ B4 ∩ B5; cobertura y acierto2_gt: toda la GT, cada método "
        "con su máscara",
        "commit": _git_commit(),
        "comando": " ".join(sys.argv),
        "segundos": seconds,
        "resumen": _jsonable(summary),
    }
    with (out_dir / "resumen.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    _print_table(summary, info.name, args.grupo, ckpt["base"])
    print(f"resultados en {out_dir}")
    return 0


def _check_same_base(
    ckpt_path: Path, depth_cfg: dict[str, Any], allow: bool, uses_b5_params: bool = False
) -> None:
    """La base depende de SGBM, alineación, monocular y (si el modelo usa B5) depth.b5: si la
    config actual difiere de la del entrenamiento, el residual aprendido ya no le corresponde."""
    record_path = ckpt_path.parent / "config.yaml"
    if not record_path.is_file():
        print(f"aviso: no encontré {record_path}; no puedo comparar la base", file=sys.stderr)
        return
    with record_path.open(encoding="utf-8") as f:
        trained = yaml.safe_load(f).get("depth", {})
    if not all(k in trained for k in ("sgbm", "alineacion", "mono", "b4")):
        print(f"aviso: {record_path} no registra la base; no puedo compararla", file=sys.stderr)
        return
    _check_base(trained, depth_cfg, allow, uses_b5_params)


def _check_base(
    trained: dict[str, Any],
    depth_cfg: dict[str, Any],
    allow: bool,
    uses_b5_params: bool | None = None,
) -> None:
    """Misma comparación que el estimador de la app (package.base_differences): parámetros
    interpretados, sin el rango de búsqueda (sale del dataset o de la geometría)."""
    from visionx.depth.fusion.package import config_base_differences

    diffs = config_base_differences(trained, depth_cfg, uses_b5_params)
    if diffs and not allow:
        raise InvalidInput(f"la config actual difiere de la del entrenamiento en {diffs}; "
                           "usar --permitir-otra-base solo a propósito")  # fmt: skip
    if diffs:
        print(f"aviso: base distinta de la del entrenamiento en {diffs}", file=sys.stderr)


def _print_table(
    summary: dict[str, dict[str, float]], dataset: str, group: str, base: str = "B4"
) -> None:
    s, sc = summary["por_muestra"], summary["por_escena"]
    print(f"\n{dataset}/{group}: {s['n_muestras']:.0f} muestras, {sc['n_escenas']:.0f} escenas. "
          f"EPE, bad-2 y acierto-2 en el soporte común GT ∩ {base} ∩ Fusion; cobertura y "
          "acierto-2 GT sobre toda la GT, cada método con su máscara")  # fmt: skip
    print(f"{'':10}{'EPE px':>9}{'bad-2 %':>9}{'acierto-2 %':>13}{'EPE escena':>12}"
          f"{'cobertura':>11}{'acierto-2 GT':>14}")  # fmt: skip
    for name, key in ((f"base {base}", "base"), ("Fusion", "fusion")):
        print(f"{name:10}{s['epe_' + key]:9.3f}{s['bad2_' + key]:9.2f}"
              f"{s['acierto2_' + key]:13.2f}{sc['epe_' + key]:12.3f}"
              f"{s['cobertura_' + key]:11.2%}{s['acierto2_gt_' + key]:14.2f}")  # fmt: skip
    print(f"\nControles fijos y Fusion en el soporte común GT ∩ B4 ∩ B5 ∩ Fusion "
          f"({s['fraccion_comun']:.1%} de la GT, media por muestra):")  # fmt: skip
    print(f"{'':10}{'EPE px':>9}{'bad-2 %':>9}{'acierto-2 %':>13}")
    for name, key in (("B4", "b4"), ("B5", "b5"), ("Fusion", "fusion")):
        print(f"{name:10}{s['epe_comun_' + key]:9.3f}{s['bad2_comun_' + key]:9.2f}"
              f"{s['acierto2_comun_' + key]:13.2f}")  # fmt: skip
    print(f"NLL Laplace de Fusion {s['nll_px']:.3f} (px), b mediana {s['b_mediana_px']:.2f} px; "
          "medias por muestra salvo 'EPE escena'")  # fmt: skip


# --- empaquetar ------------------------------------------------------------------------------


def _package(args: argparse.Namespace) -> int:
    from visionx.depth.fusion.package import empaquetar, load_package

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = Path(args.runs) / args.run
    if not run_dir.is_dir():
        raise InvalidInput(f"{args.run}: no es la carpeta de un run ni un run de {args.runs}")
    name = args.nombre or run_dir.name
    dest = empaquetar(run_dir, args.salida, name, args.checkpoint, overwrite=args.reemplazar)
    pkg = load_package(dest, "cpu")  # verifica lo recién escrito (sha256, canales)
    metrics = pkg.meta["metricas_val"]
    epe = (f"val EPE Fusion {metrics['epe_fusion_px']:.3f} px vs base {pkg.base_variant} "
           f"{metrics['epe_base_px']:.3f} px" if isinstance(metrics, dict)
           else "val no medida")  # fmt: skip
    print(f"{pkg.model_id}: {args.checkpoint} de {run_dir.name} (época "
          f"{pkg.meta['checkpoint']['epoca']}; {epe})")  # fmt: skip
    print(f"paquete en {dest}")
    return 0


# --- ajustar B5 ------------------------------------------------------------------------------


def _tune_b5(args: argparse.Namespace) -> int:
    """Grilla de B5 en InStereo2K val y confirmación de los mejores en train (protocolo T05)."""
    from visionx.depth.fusion import b5_tuning
    from visionx.depth.hybrid import B5Params

    depth_cfg = _depth_cfg(args.config_base)
    info = CATALOG["instereo2k"]
    data_root = Path(args.datos)
    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    spec = _cache_spec(depth_cfg)
    cands = b5_tuning.candidates(B5Params.from_config(depth_cfg))

    def ndisp(sample: Any) -> int:
        return search_range(sample, rule, data_root)

    val_refs = select_refs(info, data_root, "val", args.limite)
    print(f"B5: {len(cands)} candidatas (B4, rectas globales y grilla) en {len(val_refs)} "
          "pares de val, desde la caché")  # fmt: skip
    start = time.perf_counter()
    val_rows = b5_tuning.run(val_refs, ndisp, spec, data_root, cands)
    table = b5_tuning.summarize(val_rows, cands)
    best = b5_tuning.ranking(table)
    top = [c for c in cands if c.name in {r["candidato"] for r in best[: args.mejores]}]
    refs_conf = [c for c in cands if c.kind != "B5"]  # B4 y rectas globales, de referencia
    conf_refs = select_refs(info, data_root, "train", args.confirmacion)
    print(f"confirmación: {len(top)} mejores en {len(conf_refs)} pares de train")
    conf_rows = b5_tuning.run(conf_refs, ndisp, spec, data_root, refs_conf + top)
    conf = b5_tuning.summarize(conf_rows, refs_conf + top)
    conf_rank = [r["candidato"] for r in b5_tuning.ranking(conf)]
    val_rank = [r["candidato"] for r in best[: args.mejores]]
    minutes = (time.perf_counter() - start) / 60

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-ajuste-b5-{info.name}-val"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("grilla.csv", table),
        ("confirmacion.csv", conf),
        ("muestras_val.csv", val_rows),
        ("muestras_train.csv", conf_rows),
    ):
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    chosen = next(c for c in cands if c.name == val_rank[0])
    report = {
        "dataset": info.name,
        "objetivo": f"{b5_tuning.OBJECTIVE}, media por muestra, toda la GT",
        "commit": _git_commit(),
        "comando": " ".join(sys.argv),
        "minutos": minutes,
        "pares_val": len(val_refs),
        "pares_confirmacion": len(conf_refs),
        "elegido": chosen.name,
        "propuesta_config": {
            "method": chosen.params.align.method,
            **{
                k: getattr(chosen.params.local, k)
                for k in ("sigma_px", "slope_lambda_px2", "exclude_px")
            },
        },  # fmt: skip
        "orden_val": val_rank,
        "orden_train": conf_rank,
        "orden_se_mantiene": conf_rank[: len(val_rank)] == val_rank,
        "val": _jsonable({"t": [r for r in table if r["tipo"] != "B5"] + best[:3]})["t"],
        "train": _jsonable({"t": conf})["t"],
    }
    with (out_dir / "elegidos.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)

    def show(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            print(f"  {r['candidato']:42}{r['acierto2']:9.2f}{r['epe']:8.3f}{r['bad2']:8.2f}"
                  f"{r['cobertura']:9.2%}{r['ms']:7.0f}")  # fmt: skip

    header = f"  {'':42}{'acierto2':>9}{'EPE px':>8}{'bad-2':>8}{'cobert.':>9}{'ms':>7}"
    print(f"\nval ({len(val_refs)} pares, media por muestra, toda la GT):\n{header}")
    show([r for r in table if r["tipo"] != "B5"] + best)
    print(f"\ntrain ({len(conf_refs)} pares repartidos; confirma, no elige):\n{header}")
    show(sorted(conf, key=lambda r: -r["acierto2"]))
    print(
        f"\nelegido en val: {chosen.name}; orden de los {len(val_rank)} mejores en train: "
        f"{'se mantiene' if report['orden_se_mantiene'] else 'CAMBIA'} ({conf_rank})"
    )
    print(f"resultados en {out_dir}")
    return 0


# --- main -----------------------------------------------------------------------------------


def _latency(args: argparse.Namespace) -> int:
    """M9 de Fusion sobre un clip de la ELP: el mismo protocolo que `visionx.depth latencia`.

    Mide `FusionEstimator.predict_detailed` como lo usa la app: reescalar a la resolución de
    cómputo, SGBM, monocular, recta(s) global(es) y corrección local, canales, red y armado del
    DepthResult. No entran captura, JPEG ni rectificación. La red devuelve su salida a CPU antes
    de terminar, así que la GPU (MPS) queda sincronizada en cada par.
    """
    import platform

    from visionx.calibration import StereoCalibration
    from visionx.camera.rectified import RectifiedSource
    from visionx.camera.replay import ReplaySource
    from visionx.depth import latency
    from visionx.depth.__main__ import _parse_size
    from visionx.depth.fusion.estimator import load_estimator
    from visionx.depth.mono import DepthAnythingV2, MonoConfig

    cfg = load_config(args.config_base)
    calib_path = ROOT / cfg["camera"]["calibration"]
    source = RectifiedSource(ReplaySource(args.clip), StereoCalibration.load(calib_path))
    frames = []
    while (frame := source.read()) is not None and len(frames) < args.pares:
        frames.append(frame)
    source.close()
    capture = source.geometry
    print(f"{len(frames)} pares rectificados de {args.clip} ({capture.size[0]}x{capture.size[1]})")

    # Un solo monocular para todas las resoluciones (la red se carga una vez, como en la app).
    mono = DepthAnythingV2(MonoConfig.from_config(cfg["depth"]["mono"]))
    results: dict[str, Any] = {}
    model_id = None
    for text in args.resoluciones:
        size = _parse_size(text)
        est = load_estimator(cfg, capture, size, package=args.paquete, mono=mono,
                             device=args.dispositivo)  # fmt: skip
        model_id = est.model_id

        def run(frame: Any, est: Any = est) -> dict[str, float]:
            result, out = est.predict_detailed(frame)
            st = out.stats
            return {"total_ms": result.inference_ms, "estereo_ms": st["ms_estereo"],
                    "mono_ms": st["ms_mono"], "alineacion_ms": st["ms_alineacion"],
                    "canales_ms": st["ms_canales"], "red_ms": st["ms_red"],
                    "alin_aceptada": st["alin_aceptada"], "cobertura": st["cobertura"]}  # fmt: skip

        print(f"{text}: {args.calentamiento} de calentamiento + {args.medidos} medidos…")
        stats = latency.measure(frames, run, args.calentamiento, args.medidos)
        results[text] = stats
        t = stats["total_ms"]
        parts = ", ".join(
            f"{k[:-3]} {stats[k]['p50']:.0f}"
            for k in ("estereo_ms", "mono_ms", "alineacion_ms", "canales_ms", "red_ms")
        )
        print(f"  total P50 {t['p50']:.0f} ms, P95 {t['p95']:.0f} ms (~{1000 / t['p50']:.1f} "
              f"pares/s); P50 por etapa: {parts} ms")  # fmt: skip

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-latencia-{model_id}-{Path(args.clip).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "modelo": model_id,
        "paquete": str(args.paquete),
        "clip": str(args.clip),
        "session_id": frames[0].session_id if frames else None,
        "pares_distintos": len(frames),
        "calibracion": cfg["camera"]["calibration"],
        "commit": _git_commit(),
        "comando": " ".join(sys.argv),
        "equipo": {
            "sistema": platform.platform(),
            "procesador": platform.processor(),
            "python": platform.python_version(),
            "dispositivo_mono": mono.device,
            "dispositivo_red": args.dispositivo or "auto",
        },  # fmt: skip
        "memoria_pico_mb": latency.peak_rss_mb(),
        "resoluciones": results,
    }
    with (out_dir / "latencia.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_jsonable(report), f, allow_unicode=True, sort_keys=False)
    print(f"resultados en {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.depth.fusion",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config-base", default=None, help="override YAML de default.yaml")
        p.add_argument("--datos", default=str(ROOT / "data" / "raw"))

    def recipe(p: argparse.ArgumentParser) -> None:
        p.add_argument("--norm", choices=("group", "batch", "none"), default=None,
                       help="normalización de la U-Net (pisa model.norm del YAML)")  # fmt: skip
        p.add_argument("--perdida", choices=("nll", "l1"), default=None, help="pisa loss")
        p.add_argument("--peso-l1", type=float, default=None, help="pisa l1_weight")

    ca = sub.add_parser("cache", help="calcular estéreo, monocular y alineación una vez")
    ca.add_argument("--dataset", default="instereo2k")
    ca.add_argument("--grupo", nargs="+", default=["train", "val"])
    ca.add_argument("--limite", type=int, default=0, help="N muestras repartidas por grupo")
    ca.add_argument("--limite-train", type=int, default=0)
    ca.add_argument("--limite-val", type=int, default=0)
    ca.add_argument("--cache-root", default=None, help="otra raíz (por defecto, la de la config)")
    ca.add_argument("--test-congelado", action="store_true")
    common(ca)

    of = sub.add_parser("overfit", help="M5: aprender de memoria pocos recortes fijos")
    of.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    of.add_argument("--limite-train", type=int, default=None,
                    help="elegir entre N imágenes de train repartidas")  # fmt: skip
    of.add_argument("--imagenes", type=int, default=0)
    of.add_argument("--recortes-por-imagen", type=int, default=0)
    of.add_argument("--pasos", type=int, default=None)
    of.add_argument("--lr", type=float, default=None)
    of.add_argument("--dispositivo", default=None)
    of.add_argument("--salida", default=str(DEFAULT_OUTPUT / "overfit"))
    recipe(of)
    common(of)

    tr = sub.add_parser("entrenar", help="entrenar (o reanudar) un run")
    tr.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    tr.add_argument("--run", default=None, help="nombre del run (por defecto, fecha-variante)")
    tr.add_argument("--reanudar", default=None, help="run a continuar desde ultimo.pt")
    tr.add_argument("--limite-train", type=int, default=None)
    tr.add_argument("--limite-val", type=int, default=None)
    tr.add_argument("--epocas", type=int, default=None)
    tr.add_argument("--dispositivo", default=None)
    tr.add_argument("--workers", type=int, default=None)
    tr.add_argument("--semilla", type=int, default=None, help="pisa seed de la config")
    tr.add_argument("--salida", default=str(DEFAULT_OUTPUT))
    recipe(tr)
    common(tr)

    ev = sub.add_parser("evaluar", help="checkpoint contra B4 en un grupo (mismo soporte)")
    ev.add_argument("checkpoint", help="ultimo.pt o mejor.pt de un run, o carpeta de paquete")
    ev.add_argument("--dataset", default="instereo2k")
    ev.add_argument("--grupo", default="val")
    ev.add_argument("--limite", type=int, default=0)
    ev.add_argument("--test-congelado", action="store_true")
    ev.add_argument("--permitir-otra-base", action="store_true",
                    help="aunque SGBM/alineación/mono difieran del run")  # fmt: skip
    ev.add_argument("--dispositivo", default=None)
    ev.add_argument("--workers", type=int, default=0)
    ev.add_argument("--salida", default=str(DEFAULT_OUTPUT))
    common(ev)

    em = sub.add_parser("empaquetar", help="pesos + metadatos + model card para la app")
    em.add_argument("run", help="carpeta del run o su nombre dentro de --runs")
    em.add_argument("--nombre", default=None, help="nombre del paquete (por defecto, el del run)")
    em.add_argument("--checkpoint", default="mejor.pt", help="mejor.pt (por defecto) o ultimo.pt")
    em.add_argument("--runs", default=str(DEFAULT_OUTPUT))
    em.add_argument("--salida", default=str(DEFAULT_OUTPUT / "paquetes"))
    em.add_argument("--reemplazar", action="store_true", help="pisar un paquete con ese nombre")

    la = sub.add_parser("latencia", help="M9: Fusion completa sobre un clip de la ELP")
    la.add_argument("paquete", help="carpeta de un paquete (empaquetar)")
    la.add_argument("clip", help="carpeta de una grabación (recordings/...)")
    la.add_argument("--resoluciones", nargs="+", default=["640x360", "960x540", "1280x720"])
    la.add_argument("--pares", type=int, default=100, help="pares distintos (se recorren en ciclo)")
    la.add_argument("--calentamiento", type=int, default=50)
    la.add_argument("--medidos", type=int, default=300)
    la.add_argument("--dispositivo", default=None, help="de la red: auto | mps | cpu")
    la.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    common(la)

    b5 = sub.add_parser("ajustar-b5", help="grilla de B5 en val y confirmación en train")
    b5.add_argument("--limite", type=int, default=0, help="N pares de val repartidos (prueba)")
    b5.add_argument("--confirmacion", type=int, default=100, help="pares de train repartidos")
    b5.add_argument("--mejores", type=int, default=3, help="cuántos mejores se confirman")
    b5.add_argument("--salida", default=str(ROOT / "outputs" / "b5"))
    common(b5)

    args = parser.parse_args(argv)
    try:
        if args.comando == "ajustar-b5":
            return _tune_b5(args)
        if args.comando == "cache":
            return _cache(args)
        if args.comando == "overfit":
            return _overfit(args)
        if args.comando == "entrenar":
            return _train(args)
        if args.comando == "empaquetar":
            return _package(args)
        if args.comando == "latencia":
            return _latency(args)
        return _evaluate(args)
    except (InvalidInput, EstimatorError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
