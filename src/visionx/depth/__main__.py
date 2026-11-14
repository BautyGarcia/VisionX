"""Evaluar métodos de profundidad sobre datasets públicos (T05).

    uv run python -m visionx.depth evaluar B1 instereo2k --grupo val
    uv run python -m visionx.depth evaluar B2 instereo2k --grupo val --limite 20
    uv run python -m visionx.depth evaluar B4 instereo2k --grupo val      # necesita --extra ml
    uv run python -m visionx.depth evaluar B1 middeval3 --grupo test --test-congelado
    uv run python -m visionx.depth bajar-mono   # una vez: pesos de DA V2 Small con hash
    uv run --extra ml python -m visionx.depth ajustar   # T05 PR 3: elige con InStereo2K val
    uv run --extra ml python -m visionx.depth latencia recordings/dev-1114-distancias  # M9

**El test público no elige nada.** MiddEval3 y ETH3D son test (TECNICO T04). Los parámetros
se ajustan con la validación de InStereo2K; recién con todo congelado en la config se corre
el test, una vez, con `--test-congelado`. Sin esa bandera el grupo test se rechaza.

Salida en outputs/profundidad/<fecha>-<metodo>-<dataset>-<grupo>/ (fuera de Git):
    muestras.csv  una fila por muestra y región ("all" y, si hay, "noc")
    resumen.yaml  agregados por escena y por píxel, parámetros, commit y comando
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visionx.depth.align import AlignParams
from visionx.depth.datasets import middlebury
from visionx.depth.datasets.catalog import CATALOG
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.hybrid import HYBRID_VARIANTS, HybridMatcher
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.mono import DepthAnythingV2, MonoConfig, MonoModel
from visionx.depth.mono import download as download_mono
from visionx.depth.stereo import VARIANTS, SgbmMatcher, SgbmParams, round_up_16
from visionx.depth.tuning import Pair
from visionx.shared.config import ROOT, load_config
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError

METHODS = VARIANTS + HYBRID_VARIANTS

TEST_GROUPS = {"test"}


def search_range(sample: StereoSample, rule: Any, data_root: Path) -> int:
    """num_disparities para una muestra según la regla de la config.

    - "calib": el `ndisp` que publica Middlebury en calib.txt (por escena), redondeado a 16.
    - un entero: rango fijo para todo el dataset (ver comentarios en la config).
    """
    if rule == "calib":
        calib_files = [f for f in sample.files if f.endswith("calib.txt")]
        if not calib_files:
            raise InvalidInput(f"{sample.sample_id}: la regla 'calib' necesita calib.txt")
        ndisp = middlebury.read_calib(data_root / sample.dataset / calib_files[0]).get("ndisp")
        if not isinstance(ndisp, float):
            raise InvalidInput(f"{sample.sample_id}: calib.txt sin ndisp")
        return round_up_16(ndisp)
    if isinstance(rule, int) and not isinstance(rule, bool):
        return round_up_16(rule)
    raise InvalidInput(f"regla de rango {rule!r}: se esperaba 'calib' o un entero")


def select_refs(info: Any, data_root: Path, group: str, limit: int = 0) -> list[Any]:
    """Muestras de un grupo de la partición; con `limit`, N repartidas a lo largo de la lista
    (no las primeras N, que serían escenas vecinas)."""
    refs = info.index(data_root / info.name)
    groups = info.partition(refs)
    if group not in groups:
        raise InvalidInput(f"{info.name}: grupos disponibles {sorted(groups)}")
    scenes = set(groups[group])
    refs = [r for r in refs if r.scene_id in scenes]
    if limit:
        idx = np.linspace(0, len(refs) - 1, min(limit, len(refs))).round().astype(int)
        refs = [refs[i] for i in sorted(set(idx.tolist()))]
    return refs


def make_matcher(
    method: str, params: SgbmParams, depth_cfg: dict[str, Any], mono: MonoModel | None
) -> SgbmMatcher | HybridMatcher:
    if method in VARIANTS:
        return SgbmMatcher(method, params)
    if mono is None:
        raise InvalidInput(f"{method} necesita el modelo monocular")
    stereo_variant = depth_cfg["b4"]["estereo"]
    align_params = AlignParams.from_config(depth_cfg["alineacion"])
    return HybridMatcher(method, stereo_variant, params, mono, align_params)


def evaluate_sample(
    sample: StereoSample, matcher: SgbmMatcher | HybridMatcher, num_disparities: int
) -> list[dict[str, Any]]:
    doffs = sample.geometry.doffs if sample.geometry else 0.0
    start = time.perf_counter()
    out = matcher.compute(sample.left, sample.right, doffs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    extra: dict[str, Any] = {}
    alignment = getattr(out, "alignment", None)
    if alignment is not None:
        # Diagnóstico de B3/B4. Como floats se promedian en el resumen (p. ej. tasa de
        # alineaciones aceptadas); el motivo queda como texto en el CSV.
        extra = {
            "alin_aceptada": float(alignment.accepted),
            "alin_motivo": alignment.reason or "",
            "alin_s": alignment.scale,
            "alin_t": alignment.offset,
            "alin_grilla": alignment.grid_coverage,
            "alin_inliers": alignment.inlier_ratio,
            "alin_reservadas": alignment.holdout_inlier_ratio,
            "alin_reservadas_p50_px": alignment.holdout_p50_px,
            "fraccion_mono": out.stats["fraccion_mono"],
            "ms_mono": out.stats["ms_mono"],
        }
    regions = {"all": sample.valid}
    if sample.nonoccluded is not None:
        regions["noc"] = sample.nonoccluded
    rows = []
    for region, mask in regions.items():
        m = disparity_metrics(out.disparity_px, out.valid, sample.disparity, mask,
                              geometry=sample.geometry)  # fmt: skip
        rows.append({"muestra": sample.sample_id, "escena": sample.scene_id, "region": region,
                     "num_disparities": float(num_disparities), "ms": elapsed_ms,
                     **m, **extra})  # fmt: skip
    return rows


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return f"{sha}{'-con-cambios' if dirty else ''}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and value != value:  # NaN no es YAML portable
        return None
    if isinstance(value, dict):
        return {k: _jsonable(x) for k, x in value.items()}
    return value


def _evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    if args.dataset not in CATALOG:
        raise InvalidInput(f"dataset {args.dataset!r}: se esperaba uno de {sorted(CATALOG)}")
    info = CATALOG[args.dataset]
    if args.grupo in TEST_GROUPS and not args.test_congelado:
        print("El grupo test no se usa para ajustar. Si los parámetros ya están congelados en "
              "la config y registrados en BITACORA, repetir con --test-congelado.",
              file=sys.stderr)  # fmt: skip
        return 2

    data_root = Path(args.datos)
    refs = select_refs(info, data_root, args.grupo, args.limite)

    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    mono = None
    if args.metodo in HYBRID_VARIANTS:
        mono = DepthAnythingV2(MonoConfig.from_config(depth_cfg["mono"]))
        print(f"monocular: {mono.model_id} en {mono.device}")
    base = SgbmParams.from_config(depth_cfg["sgbm"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-{args.metodo}-{info.name}-{args.grupo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, ref in enumerate(refs, 1):
        sample = ref.load()
        ndisp = search_range(sample, rule, data_root)
        params = SgbmParams.from_config(depth_cfg["sgbm"], num_disparities=ndisp)
        matcher = make_matcher(args.metodo, params, depth_cfg, mono)
        sample_rows = evaluate_sample(sample, matcher, ndisp)
        rows.extend(sample_rows)
        first = sample_rows[0]
        print(f"[{i}/{len(refs)}] {ref.sample_id}: EPE {first['epe']:.2f} px, "
              f"bad-2 {first['bad2']:.1f} %, cobertura {first['cobertura']:.0%}, "
              f"{first['ms']:.0f} ms")  # fmt: skip

    with (out_dir / "muestras.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "metodo": args.metodo,
        "dataset": info.name,
        "grupo": args.grupo,
        "test_congelado": bool(args.test_congelado),
        "commit": _git_commit(),
        "comando": " ".join(sys.argv),
        "rango_busqueda": rule,
        "parametros": asdict(base),
        "hibrido": None
        if mono is None
        else {
            "monocular": mono.model_id,
            "estereo_anclas": depth_cfg["b4"]["estereo"],
            "alineacion": asdict(AlignParams.from_config(depth_cfg["alineacion"])),
        },
        "regiones": {
            region: _jsonable(aggregate(r for r in rows if r["region"] == region))
            for region in sorted({r["region"] for r in rows})
        },
    }
    with (out_dir / "resumen.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    main_region = "noc" if "noc" in summary["regiones"] else "all"
    s = summary["regiones"][main_region]["por_escena"]
    print(
        f"\n{args.metodo} en {info.name}/{args.grupo} ({main_region}, media por escena): "
        f"EPE {s['epe']:.2f} px, bad-2 {s['bad2']:.1f} %, cobertura {s['cobertura']:.0%}"
    )
    print(f"resultados en {out_dir}")
    return 0


TUNING_DATASET = "instereo2k"  # el único con val (TECNICO T04); el test no se ajusta


def _load_pairs(refs: list[Any], rule: Any, data_root: Path, mono: MonoModel) -> list[Pair]:
    pairs = []
    for i, ref in enumerate(refs, 1):
        sample = ref.load()
        start = time.perf_counter()
        relative = mono.predict(sample.left)
        mono_ms = (time.perf_counter() - start) * 1e3
        pairs.append(Pair(sample, search_range(sample, rule, data_root), relative, mono_ms))
        if i % 25 == 0 or i == len(refs):
            print(f"  {i}/{len(refs)} pares cargados con monocular")
    return pairs


def _tune(args: argparse.Namespace) -> int:
    from visionx.depth import tuning

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    with Path(args.grillas).open(encoding="utf-8") as f:
        grids = yaml.safe_load(f)
    info = CATALOG[TUNING_DATASET]
    data_root = Path(args.datos)
    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    mono = DepthAnythingV2(MonoConfig.from_config(depth_cfg["mono"]))
    print(f"monocular: {mono.model_id} en {mono.device}")
    mono.predict(np.zeros((64, 64, 3), np.uint8))  # calentamiento: el primero compila

    val = _load_pairs(select_refs(info, data_root, "val", args.limite), rule, data_root, mono)
    base = SgbmParams.from_config(depth_cfg["sgbm"])
    align_base = AlignParams.from_config(depth_cfg["alineacion"])
    start = time.perf_counter()
    trials = tuning.tune(val, base, align_base, grids)
    minutes = (time.perf_counter() - start) / 60
    mono_ms = float(np.median([p.mono_ms for p in val]))
    del val  # libera ~1 GB antes de cargar la confirmación

    conf = grids["confirmacion"]
    n_conf = args.limite or conf["muestras"]
    print(f"confirmación: {conf['mejores']} mejores B4 en {n_conf} pares de {conf['grupo']}")
    other = _load_pairs(select_refs(info, data_root, conf["grupo"], n_conf), rule, data_root,
                        mono)  # fmt: skip
    candidates = tuning.ranking(trials, "B4", conf["mejores"])
    confirmation = tuning.confirm(other, candidates)
    holds = max(confirmation, key=lambda c: c["acierto2_confirmacion"]) is confirmation[0]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-ajuste-{info.name}-val"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [t.flat() for t in trials]
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "ensayos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": info.name,
        "grupo": "val",
        "objetivo": f"{tuning.OBJECTIVE}, media por muestra",
        "commit": _git_commit(),
        "comando": " ".join(sys.argv),
        "monocular": mono.model_id,
        "ms_mono_mediana": mono_ms,
        "minutos": minutes,
        "ensayos": len(trials),
        "propuesta_config": tuning.frozen(trials),
        "tabla_val": _jsonable(tuning.table(trials)),
        "confirmacion": {
            "grupo": conf["grupo"],
            "pares": len(other),
            "orden_se_mantiene": holds,
            "candidatos": _jsonable({"c": confirmation})["c"],
        },  # fmt: skip
    }
    with (out_dir / "elegidos.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print("\nB0-B4 con lo elegido (val, media por muestra):")
    for name, s in report["tabla_val"].items():
        print(
            f"  {name}: acierto-2 {s['acierto2']:.2f} %, EPE {s['epe']:.2f} px, "
            f"bad-2 {s['bad2']:.1f} %, cobertura {s['cobertura']:.0%}, {s['ms']:.0f} ms"
        )
    print(f"confirmación en {conf['grupo']}: el orden {'se mantiene' if holds else 'CAMBIA'}")
    print(f"resultados en {out_dir}")
    return 0


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise InvalidInput(f"resolución {text!r}: se esperaba ANCHOxALTO") from exc
    return w, h


def _latency(args: argparse.Namespace) -> int:
    import platform

    from visionx.calibration import StereoCalibration
    from visionx.camera.rectified import RectifiedSource
    from visionx.camera.replay import ReplaySource
    from visionx.depth import latency
    from visionx.depth.hybrid import HybridEstimator
    from visionx.depth.stereo import SgbmEstimator, compute_geometry, num_disparities_for

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    calib_path = ROOT / cfg["camera"]["calibration"]
    source = RectifiedSource(ReplaySource(args.clip), StereoCalibration.load(calib_path))
    frames = []
    while (frame := source.read()) is not None and len(frames) < args.pares:
        frames.append(frame)
    source.close()
    capture = source.geometry
    print(f"{len(frames)} pares rectificados de {args.clip} ({capture.size[0]}x{capture.size[1]})")

    mono = None
    if args.metodo in HYBRID_VARIANTS:
        mono = DepthAnythingV2(MonoConfig.from_config(depth_cfg["mono"]))
        print(f"monocular: {mono.model_id} en {mono.device}")
    min_depth = float(depth_cfg["range_m"][0])
    results: dict[str, Any] = {}
    for text in args.resoluciones:
        size = _parse_size(text)
        geom = compute_geometry(capture, size)
        ndisp = num_disparities_for(geom, min_depth)  # ver desde range_m[0] a esa resolución
        params = SgbmParams.from_config(depth_cfg["sgbm"], num_disparities=ndisp)
        if mono is None:
            est = SgbmEstimator(args.metodo, params, capture, size)

            def run(frame: Any, est: Any = est) -> dict[str, float]:
                result = est.predict(frame)
                return {"total_ms": result.inference_ms,
                        "cobertura": float(result.valid_mask.mean())}  # fmt: skip
        else:
            matcher = make_matcher(args.metodo, params, depth_cfg, mono)
            est = HybridEstimator(matcher, capture, size)

            def run(frame: Any, est: Any = est) -> dict[str, float]:
                result, out = est.predict_detailed(frame)
                return {"total_ms": result.inference_ms, "estereo_ms": out.stats["ms_estereo"],
                        "mono_ms": out.stats["ms_mono"],
                        "alineacion_ms": out.stats["ms_alineacion"],
                        "alin_aceptada": float(out.alignment.accepted),
                        "cobertura": out.stats["cobertura"],
                        "fraccion_mono": out.stats["fraccion_mono"]}  # fmt: skip

        print(f"{text}: rango {ndisp} px, {args.calentamiento} de calentamiento + "
              f"{args.medidos} medidos…")  # fmt: skip
        stats = latency.measure(frames, run, args.calentamiento, args.medidos)
        results[text] = {"num_disparities": ndisp, **stats}
        t = stats["total_ms"]
        print(f"  total P50 {t['p50']:.0f} ms, P95 {t['p95']:.0f} ms "
              f"(~{1000 / t['p50']:.1f} pares/s)")  # fmt: skip

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-latencia-{args.metodo}-{Path(args.clip).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metodo": args.metodo,
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
            "dispositivo_mono": getattr(mono, "device", None),
        },  # fmt: skip
        "parametros": asdict(SgbmParams.from_config(depth_cfg["sgbm"])),
        "alineacion": depth_cfg["alineacion"] if mono else None,
        "anclas": depth_cfg["b4"]["estereo"] if mono else None,
        "memoria_pico_mb": latency.peak_rss_mb(),
        "resoluciones": results,
    }
    with (out_dir / "latencia.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(f"resultados en {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.depth",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    ev = sub.add_parser("evaluar", help="correr un método sobre un dataset y medir")
    ev.add_argument("metodo", choices=METHODS)
    ev.add_argument("dataset")
    ev.add_argument("--grupo", default="val", help="grupo de la partición (val, train, test)")
    ev.add_argument("--limite", type=int, default=0, help="evaluar solo N muestras repartidas")
    ev.add_argument(
        "--test-congelado",
        action="store_true",
        help="confirma que los parámetros ya están congelados (habilita test)",
    )
    ev.add_argument("--config", default=None, help="override YAML explícito")
    ev.add_argument("--datos", default=str(ROOT / "data" / "raw"))
    ev.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    sub.add_parser("bajar-mono", help="bajar una vez los pesos fijados de DA V2 Small")
    aj = sub.add_parser("ajustar", help="elegir parámetros de B0-B4 con InStereo2K val")
    aj.add_argument("--grillas", default=str(ROOT / "configs" / "ajuste-t05.yaml"))
    aj.add_argument("--limite", type=int, default=0, help="prueba rápida con N pares")
    aj.add_argument("--config", default=None, help="override YAML explícito")
    aj.add_argument("--datos", default=str(ROOT / "data" / "raw"))
    aj.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    la = sub.add_parser("latencia", help="latencia M9 de un método sobre un clip ELP")
    la.add_argument("clip", help="carpeta de una grabación sin rectificar (T02)")
    la.add_argument("--metodo", choices=METHODS, default="B4")
    la.add_argument("--resoluciones", nargs="+", default=["640x360", "960x540", "1280x720"])
    la.add_argument("--calentamiento", type=int, default=50)
    la.add_argument("--medidos", type=int, default=300)
    la.add_argument("--pares", type=int, default=100, help="pares distintos en memoria")
    la.add_argument("--config", default=None, help="override YAML explícito")
    la.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    args = parser.parse_args(argv)
    try:
        if args.comando == "bajar-mono":
            cfg = MonoConfig.from_config(load_config(None)["depth"]["mono"])
            print(f"pesos verificados (SHA-256 {cfg.weights_sha256[:12]}…): {download_mono(cfg)}")
            return 0
        if args.comando == "ajustar":
            return _tune(args)
        if args.comando == "latencia":
            return _latency(args)
        return _evaluate(args)
    except (InvalidInput, EstimatorError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
