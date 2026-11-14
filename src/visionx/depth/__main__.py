"""Evaluar métodos de profundidad sobre datasets públicos (T05).

    uv run python -m visionx.depth evaluar B1 instereo2k --grupo val
    uv run python -m visionx.depth evaluar B2 instereo2k --grupo val --limite 20
    uv run python -m visionx.depth evaluar B1 middeval3 --grupo test --test-congelado

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

from visionx.depth.datasets import middlebury
from visionx.depth.datasets.catalog import CATALOG
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.metrics import aggregate, disparity_metrics
from visionx.depth.stereo import VARIANTS, SgbmMatcher, SgbmParams, round_up_16
from visionx.shared.config import ROOT, load_config
from visionx.shared.errors import InvalidInput

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


def evaluate_sample(sample: StereoSample, variant: str, params: SgbmParams) -> list[dict[str, Any]]:
    matcher = SgbmMatcher(variant, params)
    doffs = sample.geometry.doffs if sample.geometry else 0.0
    start = time.perf_counter()
    out = matcher.compute(sample.left, sample.right, doffs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    regions = {"all": sample.valid}
    if sample.nonoccluded is not None:
        regions["noc"] = sample.nonoccluded
    rows = []
    for region, mask in regions.items():
        m = disparity_metrics(out.disparity_px, out.valid, sample.disparity, mask,
                              geometry=sample.geometry)  # fmt: skip
        rows.append({"muestra": sample.sample_id, "escena": sample.scene_id, "region": region,
                     "num_disparities": float(params.num_disparities), "ms": elapsed_ms,
                     **m})  # fmt: skip
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
    refs = info.index(data_root / info.name)
    groups = info.partition(refs)
    if args.grupo not in groups:
        raise InvalidInput(f"{info.name}: grupos disponibles {sorted(groups)}")
    scenes = set(groups[args.grupo])
    refs = [r for r in refs if r.scene_id in scenes]
    if args.limite:
        # Repartidas a lo largo de la lista, no las primeras N (serían escenas vecinas).
        idx = np.linspace(0, len(refs) - 1, min(args.limite, len(refs))).round().astype(int)
        refs = [refs[i] for i in sorted(set(idx.tolist()))]

    rule = depth_cfg["evaluacion"]["rango_busqueda"][info.name]
    base = SgbmParams.from_config(depth_cfg["sgbm"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-{args.metodo}-{info.name}-{args.grupo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, ref in enumerate(refs, 1):
        sample = ref.load()
        params = SgbmParams.from_config(depth_cfg["sgbm"], num_disparities=search_range(
            sample, rule, data_root))  # fmt: skip
        sample_rows = evaluate_sample(sample, args.metodo, params)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.depth",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    ev = sub.add_parser("evaluar", help="correr un método sobre un dataset y medir")
    ev.add_argument("metodo", choices=VARIANTS)
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
    args = parser.parse_args(argv)
    try:
        return _evaluate(args)
    except InvalidInput as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
