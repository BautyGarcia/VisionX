"""CLI del GT-sintético-ELP.

uv run python -m visionx.gtsynth resumen  data/gt-sintetico-elp [--variante elp]
uv run python -m visionx.gtsynth objetos  data/gt-sintetico-elp --csv outputs/gtsynth/objetos.csv
uv run python -m visionx.gtsynth verificar data/gt-sintetico-elp [--max 20]
uv run --extra ml python -m visionx.gtsynth evaluar data/gt-sintetico-elp --particion val \
    --variante elp --profundidad crestereo-preciso --csv outputs/gtsynth/val-elp.csv
    [--alinear clasico|red|clasico+red --red mejor.pt] [--vistas-por-camara 8]
    [--distancia-conocida]

Variantes de `evaluar`: elp, hermana-0.5, hermana-1 (sobre las vistas elp) y distinta.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from visionx.gtsynth.dataset import consistencia, objetos_de, particion_de, vistas


def cmd_resumen(args: argparse.Namespace) -> int:
    por_part: Counter = Counter()
    clases: Counter = Counter()
    rangos: Counter = Counter()
    n_obj = 0
    for v in vistas(args.raiz, args.variante):
        por_part[particion_de(v.escena) or "sin-particion"] += 1
        for o in objetos_de(v):
            n_obj += 1
            clases[o.coco or f"({o.clase})"] += 1
            r = o.z_frente_m
            rangos["0,5–1" if r < 1 else "1–2" if r < 2 else "2–3" if r < 3 else "3–4" if r < 4
                   else "4+"] += 1  # fmt: skip
    print(f"vistas por partición: {dict(por_part)}")
    print(f"objetos: {n_obj}; por rango (m): {dict(sorted(rangos.items()))}")
    print("clases más frecuentes (entre paréntesis: sin equivalente en el detector):")
    for c, k in clases.most_common(25):
        print(f"  {c:28} {k}")
    return 0


def cmd_objetos(args: argparse.Namespace) -> int:
    rows = []
    for v in vistas(args.raiz, args.variante):
        for o in objetos_de(v, min_px=args.min_px):
            row = dataclasses.asdict(o)
            row["caja"] = " ".join(map(str, o.caja))
            row["particion"] = particion_de(v.escena)
            rows.append(row)
    if not rows:
        print("no hay objetos", file=sys.stderr)
        return 1
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} objetos en {path}")
    return 0


def cmd_verificar(args: argparse.Namespace) -> int:
    malas = 0
    exactas = []
    for v in islice(vistas(args.raiz, args.variante), args.max):
        e = consistencia(v)
        exactas.append(e["exacta"])
        gana = e["exacta"] < min(e["sin_mover"], e["mas_10pct"], e["menos_10pct"])
        malas += not gana
        print(f"{v.clave}: exacta {e['exacta']:.2f}  sin mover {e['sin_mover']:.2f}  "
              f"+10 % {e['mas_10pct']:.2f}  -10 % {e['menos_10pct']:.2f}  "
              f"{'ok' if gana else 'MAL'}")  # fmt: skip
    if not exactas:
        print("no hay vistas", file=sys.stderr)
        return 1
    print(f"mediana del error con la disparidad exacta: {np.median(exactas):.2f} niveles de gris; "
          f"vistas donde no gana: {malas}")  # fmt: skip
    return 0 if malas == 0 else 1


def cmd_evaluar(args: argparse.Namespace) -> int:
    import json
    import time

    from visionx.gtsynth.dataset import ELP_RENDER
    from visionx.gtsynth.evaluar import Medidor, medir, resumen
    from visionx.gtsynth.variantes import es_hermana
    from visionx.shared.config import load_config

    cfg = load_config(args.config)
    carpeta = "elp" if es_hermana(args.variante) else args.variante
    lista = list(vistas(args.raiz, carpeta, args.particion))
    if not lista:
        print(f"no hay vistas {carpeta}/{args.particion}", file=sys.stderr)
        return 1
    m = Medidor.armar(args.profundidad, cfg, ELP_RENDER)
    t0 = time.time()

    def progreso(n, v):
        if n % 10 == 0 or n == len(lista):
            print(f"  {n}/{len(lista)} vistas ({time.time() - t0:.0f} s)", flush=True)

    alineador = None
    if args.alinear != "ninguna":
        from visionx.depth.crestereo_adapt.align_apply import Alineador
        from visionx.gtsynth.variantes import distorsion_nominal

        alineador = Alineador(args.alinear, ELP_RENDER, distorsion_nominal(), args.red)
    filas = medir(
        m, lista, args.variante, args.max, args.min_px, progreso, alineador,
        args.vistas_por_camara, args.distancia_conocida,
    )  # fmt: skip
    if not filas:
        print("no se midió ningún objeto", file=sys.stderr)
        return 1
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)
    r = resumen(filas)
    path.with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                         encoding="utf-8")  # fmt: skip
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"{len(filas)} objetos en {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m visionx.gtsynth", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    vis = sub.add_parser("visor", help="láminas del GT para el stand o el póster")
    vis.add_argument("raiz")
    vis.add_argument("--particion", choices=["train", "val", "test"], default=None)
    vis.add_argument("--cada", type=float, default=5.0, help="segundos por lámina")
    vis.add_argument("--exportar", help="carpeta: guardar JPG en vez de mostrar")
    vis.add_argument("--n", type=int, default=12, help="láminas a exportar")
    vis.add_argument("--semilla", type=int, default=0)
    ev = sub.add_parser("evaluar")
    ev.add_argument("raiz")
    ev.add_argument("--variante", default="elp")
    ev.add_argument("--particion", choices=["train", "val", "test"], required=True)
    ev.add_argument("--profundidad", default="crestereo-preciso")
    ev.add_argument("--config", default=None)
    ev.add_argument("--csv", required=True)
    ev.add_argument("--max", type=int, default=None)
    ev.add_argument("--min-px", type=int, default=400)
    ev.add_argument(
        "--alinear",
        choices=["ninguna", "clasico", "red", "clasico+red"],
        default="ninguna",
        help="corrección del ojo derecho antes de la profundidad (T12)",
    )
    ev.add_argument("--red", default=None, help="checkpoint de la red de alineación")
    ev.add_argument(
        "--vistas-por-camara",
        type=int,
        default=1,
        help="vistas seguidas que ve una misma cámara (hermana compartida, alineación conjunta)",
    )
    ev.add_argument(
        "--distancia-conocida",
        action="store_true",
        help="el objeto apuntado de la primera vista de cada cámara corrige la escala",
    )
    for name in ("resumen", "objetos", "verificar"):
        p = sub.add_parser(name)
        p.add_argument("raiz")
        p.add_argument("--variante", default="elp")
        if name == "objetos":
            p.add_argument("--csv", required=True)
            p.add_argument("--min-px", type=int, default=400)
        if name == "verificar":
            p.add_argument("--max", type=int, default=20)
    args = ap.parse_args(argv)
    cmds = {"resumen": cmd_resumen, "objetos": cmd_objetos, "verificar": cmd_verificar,
            "evaluar": cmd_evaluar}  # fmt: skip
    if args.cmd == "visor":
        from visionx.gtsynth.visor import run

        run(args.raiz, args.particion, args.cada, args.exportar, args.n, args.semilla)
        return 0
    if args.cmd == "evaluar" and args.particion == "test":
        print("test está congelado hasta la evaluación final (docs/INCERTIDUMBRE-Y-GT.md)",
              file=sys.stderr)  # fmt: skip
    return cmds[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
