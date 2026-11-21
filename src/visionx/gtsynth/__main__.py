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
from visionx.shared.config import ROOT


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


def cmd_camaras(args: argparse.Namespace) -> int:
    """Precalcular la demo «Cámaras» del stand (CREStereo con cada cámara)."""
    from visionx.gtsynth.camaras import elegir_poses, preparar

    poses = elegir_poses(args.raiz, args.escenas)
    print(f"poses: {poses}")
    preparar(args.raiz, args.salida, poses, args.profundidad)
    print(f"listo en {args.salida}; verla con: python -m visionx.gtsynth camaras-ver {args.salida}")
    return 0


def cmd_deteccion(args: argparse.Namespace) -> int:
    """Recall y precisión de uno o más detectores contra el GT (ver gtsynth/deteccion.py)."""
    from itertools import islice

    from visionx.gtsynth.deteccion import escribir_csv, evaluar, predictor_yolo

    resultados = []
    for modelo in args.modelos:
        path = modelo if modelo.endswith(".pt") else str(ROOT / "weights" / f"{modelo}.pt")
        predecir, conocidas = predictor_yolo(path, args.imgsz, args.conf, args.dispositivo)
        vs = []
        for particion in args.particion:
            vs += list(vistas(args.raiz, args.variante, particion))
        vs = vs[:: max(1, len(vs) // args.max)] if args.max else vs
        r = evaluar(islice(vs, None), predecir, Path(path).stem, conocidas, args.iou, args.min_px)
        resultados.append(r)
        tot = r.total()
        ms = sorted(r.ms)[len(r.ms) // 2] if r.ms else float("nan")
        print(f"{r.modelo}: {r.vistas} vistas, recall {tot.recall:.3f} ({tot.encontrados}/"
              f"{tot.gt}), precisión ≥ {tot.precision:.3f} ({tot.det_ok}/{tot.det}), "
              f"{ms:.0f} ms/cuadro")  # fmt: skip
    if len(resultados) > 1:  # mismas clases para todos: las que conoce el primero
        base = {c for c, k in resultados[0].por_clase.items() if k.gt}
        for r in resultados:
            tot = r.total(base)
            print(f"  en las clases de {resultados[0].modelo}: {r.modelo} recall "
                  f"{tot.recall:.3f}, precisión ≥ {tot.precision:.3f}")  # fmt: skip
    escribir_csv(resultados, Path(args.csv))
    print(f"por clase en {args.csv}")
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
    ca = sub.add_parser("camaras", help="precalcular la demo de cámaras distintas del stand")
    ca.add_argument("raiz")
    ca.add_argument("--salida", default=str(ROOT / "data" / "stand-camaras"))
    ca.add_argument("--escenas", type=int, default=4, help="cuántas vistas de val (una por escena)")
    ca.add_argument("--profundidad", default="crestereo-preciso")
    cv_ = sub.add_parser("camaras-ver", help="mostrar la demo de cámaras en una ventana")
    cv_.add_argument(
        "carpeta",
        nargs="?",
        default=None,
        help="por defecto data/stand-video si tiene videos, si no data/stand-camaras",
    )
    vi = sub.add_parser("video", help="videos de la demo de cámaras (recorridos, 4 versiones)")
    vi.add_argument(
        "raiz", help="carpeta con las variantes video-* (render_tanda --tipo-pose recorrido)"
    )
    vi.add_argument("--salida", default=str(ROOT / "data" / "stand-video"))
    vi.add_argument("--red", default=None, help="checkpoint de la red de alineación (v3)")
    vi.add_argument("--solo", nargs="+", default=None, help="rehacer solo estas variantes")
    sc = sub.add_parser("sin-calibrar", help="E0: cámaras distintas suponiendo que son la ELP")
    sc.add_argument("raiz", help="carpeta con las variantes video-*")
    sc.add_argument("--salida", default=str(ROOT / "outputs" / "gtsynth" / "e0.json"))
    sc.add_argument("--variantes", nargs="+", default=None)
    sc.add_argument("--cada", type=int, default=6, help="un cuadro de cada N")
    sc.add_argument("--escenas", nargs="+", default=None, help="solo estas escenas")
    mo = sub.add_parser("mosaico", help="las 6 cámaras de la expo a la vez (desde los videos)")
    mo.add_argument("carpeta", help="carpeta de `gtsynth video` (data/stand-video)")
    mo.add_argument("--e0", default=str(ROOT / "outputs" / "gtsynth" / "e0.json"))
    de = sub.add_parser("deteccion", help="recall y precisión de detectores contra el GT")
    de.add_argument("raiz")
    de.add_argument("--modelos", nargs="+", required=True, help="nombres en weights/ o rutas .pt")
    de.add_argument("--variante", default="elp")
    de.add_argument("--particion", nargs="+", default=["val"], choices=["train", "val", "test"])
    de.add_argument("--csv", required=True)
    de.add_argument("--max", type=int, default=0, help="como máximo N vistas, repartidas")
    de.add_argument("--imgsz", type=int, default=960)
    de.add_argument("--conf", type=float, default=0.25)
    de.add_argument("--iou", type=float, default=0.4)
    de.add_argument("--min-px", type=int, default=400)
    de.add_argument("--dispositivo", default="cpu")
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
            "evaluar": cmd_evaluar, "camaras": cmd_camaras, "deteccion": cmd_deteccion}  # fmt: skip
    if args.cmd == "camaras-ver":
        from visionx.app.stand_camaras import carpeta_por_defecto, mostrar

        mostrar(args.carpeta or carpeta_por_defecto())
        return 0
    if args.cmd == "video":
        from visionx.gtsynth.video import preparar as preparar_video

        solo = tuple(args.solo) if args.solo else None
        preparar_video(args.raiz, args.salida, args.red, solo=solo)
        return 0
    if args.cmd == "mosaico":
        from visionx.gtsynth.mosaico import armar_todos

        for path in armar_todos(args.carpeta, args.e0):
            print(path)
        return 0
    if args.cmd == "sin-calibrar":
        from visionx.gtsynth.sin_calibrar import correr_todo

        correr_todo(args.raiz, args.salida, args.variantes, cada=args.cada, escenas=args.escenas)
        return 0
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
