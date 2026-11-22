"""CLI del GT-sintético-ELP.

uv run python -m visionx.gtsynth resumen  data/gt-sintetico-elp [--variante elp]
uv run python -m visionx.gtsynth objetos  data/gt-sintetico-elp --csv outputs/gtsynth/objetos.csv
uv run python -m visionx.gtsynth verificar data/gt-sintetico-elp [--max 20]
uv run --extra ml python -m visionx.gtsynth evaluar data/gt-sintetico-elp --particion val \
    --variante elp --profundidad crestereo-preciso --csv outputs/gtsynth/val-elp.csv
    [--alinear clasico|red|clasico+red --red mejor.pt] [--vistas-por-camara 8]
    [--distancia-conocida [corrimiento|escala]]

Variantes de `evaluar`: elp, hermana-0.5, hermana-1 (sobre las vistas elp) y distinta.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
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
    lista = list(vistas(args.raiz, carpeta, args.particion))[:: max(1, args.cada)]
    if not lista:
        print(f"no hay vistas {carpeta}/{args.particion}", file=sys.stderr)
        return 1
    if args.geometria == "estimada":
        return _evaluar_estimada(args, cfg, carpeta, lista)
    geometria = ELP_RENDER
    if args.geometria != "elp":  # la cámara de las vistas (recorte nominal)
        from visionx.gtsynth.dataset import VistaNominal

        lista = [VistaNominal(v.carpeta) for v in lista]
        ids = {v.geometria.geometry_id for v in lista}
        if len(ids) != 1:
            print(f"--geometria {args.geometria} pide una sola cámara por variante ({len(ids)})",
                  file=sys.stderr)  # fmt: skip
            return 1
        geometria = lista[0].geometria
        if args.geometria == "elp-supuesta":  # otra cámara, pero el sistema cree que es la ELP
            from visionx.gtsynth.sin_calibrar import geometria_elp

            geometria = geometria_elp(geometria)
    m = Medidor.armar(args.profundidad, cfg, geometria)
    t0 = time.time()

    def progreso(n, v):
        if n % 10 == 0 or n == len(lista):
            print(f"  {n}/{len(lista)} vistas ({time.time() - t0:.0f} s)", flush=True)

    alineador = None
    if args.alinear != "ninguna":
        from visionx.depth.crestereo_adapt.align_apply import Alineador
        from visionx.gtsynth.variantes import distorsion_nominal

        if args.geometria == "elp":
            alineador = Alineador(args.alinear, ELP_RENDER, distorsion_nominal(), args.red)
        else:  # renders pinhole: sin distorsión, con la geometría que cree tener el sistema
            alineador = Alineador(args.alinear, geometria, np.zeros(5), args.red)
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


def _evaluar_estimada(args: argparse.Namespace, cfg: dict, carpeta: str, lista: list) -> int:
    """Por objeto con la cámara que estima el sistema, un recorrido (escena) a la vez.

    Focal de red + movimiento (`--focal-movimiento`, JSON de `gtsynth autocalib`), centro de la
    red (`--red-camara`, CSV de `camara_net evaluar`) y la separación conocida (hoja de datos).
    El resto igual que `--geometria vista`: alineación y distancia conocida a elección.
    """
    import json
    import time

    from visionx.gtsynth.dataset import VistaNominal
    from visionx.gtsynth.evaluar import Medidor, medir, resumen
    from visionx.gtsynth.sin_calibrar import camara_de_la_red, geometria_red, leer_red

    if not (args.focal_movimiento and args.red_camara):
        print("--geometria estimada pide --focal-movimiento y --red-camara", file=sys.stderr)
        return 1
    filas_fm = json.loads(Path(args.focal_movimiento).read_text(encoding="utf-8"))["filas"]
    fm = {(f["variante"], f["escena"]): f["fx_real"] * (1 + f["comb_err_pct"] / 100)
          for f in filas_fm}  # fmt: skip
    red = leer_red(args.red_camara)
    todas: dict[str, list] = {}
    for v in vistas(args.raiz, carpeta, args.particion):
        todas.setdefault(v.escena, []).append(v)
    por_escena: dict[str, list] = {}
    for v in lista:
        por_escena.setdefault(v.escena, []).append(VistaNominal(v.carpeta))
    filas: list[dict] = []
    t0 = time.time()
    for escena in sorted(por_escena):
        vs = sorted(todas[escena], key=lambda v: v.meta["indice"])
        cam = camara_de_la_red(vs, red)
        g = por_escena[escena][0].geometria
        if cam is None or (args.variante, escena) not in fm:
            print(f"  {escena}: sin estimación de la red o del movimiento; se saltea")
            continue
        cam = {**cam, "fx": fm[(args.variante, escena)] * cam["ancho"] / g.width}
        g_est = geometria_red(g, cam, g.baseline_m)
        m = Medidor.armar(args.profundidad, cfg, g_est)
        alineador = None
        if args.alinear != "ninguna":
            from visionx.depth.crestereo_adapt.align_apply import Alineador

            alineador = Alineador(args.alinear, g_est, np.zeros(5), args.red)
        nuevas = medir(m, por_escena[escena], args.variante, args.max, args.min_px, None,
                       alineador, args.vistas_por_camara, args.distancia_conocida)  # fmt: skip
        for f in nuevas:
            f["fx_estimada"] = round(g_est.fx, 2)
            f["fx_real"] = round(g.fx, 2)
        filas += nuevas
        print(f"  {escena}: {len(nuevas)} objetos, focal {100 * (g_est.fx / g.fx - 1):+.2f} % "
              f"({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
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
    ev.add_argument("--cada", type=int, default=1, help="una de cada N vistas (recorridos)")
    ev.add_argument(
        "--geometria",
        choices=["elp", "vista", "elp-supuesta", "estimada"],
        default="elp",
        help="elp: renders de la ELP (por defecto); vista: la cámara de las vistas, recortada a "
        "su tamaño nominal (gt-video, perfiles); elp-supuesta: esas vistas, pero el sistema cree "
        "que la cámara es una ELP (E0, plug and play); estimada: la que estima el sistema con "
        "red + movimiento y la separación conocida, por recorrido",
    )
    ev.add_argument("--focal-movimiento", default=None, help="JSON de `gtsynth autocalib`")
    ev.add_argument("--red-camara", default=None, help="CSV de `camara_net evaluar`")
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
        nargs="?",
        const="corrimiento",
        default=None,
        choices=["corrimiento", "escala"],
        help="el objeto apuntado de la primera vista de cada cámara corrige un corrimiento de "
        "columnas (por defecto, misma cámara) o la escala f·B (cámara distinta o estimada)",
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
    vi.add_argument("--escenas", nargs="+", default=None, help="solo estas escenas")
    sc = sub.add_parser("sin-calibrar", help="E0: cámaras distintas suponiendo que son la ELP")
    sc.add_argument("raiz", help="carpeta con las variantes video-*")
    sc.add_argument("--salida", default=str(ROOT / "outputs" / "gtsynth" / "e0.json"))
    sc.add_argument("--variantes", nargs="+", default=None)
    sc.add_argument("--cada", type=int, default=6, help="un cuadro de cada N")
    sc.add_argument("--escenas", nargs="+", default=None, help="solo estas escenas")
    sc.add_argument("--versiones", nargs="+", default=None, help="solo estas versiones (nombres)")
    sc.add_argument(
        "--red-camara",
        default=None,
        metavar="CARPETA",
        help="E3: CSV de `camara_net evaluar`; agrega las versiones «red …»",
    )
    sc.add_argument(
        "--focal-movimiento",
        default=None,
        metavar="JSON",
        help="E3 con la focal de `gtsynth autocalib` (red + movimiento)",
    )
    sc.add_argument(
        "--cuadros-red",
        type=int,
        default=8,
        help="cuántos cuadros del principio del recorrido mira la red",
    )
    ae = sub.add_parser("autoetiquetas", help="etiquetas red + movimiento para reentrenar la red")
    ae.add_argument("raiz", help="carpeta con las variantes (gt-sintetico-elp)")
    ae.add_argument("--variante", default="hssd-recorridos-sorteada-b")
    ae.add_argument("--red", default=str(ROOT / "outputs" / "camara_net" / "e2-v2" / "mejor.pt"))
    ae.add_argument("--salida", default=str(ROOT / "outputs" / "autoetiquetas" / "b.csv"))
    ae.add_argument("--limite", type=int, default=0, help="solo los primeros N recorridos")
    ac = sub.add_parser("autocalib", help="autocalibración por movimiento contra la cámara real")
    ac.add_argument("raiz", help="carpeta con las variantes video-*")
    ac.add_argument("--salida", default=str(ROOT / "outputs" / "gtsynth" / "autocalib.json"))
    ac.add_argument("--red-camara", default=None, metavar="CARPETA", help="CSV de la red")
    ac.add_argument("--variantes", nargs="+", default=None)
    ac.add_argument("--cuadros", type=int, default=60)
    ac.add_argument("--saltos", type=int, nargs="+", default=[3, 6, 12])
    e3 = sub.add_parser("e3-simulado", help="E3 con una red de entrada simulada (focal ± σ)")
    e3.add_argument("carpeta", help="CSV por cámara de `evaluar --geometria vista`")
    e3.add_argument("--salida", default=str(ROOT / "outputs" / "gtsynth" / "e3-simulado.json"))
    e3.add_argument("--k", type=float, default=1.0, help="factor de la b de la U-Net")
    e3.add_argument("--sorteos", type=int, default=500)
    mo = sub.add_parser("mosaico", help="las 6 cámaras de la expo a la vez (desde los videos)")
    mo.add_argument("carpeta", help="carpeta de `gtsynth video` (data/stand-video)")
    mo.add_argument("--e0", default=str(ROOT / "outputs" / "gtsynth" / "e0.json"))
    di = sub.add_parser("disparidades", help="CREStereo precalculado sobre el GT (para entrenar)")
    di.add_argument("raiz")
    di.add_argument("--variantes", nargs="+", required=True)
    di.add_argument("--salida", default=str(ROOT / "data" / "cache-crestereo"))
    di.add_argument("--limite", type=int, default=0, help="solo N vistas por variante")
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
        escenas = tuple(args.escenas) if args.escenas else None
        preparar_video(args.raiz, args.salida, args.red, solo=solo, solo_escenas=escenas)
        return 0
    if args.cmd == "disparidades":
        from visionx.gtsynth.disparidades import calcular

        print(calcular(args.raiz, args.variantes, args.salida, limite=args.limite))
        return 0
    if args.cmd == "mosaico":
        from visionx.gtsynth.mosaico import armar_todos

        for path in armar_todos(args.carpeta, args.e0):
            print(path)
        return 0
    if args.cmd == "sin-calibrar":
        from visionx.gtsynth.sin_calibrar import correr_todo

        correr_todo(args.raiz, args.salida, args.variantes, cada=args.cada,
                    escenas=args.escenas, versiones=args.versiones, red=args.red_camara,
                    cuadros_red=args.cuadros_red, focal_mov=args.focal_movimiento)  # fmt: skip
        return 0
    if args.cmd == "autoetiquetas":
        from visionx.gtsynth.autoetiquetas import correr as autoetiquetas

        q = autoetiquetas(args.raiz, args.variante, args.red, args.salida,
                          limite=args.limite)  # fmt: skip
        print(json.dumps(q, ensure_ascii=False))
        return 0
    if args.cmd == "autocalib":
        from visionx.gtsynth.autocalib import correr_todo as autocalib

        r = autocalib(args.raiz, args.salida, args.red_camara, args.variantes, args.cuadros,
                      tuple(args.saltos))  # fmt: skip
        print(json.dumps(r["resumen"], ensure_ascii=False))
        return 0
    if args.cmd == "e3-simulado":
        from visionx.gtsynth.e3_simulado import correr

        r = correr(args.carpeta, args.salida, k=args.k, sorteos=args.sorteos)
        for s, d in r["por_sigma"].items():
            print(f"focal ±{s}: {d['media']}")
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
