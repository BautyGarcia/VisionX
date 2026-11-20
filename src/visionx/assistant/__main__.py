"""CLI del asistente.

    uv run --extra ia python -m visionx.assistant banco-incertidumbre --seco     # ver los casos
    uv run --extra ia python -m visionx.assistant banco-incertidumbre \\
        --csv outputs/vlm/banco-incertidumbre.csv                                 # ~0,4 USD

Usa la key de VisionX (~/.visionx/anthropic_api_key), nunca ANTHROPIC_API_KEY, y anota el gasto
en el ledger de `assistant` (corta en `budget_usd`).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

IMAGEN = "recordings/dev-1114-distancias/frames/000000_L.jpg"


def cmd_banco(args: argparse.Namespace) -> int:
    import cv2

    from visionx.assistant.banco import casos, evaluar
    from visionx.assistant.core import Assistant, AssistantSettings
    from visionx.shared.config import ROOT, load_config

    lista = casos(args.semilla)[: args.casos]
    if args.seco:
        for i, c in enumerate(lista):
            print(f"{i:2} [{c.grupo}] {c.pedido}")
            for o in c.objetos:
                print(f"     - {o}")
        return 0
    imagen = cv2.imread(str(ROOT / args.imagen))
    if imagen is None:
        print(f"no está la imagen {args.imagen}", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    # Sin memoria entre casos: cada uno es una pregunta suelta.
    s = dataclasses.replace(AssistantSettings.from_config(cfg), memory_s=0.0)
    a = Assistant(s)
    if not a.available:
        print("asistente no disponible (key, extra ia o tope de gasto)", file=sys.stderr)
        return 1
    filas = []
    for i, c in enumerate(lista):
        job = a.submit(i + 1, c.kind, imagen, request=c.pedido, objects=c.objetos)
        job.done.wait(s.timeout_s + 10)
        dichas = [t for _, t in job.sentences]
        fila = {"caso": i, "grupo": c.grupo, "pedido": c.pedido, "objetos": " | ".join(c.objetos),
                "crudo": job.raw.strip(), "dicho": " ".join(dichas),
                "descartadas": len(job.dropped), "error": job.error or "",
                "primer_texto_s": job.first_s, "total_s": job.total_s, "usd": job.usd,
                **evaluar(c, job.raw, dichas, s.clause_chars)}  # fmt: skip
        filas.append(fila)
        marca = "ok" if fila["fiel_crudo"] and fila["nivel_ok"] and fila["camara_ok"] else "MAL"
        print(f"{i:2} [{c.grupo:10}] {marca}  {fila['crudo'][:110]}", flush=True)
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        for fila in filas:
            w.writerow({k: (" ; ".join(v) if isinstance(v, list) else v) for k, v in fila.items()})
    resumen = resumir(filas)
    path.with_suffix(".json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                                         encoding="utf-8")  # fmt: skip
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    return 0


def resumir(filas: list[dict]) -> dict:
    def tasa(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(map(bool, xs)) / len(xs), 3) if xs else None

    por_grupo = defaultdict(list)
    for f in filas:
        por_grupo[f["grupo"]].append(f)
    out = {
        "casos": len(filas),
        "errores": sum(1 for f in filas if f["error"]),
        "fidelidad_cruda": tasa(f["fiel_crudo"] for f in filas),
        "fidelidad_dicha": tasa(f["fiel_dicho"] for f in filas),
        "nivel_respetado": tasa(f["nivel_ok"] for f in filas),
        "camara_dicha": tasa(f["camara_ok"] for f in filas if f["grupo"] == "camara"),
        "metros_en_el_primer_tramo": tasa(f["metros_primero"] for f in filas),
        "tramos_descartados": sum(f["descartadas"] for f in filas),
        "primer_texto_s_mediana": _mediana(f["primer_texto_s"] for f in filas),
        "total_s_mediana": _mediana(f["total_s"] for f in filas),
        "usd": round(sum(f["usd"] or 0 for f in filas), 4),
        "por_grupo": {g: {"n": len(fs), "fidelidad_cruda": tasa(f["fiel_crudo"] for f in fs),
                          "nivel_respetado": tasa(f["nivel_ok"] for f in fs)}
                      for g, fs in por_grupo.items()},
    }  # fmt: skip
    return out


def _mediana(xs):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 2) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m visionx.assistant", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("banco-incertidumbre", help="60 casos: niveles de certeza y números")
    p.add_argument("--imagen", default=IMAGEN)
    p.add_argument("--casos", type=int, default=60)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--csv", default="outputs/vlm/banco-incertidumbre.csv")
    p.add_argument("--config")
    p.add_argument("--seco", action="store_true", help="solo mostrar los casos, sin Claude")
    p.set_defaults(fn=cmd_banco)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
