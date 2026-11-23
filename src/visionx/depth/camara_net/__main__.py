"""Línea de comandos de la red que estima la cámara (E2).

    uv run --extra ml python -m visionx.depth.camara_net entrenar --salida outputs/camara_net/e2-v1
    uv run --extra ml python -m visionx.depth.camara_net evaluar outputs/camara_net/e2-v1/mejor.pt

`evaluar` mide, sin entrenar con nada de eso:
- val de `camara-sorteada` y `hssd-camara-sorteada`, tal cual (pinhole) y con lente sorteada;
- los recorridos de `data/gt-video` (las 6 cámaras de la expo y 3 genéricas), tal cual, por
  cuadro, de a 8 cuadros y con todos los cuadros juntos.
No usa nunca la partición test.

Parte real de D (video de una cámara real, etiquetado por la geometría):

    uv run python -m visionx.depth.camara_net desde-grabacion recordings/<grabación> \
        --calibracion configs/calib/elp-2026-11-11-corregida.yaml
    uv run --extra ml python -m visionx.depth.camara_net etiquetar-real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionx.shared.config import ROOT


def _entrenar(a: argparse.Namespace) -> None:
    from visionx.depth.camara_net.entrenar import Params, entrenar

    p = Params(raiz=a.raiz, epocas=a.epocas, lote=a.lote, trabajadores=a.trabajadores,
               semilla=a.semilla, max_items=a.max_items,
               zoom=tuple(a.zoom) if a.zoom else None, lr=a.lr, lr_tronco=a.lr_tronco,
               calentamiento=a.calentamiento, etiquetas=a.etiquetas, sigma_max=a.sigma_max,
               etiqueta_real=a.etiqueta_real, desde=a.desde,
               variantes=tuple(a.variantes))  # fmt: skip
    print(json.dumps(entrenar(p, a.salida, a.device, a.continuar), ensure_ascii=False))


def _evaluar(a: argparse.Namespace) -> None:
    from visionx.depth.camara_net.datos import Aumento
    from visionx.depth.camara_net.entrenar import cargar, evaluar_conjunto, items_de

    modelo, centro, device = cargar(a.modelo, a.device)
    salida = Path(a.salida or Path(a.modelo).parent / "eval")
    salida.mkdir(parents=True, exist_ok=True)
    res = {}
    val = items_de(a.raiz, ("camara-sorteada", "hssd-camara-sorteada"), "val")
    for nombre, aum in (("val-pinhole", Aumento.ninguno()), ("val-lente", Aumento())):
        print(f"{nombre}: {len(val)} vistas", flush=True)
        res[nombre] = evaluar_conjunto(modelo, val, aum, device, centro, a.trabajadores,
                                       salida / f"{nombre}.csv")  # fmt: skip
    video = Path(a.video)
    if video.is_dir():
        for v in sorted(x.name for x in video.iterdir() if x.is_dir()):
            items = items_de(video, (v,), None)
            print(f"{v}: {len(items)} cuadros", flush=True)
            res[v] = evaluar_conjunto(modelo, items, Aumento.ninguno(), device, centro,
                                      a.trabajadores, salida / f"{v}.csv")  # fmt: skip
    (salida / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    print(f"listo: {salida / 'resumen.json'}")


def _desde_grabacion(a: argparse.Namespace) -> None:
    from visionx.depth.camara_net.grabacion import convertir

    for g in a.grabaciones:
        r = convertir(g, a.raiz, a.variante, a.cuadros, a.paso, a.ancho, a.calibracion,
                      a.sin_lente)  # fmt: skip
        print(json.dumps(r, ensure_ascii=False), flush=True)


def _etiquetar_real(a: argparse.Namespace) -> None:
    # Las escenas reales no están en ninguna partición de la simulación: particion=None.
    from visionx.gtsynth.autoetiquetas import correr

    q = correr(a.raiz, a.variante, a.red, a.salida, particion=None)
    print(json.dumps(q, ensure_ascii=False, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m visionx.depth.camara_net")
    sub = ap.add_subparsers(dest="cmd", required=True)
    en = sub.add_parser("entrenar", help="entrenar la red (Mac, MPS)")
    en.add_argument("--raiz", default=str(ROOT / "data" / "gt-sintetico-elp"))
    en.add_argument("--salida", required=True)
    en.add_argument("--epocas", type=int, default=20)
    en.add_argument("--lote", type=int, default=12)
    en.add_argument("--trabajadores", type=int, default=8)
    en.add_argument("--semilla", type=int, default=0)
    en.add_argument("--max-items", type=int, default=0, help="0 = todos (para pruebas rápidas)")
    en.add_argument("--zoom", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="recortes con zoom (p. ej. 0.55 1.2); por defecto sin zoom")  # fmt: skip
    en.add_argument("--variantes", nargs="+", default=["camara-sorteada", "hssd-camara-sorteada"],
                    help="variantes de una cámara por vista para entrenar (y validar)")  # fmt: skip
    en.add_argument("--lr", type=float, default=3e-4, help="la cabeza")
    en.add_argument("--lr-tronco", type=float, default=1e-4)
    en.add_argument("--calentamiento", type=int, default=300, help="pasos")
    en.add_argument("--etiquetas", default=None,
                    help="v3-auto: CSV de `gtsynth autoetiquetas` (b-vistas.csv)")  # fmt: skip
    en.add_argument("--sigma-max", type=float, default=0.03, help="duda máxima de la etiqueta")
    en.add_argument("--etiqueta-real", action="store_true",
                    help="control: la focal real en los mismos cuadros")  # fmt: skip
    en.add_argument("--desde", default=None, help="checkpoint de partida (p. ej. la v2)")
    en.add_argument("--device", default="auto")
    en.add_argument("--continuar", action="store_true", help="seguir desde ultimo.pt")
    en.set_defaults(fn=_entrenar)
    ev = sub.add_parser("evaluar", help="E2: val y cámaras de la expo")
    ev.add_argument("modelo")
    ev.add_argument("--raiz", default=str(ROOT / "data" / "gt-sintetico-elp"))
    ev.add_argument("--video", default=str(ROOT / "data" / "gt-video"))
    ev.add_argument("--salida", default=None)
    ev.add_argument("--trabajadores", type=int, default=6)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(fn=_evaluar)
    real = ROOT / "data" / "real"
    dg = sub.add_parser("desde-grabacion", help="D real: grabación cruda → recorridos (vistas)")
    dg.add_argument("grabaciones", nargs="+", help="carpetas de `camera grabar` (par crudo)")
    dg.add_argument("--raiz", default=str(real))
    dg.add_argument("--variante", default="elp-recorridos-real")
    dg.add_argument("--cuadros", type=int, default=60, help="cuadros por recorrido (como la app)")
    dg.add_argument("--paso", type=int, default=2, help="uno de cada N pares grabados")
    dg.add_argument("--ancho", type=int, default=960)
    dg.add_argument("--calibracion", default=None,
                    help="calibración de tablero: solo para el control (fx_real)")  # fmt: skip
    dg.add_argument("--sin-lente", action="store_true",
                    help="control: sin la lente del tablero (otra --variante)")  # fmt: skip
    dg.set_defaults(fn=_desde_grabacion)
    er = sub.add_parser("etiquetar-real", help="D real: red + movimiento sobre esos recorridos")
    er.add_argument("--raiz", default=str(real))
    er.add_argument("--variante", default="elp-recorridos-real")
    er.add_argument("--red", default=str(ROOT / "outputs" / "camara_net" / "e2-v2" / "mejor.pt"))
    er.add_argument("--salida", default=str(ROOT / "outputs" / "autoetiquetas" / "real.csv"))
    er.set_defaults(fn=_etiquetar_real)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
