"""Planillas y evaluación offline M7/M8/M10; pesos locales, dispositivos solo en voz grabar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    metrics = parser.add_subparsers(dest="metrica", required=True)
    for metric in ("m7", "m8", "voz", "ocr", "describir"):
        p = metrics.add_parser(metric)
        commands = p.add_subparsers(dest="accion", required=True)
        template = commands.add_parser("plantilla", help="crear CSV y guía de grabación")
        template.add_argument("--salida", type=Path, required=True)
        ev = commands.add_parser("evaluar", help="guardar detalle.csv y resumen.json")
        ev.add_argument("--planilla", type=Path, required=True)
        ev.add_argument("--salida", type=Path, required=True)
        ev.add_argument("--config", help="override YAML explícito")
        if metric in ("m7", "describir"):
            ev.add_argument(
                "--metodos", nargs="+", default=["B0"], help="B0–B5, fusion o depth.estimadores"
            )
            ev.add_argument("--ventana", type=float, default=0.4, help="ancho total en s")
            ev.add_argument("--frames", type=int, default=3)
        if metric == "m8":
            ev.add_argument("--traza", type=Path, help="una traza para todas las filas")
        if metric == "ocr":
            ev.add_argument("--imagen", choices=("original", "rectified"), default="original")
        if metric == "voz":
            synth = commands.add_parser("sintetizar", help="Piper -> WAV, sin sonido ni micrófono")
            synth.add_argument("--planilla", type=Path, required=True)
            synth.add_argument("--config")
            commands.add_parser("dispositivos", help="listar; no abre micrófono")
            rec = commands.add_parser("grabar", help="Enter graba, r reintenta, s saltea, q sale")
            rec.add_argument("--planilla", type=Path, required=True)
            rec.add_argument(
                "--dispositivo",
                help="índice o subcadena sin acentos; omitir usa entrada por defecto",
            )
            rec.add_argument(
                "--segundos", type=float, default=4, help="duración por frase (default: 4 s)"
            )
            rec.add_argument("--cuenta", type=int, default=3)
            rec.add_argument("--revisar-saltadas", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.accion == "plantilla":
            from .templates import plantilla

            print(plantilla(args.metrica, args.salida))
            return 0
        if args.accion == "dispositivos":
            from .audio_devices import input_devices

            for device in input_devices():
                print(f"{device['index']}: {device['name']} [{device['api']}]")
            return 0
        if args.accion == "grabar":
            from .voice import record

            device = args.dispositivo
            if device is not None and device.isdigit():
                device = int(device)
            record(
                args.planilla,
                seconds=args.segundos,
                device=device,
                countdown=args.cuenta,
                revisit=args.revisar_saltadas,
            )
            return 0
        from visionx.shared.config import load_config

        cfg = load_config(args.config)
        if args.accion == "sintetizar":
            from .voice import synthesize_session

            synthesize_session(args.planilla, cfg)
            return 0
        if args.metrica == "m7":
            from .objects import evaluate

            evaluate(
                args.planilla,
                args.salida,
                cfg,
                args.metodos,
                window_s=args.ventana,
                samples=args.frames,
            )
        elif args.metrica == "m8":
            from .usage import evaluate

            evaluate(args.planilla, args.salida, trace=args.traza)
        elif args.metrica == "voz":
            from .voice import evaluate

            evaluate(args.planilla, args.salida, cfg)
        elif args.metrica == "ocr":
            from .text import evaluate

            evaluate(args.planilla, args.salida, cfg, image=args.imagen)
        else:
            from .scenes import evaluate

            if len(args.metodos) != 1:
                raise ValueError("describir usa un método por corrida")
            evaluate(
                args.planilla,
                args.salida,
                cfg,
                method=args.metodos[0],
                window_s=args.ventana,
                samples=args.frames,
            )
        return 0
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
