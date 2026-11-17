"""Descargar modelos OCR explícitamente o evaluar segmentos M10 en CPU, sin dispositivos."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from visionx.reading import RapidReader, ReadingSettings
from visionx.reading.evaluate import evaluate
from visionx.reading.rapid import download_models
from visionx.shared.config import load_config
from visionx.shared.errors import EstimatorError, SourceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("descargar", help="bajar/verificar PP-OCRv5 latino, sin inferencia")
    download.add_argument("--config")
    ev = sub.add_parser("evaluar", help="un frame por segmento: CER y tiempos; sin parlantes")
    ev.add_argument("--fuente", action="append", required=True, type=Path)
    ev.add_argument("--csv", required=True, type=Path)
    ev.add_argument("--config")
    ev.add_argument("--imagen", choices=("original", "rectified", "ambas"))
    ev.add_argument("--crop", choices=("none", "center", "text", "object"))
    ev.add_argument(
        "--calentar", action="store_true", help="OCR previo sin medir sobre primer frame"
    )
    ev.add_argument("--ventana", type=float, help="s de selección desde inicio del segmento")
    ev.add_argument("--tts", action="store_true", help="sintetizar primer bloque SIN reproducir")
    ev.add_argument(
        "--permitir-incompletos",
        action="store_true",
        help="piloto explícito con frames disponibles; registrar omisiones en JSON",
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        settings = ReadingSettings.from_config(cfg)
        if args.command == "descargar":
            for name, digest in download_models(settings).items():
                print(f"{name}: {digest}")
            return 0
        if args.ventana is not None:
            settings = replace(settings, window_s=args.ventana)
        if args.crop is not None:
            settings = replace(settings, crop=args.crop)
        route = args.imagen or settings.image_source
        routes = ("original", "rectified") if route == "ambas" else (route,)
        synth = None
        if args.tts:
            from visionx.voice.engines import PiperSynth, VoiceSettings

            synth = PiperSynth(VoiceSettings.from_config(cfg))
        evaluate(
            args.fuente,
            args.csv,
            RapidReader(settings),
            settings,
            cfg,
            routes=routes,
            synth=synth,
            allow_incomplete=args.permitir_incompletos,
            warmup=args.calentar,
        )
        return 0
    except (OSError, ValueError, EstimatorError, SourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
