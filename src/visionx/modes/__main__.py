"""Modos buscar y distancia (T09) sobre una grabación, con tiempo grabado y latencias emuladas.

    uv run python -m visionx.modes simular --fuente recordings/dev-1114-buscar-a \\
        --orden "1:buscá la botella" [--orden "20:a qué distancia está"] \\
        [--profundidad B0|B1|B2|ninguna] [--latencia-detector 0.2] [--latencia-profundidad 1.0] \\
        [--cada 1] [--traza X.jsonl] [--config X]

    uv run python -m visionx.modes simular --fuente sintetica --orden "0:buscá la taza"

Imprime lo que el sistema pediría decir, con el segundo del clip, y los eventos de los modos
(adquirido, perdido, elección, respuesta de distancia). Ver modes/simulate.py: el tiempo es el
grabado, no el de la CPU, así que el resultado no depende de la velocidad de esta máquina.
Con una grabación hace falta `uv sync --extra deteccion`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

from visionx.app.controller import Controller
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes import build_handlers
from visionx.modes.simulate import simulate
from visionx.shared.clock import ManualClock
from visionx.shared.config import ROOT, load_config
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError
from visionx.voice.commands import CommandParser


def parse_order(text: str) -> tuple[float, str]:
    """ "12.5:buscá la taza" -> (12.5, "buscá la taza")."""
    t, sep, utterance = text.partition(":")
    if not sep or not utterance.strip():
        raise InvalidInput(f"orden {text!r}: se esperaba SEGUNDO:frase")
    return float(t), utterance.strip()


def load_orders(args: argparse.Namespace) -> list[tuple[float, str]]:
    orders = [parse_order(o) for o in args.orden or []]
    if args.ordenes:
        for line in Path(args.ordenes).read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                orders.append((float(data["t_s"]), str(data["texto"])))
    if not orders:
        raise InvalidInput("hace falta al menos una --orden o --ordenes")
    return sorted(orders)


def build(args: argparse.Namespace, cfg: dict, clock: ManualClock):
    """(fuente, geometría del frame, detector, profundidad, ubicación, tick_s)."""
    from visionx.app.__main__ import build_depth  # mismas piezas que `app ejecutar`

    if args.fuente == "sintetica":
        from visionx.app.fakes import ScriptedDetector, SyntheticStereoSource

        source = SyntheticStereoSource(fps=None, n_frames=args.frames or 300, clock=clock)
        detector = ScriptedDetector(source)
        ns = SimpleNamespace(profundidad="fake", retraso_profundidad=0.0)
        depth, spatial, _ = build_depth(ns, cfg, source.geometry, source)
        return source, source.geometry, detector, depth, spatial, 1 / 30
    from visionx.calibration import StereoCalibration
    from visionx.camera import RectifiedSource, ReplaySource
    from visionx.detection import DetectorSettings, YoloDetector

    calib_path = Path((cfg.get("camera") or {}).get("calibration") or "")
    calibration = StereoCalibration.load(
        calib_path if calib_path.is_absolute() else ROOT / calib_path
    )
    source = RectifiedSource(ReplaySource(args.fuente, clock=clock), calibration)
    detector = YoloDetector(DetectorSettings.from_config(cfg))
    depth = spatial = None
    if args.profundidad != "ninguna":
        ns = SimpleNamespace(profundidad=args.profundidad, retraso_profundidad=0.0)
        depth, spatial, _ = build_depth(ns, cfg, source.geometry, None)
    return source, source.geometry, detector, depth, spatial, None


def cmd_simular(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    clock = ManualClock(1_000_000_000)
    source, geometry, detector, depth, spatial, tick_s = build(args, cfg, clock)
    catalog = ClassCatalog(COCO_CLASSES)
    deadlines = {
        str(k): float(v) for k, v in ((cfg.get("audio") or {}).get("deadlines_s") or {}).items()
    }

    def make_controller(latest_view, sink, on_event):
        return Controller(
            CommandParser(catalog),
            sink,
            latest_view=latest_view,
            handlers=build_handlers(cfg, geometry, catalog.spoken, on_event),
            deadlines_s=deadlines,
            clock=clock,
            on_event=on_event,
        )

    result = simulate(
        source,
        detector,
        make_controller,
        clock,
        load_orders(args),
        depth=depth,
        spatial=spatial,
        det_latency_s=args.latencia_detector,
        depth_latency_s=args.latencia_profundidad,
        every=args.cada,
        tick_s=tick_s,
        tail_s=args.cola,
    )
    print(f"fuente: {args.fuente}  ({result.frames} frames, {result.duration_s:.1f} s grabados)")
    print(f"detector: {detector.model_id} ({result.detections} frames procesados, "
          f"latencia emulada {args.latencia_detector} s)")  # fmt: skip
    if depth is not None:
        print(f"profundidad: {depth.model_id} ({result.measurements} frames, "
              f"latencia emulada {args.latencia_profundidad} s)")  # fmt: skip
    print("\nlo que diría:")
    for s in result.spoken:
        print(f"  {s.t_s:7.2f} s  [#{s.request_id}] {s.text}")
    counts = Counter(
        kind for _, kind, _ in result.events if kind.startswith(("buscar", "distancia"))
    )
    print("\neventos:", dict(counts) or "-")
    for t, kind, data in result.events:
        if kind.startswith("distancia-respuesta"):
            print(f"  {t:7.2f} s  {data['clase']}: z={data['z_m']} motivo={data['motivo']} "
                  f"ángulo={data['angulo_h']} frame={data['frame_id']}")  # fmt: skip
    for e in result.errors:
        print("error:", e)
    if args.traza:
        with open(args.traza, "w", encoding="utf-8", newline="\n") as f:
            for s in result.spoken:
                f.write(json.dumps({"t_s": s.t_s, "tipo": "dice", "request_id": s.request_id,
                                    "texto": s.text}, ensure_ascii=False) + "\n")  # fmt: skip
            for t, kind, data in result.events:
                f.write(json.dumps({"t_s": t, "tipo": kind, **data}, ensure_ascii=False,
                                   default=str) + "\n")  # fmt: skip
        print(f"traza: {args.traza}")
    summary = {"frames": result.frames, "dichos": len(result.spoken), "eventos": dict(counts)}
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.modes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("simular", help="buscar/distancia sobre una grabación, tiempo grabado")
    p.add_argument("--fuente", required=True, help="carpeta de grabación o 'sintetica'")
    p.add_argument("--orden", action="append", help="SEGUNDO:frase (se puede repetir)")
    p.add_argument("--ordenes", help="JSONL con {t_s, texto}")
    p.add_argument("--profundidad", choices=["B0", "B1", "B2", "ninguna"], default="B0")
    p.add_argument("--latencia-detector", type=float, default=0.0, help="s emulados")
    p.add_argument("--latencia-profundidad", type=float, default=0.0, help="s emulados")
    p.add_argument("--cada", type=int, default=1, help="el detector mira 1 de cada N frames")
    p.add_argument("--frames", type=int, default=0, help="solo sintética: cuántos frames")
    p.add_argument("--cola", type=float, default=3.0, help="s simulados tras el último frame")
    p.add_argument("--traza", help="JSONL con lo dicho y los eventos")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_simular)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (EstimatorError, SourceError, InvalidInput) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
