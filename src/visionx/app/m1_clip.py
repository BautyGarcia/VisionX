"""Complemento de M1 sobre clip real: YOLO+B0, sin reproducir audio ni usar dispositivos.

Se informan presencia/pérdida observadas; no reemplazan los ocho casos sintéticos ni
una anotación humana del clip. Tiempo de video lógico y coste de pared separados.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from visionx.app.controller import Controller
from visionx.app.m1 import write_results
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes.common import ModeSettings
from visionx.modes.distance import DistanceHandler
from visionx.modes.search import SearchHandler
from visionx.modes.simulate import simulate
from visionx.shared.clock import ManualClock
from visionx.shared.config import load_config
from visionx.shared.contracts import Mode
from visionx.shared.errors import InvalidInput
from visionx.voice.commands import CommandParser


def clip_inputs(args, clock, cfg):
    from visionx.app.__main__ import build_depth
    from visionx.calibration import StereoCalibration
    from visionx.camera import RectifiedSource, ReplaySource
    from visionx.detection import DetectorSettings, YoloDetector
    from visionx.shared.config import ROOT

    path = Path(cfg["camera"]["calibration"])
    calibration = StereoCalibration.load(path if path.is_absolute() else ROOT / path)
    source = RectifiedSource(ReplaySource(args.clip, clock=clock), calibration)
    detector = YoloDetector(DetectorSettings.from_config(cfg))
    depth, spatial, _ = build_depth(SimpleNamespace(profundidad="B0"), cfg, source.geometry, None)
    return source, detector, depth, spatial


def run_clip(args, output: Path) -> dict:
    if args.cada < 1:
        raise InvalidInput("--cada debe ser >= 1")
    from visionx.app.__main__ import load_script

    cfg = load_config(args.config)
    clock = ManualClock(1_000_000_000)
    wall = time.perf_counter_ns()
    source, detector, depth, spatial = clip_inputs(args, clock, cfg)
    catalog = ClassCatalog(COCO_CLASSES)
    settings = ModeSettings.from_config(cfg)
    controllers = []
    rows = []

    def make_controller(latest, sink, on_event):
        controller = Controller(
            CommandParser(catalog),
            sink,
            latest_view=latest,
            handlers={
                Mode.SEARCH: SearchHandler(
                    source.geometry, catalog.spoken, settings, on_event=on_event
                ),
                Mode.DISTANCE: DistanceHandler(
                    source.geometry, catalog.spoken, settings, on_event=on_event
                ),
            },
            deadlines_s=(cfg.get("audio") or {}).get("deadlines_s"),
            clock=clock,
            on_event=on_event,
        )
        controllers.append(controller)
        return controller

    def on_frame(t_s, frame, view, sink):
        if frame is not None and not rows:
            rows.append(
                {
                    "caso": "clip",
                    "repeticion": 1,
                    "evento": "primer-frame",
                    "logical_ns": clock.now_ns(),
                    "wall_ns": time.perf_counter_ns() - wall,
                    "frame_id": frame.frame_id,
                }
            )

    try:
        orders = load_script(args.ordenes) if args.ordenes else [(0, "buscá la taza")]
        result = simulate(
            source,
            detector,
            make_controller,
            clock,
            orders,
            depth=depth,
            spatial=spatial,
            every=args.cada,
            tail_s=3,
            on_frame=on_frame,
        )
    finally:
        for controller in controllers:
            controller.close()
        source.close()
        detector.close()
        depth.close()
    for t_s, kind, data in result.events:
        rows.append(
            {
                "caso": "clip",
                "repeticion": 1,
                "evento": kind,
                "logical_ns": int(t_s * 1e9),
                "wall_ns": None,
                **data,
            }
        )
    for spoken in result.spoken:
        rows.append(
            {
                "caso": "clip",
                "repeticion": 1,
                "evento": "audio-solicitado",
                "logical_ns": int(spoken.t_s * 1e9),
                "wall_ns": None,
                **asdict(spoken),
            }
        )
    events = {kind for _, kind, _ in result.events}
    criteria = {
        "sin_errores_modelos": "PASA" if not result.errors else "FALLA",
        "presente": "PASA" if "buscar-adquirido" in events else "FALLA",
        "perdido": "PASA" if "buscar-perdido" in events else "NO OBSERVADO",
    }
    # No otorgar latencia de reproducción a RecordingSink: solo registra solicitudes.
    summary = {
        "tipo": "clip-real-YOLO-B0-sin-audio",
        "repeticiones": 1,
        "fuente": str(args.clip),
        "ordenes": orders,
        "cada": args.cada,
        "configuracion": cfg,
        "detector": detector.model_id,
        "depth": depth.model_id,
        "frames": result.frames,
        "detecciones": result.detections,
        "mediciones": result.measurements,
        "duracion_video_s": result.duration_s,
        "wall_ms": (time.perf_counter_ns() - wall) / 1e6,
        "errores": result.errors,
        "criterios": criteria,
        "metricas": {},
        "corridas": [],
        "resultado": "FALLA" if "FALLA" in criteria.values() else "PASA",
        "limites": "Tiempo de video; latencias de inferencia emuladas en 0. "
        "Audio solicitado, no reproducido. Sin GT humano ni exactitud del detector validada.",
        "hashes_entrada": {},
    }
    for path in (
        Path(args.clip) / "manifest.yaml",
        Path(args.clip) / "frames.jsonl",
        Path(cfg["camera"]["calibration"]),
    ):
        if path.is_file():
            summary["hashes_entrada"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_results(summary, rows, output)
    print(
        f"Clip aparte: {criteria}; {result.frames} frames; {summary['wall_ms'] / 1000:.1f} s pared"
    )
    return summary
