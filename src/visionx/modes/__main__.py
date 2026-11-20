"""Modos buscar, distancia y describir sobre clips, con tiempo y latencias emulados.

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
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

from visionx.app.camera_status import CameraStatus, CameraStatusSettings
from visionx.app.controller import Controller
from visionx.app.proximity import ProximityAlert, ProximitySettings
from visionx.assistant import build_assistant
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
    catalog = ClassCatalog(getattr(detector, "class_names", None) or COCO_CLASSES)
    deadlines = {
        str(k): float(v) for k, v in ((cfg.get("audio") or {}).get("deadlines_s") or {}).items()
    }

    assistant = build_assistant(cfg)

    def make_controller(latest_view, sink, on_event):
        return Controller(
            CommandParser(catalog, questions=assistant is not None),
            sink,
            latest_view=latest_view,
            handlers=build_handlers(cfg, geometry, catalog.spoken, on_event, assistant=assistant),
            deadlines_s=deadlines,
            clock=clock,
            on_event=on_event,
            watchers=[
                CameraStatus(CameraStatusSettings.from_config(cfg)),
                ProximityAlert(ProximitySettings.from_config(cfg), geometry=geometry),
            ],
        )

    # Capturas del frame que respalda lo dicho, no del frame actual de la cámara: detector
    # y profundidad pueden llegar más tarde. Sirve para revisar cualquier modo sin audio.
    spoken_count = 0

    def capture(t_s, frame, view, sink):
        nonlocal spoken_count
        if not args.capturas or len(sink.spoken) == spoken_count:
            return
        import cv2

        folder = Path(args.capturas)
        folder.mkdir(parents=True, exist_ok=True)
        spoken_count = len(sink.spoken)
        for kind, joint in (("seen", view.seen), ("measured", view.measured)):
            if joint is None:
                continue
            canvas = joint.frame.left.copy()
            for i, (box, label) in enumerate(
                zip(joint.detection.boxes, joint.detection.classes, strict=True)
            ):
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    canvas,
                    f"{i}: {label}",
                    (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            path = folder / f"{t_s:07.3f}-{kind}-{joint.frame.frame_id}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise InvalidInput(f"no se pudo escribir {path}")

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
        on_frame=capture,
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
        kind
        for _, kind, _ in result.events
        if kind.startswith(("buscar", "distancia", "describir"))
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


def cmd_demo(args: argparse.Namespace) -> int:
    """Como `simular`, pero además escribe un MP4 con la imagen anotada, subtítulos y voz."""
    import tempfile

    import cv2

    from visionx.modes.common import ModeSettings
    from visionx.modes.demo import (
        AudioSchedule,
        DemoState,
        H,
        Painter,
        W,
        add_subtitles,
        mix_sonar,
        mux,
        render_frame,
        write_wav,
    )

    cfg = load_config(args.config)
    clock = ManualClock(1_000_000_000)
    source, geometry, detector, depth, spatial, tick_s = build(args, cfg, clock)
    catalog = ClassCatalog(getattr(detector, "class_names", None) or COCO_CLASSES)
    settings = ModeSettings.from_config(cfg)
    deadlines = {
        str(k): float(v) for k, v in ((cfg.get("audio") or {}).get("deadlines_s") or {}).items()
    }
    holder: dict = {}
    assistant = build_assistant(cfg)
    print(f"asistente: {assistant.s.model}" if assistant else "asistente: apagado")

    def make_controller(latest_view, sink, on_event):
        holder["handlers"] = build_handlers(
            cfg, geometry, catalog.spoken, on_event, assistant=assistant
        )
        holder["controller"] = Controller(
            CommandParser(catalog, questions=assistant is not None),
            sink,
            latest_view=latest_view,
            handlers=holder["handlers"],
            deadlines_s=deadlines,
            clock=clock,
            on_event=on_event,
            watchers=[
                CameraStatus(CameraStatusSettings.from_config(cfg)),
                ProximityAlert(ProximitySettings.from_config(cfg), geometry=geometry),
            ],
        )
        return holder["controller"]

    synth = None
    sample_rate = 22050
    if not args.sin_voz:
        from visionx.voice.engines import VoiceSettings, build_synth

        synth_obj = build_synth(VoiceSettings.from_config(cfg))
        synth, sample_rate = synth_obj.synthesize, synth_obj.sample_rate
    schedule = AudioSchedule(synth, sample_rate)
    from visionx.modes.sonar import Sonar, SonarSettings, target_of

    sonar = Sonar(SonarSettings()) if settings.guide_style == "sonar" else None
    sonar_cues: list = []
    painter = Painter()
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="visionx-demo-"))
    writer = cv2.VideoWriter(
        str(tmp / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H)
    )
    info = [f"detector: {detector.model_id.split('@')[0]}",
            f"  latencia emulada {args.latencia_detector:g} s"]  # fmt: skip
    if depth is not None:
        info += [
            f"profundidad: {depth.model_id}",
            f"  latencia emulada {args.latencia_profundidad:g} s",
        ]
    st = DemoState()
    cursor = {"base": None, "next": 0.0}

    def on_frame(t_s, frame, view, sink):
        schedule.update(sink)
        ctl = holder["controller"]
        active = ctl.active
        if active is not None:
            names = {"search": "buscar", "distance": "distancia", "choose": "elegir"}
            st.order, st.label = active.utterance, active.target
            st.mode = names.get(active.mode.value, active.mode.value)
            handler = holder["handlers"].get(active.mode)
            target = getattr(handler, "target", None)
            st.state = getattr(handler, "state", "midiendo")
            st.target_angle = (target.angle_h_deg, target.angle_v_deg) if target else None
        else:
            st.mode, st.state, st.target_angle = "-", "inactivo", None
        if sonar is not None:
            guiding = active is not None and active.mode.value == "search"
            handler = holder["handlers"].get(active.mode) if guiding else None
            cue = sonar.step(t_s, target_of(handler, clock.now_ns()) if guiding else None)
            if cue is not None:
                sonar_cues.append(cue)
        while cursor["base"] is not None and cursor["next"] <= t_s:
            writer.write(add_subtitles(cursor["base"], schedule, cursor["next"], painter))
            cursor["next"] += 1 / args.fps
        if frame is not None:
            cursor["base"] = render_frame(frame, view, st, geometry, catalog.spoken, t_s, info,
                                          settings, painter)  # fmt: skip

    result = simulate(
        source, detector, make_controller, clock, load_orders(args), depth=depth, spatial=spatial,
        det_latency_s=args.latencia_detector, depth_latency_s=args.latencia_profundidad,
        every=args.cada, tick_s=tick_s, tail_s=args.cola, on_frame=on_frame,
    )  # fmt: skip
    if args.traza:
        with open(args.traza, "w", encoding="utf-8", newline="\n") as f:
            for s in result.spoken:
                f.write(json.dumps({"t_s": s.t_s, "tipo": "dice", "request_id": s.request_id,
                                    "texto": s.text}, ensure_ascii=False) + "\n")  # fmt: skip
            for t, kind, data in result.events:
                f.write(json.dumps({"t_s": t, "tipo": kind, **data}, ensure_ascii=False,
                                   default=str) + "\n")  # fmt: skip
    end = max([result.duration_s] + [s.end_s for s in schedule.segments]) + 0.5
    while cursor["base"] is not None and cursor["next"] <= end:
        writer.write(add_subtitles(cursor["base"], schedule, cursor["next"], painter))
        cursor["next"] += 1 / args.fps
    writer.release()
    audio = schedule.mix(end)
    if sonar is not None and sonar_cues:
        audio = mix_sonar(audio, sonar_cues, sonar.s, sample_rate)
    write_wav(tmp / "audio.wav", audio, sample_rate)
    if mux(tmp / "video.mp4", tmp / "audio.wav", out):
        print(f"demo: {out} ({end:.1f} s, con audio)")
    else:
        silent = out.with_name(out.stem + "-sin-audio.mp4")
        # move y no replace: el temporal puede estar en otro disco (C: contra D: en Windows).
        shutil.move(tmp / "video.mp4", silent)
        shutil.move(tmp / "audio.wav", out.with_suffix(".wav"))
        print(f"sin ffmpeg: video {silent} y audio {out.with_suffix('.wav')} por separado")
    for s in result.spoken:
        print(f"  {s.t_s:7.2f} s  [#{s.request_id}] {s.text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.modes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("simular", help="modos sobre una grabación, tiempo grabado")
    p.add_argument("--fuente", required=True, help="carpeta de grabación o 'sintetica'")
    p.add_argument("--orden", action="append", help="SEGUNDO:frase (se puede repetir)")
    p.add_argument("--ordenes", help="JSONL con {t_s, texto}")
    p.add_argument(
        "--profundidad",
        default="B0",
        help="B0–B5, fusion, crestereo, crestereo-preciso, depth.estimadores o ninguna",
    )
    p.add_argument("--latencia-detector", type=float, default=0.0, help="s emulados")
    p.add_argument("--latencia-profundidad", type=float, default=0.0, help="s emulados")
    p.add_argument("--cada", type=int, default=1, help="el detector mira 1 de cada N frames")
    p.add_argument("--frames", type=int, default=0, help="solo sintética: cuántos frames")
    p.add_argument("--cola", type=float, default=3.0, help="s simulados tras el último frame")
    p.add_argument("--traza", help="JSONL con lo dicho y los eventos")
    p.add_argument(
        "--capturas", help="carpeta para frames anotados al pedir audio (sin reproducirlo)"
    )
    p.add_argument("--config")
    p.set_defaults(fn=cmd_simular)

    d = sub.add_parser("demo", help="como simular, más un MP4 con imagen, subtítulos y voz")
    for action in p._actions:  # mismas opciones que simular
        if action.dest not in ("help", "capturas"):
            d._add_action(action)
    d.add_argument("--salida", required=True, help="archivo .mp4")
    d.add_argument("--sin-voz", action="store_true", help="sin Piper: subtítulos y silencio")
    d.add_argument("--fps", type=float, default=15.0, help="cuadros por segundo del video")
    d.set_defaults(fn=cmd_demo)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (EstimatorError, SourceError, InvalidInput) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
