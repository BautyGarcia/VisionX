"""Aplicación VisionX: ejecución interactiva y comparación científica (T07).

    uv run python -m visionx.app ejecutar [--fuente sintetica|CARPETA|usb] [--detector fake|yolo]
        [--profundidad fake|ninguna|B0|B1|B2|B3|B4|REGISTRO]
        [--profundidad-precisa NOMBRE|ninguna] [--segundos 0] [--sin-ventana] [--voz] [--consola]
        [--ordenes ordenes.jsonl] [--traza X.jsonl] [--captura X.png]
        [--retraso-detector MS] [--retraso-profundidad MS]
    uv run python -m visionx.app comparar [--fuente ...] [--detector ...] --csv X

Teclas en la ventana: c = cancelar (botón), espacio = hablar/terminar (con --voz), q = salir,
v = stand/diagnóstico, f = pantalla completa, a = panel de audio (micrófono y salida).
--consola permite escribir órdenes en la terminal. Con --ordenes, un JSONL {"t_s": 2, "texto":
"buscá la taza"} reproduce una sesión. Sin --voz, lo que se diría se imprime y no suena.

La profundidad "fake" y la fuente "sintetica" son sustitutos (app/fakes.py): no miden nada.
B0–B2 usan SGBM; B3/B4 usan el híbrido de T05. Necesitan un par rectificado y B3/B4,
pesos DA V2 Small locales. `depth.estimadores` admite fábricas públicas para otros métodos.
La ubicación por objeto es la provisoria de app/spatial.py hasta que A entregue T09/T13.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visionx.app.calibrate import build_calibrator
from visionx.app.camera_status import CameraStatus, CameraStatusSettings
from visionx.app.controller import Controller
from visionx.app.dashboard import DashboardState, render
from visionx.app.depth_registry import build_estimator
from visionx.app.fakes import (
    FakeDepthEstimator,
    ScriptedDetector,
    SyntheticStereoSource,
)
from visionx.app.precision import PrecisionWorker
from visionx.app.proximity import ProximityAlert, ProximitySettings
from visionx.app.runtime import Pipeline
from visionx.app.science import TraceWriter, compare
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes import ModeSettings, build_handlers
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import AudioEvent, Priority
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError
from visionx.shared.interfaces import SpatialEstimator
from visionx.voice.commands import CommandParser
from visionx.voice.output import AudioOutput, FakePlayer, PlaybackRecord


class PrintSynth:
    """Sin voz: imprime la frase y devuelve silencio breve. Para correr sin parlantes."""

    sample_rate = 8000

    def synthesize(self, text: str) -> np.ndarray:
        print(f"  [diría] {text}", flush=True)
        return np.zeros(80, np.float32)


def build_source(args: argparse.Namespace, cfg: dict[str, Any]):
    """(fuente, geometría de la profundidad, fuente sintética o None)."""
    if args.fuente == "sintetica":
        source = SyntheticStereoSource(fps=args.fps, n_frames=args.frames or None)
        return source, source.geometry, source
    from visionx.calibration import StereoCalibration
    from visionx.camera import (
        CameraSettings,
        RectifiedSource,
        ReplaySource,
        UsbStereoSource,
    )

    calib_path = Path((cfg.get("camera") or {}).get("calibration") or "")
    if not str(calib_path):
        raise InvalidInput("camera.calibration hace falta para rectificar")
    calibration = StereoCalibration.load(
        calib_path if calib_path.is_absolute() else ROOT / calib_path
    )
    if args.fuente == "usb":
        raw = UsbStereoSource(CameraSettings.from_config(cfg))
    else:
        raw = ReplaySource(args.fuente)
    source = RectifiedSource(raw, calibration)
    return source, source.geometry, None


def build_detector(args: argparse.Namespace, cfg: dict[str, Any], synthetic):
    if args.detector == "yolo":
        from visionx.detection import DetectorSettings
        from visionx.detection.yolo import build_detector as build_yolo

        return build_yolo(DetectorSettings.from_config(cfg))
    if synthetic is None:
        raise InvalidInput("--detector fake solo sirve con --fuente sintetica")
    return ScriptedDetector(synthetic, delay_ms=args.retraso_detector)


def build_output(args: argparse.Namespace, cfg: dict[str, Any], on_record):
    if not args.voz:
        return AudioOutput(PrintSynth(), FakePlayer(), on_record=on_record), None
    from visionx.voice.engines import PushToTalk, VoiceSettings, build_synth, build_transcriber
    from visionx.voice.output import SoundDevicePlayer

    s = VoiceSettings.from_config(cfg)
    synth = build_synth(s)
    synth.preload(s.preload)
    output = AudioOutput(
        synth,
        SoundDevicePlayer(s.output_device, s.block_ms, s.output_latency_s),
        on_record=on_record,
    )
    stt = build_transcriber(s)
    print(f"voz a texto: {stt.model_id}")
    return output, (PushToTalk(s.input_device, s.max_record_s), stt)


def load_script(path: str | None) -> list[tuple[float, str]]:
    if not path:
        return []
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            items.append((float(data["t_s"]), str(data["texto"])))
    return sorted(items)


def build_spatial(cfg: dict[str, Any], geometry, depth_geometry=None) -> SpatialEstimator:
    """Ubicación por objeto de A (`depth/spatial`, sección `spatial`).

    `spatial_provisional.usar: true` vuelve a la ubicación provisoria de B, para comparar.
    """
    s = cfg.get("spatial_provisional") or {}
    if s.get("usar"):
        return ProvisionalSpatialEstimator(
            geometry,
            min_valid=float(s.get("min_valid", 0.5)),
            depth_geometry=depth_geometry,
            max_rel_spread=float(s.get("max_rel_spread", 0.5)),
            border_px=float(s.get("border_px", 2.0)),
            layer_min_frac=float(s.get("layer_min_frac", 0.25)),
            front_percentile=float(s.get("front_percentile", 20.0)),
            spread_allowance_m=float(s.get("spread_allowance_m", 0.4)),
        )
    from visionx.depth.spatial import build_locator

    return build_locator(cfg, geometry, depth_geometry)


def build_depth(args: argparse.Namespace, cfg: dict[str, Any], geometry, synthetic):
    """(estimador de profundidad, ubicación espacial, geometría de la profundidad)."""
    if args.profundidad == "ninguna":
        return None, None, geometry
    if args.profundidad == "fake":
        depth = FakeDepthEstimator(geometry, source=synthetic, delay_ms=args.retraso_profundidad)
        return depth, build_spatial(cfg, geometry), geometry
    if synthetic is not None and args.profundidad in {"B0", "B1", "B2", "B3", "B4", "B5", "fusion"}:
        raise InvalidInput(f"--profundidad {args.profundidad} necesita una grabación o usb")
    depth = build_estimator(args.profundidad, cfg, geometry)
    depth_geometry = getattr(depth, "geometry", geometry)
    return depth, build_spatial(cfg, geometry, depth_geometry), depth_geometry


def build_precision(args: argparse.Namespace, cfg: dict[str, Any], geometry, on_event=None):
    """La resolución precisa puede diferir de la continua; no cambia parámetros congelados."""
    section = (cfg.get("depth") or {}).get("precisa") or {}
    name = args.profundidad_precisa
    if name is None:
        name = section.get("estimador", "ninguna")
    if name == "ninguna":
        return None
    precise_cfg = deepcopy(cfg)
    if section.get("compute"):
        precise_cfg.setdefault("depth", {})["compute"] = section["compute"]
    estimator = build_estimator(name, precise_cfg, geometry)
    return PrecisionWorker(
        estimator,
        build_spatial(cfg, geometry, getattr(estimator, "geometry", geometry)),
        wait_s=float(section.get("espera_s", 2.0)),
        on_event=on_event,
    )


def cmd_ejecutar(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    voice_cfg = cfg.get("audio") or {}
    trace = TraceWriter(args.traza) if args.traza else None
    from visionx.app.stand import LiveLog

    live = LiveLog()  # conversación y tiempos para la pantalla del stand

    def on_event(kind: str, data: dict) -> None:
        live.feed(kind, data)
        if trace is not None:
            trace(kind, data)

    source, geometry, synthetic = build_source(args, cfg)
    detector = build_detector(args, cfg, synthetic)
    depth, spatial, depth_geometry = build_depth(args, cfg, geometry, synthetic)
    precise = build_precision(args, cfg, geometry, on_event)

    last_audio: list[str] = ["-"]

    def on_record(record: PlaybackRecord) -> None:
        last_audio[0] = f"{record.text_or_sound} ({record.outcome})"
        on_event(
            "audio",
            record.trace_data(),
        )  # fmt: skip

    output, voice = build_output(args, cfg, on_record)
    pipeline = Pipeline(source, detector, depth, spatial, on_event=on_event, precise=precise)
    # Las clases del detector cargado (COCO o el vocabulario de YOLOE): «buscá el pizarrón»
    # solo se entiende si el modelo lo puede ver.
    catalog = ClassCatalog(getattr(detector, "class_names", None) or COCO_CLASSES)
    from visionx.assistant import build_assistant

    assistant = build_assistant(cfg)
    print(
        f"asistente: {assistant.s.model} (necesita internet)"
        if assistant
        else "asistente: apagado (sin ~/.visionx/anthropic_api_key, sin extra ia o tope de gasto)"
    )
    from visionx.app.onboarding import apply_preference, intro_phrases, restore
    from visionx.modes import body
    from visionx.voice.engines import VoiceSettings

    voice_synth = output.synth if args.voz else None
    base_speed = VoiceSettings.from_config(cfg).speech_rate
    controller = Controller(
        CommandParser(catalog, questions=assistant is not None),
        output,
        latest_view=pipeline.view,
        handlers=build_handlers(
            cfg,
            geometry,
            catalog.spoken,
            on_event,
            prewarm_reading=synthetic is None,
            assistant=assistant,
        ),
        deadlines_s={str(k): float(v) for k, v in (voice_cfg.get("deadlines_s") or {}).items()},
        on_event=on_event,
        watchers=[
            CameraStatus(CameraStatusSettings.from_config(cfg)),
            ProximityAlert(ProximitySettings.from_config(cfg), geometry=geometry),
            # El selector de vocabulario necesita saber qué se busca (detection/switch.py).
            *([detector] if hasattr(detector, "uses_extra") else []),
        ],
        calibrate=build_calibrator(depth, geometry, catalog.spoken, ModeSettings.from_config(cfg)),
        intro=lambda: intro_phrases(body.current(), body.preferences()),
        preferences=lambda change: apply_preference(change, voice_synth, assistant, base_speed),
    )
    script = load_script(args.ordenes)
    range_m = tuple((cfg.get("depth") or {}).get("range_m", [0.5, 3.0]))
    # El cuerpo de la persona: su altura guardada y la cámara estimada del piso (app/floor.py),
    # y cómo quiere que hable (velocidad y largo de las respuestas, app/onboarding.py).
    from visionx.app.floor import FloorTracker

    body.load()
    prefs = restore(voice_synth, assistant)
    if prefs:
        print(f"preferencias de la persona: {prefs}")
    floor = FloorTracker()
    # Sin piso: la cámara al 72 % de la estatura y nivelada (solo proporciones).
    use_floor = bool((cfg.get("persona") or {}).get("usar_piso", True))
    floor_key: list = [None]
    if args.voz and not args.ordenes:
        controller.announce("VisionX está listo. Decí: hola VisionX, y te cuento cómo funciono.")
    sonar = None
    if args.voz and ModeSettings.from_config(cfg).guide_style == "sonar":
        # Bips de la guía de buscar en su propio stream estéreo, al lado de la voz.
        from visionx.modes.sonar import SonarPlayer
        from visionx.shared.contracts import Mode

        out_device = (cfg.get("audio") or {}).get("output_device")
        sonar = SonarPlayer(controller, controller.handlers.get(Mode.SEARCH), device=out_device)
        sonar.start()
    if args.consola:

        def console() -> None:
            for line in sys.stdin:
                if line.strip():
                    controller.handle_text(line.strip())

        threading.Thread(target=console, name="visionx-consola", daemon=True).start()

    recording = False
    max_queues: dict[str, int] = {}
    # Panel de audio (tecla «a»): elegir micrófono y salida en vivo, solo para esta sesión.
    audio_ui: list[Any] = [None, None]  # [PanelAudio, MedidorNivel]

    def dashboard_state() -> DashboardState:
        return DashboardState(
            joint=pipeline.latest_joint(),
            statuses=(pipeline.capture_status, pipeline.detector_status, pipeline.depth_status)
            + ((precise.status,) if precise else ()),
            now_ns=pipeline.clock.now_ns(),
            range_m=range_m,  # type: ignore[arg-type]
            dropped=pipeline.dropped(),
            queues=pipeline.queue_sizes(),
            command=str(controller.active.utterance if controller.active else "-")
            + (" [grabando]" if recording else ""),
            audio=last_audio[0],
            geometry=depth_geometry,  # la vista de arriba usa la geometría de la profundidad
            measurement=precise.selection if precise else None,
            target_angle=guided_angle(),
            frame_geometry=geometry,
        )

    def guided_angle() -> tuple[float, float] | None:
        """Ángulos del objetivo de buscar mientras se guía (la flecha del dashboard)."""
        active = controller.active
        if active is None or active.mode.value != "search":
            return None
        handler = controller.handlers.get(active.mode)
        target = getattr(handler, "target", None)
        if target is None or getattr(handler, "state", "") != "guiando":
            return None
        return target.angle_h_deg, target.angle_v_deg or 0.0

    from visionx.app.stand import StandState, mode_label, render_stand

    mode_settings = ModeSettings.from_config(cfg)
    view_mode = [args.vista]

    def stand_state() -> StandState:
        turns, timing = live.snapshot()
        chip, target = mode_label(controller, catalog.spoken, recording)
        return StandState(
            joint=pipeline.latest_joint(),
            statuses=(pipeline.capture_status, pipeline.detector_status, pipeline.depth_status)
            + ((precise.status,) if precise else ()),
            geometry=depth_geometry,
            frame_geometry=geometry,
            range_m=range_m,  # type: ignore[arg-type]
            settings=mode_settings,
            mode=chip,
            recording=recording,
            turns=turns,
            timing=timing,
            body=body.current(),
            target_angle=guided_angle(),
            target_label=target,
            spoken=catalog.spoken,
        )

    def frame_image():
        if view_mode[0] == "camaras":
            from visionx.app.stand_camaras import render_camaras

            return render_camaras(args.camaras)
        return render_stand(stand_state()) if view_mode[0] == "stand" else render(dashboard_state())

    def open_audio_panel() -> None:
        """Lista los dispositivos de ahora (pudieron conectarse otros) y abre el panel."""
        from visionx.app.audio_panel import MedidorNivel, PanelAudio, listar, posicion

        try:
            import sounddevice as sd

            entradas, salidas = listar(sd)
            default_in, default_out = sd.default.device
        except Exception as exc:  # sin sounddevice o sin audio en el sistema
            panel = PanelAudio([], [])
            panel.abrir()
            panel.error = f"no se pudo leer el audio del sistema: {exc}"
            audio_ui[:] = [panel, None]
            return
        mic_now = voice[0].device if voice else None
        out_now = getattr(output.player, "device", None)
        panel = PanelAudio(
            entradas,
            salidas,
            mic=posicion(entradas, mic_now, default_in),
            salida=posicion(salidas, out_now, default_out),
        )
        panel.abrir()
        if voice is None:
            panel.error = "la voz está apagada: arrancá la app con --voz para cambiar el audio"
        audio_ui[:] = [panel, MedidorNivel(sd)]
        measure_mic(panel.bajo_cursor())

    def measure_mic(device) -> None:
        panel, meter = audio_ui
        if meter is None or device is None:
            return
        try:
            meter.abrir(device)
        except Exception as exc:
            meter.cerrar()
            panel.error = f"{device.nombre} no abre: {exc}"

    def close_audio_panel() -> None:
        panel, meter = audio_ui
        if meter is not None:
            meter.cerrar()
        if panel is not None:
            panel.cerrar()

    def audio_panel_key(name: str | None) -> None:
        """Una tecla con el panel abierto. Usar un dispositivo lo valida antes de aplicarlo."""
        panel = audio_ui[0]
        action = panel.manejar(name)
        if action is None:
            return
        kind, device = action
        if kind == "cerrar":
            close_audio_panel()
        elif kind == "medir":
            measure_mic(device)
        elif voice is not None:
            import sounddevice as sd

            try:
                if kind == "mic":
                    sd.check_input_settings(device=device.indice, channels=1, dtype="float32")
                    voice[0].device = device.indice  # vale desde la próxima vez que se hable
                else:
                    sd.check_output_settings(device=device.indice, channels=1, dtype="float32")
                    output.cancel_all()
                    output.player.set_device(device.indice)
                    # Bip por la salida nueva: confirma por dónde va a hablar VisionX.
                    output.submit(
                        AudioEvent(0, Priority.SHORT, "sonido:bip",
                                   created_ns=output.clock.now_ns())
                    )  # fmt: skip
            except Exception as exc:
                panel.error = f"{device.nombre} no se puede usar: {exc}"
                return
            panel.usar(kind, device)
            tipo = "entrada" if kind == "mic" else "salida"
            on_event("audio-dispositivo", {"tipo": tipo, "nombre": device.nombre,
                                           "indice": device.indice})  # fmt: skip

    def window_image():
        image = frame_image()
        panel, meter = audio_ui
        if panel is None or not panel.abierto:
            return image
        from visionx.app.audio_panel import dibujar_panel

        level, silent = meter.nivel() if meter is not None else (None, False)
        return dibujar_panel(image, panel, level, silent)

    window = not args.sin_ventana
    if window:
        import cv2

        cv2.namedWindow("VisionX", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VisionX", 1600, 900)
    pipeline.start()
    t0 = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - t0
            if args.segundos and elapsed >= args.segundos:
                break
            if pipeline.finished.is_set() and not pipeline.running:
                break
            while script and script[0][0] <= elapsed:
                controller.handle_text(script.pop(0)[1])
            if not recording:
                # Con el botón apretado los modos no avanzan: si la guía hablara ahora, el
                # micrófono grabaría la voz de VisionX encima de la orden (con parlantes,
                # Whisper la transcribe). La vigencia se revisa en el primer tick al soltar.
                controller.tick()
            for name, size in pipeline.queue_sizes().items():
                max_queues[name] = max(max_queues.get(name, 0), size)
            joint = pipeline.latest_joint()
            if (
                use_floor
                and joint is not None
                and joint.depth is not None
                and joint.depth.depth_m is not None
                and joint.frame.key != floor_key[0]
                and depth_geometry.size == joint.depth.size
            ):
                floor_key[0] = joint.frame.key
                floor.update(joint.depth.depth_m, joint.depth.valid_mask, depth_geometry, elapsed)
            if window:
                from visionx.app.audio_panel import tecla

                cv2.imshow("VisionX", window_image())
                # waitKeyEx: las flechas del panel de audio no entran en un byte.
                name = tecla(cv2.waitKeyEx(30))
                if audio_ui[0] is not None and audio_ui[0].abierto:
                    # Con el panel abierto las teclas son del panel: no se habla ni se sale
                    # sin querer, y el micrófono lo está usando la barra de nivel.
                    audio_panel_key(name)
                    continue
                key = ord(name) if name is not None and len(name) == 1 else -1
                if key == ord("a") and not recording:
                    open_audio_panel()
                    continue
                if key == ord("q"):
                    break
                if key == ord("v"):  # stand <-> diagnóstico
                    view_mode[0] = "diagnostico" if view_mode[0] == "stand" else "stand"
                if key == ord("k"):  # stand <-> demo de cámaras (precalculada)
                    view_mode[0] = "stand" if view_mode[0] == "camaras" else "camaras"
                if key == ord("f"):  # pantalla completa
                    full = cv2.getWindowProperty("VisionX", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("VisionX", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN
                                          else cv2.WINDOW_FULLSCREEN)  # fmt: skip
                if key == ord("c"):
                    controller.cancel()
                if key == ord(" ") and voice is not None:
                    mic, stt = voice
                    if not recording:
                        output.cancel_all()  # hablar corta lo que se estaba diciendo
                        mic.start()
                        recording = True
                    else:
                        audio = mic.stop()
                        recording = False
                        on_event("boton-soltado", {"audio_s": round(len(audio) / 16000, 2)})

                        def recognize(audio=audio, stt=stt) -> None:
                            result = stt.transcribe(audio)
                            on_event("stt", {"texto": result.text, "ms": result.elapsed_ms})
                            controller.handle_text(result.text)

                        threading.Thread(target=recognize, daemon=True).start()
            else:
                time.sleep(0.03)
    except KeyboardInterrupt:
        print("interrumpido")
    finally:
        if args.captura:
            # Lo último que se unió, dibujado igual que en la ventana: evidencia sin pantalla.
            # Antes de cerrar el controlador, para que se vea la orden activa.
            import cv2

            Path(args.captura).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.captura), frame_image())
            print(f"captura del dashboard: {args.captura}")
        if sonar is not None:
            sonar.close()
        if audio_ui[0] is not None:
            close_audio_panel()  # suelta el micrófono de la barra de nivel
        controller.close()
        pipeline.stop()
        output.wait_idle(timeout_s=5)
        output.close()
        if window:
            cv2.destroyAllWindows()
    summary = {
        "fuente": args.fuente,
        "segundos": round(time.monotonic() - t0, 1),
        "frames_capturados": pipeline.capture_status.processed,
        "detector": {"modelo": pipeline.detector_status.model_id,
                     "procesados": pipeline.detector_status.processed,
                     "errores": pipeline.detector_status.errors},
        "profundidad": {"modelo": pipeline.depth_status.model_id,
                        "procesados": pipeline.depth_status.processed,
                        "errores": pipeline.depth_status.errors},
        "precisa": {"modelo": precise.status.model_id, "procesados": precise.status.processed,
                    "errores": precise.status.errors} if precise else None,
        "salteados": pipeline.dropped(),
        "colas_maximas": max_queues,
        "errores_espaciales": pipeline.spatial_errors,
        "error_fuente": pipeline.error,
        "audio": {str(o): sum(1 for r in output.records if r.outcome == o)
                  for o in {r.outcome for r in output.records}},
    }  # fmt: skip
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False, default_flow_style=None))
    if trace is not None:
        trace("resumen", summary)
        trace.close()
        print(f"traza: {trace.path} ({trace.lines} eventos)")
    return 1 if pipeline.error else 0


def cmd_comparar(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    source, geometry, synthetic = build_source(args, cfg)
    methods: dict[str, Any] = {"detector": build_detector(args, cfg, synthetic)}
    depth, _, _ = build_depth(args, cfg, geometry, synthetic)
    if depth is not None:
        methods["profundidad"] = depth
    result = compare(source, methods, args.csv, max_frames=args.frames or None)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    print(f"detalle por frame en {args.csv}")
    return 0


def cmd_latencias(args: argparse.Namespace) -> int:
    from visionx.app.latency import load, summary

    print(summary(load(args.traza)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.app",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    from visionx.app.m1 import register

    register(sub)
    from visionx.app.recepcion import register as register_reception

    register_reception(sub)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--fuente", default="sintetica", help="sintetica, usb o carpeta de grabación"
        )
        p.add_argument("--detector", choices=["fake", "yolo"], default="fake")
        p.add_argument(
            "--profundidad",
            default="fake",
            help="fake, ninguna, B0–B5, fusion, crestereo o depth.estimadores",
        )
        p.add_argument("--fps", type=float, default=30.0, help="ritmo de la fuente sintética")
        p.add_argument("--frames", type=int, default=0, help="cortar la fuente tras N frames")
        p.add_argument("--retraso-detector", type=float, default=0.0, help="ms extra (fake)")
        p.add_argument("--retraso-profundidad", type=float, default=0.0, help="ms extra (fake)")
        p.add_argument("--config")

    p = sub.add_parser("ejecutar", help="recorrido interactivo con dashboard")
    common(p)
    p.add_argument(
        "--profundidad-precisa",
        default=None,
        help="B0–B5, fusion, crestereo-preciso, depth.estimadores o ninguna (default config)",
    )
    p.add_argument("--segundos", type=float, default=0.0, help="0 = hasta q o fin de la fuente")
    p.add_argument("--sin-ventana", action="store_true")
    p.add_argument("--voz", action="store_true", help="Piper + micrófono con espacio")
    p.add_argument("--consola", action="store_true", help="escribir órdenes en la terminal")
    p.add_argument("--ordenes", help="JSONL con {t_s, texto}")
    p.add_argument("--traza", help="JSONL de eventos (frames, resultados, órdenes, audio)")
    p.add_argument("--captura", help="PNG del dashboard al terminar (sirve con --sin-ventana)")
    p.add_argument(
        "--vista",
        choices=["stand", "diagnostico", "camaras"],
        default="stand",
        help="pantalla del stand (por defecto), el dashboard de diagnóstico (v alterna) o la "
        "demo de cámaras distintas (k alterna)",
    )
    p.add_argument(
        "--camaras",
        default=str(ROOT / "data" / "stand-camaras"),
        help="carpeta de la demo de cámaras (python -m visionx.gtsynth camaras)",
    )
    p.set_defaults(fn=cmd_ejecutar)

    p = sub.add_parser("latencias", help="tabla de latencias por orden de una traza en vivo")
    p.add_argument("--traza", required=True)
    p.set_defaults(fn=cmd_latencias)

    p = sub.add_parser("comparar", help="todos los métodos sobre todos los pares, sin descartes")
    common(p)
    p.add_argument("--csv", required=True)
    p.set_defaults(fn=cmd_comparar)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (EstimatorError, SourceError, InvalidInput) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
