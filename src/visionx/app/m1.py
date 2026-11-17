"""Banco M1 sin dispositivos: lógica real, percepción y audio falsos rotulados.

El reloj manual avanza 100 ms entre capturas y 20 ms de detección lista a tick.
AudioOutput ejecuta su cola/síntesis/cancelación reales; el reproductor no suena.
Las barreras no avanzan el reloj: 0 ms lógico significa el mismo instante simulado,
no una medición física. Los tiempos de pared incluyen planificación de hilos/CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import threading
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from visionx.app.controller import Controller
from visionx.app.fakes import (
    FakeDepthEstimator,
    ScriptedDetector,
    SyntheticObject,
    SyntheticStereoSource,
    synthetic_geometry,
)
from visionx.app.runtime import Pipeline, View
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.modes.common import ModeSettings
from visionx.modes.distance import DistanceHandler
from visionx.modes.search import SearchHandler
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import AudioEvent, DepthStatus, Mode
from visionx.voice.commands import CommandParser
from visionx.voice.output import AudioOutput, FakePlayer, Outcome, PlaybackRecord

CORE_CASES = (
    "presente",
    "ausente",
    "duplicado",
    "perdido",
    "cancelado",
    "vencido",
    "mascara-vacia",
    "depth-invalida",
)
EXTRA_CASES = (
    "deteccion-duplicada",
    "modelo-ausente",
    "camara-desconectada",
    "audio-desconectado",
    "fuera-de-orden",
    "cancelar-ocr",
    "cancelar-inferencia",
    "cancelar-tts",
    "vencido-tts",
)
WAIT_S = 10.0  # límite de sincronización, nunca se usa como umbral de rendimiento


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_until(predicate, message: str) -> None:
    end = time.monotonic() + WAIT_S
    while not predicate():
        require(time.monotonic() < end, message)
        time.sleep(0.001)


class SilentSynth:
    """Fake rotulado: 10 ms de ceros; no Piper ni latencia de síntesis emulada."""

    sample_rate = 8000

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(80, np.float32)


class GateSynth(SilentSynth):
    """Detiene la síntesis de una frase real hasta liberar la barrera."""

    def __init__(self, match: str = "") -> None:
        self.match = match
        self.entered, self.release = threading.Event(), threading.Event()

    def synthesize(self, text: str) -> np.ndarray:
        if self.match in text:
            self.entered.set()
            require(self.release.wait(WAIT_S), "no se liberó síntesis fake")
        return super().synthesize(text)


class GatePlayer(FakePlayer):
    """Un audio queda sonando hasta cancel; observa el stop del AudioOutput real."""

    def __init__(self) -> None:
        super().__init__()
        self.cut = threading.Event()
        self.block = True

    def play(self, samples, sample_rate, stop) -> bool:
        self.started.set()
        if self.block:
            require(stop.wait(WAIT_S), "audio fake no recibió stop")
            self.block = False
            self.cut.set()
            return False
        return super().play(samples, sample_rate, stop)


class AuditSink:
    """Observa submit/cancel sin sustituir la semántica del AudioOutput."""

    def __init__(self, bank: Scenario, output: AudioOutput) -> None:
        self.bank, self.output = bank, output
        self.events: list[AudioEvent] = []

    def submit(self, event: AudioEvent) -> None:
        self.events.append(event)
        seen, measured = self.bank.view.seen, self.bank.view.measured
        # La frase de pérdida describe explícitamente la última observación pasada;
        # no se presenta su edad como una dirección actual.
        reference = seen
        if " metro" in event.text_or_sound:
            reference = measured
        if event.text_or_sound.startswith("Perdí"):
            reference = self.bank.last_positive
        factual = not event.text_or_sound.startswith(("Busco", "Leo", "sonido:", "No encontré"))
        observed = reference.frame.received_ns if reference and factual else None
        self.bank.log(
            "audio-submit",
            {
                "request_id": event.request_id,
                "texto": event.text_or_sound,
                "created_ns": event.created_ns,
                "deadline_ns": event.deadline_ns,
                "observed_ns": observed,
                "frame_id": reference.frame.frame_id if observed is not None else None,
                "tipo_edad": "historica" if event.text_or_sound.startswith("Perdí") else "actual",
            },
        )
        self.output.submit(event)

    def cancel(self, request_id: int) -> None:
        self.bank.log("audio-cancel", {"request_id": request_id})
        self.output.cancel(request_id)

    def cancel_all(self) -> None:
        self.bank.log("audio-cancel-all", {})
        self.output.cancel_all()

    def close(self) -> None:
        self.output.close()


class ObservedPlayer:
    """Marca entrada/salida al reproductor falso, sin confundir submit con inicio."""

    def __init__(self, bank: Scenario, player) -> None:
        self.bank, self.player = bank, player

    def play(self, samples, sample_rate, stop):
        record = next(
            r for r in self.bank.output.records if r.started_ns is not None and r.outcome is None
        )
        data = {"request_id": record.request_id, "texto": record.text_or_sound}
        self.bank.log("audio-inicio", data)
        try:
            return self.player.play(samples, sample_rate, stop)
        finally:
            self.bank.log("audio-fin", data)

    def close(self) -> None:
        self.player.close()


class Scenario:
    """Una ejecución aislada. Sin hilos de percepción salvo escenarios que los prueban."""

    def __init__(self, case: str, repetition: int, *, synth=None, player=None) -> None:
        self.case, self.repetition = case, repetition
        self.clock = ManualClock(1_000_000_000)
        self.wall_start = time.perf_counter_ns()
        self.rows: list[dict[str, Any]] = []
        self.checks: dict[str, bool] = {}
        self.metrics: dict[str, list[float]] = {}
        self.view = View()
        self.last_positive = None
        self.geometry = synthetic_geometry(160, 96)
        self.source = SyntheticStereoSource(
            geometry=self.geometry, fps=None, clock=self.clock, session_id=f"m1-{case}"
        )
        self.detector = ScriptedDetector(self.source)
        self.depth = FakeDepthEstimator(self.geometry, source=self.source)
        self.pipeline = Pipeline(
            self.source,
            self.detector,
            self.depth,
            ProvisionalSpatialEstimator(self.geometry),
            clock=self.clock,
            on_event=self.log,
        )
        self.player = player or FakePlayer()
        self.output = AudioOutput(
            synth or SilentSynth(),
            ObservedPlayer(self, self.player),
            clock=self.clock,
            on_record=self.record,
        )
        self.sink = AuditSink(self, self.output)
        catalog = ClassCatalog(COCO_CLASSES)
        self.settings = ModeSettings()
        self.controller = Controller(
            CommandParser(catalog),
            self.sink,
            latest_view=lambda: self.view,
            handlers={
                Mode.SEARCH: SearchHandler(
                    self.geometry, catalog.spoken, self.settings, on_event=self.log
                ),
                Mode.DISTANCE: DistanceHandler(
                    self.geometry, catalog.spoken, self.settings, on_event=self.log
                ),
            },
            deadlines_s={"search": 12, "distance": 8, "read": 30},
            clock=self.clock,
            on_event=self.log,
        )

    def log(self, kind: str, data: dict[str, Any]) -> None:
        # append es atómico bajo CPython; no avanzar el reloj desde los workers.
        self.rows.append(
            {
                "caso": self.case,
                "repeticion": self.repetition,
                "evento": kind,
                "logical_ns": self.clock.now_ns(),
                "wall_ns": time.perf_counter_ns() - self.wall_start,
                **data,
            }
        )

    def record(self, record: PlaybackRecord) -> None:
        self.log("audio-record", asdict(record))

    def check(self, name: str, condition: Any) -> None:
        self.checks[name] = bool(condition)

    def drain(self) -> None:
        require(self.output.wait_idle(WAIT_S), "AudioOutput no quedó inactivo")

    def order(self, text: str) -> None:
        self.controller.handle_text(text)

    def step(self, *, visible=True, empty_mask=False, invalid_depth=False, duplicate=False) -> None:
        self.clock.advance(100_000_000)
        self.source.objects = (SyntheticObject("cup", 1.2),) if visible else ()
        frame = self.source.read()
        require(frame is not None, "fuente sintética agotada")
        det = self.detector.predict(frame)
        if duplicate:
            # Dos instancias separadas: el usuario debe poder elegir entre iguales.
            boxes = np.concatenate([det.boxes, det.boxes + np.array([80, 0, 80, 0])])
            masks = np.concatenate([det.masks, np.roll(det.masks, 80, axis=2)])
            det = replace(
                det,
                boxes=boxes.astype(np.float32),
                masks=masks,
                classes=("cup", "cup"),
                scores=np.array([0.9, 0.8], np.float32),
                track_ids=(1, 2),
            )
        if empty_mask:
            det = replace(det, masks=np.zeros_like(det.masks))
        depth = self.depth.predict(frame)
        if invalid_depth:
            depth = replace(
                depth,
                valid_mask=np.zeros_like(depth.valid_mask),
                status=DepthStatus.UNMEASURABLE,
                reason="fake-M1-depth-invalida",
            )
        self.pipeline.board.put_detection(det, frame)
        self.pipeline.board.put_depth(depth)
        joint = self.pipeline.joint_for(frame.key)
        require(joint is not None, "no se pudo unir el mismo frame")
        self.view = View(replace(joint, depth=None, objects=()), joint, frame)
        if visible:
            self.last_positive = self.view.seen
        self.log("deteccion-lista", {"frame_id": frame.frame_id, "visible": visible})
        self.clock.advance(20_000_000)  # tick periódico, explícito y reproducible
        self.controller.tick()

    def acquire(self, *, duplicate=False) -> None:
        self.order("buscá la taza")
        self.drain()
        for _ in range(self.settings.confirm_hits):
            self.step(duplicate=duplicate)
            self.drain()

    def texts(self) -> list[str]:
        return [r.text_or_sound for r in self.output.records if r.started_ns is not None]

    def has(self, event: str) -> bool:
        return any(r["evento"] == event for r in self.rows)

    def cancel(self, release: threading.Event | None = None) -> None:
        old = {e.request_id for e in self.sink.events}
        start = self.clock.now_ns()
        wall = time.perf_counter_ns()
        self.log("cancelar-inicio", {})
        begin = self.rows[-1]
        self.order("cancelar")
        self.log("modo-detenido", {"detenido": self.controller.active is None})
        mode_wall = self.rows[-1]["wall_ns"]
        if release is not None:
            self.check(
                "cancela_sin_esperar_tts", self.controller.active is None and not release.is_set()
            )
            release.set()
        self.drain()
        # Medimos hasta la terminación de los registros del pedido anterior y el modo parado.
        ended = max(
            (r.ended_ns or start for r in self.output.records if r.request_id in old), default=start
        )
        self.metrics["cancelacion_logica_ms"] = [(max(start, ended) - start) / 1e6]
        stops = [
            r["wall_ns"]
            for r in self.rows
            if r["evento"] == "audio-fin"
            and r["request_id"] in old
            and r["wall_ns"] >= begin["wall_ns"]
        ]
        self.metrics["cancelacion_pared_ms"] = [(max([mode_wall, *stops]) - begin["wall_ns"]) / 1e6]
        self.metrics["cancelacion_hasta_idle_pared_ms"] = [(time.perf_counter_ns() - wall) / 1e6]
        self.check("modo_detenido", self.controller.active is None)
        self.check(
            "sin_respuesta_cancelada",
            not any(
                r["evento"] == "audio-inicio"
                and r["wall_ns"] >= begin["wall_ns"]
                and r["request_id"] in old
                for r in self.rows
            ),
        )
        self.log("cancelar-fin", {"modo_detenido": self.controller.active is None})

    def finish(self) -> dict[str, Any]:
        self.controller.close()
        self.output.close()
        self.source.close()
        # El vínculo se hace con el submit observado, sin atribuir frames a confirmaciones.
        submits = [r for r in self.rows if r["evento"] == "audio-submit"]
        used: set[int] = set()
        for record in self.output.records:
            index = next(
                (
                    i
                    for i, s in enumerate(submits)
                    if i not in used
                    and s["request_id"] == record.request_id
                    and s["texto"] == record.text_or_sound
                ),
                None,
            )
            if index is None:
                continue
            used.add(index)
            submit = submits[index]
            if record.started_ns is None:
                continue
            if submit["observed_ns"] is not None:
                age = (record.started_ns - submit["observed_ns"]) / 1e6
                metric = (
                    "edad_hecho_historico_ms"
                    if submit["tipo_edad"] == "historica"
                    else "edad_dicho_ms"
                )
                self.metrics.setdefault(metric, []).append(age)
                self.log(
                    "edad-dicho",
                    {
                        "texto": record.text_or_sound,
                        "edad_ms": age,
                        "tipo_edad": submit["tipo_edad"],
                        "frame_id": submit["frame_id"],
                        "audio_started_ns": record.started_ns,
                    },
                )
            if record.text_or_sound == "sonido:bip":
                ready = [
                    r
                    for r in self.rows
                    if r["evento"] == "deteccion-lista"
                    and r["logical_ns"] <= record.started_ns
                    and r.get("visible")
                ]
                if ready:
                    # Incluye confirmación de presencia desde la PRIMERA detección positiva.
                    self.metrics.setdefault("deteccion_bip_logica_ms", []).append(
                        (record.started_ns - ready[0]["logical_ns"]) / 1e6
                    )
                    starts = [
                        r
                        for r in self.rows
                        if r["evento"] == "audio-inicio"
                        and r["request_id"] == record.request_id
                        and r["texto"] == record.text_or_sound
                    ]
                    if starts:
                        self.metrics.setdefault("deteccion_bip_pared_ms", []).append(
                            (starts[0]["wall_ns"] - ready[0]["wall_ns"]) / 1e6
                        )
        expired = [
            r
            for r in self.output.records
            if r.started_ns is not None
            and any(
                s["request_id"] == r.request_id
                and s["texto"] == r.text_or_sound
                and s["deadline_ns"] is not None
                and r.started_ns >= s["deadline_ns"]
                for s in submits
            )
        ]
        self.check("cero_audio_vencido", not expired)
        self.check(
            "audio_sin_errores_inesperados",
            self.case == "audio-desconectado"
            or not any(r.outcome == Outcome.FAILED for r in self.output.records),
        )
        return {
            "caso": self.case,
            "repeticion": self.repetition,
            "grupo": "base" if self.case in CORE_CASES else "adicional",
            "resultado": "PASA" if self.checks and all(self.checks.values()) else "FALLA",
            "criterios": self.checks,
            "metricas": self.metrics,
            "frases_iniciadas": self.texts(),
            "wall_ms": (time.perf_counter_ns() - self.wall_start) / 1e6,
        }


def core_case(bank: Scenario) -> None:
    case = bank.case
    if case in ("mascara-vacia", "depth-invalida"):
        bank.order("a qué distancia está la taza")
        bank.drain()
        bank.step(empty_mask=case == "mascara-vacia", invalid_depth=case == "depth-invalida")
        bank.drain()
        bank.check("abstencion_explicita", any("no puedo medir" in t for t in bank.texts()))
        bank.check("sin_metros", not any(" metro" in t for t in bank.texts()))
        bank.check("respuesta_distancia", bank.has("distancia-respuesta"))
        return
    if case == "ausente":
        bank.order("buscá la taza")
        for _ in range(3):
            bank.step(visible=False)
            bank.drain()
        command = bank.controller.active
        require(command is not None, "la búsqueda terminó antes de tiempo")
        bank.clock.set(command.deadline_ns - int(bank.settings.end_margin_s * 1e9))
        bank.controller.tick()
        bank.drain()
        bank.check("ausencia_explicita", bank.texts()[-1] == "No encontré taza.")
        bank.check(
            "sin_bip_ni_guia", not bank.has("buscar-adquirido") and "sonido:bip" not in bank.texts()
        )
        bank.check("modo_terminado", bank.controller.active is None)
        return
    if case == "vencido":
        # La orden sigue vigente, pero el frame que originó la guía vence mientras
        # espera síntesis. Comprobarlo al iniciar audio, además del deadline del pedido.
        bank.order("buscá la taza")
        bank.drain()
        bank.step()
        bank.drain()
        bank.step()
        gate = bank.output.synth
        require(gate.entered.wait(WAIT_S), "la guía no llegó a síntesis")
        bank.clock.advance(int((bank.settings.max_age_s + 0.1) * 1e9))
        gate.release.set()
        bank.drain()
        guide = [r for r in bank.output.records if r.text_or_sound.startswith("Taza")]
        bank.check(
            "guia_en_cola_vieja_no_suena", bool(guide) and all(r.started_ns is None for r in guide)
        )
    else:
        bank.acquire(duplicate=case == "duplicado")
    if case == "duplicado":
        bank.check(
            "pide_eleccion", bank.has("buscar-varios") and any("¿Cuál?" in t for t in bank.texts())
        )
        bank.check("no_elige_solo", not bank.has("buscar-adquirido"))
        bank.order("la de la derecha")
        bank.drain()
        bank.check(
            "eleccion_respetada", bank.has("buscar-eleccion") and bank.has("buscar-adquirido")
        )
        return
    bank.check("presencia_confirmada", bank.has("buscar-adquirido"))
    bank.check("un_bip", bank.texts().count("sonido:bip") == 1)
    if case == "perdido":
        for _ in range(23):  # 2,76 s de frames frescos sin objeto; lost_s por defecto = 2,5
            bank.step(visible=False)
            bank.drain()
        bank.check(
            "perdida_explicita",
            bank.has("buscar-perdido") and sum(t.startswith("Perdí") for t in bank.texts()) == 1,
        )
        after = len(bank.sink.events)
        for _ in range(4):
            bank.step(visible=False)
            bank.drain()
        bank.check("sin_guia_tras_perder", len(bank.sink.events) == after)
    elif case == "cancelado":
        player = bank.player
        player.started.clear()
        player.block = True
        bank.clock.advance(4_000_000_000)
        bank.step()
        require(player.started.wait(WAIT_S), "no empezó la guía que se quiere cortar")
        bank.cancel()
        bank.check(
            "audio_cortado",
            player.cut.is_set()
            and any(r.outcome == Outcome.CANCELLED for r in bank.output.records),
        )
        after = len(bank.sink.events)
        for _ in range(5):
            bank.step()
        bank.drain()
        bank.check(
            "silencio_tras_cancelar",
            len(bank.sink.events) == after and bank.texts()[-1] == "sonido:cancelado",
        )
    elif case == "vencido":
        after = len(bank.sink.events)
        bank.clock.advance(4_000_000_000)  # frame ya viejo, todavía no vence la orden
        bank.controller.tick()
        bank.drain()
        bank.check("sin_guia_de_frame_viejo", len(bank.sink.events) == after)
        command = bank.controller.active
        require(command is not None, "la búsqueda no sigue activa")
        bank.clock.set(command.deadline_ns)
        bank.controller.tick()
        bank.drain()
        bank.check("orden_vencida", bank.controller.active is None and bank.has("vencido"))
        bank.step()
        bank.drain()
        bank.check("silencio_tras_vencer", len(bank.sink.events) == after)


def percentiles(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "unidad": "ms",
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def run_bank(repetitions: int = 3, *, cases: tuple[str, ...] | None = None) -> tuple[dict, list]:
    """Ejecutar sin archivos. Fallos de un caso quedan visibles y no impiden los demás."""
    from visionx.app.m1_faults import fault_case, make_bank

    if repetitions < 1:
        raise ValueError("repeticiones debe ser >= 1")
    runs, rows = [], []
    for case in cases or CORE_CASES + EXTRA_CASES:
        if case not in CORE_CASES + EXTRA_CASES:
            raise ValueError(f"caso desconocido: {case}")
        for repetition in range(1, repetitions + 1):
            bank = make_bank(case, repetition)
            try:
                if case in CORE_CASES:
                    core_case(bank)
                else:
                    fault_case(bank)
            except Exception as exc:
                bank.log("fallo-banco", {"error": f"{type(exc).__name__}: {exc}"})
                bank.check("ejecucion_sin_excepcion", False)
            finally:
                release = getattr(bank.output.synth, "release", None)
                if release is not None:
                    release.set()
                runs.append(bank.finish())
                rows.extend(bank.rows)
    metrics: dict[str, list[float]] = {}
    for run in runs:
        for name, values in run["metricas"].items():
            metrics.setdefault(name, []).extend(values)
    core = [r for r in runs if r["grupo"] == "base"]
    aggregated = {name: percentiles(values) for name, values in metrics.items()}
    checks = {
        "eventos_base_correctos": len(core) == 8 * repetitions
        and all(r["resultado"] == "PASA" for r in core),
        "cero_audio_viejo": all(r["criterios"].get("cero_audio_vencido", False) for r in runs),
        "cancelacion_logica_p95_200_ms": bool(metrics.get("cancelacion_logica_ms"))
        and aggregated["cancelacion_logica_ms"]["p95"] <= 200,
        "deteccion_bip_p95_500_ms": bool(metrics.get("deteccion_bip_logica_ms"))
        and aggregated["deteccion_bip_logica_ms"]["p95"] <= 500,
    }
    checks["cero_audio_viejo"] = checks["cero_audio_viejo"] and all(
        r["criterios"].get("sin_respuesta_cancelada", True)
        and r["criterios"].get("sin_guia_de_frame_viejo", True)
        and r["criterios"].get("guia_en_cola_vieja_no_suena", True)
        for r in runs
    )
    return {
        "schema_banco": 1,
        "repeticiones": repetitions,
        "tipo": "sintetico-sin-dispositivos",
        "fecha_utc": datetime.now(UTC).isoformat(),
        "configuracion": {
            "geometria": "160x96/fx152/B0.06m",
            "settings_modos": asdict(ModeSettings()),
            "modelos": ["fake-detector-sintetico", "fake-profundidad", "fake-reader"],
            "sintesis": "fake-silencio-10ms",
            "dispositivos": False,
        },
        "reloj": "ManualClock; captura 100 ms + tick 20 ms; audio instantáneo salvo barreras",
        "limites": "No mide latencia audible, precisión YOLO/OCR, GPU ni hardware. "
        "0 ms lógico = mismo instante virtual. Pared incluye competencia de CPU.",
        "entorno": {"python": platform.python_version(), "plataforma": platform.platform()},
        "criterios": {k: "PASA" if v else "FALLA" for k, v in checks.items()},
        "base_correctos": sum(r["resultado"] == "PASA" for r in core),
        "base_total": len(core),
        "metricas": aggregated,
        "corridas": runs,
        "resultado": "PASA"
        if all(checks.values()) and all(r["resultado"] == "PASA" for r in runs)
        else "FALLA",
    }, rows


def table(summary: dict) -> str:
    lines = ["| Caso / criterio | Resultado |", "|---|---|"]
    for case in dict.fromkeys(r["caso"] for r in summary["corridas"]):
        runs = [r for r in summary["corridas"] if r["caso"] == case]
        passed = sum(r["resultado"] == "PASA" for r in runs)
        lines.append(
            f"| {case} | {'PASA' if passed == len(runs) else 'FALLA'} {passed}/{len(runs)} |"
        )
    lines += [f"| {k} | {v} |" for k, v in summary["criterios"].items()]
    lines += ["", "| Métrica | n | P50 ms | P95 ms |", "|---|---:|---:|---:|"]
    for name, metric in sorted(summary["metricas"].items()):
        lines.append(f"| {name} | {metric['n']} | {metric['p50']:.3f} | {metric['p95']:.3f} |")
    return "\n".join(lines) + "\n"


def write_results(summary: dict, rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        summary["codigo_sin_commit"] = bool(status.strip())
        summary["git_status"] = status.splitlines()
    except (OSError, subprocess.CalledProcessError):
        summary["git_head"] = None
        summary["codigo_sin_commit"] = None
    package_root = Path(__file__).resolve().parents[1]
    summary["sha256_codigo"] = {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
        if path.parts[-2] in ("app", "modes", "voice", "reading")
    }
    source_root = Path(__file__).resolve().parents[1]
    summary["codigo_sha256"] = {
        p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("app", "modes", "voice", "reading")
        for p in sorted((source_root / folder).glob("*.py"))
    }
    (output / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "tabla.md").write_text(table(summary), encoding="utf-8", newline="\n")
    columns = ["caso", "repeticion", "evento", "logical_ns", "wall_ns", "datos_json"]
    with (output / "eventos.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{k: row[k] for k in columns[:-1]},
                    "datos_json": json.dumps(
                        {k: v for k, v in row.items() if k not in columns}, ensure_ascii=False
                    ),
                }
            )


def cmd_m1(args: argparse.Namespace) -> int:
    if args.repeticiones < 1:
        raise ValueError("--repeticiones debe ser >= 1")
    summary, rows = run_bank(args.repeticiones)
    write_results(summary, rows, Path(args.salida))
    print(table(summary))
    print(summary["limites"])
    if args.clip:
        from visionx.app.m1_clip import run_clip

        clip = run_clip(args, Path(args.salida) / "clip")
        return int(summary["resultado"] != "PASA" or clip["resultado"] != "PASA")
    return int(summary["resultado"] != "PASA")


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("se necesita un entero >= 1") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("se necesita un entero >= 1")
    return value


def register(subparsers) -> None:
    parser = subparsers.add_parser("m1", help="banco M1 reproducible, sin dispositivos")
    parser.add_argument("--repeticiones", type=positive_int, default=3)
    parser.add_argument("--salida", default="outputs/m1")
    parser.add_argument("--clip", help="carpeta de replay opcional: YOLO+B0, informe separado")
    parser.add_argument("--config", help="override de calibración/modelo para el clip")
    parser.add_argument("--ordenes", help="JSONL {t_s, texto}; por defecto busca la taza en t=0")
    parser.add_argument(
        "--cada", type=positive_int, default=10, help="detectar uno de cada N frames del clip"
    )
    parser.set_defaults(fn=cmd_m1)
