"""Fallos adicionales de M1 con adaptadores falsos y barreras, sin dispositivos."""

from __future__ import annotations

import tempfile
import threading
from dataclasses import replace

import numpy as np

from visionx.app.fakes import ScriptedDetector
from visionx.app.m1 import (
    WAIT_S,
    GatePlayer,
    GateSynth,
    Scenario,
    require,
    wait_until,
)
from visionx.app.runtime import Pipeline
from visionx.modes.read import ReadHandler
from visionx.reading import FakeReader, ReadingSettings, TextBlock
from visionx.shared.contracts import Mode
from visionx.shared.errors import EstimatorError, SourceError
from visionx.voice.output import FakePlayer, Outcome


class BrokenPlayer(FakePlayer):
    def play(self, samples, sample_rate, stop):
        raise OSError("fake-M1-audio-desconectado")


def make_bank(case: str, repetition: int) -> Scenario:
    if case == "cancelado":
        player = GatePlayer()
        player.block = False
        return Scenario(case, repetition, player=player)
    if case == "cancelar-tts":
        return Scenario(case, repetition, synth=GateSynth())
    if case == "audio-desconectado":
        return Scenario(case, repetition, player=BrokenPlayer())
    if case == "vencido":
        return Scenario(case, repetition, synth=GateSynth("Taza"))
    if case == "vencido-tts":
        return Scenario(case, repetition, synth=GateSynth())
    return Scenario(case, repetition)


def fault_case(bank: Scenario) -> None:
    case = bank.case
    if case == "modelo-ausente":
        from visionx.detection import DetectorSettings, YoloDetector

        with tempfile.TemporaryDirectory(prefix="visionx-m1-") as directory:
            try:
                YoloDetector(DetectorSettings(weights_dir=directory))
            except EstimatorError as exc:
                bank.log("modelo-error", {"error": str(exc)})
                bank.check("error_peso_ausente", "no están los pesos" in str(exc))
            else:
                bank.check("error_peso_ausente", False)
        bank.check("sin_audio_inventado", not bank.sink.events)
    elif case == "audio-desconectado":
        bank.acquire()
        bank.check(
            "fallo_audio_visible",
            bool(bank.output.records)
            and all(
                r.outcome == Outcome.FAILED and "dispositivo" in (r.error or "")
                for r in bank.output.records
            ),
        )
        bank.cancel()
        bank.check("controlador_sigue_operativo", bank.controller.active is None)
    elif case == "vencido-tts":
        gate = bank.output.synth
        bank.order("buscá la taza")
        require(gate.entered.wait(WAIT_S), "TTS fake no entró")
        command = bank.controller.active
        bank.clock.set(command.deadline_ns + 1)
        gate.release.set()
        bank.drain()
        bank.check(
            "tts_vencido_no_reproducido",
            bank.output.records[0].outcome == Outcome.EXPIRED
            and bank.output.records[0].started_ns is None,
        )
        bank.controller.tick()
        bank.check("modo_vencido", bank.controller.active is None)
    elif case == "cancelar-tts":
        gate = bank.output.synth
        bank.order("buscá la taza")
        require(gate.entered.wait(WAIT_S), "TTS fake no entró")
        try:
            bank.cancel(release=gate.release)
        finally:
            gate.release.set()
        bank.check(
            "tts_viejo_no_reproducido",
            bank.output.records[0].outcome == Outcome.CANCELLED
            and bank.output.records[0].started_ns is None,
        )
        bank.check("solo_confirmacion_cancelar", bank.texts() == ["sonido:cancelado"])
    elif case == "cancelar-ocr":
        _cancel_ocr(bank)
    elif case in ("camara-desconectada", "cancelar-inferencia", "fuera-de-orden"):
        _pipeline_fault(bank)
    elif case == "deteccion-duplicada":
        _duplicate(bank)


def _duplicate(bank: Scenario) -> None:
    """Además de dos instancias, verificar dos cajas superpuestas del mismo objeto."""
    bank.step()
    bank.drain()
    joint = bank.view.seen
    det = joint.detection
    det = replace(
        det,
        boxes=np.concatenate([det.boxes, det.boxes]),
        masks=np.concatenate([det.masks, det.masks]),
        classes=("cup", "cup"),
        scores=np.array([0.9, 0.8], np.float32),
        track_ids=(1, 2),
    )
    bank.view = replace(bank.view, seen=replace(joint, detection=det), measured=None)
    bank.order("buscá la taza")
    bank.drain()
    bank.step()
    bank.drain()
    bank.check("no_pide_eleccion_falsa", not bank.has("buscar-varios"))
    bank.check(
        "adquiere_un_objeto", bank.has("buscar-adquirido") and bank.texts().count("sonido:bip") == 1
    )


def _cancel_ocr(bank: Scenario) -> None:
    class GatedReader(FakeReader):
        def __init__(self):
            super().__init__([TextBlock("RESULTADO VIEJO", 0.99, (10, 10, 100, 40))])
            self.entered, self.release, self.returned = (threading.Event() for _ in range(3))

        def read(self, image):
            self.entered.set()
            require(self.release.wait(WAIT_S), "OCR fake no liberado")
            self.returned.set()
            return super().read(image)

    reader = GatedReader()
    handler = ReadHandler(reader, ReadingSettings(window_s=0.01, crop="none"), bank.log)
    bank.controller.handlers[Mode.READ] = handler
    bank.step()
    bank.drain()
    bank.order("leé")
    bank.clock.advance(20_000_000)
    bank.controller.tick()
    require(reader.entered.wait(WAIT_S), "OCR fake no entró")
    try:
        bank.cancel()
        bank.check(
            "cancela_sin_esperar_ocr",
            bank.controller.active is None and not reader.release.is_set(),
        )
    finally:
        reader.release.set()
    require(reader.returned.wait(WAIT_S), "OCR no devolvió el resultado tardío")
    bank.controller.tick()
    bank.drain()
    bank.check(
        "ocr_viejo_ignorado",
        not any("RESULTADO VIEJO" in t for t in bank.texts()) and not bank.has("leer-resultado"),
    )


def _pipeline_fault(bank: Scenario) -> None:
    case = bank.case
    source = bank.source
    entered, release = threading.Event(), threading.Event()

    class CuttingSource:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return source.read()
            if case == "camara-desconectada":
                raise SourceError("fake-M1-camara-desconectada")
            return None

        def close(self):
            source.close()

    class GatedDetector(ScriptedDetector):
        def predict(self, frame):
            if case == "cancelar-inferencia":
                entered.set()
                require(release.wait(WAIT_S), "inferencia fake no liberada")
            result = super().predict(frame)
            # Un adaptador que retorna una identidad incorrecta: Pipeline debe rechazarla.
            return (
                replace(result, frame_id=frame.frame_id + 1) if case == "fuera-de-orden" else result
            )

    if case == "camara-desconectada":
        bank.acquire()
        bank.check("guia_activa_antes_del_corte", bank.has("buscar-adquirido"))
    pipe = Pipeline(CuttingSource(), GatedDetector(source), clock=bank.clock, on_event=bank.log)
    bank.controller.latest_view = pipe.view
    if case != "camara-desconectada":
        bank.order("buscá la taza")
    pipe.start()
    try:
        if case == "cancelar-inferencia":
            require(entered.wait(WAIT_S), "inferencia no entró")
            bank.cancel()
            bank.check(
                "cancela_sin_esperar_inferencia",
                not release.is_set() and bank.controller.active is None,
            )
            release.set()
        wait_until(lambda: not pipe.running, "Pipeline no terminó")
        bank.controller.tick()
        bank.drain()
        if case == "camara-desconectada":
            bank.check("error_fuente_explicito", "fake-M1-camara" in (pipe.error or ""))
            # Sin video fresco, el modo no debe fabricar una pérdida ni seguir guiando.
            bank.clock.advance(4_000_000_000)
            after = len(bank.sink.events)
            bank.controller.tick()
            bank.drain()
            bank.check(
                "sin_guia_ni_perdida_inventada",
                len(bank.sink.events) == after and not bank.has("buscar-perdido"),
            )
        elif case == "fuera-de-orden":
            bank.check(
                "identidad_incorrecta_rechazada",
                pipe.detector_status.errors == 1 and pipe.latest_seen() is None,
            )
            # Resultados válidos que terminan en orden inverso tampoco hacen retroceder
            # la vista pública del controlador.
            older, newer = source.read(), source.read()
            detector = ScriptedDetector(source)
            pipe.board.put_detection(detector.predict(newer), newer)
            before = pipe.latest_seen()
            pipe.board.put_detection(detector.predict(older), older)
            after = pipe.latest_seen()
            bank.check("vista_no_retrocede", before.frame.key == after.frame.key == newer.key)
            bank.log(
                "salidas-invertidas",
                {"nuevo": newer.frame_id, "tardio": older.frame_id, "vista": after.frame.frame_id},
            )
        else:
            bank.check(
                "inferencia_tardia_no_habla",
                bank.texts()[-1] == "sonido:cancelado" and not bank.has("buscar-adquirido"),
            )
    finally:
        release.set()
        pipe.stop()
