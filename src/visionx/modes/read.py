"""Modo LEER beta: seleccionar izquierda nítida, OCR asíncrono y bloques cancelables."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from visionx.app.controller import Say
from visionx.app.runtime import View
from visionx.assistant import Delivery
from visionx.modes.common import ModeSettings, closing_due, closing_text, speech_s
from visionx.reading import RapidReader, ReadingSettings, TextReader, reading_order
from visionx.reading.core import framing_hint, spoken_chunks
from visionx.reading.worker import ReadingWorker, ReadJob
from visionx.shared.contracts import Command, Priority, StereoFrame
from visionx.voice.commands import READ_ALL


class ReadHandler:
    def __init__(
        self,
        reader: TextReader | None = None,
        settings: ReadingSettings | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        prewarm: bool = False,
        mode_settings: ModeSettings | None = None,
        assistant: Any = None,
    ) -> None:
        self.s = settings or ReadingSettings()
        # Asistente (Claude): con el OCR de la imagen elegida dice qué es el objeto y lo más
        # importante que dice («Adelante tenés una caja de TITA…»). «Leé todo» y cualquier falla
        # del asistente usan la lectura por bloques de siempre.
        self.assistant = assistant
        self._delivery: Any = None
        self._fallback: list[str] = []
        self._timing = replace(
            mode_settings or ModeSettings(),
            speech_base_s=self.s.speech_base_s,
            speech_s_per_char=self.s.speech_s_per_char,
        )
        self.reader = reader if reader is not None else RapidReader(self.s)
        self._event = on_event or (lambda kind, data: None)
        self._worker: ReadingWorker | None = (
            ReadingWorker(self.reader, self.s, prewarm=True) if prewarm else None
        )
        self._job: ReadJob | None = None
        self._chunks: deque[str] = deque()
        self._last_key = None
        self._loaded = False
        self._next_say_ns = 0

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self.cancel()
        self._job = ReadJob(
            command.request_id,
            earliest_ns=max(
                command.issued_ns - int(self.s.pre_s * 1e9),
                now_ns - int(self.s.max_age_s * 1e9),
            ),
            end_ns=command.issued_ns + int(self.s.window_s * 1e9),
        )
        self._loaded = False
        self._next_say_ns = now_ns
        self._last_key = None  # el pedido conserva también la captura actual durante warmup
        if self._worker is None:
            self._worker = ReadingWorker(self.reader, self.s)
        self.observe(view, now_ns)
        self._worker.submit(self._job)
        # Controller ya confirma READ con «Leo.» (precacheada por T08).
        return []

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled.set()
        self._chunks.clear()
        if self._delivery is not None:
            self._delivery.job.cancel()
            self._delivery = None

    def close(self) -> None:
        self.cancel()
        if self._worker is not None:
            self._worker.close()

    def tick(self, command: Command, view: View, now_ns: int) -> tuple[list[Say], bool]:
        job = self._job
        if job is None or job.cancelled.is_set() or command.request_id != job.request_id:
            return [], True
        if command.deadline_ns is not None and now_ns >= command.deadline_ns:
            self.cancel()
            return [], True
        if self._delivery is not None:
            return self._deliver(command, now_ns)
        closing = closing_text(
            command,
            command.issued_ns,
            [
                "No terminé de leer a tiempo.",
                "No pude leer.",
            ],
            self._timing,
        )
        if closing_due(command, now_ns, closing, self._timing) and now_ns >= self._next_say_ns:
            self.cancel()
            return self._finish(closing, command, now_ns)
        end_ns = command.issued_ns + int(self.s.window_s * 1e9)
        if not job.complete.is_set():
            self.observe(view, now_ns)
        if now_ns >= end_ns:
            job.complete.set()
        if not self._loaded:
            if not job.result.done():
                return [], False  # nunca esperar un Future bajo el lock del controlador
            self._loaded = True
            try:
                raw = job.result.result()
            except Exception as exc:
                self._event("leer-error", {"request_id": job.request_id, "error": str(exc)})
                return self._finish("No pude leer. Probá de nuevo.", command, now_ns)
            ordered = reading_order(raw, self.s)
            self._event(
                "leer-resultado",
                {
                    "request_id": job.request_id,
                    "frame_key": str(job.selected_key),
                    "ruta": self.s.image_source,
                    "nitidez": job.sharpness,
                    "corte_temprano": job.early,
                    "bloques": len(ordered),
                    "modelo": self.reader.model_id,
                },
            )
            if (
                command.target != READ_ALL
                and self.assistant is not None
                and self.assistant.available
                and job.selected_image is not None
            ):
                lines = [b.text for b in ordered]
                fallback = spoken_chunks(ordered, self.s) if ordered else []
                if not ordered:
                    hint = (
                        framing_hint(raw, job.selected_shape, self.s) if job.selected_shape else ""
                    )
                    fallback = ["No encuentro texto." + (f" {hint}" if hint else "")]
                kind = "leer"
                ajob = self.assistant.submit(job.request_id, kind, job.selected_image, ocr=lines)
                self._delivery = Delivery(ajob, now_ns)
                self._fallback = fallback
                self._event("asistente-pedido", {"request_id": job.request_id, "tipo": kind,
                                                 "ocr": lines})  # fmt: skip
                return [], False
            if not ordered:
                hint = framing_hint(raw, job.selected_shape, self.s) if job.selected_shape else ""
                return self._finish(
                    "No encuentro texto." + (f" {hint}" if hint else ""), command, now_ns
                )
            self._chunks.extend(spoken_chunks(ordered, self.s))
            if not self._chunks:
                return self._finish(
                    "No pude leer con suficiente confianza. Probá acercarlo.", command, now_ns
                )
        if now_ns < self._next_say_ns:
            return [], False
        if not self._chunks:
            return [], True
        text = self._chunks.popleft()
        # No empezar un bloque que el deadline cortaría; dejar lugar para avisarlo.
        if (
            command.deadline_ns is not None
            and now_ns
            + int(
                (
                    speech_s(text, self._timing)
                    + speech_s(closing, self._timing)
                    + self._timing.end_margin_s
                )
                * 1e9
            )
            > command.deadline_ns
        ):
            self.cancel()
            return self._finish(closing, command, now_ns)
        # Como T09: espaciar por duración estimada evita llenar la cola y perder leyendas.
        # Es una estimación de Piper es_AR, no latencia audible medida en esta PC.
        self._next_say_ns = now_ns + int(
            (self.s.speech_base_s + len(text) * self.s.speech_s_per_char) * 1e9
        )
        return [Say(text, Priority.LONG)], False

    def _deliver(self, command: Command, now_ns: int) -> tuple[list[Say], bool]:
        d = self._delivery
        if d.failed:
            # Sin asistente (red, tope de gasto): lectura por bloques de siempre.
            self._delivery = None
            self._event("asistente-error", {"request_id": d.job.request_id, "error": d.job.error})
            self._chunks.extend(self._fallback)
            if not self._chunks:
                return self._finish("No pude leer. Probá de nuevo.", command, now_ns)
            return [], False
        says = [Say(s, Priority.LONG) for s in d.ready(now_ns)]
        if d.finished:
            self._delivery = None
            job = d.job
            self._event(
                "asistente-respuesta",
                {
                    "request_id": job.request_id,
                    "tipo": job.kind,
                    "texto": job.text,
                    "primer_texto_s": job.first_s,
                    "total_s": job.total_s,
                    "usd": job.usd,
                    "citas_no_confirmadas": job.unconfirmed_quotes,
                    "ocr": job.ocr,
                    "error": job.error,
                },
            )
            return says, True
        return says, False

    def _finish(self, text: str, command: Command, now_ns: int) -> tuple[list[Say], bool]:
        closing = closing_text(
            command, now_ns, [text, "No pude leer."], self._timing, reserve=False
        )
        return ([Say(closing)] if closing else []), True

    def _frame(self, view: View) -> StereoFrame | None:
        if view is None:
            return None
        if self.s.image_source == "original":
            if view.original is not None:
                return view.original
            # Nunca llamar «original» a un frame rectificado cuando no se lo conservó.
            fallback = view.captured or (view.seen.frame if view.seen else None)
            return fallback if fallback is not None and not fallback.rectified else None
        fallback = view.captured or (view.seen.frame if view.seen else None)
        return fallback if fallback is not None and fallback.rectified else None

    def observe(self, view: View, now_ns: int) -> None:
        """Buzón de captura aun sin orden; el worker calcula y conserva la nitidez."""
        frame = self._frame(view)
        if frame is None:
            return
        if frame.received_ns > now_ns:
            return
        if now_ns - frame.received_ns > self.s.max_age_s * 1e9 or frame.key == self._last_key:
            return
        self._last_key = frame.key
        object_box = None
        if self.s.crop == "object" and view.seen is not None:
            detection = view.seen.detection
            # Igual identidad no basta para aplicar cajas rectificadas sobre la original.
            if (
                detection.key == frame.key
                and detection.geometry_id == frame.geometry_id
                and detection.calibration_id == frame.calibration_id
                and detection.image_size == (frame.left.shape[1], frame.left.shape[0])
            ):
                w, h = detection.image_size
                valid = [b for b in detection.boxes if b[2] > b[0] and b[3] > b[1]]
                if valid:
                    box = min(
                        valid,
                        key=lambda b: (
                            (((b[0] + b[2]) / 2 - w / 2) / w) ** 2
                            + (((b[1] + b[3]) / 2 - h / 2) / h) ** 2
                        ),
                    )
                    object_box = tuple(float(v) for v in box)
        if self._worker is None:
            self._worker = ReadingWorker(self.reader, self.s)
        candidate = (
            frame.received_ns,
            frame.key,
            frame.left,
            (frame.session_id, frame.geometry_id, frame.calibration_id, frame.left.shape),
            object_box,
        )
        job = self._job
        if job is not None and not job.cancelled.is_set() and not job.complete.is_set():
            if job.earliest_ns <= frame.received_ns <= job.end_ns:
                job.latest = candidate
        self._worker.offer(
            frame.key,
            frame.left,
            frame.received_ns,
            candidate[3],
            object_box,
        )
