"""Modo LEER beta: seleccionar izquierda nítida, OCR asíncrono y bloques cancelables."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from visionx.app.controller import Say
from visionx.app.runtime import View
from visionx.reading import RapidReader, ReadingSettings, TextReader, reading_order
from visionx.reading.core import framing_hint, speech_chunks
from visionx.reading.worker import ReadingWorker, ReadJob
from visionx.shared.contracts import Command, Priority, StereoFrame


class ReadHandler:
    def __init__(
        self,
        reader: TextReader | None = None,
        settings: ReadingSettings | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.s = settings or ReadingSettings()
        self.reader = reader if reader is not None else RapidReader(self.s)
        self._event = on_event or (lambda kind, data: None)
        self._worker: ReadingWorker | None = None  # ni hilo ni modelos antes de la primera orden
        self._job: ReadJob | None = None
        self._chunks: deque[str] = deque()
        self._last_key = None
        self._loaded = False
        self._next_say_ns = 0

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self.cancel()
        self._job = ReadJob(command.request_id)
        self._loaded = False
        self._last_key = None
        self._next_say_ns = now_ns
        if self._worker is None:
            self._worker = ReadingWorker(self.reader)
        self._worker.submit(self._job)
        self._observe(command, view, now_ns)
        # Controller ya confirma READ con «Leo.» (precacheada por T08).
        return []

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled.set()
        self._chunks.clear()

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
        end_ns = command.issued_ns + int(self.s.window_s * 1e9)
        self._observe(command, view, now_ns)
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
                return [Say("No pude leer. Probá de nuevo.")], True
            ordered = reading_order(raw, self.s)
            self._event(
                "leer-resultado",
                {
                    "request_id": job.request_id,
                    "frame_key": str(job.selected_key),
                    "ruta": self.s.image_source,
                    "nitidez": job.sharpness,
                    "bloques": len(ordered),
                    "modelo": self.reader.model_id,
                },
            )
            if not ordered:
                hint = framing_hint(raw, job.selected_shape, self.s) if job.selected_shape else ""
                return [Say("No encuentro texto." + (f" {hint}" if hint else ""))], True
            self._chunks.extend(speech_chunks(ordered, self.s.max_chars))
        if now_ns < self._next_say_ns:
            return [], False
        if not self._chunks:
            return [], True
        text = self._chunks.popleft()
        # Como T09: espaciar por duración estimada evita llenar la cola y perder leyendas.
        # Es una estimación de Piper es_AR, no latencia audible medida en esta PC.
        self._next_say_ns = now_ns + int(
            (self.s.speech_base_s + len(text) * self.s.speech_s_per_char) * 1e9
        )
        return [Say(text, Priority.LONG)], False

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

    def _observe(self, command: Command, view: View, now_ns: int) -> None:
        frame = self._frame(view)
        if frame is None or self._job is None or self._job.complete.is_set():
            return
        end_ns = command.issued_ns + int(self.s.window_s * 1e9)
        if not command.issued_ns <= frame.received_ns <= min(now_ns, end_ns):
            return
        if now_ns - frame.received_ns > self.s.max_age_s * 1e9 or frame.key == self._last_key:
            return
        self._last_key = frame.key
        self._job.offer(frame.key, frame.left)
