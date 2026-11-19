"""Selección continua y OCR serializados en un worker; memoria y colas acotadas.

Cancelar descarta el resultado de ONNX, que no se puede interrumpir. La carga inicial,
la nitidez y las inferencias comparten hilo: una orden nunca duplica la carga.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Full, Queue

import numpy as np

from visionx.reading.core import ReadingSettings, TextBlock, TextReader, sharpness


@dataclass(slots=True)
class ReadJob:
    request_id: int
    earliest_ns: int = 0
    end_ns: int = 0
    complete: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: Future = field(default_factory=Future)
    selected_key: object = None
    selected_shape: tuple[int, ...] = ()
    selected_image: object = None  # la imagen leída: el asistente la mira junto con el OCR
    sharpness: float = -1.0
    early: bool = False
    latest: tuple | None = None


class ReadingWorker:
    def __init__(
        self, reader: TextReader, settings: ReadingSettings, *, prewarm: bool = False
    ) -> None:
        self.reader, self.settings = reader, settings
        self.ready = threading.Event()
        self._warmup_error: Exception | None = None
        self._prewarm = prewarm
        self._cond = threading.Condition()
        self._pending: ReadJob | None = None
        self._closed = False
        self._frames: Queue = Queue(maxsize=1)
        # Máximo 32 izquierdas, además de selección/OCR, buzón y captura del pedido.
        # Se purgan por tiempo y contexto; no conservar una escena/sesión vieja.
        self._history: deque = deque(maxlen=32)
        self._thread = threading.Thread(target=self._run, name="visionx-ocr", daemon=True)
        self._thread.start()

    def offer(self, key, image: np.ndarray, received_ns: int, context, object_box=None) -> None:
        # Solo un productor (controlador), sin copia de píxeles ni cálculo bajo su lock.
        try:
            self._frames.put_nowait((received_ns, key, image, context, object_box))
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            self._frames.put_nowait((received_ns, key, image, context, object_box))
        with self._cond:
            self._cond.notify()

    def submit(self, job: ReadJob) -> None:
        with self._cond:
            if self._pending is not None:
                self._pending.cancelled.set()
            self._pending = job
            self._cond.notify()

    def close(self) -> None:
        # No join del OCR: tampoco esperar una carga/inferencia nativa al cerrar.
        with self._cond:
            self._closed = True
            if self._pending is not None:
                self._pending.cancelled.set()
            self._cond.notify()

    def _capture(self) -> None:
        try:
            received, key, image, context, box = self._frames.get_nowait()
        except Empty:
            return
        if self._history and (context != self._history[-1][3] or received < self._history[-1][0]):
            self._history.clear()
        cutoff = received - int(self.settings.pre_s * 1e9)
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        score = sharpness(image)
        # Un frame más nuevo y más nítido domina al anterior incluso al expirar la ventana.
        # Cola monótona: conservar el máximo móvil sin guardar 32 imágenes por rutina.
        while self._history and self._history[-1][5] <= score:
            self._history.pop()
        self._history.append((received, key, image, context, box, score))

    def _run(self) -> None:
        try:
            warmup = getattr(self.reader, "warmup", None)
            if self._prewarm and warmup is not None:
                warmup()
        except Exception as exc:
            self._warmup_error = exc
        finally:
            self.ready.set()
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._closed or self._pending is not None or not self._frames.empty()
                )
                if self._closed:
                    return
                job, self._pending = self._pending, None
            self._capture()
            if job is None:
                continue
            try:
                if self._warmup_error is not None:
                    raise self._warmup_error
                job.result.set_result(self._read(job))
            except Exception as exc:
                job.result.set_exception(exc)

    def _read(self, job: ReadJob) -> list[TextBlock]:
        best = best_box = None
        seen = set()
        last_context = None
        while not job.cancelled.is_set() and not self._closed:
            self._capture()
            candidates = list(self._history)
            latest = job.latest
            # Si warmup dura más que la ventana, el buzón global puede estar fuera de ella.
            # El pedido conserva su última captura elegible; se calcula aquí, nunca en UI.
            if latest is not None and latest[1] not in seen:
                candidates.append((*latest, sharpness(latest[2])))
            for received, key, image, context, box, score in candidates:
                if key in seen or not job.earliest_ns <= received <= job.end_ns:
                    continue
                if context != last_context:
                    last_context = context
                    best = best_box = None
                    job.sharpness, job.selected_key = -1.0, None
                    seen.clear()
                seen.add(key)
                if score > job.sharpness:
                    job.sharpness, job.selected_key = score, key
                    job.selected_shape = image.shape
                    best, best_box = image, box
            # Umbral absoluto en la MISMA ruta/resolución; 0 deshabilita el corte.
            if best is not None and self.settings.early_sharpness > 0:
                if job.sharpness >= self.settings.early_sharpness:
                    job.early = True
                    job.complete.set()
            if job.complete.is_set():
                break
            with self._cond:
                self._cond.wait(timeout=0.02)
        if job.cancelled.is_set() or self._closed or best is None:
            return []
        job.selected_image = best
        read_region = getattr(self.reader, "read_region", None)
        return read_region(best, best_box) if read_region else self.reader.read(best)
