"""Un worker persistente: selección y OCR fuera del controlador; memoria acotada.

No se puede interrumpir una llamada nativa de ONNX. Cancelar descarta su resultado y
reemplaza el trabajo pendiente; nunca crea otro hilo ni una cola de inferencias viejas.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Full, Queue

import numpy as np

from visionx.reading.core import TextBlock, TextReader, sharpness


@dataclass(slots=True)
class ReadJob:
    request_id: int
    frames: Queue = field(default_factory=lambda: Queue(maxsize=1))
    complete: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: Future = field(default_factory=Future)
    selected_key: object = None
    selected_shape: tuple[int, ...] = ()
    sharpness: float = -1.0

    def offer(self, key: object, image: np.ndarray) -> None:
        # Solo hay un productor (controlador). Si selección tarda, conservar lo más nuevo.
        try:
            self.frames.put_nowait((key, image))
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait((key, image))


class ReadingWorker:
    def __init__(self, reader: TextReader) -> None:
        self.reader = reader
        self._cond = threading.Condition()
        self._pending: ReadJob | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="visionx-ocr", daemon=True)
        self._thread.start()

    def submit(self, job: ReadJob) -> None:
        with self._cond:
            if self._pending is not None:
                self._pending.cancelled.set()
            self._pending = job
            self._cond.notify()

    def close(self) -> None:
        # No join del OCR: cerrar la aplicación tampoco espera a una inferencia nativa.
        with self._cond:
            self._closed = True
            if self._pending is not None:
                self._pending.cancelled.set()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                job, self._pending = self._pending, None
            try:
                blocks = self._read(job)
                job.result.set_result(blocks)
            except Exception as exc:
                # La UI recibe el error en tick; el hilo queda disponible para otra orden.
                job.result.set_exception(exc)

    def _read(self, job: ReadJob) -> list[TextBlock]:
        best = None
        while not job.cancelled.is_set():
            try:
                key, image = job.frames.get(timeout=0.02)
            except Empty:
                if job.complete.is_set():
                    break
                continue
            score = sharpness(image)
            if score > job.sharpness:
                job.sharpness = score
                job.selected_key = key
                job.selected_shape = image.shape
                best = image
        if job.cancelled.is_set() or best is None:
            return []
        return self.reader.read(best)
