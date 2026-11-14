"""Salida de audio con prioridad, vencimiento y cancelación (T08). Cumple la interfaz AudioSink.

Un hilo propio sintetiza y reproduce los AudioEvent de a uno:
- El de mayor prioridad sale primero; a igual prioridad, el más viejo.
- Un evento más prioritario que el que está sonando lo corta si el actual es cancelable
  (una alerta interrumpe una descripción). Lo cortado no se retoma: sería información vieja.
- Antes de sonar se revisa el deadline con el reloj elegido: lo vencido no se reproduce.
- cancel(request_id) saca lo pendiente de ese pedido y corta lo que suena. El corte se revisa
  entre bloques de `block_ms`: esa es la demora lógica máxima, más el búfer del dispositivo.

Cada evento deja un PlaybackRecord con su resultado y tiempos, para medir latencias (M1/M10).
El callback del dispositivo nunca sintetiza ni bloquea: solo recibe bloques ya calculados.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import AudioEvent, check_schema
from visionx.shared.errors import InvalidInput

SOUND_PREFIX = "sonido:"


class Synth(Protocol):
    """Texto -> muestras float32 mono en [-1, 1] a `sample_rate` Hz."""

    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


class Player(Protocol):
    """Reproduce muestras; devuelve False si `stop` cortó la reproducción antes del final."""

    def play(self, samples: np.ndarray, sample_rate: int, stop: threading.Event) -> bool: ...

    def close(self) -> None: ...


# --- sonidos ----------------------------------------------------------------------------


def tone(
    freqs_hz: Iterable[float], ms: float, sample_rate: int, volume: float = 0.4, gap_ms: float = 40
) -> np.ndarray:
    """Secuencia de tonos senoidales con rampa de 5 ms (sin rampa se oye un clic)."""
    parts = []
    n = int(sample_rate * ms / 1000)
    ramp = min(int(sample_rate * 0.005), n // 2)
    envelope = np.ones(n, np.float32)
    if ramp:
        envelope[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
        envelope[-ramp:] = np.linspace(1, 0, ramp, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    silence = np.zeros(int(sample_rate * gap_ms / 1000), np.float32)
    for i, f in enumerate(freqs_hz):
        if i:
            parts.append(silence)
        parts.append((volume * np.sin(2 * np.pi * f * t) * envelope).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


# Sonidos con nombre: `sonido:bip` en AudioEvent.text_or_sound. Frecuencias en Hz, duración en ms.
SOUNDS: dict[str, tuple[tuple[float, ...], float]] = {
    "bip": ((880.0,), 120),  # objeto buscado a la vista
    "bip-agudo": ((1320.0,), 80),
    "bip-grave": ((440.0,), 150),
    "ok": ((660.0, 990.0), 90),  # orden entendida
    "error": ((660.0, 440.0), 140),  # no se entendió / falló
    "cancelado": ((520.0,), 70),
}


def render_sound(name: str, sample_rate: int) -> np.ndarray:
    if name not in SOUNDS:
        raise InvalidInput(f"sonido desconocido {name!r}; hay {sorted(SOUNDS)}")
    freqs, ms = SOUNDS[name]
    return tone(freqs, ms, sample_rate)


class CachedSynth:
    """Guarda las últimas frases sintetizadas: las frecuentes ("Busco la taza.") no se vuelven a
    calcular. preload() las prepara al arrancar, cuando nadie está esperando."""

    def __init__(self, synth: Synth, max_items: int = 64) -> None:
        self.synth = synth
        self.sample_rate = synth.sample_rate
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                self.hits += 1
                return self._cache[text]
        samples = self.synth.synthesize(text)  # fuera del lock: puede tardar cientos de ms
        samples.setflags(write=False)  # compartido entre reproducciones: nadie lo modifica
        with self._lock:
            self.misses += 1
            self._cache[text] = samples
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return samples

    def preload(self, phrases: Iterable[str]) -> None:
        for phrase in phrases:
            self.synthesize(phrase)


# --- registro ---------------------------------------------------------------------------


class Outcome(StrEnum):
    PLAYED = "reproducido"
    CANCELLED = "cancelado"  # cancel() del pedido o cancel_all()
    PREEMPTED = "desplazado"  # lo cortó o lo sacó de la cola algo más prioritario
    EXPIRED = "vencido"  # pasó su deadline antes de sonar
    DROPPED = "descartado"  # la cola estaba llena
    FAILED = "error"  # falló la síntesis o el dispositivo


@dataclass(slots=True)
class PlaybackRecord:
    request_id: int
    priority: int
    text_or_sound: str
    submitted_ns: int
    outcome: Outcome | None = None
    started_ns: int | None = None  # empezó a sonar (después de sintetizar)
    ended_ns: int | None = None
    error: str | None = None

    @property
    def wait_ms(self) -> float | None:
        """Desde submit() hasta que empezó a sonar: cola + síntesis."""
        if self.started_ns is None:
            return None
        return (self.started_ns - self.submitted_ns) / 1e6


@dataclass(order=True)
class _Queued:
    sort_key: tuple[int, int]  # (-prioridad, orden de llegada)
    event: AudioEvent = None  # type: ignore[assignment]
    record: PlaybackRecord = None  # type: ignore[assignment]


class AudioOutput:
    """AudioSink real: un hilo, una cola con prioridad y un reproductor intercambiable."""

    def __init__(
        self,
        synth: Synth,
        player: Player,
        *,
        clock: Clock | None = None,
        max_queue: int = 16,
        on_record: Callable[[PlaybackRecord], None] | None = None,
    ) -> None:
        self.synth = synth
        self.player = player
        self.clock = clock or MonotonicClock()
        self.max_queue = max_queue
        self.records: list[PlaybackRecord] = []
        self._on_record = on_record
        self._queue: list[_Queued] = []
        self._order = itertools.count()
        self._cond = threading.Condition()
        self._current: _Queued | None = None
        self._stop_current = threading.Event()
        self._current_outcome: Outcome | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="visionx-audio", daemon=True)
        self._thread.start()

    # -- API AudioSink -------------------------------------------------------------------

    def submit(self, event: AudioEvent) -> None:
        check_schema(event)
        now = self.clock.now_ns()
        record = PlaybackRecord(
            event.request_id, int(event.priority), event.text_or_sound, submitted_ns=now
        )
        with self._cond:
            if self._closed:
                raise RuntimeError("AudioOutput cerrado")
            self.records.append(record)
            if event.deadline_ns is not None and now > event.deadline_ns:
                self._finish(record, Outcome.EXPIRED, now)
                return
            heapq.heappush(
                self._queue, _Queued((-event.priority, next(self._order)), event, record)
            )
            current = self._current
            if current is not None and current.event.cancelable:
                if event.priority > current.event.priority:
                    self._interrupt(Outcome.PREEMPTED)
            while len(self._queue) > self.max_queue:
                # Se descarta el menos prioritario y, a igualdad, el más nuevo.
                worst = max(self._queue)
                self._queue.remove(worst)
                heapq.heapify(self._queue)
                self._finish(worst.record, Outcome.DROPPED, now)
            self._cond.notify()

    def cancel(self, request_id: int) -> None:
        """Descarta lo pendiente de ese pedido y corta lo que suena, si es cancelable."""
        self._cancel(lambda e: e.request_id == request_id)

    def cancel_all(self) -> None:
        """El botón de cancelar: corta todo lo cancelable, sin depender de reconocer una palabra."""
        self._cancel(lambda e: True)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._interrupt(Outcome.CANCELLED)
            self._cond.notify()
        self._thread.join(timeout=5)
        self.player.close()

    def wait_idle(self, timeout_s: float = 10.0) -> bool:
        """Espera a que no quede nada pendiente ni sonando (para CLI y tests)."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._queue and self._current is None, timeout=timeout_s
            )

    # -- interno ---------------------------------------------------------------------------

    def _cancel(self, match: Callable[[AudioEvent], bool]) -> None:
        now = self.clock.now_ns()
        with self._cond:
            keep = []
            for item in self._queue:
                if item.event.cancelable and match(item.event):
                    self._finish(item.record, Outcome.CANCELLED, now)
                else:
                    keep.append(item)
            self._queue = keep
            heapq.heapify(self._queue)
            current = self._current
            if current is not None and current.event.cancelable and match(current.event):
                self._interrupt(Outcome.CANCELLED)

    def _interrupt(self, outcome: Outcome) -> None:
        """Pide cortar lo que suena (llamar con el lock tomado)."""
        if self._current is not None and self._current_outcome is None:
            self._current_outcome = outcome
            self._stop_current.set()

    def _finish(self, record: PlaybackRecord, outcome: Outcome, now: int, error: str | None = None):
        record.outcome = outcome
        record.ended_ns = now
        record.error = error
        if self._on_record is not None:
            self._on_record(record)

    def _render(self, event: AudioEvent) -> tuple[np.ndarray, int]:
        text = event.text_or_sound
        rate = self.synth.sample_rate
        if text.startswith(SOUND_PREFIX):
            return render_sound(text[len(SOUND_PREFIX) :], rate), rate
        return self.synth.synthesize(text), rate

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    now = self.clock.now_ns()
                    for item in self._queue:
                        self._finish(item.record, Outcome.CANCELLED, now)
                    self._queue.clear()
                    self._cond.notify_all()
                    return
                item = heapq.heappop(self._queue)
                self._current = item
                self._current_outcome = None
                self._stop_current.clear()
            outcome, error = self._play(item)
            with self._cond:
                self._finish(item.record, outcome, self.clock.now_ns(), error)
                self._current = None
                self._cond.notify_all()

    def _play(self, item: _Queued) -> tuple[Outcome, str | None]:
        event, record = item.event, item.record
        try:
            samples, rate = self._render(event)
        except Exception as exc:  # una frase que no se puede sintetizar no tumba la salida
            return Outcome.FAILED, f"síntesis: {exc}"
        with self._cond:
            # La síntesis tarda: mientras tanto pudo llegar un cancel o vencer el pedido.
            if self._current_outcome is not None:
                return self._current_outcome, None
            now = self.clock.now_ns()
            if event.deadline_ns is not None and now > event.deadline_ns:
                return Outcome.EXPIRED, None
            record.started_ns = now
        try:
            completed = self.player.play(samples, rate, self._stop_current)
        except Exception as exc:
            return Outcome.FAILED, f"dispositivo: {exc}"
        with self._cond:
            if completed:
                return Outcome.PLAYED, None
            return self._current_outcome or Outcome.CANCELLED, None


# --- reproductores ----------------------------------------------------------------------


class FakePlayer:
    """Reproductor sin dispositivo. Con `realtime` tarda lo que duraría el audio (en bloques,
    mirando `stop`); si no, termina al instante. Guarda qué reprodujo y si lo cortaron."""

    def __init__(self, *, realtime: bool = False, block_ms: float = 10.0, speed: float = 1.0):
        self.realtime = realtime
        self.block_ms = block_ms
        self.speed = speed
        self.played: list[tuple[int, int, bool]] = []  # (muestras, Hz, completo)
        self.started = threading.Event()
        self.closed = False

    def play(self, samples: np.ndarray, sample_rate: int, stop: threading.Event) -> bool:
        self.started.set()
        completed = True
        if self.realtime:
            duration_s = len(samples) / sample_rate / self.speed
            block_s = self.block_ms / 1000
            elapsed = 0.0
            while elapsed < duration_s:
                if stop.wait(block_s):
                    completed = False
                    break
                elapsed += block_s
        elif stop.is_set():
            completed = False
        self.played.append((len(samples), sample_rate, completed))
        return completed

    def close(self) -> None:
        self.closed = True


class SoundDevicePlayer:
    """Parlantes/auriculares con sounddevice (PortAudio). Escribe bloques de `block_ms` y
    revisa `stop` entre bloques. Al cortar usa abort(), que descarta lo que quedó en el búfer
    del dispositivo en vez de dejarlo sonar."""

    def __init__(self, device: int | str | None = None, block_ms: float = 20.0) -> None:
        import sounddevice as sd

        self._sd = sd
        self.device = device
        self.block_ms = block_ms
        self._stream = None
        self._rate: int | None = None

    def _open(self, sample_rate: int) -> None:
        if self._stream is not None and self._rate == sample_rate:
            return
        self.close()
        block = max(1, int(sample_rate * self.block_ms / 1000))
        self._stream = self._sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=block,
            latency="low",
        )
        self._rate = sample_rate

    def play(self, samples: np.ndarray, sample_rate: int, stop: threading.Event) -> bool:
        self._open(sample_rate)
        stream = self._stream
        block = max(1, int(sample_rate * self.block_ms / 1000))
        data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1)
        stream.start()
        try:
            for start in range(0, len(data), block):
                if stop.is_set():
                    stream.abort()
                    return False
                stream.write(data[start : start + block])
            stream.stop()  # espera a que termine de sonar lo escrito
            return True
        except BaseException:
            stream.abort()
            raise

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._rate = None
