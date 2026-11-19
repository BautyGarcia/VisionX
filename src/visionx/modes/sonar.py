"""Sonar de la guía de buscar (T09, pedido de Tomás el 19/11).

En vez de frases de dirección que se cortan cuando la cámara gira, un bip corto en estéreo:

- **Lado:** suena del lado del objeto (paneo de potencia constante según el ángulo horizontal).
- **Centrado:** cuanto más cerca del centro, más seguido y más agudo; al frente, rápido y agudo.
- **Silencio** si no se lo ve hace más de `max_age_s` (la guía dice «Perdí…»).

Las palabras quedan para los momentos clave (encontrarlo, metros cada tanto, «Llegaste»). La
misma lógica arma los bips en vivo (`SonarPlayer`, un canal aparte de la voz) y en la demo.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SonarSettings:
    center_deg: float = 8.0  # |ángulo| <= esto: al frente (bip agudo y rápido)
    max_deg: float = 55.0  # ángulo de paneo total (borde de la imagen de la ELP)
    fast_s: float = 0.12  # intervalo entre bips al frente
    slow_s: float = 0.7  # intervalo con el objeto en el borde
    freq_center_hz: float = 1320.0
    freq_side_hz: float = 660.0
    beep_ms: float = 60.0
    volume: float = 0.35
    max_age_s: float = 1.0  # sin ver el objetivo durante esto: silencio


@dataclass(frozen=True, slots=True)
class Cue:
    t_s: float
    pan: float  # -1 izquierda, +1 derecha
    freq_hz: float


def cue_for(angle_h_deg: float, s: SonarSettings) -> tuple[float, float, float]:
    """(pan, frecuencia, intervalo) para un ángulo horizontal."""
    a = max(-s.max_deg, min(s.max_deg, angle_h_deg))
    pan = a / s.max_deg
    if abs(angle_h_deg) <= s.center_deg:
        return pan, s.freq_center_hz, s.fast_s
    k = min(1.0, (abs(a) - s.center_deg) / (s.max_deg - s.center_deg))  # 0 cerca, 1 en el borde
    freq = s.freq_center_hz + (s.freq_side_hz - s.freq_center_hz) * k
    return pan, freq, s.fast_s + (s.slow_s - s.fast_s) * (0.35 + 0.65 * k)


def beep(pan: float, freq_hz: float, s: SonarSettings, rate: int) -> np.ndarray:
    """Bip estéreo (N x 2) con rampa y paneo de potencia constante."""
    n = int(rate * s.beep_ms / 1000)
    t = np.arange(n, dtype=np.float32) / rate
    ramp = max(1, int(rate * 0.006))
    env = np.ones(n, np.float32)
    env[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
    env[-ramp:] = np.linspace(1, 0, ramp, dtype=np.float32)
    mono = s.volume * np.sin(2 * np.pi * freq_hz * t) * env
    theta = (pan + 1) * math.pi / 4  # 0 = izquierda, pi/2 = derecha
    return np.stack([mono * math.cos(theta), mono * math.sin(theta)], axis=1).astype(np.float32)


class Sonar:
    """Decide cuándo suena el próximo bip a partir del objetivo actual."""

    def __init__(self, settings: SonarSettings | None = None) -> None:
        self.s = settings or SonarSettings()
        self._next_s: float | None = None

    def step(self, now_s: float, target: tuple[float, float] | None) -> Cue | None:
        """`target` = (ángulo horizontal, segundos desde que se lo vio) o None."""
        if target is None or target[1] > self.s.max_age_s:
            self._next_s = None
            return None
        pan, freq, interval = cue_for(target[0], self.s)
        if self._next_s is not None and now_s < self._next_s:
            # Si se centró, no esperar el intervalo lento que venía de antes.
            self._next_s = min(self._next_s, now_s + interval)
            if now_s < self._next_s:
                return None
        self._next_s = now_s + interval
        return Cue(now_s, pan, freq)


def target_of(handler, now_ns: int) -> tuple[float, float] | None:
    """(ángulo, edad en s) del objetivo que se está guiando, o None."""
    get = getattr(handler, "sonar_target", None)
    return get(now_ns) if get is not None else None


class SonarPlayer:
    """En vivo: un hilo que mira el modo buscar y toca los bips en su propio stream estéreo."""

    def __init__(
        self,
        controller,
        handler,
        settings: SonarSettings | None = None,
        device: int | str | None = None,
        rate: int = 24_000,
    ) -> None:
        self.controller, self.handler = controller, handler
        self.sonar = Sonar(settings)
        self.rate, self.device = rate, device
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="visionx-sonar", daemon=True)

    def start(self) -> SonarPlayer:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        import sounddevice as sd

        with sd.OutputStream(samplerate=self.rate, channels=2, dtype="float32",
                             device=self.device, latency="low") as stream:  # fmt: skip
            while not self._stop.is_set():
                now = time.monotonic_ns()
                active = self.controller.active
                guiding = active is not None and active.mode.value == "search"
                cue = self.sonar.step(now / 1e9, target_of(self.handler, now) if guiding else None)
                if cue is not None:
                    stream.write(beep(cue.pan, cue.freq_hz, self.sonar.s, self.rate))
                else:
                    time.sleep(0.01)
