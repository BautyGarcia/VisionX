"""Motores locales de voz (T08): TTS con Piper, STT con faster-whisper y micrófono por botón.

Todos se importan recién al construirse (extra `voz`) y cargan modelos ya bajados con
`python -m visionx.voice descargar`: ninguno descarga solo ni necesita red al usarse.
"""

from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visionx.shared.config import ROOT
from visionx.shared.errors import EstimatorError, SourceError

# Voces de Piper fijadas a la versión v1.0.0 del repositorio (no a "main", que puede cambiar).
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
WHISPER_RATE = 16_000  # Whisper espera audio mono a 16 kHz


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    """Sección `audio` de la config (unidades en configs/default.yaml)."""

    tts_voice: str = "es_AR-daniela-high"
    voices_dir: str = "weights/piper"
    speech_rate: float = 1.0
    stt_model: str = "small"
    stt_models_dir: str = "weights/whisper"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_initial_prompt: str | None = None
    input_device: int | str | None = None
    output_device: int | str | None = None
    block_ms: float = 20.0
    max_record_s: float = 8.0
    deadlines_s: dict[str, float] = field(default_factory=dict)
    preload: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VoiceSettings:
        a = cfg.get("audio") or {}
        tts = a.get("tts") or {}
        stt = a.get("stt") or {}
        d = cls()
        return cls(
            tts_voice=str(tts.get("voice", d.tts_voice)),
            voices_dir=str(tts.get("voices_dir", d.voices_dir)),
            speech_rate=float(a.get("speech_rate", d.speech_rate)),
            stt_model=str(stt.get("model", d.stt_model)),
            stt_models_dir=str(stt.get("models_dir", d.stt_models_dir)),
            stt_device=str(stt.get("device", d.stt_device)),
            stt_compute_type=str(stt.get("compute_type", d.stt_compute_type)),
            stt_beam_size=int(stt.get("beam_size", d.stt_beam_size)),
            stt_initial_prompt=stt.get("initial_prompt"),
            input_device=a.get("input_device"),
            output_device=a.get("output_device"),
            block_ms=float(a.get("block_ms", d.block_ms)),
            max_record_s=float(a.get("max_record_s", d.max_record_s)),
            deadlines_s={str(k): float(v) for k, v in (a.get("deadlines_s") or {}).items()},
            preload=tuple(str(p) for p in (tts.get("preload") or ())),
        )

    @property
    def voice_path(self) -> Path:
        return _abs(self.voices_dir) / f"{self.tts_voice}.onnx"

    @property
    def stt_path(self) -> Path:
        return _abs(self.stt_models_dir) / self.stt_model


# --- descargas --------------------------------------------------------------------------


def _download(url: str, target: Path, min_bytes: int) -> Path:
    """Descarga a un temporal y renombra: un corte no deja un archivo a medias usable."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".parcial")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if partial.stat().st_size < min_bytes:
        partial.unlink()
        raise OSError(f"{url}: la descarga no parece un modelo")
    partial.replace(target)
    return target


def download_piper_voice(voice: str, folder: Path) -> list[Path]:
    """Baja `<voz>.onnx` y su `.onnx.json`. voice = idioma_PAÍS-nombre-calidad."""
    lang, name, quality = voice.split("-", 2)
    base = f"{PIPER_VOICES_URL}/{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}"
    return [
        _download(f"{base}.onnx", folder / f"{voice}.onnx", 1_000_000),
        _download(f"{base}.onnx.json", folder / f"{voice}.onnx.json", 100),
    ]


def download_whisper(model: str, folder: Path) -> Path:
    """Baja el modelo CTranslate2 de faster-whisper (Hugging Face Systran) a folder/model."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise EstimatorError("falta faster-whisper: `uv sync --extra voz`") from exc
    target = folder / model
    download_model(model, output_dir=str(target))
    return target


# --- TTS --------------------------------------------------------------------------------


class PiperSynth:
    """Texto -> audio con una voz de Piper (ONNX en CPU)."""

    def __init__(self, settings: VoiceSettings) -> None:
        path = settings.voice_path
        if not path.is_file():
            raise EstimatorError(
                f"no está la voz {path}: "
                f"`python -m visionx.voice descargar --voz {settings.tts_voice}`"
            )
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise EstimatorError("falta piper-tts: `uv sync --extra voz`") from exc
        self.voice_id = f"{settings.tts_voice}@{sha256_file(path)[:12]}"
        self._voice = PiperVoice.load(str(path))
        self.sample_rate = int(self._voice.config.sample_rate)
        # length_scale > 1 alarga los fonemas: hablar más lento es 1/velocidad.
        self._config = SynthesisConfig(length_scale=1.0 / settings.speech_rate)
        self.last_ms = 0.0

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        chunks = [c.audio_float_array for c in self._voice.synthesize(text, self._config)]
        self.last_ms = (time.perf_counter() - t0) * 1000
        if not chunks:
            return np.zeros(0, np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)


# --- STT --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    audio_s: float  # duración del audio recibido
    elapsed_ms: float  # tiempo de transcripción (sin grabar)
    language: str
    language_probability: float


class WhisperTranscriber:
    """Audio mono 16 kHz -> texto con faster-whisper (CTranslate2; CPU o CUDA, no MPS)."""

    def __init__(self, settings: VoiceSettings) -> None:
        path = settings.stt_path
        if not (path / "model.bin").is_file():
            raise EstimatorError(
                f"no está el modelo {path}: `python -m visionx.voice descargar --whisper "
                f"{settings.stt_model}`"
            )
        # Se carga de una carpeta local: sin consultas a Hugging Face al usarlo.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EstimatorError("falta faster-whisper: `uv sync --extra voz`") from exc
        self.settings = settings
        self.model_id = f"whisper-{settings.stt_model}@{sha256_file(path / 'model.bin')[:12]}"
        self._model = WhisperModel(
            str(path), device=settings.stt_device, compute_type=settings.stt_compute_type
        )

    def transcribe(self, samples: np.ndarray) -> Transcript:
        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language="es",
            beam_size=self.settings.stt_beam_size,
            # Órdenes cortas y sueltas: no condicionar en texto previo evita que una frase
            # anterior "contamine" la siguiente.
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.settings.stt_initial_prompt,
            vad_filter=False,  # el botón ya delimita la frase
        )
        text = " ".join(s.text.strip() for s in segments).strip()  # generador: acá transcribe
        return Transcript(
            text=text,
            audio_s=len(audio) / WHISPER_RATE,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            language=info.language,
            language_probability=float(info.language_probability),
        )


# --- micrófono --------------------------------------------------------------------------


def resample(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Remuestreo limitado en banda por FFT: descarta lo que está por encima de la nueva
    frecuencia de Nyquist en vez de doblarlo sobre la voz (aliasing), como haría una
    interpolación lineal al bajar de 22 050 o 48 000 Hz a 16 000 Hz. Pensado para órdenes de
    pocos segundos; para audio largo convendría un filtro polifásico."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if rate_in == rate_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = max(1, int(round(len(x) * rate_out / rate_in)))
    spectrum = np.fft.rfft(x)
    bins = n_out // 2 + 1
    if bins <= len(spectrum):
        spectrum = spectrum[:bins]
    else:
        spectrum = np.concatenate([spectrum, np.zeros(bins - len(spectrum), complex)])
    # irfft normaliza por n_out; el factor conserva la amplitud de la señal original.
    return (np.fft.irfft(spectrum, n_out) * (n_out / len(x))).astype(np.float32)


class PushToTalk:
    """Graba mientras el botón está apretado: start() ... stop() -> muestras a 16 kHz.

    El callback de PortAudio solo copia el bloque a una cola (TECNICO T08: nada pesado ahí).
    Si el dispositivo no acepta 16 kHz se graba a su frecuencia y se remuestrea al final.
    """

    def __init__(self, device: int | str | None = None, max_seconds: float = 8.0) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SourceError("falta sounddevice: `uv sync --extra voz`") from exc
        self._sd = sd
        self.device = device
        self.max_seconds = max_seconds
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self.rate = WHISPER_RATE
        self.overflows = 0
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:  # hilo de PortAudio
        if status.input_overflow:
            self.overflows += 1
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._queue = queue.Queue()
            for rate in (WHISPER_RATE, None):
                try:
                    if rate is None:
                        info = self._sd.query_devices(self.device, "input")
                        rate = int(info["default_samplerate"])
                    stream = self._sd.InputStream(
                        samplerate=rate,
                        channels=1,
                        dtype="float32",
                        device=self.device,
                        callback=self._callback,
                    )
                    break
                except self._sd.PortAudioError as exc:
                    last = exc
            else:
                raise SourceError(f"no se pudo abrir el micrófono: {last}")
            self.rate = rate
            self._stream = stream
            stream.start()

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                return np.zeros(0, np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None
        blocks = []
        while not self._queue.empty():
            blocks.append(self._queue.get_nowait())
        audio = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        audio = audio[: int(self.max_seconds * self.rate)]
        return resample(audio, self.rate, WHISPER_RATE)
