"""Motores de voz (T08): TTS con Edge o Piper, STT con faster-whisper y micrófono por botón.

Todos se importan recién al construirse (extra `voz`). Piper y Whisper cargan modelos ya bajados
con `python -m visionx.voice descargar` y no necesitan red. La voz por defecto (`audio.tts.engine:
edge`, es-UY-MateoNeural, elegida por Tomás el 18/11) es una voz neuronal de Microsoft Edge que
SÍ usa internet: el texto dicho sale de la máquina. Sin red, la app sigue con Piper sola
(FallbackSynth) y lo ya dicho alguna vez sale de la caché en disco.
"""

from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visionx.shared.config import ROOT
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError

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

    tts_voice: str = "es_ES-davefx-medium"  # Piper: voz única con engine piper; respaldo con edge
    voices_dir: str = "weights/piper"
    tts_engine: str = "piper"  # piper | edge
    edge_voice: str = "es-UY-MateoNeural"
    edge_timeout_s: float = 4.0
    edge_retry_s: float = 30.0
    tts_cache_dir: str | None = "data/cache/voz"
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
            tts_engine=str(tts.get("engine", d.tts_engine)),
            edge_voice=str(tts.get("edge_voice", d.edge_voice)),
            edge_timeout_s=float(tts.get("edge_timeout_s", d.edge_timeout_s)),
            edge_retry_s=float(tts.get("edge_retry_s", d.edge_retry_s)),
            tts_cache_dir=tts.get("cache_dir", d.tts_cache_dir) or None,
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


EDGE_RATE = 24_000  # Edge entrega MP3 mono a 24 kHz


def edge_rate(speech_rate: float) -> str:
    """speech_rate (1.0 = normal) -> el formato de Edge: "+0%", "-10%", "+15%"."""
    pct = int(round((speech_rate - 1.0) * 100))
    return f"{pct:+d}%"


def decode_mp3(data: bytes, rate: int = EDGE_RATE) -> np.ndarray:
    """MP3 en memoria -> float32 mono a `rate` Hz, con PyAV (ya viene con faster-whisper)."""
    import io

    import av

    parts = []
    with av.open(io.BytesIO(data)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in container.decode(audio=0):
            parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def trim_silence(
    samples: np.ndarray, rate: int, threshold: float = 0.01, keep_start_ms=30, keep_end_ms=120
) -> np.ndarray:
    """Recortar el silencio de los bordes. Edge entrega ~150 ms antes de hablar y 0,8-1,2 s
    después (medido el 18/11): demora la primera palabra y ocupa el parlante sin decir nada."""
    loud = np.nonzero(np.abs(samples) > threshold)[0]
    if loud.size == 0:
        return samples
    start = max(0, int(loud[0]) - int(rate * keep_start_ms / 1000))
    end = min(len(samples), int(loud[-1]) + 1 + int(rate * keep_end_ms / 1000))
    return samples[start:end]


def _edge_fetch(text: str, voice: str, rate: str, timeout_s: float) -> bytes:
    """Pedir el MP3 a Edge. Corre su propio bucle asyncio (se llama desde hilos sin bucle)."""
    import asyncio

    try:
        import edge_tts
    except ImportError as exc:
        raise EstimatorError("falta edge-tts: `uv sync --extra voz`") from exc

    async def collect() -> bytes:
        out = bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
            if chunk["type"] == "audio":
                out += chunk["data"]
        return bytes(out)

    return asyncio.run(asyncio.wait_for(collect(), timeout_s))


class EdgeSynth:
    """Texto -> audio con una voz neuronal de Microsoft Edge (edge-tts). NECESITA INTERNET.

    Medido en la PC de escritorio el 18/11: ~0,9-1,3 s hasta el primer audio (Piper: ~0,15 s).
    `fetch` se inyecta en los tests para no salir a la red.
    """

    sample_rate = EDGE_RATE

    def __init__(self, settings: VoiceSettings, fetch: Callable[..., bytes] | None = None) -> None:
        self.voice = settings.edge_voice
        self.rate = edge_rate(settings.speech_rate)
        self.timeout_s = settings.edge_timeout_s
        self._fetch = fetch or _edge_fetch
        self.voice_id = f"edge:{self.voice}{'' if self.rate == '+0%' else self.rate}"
        self.last_ms = 0.0

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        data = self._fetch(text, self.voice, self.rate, self.timeout_s)
        if not data:
            raise EstimatorError(f"Edge no devolvió audio para {text!r}")
        samples = trim_silence(decode_mp3(data, self.sample_rate), self.sample_rate)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return samples


class DiskCachedSynth:
    """Guarda en disco cada frase sintetizada (`<carpeta>/<voz>/v1/<sha256>.npy`).

    Lo dicho una vez sale después al instante y también sin red. La clave incluye la voz y la
    velocidad: cambiar de voz no reutiliza audio de otra.
    """

    def __init__(self, synth, folder: Path) -> None:
        self.synth = synth
        self.sample_rate = synth.sample_rate
        self.voice_id = getattr(synth, "voice_id", "voz")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.voice_id)
        self.folder = folder / safe / "v1"  # v1: con trim_silence; cambiar el audio sube la versión
        self.last_ms = 0.0
        self.hits = 0

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.voice_id}\n{text}".encode()).hexdigest()[:32]
        return self.folder / f"{key}.npy"

    def synthesize(self, text: str) -> np.ndarray:
        path = self._path(text)
        if path.is_file():
            try:
                samples = np.load(path)
                self.hits += 1
                self.last_ms = 0.0
                return samples
            except (OSError, ValueError):
                path.unlink(missing_ok=True)  # archivo roto: se vuelve a pedir
        samples = self.synth.synthesize(text)
        self.last_ms = getattr(self.synth, "last_ms", 0.0)
        self.folder.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.stem + ".parcial.npy")
        np.save(partial, samples)
        partial.replace(path)  # un corte no deja un .npy a medias
        return samples


class FallbackSynth:
    """Voz principal (Edge) con respaldo local (Piper) si falla o no hay red.

    Tras una falla no se vuelve a probar la principal durante `retry_s`: sin internet, cada
    frase esperaría el timeout. Lo dicho por el respaldo se lleva a la frecuencia de la principal
    para que el reproductor reciba siempre la misma. Cada motor tiene su propia caché en
    memoria (build_synth): una frase dicha por Piper durante un corte no queda fija con esa voz.
    """

    def __init__(self, primary, fallback, retry_s: float = 30.0, clock=time.monotonic) -> None:
        self.primary = primary
        self.fallback = fallback
        self.sample_rate = primary.sample_rate
        self.retry_s = retry_s
        self._clock = clock
        self._down_until = float("-inf")
        self._lock = threading.Lock()
        names = (
            getattr(primary, "voice_id", None) or "principal",
            getattr(fallback, "voice_id", None) or "respaldo",
        )
        self.voice_id = "|".join(names)
        self.failures = 0
        self.last_engine = ""
        self.last_error: str | None = None

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            try_primary = self._clock() >= self._down_until
        if try_primary:
            try:
                samples = self.primary.synthesize(text)
                self.last_engine = "principal"
                return samples
            except Exception as exc:  # red caída, timeout, servicio que cambió: seguir con Piper
                with self._lock:
                    self._down_until = self._clock() + self.retry_s
                    self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
        samples = self.fallback.synthesize(text)
        self.last_engine = "respaldo"
        return resample(samples, self.fallback.sample_rate, self.sample_rate)

    def preload(self, phrases: Iterable[str]) -> None:
        for phrase in phrases:
            self.synthesize(phrase)

    def prefetch(self, text: str) -> None:
        """Adelantar la voz principal mientras está arriba (Piper es local y rápido)."""
        with self._lock:
            up = self._clock() >= self._down_until
        prefetch = getattr(self.primary, "prefetch", None)
        if up and prefetch is not None:
            prefetch(text)


def build_synth(settings: VoiceSettings):
    """Sintetizador de la app según `audio.tts.engine`, con caché en memoria (y preload()).

    piper: Piper sola. edge: Edge con caché en disco y Piper de respaldo.
    """
    from visionx.voice.output import CachedSynth

    if settings.tts_engine == "piper":
        return CachedSynth(PiperSynth(settings))
    if settings.tts_engine != "edge":
        raise InvalidInput(f"audio.tts.engine {settings.tts_engine!r}: use piper o edge")
    primary = EdgeSynth(settings)
    if settings.tts_cache_dir:
        primary = DiskCachedSynth(primary, _abs(settings.tts_cache_dir))
    return FallbackSynth(
        CachedSynth(primary, prefetch=True),
        CachedSynth(PiperSynth(settings)),
        retry_s=settings.edge_retry_s,
    )


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
