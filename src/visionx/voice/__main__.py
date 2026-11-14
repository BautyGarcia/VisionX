"""Comandos de voz (T08). Requieren `uv sync --extra voz` salvo `orden` y `frases`.

    uv run python -m visionx.voice dispositivos
    uv run python -m visionx.voice descargar [--voz es_AR-daniela-high] [--whisper base small]
    uv run python -m visionx.voice orden "¿A qué distancia está la taza?"
    uv run python -m visionx.voice frases [tests/fixtures/voz-frases.csv]
    uv run python -m visionx.voice decir "Busco la taza." [--wav salida.wav]
    uv run python -m visionx.voice transcribir audio.wav
    uv run python -m visionx.voice bucle [--whisper base small] [--csv X]
    uv run python -m visionx.voice escuchar

`bucle` sintetiza cada frase con Piper, la transcribe con Whisper y la interpreta: mide el
circuito sin micrófono, con una voz sintética. No reemplaza las frases dichas por personas.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import AudioEvent, Priority
from visionx.shared.errors import EstimatorError, SourceError
from visionx.voice.commands import CommandParser, Interpretation, acknowledgement
from visionx.voice.engines import (
    WHISPER_RATE,
    VoiceSettings,
    download_piper_voice,
    download_whisper,
    resample,
    sha256_file,
)

DEFAULT_PHRASES = ROOT / "tests" / "fixtures" / "voz-frases.csv"


def _parser() -> CommandParser:
    return CommandParser(ClassCatalog(COCO_CLASSES))


def read_phrases(path: Path) -> Iterator[dict[str, str]]:
    """Filas frase,modo,objetivo; las líneas que empiezan con # son comentarios."""
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(line for line in f if not line.startswith("#"))


def outcome(interp: Interpretation) -> tuple[str, str]:
    """(modo, objetivo) comparable con el CSV: "aclarar" si hay que preguntar."""
    if not interp.ok:
        return "aclarar", ""
    return interp.mode.value, interp.target or ""  # type: ignore[union-attr]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SourceError(f"{path}: se esperaba PCM de 16 bits")
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        channels, rate = w.getnchannels(), w.getframerate()
    return data.reshape(-1, channels).mean(axis=1), rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# --- comandos ---------------------------------------------------------------------------


def cmd_dispositivos(args: argparse.Namespace) -> int:
    import sounddevice as sd

    print(sd.query_devices())
    print("> = entrada por defecto, < = salida por defecto. Usar el índice en audio.*_device.")
    return 0


def cmd_descargar(args: argparse.Namespace) -> int:
    s = VoiceSettings.from_config(load_config(args.config))
    for voice in args.voz or [s.tts_voice]:
        for path in download_piper_voice(voice, Path(s.voice_path.parent)):
            print(f"{path}  {path.stat().st_size} bytes  sha256 {sha256_file(path)}")
    for model in args.whisper or [s.stt_model]:
        folder = download_whisper(model, s.stt_path.parent)
        print(f"{folder}  model.bin sha256 {sha256_file(folder / 'model.bin')}")
    print("Registrar los sha256 en BITACORA.")
    return 0


def cmd_orden(args: argparse.Namespace) -> int:
    parser = _parser()
    status = 0
    for text in args.texto:
        interp = parser.parse(text)
        mode, target = outcome(interp)
        print(f"{text!r} -> {mode} {target} (regla {interp.rule})")
        reply = acknowledgement(interp, parser.catalog)
        if reply:
            print(f"   respuesta: {reply}")
        status |= not interp.ok
    return status


def cmd_frases(args: argparse.Namespace) -> int:
    parser = _parser()
    rows = list(read_phrases(Path(args.archivo)))
    failures = []
    for row in rows:
        got = outcome(parser.parse(row["frase"]))
        if got != (row["modo"], row["objetivo"]):
            failures.append((row["frase"], got, (row["modo"], row["objetivo"])))
    for phrase, got, expected in failures:
        print(f"FALLA {phrase!r}: {got} en vez de {expected}")
    print(f"{len(rows) - len(failures)} de {len(rows)} frases interpretadas como se esperaba")
    return 1 if failures else 0


def cmd_decir(args: argparse.Namespace) -> int:
    from visionx.voice.engines import PiperSynth

    s = VoiceSettings.from_config(load_config(args.config))
    synth = PiperSynth(s)
    samples = synth.synthesize(args.texto)
    duration = len(samples) / synth.sample_rate
    print(f"{synth.voice_id}: síntesis {synth.last_ms:.0f} ms para {duration:.2f} s de audio")
    if args.wav:
        write_wav(Path(args.wav), samples, synth.sample_rate)
        print(f"guardado en {args.wav}")
        return 0
    from visionx.voice.output import AudioOutput, SoundDevicePlayer

    output = AudioOutput(synth, SoundDevicePlayer(s.output_device, s.block_ms))
    output.submit(AudioEvent(1, Priority.SHORT, args.texto, created_ns=output.clock.now_ns()))
    output.wait_idle(timeout_s=duration + 10)
    output.close()
    record = output.records[0]
    print(f"{record.outcome}; empezó a sonar {record.wait_ms:.0f} ms después de pedirlo")
    return 0


def cmd_transcribir(args: argparse.Namespace) -> int:
    from visionx.voice.engines import WhisperTranscriber

    s = VoiceSettings.from_config(load_config(args.config))
    if args.whisper:
        s = dataclasses.replace(s, stt_model=args.whisper)
    stt = WhisperTranscriber(s)
    audio, rate = read_wav(Path(args.wav))
    result = stt.transcribe(resample(audio, rate, WHISPER_RATE))
    print(f"{stt.model_id}: {result.text!r} ({result.elapsed_ms:.0f} ms, {result.audio_s:.2f} s)")
    interp = _parser().parse(result.text)
    print(f"-> {outcome(interp)} (regla {interp.rule})")
    return 0


def cmd_bucle(args: argparse.Namespace) -> int:
    from visionx.voice.engines import PiperSynth, WhisperTranscriber

    s = VoiceSettings.from_config(load_config(args.config))
    parser = _parser()
    synth = PiperSynth(s)
    rows = [r for r in read_phrases(Path(args.archivo)) if r["frase"].strip()]
    audios = []
    tts_ms = []
    for row in rows:  # sintetizar una sola vez: todos los modelos oyen el mismo audio
        samples = synth.synthesize(row["frase"])
        tts_ms.append(synth.last_ms)
        audios.append(resample(samples, synth.sample_rate, WHISPER_RATE))
    print(f"{synth.voice_id}: {len(rows)} frases, síntesis mediana {np.median(tts_ms):.0f} ms")
    table = []
    for model in args.whisper or [s.stt_model]:
        stt = WhisperTranscriber(dataclasses.replace(s, stt_model=model))
        stt.transcribe(audios[0])  # calentamiento: la primera llamada inicializa
        ok = 0
        times = []
        for row, audio in zip(rows, audios, strict=True):
            tr = stt.transcribe(audio)
            times.append(tr.elapsed_ms)
            got = outcome(parser.parse(tr.text))
            hit = got == (row["modo"], row["objetivo"])
            ok += hit
            table.append(
                {
                    "modelo": stt.model_id,
                    "frase": row["frase"],
                    "transcripcion": tr.text,
                    "esperado": f"{row['modo']} {row['objetivo']}".strip(),
                    "obtenido": f"{got[0]} {got[1]}".strip(),
                    "acierto": int(hit),
                    "stt_ms": round(tr.elapsed_ms, 1),
                    "audio_s": round(tr.audio_s, 2),
                }
            )
            if not hit and args.detalle:
                print(f"  {row['frase']!r} -> {tr.text!r} -> {got}")
        t = np.asarray(times)
        print(
            f"{stt.model_id}: {ok}/{len(rows)} órdenes correctas; STT mediana "
            f"{np.median(t):.0f} ms, p90 {np.percentile(t, 90):.0f} ms"
        )
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"detalle en {path}")
    return 0


def cmd_escuchar(args: argparse.Namespace) -> int:
    """Prueba manual con micrófono: Enter empieza a grabar, Enter termina; q + Enter sale."""
    from visionx.voice.engines import PiperSynth, PushToTalk, WhisperTranscriber
    from visionx.voice.output import AudioOutput, CachedSynth, SoundDevicePlayer

    s = VoiceSettings.from_config(load_config(args.config))
    parser = _parser()
    stt = WhisperTranscriber(s)
    synth = CachedSynth(PiperSynth(s))
    synth.preload(s.preload)
    output = AudioOutput(synth, SoundDevicePlayer(s.output_device, s.block_ms))
    mic = PushToTalk(s.input_device, s.max_record_s)
    request_id = 0
    print("Enter para hablar, Enter para terminar. 'q' + Enter para salir.")
    try:
        while input("> listo ").strip().lower() != "q":
            output.cancel_all()  # hablar corta lo que estaba diciendo
            mic.start()
            input("  grabando... Enter para terminar ")
            t_release = time.perf_counter()
            audio = mic.stop()
            tr = stt.transcribe(audio)
            interp = parser.parse(tr.text)
            request_id += 1
            reply = acknowledgement(interp, parser.catalog) or "sonido:ok"
            output.submit(
                AudioEvent(request_id, Priority.SHORT, reply, created_ns=output.clock.now_ns())
            )
            output.wait_idle()
            record = output.records[-1]
            total = (time.perf_counter() - t_release) * 1000
            print(
                f"  {tr.text!r} -> {outcome(interp)}; STT {tr.elapsed_ms:.0f} ms; "
                f"de soltar a terminar la respuesta {total:.0f} ms; "
                f"respuesta empezó {record.wait_ms or 0:.0f} ms después de pedirla"
            )
    finally:
        output.close()
    if mic.overflows:
        print(f"aviso: el micrófono perdió bloques {mic.overflows} veces")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.voice",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dispositivos", help="micrófonos y salidas de audio disponibles")
    p.set_defaults(fn=cmd_dispositivos)

    p = sub.add_parser("descargar", help="bajar voz de Piper y modelos de Whisper")
    p.add_argument("--voz", nargs="+")
    p.add_argument("--whisper", nargs="+", help="tiny, base, small, medium...")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_descargar)

    p = sub.add_parser("orden", help="interpretar frases escritas (sin audio)")
    p.add_argument("texto", nargs="+")
    p.set_defaults(fn=cmd_orden)

    p = sub.add_parser("frases", help="comparar las reglas con un CSV frase,modo,objetivo")
    p.add_argument("archivo", nargs="?", default=str(DEFAULT_PHRASES))
    p.set_defaults(fn=cmd_frases)

    p = sub.add_parser("decir", help="sintetizar una frase y reproducirla o guardarla")
    p.add_argument("texto")
    p.add_argument("--wav", help="guardar en vez de reproducir")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_decir)

    p = sub.add_parser("transcribir", help="transcribir un WAV e interpretarlo")
    p.add_argument("wav")
    p.add_argument("--whisper", help="pisa audio.stt.model")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_transcribir)

    p = sub.add_parser("bucle", help="Piper -> Whisper -> reglas sobre el CSV de frases")
    p.add_argument("archivo", nargs="?", default=str(DEFAULT_PHRASES))
    p.add_argument("--whisper", nargs="+", help="modelos a comparar")
    p.add_argument("--csv", help="guardar el detalle por frase")
    p.add_argument("--detalle", action="store_true", help="mostrar cada error")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_bucle)

    p = sub.add_parser("escuchar", help="prueba con micrófono y botón (Enter)")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_escuchar)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (EstimatorError, SourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
