"""M10 voz: captura recuperable y comparación pareada de Whisper base/small."""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from .audio_devices import readable_name, resolve_device
from .common import percentile, read_csv, report, sha, write_json


def phrases() -> list[dict]:
    with Path(__file__).with_name("frases.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [dict(id=i, **r, notas="") for i, r in enumerate(rows, 1) if r["frase"].strip()]


def capture(seconds: float, device=None) -> tuple[np.ndarray, dict]:
    """Solo se abre por `voz grabar`; callback ligero y medidor desde el hilo principal."""
    import sounddevice as sd

    from visionx.voice.engines import resample

    device = resolve_device(device, sd)
    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])
    blocks, overflows = [], []

    def callback(data, frames, timing, status):
        blocks.append(data[:, 0].copy())
        if status:
            overflows.append(str(status))

    with sd.InputStream(
        device=device, channels=1, samplerate=rate, dtype="float32", callback=callback
    ):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            peak = float(np.max(np.abs(blocks[-1]))) if blocks else 0.0
            bar = "#" * min(30, int(peak * 60))
            print(f"\rGRABANDO [{bar:30}] pico {peak:.3f}", end="", flush=True)
            time.sleep(min(0.1, max(0, end - time.monotonic())))
    print()
    audio = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
    audio = resample(audio[: int(seconds * rate)], rate, 16000)
    return audio, {
        "dispositivo": str(device),
        "nombre": readable_name(info["name"]),
        "rate_original": rate,
        "overflows": overflows,
    }


def record(
    sheet: Path,
    *,
    seconds=4.0,
    device=None,
    countdown=3,
    recorder=capture,
    ask=input,
    sleep=time.sleep,
    revisit=False,
) -> dict:
    from visionx.voice.__main__ import write_wav

    rows = read_csv(sheet)
    if seconds <= 0 or not np.isfinite(seconds) or countdown < 0:
        raise ValueError("duración positiva y cuenta regresiva no negativa requeridas")
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids) or any(not i.isdigit() for i in ids):
        raise ValueError("id de frase debe ser entero único")
    state_path = sheet.parent / "sesion.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    digest = sha(sheet)
    if state and state.get("planilla_sha256") != digest:
        raise ValueError("la planilla cambió desde la captura; conservarla o crear otra sesión")
    state.setdefault("planilla_sha256", digest)
    if state.get("origen_audio", "microfono") != "microfono":
        raise ValueError("sesión sintética: usar otra carpeta para grabar personas")
    state["origen_audio"] = "microfono"
    items = state.setdefault("frases", {})
    try:
        for row in rows:
            key = str(row["id"])
            prior = items.get(key, {})
            if prior.get("estado") == "ok" and (sheet.parent / prior["wav"]).is_file():
                continue
            if prior.get("estado") == "saltada" and not revisit:
                continue
            while True:
                action = ask(
                    f"[{key}/{len(rows)}] «{row['frase']}» Enter=grabar s=saltear q=salir: "
                )
                if action.lower() == "q":
                    return state
                if action.lower() == "s":
                    items[key] = {**prior, "estado": "saltada"}
                    write_json(state_path, state)
                    break
                for n in range(countdown, 0, -1):
                    print(f"{n}…", flush=True)
                    sleep(1)
                try:
                    audio, meta = recorder(seconds, device)
                except Exception as exc:
                    # PortAudio tiene tipos propios: conservar error e interacción sin perder WAVs.
                    prior = {**prior, "error": str(exc), "estado": "error"}
                    items[key] = prior
                    write_json(state_path, state)
                    print(f"No se pudo grabar: {exc}. Podés reintentar, saltear o salir.")
                    continue
                audio = np.asarray(audio, np.float32).reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    print("Audio vacío o inválido: reintentar.")
                    continue
                peak = float(np.max(np.abs(audio)))
                rms = float(np.sqrt(np.mean(audio**2)))
                # WAV de captura: mono 16 kHz. Energía RMS al final indica posible corte,
                # no demuestra que la frase haya terminado: conservar aviso y permitir reintento.
                tail_rms = float(np.sqrt(np.mean(audio[-3200:] ** 2)))
                warnings = []
                if peak < 0.02:
                    warnings.append("casi silencio")
                if np.mean(np.abs(audio) >= 0.999) > 0.001:
                    warnings.append("saturación")
                if meta.get("overflows"):
                    warnings.append("pérdida de bloques")
                if tail_rms >= max(0.02, 0.5 * rms):
                    warnings.append(
                        "posible frase cortada: energía alta en últimos 200 ms; aumentar --segundos"
                    )
                attempt = int(prior.get("intentos", 0)) + 1
                wav = f"wav/{int(key):03}-{attempt:03}.wav"
                write_wav(sheet.parent / wav, audio, 16000)
                prior = {
                    "estado": "pendiente",
                    "wav": wav,
                    "intentos": attempt,
                    "pico": peak,
                    "rms": rms,
                    "rms_ultimos_200ms": tail_rms,
                    "advertencias": warnings,
                    **meta,
                }
                items[key] = prior
                write_json(state_path, state)  # también conserva intentos rechazados en disco
                print(f"Guardado {wav}; pico {peak:.3f}, RMS {rms:.3f}; {', '.join(warnings)}")
                action = ask(
                    "a=aceptar r=reintentar s=saltear q=guardar y salir: "
                    if warnings
                    else "Enter=aceptar r=reintentar s=saltear q=salir: "
                )
                if action.lower() == "q":
                    return state
                if action.lower() == "s":
                    items[key]["estado"] = "saltada"
                elif (not warnings and not action) or action.lower() == "a":
                    items[key]["estado"] = "ok"
                else:
                    continue
                write_json(state_path, state)
                break
    except (KeyboardInterrupt, EOFError):
        print("\nSesión guardada; repetir el comando para retomar.")
    finally:
        write_json(state_path, state)
    return state


def synthesize_session(sheet: Path, cfg: dict) -> dict:
    """Piper -> WAV, sin reproducir; sustituto explícito de personas para probar la cadena."""
    from visionx.voice.__main__ import write_wav
    from visionx.voice.engines import PiperSynth, VoiceSettings, resample

    state_path = sheet.parent / "sesion.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state and (
        state.get("origen_audio") != "sintetico_piper" or state["planilla_sha256"] != sha(sheet)
    ):
        raise ValueError("sesión existente incompatible; usar otra carpeta")
    state.update(planilla_sha256=sha(sheet), origen_audio="sintetico_piper")
    items = state.setdefault("frases", {})
    synth = PiperSynth(VoiceSettings.from_config(cfg))
    state["sintetizador"] = synth.voice_id
    for row in read_csv(sheet):
        key = str(row["id"])
        if not key.isdigit():
            raise ValueError("id de frase debe ser entero")
        if items.get(key, {}).get("estado") == "ok":
            continue
        audio = resample(synth.synthesize(row["frase"]), synth.sample_rate, 16000)
        wav = f"wav/{int(key):03}-piper.wav"
        write_wav(sheet.parent / wav, audio, 16000)
        items[key] = {"estado": "ok", "wav": wav, "intentos": 1}
        write_json(state_path, state)
    return state


def evaluate(sheet: Path, out: Path, cfg: dict, *, factories=None) -> dict:
    from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
    from visionx.voice.__main__ import outcome, read_wav
    from visionx.voice.commands import CommandParser
    from visionx.voice.engines import VoiceSettings, WhisperTranscriber, resample

    state_path = sheet.parent / "sesion.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["planilla_sha256"] != sha(sheet):
        raise ValueError("la planilla cambió desde la grabación")
    rows = read_csv(sheet)
    accepted = [r for r in rows if state["frases"].get(str(r["id"]), {}).get("estado") == "ok"]
    if not accepted:
        raise ValueError("no hay WAV aceptados; ejecutar voz grabar")
    parser = CommandParser(ClassCatalog(COCO_CLASSES))
    settings = VoiceSettings.from_config(cfg)
    table, summaries = [], []
    inputs = [sheet, state_path]
    cache_path = out / "transcripciones.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for model in ("base", "small"):
        if factories is None:
            inputs.append(dataclasses.replace(settings, stt_model=model).stt_path / "model.bin")
        stt = (
            factories[model]()
            if factories
            else WhisperTranscriber(dataclasses.replace(settings, stt_model=model))
        )
        model_id = stt.model_id
        # Warmup separado. No se usan sus tiempos como latencia medida.
        stt.transcribe(np.zeros(16000, np.float32))
        current = []
        for r in accepted:
            wav = sheet.parent / state["frases"][str(r["id"])]["wav"]
            inputs.append(wav)
            fingerprint = f"{model_id}:{sha(wav)}:{dataclasses.asdict(settings)}"
            reused = fingerprint in cache
            if fingerprint not in cache:
                samples, rate = read_wav(wav)
                if not samples.size:
                    raise ValueError(f"WAV vacío: {wav}")
                tr = stt.transcribe(resample(samples, rate, 16000))
                cache[fingerprint] = {"texto": tr.text, "ms": tr.elapsed_ms}
                write_json(cache_path, cache)
            tr = cache[fingerprint]
            mode, target = outcome(parser.parse(tr["texto"]))
            intent_ok = mode == r["modo"] and (
                r["modo"] not in ("control", "choose") or target == r["objetivo"]
            )
            object_case = r["modo"] in ("search", "distance") and bool(r["objetivo"])
            object_ok = target == r["objetivo"] if object_case else None
            item = {
                "id": r["id"],
                "modelo": model,
                "model_id": model_id,
                "frase": r["frase"],
                "modo_esperado": r["modo"],
                "objeto_esperado": r["objetivo"],
                "texto": tr["texto"],
                "modo": mode,
                "objetivo": target,
                "intencion_ok": intent_ok,
                "objeto_ok": object_ok,
                "conjunto_ok": intent_ok and target == r["objetivo"],
                "stt_ms": tr["ms"],
                "tiempo_de_cache": reused,
                "wav": str(wav),
            }
            current.append(item)
            print(f"{model} [{r['id']}] {tr['texto']!r} -> {mode} {target}", flush=True)
        objects = [r for r in current if r["objeto_ok"] is not None]
        summaries.append(
            {
                "modelo": model,
                "n": len(current),
                "n_planificados": len(rows),
                "n_pendientes_o_saltadas": len(rows) - len(current),
                "intencion_pct": 100 * sum(r["intencion_ok"] for r in current) / len(current),
                "n_objeto": len(objects),
                "objeto_pct": 100 * sum(r["objeto_ok"] for r in objects) / len(objects)
                if objects
                else None,
                "conjunto_pct": 100 * sum(r["conjunto_ok"] for r in current) / len(current),
                "stt_p50_ms": percentile([r["stt_ms"] for r in current], 50),
                "stt_p95_ms": percentile([r["stt_ms"] for r in current], 95),
            }
        )
        table.extend(current)
        del stt  # un modelo residente por vez; otros agentes comparten CPU/RAM
    return report(
        out,
        table,
        {
            "simulado": factories is not None,
            "origen_audio": state.get("origen_audio", "no_declarado"),
            "latencia": "STT tras warmup; cache conserva tiempos de la primera corrida",
            "modelos": summaries,
        },
        inputs,
        cfg,
    )
