"""M10 piloto: un frame por segmento y ruta, CER de etiqueta completa y latencias.

Protocolo deliberadamente sin seleccionar leyendas por parecido al OCR: la referencia son
TODAS las comillas de textos.txt, en su orden, unidas con un espacio. La hipótesis concatena
TODOS los bloques aceptados, en orden visual. Así las omisiones, texto de fondo, repeticiones
y diferencias de orden se penalizan. No es CER solo de palabras reconocidas ni de una ROI
anotada. Los envases curvos pueden tener leyendas de referencia invisibles en esta vista.
Se guardan ambas cadenas y bloques para auditar esas limitaciones.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.camera.replay import load_recording
from visionx.reading.core import (
    ReadingSettings,
    TextReader,
    normalize_text,
    reading_order,
    sharpness,
    speech_chunks,
)
from visionx.reading.rapid import MODELS, file_hash, weights_folder
from visionx.shared.errors import SourceError


def comparison_text(text: str, *, fold_accents: bool = False) -> str:
    text = normalize_text(text).casefold()
    if fold_accents:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
        )
    return text


def edit_distance(ref: str | list, hyp: str | list) -> int:
    """Levenshtein, memoria O(longitud de hipótesis); inserción/borrado/sustitución = 1."""
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = comparison_text(reference), comparison_text(hypothesis)
    if not ref:
        raise ValueError("CER exige una referencia no vacía")
    folded_ref = comparison_text(reference, fold_accents=True)
    folded_hyp = comparison_text(hypothesis, fold_accents=True)
    # Tokens numéricos separados: "684", "36", "19", "24%"; no se confunde 120 con 12.
    number_pattern = r"\d+(?:[.,]\d+)?%?"
    rn, hn = re.findall(number_pattern, ref), re.findall(number_pattern, hyp)
    return {
        "cer": edit_distance(ref, hyp) / len(ref),
        "cer_sin_acentos": edit_distance(folded_ref, folded_hyp) / len(folded_ref),
        "caracteres_ref": len(ref),
        "ediciones": edit_distance(ref, hyp),
        "numeros_ref": " | ".join(rn),
        "numeros_ocr": " | ".join(hn),
        "errores_numericos": edit_distance(rn, hn),
        "n_numeros_ref": len(rn),
    }


def references(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not re.match(r"\d+\.", line):
            continue
        title = line.split(":", 1)[0]
        legends = re.findall(r'"([^"]+)"', line)
        if not legends:
            raise ValueError(f"{path}: referencia sin leyendas: {title}")
        out[comparison_text(title, fold_accents=True)] = " ".join(legends)
    if not out:
        raise ValueError(f"{path}: no hay referencias numeradas entre comillas")
    return out


def reference_for(obj: str, refs: dict[str, str]) -> str:
    # Etiqueta del CSV como palabra del título (Fusion -> Bariatric Fusion), sin fuzzy match.
    pattern = rf"\b{re.escape(comparison_text(obj, fold_accents=True))}\b"
    matches = [text for title, text in refs.items() if re.search(pattern, title)]
    if len(matches) != 1:
        raise ValueError(f"objeto {obj!r}: se esperaba una referencia, hay {len(matches)}")
    return matches[0]


def load_left(folder: Path, entry: dict[str, Any]) -> np.ndarray:
    path = folder / entry["left"]
    data = path.read_bytes()  # si Drive sigue copiando, fallar explícitamente; nunca saltear
    if hashlib.sha256(data).hexdigest() != entry["sha256_left"]:
        raise SourceError(f"hash incorrecto: {path}")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(f"imagen ilegible: {path}")
    return image


def evaluate(
    sources: list[Path],
    destination: Path,
    reader: TextReader,
    settings: ReadingSettings,
    cfg: dict[str, Any],
    *,
    routes: tuple[str, ...] = ("original",),
    synth=None,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    rows, details, input_hashes = [], [], {}
    skipped = []
    maps, calibration = None, None
    if "rectified" in routes:
        from visionx.calibration import StereoCalibration
        from visionx.shared.config import ROOT

        path = Path(cfg["camera"]["calibration"])
        path = path if path.is_absolute() else ROOT / path
        calibration = StereoCalibration.load(path)
        maps = calibration.rectify_maps()[0]  # solo izquierda: OCR no necesita ojo derecho
        input_hashes[str(path)] = file_hash(path)
    cold = True
    for folder in sources:
        manifest, entries, truncated = load_recording(folder)
        if truncated or len(entries) != manifest.get("pares"):
            raise SourceError(f"{folder}: índice incompleto; esperar copia de Drive")
        if manifest.get("rectificado"):
            raise SourceError(f"{folder}: la comparación exige grabación original")
        refs = references(folder / "textos.txt")
        for name in ("textos.txt", "segmentos.csv", "frames.jsonl", "manifest.yaml"):
            input_hashes[str(folder / name)] = file_hash(folder / name)
        with (folder / "segmentos.csv").open(encoding="utf-8-sig", newline="") as f:
            segments = list(csv.DictReader(f))
        for segment in segments:
            start, finish = int(segment["inicio_ns"]), int(segment["fin_ns"])
            if finish <= start:
                raise ValueError("segmento con fin <= inicio")
            end = min(finish, start + int(settings.window_s * 1e9))
            candidates = [e for e in entries if start <= e["received_ns"] <= end]
            if not candidates:
                raise SourceError(f"{folder}: no hay frames en {segment}")
            expected = len(candidates)
            missing = [e["left"] for e in candidates if not (folder / e["left"]).is_file()]
            if missing:
                if not allow_incomplete:
                    raise SourceError(
                        f"{folder}: faltan {len(missing)}/{expected} izquierdas "
                        f"en {segment}; esperar copia (primera: {missing[0]})"
                    )
                candidates = [e for e in candidates if (folder / e["left"]).is_file()]
                skipped.append(
                    {
                        "fuente": str(folder),
                        "segmento": segment,
                        "faltantes": missing,
                        "n_disponibles": len(candidates),
                    }
                )
                if not candidates:
                    print(
                        f"{folder.name} {segment['objeto']} {segment['distancia_m']}m: "
                        "SIN FRAMES, sin métrica",
                        flush=True,
                    )
                    continue
            reference = reference_for(segment["objeto"], refs)
            for route in routes:
                t0 = time.perf_counter()
                best_score, selected, best = -1.0, None, None
                for entry in candidates:
                    image = load_left(folder, entry)
                    if route == "rectified":
                        if (image.shape[1], image.shape[0]) != tuple(calibration.image_size):
                            raise SourceError("la calibración no corresponde al tamaño del clip")
                        image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
                    score = sharpness(image)
                    if score > best_score:
                        best_score, selected, best = score, entry, image
                select_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                raw = reader.read(best)
                ocr_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                blocks = reading_order(raw, settings)
                hypothesis = " ".join(b.text for b in blocks)
                chunks = speech_chunks(blocks, settings.max_chars)
                post_ms = (time.perf_counter() - t0) * 1000
                # Latencia lógica: ventana en tiempo de video + OCR/post. La selección offline
                # incluye JPEG/disco y se reporta separada: no fingir que mide captura en vivo.
                ready_s = (end - start) / 1e9 + (ocr_ms + post_ms) / 1000 if blocks else None
                tts_ms, onset_s, first_word_s = None, None, None
                if chunks and synth is not None:
                    t0 = time.perf_counter()
                    samples = synth.synthesize(chunks[0])  # solo archivo en memoria; sin player
                    tts_ms = (time.perf_counter() - t0) * 1000
                    active = np.flatnonzero(np.abs(samples) >= 0.01)
                    if active.size:
                        onset_s = int(active[0]) / synth.sample_rate
                        first_word_s = ready_s + tts_ms / 1000 + onset_s
                row = {
                    "fuente": folder.name,
                    "objeto": segment["objeto"],
                    "distancia_m": float(segment["distancia_m"]),
                    "imagen": route,
                    "inicio_ns": start,
                    "fin_ventana_ns": end,
                    "n_frames": len(candidates),
                    "n_frames_esperados": expected,
                    "copia_completa_ventana": not missing,
                    "frame_id": selected["frame_id"],
                    "received_ns": selected["received_ns"],
                    "sha256_left": selected["sha256_left"],
                    "ancho": best.shape[1],
                    "alto": best.shape[0],
                    "nitidez": best_score,
                    "ocr_frio": cold,
                    "seleccion_offline_ms": select_ms,
                    "ocr_ms": ocr_ms,
                    "post_ms": post_ms,
                    "primera_palabra_lista_s": ready_s,
                    "tts_ms": tts_ms,
                    "inicio_senal_s": onset_s,
                    "primera_palabra_estimada_s": first_word_s,
                    "bloques": len(blocks),
                    "referencia": reference,
                    "ocr": hypothesis,
                    **text_metrics(reference, hypothesis),
                }
                cold = False
                rows.append(row)
                details.append({**row, "bloques_crudos": [asdict(b) for b in raw]})
                print(
                    f"{folder.name} {segment['objeto']} {segment['distancia_m']}m {route}: "
                    f"CER {row['cer']:.1%}, OCR {ocr_ms:.0f} ms",
                    flush=True,
                )
    if not rows:
        raise SourceError("ningún segmento con frames disponibles; no hay métricas")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "desconocido", None
    hashes = {
        name: file_hash(weights_folder(settings) / name)
        for name, _ in MODELS.values()
        if (weights_folder(settings) / name).is_file()
    }
    metadata = {
        "commit": commit,
        "dirty": dirty,
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "procesador": platform.processor(),
        "reader": reader.model_id,
        "reading": asdict(settings),
        "config": cfg,
        "modelos_sha256": hashes,
        "inputs_sha256": input_hashes,
        "voz": getattr(synth, "voice_id", None),
        "resultados": details,
        "copias_incompletas": skipped,
        "protocolo": __doc__,
        "tiempos": "Primera palabra lista = ventana + OCR + post. Estimada agrega síntesis "
        "completa del primer Say e inicio de señal |x|>=0.01. Sin reproducción, cola, captura "
        "ni tick; no es latencia audible. OCR frío incluye carga/import/hashes, una vez.",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows
