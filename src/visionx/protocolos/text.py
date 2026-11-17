"""M10 OCR: mismas reglas públicas de normalización y métricas que reading.evaluate."""

import time
from pathlib import Path

from .common import number, percentile, read_csv, report, resolve


def evaluate(sheet: Path, out: Path, cfg: dict, *, reader=None, backend=None, image="original"):
    from visionx.reading.core import ReadingSettings, reading_order
    from visionx.reading.evaluate import text_metrics
    from visionx.reading.rapid import MODELS, RapidReader, weights_folder

    from .vision import OfflineVision

    annotations = read_csv(sheet)
    for r in annotations:
        number(r["t_s"], "t_s")
        number(r["distancia_m"], "distancia", positive=True)
        if not r["texto_referencia"].strip():
            raise ValueError("falta texto de referencia")
    settings = ReadingSettings.from_config(cfg)
    simulated = reader is not None or backend is not None
    reader = reader or RapidReader(settings)
    backend = backend or OfflineVision(cfg, detection=False)
    rows = []
    try:
        for r in annotations:
            frame = next(
                iter(
                    backend.frames(
                        resolve(sheet, r["clip"]),
                        float(r["t_s"]),
                        0,
                        1,
                        rectified=image == "rectified",
                    )
                )
            )
            start = time.perf_counter()
            blocks = reading_order(reader.read(frame.left), settings)
            elapsed = (time.perf_counter() - start) * 1000
            hypothesis = " ".join(b.text for b in blocks)
            rows.append(
                {
                    **r,
                    "frame_id": frame.frame_id,
                    "imagen": image,
                    "ocr": hypothesis,
                    "ocr_ms": elapsed,
                    **text_metrics(r["texto_referencia"], hypothesis),
                }
            )
    finally:
        backend.close()
    inputs = [sheet, *backend.inputs]
    if not simulated:
        inputs.extend(weights_folder(settings) / name for name, _ in MODELS.values())
    summary = {
        "simulado": simulated,
        "n": len(rows),
        "cer_micro": sum(r["ediciones"] for r in rows) / sum(r["caracteres_ref"] for r in rows),
        "recall_palabras_micro": sum(r["aciertos_palabras"] for r in rows)
        / sum(r["n_palabras_ref"] for r in rows)
        if sum(r["n_palabras_ref"] for r in rows)
        else None,
        "precision_palabras_micro": sum(r["aciertos_palabras_exactos"] for r in rows)
        / sum(r["n_palabras_ocr"] for r in rows)
        if sum(r["n_palabras_ocr"] for r in rows)
        else 0,
        "errores_numericos": sum(r["errores_numericos"] for r in rows),
        "ocr_p50_ms": percentile([r["ocr_ms"] for r in rows], 50),
        "ocr_p95_ms": percentile([r["ocr_ms"] for r in rows], 95),
    }
    return report(out, rows, summary, inputs, cfg)
