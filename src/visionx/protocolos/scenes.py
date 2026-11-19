"""M10 descripción: ejecutar el handler y auditar lo realmente mencionado en su texto."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from .common import number, percentile, read_csv, report, resolve


def audit(phrases: list[str], facts: dict, references: list[dict], spoken, round_m=0.1):
    """Auditor del compositor local actual; texto libre futuro requiere revisión humana."""
    from visionx.detection.catalog import COCO_CLASSES
    from visionx.modes.describe import COMPOUND_PLURALS
    from visionx.modes.target import plural

    names = {}
    for label in COCO_CLASSES:
        name = "pantalla" if label == "tv" else "esquí" if label == "skis" else spoken(label)
        names[label] = (name, COMPOUND_PLURALS.get(label, plural(name)))
    mentioned, unsupported_meters, numeric = Counter(), 0, []
    count_words = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
    presence = facts.get("presence", [])
    distances = {m["detection_index"]: m["z_m"] for m in facts.get("measured", [])}
    for phrase in phrases:
        lower = phrase.casefold()
        zone = next(
            (z for z in ("a la izquierda", "a la derecha", "al frente") if lower.startswith(z)),
            None,
        )
        hits = []
        for label, variants in names.items():
            pattern = (
                r"\b(un|una|dos|tres|cuatro|cinco|\d+)\s+(?:"
                + "|".join(re.escape(v) for v in sorted(set(variants), key=len, reverse=True))
                + r")\b"
            )
            for match in re.finditer(pattern, lower):
                count = count_words.get(match[1], int(match[1]) if match[1].isdigit() else 1)
                hits.append((match.start(), match.end(), label))
                mentioned[label, zone] += count
        for match in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s+metros?\b", lower):
            value = float(match[1].replace(",", "."))
            previous = [h for h in hits if h[1] <= match.start()]
            label = max(previous, default=(0, 0, None))[2]
            supported = [
                p
                for p in presence
                if p["label"] == label
                and p["zone"] == zone
                and p["detection_index"] in distances
                and p.get("confidence") == "high"
            ]
            ok = any(
                abs(distances[p["detection_index"]] - value) <= round_m / 2 + 1e-6
                for p in supported
            )
            unsupported_meters += not ok
            numeric.append({"clase": label, "zona": zone, "z_dicha_m": value, "respaldada": ok})
    expected = Counter((r["clase"], r["zona"]) for r in references)
    hits = sum((mentioned & expected).values())
    extra = sum((mentioned - expected).values())
    return {
        "objetos_referencia": sum(expected.values()),
        "objetos_mencionados": sum(mentioned.values()),
        "objetos_reales_mencionados": hits,
        "afirmaciones_sin_respaldo": extra,
        "metros_inventados": unsupported_meters,
        "metros_dichos": len(numeric),
        "recall_objetos": hits / sum(expected.values()) if expected else None,
        "precision_objetos": hits / sum(mentioned.values()) if mentioned else None,
        "detalle_metros": numeric,
    }


def evaluate(
    sheet: Path, out: Path, cfg: dict, *, method="B0", window_s=0.4, samples=3, backend=None
):
    from visionx.app.runtime import Joint, View
    from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
    from visionx.modes.common import ModeSettings
    from visionx.modes.describe import DescribeHandler, DescribeSettings
    from visionx.shared.contracts import Command, Mode

    from .vision import OfflineVision

    annotations = read_csv(sheet)
    references = []
    for r in annotations:
        number(r["t_s"], "t_s")
        refs = json.loads(r["objetos_json"])
        if not isinstance(refs, list):
            raise ValueError("objetos_json debe ser lista de instancias (puede ser [])")
        for obj in refs:
            if obj["clase"] not in COCO_CLASSES or obj["zona"] not in (
                "a la izquierda",
                "a la derecha",
                "al frente",
            ):
                raise ValueError("clase/zona desconocida en referencia")
            number(obj["distancia_m"], "distancia aproximada", positive=True)
        references.append(refs)
    simulated = backend is not None
    backend = backend or OfflineVision(cfg)
    settings, ds = ModeSettings.from_config(cfg), DescribeSettings.from_config(cfg)
    spoken = ClassCatalog(COCO_CLASSES).spoken
    rows = []
    try:
        for r, refs in zip(annotations, references, strict=True):
            frames = list(
                backend.frames(resolve(sheet, r["clip"]), float(r["t_s"]), window_s, samples)
            )
            if not frames:
                raise ValueError("sin frames para describir")
            # Ventana al final de la espera: no fingir observaciones persistentes duplicando frames.
            limit = frames[-1].received_ns
            issued = max(0, limit - int(ds.wait_measure_s * 1e9))
            events = []
            handler = DescribeHandler(
                backend.geometry,
                spoken,
                settings,
                ds,
                on_event=lambda kind, data, events=events: events.append(data),
            )
            cmd = Command(1, Mode.DESCRIBE, "describí", issued, deadline_ns=limit + int(10e9))
            phrases = []
            handler.start(cmd, View(), issued)
            start = time.perf_counter()
            for frame in frames:
                detection, predictions = backend.observe(frame, [method])
                depth, objects = predictions[method]
                joint = Joint(frame, detection, depth, tuple(objects))
                says, done = handler.tick(cmd, View(joint, joint), frame.received_ns)
                phrases.extend(s.text_or_sound for s in says)
                if done:
                    break
            elapsed = (time.perf_counter() - start) * 1000
            event = next((e for e in reversed(events) if "hechos" in e), {})
            facts = event.get("hechos", {})
            metrics = audit(phrases, facts, refs, spoken, settings.round_m)
            rows.append(
                {
                    **r,
                    "metodo": method,
                    "texto": " ".join(phrases),
                    "hechos_json": json.dumps(facts, ensure_ascii=False),
                    "duracion_estimada_s": event.get("duracion_estimada_s"),
                    "evaluacion_offline_ms": elapsed,
                    "detalle_metros_json": json.dumps(metrics.pop("detalle_metros")),
                    **metrics,
                }
            )
    finally:
        backend.close()
    summary = {
        "simulado": simulated,
        "n": len(rows),
        "n_con_respuesta": sum(bool(r["texto"]) for r in rows),
        "objetos_reales_mencionados": sum(r["objetos_reales_mencionados"] for r in rows),
        "objetos_referencia": sum(r["objetos_referencia"] for r in rows),
        "afirmaciones_sin_respaldo": sum(r["afirmaciones_sin_respaldo"] for r in rows),
        "metros_inventados": sum(r["metros_inventados"] for r in rows),
        "duracion_p50_s": percentile(
            [r["duracion_estimada_s"] for r in rows if r["duracion_estimada_s"] is not None], 50
        ),
        "limites": "Duración de habla estimada por el modo, sin audio. "
        "Auditor del vocabulario/compositor local; revisar texto y hechos manualmente "
        "para afirmaciones libres. Distancia humana aproximada conservada para auditoría.",
    }
    return report(out, rows, summary, [sheet, *backend.inputs], cfg)
