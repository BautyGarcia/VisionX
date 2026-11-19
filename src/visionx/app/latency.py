"""Latencias de punta a punta de una traza de `ejecutar --traza` (prueba en vivo, T07/M1).

Por orden: desde que se soltó el botón de hablar (evento `boton-soltado`) hasta el texto del
STT, el acuse (bip u «Leo.») y la primera frase con contenido; y, si respondió el asistente,
cuánto tardó Claude en dar su primera frase. Todo en el reloj monotónico de la traza.

    uv run python -m visionx.app latencias --traza outputs/beta/1119/vivo-1.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ACKS = {"Leo.", "Describo.", "Mido la distancia."}


def load(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def order_latencies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Una fila por orden con request_id: tiempos en ms desde soltar el botón (o la orden)."""
    out = []
    released: int | None = None
    stt_ms: float | None = None
    for i, r in enumerate(rows):
        kind = r.get("tipo")
        if kind == "boton-soltado":
            released, stt_ms = r["t_ns"], None
        elif kind == "stt":
            stt_ms = r.get("ms")
        elif kind == "orden" and r.get("request_id") is not None:
            rid, start = r["request_id"], released if released is not None else r["issued_ns"]
            audio = [a for a in rows[i:] if a.get("tipo") == "audio"
                     and a.get("request_id") == rid and a.get("started_ns")]  # fmt: skip
            first_any = min((a["started_ns"] for a in audio), default=None)
            content = [
                a["started_ns"]
                for a in audio
                if not str(a["texto"]).startswith("sonido:") and a["texto"] not in ACKS
            ]
            answer = next((a for a in rows[i:] if a.get("tipo") == "asistente-respuesta"
                           and a.get("request_id") == rid), None)  # fmt: skip
            out.append({
                "request_id": rid,
                "texto": r.get("texto"),
                "regla": r.get("regla"),
                "stt_ms": stt_ms,
                "acuse_ms": (first_any - start) / 1e6 if first_any else None,
                "primera_frase_ms": (min(content) - start) / 1e6 if content else None,
                "claude_primer_texto_ms": answer["primer_texto_s"] * 1e3
                if answer and answer.get("primer_texto_s") is not None else None,
                "desde_boton": released is not None,
            })  # fmt: skip
            released, stt_ms = None, None
    return out


def summary(rows: list[dict[str, Any]]) -> str:
    lines = ["| # | Orden | Regla | STT ms | Acuse ms | 1.ª frase ms | Claude 1.er texto ms |",
             "|---:|---|---|---:|---:|---:|---:|"]  # fmt: skip

    def fmt(v):
        return "—" if v is None else f"{v:.0f}"

    table = order_latencies(rows)
    for o in table:
        lines.append(f"| {o['request_id']} | {o['texto']} | {o['regla']} | {fmt(o['stt_ms'])} | "
                     f"{fmt(o['acuse_ms'])} | {fmt(o['primera_frase_ms'])} | "
                     f"{fmt(o['claude_primer_texto_ms'])} |")  # fmt: skip
    by_rule: dict[str, list[float]] = {}
    for o in table:
        if o["primera_frase_ms"] is not None:
            by_rule.setdefault(o["regla"] or "?", []).append(o["primera_frase_ms"])
    lines.append("")
    for rule, values in sorted(by_rule.items()):
        v = np.array(values)
        lines.append(f"- {rule}: n={len(v)}, primera frase mediana {np.median(v):.0f} ms, "
                     f"máx {v.max():.0f} ms")  # fmt: skip
    if table and not any(o["desde_boton"] for o in table):
        lines.append("- (sin `boton-soltado`: tiempos desde que la orden llegó al controlador)")
    return "\n".join(lines)
