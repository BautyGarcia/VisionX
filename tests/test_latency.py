"""Latencias de punta a punta desde una traza (app/latency.py)."""

from visionx.app.latency import order_latencies, summary

MS = 1_000_000


def test_latency_from_button_release_to_first_content():
    rows = [
        {"t_ns": 1000 * MS, "tipo": "boton-soltado"},
        {"t_ns": 2300 * MS, "tipo": "stt", "ms": 1300},
        {
            "t_ns": 2301 * MS,
            "tipo": "orden",
            "request_id": 4,
            "texto": "Leé.",
            "regla": "leer",
            "issued_ns": 2301 * MS,
        },
        {
            "t_ns": 2400 * MS,
            "tipo": "audio",
            "request_id": 4,
            "texto": "Leo.",
            "started_ns": 2450 * MS,
        },
        {"t_ns": 4000 * MS, "tipo": "asistente-respuesta", "request_id": 4, "primer_texto_s": 0.9},
        {
            "t_ns": 5000 * MS,
            "tipo": "audio",
            "request_id": 4,
            "texto": "Adelante tenés una caja.",
            "started_ns": 4100 * MS,
        },
    ]
    (o,) = order_latencies(rows)
    assert o["stt_ms"] == 1300 and o["acuse_ms"] == 1450 and o["primera_frase_ms"] == 3100
    assert o["claude_primer_texto_ms"] == 900
    assert "leer: n=1, primera frase mediana 3100 ms" in summary(rows)
