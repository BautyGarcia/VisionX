"""Regresiones de lectura hablada: conservar sentido y señalar huecos, sin TTS."""

import pytest

from visionx.reading import ReadingSettings, TextBlock
from visionx.reading.core import spoken_chunks


@pytest.mark.parametrize("negation", ["NO", "no", "NI", "ni", "SIN", "sin"])
def test_standalone_negation_is_preserved_before_next_line(negation):
    blocks = [
        TextBlock(negation, 0.99, (10, 10, 50, 30)),
        TextBlock("APTO PARA CELÍACOS", 0.99, (10, 34, 220, 54)),
    ]
    assert spoken_chunks(blocks, ReadingSettings()) == [f"{negation} APTO PARA CELÍACOS"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Tomar 1/2 comprimido", "Tomar 1/2 comprimido"),
        ("Conservar a -5 grados", "Conservar a menos 5 grados"),
        ("Conservar a −5 grados", "Conservar a menos 5 grados"),
        ("VENCE 11/2026", "VENCE 11/2026"),
        ("Abrir 08:05", "Abrir 08 horas 05"),
        ("Hora 14:30:05", "Hora 14 horas 30 minutos 05 segundos"),
        ("Dosis -1/2", "Dosis menos 1/2"),
        ("Descuento 12,5%", "Descuento 12,5%"),
        ("Temperatura -5,25", "Temperatura menos 5,25"),
        ("Contenido 0,05 g", "Contenido 0,05 g"),
    ],
)
def test_numbers_reach_speech_without_losing_sign_or_separator(text, expected):
    blocks = [TextBlock(text, 0.99, (10, 10, 300, 30))]
    assert spoken_chunks(blocks, ReadingSettings()) == [expected]
    assert blocks[0].text == text  # la evaluación OCR conserva la transcripción cruda


@pytest.mark.parametrize(
    "text,expected",
    [
        ("NO @@@ APTO", "NO (ilegible) APTO"),
        ("NO abc123 bcdfgh APTO", "NO (ilegible) APTO"),
        ("@@@ APTO", "(ilegible) APTO"),
        ("NO @@@", "NO (ilegible)"),
        ("@@@ abc123", ""),
        ("NO APTO", "NO APTO"),
    ],
)
def test_rejected_tokens_cannot_form_an_apparently_complete_phrase(text, expected):
    chunks = spoken_chunks([TextBlock(text, 0.99, (10, 10, 300, 30))], ReadingSettings())
    assert chunks == ([expected] if expected else [])


@pytest.mark.parametrize("rejected,score", [("@@@", 0.99), ("PARA", 0.8)])
def test_rejected_block_between_retained_lines_is_announced(rejected, score):
    blocks = [
        TextBlock("NO", 0.99, (10, 10, 100, 30)),
        TextBlock(rejected, score, (10, 32, 100, 52)),
        TextBlock("APTO", 0.99, (10, 54, 100, 74)),
    ]
    assert spoken_chunks(blocks, ReadingSettings()) == ["NO (ilegible) APTO"]


def test_rejected_run_spanning_blocks_gets_only_one_notice():
    blocks = [
        TextBlock("NO @@@", 0.99, (10, 10, 100, 30)),
        TextBlock("@@@", 0.99, (10, 32, 100, 52)),
        TextBlock("@@@ APTO", 0.99, (10, 54, 100, 74)),
    ]
    assert spoken_chunks(blocks, ReadingSettings()) == ["NO (ilegible) APTO"]
