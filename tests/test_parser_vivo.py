"""Órdenes que en el vivo del 19/11 terminaban en Claude (traza de Bauty: 6 de 26).

Con el asistente activo, lo que no empieza con el verbo iba como pregunta libre: Claude
describe en vez de guiar, tarda más y cuesta. Estas frases tienen que ser la orden.
"""

from __future__ import annotations

import pytest

from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.contracts import Mode
from visionx.voice.commands import CommandParser

CON_CLAUDE = CommandParser(ClassCatalog(COCO_CLASSES), questions=True)


@pytest.mark.parametrize(
    "frase,modo,objetivo",
    [
        ("Sí, puedes buscar mi teléfono.", Mode.SEARCH, "cell phone"),
        ("¿Podés buscar mi teléfono?", Mode.SEARCH, "cell phone"),
        ("Quiero que me guíes a la taza.", Mode.SEARCH, "cup"),
        ("¿Me podrías llevar a la silla?", Mode.SEARCH, "chair"),
        ("Necesito encontrar la botella.", Mode.SEARCH, "bottle"),
        ("¿Me ayudás a buscar la mochila?", Mode.SEARCH, "backpack"),
        ("Diame a la taza.", Mode.SEARCH, "cup"),  # Whisper por «guiame»
        ("Kiame al baño.", Mode.SEARCH, "toilet"),
        ("Viame hasta la silla.", Mode.SEARCH, "chair"),
        ("Buscá la tasa.", Mode.SEARCH, "cup"),  # Whisper por «taza»
        ("Guíame a la taza.", Mode.SEARCH, "cup"),
        ("¿Me podés leer esto?", Mode.READ, None),
        ("¿Podés decirme a qué distancia está la silla?", Mode.DISTANCE, "chair"),
    ],
)
def test_ordenes_mal_dichas_no_van_a_claude(frase, modo, objetivo):
    i = CON_CLAUDE.parse(frase)
    assert i.mode is modo, (frase, i)
    if objetivo is not None:
        assert i.target == objetivo, (frase, i)


@pytest.mark.parametrize(
    "frase",
    ["¿Qué hora es?", "Diame la verdad, ¿es dulce?", "¿Para qué sirve esto?", "¿Es de vidrio?"],
)
def test_preguntas_de_verdad_siguen_yendo_a_claude(frase):
    i = CON_CLAUDE.parse(frase)
    assert i.mode is Mode.DESCRIBE and i.rule == "pregunta", (frase, i)


def test_objeto_desconocido_no_va_a_claude():
    i = CON_CLAUDE.parse("Guiame al pizarrón.")
    assert i.mode is Mode.SEARCH and not i.ok and "No sé buscar pizarron" in i.reply
