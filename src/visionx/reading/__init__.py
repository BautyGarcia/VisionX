"""OCR T10, independiente de los modos y sin importar motores opcionales al arrancar."""

from visionx.reading.core import (
    FakeReader,
    ReadingSettings,
    TextBlock,
    TextReader,
    normalize_text,
    reading_order,
    sharpness,
)
from visionx.reading.rapid import RapidReader

__all__ = ["FakeReader", "RapidReader", "ReadingSettings", "TextBlock", "TextReader",
           "normalize_text", "reading_order", "sharpness"]  # fmt: skip
