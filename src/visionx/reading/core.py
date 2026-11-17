"""Contratos locales de OCR y operaciones que comparten modo y evaluación.

Una caja pertenece siempre a la imagen BGR entregada al lector. No se interpreta como
una detección ni como geometría estéreo; no cambia schema 3.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles, origen arriba/izquierda

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score OCR fuera de [0, 1]")
        if not all(math.isfinite(x) for x in self.box) or not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("caja OCR inválida")


class TextReader(Protocol):
    model_id: str

    def read(self, image: np.ndarray) -> list[TextBlock]: ...


class FakeReader:
    """Sustituto rotulado: devuelve bloques elegidos, nunca mide precisión OCR."""

    model_id = "fake-reader"

    def __init__(self, blocks: Sequence[TextBlock] = ()) -> None:
        self.blocks = list(blocks)

    def read(self, image: np.ndarray) -> list[TextBlock]:
        return list(self.blocks)


@dataclass(frozen=True, slots=True)
class ReadingSettings:
    window_s: float = 0.4
    pre_s: float = 0.8
    early_sharpness: float = 90.0
    min_score: float = 0.65
    speech_min_score: float = 0.90
    line_height_ratio: float = 0.5
    region_gap_ratio: float = 0.15
    section_gap_ratio: float = 1.5
    # En el piloto ELP la rectificación mejora CER; conservar original configurable para
    # otras lentes/encuadres, donde el remapeo puede deformar letras o perder bordes.
    image_source: str = "rectified"
    max_age_s: float = 1.0
    border_px: float = 8.0
    small_text_px: float = 20.0
    max_chars: int = 160
    speech_max_blocks: int = 6
    speech_max_chars: int = 320
    speech_base_s: float = 0.3
    speech_s_per_char: float = 0.065
    weights_dir: str = "weights/rapidocr"
    threads: int = 4
    det_limit_px: int = 1280
    crop: str = "center"
    center_fraction: float = 0.6
    crop_margin: float = 0.15
    text_probe_px: int = 640

    def __post_init__(self) -> None:
        for name in (
            "window_s",
            "max_age_s",
            "line_height_ratio",
            "region_gap_ratio",
            "section_gap_ratio",
            "speech_s_per_char",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"reading.{name} debe ser positivo y finito")
        for name in (
            "border_px",
            "small_text_px",
            "speech_base_s",
            "crop_margin",
            "pre_s",
            "early_sharpness",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reading.{name} debe ser no negativo y finito")
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("reading.min_score fuera de [0, 1]")
        if (
            not math.isfinite(self.speech_min_score)
            or not self.min_score <= self.speech_min_score <= 1
        ):
            raise ValueError("reading.speech_min_score debe estar entre min_score y 1")
        if self.image_source not in ("original", "rectified"):
            raise ValueError("reading.image_source: original | rectified")
        if self.crop not in ("none", "center", "text", "object"):
            raise ValueError("reading.crop: none | center | text | object")
        if not math.isfinite(self.center_fraction) or not 0 < self.center_fraction <= 1:
            raise ValueError("reading.center_fraction fuera de (0, 1]")
        for name in (
            "max_chars",
            "speech_max_blocks",
            "speech_max_chars",
            "threads",
            "det_limit_px",
            "text_probe_px",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"reading.{name} debe ser entero positivo")
        if self.speech_max_chars < 24:
            raise ValueError("reading.speech_max_chars debe ser >=24 para el aviso de corte")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ReadingSettings:
        return cls(**(cfg.get("reading") or {}))


def normalize_text(text: str) -> str:
    """NFC y espacios uniformes; preserva acentos, números, puntuación y mayúsculas."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def sharpness(image: np.ndarray) -> float:
    """Varianza del Laplaciano (intensidad²/px⁴): destaca bordes finos y castiga desenfoque.

    Comparar solo frames de la misma ruta/resolución. El fondo con textura puede ganar a un
    envase: esta beta no tiene ROI de texto antes del OCR. No es una medida de legibilidad.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def reading_order(blocks: Sequence[TextBlock], settings: ReadingSettings) -> list[TextBlock]:
    """Separa regiones por corredores vacíos (XY-cut), luego ordena sus renglones.

    Primero secciones con separación amplia, luego columnas y separaciones menores.
    Un título que abarca columnas se separa por el espacio horizontal antes de buscar
    columnas debajo. El mínimo corredor escala con la altura mediana del texto.
    """
    return [
        b
        for region in _reading_regions(blocks, settings, separate_lines=True)
        for b in _line_order(region, settings)
    ]


def _reading_regions(
    blocks: Sequence[TextBlock], settings: ReadingSettings, *, separate_lines: bool
) -> list[list[TextBlock]]:
    clean = [
        TextBlock(normalize_text(b.text), b.score, b.box)
        for b in blocks
        if b.score >= settings.min_score and normalize_text(b.text)
    ]

    def regions(items: list[TextBlock]) -> list[list[TextBlock]]:
        if len(items) < 2:
            return [items]
        height = float(np.median([b.box[3] - b.box[1] for b in items]))
        gap_min = settings.region_gap_ratio * height
        candidates = {}
        for axis in (0, 1):
            ordered = sorted(items, key=lambda b: b.box[axis])
            edge = ordered[0].box[axis + 2]
            cuts = []
            for i, b in enumerate(ordered[1:], 1):
                gap = b.box[axis] - edge
                if gap >= gap_min:
                    cuts.append((gap, i))
                edge = max(edge, b.box[axis + 2])
            if cuts:
                gap, i = max(cuts)
                candidates[axis] = (gap, i, ordered)
        # Un título puede terminar justo sobre el gutter entre dos sellos. Separar
        # antes el encabezado evita leer marca + primer sello antes del título.
        if 1 in candidates and candidates[1][0] >= settings.section_gap_ratio * height:
            axis = 1
        elif 0 in candidates:
            axis = 0
        elif separate_lines and 1 in candidates:
            axis = 1
        else:
            return [items]
        _, i, ordered = candidates[axis]
        return regions(ordered[:i]) + regions(ordered[i:])

    # Para voz, conservar juntos los renglones de una columna hasta la próxima
    # sección amplia. La evaluación conserva exactamente sus cortes anteriores.
    return regions(clean) if clean else []


def _line_order(blocks: Sequence[TextBlock], settings: ReadingSettings) -> list[TextBlock]:
    return [b for line in _line_groups(blocks, settings) for b in line]


def _line_groups(blocks: Sequence[TextBlock], settings: ReadingSettings) -> list[list[TextBlock]]:
    # La menor altura impide que un título grande absorba dos renglones pequeños.
    lines: list[list[TextBlock]] = []
    for b in sorted(blocks, key=lambda b: ((b.box[1] + b.box[3]) / 2, b.box[0])):
        cy = (b.box[1] + b.box[3]) / 2
        height = b.box[3] - b.box[1]
        matches = []
        for i, line in enumerate(lines):
            center = float(np.median([(a.box[1] + a.box[3]) / 2 for a in line]))
            h = float(np.median([a.box[3] - a.box[1] for a in line]))
            delta = abs(center - cy)
            # Cajas que se solapan horizontalmente son renglones apilados, aunque
            # el detector haya inflado su altura (pasa en GRASAS / TOTALES).
            disjoint = all(b.box[0] >= a.box[2] or b.box[2] <= a.box[0] for a in line)
            if disjoint and delta <= settings.line_height_ratio * min(h, height):
                matches.append((delta, i))
        if matches:
            lines[min(matches)[1]].append(b)
        else:
            lines.append([b])
    lines.sort(key=lambda line: float(np.median([(b.box[1] + b.box[3]) / 2 for b in line])))
    return [sorted(line, key=lambda b: b.box[0]) for line in lines]


def speech_chunks(blocks: Sequence[TextBlock], max_chars: int) -> list[str]:
    """Un Say por texto/región, cortando por palabras sin perder caracteres."""
    chunks = []
    for block in blocks:
        chunk = ""
        for word in block.text.split():
            if chunk and len(chunk) + len(word) + 1 > max_chars:
                chunks.append(chunk)
                chunk = ""
            # Una palabra excepcionalmente larga se corta también para acotar síntesis.
            while len(word) > max_chars:
                if chunk:
                    chunks.append(chunk)
                    chunk = ""
                chunks.append(word[:max_chars])
                word = word[max_chars:]
            chunk = f"{chunk} {word}".strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _plausible_line(text: str, *, context: bool = False) -> str:
    """Filtro ortográfico de habla, sin diccionario ni correcciones.

    Palabras de 3–30 letras: >=80 % letras, vocal y solo letras con guion/apóstrofo
    internos. Tokens de 1–2 letras requieren otro token largo o numérico en el
    mismo renglón (incluye unidades como g). Números decimales y % se conservan.
    No certifica que una palabra exista: errores plausibles pueden pasar.
    """
    tokens = normalize_text(text).split()
    candidates: list[tuple[str, bool]] = []
    for token in tokens:
        token = token.strip('.,;:!?¿¡()[]{}«»"“”‘’…')
        if re.fullmatch(r"\d+(?:[.,]\d+)?%?", token):
            candidates.append((token, True))
            continue
        if token == "%":
            candidates.append((token, True))
            continue
        letters = [c for c in token if c.isalpha()]
        if not token or not 1 <= len(letters) <= 30 or len(letters) / len(token) < 0.8:
            continue
        if any(
            not c.isalpha()
            and not (
                c in "-'’"
                and 0 < i < len(token) - 1
                and token[i - 1].isalpha()
                and token[i + 1].isalpha()
            )
            for i, c in enumerate(token)
        ):
            continue
        folded = unicodedata.normalize("NFD", token.casefold())
        if len(letters) <= 2:
            candidates.append((token, False))
        elif any(c in "aeiou" for c in folded):
            candidates.append((token, True))
    context = context or any(long_or_number for _, long_or_number in candidates)
    return " ".join(t for t, long_or_number in candidates if long_or_number or context)


def spoken_chunks(blocks: Sequence[TextBlock], settings: ReadingSettings) -> list[str]:
    """Región del título (mayor altura, desempate por área) primero; resto visual.

    Límites totales independientes del tamaño por Say. El aviso ocupa parte del tope
    de caracteres y no propone diálogo. Las métricas siempre usan TODOS los bloques.
    """
    regions = []
    for region in _reading_regions(blocks, settings, separate_lines=False):
        filtered = []
        confident = [b for b in region if b.score >= settings.speech_min_score]
        for line in _line_groups(confident, settings):
            context = bool(_plausible_line(" ".join(b.text for b in line)))
            filtered.extend(
                TextBlock(text, b.score, b.box)
                for b in line
                if (text := _plausible_line(b.text, context=context))
            )
        if filtered:
            regions.append(filtered)
    if not regions:
        return []
    title = max(
        range(len(regions)),
        key=lambda i: max(
            (b.box[3] - b.box[1], (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])) for b in regions[i]
        ),
    )
    ordered = [regions[title]] + [r for i, r in enumerate(regions) if i != title]
    chosen = ordered[: settings.speech_max_blocks]
    full = speech_chunks(
        [TextBlock(" ".join(b.text for b in r), 1.0, r[0].box) for r in chosen], settings.max_chars
    )
    truncated = len(chosen) < len(ordered) or sum(map(len, full)) > settings.speech_max_chars
    if not truncated:
        return full
    suffix = "…y más texto chico."
    remaining = settings.speech_max_chars - len(suffix)
    chunks = []
    for text in full:
        if len(text) <= remaining:
            chunks.append(text)
            remaining -= len(text)
        else:
            fragment = text[:remaining]
            if remaining < len(text) and text[remaining : remaining + 1] != " ":
                fragment = fragment.rsplit(" ", 1)[0] if " " in fragment else ""
            if fragment.strip():
                chunks.append(fragment.strip())
            break
    # Acotar también cada Say incluso si se configuró max_chars menor que el aviso.
    chunks.extend(speech_chunks([TextBlock(suffix, 1.0, (0, 0, 1, 1))], settings.max_chars))
    return chunks


def framing_hint(blocks: Sequence[TextBlock], shape: tuple[int, ...], s: ReadingSettings) -> str:
    """Guía solo con evidencia de cajas, incluso las descartadas por score.

    Es una sugerencia de encuadre, no una certeza de que exista texto fuera de campo.
    Sin cajas no se infiere ni distancia ni movimiento.
    """
    h, w = shape[:2]
    for b in blocks:
        if b.box[0] <= s.border_px:
            return "Probá encuadrar un poco hacia la izquierda."
        if b.box[2] >= w - s.border_px:
            return "Probá encuadrar un poco hacia la derecha."
        if b.box[1] <= s.border_px:
            return "Probá encuadrar un poco hacia arriba."
        if b.box[3] >= h - s.border_px:
            return "Probá encuadrar un poco hacia abajo."
    if blocks and float(np.median([b.box[3] - b.box[1] for b in blocks])) < s.small_text_px:
        return "Acercalo."
    return ""
