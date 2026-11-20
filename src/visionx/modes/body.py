"""El cuerpo de quien usa VisionX: su altura y dónde está la cámara (en el pecho).

Sirve para decir la altura de un objeto respecto del cuerpo de ESA persona («a la altura de tu
cintura»): no es lo mismo para alguien de 1,50 m que de 1,90 m. Dos fuentes:

- **La altura de la persona:** se dice una vez («mido 1,80») y se guarda en
  `~/.visionx/persona.json`. Sin ese dato se usa 1,70 m y la app la pide al arrancar.
- **La cámara:** altura sobre el piso e inclinación, estimadas del piso con la profundidad
  (`floor.FloorTracker`); sin piso visible, la cámara a ~72 % de la altura (arnés de pecho) y
  nivelada.

Proporciones del cuerpo (fracciones de la estatura, antropometría de adultos): centro de la
cabeza ~0,93, hombros ~0,82, pecho ~0,72, ombligo ~0,60, cadera ~0,48, rodilla ~0,28. Cada banda
va de la mitad entre dos referencias a la mitad entre las siguientes.
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_HEIGHT_M = 1.70
CHEST_FRACTION = 0.72  # altura de un arnés de pecho respecto de la estatura
MIN_HEIGHT_M, MAX_HEIGHT_M = 1.0, 2.3

# (límite inferior como fracción de la estatura, frase); de arriba hacia abajo.
BANDS = (
    (1.03, "más arriba de tu cabeza"),
    (0.87, "a la altura de tu cabeza"),
    (0.66, "a la altura de tu pecho"),
    (0.54, "a la altura de tu cintura"),
    (0.38, "a la altura de tu cadera"),
    (0.14, "a la altura de tus rodillas"),
)


@dataclass(frozen=True, slots=True)
class Body:
    height_m: float = DEFAULT_HEIGHT_M
    height_known: bool = False  # la persona la dijo (si no, se usa el promedio)
    camera_height_m: float | None = None  # del piso; None = estimada por proporción
    pitch_deg: float = 0.0  # inclinación de la cámara: + mira arriba, − mira abajo
    floor_seen: bool = False

    @property
    def camera_m(self) -> float:
        return (
            self.camera_height_m
            if self.camera_height_m is not None
            else (CHEST_FRACTION * self.height_m)
        )

    def object_height_m(self, z_m: float, angle_v_deg: float) -> float:
        """Altura sobre el piso de un punto a Z axial y ángulo vertical en la imagen."""
        y_up = z_m * math.tan(math.radians(angle_v_deg))  # en el marco de la cámara
        p = math.radians(self.pitch_deg)
        return self.camera_m + y_up * math.cos(p) + z_m * math.sin(p)

    def band(self, z_m: float, angle_v_deg: float) -> str:
        frac = self.object_height_m(z_m, angle_v_deg) / self.height_m
        for limit, phrase in BANDS:
            if frac >= limit:
                return phrase
        return "cerca del piso"


_lock = threading.Lock()
_body = Body()


def current() -> Body:
    with _lock:
        return _body


def _store() -> Path:
    return Path.home() / ".visionx" / "persona.json"


def load() -> Body:
    """Lee la altura guardada (si hay) al arrancar la app."""
    global _body
    try:
        data = json.loads(_store().read_text(encoding="utf-8"))
        h = float(data["altura_m"])
        if MIN_HEIGHT_M <= h <= MAX_HEIGHT_M:
            with _lock:
                _body = replace(_body, height_m=h, height_known=True)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return current()


def set_height(height_m: float, *, persist: bool = True) -> Body:
    global _body
    if not MIN_HEIGHT_M <= height_m <= MAX_HEIGHT_M:
        raise ValueError(f"altura fuera de rango: {height_m}")
    with _lock:
        _body = replace(_body, height_m=height_m, height_known=True)
    if persist:
        path = _store()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"altura_m": round(height_m, 2)}), encoding="utf-8")
    return current()


def set_camera(height_m: float | None, pitch_deg: float, floor_seen: bool) -> Body:
    global _body
    with _lock:
        _body = replace(_body, camera_height_m=height_m, pitch_deg=pitch_deg,
                        floor_seen=floor_seen)  # fmt: skip
    return current()


def reset() -> None:
    """Para tests."""
    global _body
    with _lock:
        _body = Body()


# --- «mido 1,80» -----------------------------------------------------------------------------

_TENS = {"cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
         "noventa": 90}  # fmt: skip
_UNITS = {"uno": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
          "siete": 7, "ocho": 8, "nueve": 9}  # fmt: skip
_TEENS = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
          "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
          "treinta": 30}  # fmt: skip
_HEIGHT_START = re.compile(r"^(yo )?(mido|medo)\b|^mi (altura|estatura) es\b|^soy de\b")


def _plain(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def parse_height(text: str) -> float | None:
    """«mido 1,80», «mido 180», «un metro ochenta y cinco», «1.75 m» -> metros (o None)."""
    t = _plain(text)
    m = re.search(r"(\d+(?:[.,]\d+)?)", t)
    if m:
        v = float(m.group(1).replace(",", "."))
        v = v / 100 if v > 3 else v
        return v if MIN_HEIGHT_M <= v <= MAX_HEIGHT_M else None
    m = re.search(r"\b(un|uno|dos) metros?\b(?: y)?\s*(\w+)?(?: y (\w+))?", t)
    if not m:
        return None
    meters = 1 if m.group(1) in ("un", "uno") else 2
    cm = 0
    for word in (m.group(2), m.group(3)):
        if word in _TENS:
            cm += _TENS[word]
        elif word in _TEENS:
            cm += _TEENS[word]
        elif word in _UNITS:
            cm += _UNITS[word]
    v = meters + cm / 100
    return v if MIN_HEIGHT_M <= v <= MAX_HEIGHT_M else None


def is_height_statement(text: str) -> bool:
    return bool(_HEIGHT_START.search(_plain(text).strip(" ¿?¡!.,")))
