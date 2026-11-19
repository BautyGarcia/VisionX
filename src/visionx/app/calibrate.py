"""«Calibrar distancia» (acordado con Bauty el 18/11 para T12 «CREStereo adaptable»).

La persona apunta a un objeto a distancia conocida y dice «calibrá, está a 2 metros». La app
elige el objeto detectado más cerca del centro en la última unión medida y llama al método
opcional del estimador de profundidad:

    calibrar_distancia(frame: StereoFrame, mask: np.ndarray | None, z_m: float) -> bool

`mask` es la máscara del objeto en la imagen del detector (None si el detector no da
máscaras). Si el estimador no tiene el método, la app lo dice: la escala de los estimadores
actuales sale de la calibración y no se corrige con una distancia.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from visionx.modes.common import ModeSettings, age_s, candidates, meters

_NUMBERS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}  # fmt: skip
_DIGITS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(metros?|m\b|centimetros?|cm\b)?")
_WORDS = re.compile(r"\b(" + "|".join(_NUMBERS) + r")\s+metros?(\s+y\s+medio)?")
_HALF = re.compile(r"\bmedio metro\b|\bmetro y medio\b")


def parse_meters(text: str) -> float | None:
    """«a 1,5 metros», «150 centímetros», «dos metros y medio», «metro y medio» -> metros."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in decomposed if not unicodedata.combining(c))
    if "medio metro" in t:
        return 0.5
    if "metro y medio" in t and not _WORDS.search(t):
        return 1.5
    m = _DIGITS.search(t)
    if m:
        value = float(m.group(1).replace(",", "."))
        unit = m.group(2) or ""
        return value / 100 if unit.startswith(("c", "cm")) else value
    m = _WORDS.search(t)
    if m:
        return _NUMBERS[m.group(1)] + (0.5 if m.group(2) else 0.0)
    return None


def build_calibrator(
    depth: Any, geometry: Any, spoken: Callable[[str], str], settings: ModeSettings | None = None
) -> Callable[[float, Any, int], list[str]]:
    """Función que el controlador llama con (metros, vista, ahora): devuelve qué decir."""
    s = settings or ModeSettings()

    def calibrate(z_m: float, view: Any, now_ns: int) -> list[str]:
        method = getattr(depth, "calibrar_distancia", None)
        if method is None:
            return [
                "Esta profundidad no se calibra con una distancia: usa la calibración de la cámara."
            ]
        if not 0.3 <= z_m <= 6.0:
            return ["Decime una distancia entre 30 centímetros y 6 metros."]
        joint = getattr(view, "measured", None)
        if joint is None or age_s(joint, now_ns) > s.max_measure_age_s:
            return ["Todavía no tengo una medición reciente. Probá de nuevo en un momento."]
        cands = [c for c in candidates(joint, None, geometry) if abs(c.angle_h_deg) <= s.side_deg]
        if not cands:
            return ["No veo un objeto en el centro. Apuntá a uno y repetí."]
        target = min(cands, key=lambda c: abs(c.angle_h_deg))
        masks = joint.detection.masks
        mask = masks[target.index] if masks is not None else None
        try:
            ok = bool(method(joint.frame, mask, float(z_m)))
        except Exception:  # el estimador decide; la app no se cae por una calibración
            ok = False
        name = spoken(target.label)
        if not ok:
            return [f"No pude calibrar con {name}. Probá con otro objeto."]
        return [f"Listo, calibré con {name} a {meters(z_m, s, approx=False)}."]

    return calibrate
