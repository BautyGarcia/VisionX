"""Estado global de la cámara dicho por voz (T07, para T12 «CREStereo adaptable»).

Acordado con Bauty el 18/11: el estimador de profundidad avisa el estado de la cámara en cada
`DepthResult` con `status=UNMEASURABLE` y un código en `reason` (puede seguir un detalle después
de «:», p. ej. "alineando: confianza 0,4"):

- `alineando`: la red de alineación todavía no convergió («ajustando la cámara»).
- `camara-desconocida`: ningún perfil cierra; sin datos de la cámara no hay metros.
- `no-medible`: incertidumbre alta en toda la imagen durante varios segundos.

La app lo dice una vez al entrar en el estado (si dura `hold_s`, para no anunciar un parpadeo),
lo recuerda cada `repeat_s` si sigue igual, y avisa cuando vuelve a medir. Los códigos y las
frases viven en `camera_status` de la config: renombrar un código no toca el código.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visionx.app.controller import Say
from visionx.shared.contracts import DepthStatus

DEFAULT_PHRASES = {
    "alineando": "Estoy ajustando la cámara. Esperá un momento para pedirme distancias.",
    "camara-desconocida": (
        "No reconozco esta cámara. Puedo decirte dónde están las cosas, pero no a cuántos metros."
    ),
    "no-medible": "Ahora no puedo medir distancias con esta cámara.",
}
OK = "ok"


@dataclass(frozen=True, slots=True)
class CameraStatusSettings:
    phrases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PHRASES))
    ready_phrase: str = "Listo, ya puedo medir distancias."
    hold_s: float = 1.0
    repeat_s: float = 60.0  # 0 = no recordar

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CameraStatusSettings:
        c = cfg.get("camera_status") or {}
        d = cls()
        return cls(
            phrases={str(k): str(v) for k, v in (c.get("frases") or d.phrases).items()},
            ready_phrase=str(c.get("lista", d.ready_phrase)),
            hold_s=float(c.get("hold_s", d.hold_s)),
            repeat_s=float(c.get("repeat_s", d.repeat_s)),
        )


def status_code(depth: Any, phrases: dict[str, str]) -> str:
    """Código del estado global de la cámara en un DepthResult; OK si mide o es otro motivo."""
    if depth is None or depth.status is not DepthStatus.UNMEASURABLE or not depth.reason:
        return OK
    code = str(depth.reason).split(":", 1)[0].strip()
    return code if code in phrases else OK


class CameraStatus:
    """Observador del controlador: `observe(view, now_ns) -> list[Say]`."""

    def __init__(self, settings: CameraStatusSettings | None = None) -> None:
        self.s = settings or CameraStatusSettings()
        self.announced = OK
        self._candidate = OK
        self._since_ns = 0
        self._said_ns = 0

    def observe(self, view: Any, now_ns: int) -> list[Say]:
        measured = getattr(view, "measured", None)
        depth = getattr(measured, "depth", None)
        if depth is None:
            return []  # sin medición nueva no hay nada que decir del estado
        state = status_code(depth, self.s.phrases)
        if state != self._candidate:
            self._candidate, self._since_ns = state, now_ns
        if now_ns - self._since_ns < self.s.hold_s * 1e9:
            return []
        if state != self.announced:
            previous, self.announced, self._said_ns = self.announced, state, now_ns
            if state == OK:
                return [Say(self.s.ready_phrase)] if previous != OK else []
            return [Say(self.s.phrases[state])]
        repeat = self.s.repeat_s > 0 and now_ns - self._said_ns >= self.s.repeat_s * 1e9
        if state != OK and repeat:
            self._said_ns = now_ns
            return [Say(self.s.phrases[state])]
        return []
