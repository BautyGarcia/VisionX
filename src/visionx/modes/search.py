"""Modo buscar y guiar (T09).

Estados de una búsqueda:

    buscando --(presencia confirmada, uno)--> guiando
    buscando --(presencia confirmada, varios iguales)--> eligiendo --(elección)--> guiando
    guiando --(sin verlo durante lost_s)--> buscando   ("Perdí la taza.")
    guiando --(al frente y a arrive_m o menos)--> fin  ("Llegaste...")
    cualquiera --(se acerca el vencimiento)--> fin     ("No encontré..." / "Termino...")

Reglas (TECNICO T09):
- Presencia sin esperar metros: la detección sola alcanza para el bip y la dirección.
- La guía habla cuando cambia la dirección o cada `repeat_s`, nunca en cada frame, y con un
  mínimo de `min_interval_s` entre frases.
- Nada viejo: una observación con más de `max_age_s` no guía; un Z con más de
  `max_measure_age_s` no se dice. Si el sistema se atrasa, la guía calla; "Perdí" exige
  `lost_s` de video fresco sin el objetivo (un pico del detector no es una pérdida).
- Metros solo con la política SPEAK_METERS de quien ubica el objeto y dentro del rango.
- Con varios iguales se pregunta cuál; perdido el elegido, se suspende la guía y no se le
  asigna su identidad al primero parecido.

Todo lo que se dice sale con prioridad SHORT: el bip y la dirección deben sonar DESPUÉS del
"Busco taza." de confirmación, no cortarlo (una prioridad mayor lo interrumpiría).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from visionx.app.controller import Say
from visionx.app.runtime import Joint
from visionx.modes.common import (
    Candidate,
    Measure,
    ModeSettings,
    View,
    age_s,
    candidates,
    direction,
    horizontal,
    measure,
    measure_phrase,
    speech_s,
)
from visionx.modes.target import Target, TargetMemory, plural
from visionx.shared.contracts import Command
from visionx.shared.geometry import StereoGeometry

BIP = "sonido:bip"
SEARCHING, CHOOSING, GUIDING = "buscando", "eligiendo", "guiando"


class SearchHandler:
    def __init__(
        self,
        geometry: StereoGeometry,
        spoken: Callable[[str], str],
        settings: ModeSettings | None = None,
        memory: TargetMemory | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.geometry = geometry
        self.spoken = spoken
        self.s = settings or ModeSettings()
        self.memory = memory or TargetMemory()
        self._on_event = on_event or (lambda kind, data: None)
        self._reset(None)

    def _reset(self, command: Command | None) -> None:
        self.request_id = command.request_id if command else None
        self.state = SEARCHING
        self.target: Target | None = None
        self.found_once = False
        self._hits: deque[tuple[int, int]] = deque()  # (received_ns, cuántos) con candidatos
        self._single_since: int | None = None  # eligiendo: desde cuándo se ve uno solo
        self._choices: list[Candidate] = []
        self._last_seen_key = None
        self._last_measured_key = None
        self._measure: tuple[Measure, int] | None = None  # (medida, received_ns de su frame)
        self._last_phrase: str | None = None
        self._last_say_ns: int | None = None
        self._finished = False  # ya dijo su frase de cierre (puede pasar dentro de start())
        self._stall_reported = False

    # -- interfaz ModeHandler --------------------------------------------------------------

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self._reset(command)
        self.memory.remember(None)  # una búsqueda nueva reemplaza al objetivo anterior
        says, _ = self.tick(command, view, now_ns)
        return says

    def tick(self, command: Command, view: View, now_ns: int) -> tuple[list[Say], bool]:
        if command.request_id != self.request_id:
            self._reset(command)
        if self._finished:
            return [], True
        label = command.target or ""
        name = self.spoken(label)
        if command.deadline_ns is not None and now_ns >= command.deadline_ns - int(
            self.s.end_margin_s * 1e9
        ):
            self._event("fin", {"motivo": "vencimiento", "encontrado": self.found_once})
            self._finished = True
            return [Say(f"Terminé de buscar {name}." if self.found_once
                        else f"No encontré {name}.")], True  # fmt: skip

        self._update_measure(view, label, now_ns)
        says: list[Say] = []
        seen = view.seen
        fresh = seen is not None and age_s(seen, now_ns) <= self.s.max_age_s
        if fresh and seen.frame.key != self._last_seen_key:  # type: ignore[union-attr]
            self._last_seen_key = seen.frame.key  # type: ignore[union-attr]
            cands = candidates(seen, label, self.geometry)  # type: ignore[arg-type]
            says += self._on_frame(cands, name, now_ns)
        if self.state == CHOOSING and self._last_say_ns is not None:
            # Sin respuesta, se vuelve a preguntar de vez en cuando (no se elige por el usuario).
            if (now_ns - self._last_say_ns) / 1e9 >= 2 * self.s.repeat_s and self._choices:
                says.append(
                    self._phrase(
                        f"¿Cuál {name}? Decí izquierda, derecha, centro o la más cercana.", now_ns
                    )
                )
        if self.state == GUIDING and self.target is not None:
            # "Perdí" necesita evidencia de ausencia: frames frescos que cubran lost_s de video
            # sin el objetivo, medido en tiempo de FRAME. Si el sistema se atrasa (no llegan
            # frames frescos), la guía se suspende en silencio: no es que el objeto se fue.
            gap_s = (
                (seen.frame.received_ns - self.target.last_seen_ns) / 1e9  # type: ignore[union-attr]
                if fresh
                else 0.0
            )
            stalled = not fresh and (now_ns - self.target.last_seen_ns) / 1e9 > self.s.lost_s
            if stalled and not self._stall_reported:
                self._stall_reported = True
                self._event("demora", {"sin_frames_frescos_s": self.s.lost_s})
            if fresh:
                self._stall_reported = False
            if gap_s > self.s.lost_s:
                last = self.target.angle_h_deg
                self._event("perdido", {"angulo_h": round(last, 1)})
                self.state, self.target = SEARCHING, None
                self._hits.clear()
                self.memory.remember(None)
                # Si salió por un costado, decir hacia dónde quedó ayuda a volver a girar.
                where = horizontal(last, self.s)
                lost = (
                    f"Perdí {name}: quedó {where}."
                    if where.startswith("a la")
                    else f"Perdí {name}."
                )
                says.append(self._phrase(lost, now_ns))
            elif fresh and gap_s == 0:
                # Solo se guía en frames donde el objetivo está: mientras no se lo ve, repetir
                # su última dirección sería guiar con información vieja.
                says += self._guide(name, now_ns)
        done = any(s.text_or_sound.startswith("Llegaste") for s in says)
        if done:
            self._finished = True
            self._event("fin", {"motivo": "llegada"})
        return [s for s in says if s.text_or_sound], done

    def choose(self, command: Command, choice: str, view: View, now_ns: int) -> list[Say]:
        """Respuesta a "la de la izquierda" / "la más cercana" mientras se eligen iguales."""
        name = self.spoken(command.target or "")
        if self.state == GUIDING:
            return [Say(f"Ya te estoy guiando hacia {name}.")]
        if self.state != CHOOSING or not self._choices:
            return [Say("No hay nada para elegir ahora.")]
        chosen: Candidate | None
        if choice == "izquierda":
            chosen = self._choices[0]
        elif choice == "derecha":
            chosen = self._choices[-1]
        elif choice == "centro":
            chosen = min(self._choices, key=lambda c: abs(c.angle_h_deg))
        else:  # cerca: solo con distancias que se pueden decir
            chosen = self._nearest(command.target or "", view, now_ns)
            if chosen is None:
                return [
                    Say("No puedo saber cuál está más cerca. Decí izquierda, derecha o centro.")
                ]
        self._event("eleccion", {"opcion": choice, "angulo_h": round(chosen.angle_h_deg, 1)})
        return self._acquire(chosen, name, now_ns)

    # -- interno ---------------------------------------------------------------------------

    def _on_frame(self, cands: list[Candidate], name: str, now_ns: int) -> list[Say]:
        if self.state == GUIDING and self.target is not None:
            match = self.target.match(cands, self.s)
            if match is not None:
                self.target.update(match)
                self.memory.remember(self.target)
            return []
        if not cands:
            return []
        received = cands[0].received_ns
        self._hits.append((received, len(cands)))
        while self._hits and (received - self._hits[0][0]) / 1e9 > self.s.confirm_s:
            self._hits.popleft()
        if self.state == CHOOSING:
            self._choices = cands  # la elección se aplica a lo último que se vio
            # Si queda uno solo de forma estable, ya no hay entre qué elegir: se guía a ese.
            if len(cands) == 1:
                self._single_since = self._single_since or received
                if (received - self._single_since) / 1e9 >= self.s.confirm_s:
                    self._event(
                        "eleccion", {"opcion": "unico", "angulo_h": round(cands[0].angle_h_deg, 1)}
                    )
                    return self._acquire(cands[0], name, now_ns)
            else:
                self._single_since = None
            return []
        if len(self._hits) < self.s.confirm_hits:
            return []
        # "Varios" tiene que ser consistente en la ventana de confirmación: una detección suelta
        # de más en un frame (pasa con modelos sensibles) no justifica preguntar.
        several = min(n for _, n in self._hits) > 1
        if not several:
            return self._acquire(max(cands, key=lambda c: c.score), name, now_ns)
        if len(cands) > 1:
            self.state, self._choices = CHOOSING, cands
            self._event("varios", {"cantidad": len(cands)})
            listed = min(len(cands), self.s.max_listed)
            more = "o más " if len(cands) > listed else ""
            return [
                Say(BIP),
                self._phrase(
                    f"Veo {listed} {more}{plural(name)}. ¿Cuál? Decí izquierda, derecha, centro "
                    "o la más cercana.",
                    now_ns,
                ),
            ]
        return self._acquire(cands[0], name, now_ns)

    def _acquire(self, c: Candidate, name: str, now_ns: int) -> list[Say]:
        reacquired = self.found_once
        self.state, self.found_once = GUIDING, True
        self.target = Target(c.label, c.angle_h_deg, c.track_id, c.received_ns, c.angle_v_deg)
        self.memory.remember(self.target)
        self._measure = None
        self._event("adquirido", {"angulo_h": round(c.angle_h_deg, 1), "track_id": c.track_id,
                                  "reaparicion": reacquired})  # fmt: skip
        return [Say(BIP), self._phrase(self._guide_text(name, now_ns), now_ns)]

    def _guide(self, name: str, now_ns: int) -> list[Say]:
        text = self._guide_text(name, now_ns)
        if text == "" or text.startswith("Llegaste"):
            return [self._phrase(text, now_ns)] if text else []
        since = None if self._last_say_ns is None else (now_ns - self._last_say_ns) / 1e9
        changed = text != self._last_phrase
        # Ni antes del mínimo ni mientras la frase anterior todavía suena (no encolar guía vieja).
        speaking = speech_s(self._last_phrase or "", self.s) + 0.2
        if since is not None and since < max(self.s.min_interval_s, speaking):
            return []
        if changed or since is None or since >= self.s.repeat_s:
            return [self._phrase(text, now_ns)]
        return []

    def _guide_text(self, name: str, now_ns: int) -> str:
        assert self.target is not None
        # "Al frente" (para metros y llegada) es horizontal; lo vertical solo se agrega a la frase.
        front = horizontal(self.target.angle_h_deg, self.s) == "al frente"
        where = direction(self.target.angle_h_deg, self.s, self.target.angle_v_deg)
        m = self._fresh_measure(now_ns)
        if front and m is not None:
            close = m.ok and m.z_m is not None and m.z_m <= self.s.arrive_m
            if close or m.reason == "muy-cerca":  # más cerca que el rango validado también llega
                return f"Llegaste: {name} {where}, {measure_phrase(m, self.s)}."
            if m.ok or m.reason in ("muy-cerca", "muy-lejos"):
                return f"{name.capitalize()} {where}, {measure_phrase(m, self.s)}."
        return f"{name.capitalize()} {where}."

    def _phrase(self, text: str, now_ns: int) -> Say:
        self._last_phrase, self._last_say_ns = text, now_ns
        return Say(text)

    def _update_measure(self, view: View, label: str, now_ns: int) -> None:
        """Toma el Z del objetivo de la unión medida más nueva (mismo frame que su máscara)."""
        joint = view.measured
        if (
            joint is None
            or self.target is None
            or joint.frame.key == self._last_measured_key
            or age_s(joint, now_ns) > self.s.max_measure_age_s
        ):
            return
        self._last_measured_key = joint.frame.key
        match = self.target.match(candidates(joint, label, self.geometry), self.s)
        if match is not None:
            self._measure = (measure(match.obj, self.s), joint.frame.received_ns)

    def _fresh_measure(self, now_ns: int) -> Measure | None:
        if self._measure is None:
            return None
        m, received = self._measure
        return m if (now_ns - received) / 1e9 <= self.s.max_measure_age_s else None

    def _nearest(self, label: str, view: View, now_ns: int) -> Candidate | None:
        joint: Joint | None = view.measured
        if joint is None or age_s(joint, now_ns) > self.s.max_measure_age_s:
            return None
        decible = [c for c in candidates(joint, label, self.geometry) if measure(c.obj, self.s).ok]
        return min(decible, key=lambda c: c.obj.z_m) if decible else None  # type: ignore[union-attr]

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self._on_event(f"buscar-{kind}", {"request_id": self.request_id, **data})
