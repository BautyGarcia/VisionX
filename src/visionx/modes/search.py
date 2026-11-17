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
- Se conserva la selección tras una pérdida. Solo se reasocia con continuidad suficiente;
  ante ambigüedad se suspende la guía y se pregunta con un intervalo configurable.

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
    closing_due,
    closing_text,
    horizontal,
    measure,
    measure_phrase,
    speech_s,
    vertical,
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
        self._selected: Target | None = None  # se conserva aun cuando no se puede guiar
        self._last_choice_ns: int | None = None
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
        self._closing_ns: int | None = None
        self._stall_reported = False
        self._h_word: str | None = None  # últimas zonas dichas, para la histéresis
        self._v_word: str | None = None

    # -- interfaz ModeHandler --------------------------------------------------------------

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self._reset(command)
        self.memory.remember(None)  # una búsqueda nueva reemplaza al objetivo anterior
        if command.target:
            self.memory.request(command.target, now_ns)
        says, _ = self.tick(command, view, now_ns)
        return says

    def tick(self, command: Command, view: View, now_ns: int) -> tuple[list[Say], bool]:
        if command.request_id != self.request_id:
            self._reset(command)
        if self._finished:
            return [], True
        label = command.target or ""
        name = self.spoken(label)
        closing = closing_text(
            command,
            command.issued_ns,
            [
                f"Terminé de buscar {name}." if self.found_once else f"No encontré {name}.",
                "Terminé de buscar." if self.found_once else "No lo encontré.",
                "Fin de búsqueda.",
            ],
            self.s,
        )
        self._closing_ns = (
            command.deadline_ns - int((speech_s(closing, self.s) + self.s.end_margin_s) * 1e9)
            if command.deadline_ns is not None
            else None
        )
        if closing_due(command, now_ns, closing, self.s):
            self._event("fin", {"motivo": "vencimiento", "encontrado": self.found_once})
            self._finished = True
            closing = closing_text(
                command, now_ns, [closing, "Fin de búsqueda."], self.s, reserve=False
            )
            return ([Say(closing)] if closing else []), True

        self._update_measure(view, label, now_ns)
        says: list[Say] = []
        seen = view.seen
        fresh = seen is not None and age_s(seen, now_ns) <= self.s.max_age_s
        if fresh and seen.frame.key != self._last_seen_key:  # type: ignore[union-attr]
            self._last_seen_key = seen.frame.key  # type: ignore[union-attr]
            cands = candidates(seen, label, self.geometry)  # type: ignore[arg-type]
            says += self._on_frame(cands, name, now_ns)
        if not fresh:
            self._choices = []
            self._single_since = None
        if self.state == CHOOSING and self._choices and self._can_ask(now_ns):
            says.append(
                self._ask(
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
                self._measure = None
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
        if view.seen is None or age_s(view.seen, now_ns) > self.s.max_age_s:
            return [Say(f"No veo {name} ahora. Sigo buscando.")]
        self._choices = candidates(view.seen, command.target, self.geometry)
        if not self._choices:
            return [Say(f"No veo {name} ahora. Sigo buscando.")]
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
            self._choices = []
            self._single_since = None
            return []
        received = cands[0].received_ns
        self._hits.append((received, len(cands)))
        while self._hits and (received - self._hits[0][0]) / 1e9 > self.s.confirm_s:
            self._hits.popleft()
        if self._selected is not None and len(self._hits) >= self.s.confirm_hits:
            match = self._selected.match(cands, self.s)
            if match is not None:
                return self._acquire(match, name, now_ns, continued=True)
            # Con selección previa, ni un único candidato dudoso autoriza cambiar de objeto.
            self.state, self._choices = CHOOSING, cands
            return []
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
            return (
                [
                    Say(
                        BIP,
                        valid_until_ns=received + int(self.s.max_age_s * 1e9),
                        fact_received_ns=received,
                    ),
                    self._ask(
                        f"Veo {listed} {more}{plural(name)}. ¿Cuál? "
                        "Decí izquierda, derecha, centro o la más cercana.",
                        now_ns,
                        valid_until_ns=received + int(self.s.max_age_s * 1e9),
                        fact_received_ns=received,
                    ),
                ]
                if self._can_ask(now_ns)
                else []
            )
        return self._acquire(cands[0], name, now_ns)

    def _acquire(
        self, c: Candidate, name: str, now_ns: int, *, continued: bool = False
    ) -> list[Say]:
        reacquired = self.found_once
        self.state, self.found_once = GUIDING, True
        if continued and self._selected is not None:
            self.target = self._selected
            self.target.update(c)
        else:
            self.target = Target(c.label, c.angle_h_deg, c.track_id, c.received_ns, c.angle_v_deg)
        self.target.remember_measure(c)
        self._selected = self.target
        self._choices = []
        self._h_word = self._v_word = None
        self.memory.remember(self.target)
        self._measure = None
        self._event("adquirido", {"angulo_h": round(c.angle_h_deg, 1), "track_id": c.track_id,
                                  "reaparicion": reacquired})  # fmt: skip
        text, valid_until, received = self._guide_fact(name, now_ns)
        return [
            Say(
                BIP,
                valid_until_ns=c.received_ns + int(self.s.max_age_s * 1e9),
                fact_received_ns=c.received_ns,
            ),
            self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received),
        ]

    def _guide(self, name: str, now_ns: int) -> list[Say]:
        text, valid_until, received = self._guide_fact(name, now_ns)
        if text == "" or text.startswith("Llegaste"):
            return (
                [self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received)]
                if text
                else []
            )
        since = None if self._last_say_ns is None else (now_ns - self._last_say_ns) / 1e9
        changed = text != self._last_phrase
        # Ni antes del mínimo ni mientras la frase anterior todavía suena (no encolar guía vieja).
        speaking = speech_s(self._last_phrase or "", self.s) + 0.2
        if since is not None and since < max(self.s.min_interval_s, speaking):
            return []
        if changed or since is None or since >= self.s.repeat_s:
            return [
                self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received)
            ]
        return []

    def _guide_fact(self, name: str, now_ns: int) -> tuple[str, int, int]:
        assert self.target is not None
        valid_until = self.target.last_seen_ns + int(self.s.max_age_s * 1e9)
        # Zonas con histéresis respecto de lo último dicho (no alternar en un límite).
        h_word = horizontal(self.target.angle_h_deg, self.s, self._h_word)
        v_word = vertical(self.target.angle_v_deg, self.s, self._v_word)
        self._h_word, self._v_word = h_word, v_word
        where = h_word + v_word
        # Metros y llegada con el objeto al frente o "un poco" a un lado (|h| <= side_deg): en
        # la demo de la silla, caminando hacia ella quedaba a 9–20° y nunca "llegaba".
        ahead = abs(self.target.angle_h_deg) <= self.s.side_deg
        m = self._fresh_measure(now_ns)
        if ahead and m is not None:
            assert self._measure is not None
            # Una frase compuesta vence cuando vence cualquiera de sus hechos.
            measured_until = self._measure[1] + int(self.s.max_measure_age_s * 1e9)
            received = min(self.target.last_seen_ns, self._measure[1])
            close = m.ok and m.z_m is not None and m.z_m <= self.s.arrive_m
            if close or m.reason == "muy-cerca":  # más cerca que el rango validado también llega
                return (f"Llegaste: {name} {where}, {measure_phrase(m, self.s)}.",
                        min(valid_until, measured_until), received)  # fmt: skip
            if m.ok or m.reason in ("muy-cerca", "muy-lejos"):
                return (f"{name.capitalize()} {where}, {measure_phrase(m, self.s)}.",
                        min(valid_until, measured_until), received)  # fmt: skip
        return f"{name.capitalize()} {where}.", valid_until, self.target.last_seen_ns

    def _phrase(
        self,
        text: str,
        now_ns: int,
        *,
        valid_until_ns: int | None = None,
        fact_received_ns: int | None = None,
    ) -> Say:
        if (
            self._closing_ns is not None
            and now_ns + int((speech_s(text, self.s) + self.s.end_margin_s) * 1e9)
            > self._closing_ns
        ):
            return Say("")  # dejar que drene la voz anterior antes de emitir el cierre
        self._last_phrase, self._last_say_ns = text, now_ns
        return Say(text, valid_until_ns=valid_until_ns, fact_received_ns=fact_received_ns)

    def _can_ask(self, now_ns: int) -> bool:
        return (
            self._last_choice_ns is None
            or (now_ns - self._last_choice_ns) / 1e9 >= self.s.choice_repeat_s
        )

    def _ask(
        self,
        text: str,
        now_ns: int,
        *,
        valid_until_ns: int | None = None,
        fact_received_ns: int | None = None,
    ) -> Say:
        self._last_choice_ns = now_ns
        return self._phrase(
            text, now_ns, valid_until_ns=valid_until_ns, fact_received_ns=fact_received_ns
        )

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
            self.target.remember_measure(match)
        else:
            self._measure = None  # la unión nueva ya no respalda el Z anterior

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
        # La profundidad puede ir detrás del video: elegir por Z en el par medido, pero
        # anunciar solo una instancia que todavía se pueda asociar en la vista fresca.
        for c in sorted(decible, key=lambda c: c.obj.z_m):  # type: ignore[union-attr]
            previous = Target(c.label, c.angle_h_deg, c.track_id, c.received_ns, c.angle_v_deg)
            match = previous.match(self._choices, self.s)
            if match is not None:
                return match
        return None

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self._on_event(f"buscar-{kind}", {"request_id": self.request_id, **data})
