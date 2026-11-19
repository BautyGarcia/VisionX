"""Modo buscar y guiar (T09).

Estados de una búsqueda:

    buscando --(presencia confirmada, uno)--> guiando
    buscando --(presencia confirmada, varios iguales)--> eligiendo --(elección)--> guiando
    guiando --(sin verlo durante lost_s)--> buscando   ("Perdí la taza.")
    guiando --(al frente y a arrive_m o menos)--> llegó  ("Llegaste..." + cómo terminar)
    llegó --(se aleja más de arrive_m + leave_margin_m)--> guiando
    fin: cuando la persona dice «la encontré», «listo» o «terminá» (controlador), o cancela.
    Sin verlo: «Sigo buscando…» cada searching_reminder_s.
    Con plazo (deadlines_s.search > 0, banco M1) o end_on_arrival, el cierre automático de antes:
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

import math
from collections import deque
from collections.abc import Callable
from dataclasses import replace
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
    meters,
    speech_s,
    vertical,
)
from visionx.modes.target import Target, TargetMemory, plural
from visionx.shared.contracts import Command, Priority
from visionx.shared.geometry import StereoGeometry

BIP = "sonido:bip"
SEARCHING, CHOOSING, GUIDING = "buscando", "eligiendo", "guiando"
HOW_TO_END = "Cuando lo tengas, decí: listo."  # se agrega a la primera «Llegaste»


class SearchHandler:
    def __init__(
        self,
        geometry: StereoGeometry,
        spoken: Callable[[str], str],
        settings: ModeSettings | None = None,
        memory: TargetMemory | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        assistant: Any = None,
    ) -> None:
        self.geometry = geometry
        # Con asistente, al ver varios iguales: «Veo una botella verde a la izquierda y una
        # transparente a la derecha. ¿Cuál querés?» y se puede elegir «la verde».
        self.assistant = assistant
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
        # Lo último dicho en la guía: dónde y qué Z (para no repetir lo mismo con otras palabras).
        self._said_where: str | None = None
        self._said_z: float | None = None
        # Lo que trae la frase que se está por decir (lo arma `_guide_fact`).
        self._fact_where: str | None = None
        self._fact_measure: Measure | None = None
        # Opciones descriptas por el asistente: (ángulo al preguntar, descripción) por objeto.
        if getattr(self, "_options_job", None) is not None:
            self._options_job.cancel()
        self._options_job: Any = None
        self._options_start_ns = 0
        self._options_angles: list[float] = []
        self._options: list[tuple[float, str]] = []
        self._options_question: str | None = None
        # Estilo «centrar»: instrucción dicha, si ya se centró y la descripción del apoyo.
        self._instr: str | None = None
        self._h_zone = self._v_zone = 0
        self._centered = False
        if getattr(self, "_support_job", None) is not None:
            self._support_job.cancel()
        self._support_job: Any = None
        self._support_start_ns = 0
        self._support_done = False
        self._last_frame: Any = None
        self._arrived = False  # llegó y se queda callado hasta que se aleje o termine
        self._arrived_once = False  # el aviso de cómo terminar se dice una vez

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
            self._last_frame = seen.frame  # type: ignore[union-attr]
            cands = candidates(seen, label, self.geometry)  # type: ignore[arg-type]
            says += self._on_frame(cands, name, now_ns, seen.frame)  # type: ignore[union-attr]
        says += self._options_ready(name, now_ns)
        says += self._support_ready(now_ns)
        if not fresh:
            self._choices = []
            self._single_since = None
        if (
            self.state == CHOOSING
            and self._choices
            and self._options_job is None
            and self._can_ask(now_ns)
        ):
            again = self._options_question or (
                f"¿Cuál {name}? Decí izquierda, derecha, centro o la más cercana."
            )
            says.append(self._ask(again, now_ns))
        if self._arrived and self._moved_away(now_ns):
            self._arrived = False
            self._said_z = None  # vuelve a decir los metros enseguida
            self._event("se-alejo", {})
        if self.state == GUIDING and self.target is not None and not self._arrived:
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
        arrived = any(s.text_or_sound.startswith("Llegaste") for s in says)
        done = arrived and self.s.end_on_arrival
        if done:
            self._finished = True
            self._event("fin", {"motivo": "llegada"})
        elif arrived:
            self._event("llegada", {})
            if not self._arrived_once:
                says = [replace(s, text_or_sound=f"{s.text_or_sound} {HOW_TO_END}")
                        if s.text_or_sound.startswith("Llegaste") else s for s in says]  # fmt: skip
            self._arrived = self._arrived_once = True
        if (
            self.state == SEARCHING
            and not says
            and self.s.searching_reminder_s > 0
            and (now_ns - (self._last_say_ns or command.issued_ns)) / 1e9
            >= self.s.searching_reminder_s
        ):
            says.append(self._phrase(f"Sigo buscando {name}.", now_ns))
        if self.s.guide_style == "centrar":  # sin bips: solo palabras (pedido de Tomás)
            says = [s for s in says if s.text_or_sound != BIP]
        return [s for s in says if s.text_or_sound], done

    def _moved_away(self, now_ns: int) -> bool:
        """Después de llegar: ¿se alejó lo suficiente como para volver a guiar?"""
        m = self._fresh_measure(now_ns)
        return bool(m is not None and m.ok and m.z_m is not None
                    and m.z_m > self.s.arrive_m + self.s.leave_margin_m)  # fmt: skip

    def awaiting_choice(self) -> bool:
        """¿Está preguntando «¿cuál?» con opciones descriptas (se puede elegir «la verde»)?"""
        return self.state == CHOOSING and bool(self._options)

    def choose(self, command: Command, choice: str, view: View, now_ns: int) -> list[Say]:
        """Respuesta a "la de la izquierda" / "la más cercana" / "la verde" (con opciones)."""
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
        if choice not in ("izquierda", "derecha", "centro", "cerca"):
            option = self._match_option(choice)
            if option is None:
                return [Say("No entendí cuál. Decí cómo es o de qué lado está.")]
            angle, text = option
            chosen = min(self._choices, key=lambda c: abs(c.angle_h_deg - angle))
            self._event("eleccion", {"opcion": text, "angulo_h": round(chosen.angle_h_deg, 1)})
            return self._acquire(chosen, name, now_ns)
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

    def _match_option(self, utterance: str) -> tuple[float, str] | None:
        """La opción que comparte más palabras con lo dicho («la verde» -> «verde»)."""
        from visionx.voice.commands import clean

        said = set(clean(utterance).split()) - {"la", "el", "de", "que", "esta", "los", "las"}
        scored = [
            (len(said & set(clean(text).split())), angle, text) for angle, text in self._options
        ]
        best = max(scored, default=(0, 0.0, ""))
        if best[0] == 0 or sum(1 for s in scored if s[0] == best[0]) > 1:
            return None
        return best[1], best[2]

    def _ask_options(self, cands: list[Candidate], frame, now_ns: int) -> bool:
        """Pedir al asistente cómo distinguir los iguales. False si no hay asistente."""
        if self.assistant is None or not self.assistant.available or frame is None:
            return False
        g = self.geometry
        shown = cands[: self.s.max_listed]
        lines = []
        for i, c in enumerate(shown, 1):
            where = horizontal(c.angle_h_deg, self.s)
            x = (g.cx + g.fx * math.tan(math.radians(c.angle_h_deg))) / g.width
            lines.append(f"{i}: {self.spoken(c.label)} {where} (al {100 * x:.0f} % del ancho)")
        self._options_job = self.assistant.submit(
            self.request_id or 0, "opciones", frame.left, objects=lines
        )
        self._options_start_ns = now_ns
        self._options_angles = [c.angle_h_deg for c in shown]
        self._last_choice_ns = now_ns  # que no salga la pregunta fija mientras se consulta
        self._event("opciones-pedido", {"objetos": lines})
        return True

    def _options_ready(self, name: str, now_ns: int) -> list[Say]:
        job = self._options_job
        if job is None or not job.done.is_set():
            return []
        if job.error is None and now_ns < self._options_start_ns + int((job.total_s or 0) * 1e9):
            return []  # replay: todavía no pasó lo que tardó la API
        from visionx.assistant.core import parse_options

        self._options_job = None
        found, question = (
            parse_options(job.raw, len(self._options_angles)) if not job.error else ({}, None)
        )
        self._event("opciones", {"descripciones": found, "pregunta": question, "error": job.error})
        if self.state != CHOOSING:
            return []
        if found:
            self._options = [(self._options_angles[i], found[i]) for i in sorted(found)]
            self._options_question = (
                question or "¿Cuál querés? " + ", ".join(f"la {d}" for _, d in self._options) + "."
            )
            return [self._ask(self._options_question, now_ns)]
        listed = len(self._options_angles)
        return [self._ask(f"Veo {listed} {plural(name)}. ¿Cuál? "
                          "Decí izquierda, derecha, centro o la más cercana.", now_ns)]  # fmt: skip

    def _on_frame(self, cands: list[Candidate], name: str, now_ns: int, frame=None) -> list[Say]:
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
            if len(cands) > 1 and self._options_job is None and not self._options:
                self._ask_options(cands, frame, now_ns)  # «¿cuál?» también describiendo
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
            if (
                self._options_job is None
                and not self._options
                and self._ask_options(cands, frame, now_ns)
            ):
                return [Say(BIP, valid_until_ns=received + int(self.s.max_age_s * 1e9),
                            fact_received_ns=received)]  # fmt: skip
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
        self._said_where = self._said_z = None
        self.memory.remember(self.target)
        self._measure = None
        self._event("adquirido", {"angulo_h": round(c.angle_h_deg, 1), "track_id": c.track_id,
                                  "reaparicion": reacquired})  # fmt: skip
        self._instr, self._h_zone, self._v_zone, self._centered = None, 0, 0, False
        if self.s.guide_style == "centrar":
            return [self._phrase(f"Veo {name}.", now_ns)]  # la instrucción sale en el mismo tick
        text, valid_until, received = self._guide_fact(name, now_ns)
        return [
            Say(
                BIP,
                valid_until_ns=c.received_ns + int(self.s.max_age_s * 1e9),
                fact_received_ns=c.received_ns,
            ),
            self._said(self._phrase(text, now_ns, valid_until_ns=valid_until,
                                    fact_received_ns=received)),
        ]  # fmt: skip

    def sonar_target(self, now_ns: int) -> tuple[float, float] | None:
        """(ángulo horizontal, s desde que se lo vio) mientras se guía; None si no."""
        if self._finished or self.state != GUIDING or self.target is None:
            return None
        return self.target.angle_h_deg, (now_ns - self.target.last_seen_ns) / 1e9

    # -- estilo «centrar» -----------------------------------------------------------------

    def _zone(self, a: float, limits: tuple[float, ...], prev: int) -> int:
        """Zona 0..n de |a| con histéresis: cerca de un límite se conserva la anterior."""
        raw = sum(abs(a) > lim for lim in limits)
        if raw == prev:
            return raw
        h = self.s.hysteresis_deg
        low = sum(abs(a) > lim + h for lim in limits)
        high = sum(abs(a) > lim - h for lim in limits)
        return prev if low <= prev <= high else raw

    def _instruction(self, h: float, v: float) -> str | None:
        """«Girá a la izquierda.» … «Un poco más abajo.»; None si está centrado."""
        s = self.s
        self._h_zone = self._zone(h, (s.center_deg, s.more_deg, s.turn_deg), self._h_zone)
        self._v_zone = self._zone(v, (s.center_v_deg, s.low_deg), self._v_zone)
        side = "derecha" if h > 0 else "izquierda"
        if self._h_zone:
            return {1: f"Un poco a la {side}.", 2: f"Más a la {side}.",
                    3: f"Girá a la {side}."}[self._h_zone]  # fmt: skip
        if self._v_zone:
            where = "arriba" if v > 0 else "abajo"
            return f"Un poco más {where}." if self._v_zone == 1 else f"{where.capitalize()}."
        return None

    def _guide_center(self, name: str, now_ns: int) -> list[Say]:
        assert self.target is not None
        t = self.target
        valid_until = t.last_seen_ns + int(self.s.max_age_s * 1e9)
        since = None if self._last_say_ns is None else (now_ns - self._last_say_ns) / 1e9
        instr = self._instruction(t.angle_h_deg, t.angle_v_deg or 0.0)
        if instr is not None:
            first = self._instr is None  # primera tras «Veo …» o tras centrar: va en cola
            changed = instr != self._instr
            if changed and not first and since is not None and since < self.s.preempt_min_s:
                return []
            if not changed and since is not None and since < self.s.center_repeat_s:
                return []
            self._instr = instr
            say = self._phrase(instr, now_ns, valid_until_ns=valid_until,
                               fact_received_ns=t.last_seen_ns)  # fmt: skip
            if not say.text_or_sound:
                return []
            return [say] if first else [replace(say, replace=True)]
        self._instr = None
        m = self._fresh_measure(now_ns)
        z = m.z_m if m is not None and m.ok else None
        close = m is not None and (
            m.reason == "muy-cerca" or (z is not None and z <= self.s.arrive_m)
        )
        if close:
            text = f"Llegaste: {name} adelante, {measure_phrase(m, self.s)}."  # type: ignore[arg-type]
            say = self._phrase(text, now_ns, valid_until_ns=valid_until)
            return [replace(say, replace=True)] if say.text_or_sound else []
        if not self._centered:
            self._centered = True
            text = "Centrado." + (
                f" Está a {meters(z, self.s)}, adelante." if z is not None else ""
            )
            self._said_z = z
            self._event("centrado", {"z_m": z})
            self._ask_support(name, now_ns)
            say = self._phrase(text, now_ns, valid_until_ns=valid_until)
            return [replace(say, replace=True)] if say.text_or_sound else []
        if z is None:
            return []
        if self._said_z is None:  # centrado sin medición todavía: la primera, enseguida
            self._said_z = z
            text = f"Está a {meters(z, self.s)}, adelante."
            say = self._phrase(text, now_ns, valid_until_ns=valid_until)
            return [say] if say.text_or_sound else []
        if not self._distance_news(m):  # type: ignore[arg-type]
            return []
        if since is not None and since < self.s.approach_meters_s:
            return []
        self._said_z = z
        say = self._phrase(f"A {meters(z, self.s)}.", now_ns, valid_until_ns=valid_until)
        return [say] if say.text_or_sound else []

    def _ask_support(self, name: str, now_ns: int) -> None:
        """Una vez centrado: «Está sobre la mesa de madera, al lado de una taza.»"""
        a, frame = self.assistant, self._last_frame
        if a is None or not a.available or frame is None:
            return
        if self._support_job is not None or self._support_done:
            return
        self._support_job = a.submit(self.request_id or 0, "apoyo", frame.left,
                                     objects=[f"{name} en el centro de la imagen"])  # fmt: skip
        self._support_start_ns = now_ns

    def _support_ready(self, now_ns: int) -> list[Say]:
        job = self._support_job
        if job is None or not job.done.is_set() or job.text == "" and job.error is None:
            return []
        if job.error is None and now_ns < self._support_start_ns + int((job.total_s or 0) * 1e9):
            return []  # replay: todavía no pasó lo que tardó la API
        self._support_job, self._support_done = None, True  # una sola vez por objetivo
        self._event("apoyo", {"texto": job.text, "error": job.error})
        if job.error or self.state != GUIDING:
            return []
        return [Say(job.text, Priority.SHORT)]

    def _guide(self, name: str, now_ns: int) -> list[Say]:
        if self.s.guide_style == "centrar":
            return self._guide_center(name, now_ns)
        text, valid_until, received = self._guide_fact(name, now_ns)
        if text == "" or text.startswith("Llegaste"):
            if not text:
                return []
            say = self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received)
            # «Llegaste» es la frase final: corta la guía que todavía suena (no se encola detrás).
            return [replace(say, replace=True)] if say.text_or_sound else [say]
        since = None if self._last_say_ns is None else (now_ns - self._last_say_ns) / 1e9
        if self.s.guide_style == "sonar":
            # La dirección la da el sonar; en palabras, solo metros nuevos cada tanto.
            m = self._fact_measure
            if m is None or not self._distance_news(m):
                return []
            if since is not None and since < self.s.sonar_meters_s:
                return []
            short = measure_phrase(m, self.s)
            say = self._phrase(short[0].upper() + short[1:] + ".", now_ns,
                               valid_until_ns=valid_until, fact_received_ns=received)  # fmt: skip
            return [self._said(say)]
        # Ni antes del mínimo ni mientras la frase anterior todavía suena (no encolar guía vieja).
        speaking = speech_s(self._last_phrase or "", self.s) + 0.2
        if since is not None and since < max(self.s.min_interval_s, speaking):
            # Salvo que la dirección cambió: lo que suena ya es falso. Se corta y se dice la
            # nueva (las zonas tienen histéresis; preempt_min_s evita alternar en un límite).
            changed = self._said_where is not None and self._fact_where != self._said_where
            if not changed or since < self.s.preempt_min_s:
                return []
            say = self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received)
            if not say.text_or_sound:
                return []
            self._event("guia-reemplazada", {"antes": self._said_where, "ahora": self._fact_where})
            return [self._said(replace(say, replace=True))]
        m = self._fact_measure
        if since is None or since >= self.s.repeat_s or self._fact_where != self._said_where:
            say = self._phrase(text, now_ns, valid_until_ns=valid_until, fact_received_ns=received)
            return [self._said(say)]
        if m is not None and self._distance_news(m):
            # Misma dirección y metros nuevos: solo los metros, no la frase entera otra vez.
            short = measure_phrase(m, self.s)
            say = self._phrase(short[0].upper() + short[1:] + ".", now_ns,
                               valid_until_ns=valid_until, fact_received_ns=received)  # fmt: skip
            return [self._said(say)]
        return []

    def _distance_news(self, m: Measure) -> bool:
        """¿Vale la pena decir esta medida? Sí si no se dijo ninguna o si cambió lo suficiente;
        un cambio chico (redondeo, borde del rango) no es noticia."""
        if m.z_m is None:
            return False
        if self._said_z is None:
            return True
        delta = abs(m.z_m - self._said_z)
        return delta >= max(self.s.distance_change_m, self.s.distance_change_frac * self._said_z)

    def _said(self, say: Say) -> Say:
        """Registra qué quedó dicho de la guía (dirección y Z) si la frase sale."""
        if say.text_or_sound:
            self._said_where = self._fact_where
            if self._fact_measure is not None and self._fact_measure.z_m is not None:
                self._said_z = self._fact_measure.z_m
        return say

    def _guide_fact(self, name: str, now_ns: int) -> tuple[str, int, int]:
        assert self.target is not None
        valid_until = self.target.last_seen_ns + int(self.s.max_age_s * 1e9)
        # Zonas con histéresis respecto de lo último dicho (no alternar en un límite).
        h_word = horizontal(self.target.angle_h_deg, self.s, self._h_word)
        v_word = vertical(self.target.angle_v_deg, self.s, self._v_word)
        self._h_word, self._v_word = h_word, v_word
        where = h_word + v_word
        self._fact_where, self._fact_measure = where, None
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
                self._fact_measure = m
                return (f"{self._lead(name, where)}, {measure_phrase(m, self.s)}.",
                        min(valid_until, measured_until), received)  # fmt: skip
        return self._lead(name, where) + ".", valid_until, self.target.last_seen_ns

    def _lead(self, name: str, where: str) -> str:
        """«Taza a la izquierda» la primera vez; después solo «A la izquierda» (short_guide)."""
        if self.s.short_guide and self._said_where is not None:
            return where[0].upper() + where[1:]
        return f"{name.capitalize()} {where}"

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
            if any(previous.match([choice], self.s) is not None for choice in self._choices):
                # Hay asociaciones plausibles pero ambiguas: no saltar a una más lejana.
                return None
        return None

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self._on_event(f"buscar-{kind}", {"request_id": self.request_id, **data})
