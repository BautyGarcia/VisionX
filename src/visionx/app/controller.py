"""Controlador de pedidos: de una frase a un modo activo, con vigencia y cancelación (T07).

Reglas (TECNICO T07):
- Cada orden recibe un request_id nuevo. Una orden nueva invalida la anterior: se cancela su
  audio pendiente y su modo deja de producir resultados.
- Cada orden vence a los `deadlines_s[modo]` segundos; vencida, no habla más.
- Cancelar no depende del modo: corta todo el audio cancelable y deja el controlador inactivo.
- Una frase que necesita aclaración se responde sin cancelar lo que estaba en curso.

Qué hace cada modo lo implementan T09 (buscar/distancia), T10 (leer) y T11 (describir) como
`ModeHandler`. Mientras no estén, el controlador avisa que la función no está disponible.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from visionx.app.runtime import Joint
from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import AudioEvent, Command, Mode, Priority
from visionx.shared.interfaces import AudioSink
from visionx.voice.commands import (
    CANCEL,
    REPEAT,
    CommandParser,
    Interpretation,
    acknowledgement,
)


@dataclass(frozen=True, slots=True)
class Say:
    """Algo que un modo quiere decir. El controlador le pone request_id y vigencia."""

    text_or_sound: str
    priority: Priority = Priority.SHORT
    cancelable: bool = True


class ModeHandler(Protocol):
    """Comportamiento de un modo. Recibe la unión más reciente en cada tick."""

    def start(self, command: Command, joint: Joint | None) -> list[Say]: ...

    def tick(self, command: Command, joint: Joint | None, now_ns: int) -> tuple[list[Say], bool]:
        """(qué decir, terminó)."""
        ...


class UnavailableHandler:
    """Modo todavía no implementado: lo dice y termina, sin inventar una respuesta."""

    def start(self, command: Command, joint: Joint | None) -> list[Say]:
        return [Say("Esa función todavía no está disponible.")]

    def tick(self, command: Command, joint: Joint | None, now_ns: int) -> tuple[list[Say], bool]:
        return [], True


class Controller:
    def __init__(
        self,
        parser: CommandParser,
        output: AudioSink,
        *,
        latest_joint: Callable[[], Joint | None] = lambda: None,
        handlers: dict[Mode, ModeHandler] | None = None,
        deadlines_s: dict[str, float] | None = None,
        clock: Clock | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.parser = parser
        self.output = output
        self.latest_joint = latest_joint
        self.handlers = handlers or {}
        self.deadlines_s = deadlines_s or {}
        self.clock = clock or MonotonicClock()
        self._on_event = on_event or (lambda kind, data: None)
        self._lock = threading.RLock()
        self._next_id = 0
        self.active: Command | None = None
        self.last_spoken: str | None = None
        self.history: list[tuple[Command | None, Interpretation]] = []

    # -- entrada -----------------------------------------------------------------------------

    def handle_text(self, utterance: str) -> Interpretation:
        """Frase del STT o del teclado."""
        interp = self.parser.parse(utterance)
        self.handle(interp)
        return interp

    def handle(self, interp: Interpretation) -> Command | None:
        with self._lock:
            now = self.clock.now_ns()
            self._on_event(
                "orden",
                {"texto": interp.utterance, "modo": interp.mode, "objetivo": interp.target,
                 "regla": interp.rule, "aclaracion": interp.reply},
            )  # fmt: skip
            if interp.mode is Mode.CONTROL and interp.target == CANCEL:
                self.cancel()
                self.history.append((None, interp))
                return None
            if interp.mode is Mode.CONTROL and interp.target == REPEAT:
                if self.last_spoken:
                    self._say(Say(self.last_spoken), self._new_id(), now, remember=False)
                self.history.append((None, interp))
                return None
            if not interp.ok:
                # Aclarar no interrumpe lo que estaba en curso (puede ser un error del STT).
                self._say(Say(interp.reply or "No entendí."), self._new_id(), now)
                self.history.append((None, interp))
                return None

            if self.active is not None:
                self.output.cancel(self.active.request_id)  # la orden nueva invalida la vieja
            request_id = self._new_id()
            deadline_s = self.deadlines_s.get(interp.mode.value)  # type: ignore[union-attr]
            deadline = now + int(deadline_s * 1e9) if deadline_s else None
            command = interp.to_command(request_id, now, deadline)
            self.active = command
            self.history.append((command, interp))
            ack = acknowledgement(interp, self.parser.catalog)
            if ack:
                self._say(Say(ack), request_id, now)
            handler = self.handlers.get(command.mode, UnavailableHandler())
            for say in handler.start(command, self.latest_joint()):
                self._say(say, request_id, now)
            return command

    def cancel(self) -> None:
        """El botón de cancelar (o la orden "cancelar")."""
        with self._lock:
            self.output.cancel_all()
            previous = self.active
            self.active = None
            self._on_event("cancelado", {"request_id": previous.request_id if previous else None})
            self.output.submit(
                AudioEvent(self._new_id(), Priority.SHORT, "sonido:cancelado", self.clock.now_ns())
            )

    def tick(self) -> None:
        """Llamar periódicamente (por ejemplo, en cada cuadro del dashboard)."""
        with self._lock:
            command = self.active
            if command is None:
                return
            now = self.clock.now_ns()
            if command.deadline_ns is not None and now > command.deadline_ns:
                self.active = None
                self._on_event("vencido", {"request_id": command.request_id})
                return
            handler = self.handlers.get(command.mode, UnavailableHandler())
            says, done = handler.tick(command, self.latest_joint(), now)
            for say in says:
                self._say(say, command.request_id, now)
            if done:
                self.active = None
                self._on_event("terminado", {"request_id": command.request_id})

    # -- interno ---------------------------------------------------------------------------

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _say(self, say: Say, request_id: int, now: int, *, remember: bool = True) -> None:
        deadline = None
        if self.active is not None and self.active.request_id == request_id:
            deadline = self.active.deadline_ns
        self.output.submit(
            AudioEvent(
                request_id,
                say.priority,
                say.text_or_sound,
                created_ns=now,
                deadline_ns=deadline,
                cancelable=say.cancelable,
            )
        )
        if remember and not say.text_or_sound.startswith("sonido:"):
            self.last_spoken = say.text_or_sound
