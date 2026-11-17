"""Controlador de pedidos: de una frase a un modo activo, con vigencia y cancelación (T07).

Reglas (TECNICO T07):
- Cada orden recibe un request_id nuevo. Una orden nueva invalida la anterior: se cancela su
  audio pendiente y su modo deja de producir resultados.
- Cada orden vence a los `deadlines_s[modo]` segundos; vencida, no habla más.
- Cancelar no depende del modo: corta todo el audio cancelable y deja el controlador inactivo.
- Una frase que necesita aclaración se responde sin cancelar lo que estaba en curso.

- Elegir ("la de la izquierda") no es una orden nueva: se entrega al modo activo si sabe
  elegir (buscar, con varios iguales a la vista). Sin nada para elegir, se dice.

Qué hace cada modo lo implementan T09 (buscar/distancia), T10 (leer) y T11 (describir) como
`ModeHandler`. Mientras no estén, el controlador avisa que la función no está disponible.
Los modos reciben una vista (`modes.common.View`): lo último visto (frame + detección, sin
esperar profundidad) y lo último medido (unión con profundidad del mismo frame).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

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
    """Comportamiento de un modo. `view` es lo último observado (modes.common.View)."""

    def start(self, command: Command, view: Any, now_ns: int) -> list[Say]: ...

    def tick(self, command: Command, view: Any, now_ns: int) -> tuple[list[Say], bool]:
        """(qué decir, terminó)."""
        ...


class UnavailableHandler:
    """Modo todavía no implementado: lo dice y termina, sin inventar una respuesta."""

    def start(self, command: Command, view: Any, now_ns: int) -> list[Say]:
        return [Say("Esa función todavía no está disponible.")]

    def tick(self, command: Command, view: Any, now_ns: int) -> tuple[list[Say], bool]:
        return [], True


class Controller:
    def __init__(
        self,
        parser: CommandParser,
        output: AudioSink,
        *,
        latest_view: Callable[[], Any] = lambda: None,
        handlers: dict[Mode, ModeHandler] | None = None,
        deadlines_s: dict[str, float] | None = None,
        clock: Clock | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.parser = parser
        self.output = output
        self.latest_view = latest_view
        self.handlers = handlers or {}
        self._audio_request_id: int | None = None
        self._audio_deadline_ns: int | None = None
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
            if interp.mode is Mode.CHOOSE:
                # Elegir completa la orden activa; no la reemplaza ni le cambia la vigencia.
                active = self.active
                handler = self.handlers.get(active.mode) if active is not None else None
                choose = getattr(handler, "choose", None)
                if active is None or choose is None:
                    self._say(Say("No hay nada para elegir ahora."), self._new_id(), now)
                else:
                    for say in choose(active, interp.target, self.latest_view(), now):
                        self._say(say, active.request_id, now)
                self.history.append((None, interp))
                return None

            if self._audio_request_id is not None:
                # También si el handler terminó pero su último bloque sigue sonando.
                self.output.cancel(self._audio_request_id)
            self._cancel_handler()
            request_id = self._new_id()
            deadline_s = self.deadlines_s.get(interp.mode.value)  # type: ignore[union-attr]
            deadline = now + int(deadline_s * 1e9) if deadline_s else None
            command = interp.to_command(request_id, now, deadline)
            self.active = command
            self._audio_request_id = request_id
            self._audio_deadline_ns = deadline
            self.history.append((command, interp))
            ack = acknowledgement(interp, self.parser.catalog)
            if command.mode is Mode.DISTANCE:
                # La respuesta ya nombra el objeto ("Taza a la izquierda, a 1,1 metros"): decir
                # antes "Mido la distancia a taza." la demoraba ~2 s (se ve en las demos de T09).
                # Un sonido corto confirma que se entendió sin hacer esperar el número.
                ack = "sonido:ok"
            if ack:
                self._say(Say(ack), request_id, now)
            handler = self.handlers.get(command.mode, UnavailableHandler())
            for say in handler.start(command, self.latest_view(), now):
                self._say(say, request_id, now)
            return command

    def cancel(self) -> None:
        """El botón de cancelar (o la orden "cancelar")."""
        with self._lock:
            self.output.cancel_all()
            self._audio_deadline_ns = None
            previous = self.active
            self._cancel_handler()
            self.active = None
            self._on_event("cancelado", {"request_id": previous.request_id if previous else None})
            self.output.submit(
                AudioEvent(self._new_id(), Priority.SHORT, "sonido:cancelado", self.clock.now_ns())
            )

    def tick(self) -> None:
        """Llamar periódicamente (por ejemplo, en cada cuadro del dashboard)."""
        with self._lock:
            now = self.clock.now_ns()
            if self._audio_deadline_ns is not None and now >= self._audio_deadline_ns:
                # Un modo puede haber terminado de emitir antes de acabar la reproducción.
                # La vigencia también corta su último bloque que aún esté sonando/en cola.
                self.output.cancel(self._audio_request_id)
                self._audio_deadline_ns = None
            view = self.latest_view()
            # Hook liviano: LEER conserva capturas previas aun sin una orden activa.
            # Cálculo de nitidez/modelos quedan fuera del lock, en su worker.
            for observer in self.handlers.values():
                observe = getattr(observer, "observe", None)
                if observe is not None:
                    observe(view, now)
            command = self.active
            if command is None:
                return
            if command.deadline_ns is not None and now >= command.deadline_ns:
                self._cancel_handler()
                self.active = None
                self._on_event("vencido", {"request_id": command.request_id})
                return
            handler = self.handlers.get(command.mode, UnavailableHandler())
            says, done = handler.tick(command, view, now)
            for say in says:
                self._say(say, command.request_id, now)
            if done:
                self.active = None
                self._on_event("terminado", {"request_id": command.request_id})

    # -- interno ---------------------------------------------------------------------------

    def _cancel_handler(self) -> None:
        """Hook opcional: invalidar selección/trabajo pendiente sin esperar inferencia."""
        if self.active is not None:
            cancel = getattr(self.handlers.get(self.active.mode), "cancel", None)
            if cancel is not None:
                cancel()

    def close(self) -> None:
        """Cerrar workers de modos al salir de la aplicación, sin esperar OCR nativo."""
        with self._lock:
            self._cancel_handler()
            self.active = None
            for handler in self.handlers.values():
                close = getattr(handler, "close", None)
                if close is not None:
                    close()

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
