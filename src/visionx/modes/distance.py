"""Modo distancia (T09): "¿a qué distancia está la taza?".

Reglas (TECNICO T09):
- **Congelar un par identificado.** Se usa la primera unión medida (detección + profundidad +
  objetos del MISMO frame) cuyo frame llegó después de la orden, o a lo sumo
  `accept_before_s` antes. Nunca se aplica una máscara nueva a una profundidad vieja ni se
  responde con un frame anterior a la pregunta.
- **Ningún metro sin profundidad válida.** El número sale solo si quien ubicó el objeto dijo
  SPEAK_METERS y Z cae en el rango validado; si no, se explica por qué no hay número.
- **Qué objeto.** El nombrado; si no se nombró, el que se estaba guiando hace poco; si no hay,
  el que está al frente. Con varios iguales y sin guía previa, se enumeran de izquierda a derecha.
- Si no llega una medición a tiempo, se dice; no se inventa.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from visionx.app.controller import Say
from visionx.app.runtime import Joint
from visionx.modes.common import (
    Candidate,
    ModeSettings,
    View,
    age_s,
    candidates,
    direction,
    measure,
    measure_phrase,
)
from visionx.modes.target import Target, TargetMemory, plural
from visionx.shared.contracts import Command
from visionx.shared.geometry import StereoGeometry

# Sin objeto nombrado ni guía: "al frente" tolera el doble del margen central.
FRONT_FACTOR = 2.0


class DistanceHandler:
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
        self.request_id: int | None = None
        self._label: str | None = None
        self._guided_target: Target | None = None
        self._answered = False

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self.request_id = command.request_id
        self._label = command.target
        self._answered = False  # start() puede responder ya; el tick siguiente solo cierra
        self._guided_target = None
        remembered = self.memory.recent(now_ns, self.s.target_memory_s)
        if self._label is None:
            self._label = self.memory.recent_label(now_ns, self.s.target_memory_s)
        if remembered is not None and (self._label is None or self._label == remembered.label):
            self._label = remembered.label
            self._guided_target = remembered
        if command.target:
            if remembered is None:
                self.memory.remember(None)
            self.memory.request(command.target, now_ns)
        says, _ = self.tick(command, view, now_ns)
        return says

    def tick(self, command: Command, view: View, now_ns: int) -> tuple[list[Say], bool]:
        if command.request_id != self.request_id or self._answered:
            return [], True
        joint = self._usable(view.measured, command, now_ns)
        if joint is not None:
            answer = self._answer(joint, view, now_ns)
            if answer is not None:
                self._answered = True
                return [Say(answer)], True
        if command.deadline_ns is not None and now_ns >= command.deadline_ns - int(
            self.s.end_margin_s * 1e9
        ):
            self._answered = True
            self._event("sin-medida", {"motivo": "vencimiento"})
            return [Say(self._late(view, now_ns))], True
        return [], False

    # -- interno ---------------------------------------------------------------------------

    def _usable(self, joint: Joint | None, command: Command, now_ns: int) -> Joint | None:
        if joint is None or joint.depth is None:
            return None
        if joint.frame.received_ns < command.issued_ns - int(self.s.accept_before_s * 1e9):
            return None  # frame anterior a la pregunta
        if age_s(joint, now_ns) > self.s.max_measure_age_s:
            return None
        return joint

    def _answer(self, joint: Joint, view: View, now_ns: int) -> str | None:
        """Frase final, o None si este frame no alcanza y conviene esperar el siguiente."""
        if self._label is None:
            return self._front(joint)
        name = self.spoken(self._label)
        cands = candidates(joint, self._label, self.geometry)
        if not cands:
            seen = view.seen
            still_visible = (
                seen is not None
                and age_s(seen, now_ns) <= self.s.max_age_s
                and candidates(seen, self._label, self.geometry)
            )
            if still_visible:
                return None  # se lo ve pero no en este frame medido: esperar el próximo
            self._event("sin-objeto", {"frame_id": joint.frame.frame_id})
            return f"No veo {name}."
        if self._guided_target is not None:
            chosen = self._guided_target.match(cands, self.s)
            if chosen is None:
                self._event("ambiguo", {"frame_id": joint.frame.frame_id})
                return f"No puedo distinguir {name}. Decí buscá {name} para elegir."
            return self._single(chosen, name, joint)
        if len(cands) == 1:
            return self._single(cands[0], name, joint)
        return self._several(cands, name, joint)

    def _single(self, c: Candidate, name: str, joint: Joint) -> str:
        target = Target(c.label, c.angle_h_deg, c.track_id, c.received_ns, c.angle_v_deg)
        target.remember_measure(c)
        self.memory.remember(target)
        m = measure(c.obj, self.s)
        self._event("respuesta", self._data(joint, c, m.z_m, m.reason))
        where = direction(c.angle_h_deg, self.s, c.angle_v_deg)
        return f"{name.capitalize()} {where}, {measure_phrase(m, self.s)}."

    def _several(self, cands: list[Candidate], name: str, joint: Joint) -> str:
        shown = cands[: self.s.max_listed]
        parts = []
        for c in shown:
            m = measure(c.obj, self.s)
            self._event("respuesta", self._data(joint, c, m.z_m, m.reason))
            where = direction(c.angle_h_deg, self.s, c.angle_v_deg)
            parts.append(f"{where}, {measure_phrase(m, self.s)}")
        return f"Veo {len(cands)} {plural(name)}: " + "; ".join(parts) + "."

    def _front(self, joint: Joint) -> str:
        cands = [
            c
            for c in candidates(joint, None, self.geometry)
            if abs(c.angle_h_deg) <= self.s.center_deg * FRONT_FACTOR
        ]
        if not cands:
            self._event("sin-objeto", {"frame_id": joint.frame.frame_id, "al_frente": True})
            return "No veo nada al frente que pueda medir."
        c = min(cands, key=lambda c: abs(c.angle_h_deg))
        return self._single(c, self.spoken(c.label), joint)

    def _late(self, view: View, now_ns: int) -> str:
        seen = view.seen
        if self._label is not None and seen is not None and age_s(seen, now_ns) <= self.s.max_age_s:
            cands = candidates(seen, self._label, self.geometry)
            if cands:
                where = direction(cands[0].angle_h_deg, self.s, cands[0].angle_v_deg)
                return (
                    f"Veo {self.spoken(self._label)} {where}, pero no pude medir la distancia "
                    "a tiempo."
                )
        return "No pude medir a tiempo."

    def _data(self, joint: Joint, c: Candidate, z: float | None, reason: str | None) -> dict:
        return {
            "frame_id": joint.frame.frame_id,
            "clase": c.label,
            "angulo_h": round(c.angle_h_deg, 1),
            "z_m": z,
            "motivo": reason,
            "profundidad": joint.depth.model_id if joint.depth else None,
        }

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self._on_event(f"distancia-{kind}", {"request_id": self.request_id, **data})
