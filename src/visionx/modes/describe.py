"""DESCRIBIR beta: una escena congelada, sin VLM ni generación de metros.

`SceneFacts` separa presencia de mediciones aceptadas del MISMO frame. El punto de extensión
`SceneNarrator` solo propone grupos de índices existentes, nunca texto libre ni distancias.
Un futuro VLM podrá proponer contexto no métrico con un contrato acotado y validado aparte;
no debe reemplazar el compositor de números/unidades que vive en este módulo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol

from visionx.app.controller import Say
from visionx.app.runtime import Joint, View
from visionx.modes.common import (
    Measure,
    ModeSettings,
    age_s,
    candidates,
    horizontal,
    measure,
    measure_phrase,
)
from visionx.modes.target import Target, plural
from visionx.shared.contracts import Command
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True, slots=True)
class DescribeSettings:
    max_objects: int = 5
    wait_measure_s: float = 3.0
    large_area_fraction: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.max_objects, bool) or not isinstance(self.max_objects, int):
            raise ValueError("describe.max_objects debe ser un entero")
        if self.max_objects < 1:
            raise ValueError("describe.max_objects debe ser positivo")
        if not math.isfinite(self.wait_measure_s) or self.wait_measure_s < 0:
            raise ValueError("describe.wait_measure_s debe ser finito y no negativo")
        if not 0 < self.large_area_fraction <= 1:
            raise ValueError("describe.large_area_fraction debe estar en (0, 1]")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DescribeSettings:
        section = dict(cfg.get("describe") or {})
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"describe: claves desconocidas {sorted(unknown)}")
        return cls(**section)


@dataclass(frozen=True, slots=True)
class PresenceFact:
    detection_index: int
    label: str
    zone: str
    area_fraction: float


@dataclass(frozen=True, slots=True)
class MetricFact:
    detection_index: int
    z_m: float  # solo SPEAK_METERS dentro de depth.range_m


@dataclass(frozen=True, slots=True)
class SceneFacts:
    session_id: str
    frame_id: int
    presence: tuple[PresenceFact, ...]
    measured: tuple[MetricFact, ...]
    depth_model_id: str | None


class SceneNarrator(Protocol):
    def plan(self, ordered: tuple[PresenceFact, ...]) -> tuple[tuple[int, ...], ...]:
        """Agrupar índices de presencia ya priorizados. No recibe ni devuelve metros.

        El compositor valida que el plan conserve exactamente esos objetos y que cada grupo
        tenga una sola clase y zona. No se permite texto arbitrario de un proveedor futuro.
        """
        ...


class TemplateNarrator:
    def plan(self, ordered: tuple[PresenceFact, ...]) -> tuple[tuple[int, ...], ...]:
        groups: dict[tuple[str, str], list[int]] = {}
        for fact in ordered:
            groups.setdefault((fact.zone, fact.label), []).append(fact.detection_index)
        return tuple(tuple(indices) for indices in groups.values())


# Género de los nombres hablados del catálogo COCO. Los plurales compuestos con preposición
# no admiten flexionar cada palabra ("cepillos de dientes", no "cepillos des dientes").
FEMININE = frozenset(
    {
        "person",
        "bicycle",
        "motorcycle",
        "cow",
        "sheep",
        "zebra",
        "giraffe",
        "backpack",
        "handbag",
        "tie",
        "suitcase",
        "sports ball",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "spoon",
        "banana",
        "apple",
        "orange",
        "carrot",
        "pizza",
        "donut",
        "cake",
        "chair",
        "potted plant",
        "bed",
        "dining table",
        "laptop",
        "toaster",
        "sink",
        "refrigerator",
        "scissors",
    }
)
COMPOUND_PLURALS = {
    "fire hydrant": "hidrantes",
    "stop sign": "carteles de pare",
    "skis": "esquís",
    "baseball glove": "guantes de béisbol",
    "surfboard": "tablas de surf",
    "hair drier": "secadores de pelo",
    "toothbrush": "cepillos de dientes",
    "broccoli": "brócolis",
    "snowboard": "snowboards",
}


class DescribeHandler:
    def __init__(
        self,
        geometry: StereoGeometry,
        spoken: Callable[[str], str],
        settings: ModeSettings | None = None,
        describe: DescribeSettings | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        narrator: SceneNarrator | None = None,
    ) -> None:
        self.geometry = geometry
        self.spoken = spoken
        self.s = settings or ModeSettings()
        self.describe = describe or DescribeSettings()
        self.narrator = narrator or TemplateNarrator()
        self._on_event = on_event or (lambda kind, data: None)
        self.request_id: int | None = None
        self.facts: SceneFacts | None = None
        self._answered = False
        self._zones: list[tuple[Target, str]] = []

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self.request_id = command.request_id
        self.facts = None
        self._answered = False
        self._zones = []
        says, _ = self.tick(command, view, now_ns)
        return says

    def _usable(self, joint: Joint | None, command: Command, now_ns: int, age: float) -> bool:
        return (
            joint is not None
            and command.issued_ns <= joint.frame.received_ns <= now_ns
            and age_s(joint, now_ns) <= age
        )

    def tick(self, command: Command, view: View, now_ns: int) -> tuple[list[Say], bool]:
        if command.request_id != self.request_id or self._answered:
            return [], True
        if command.deadline_ns is not None and now_ns >= command.deadline_ns:
            self._answered = True
            return [], True

        # Mantener zonas mientras esperamos profundidad. Nunca copiar sus objetos ubicados:
        # la máscara y la Z que se describen deben seguir perteneciendo a la unión elegida.
        if self._usable(view.seen, command, now_ns, self.s.max_age_s):
            self._presence(view.seen, update=True)  # type: ignore[arg-type]
        joint = view.measured
        if (
            self._usable(joint, command, now_ns, self.s.max_measure_age_s)
            and joint is not None
            and joint.depth is not None
        ):
            return self._finish(joint, measured=True)

        limit = command.issued_ns + int(self.describe.wait_measure_s * 1e9)
        if command.deadline_ns is not None:
            limit = min(limit, command.deadline_ns - int(self.s.end_margin_s * 1e9))
        if now_ns < limit:
            return [], False
        if self._usable(view.seen, command, now_ns, self.s.max_age_s):
            return self._finish(view.seen, measured=False)  # type: ignore[arg-type]
        self._answered = True
        self._event({"motivo": "sin-vista-reciente"})
        return [Say("No pude obtener una imagen reciente para describir.")], True

    def _presence(self, joint: Joint, *, update: bool = False) -> tuple[PresenceFact, ...]:
        cands = candidates(joint, None, self.geometry)
        previous = list(self._zones)
        zones = []
        facts = []
        for c in cands:
            # Asociación uno a uno, con los mismos límites que buscar. No heredar la zona
            # de un frame posterior: la unión medida puede llegar atrasada respecto de seen.
            match = next(
                (
                    p
                    for p in previous
                    if p[0].last_seen_ns <= c.received_ns
                    and (c.received_ns - p[0].last_seen_ns) / 1e9 <= self.s.max_age_s
                    and p[0].match([c], self.s) is not None
                ),
                None,
            )
            prev = match[1] if match else None
            if match:
                previous.remove(match)
            raw_zone = horizontal(c.angle_h_deg, self.s, prev)
            zone = raw_zone.replace("un poco ", "")  # tres zonas para una descripción breve
            zones.append((Target(c.label, c.angle_h_deg, c.track_id, c.received_ns), raw_zone))
            x1, y1, x2, y2 = joint.detection.boxes[c.index]
            area = float((x2 - x1) * (y2 - y1)) / (self.geometry.width * self.geometry.height)
            facts.append(PresenceFact(c.index, c.label, zone, area))
        if update:
            self._zones = zones
        return tuple(facts)

    def _finish(self, joint: Joint, *, measured: bool) -> tuple[list[Say], bool]:
        presence = self._presence(joint)
        metrics = []
        if measured:
            for c in candidates(joint, None, self.geometry):
                m = measure(c.obj, self.s)
                if m.ok and m.z_m is not None:
                    metrics.append(MetricFact(c.index, m.z_m))
        self.facts = SceneFacts(
            joint.frame.session_id,
            joint.frame.frame_id,
            presence,
            tuple(metrics),
            joint.depth.model_id if measured and joint.depth else None,
        )
        phrases = self._render(self.facts)
        if not measured:
            phrases.insert(0, "No pude medir las distancias a tiempo.")
        self._answered = True
        self._event(
            {"hechos": asdict(self.facts), "solo_presencia": not measured, "texto": phrases}
        )
        return [Say(p) for p in phrases], True

    def _render(self, scene: SceneFacts) -> list[str]:
        if not scene.presence:
            return ["No reconozco objetos al frente."]
        distances = {m.detection_index: m.z_m for m in scene.measured}

        def relevance(f: PresenceFact) -> tuple:
            z = distances.get(f.detection_index, math.inf)
            # Cercanía primero (a la resolución hablada); a igual distancia, o sin medida,
            # personas/cajas grandes antes que objetos chicos. El área no es tamaño físico.
            bucket = round(z / self.s.round_m) if math.isfinite(z) else math.inf
            important = f.label == "person" or f.area_fraction >= self.describe.large_area_fraction
            return bucket, not important, -f.area_fraction, f.detection_index

        selected = tuple(sorted(scene.presence, key=relevance)[: self.describe.max_objects])
        by_index = {f.detection_index: f for f in selected}
        plan = self.narrator.plan(selected)
        flat = [i for group in plan for i in group]
        if len(flat) != len(selected) or set(flat) != set(by_index):
            raise ValueError("SceneNarrator debe conservar exactamente los índices seleccionados")
        parts = [
            f"Veo {len(scene.presence)} " + ("cosa." if len(scene.presence) == 1 else "cosas.")
        ]
        for group in plan:
            if not group:
                raise ValueError("SceneNarrator devolvió un grupo vacío")
            first = by_index[group[0]]
            if any(
                (by_index[i].label, by_index[i].zone) != (first.label, first.zone) for i in group
            ):
                raise ValueError("SceneNarrator mezcló clases o zonas")
            name = self.spoken(first.label)
            if first.label == "skis" and name == "esquís":
                name = "esquí"  # el nombre COCO del catálogo viene en plural
            feminine = first.label in FEMININE
            if len(group) == 1:
                noun = ("una " if feminine else "un ") + name
            else:
                noun = f"{len(group)} {COMPOUND_PLURALS.get(first.label, plural(name))}"
            phrase = f"{first.zone.capitalize()}, {noun}"
            # "la más cercana" solo si TODAS las instancias del grupo tienen metros válidos.
            # Con medidas parciales no podemos afirmar cuál de ellas está más cerca.
            if all(i in distances for i in group):
                m = Measure(min(distances[i] for i in group), None)
                if len(group) > 1:
                    phrase += ", la más cercana" if feminine else ", el más cercano"
                phrase += " " + measure_phrase(m, self.s)
            parts.append(phrase + ".")
        omitted = len(scene.presence) - len(selected)
        if omitted:
            parts.append(f"Y {omitted} " + ("cosa más." if omitted == 1 else "cosas más."))
        return parts

    def _event(self, data: dict[str, Any]) -> None:
        self._on_event("describir-respuesta", {"request_id": self.request_id, **data})
