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
    speech_s,
)
from visionx.modes.target import Target, plural
from visionx.shared.contracts import Command, FrameKey
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True, slots=True)
class DescribeSettings:
    max_objects: int = 5
    wait_measure_s: float = 3.0
    large_area_fraction: float = 0.1
    min_score: float = 0.6
    persist_hits: int = 2
    persist_s: float = 3.0
    max_speech_s: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.max_objects, bool) or not isinstance(self.max_objects, int):
            raise ValueError("describe.max_objects debe ser un entero")
        if self.max_objects < 1:
            raise ValueError("describe.max_objects debe ser positivo")
        if not math.isfinite(self.wait_measure_s) or self.wait_measure_s < 0:
            raise ValueError("describe.wait_measure_s debe ser finito y no negativo")
        if not 0 < self.large_area_fraction <= 1:
            raise ValueError("describe.large_area_fraction debe estar en (0, 1]")
        if not 0 <= self.min_score <= 1:
            raise ValueError("describe.min_score debe estar en [0, 1]")
        if isinstance(self.persist_hits, bool) or not isinstance(self.persist_hits, int):
            raise ValueError("describe.persist_hits debe ser un entero")
        if not 1 <= self.persist_hits <= 256:
            raise ValueError("describe.persist_hits debe estar entre 1 y 256")
        for name in ("persist_s", "max_speech_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"describe.{name} debe ser finito y positivo")

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
        "tv",  # en DESCRIBIR decimos "pantalla": COCO también etiqueta así los monitores
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


@dataclass(frozen=True, slots=True)
class _SeenInstance:
    label: str
    track_id: int | None
    box: tuple[float, ...]


def _overlap(a: _SeenInstance, b: _SeenInstance) -> float:
    ax, ay, ar, ab = a.box
    bx, by, br, bb = b.box
    intersection = max(0.0, min(ar, br) - max(ax, bx)) * max(0.0, min(ab, bb) - max(ay, by))
    union = (ar - ax) * (ab - ay) + (br - bx) * (bb - by) - intersection
    return intersection / union if union else 0.0


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
        # Solo cajas/clases/IDs: no retener imágenes ni profundidad. Se reinicia por orden.
        self._history: dict[FrameKey, tuple[int, dict[int, _SeenInstance]]] = {}

    def start(self, command: Command, view: View, now_ns: int) -> list[Say]:
        self.request_id = command.request_id
        self.facts = None
        self._answered = False
        self._zones = []
        self._history = {}
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

        # El frame reciente anterior a la orden puede confirmar presencia, nunca dar metros.
        # seen y measured del mismo frame cuentan UNA vez; tampoco cuentan ticks repetidos.
        for observation in (view.seen, view.measured):
            if observation is not None:
                self._observe(observation, now_ns)

        limit = command.issued_ns + int(self.describe.wait_measure_s * 1e9)
        if command.deadline_ns is not None:
            limit = min(limit, command.deadline_ns - int(self.s.end_margin_s * 1e9))

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
            # Si todavía falta confirmación, aprovechar la espera ya acotada por el pedido.
            if self._confirmed(joint) or not joint.detection.classes or now_ns >= limit:
                return self._finish(joint, measured=True)

        if now_ns < limit:
            return [], False
        if self._usable(view.seen, command, now_ns, self.s.max_age_s):
            return self._finish(view.seen, measured=False)  # type: ignore[arg-type]
        self._answered = True
        self._event({"motivo": "sin-vista-reciente"})
        return [Say("No pude obtener una imagen reciente para describir.")], True

    def _instances(self, joint: Joint) -> dict[int, _SeenInstance]:
        return {
            c.index: _SeenInstance(
                c.label, c.track_id, tuple(float(v) for v in joint.detection.boxes[c.index])
            )
            for c in candidates(joint, None, self.geometry)
            if c.score >= self.describe.min_score
        }

    def _observe(self, joint: Joint, now_ns: int) -> None:
        retention = self.describe.persist_s + self.s.max_measure_age_s
        cutoff = now_ns - int(retention * 1e9)
        self._history = {key: item for key, item in self._history.items() if item[0] >= cutoff}
        if cutoff <= joint.frame.received_ns <= now_ns:
            self._history.setdefault(
                joint.frame.key, (joint.frame.received_ns, self._instances(joint))
            )
        # Además del límite temporal, un tope duro evita crecer con FPS muy altos.
        while len(self._history) > 256:
            oldest = min(self._history, key=lambda key: self._history[key][0])
            del self._history[oldest]

    def _confirmed(self, joint: Joint) -> set[int]:
        current = self._instances(joint)
        hits = dict.fromkeys(current, 0)
        end = joint.frame.received_ns
        start = end - int(self.describe.persist_s * 1e9)
        for key, (timestamp, previous) in self._history.items():
            if key.session_id != joint.frame.session_id or not start <= timestamp <= end:
                continue
            # Asociación uno a uno por frame: una silla anterior no confirma dos actuales.
            # ID estable o IoU >= 0,3 sin ID: evita sumar cualquier instancia de la clase.
            pairs = []
            for i, instance in current.items():
                for j, old in previous.items():
                    if instance.label != old.label:
                        continue
                    same_id = instance.track_id is not None and instance.track_id == old.track_id
                    different_ids = (
                        instance.track_id is not None
                        and old.track_id is not None
                        and instance.track_id != old.track_id
                    )
                    overlap = _overlap(instance, old)
                    if same_id or (not different_ids and overlap >= 0.3):
                        pairs.append((same_id, overlap, i, j))
            used_current, used_previous = set(), set()
            for _, _, i, j in sorted(pairs, reverse=True):
                if i not in used_current and j not in used_previous:
                    hits[i] += 1
                    used_current.add(i)
                    used_previous.add(j)
        return {i for i, n in hits.items() if n >= self.describe.persist_hits}

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
        confirmed = self._confirmed(joint)
        presence = tuple(p for p in self._presence(joint) if p.detection_index in confirmed)
        metrics = []
        if measured:
            for c in candidates(joint, None, self.geometry):
                m = measure(c.obj, self.s)
                if c.index in confirmed and m.ok and m.z_m is not None:
                    metrics.append(MetricFact(c.index, m.z_m))
        self.facts = SceneFacts(
            joint.frame.session_id,
            joint.frame.frame_id,
            presence,
            tuple(metrics),
            joint.depth.model_id if measured and joint.depth else None,
        )
        phrases = self._render(self.facts, measured=measured)
        self._answered = True
        self._event(
            {
                "hechos": asdict(self.facts),
                "solo_presencia": not measured,
                "texto": phrases,
                "caracteres": len(" ".join(phrases)),
                "duracion_estimada_s": sum(speech_s(p, self.s) for p in phrases),
            }
        )
        return [Say(p) for p in phrases], True

    def _render(self, scene: SceneFacts, *, measured: bool = True) -> list[str]:
        if not scene.presence:
            return ["No reconozco objetos con suficiente confianza."]
        distances = {m.detection_index: m.z_m for m in scene.measured}

        def relevance(f: PresenceFact) -> tuple:
            z = distances.get(f.detection_index, math.inf)
            # Cercanía primero (a la resolución hablada); a igual distancia, o sin medida,
            # personas/cajas grandes antes que objetos chicos. El área no es tamaño físico.
            bucket = round(z / self.s.round_m) if math.isfinite(z) else math.inf
            important = f.label == "person" or f.area_fraction >= self.describe.large_area_fraction
            return bucket, not important, -f.area_fraction, f.detection_index

        selected = tuple(sorted(scene.presence, key=relevance)[: self.describe.max_objects])
        while True:
            parts = self._compose(selected, distances, len(scene.presence))
            if not measured:
                parts.insert(0, "No pude medir distancias.")
            if sum(speech_s(p, self.s) for p in parts) <= self.describe.max_speech_s:
                return parts
            if len(selected) == 1:
                # Un presupuesto local muy pequeño puede no admitir ni un nombre completo.
                return ["No pude resumir la escena a tiempo."]
            selected = selected[:-1]  # retirar primero la instancia de menor relevancia

    def _compose(
        self, selected: tuple[PresenceFact, ...], distances: dict[int, float], total: int
    ) -> list[str]:
        by_index = {f.detection_index: f for f in selected}
        plan = self.narrator.plan(selected)
        flat = [i for group in plan for i in group]
        if len(flat) != len(selected) or set(flat) != set(by_index):
            raise ValueError("SceneNarrator debe conservar exactamente los índices seleccionados")
        zones: dict[str, list[str]] = {}
        for group in plan:
            if not group:
                raise ValueError("SceneNarrator devolvió un grupo vacío")
            first = by_index[group[0]]
            if any(
                (by_index[i].label, by_index[i].zone) != (first.label, first.zone) for i in group
            ):
                raise ValueError("SceneNarrator mezcló clases o zonas")
            # El score alto no distingue monitor de TV: usar el término visual compartido.
            # Buscar conserva el nombre y el umbral de su catálogo, sin cambios en T06.
            name = "pantalla" if first.label == "tv" else self.spoken(first.label)
            if first.label == "skis" and name == "esquís":
                name = "esquí"  # el nombre COCO del catálogo viene en plural
            feminine = first.label in FEMININE
            if len(group) == 1:
                noun = ("una " if feminine else "un ") + name
            else:
                count = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}.get(
                    len(group), str(len(group))
                )
                noun = f"{count} {COMPOUND_PLURALS.get(first.label, plural(name))}"
            phrase = noun
            # "la más cercana" solo si TODAS las instancias del grupo tienen metros válidos.
            # Con medidas parciales no podemos afirmar cuál de ellas está más cerca.
            if all(i in distances for i in group):
                m = Measure(min(distances[i] for i in group), None)
                if len(group) > 1:
                    phrase += ", la más cercana" if feminine else ", el más cercano"
                phrase += " " + measure_phrase(m, self.s)
            zones.setdefault(first.zone, []).append(phrase)
        parts = []
        for zone in ("al frente", "a la izquierda", "a la derecha"):
            elements = zones.pop(zone, [])
            if elements:
                listing = (
                    elements[0]
                    if len(elements) == 1
                    else ", ".join(elements[:-1]) + " y " + elements[-1]
                )
                parts.append(f"{zone.capitalize()}, {listing}.")
        if zones:
            raise ValueError("SceneNarrator devolvió una zona desconocida")
        omitted = total - len(selected)
        if omitted:
            parts[-1] = (
                parts[-1][:-1] + f", y {omitted} " + ("cosa más." if omitted == 1 else "cosas más.")
            )
        return parts

    def _event(self, data: dict[str, Any]) -> None:
        self._on_event("describir-respuesta", {"request_id": self.request_id, **data})
