"""DESCRIBIR beta: una escena congelada, sin VLM ni generación de metros.

`SceneFacts` separa presencia de mediciones aceptadas del MISMO frame. El punto de extensión
`SceneNarrator` solo propone grupos de índices existentes, nunca texto libre ni distancias.
Un futuro VLM podrá proponer contexto no métrico con un contrato acotado y validado aparte;
no debe reemplazar el compositor de números/unidades que vive en este módulo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Protocol

from visionx.app.controller import Say
from visionx.app.runtime import Joint, View
from visionx.modes.common import (
    Measure,
    ModeSettings,
    age_s,
    candidates,
    closing_text,
    horizontal,
    measure,
    measure_phrase,
    precision_view,
    request_precision,
    speech_s,
)
from visionx.modes.target import Target, plural
from visionx.shared.contracts import Command, FrameKey
from visionx.shared.geometry import StereoGeometry


@dataclass(frozen=True, slots=True)
class DescribeSettings:
    max_objects: int = 8
    wait_measure_s: float = 4.0
    large_area_fraction: float = 0.1
    min_score: float = 0.6
    cautious_score: float = 0.27
    # Cortes de admisión por clase: también se aplican al nivel alto. No probabilidades.
    cautious_min_scores: dict[str, float] = field(
        default_factory=lambda: {
            "bed": 0.35,
            "bench": 0.35,
            "banana": 0.4,
            "potted plant": 0.4,
            "laptop": 0.5,
            "dining table": 0.5,
            "oven": 0.65,
        }
    )
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
        if not 0 <= self.cautious_score <= self.min_score:
            raise ValueError("describe.cautious_score debe estar entre 0 y min_score")
        if not isinstance(self.cautious_min_scores, dict) or any(
            not isinstance(label, str) or not label or not 0 <= score <= 1
            for label, score in self.cautious_min_scores.items()
        ):
            raise ValueError("describe.cautious_min_scores debe mapear clases a scores en [0, 1]")
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
    confidence: Literal["high", "cautious"] = "high"


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
        tenga una sola clase, zona y confianza. No se permite texto arbitrario de un
        proveedor futuro.
        """
        ...


class TemplateNarrator:
    def plan(self, ordered: tuple[PresenceFact, ...]) -> tuple[tuple[int, ...], ...]:
        groups: dict[tuple[str, str, str], list[int]] = {}
        for fact in ordered:
            groups.setdefault((fact.zone, fact.label, fact.confidence), []).append(
                fact.detection_index
            )
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
    score: float


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
        request_precision(view, command, now_ns, self.s)
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

        selected = precision_view(view, command, now_ns, self.s, self.geometry)
        if selected is not None:
            view = selected
        if view.measured is not None:
            self._observe(view.measured, now_ns)

        limit = command.issued_ns + int(self.describe.wait_measure_s * 1e9)
        if command.deadline_ns is not None:
            fallback = closing_text(
                command,
                command.issued_ns,
                [
                    "No pude obtener una imagen reciente para describir.",
                    "No pude describir a tiempo.",
                    "Sin imagen.",
                ],
                self.s,
            )
            # Reservar el aviso; _finish comprueba además la suma real de la escena.
            limit = min(
                limit,
                command.deadline_ns - int((speech_s(fallback, self.s) + self.s.end_margin_s) * 1e9),
            )
        if selected is None and now_ns < limit:
            return [], False

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
                return self._finish(joint, measured=True, command=command, now_ns=now_ns)

        if now_ns < limit:
            return [], False
        if self._usable(view.seen, command, now_ns, self.s.max_age_s):
            return self._finish(  # type: ignore[arg-type]
                view.seen, measured=False, command=command, now_ns=now_ns
            )
        self._answered = True
        self._event({"motivo": "sin-vista-reciente"})
        closing = closing_text(
            command,
            now_ns,
            [
                "No pude obtener una imagen reciente para describir.",
                "No pude describir a tiempo.",
                "Sin imagen.",
            ],
            self.s,
            reserve=False,
        )
        return ([Say(closing)] if closing else []), True

    def _instances(self, joint: Joint) -> dict[int, _SeenInstance]:
        return {
            c.index: _SeenInstance(
                c.label,
                c.track_id,
                tuple(float(v) for v in joint.detection.boxes[c.index]),
                c.score,
            )
            for c in candidates(joint, None, self.geometry)
            if c.score >= self._threshold(c.label)
        }

    def _threshold(self, label: str, *, high: bool = False) -> float:
        return max(
            self.describe.min_score if high else self.describe.cautious_score,
            self.describe.cautious_min_scores.get(label, 0.0),
        )

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

    def _confirmed(self, joint: Joint, *, high: bool = False) -> set[int]:
        current = {
            i: instance
            for i, instance in self._instances(joint).items()
            if instance.score >= self._threshold(instance.label, high=high)
        }
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
                    if old.score < self._threshold(old.label, high=high):
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

    def _finish(
        self, joint: Joint, *, measured: bool, command: Command, now_ns: int
    ) -> tuple[list[Say], bool]:
        confirmed = self._confirmed(joint)
        # La asociación con todos los candidatos sigue mandando sobre el conjunto hablado.
        # Filtrar candidatos puede cambiar un empate IoU: no dejar métricas huérfanas.
        high = self._confirmed(joint, high=True) & confirmed
        presence = tuple(
            PresenceFact(
                p.detection_index,
                p.label,
                p.zone,
                p.area_fraction,
                "high" if p.detection_index in high else "cautious",
            )
            for p in self._presence(joint)
            if p.detection_index in confirmed
        )
        metrics = []
        if measured:
            for c in candidates(joint, None, self.geometry):
                m = measure(c.obj, self.s)
                if c.index in high and m.ok and m.z_m is not None:
                    metrics.append(MetricFact(c.index, m.z_m))
        self.facts = SceneFacts(
            joint.frame.session_id,
            joint.frame.frame_id,
            presence,
            tuple(metrics),
            joint.depth.model_id if measured and joint.depth else None,
        )
        phrases = self._render(self.facts, measured=measured)
        if (
            command.deadline_ns is not None
            and sum(speech_s(p, self.s) for p in phrases) + (self.s.end_margin_s)
            > (command.deadline_ns - now_ns) / 1e9
        ):
            brief = closing_text(
                command,
                now_ns,
                [
                    "No pude describir a tiempo.",
                    "Sin descripción.",
                ],
                self.s,
                reserve=False,
            )
            phrases = [brief] if brief else []
        self._answered = True
        self._event(
            {
                "hechos": asdict(self.facts),
                "latencia_ms": joint.depth.inference_ms if joint.depth else None,
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
        high = {p.detection_index for p in scene.presence if p.confidence == "high"}
        distances = {m.detection_index: m.z_m for m in scene.measured if m.detection_index in high}

        def relevance(f: PresenceFact) -> tuple:
            z = distances.get(f.detection_index, math.inf)
            # Cercanía primero (a la resolución hablada); a igual distancia, o sin medida,
            # personas/cajas grandes antes que objetos chicos. El área no es tamaño físico.
            bucket = round(z / self.s.round_m) if math.isfinite(z) else math.inf
            important = f.label == "person" or f.area_fraction >= self.describe.large_area_fraction
            return bucket, not important, -f.area_fraction, f.detection_index

        # Proteger la menor Z sin redondear: dos distancias pueden sonar iguales y tener
        # áreas distintas. El desempate de relevancia nunca debe desplazar la más cercana.
        nearest_by_zone: dict[str, int] = {}
        for fact in sorted(
            scene.presence, key=lambda f: (distances.get(f.detection_index, math.inf), relevance(f))
        ):
            if fact.detection_index in distances:
                nearest_by_zone.setdefault(fact.zone, fact.detection_index)
        nearest = min(distances, key=lambda i: (distances[i], i), default=None)
        protected = set(nearest_by_zone.values())
        if nearest is not None:
            protected.add(nearest)

        def priority(f: PresenceFact) -> tuple:
            return (
                f.detection_index != nearest,
                f.detection_index not in protected,
                f.confidence != "high",
                relevance(f),
            )

        selected = tuple(sorted(scene.presence, key=priority)[: self.describe.max_objects])
        detail = dict(distances)
        while True:
            parts = self._compose(selected, detail, len(scene.presence))
            if not measured:
                parts.insert(0, "No pude medir distancias.")
            if sum(speech_s(p, self.s) for p in parts) <= self.describe.max_speech_s:
                return parts

            # Recortar de menor a mayor prioridad. Una medida agrupada se quita entera:
            # retirar solo una Z podría alargar el texto al separar instancias de la clase.
            cautious = [f for f in selected if f.confidence == "cautious"]
            if cautious and len(selected) > 1:
                selected = tuple(f for f in selected if f != cautious[-1])
                continue
            metric_groups: dict[tuple[str, str], set[int]] = {}
            for fact in selected:
                if fact.detection_index in detail:
                    metric_groups.setdefault((fact.zone, fact.label), set()).add(
                        fact.detection_index
                    )
            secondary = [ids for ids in metric_groups.values() if not ids & protected]
            if secondary:
                for index in secondary[-1]:
                    detail.pop(index)
                continue

            if len(selected) == 1:
                # Si un override no admite ni el hecho obligatorio con su conteo, prevalece
                # conservar sus metros. Con la config entregada este mínimo cabe en 8 s.
                if nearest is not None:
                    return parts
                return ["No pude resumir la escena a tiempo."]
            selected = selected[:-1]  # nombres; representantes de zona al final, global nunca

    def _compose(
        self, selected: tuple[PresenceFact, ...], distances: dict[int, float], total: int
    ) -> list[str]:
        by_index = {f.detection_index: f for f in selected}
        plan = self.narrator.plan(selected)
        flat = [i for group in plan for i in group]
        if len(flat) != len(selected) or set(flat) != set(by_index):
            raise ValueError("SceneNarrator debe conservar exactamente los índices seleccionados")
        zones: dict[str, dict[str, list[str]]] = {}
        groups = []
        for group in plan:
            if not group:
                raise ValueError("SceneNarrator devolvió un grupo vacío")
            first = by_index[group[0]]
            if any(
                (by_index[i].label, by_index[i].zone, by_index[i].confidence)
                != (first.label, first.zone, first.confidence)
                for i in group
            ):
                raise ValueError("SceneNarrator mezcló clases, zonas o confianza")
            if first.confidence not in ("high", "cautious"):
                raise ValueError("SceneNarrator recibió confianza desconocida")
            # Una instancia sin medida no debe silenciar los metros de otra de su clase.
            # Separar ambos subgrupos evita llamar «la más cercana» a una medida parcial.
            with_meters = tuple(i for i in group if i in distances and first.confidence == "high")
            without_meters = tuple(i for i in group if i not in with_meters)
            groups.extend(g for g in (with_meters, without_meters) if g)
        for group in groups:
            first = by_index[group[0]]
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
            if first.confidence == "high" and all(i in distances for i in group):
                m = Measure(min(distances[i] for i in group), None)
                if len(group) > 1:
                    phrase += ", la más cercana" if feminine else ", el más cercano"
                phrase += " " + measure_phrase(m, self.s)
            zones.setdefault(first.zone, {}).setdefault(first.confidence, []).append(phrase)
        parts = []
        for zone in ("al frente", "a la izquierda", "a la derecha"):
            levels = zones.pop(zone, {})
            clauses = []
            for confidence in ("high", "cautious"):
                elements = levels.get(confidence, [])
                if elements:
                    listing = (
                        elements[0]
                        if len(elements) == 1
                        else ", ".join(elements[:-1]) + " y " + elements[-1]
                    )
                    clauses.append(("parece haber " if confidence == "cautious" else "") + listing)
            if clauses:
                parts.append(f"{zone.capitalize()}, {'; '.join(clauses)}.")
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
