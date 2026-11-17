"""El objetivo que se está guiando y cómo se lo sigue entre frames (T09).

Por qué no alcanza el ID de ByteTrack: con cámara en mano el ID no sobrevive a una salida de
cuadro y muchas detecciones llegan sin ID (BITACORA 2026-11-14, clips `buscar`). El objetivo se
sigue con, en orden:

1. el mismo track_id, si ambos lo tienen;
2. si no, continuidad de posición horizontal/vertical y Z fiable, cuando existe.

Con varios candidatos, la ventana se acota y se exige una ventaja sobre el segundo: una
pérdida larga no hace equivalentes todos los objetos de la clase. La referencia se conserva
sin guiar con ella; solo una observación fresca permite retomar la guía.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, log

from visionx.modes.common import Candidate, ModeSettings, measure
from visionx.shared.contracts import DistancePolicy


@dataclass(slots=True)
class Target:
    label: str
    angle_h_deg: float
    track_id: int | None
    last_seen_ns: int
    angle_v_deg: float = 0.0
    z_m: float | None = None

    def update(self, c: Candidate) -> None:
        self.angle_h_deg = c.angle_h_deg
        self.angle_v_deg = c.angle_v_deg
        self.track_id = c.track_id if c.track_id is not None else self.track_id
        self.last_seen_ns = c.received_ns
        self.remember_measure(c)

    def remember_measure(self, c: Candidate) -> None:
        # Solo es una pista de asociación; nunca se usa como respuesta ni como Z actual.
        obj = c.obj
        if (
            obj is not None
            and obj.policy is DistancePolicy.SPEAK_METERS
            and obj.z_m is not None
            and obj.z_m > 0
        ):
            self.z_m = obj.z_m

    def match(self, cands: list[Candidate], s: ModeSettings) -> Candidate | None:
        """El candidato que sigue siendo este objetivo, o None."""
        same = [c for c in cands if c.label == self.label]
        if self.track_id is not None:
            for c in same:
                if c.track_id == self.track_id:
                    return c
        if not same:
            return None
        # La ventana crece con el tiempo entre la última vez que se lo vio y este frame. En
        # valor absoluto: una unión con profundidad puede ser de un frame ANTERIOR al último visto.
        newest = max(c.received_ns for c in same)
        dt_s = abs(newest - self.last_seen_ns) / 1e9
        gate = s.gate_deg + s.gate_deg_per_s * dt_s
        if len(same) > 1:
            gate = min(gate, s.reassociate_gate_deg)

        def position(c: Candidate) -> float:
            return hypot(c.angle_h_deg - self.angle_h_deg, c.angle_v_deg - self.angle_v_deg)

        def cost(c: Candidate) -> float:
            m = measure(c.obj, s)
            depth = 0.0
            if self.z_m and m.ok and m.z_m and self.z_m > 0 and m.z_m > 0:
                # Cambio relativo de Z: duplicar/dividir por dos tiene el mismo coste.
                depth = s.gate_deg * abs(log(m.z_m / self.z_m))
            return position(c) + depth

        ranked = sorted(same, key=cost)
        best = ranked[0]
        if position(best) > gate:
            return None
        if len(ranked) > 1 and cost(ranked[1]) - cost(best) < s.match_margin_deg:
            return None
        return best


class TargetMemory:
    """Último objetivo buscado/medido, compartido entre buscar y distancia.

    Conserva la intención sin afirmar presencia. Una selección observada aporta además
    pistas para reasociar la instancia; una búsqueda nueva reemplaza ambas referencias.
    """

    def __init__(self) -> None:
        self.target: Target | None = None
        self.label: str | None = None
        self.mentioned_ns: int = 0

    def request(self, label: str, now_ns: int) -> None:
        """Recordar la intención, incluso si el objeto todavía no fue detectado."""
        if self.label != label:
            self.target = None
        self.label, self.mentioned_ns = label, now_ns

    def remember(self, target: Target | None) -> None:
        self.target = target
        self.label = target.label if target else None
        self.mentioned_ns = target.last_seen_ns if target else 0

    def recent_label(self, now_ns: int, within_s: float) -> str | None:
        return self.label if (now_ns - self.mentioned_ns) / 1e9 <= within_s else None

    def recent(self, now_ns: int, within_s: float) -> Target | None:
        t = self.target
        if t is None or self.recent_label(now_ns, within_s) is None:
            return None
        return t


def plural(name: str) -> str:
    """Plural simple del nombre hablado: taza -> tazas, televisor -> televisores, lápiz ->
    lápices. Cada palabra por separado ("control remoto" -> "controles remotos")."""
    out = []
    for word in name.split():
        if word[-1] in "aeiouáéó":
            out.append(word + "s")
        elif word.endswith("z"):
            out.append(word[:-1] + "ces")
        elif word[-1] == "s":
            out.append(word)
        elif word.endswith("ón"):  # ratón -> ratones: el plural pierde la tilde
            out.append(word[:-2] + "ones")
        else:
            out.append(word + "es")
    return " ".join(out)
