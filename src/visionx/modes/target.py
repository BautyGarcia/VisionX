"""El objetivo que se está guiando y cómo se lo sigue entre frames (T09).

Por qué no alcanza el ID de ByteTrack: con cámara en mano el ID no sobrevive a una salida de
cuadro y muchas detecciones llegan sin ID (BITACORA 2026-11-14, clips `buscar`). El objetivo se
sigue con, en orden:

1. el mismo track_id, si ambos lo tienen;
2. si no, el candidato de la misma clase más cercano en ángulo, dentro de una ventana que crece
   con el tiempo transcurrido (la cámara puede haber girado).

Si nadie cae en la ventana, el objetivo no se vio en ese frame. Tras `lost_s` sin verlo se da
por perdido y se olvida: TECNICO T06 pide no asignar su identidad al primero parecido. Si
reaparece, es una adquisición nueva (con su bip), no "el mismo".
"""

from __future__ import annotations

from dataclasses import dataclass

from visionx.modes.common import Candidate, ModeSettings


@dataclass(slots=True)
class Target:
    label: str
    angle_h_deg: float
    track_id: int | None
    last_seen_ns: int
    measured_z_m: float | None = None  # último Z decible, para "llegaste"

    def update(self, c: Candidate) -> None:
        self.angle_h_deg = c.angle_h_deg
        self.track_id = c.track_id if c.track_id is not None else self.track_id
        self.last_seen_ns = c.received_ns

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
        best = min(same, key=lambda c: abs(c.angle_h_deg - self.angle_h_deg))
        return best if abs(best.angle_h_deg - self.angle_h_deg) <= gate else None


class TargetMemory:
    """Último objetivo guiado, compartido entre buscar y distancia.

    "¿A qué distancia está?" sin nombrar el objeto mide lo que se estaba guiando, si se guió
    hace poco. La búsqueda lo guarda; la distancia lo lee. Una orden nueva de búsqueda lo pisa.
    """

    def __init__(self) -> None:
        self.target: Target | None = None

    def remember(self, target: Target | None) -> None:
        self.target = target

    def recent(self, now_ns: int, within_s: float) -> Target | None:
        t = self.target
        if t is None or (now_ns - t.last_seen_ns) / 1e9 > within_s:
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
