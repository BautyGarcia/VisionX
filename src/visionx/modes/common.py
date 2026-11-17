"""Piezas comunes de los modos buscar y distancia (T09).

Vocabulario:
- **Vista** (`View`): lo último que observó la ejecución. `seen` es frame + detección, sin
  esperar profundidad (alcanza para avisar presencia y dirección). `measured` es frame +
  detección + profundidad + objetos ubicados del MISMO frame (la unión de T07): solo de ahí
  salen metros.
- **Candidato**: una detección de la clase pedida, con su ángulo horizontal.
- **Fresco**: una observación cuyo frame llegó hace menos de `max_age_s`. Una guía con un
  frame viejo describiría dónde ESTABA el objeto: no se usa.

Los números de `ModeSettings` son candidatos sin medir (M1/M7/M8 los fijan); viven en la
sección `modes` de la config para poder ajustarlos sin tocar código.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from visionx.app.runtime import Joint, View
from visionx.shared.contracts import Command, DistancePolicy, SpatialObject
from visionx.shared.geometry import StereoGeometry

__all__ = ["Candidate", "Measure", "ModeSettings", "View", "age_s", "candidates", "direction",
           "horizontal", "measure", "measure_phrase", "meters", "vertical"]  # fmt: skip


@dataclass(frozen=True, slots=True)
class ModeSettings:
    # Dirección (grados desde el eje óptico; positivo a la derecha).
    center_deg: float = 8.0  # |ángulo| <= esto: "al frente"
    side_deg: float = 25.0  # <= esto: "un poco a la izquierda/derecha"; más: "a la izquierda"
    # Vertical (positivo hacia arriba, respecto del eje de la cámara): se agrega "arriba" o
    # "abajo" solo si |ángulo| supera esto. Con la cámara al pecho, la mesa suele quedar abajo.
    vertical_deg: float = 15.0
    # Histéresis de los límites (grados): una vez dicha una zona, se mantiene hasta pasar el
    # límite por este margen. Sin ella, un ángulo cerca de 15° alterna "…, abajo" / "…" en
    # cada frase (se vio en la demo de la silla).
    hysteresis_deg: float = 3.0
    # Presencia: el objeto se anuncia si se vio en `confirm_hits` frames dentro de
    # `confirm_s` segundos. En tiempo, no en frames: 3 frames duran distinto a 3 y a 30 fps.
    confirm_hits: int = 2
    confirm_s: float = 1.0
    # Observaciones más viejas que esto no se usan para guiar ni medir.
    max_age_s: float = 1.0
    max_measure_age_s: float = 3.0  # la profundidad tarda más que la detección
    # Seguimiento del objetivo elegido entre frames (sin depender del ID de ByteTrack, que no
    # sobrevive a una salida de cuadro: BITACORA 2026-11-14).
    gate_deg: float = 15.0  # salto de ángulo aceptado entre dos frames seguidos...
    gate_deg_per_s: float = 60.0  # ...más esto por segundo transcurrido (la cámara gira)
    # Sin ver el objetivo durante esto: "lo perdí". En los clips buscar (cámara en mano), pasar
    # de 1,5 a 2,5 s bajó los avisos de pérdida que se recuperan en < 3 s y no bajó la cobertura
    # de guía (BITACORA T09 mejoras). Más largo demora en avisar una pérdida real.
    lost_s: float = 2.5
    # Ritmo de la voz durante la guía.
    min_interval_s: float = 1.2  # entre dos frases de guía
    repeat_s: float = 4.0  # repetir la dirección aunque no cambie
    # Duración estimada de una frase hablada: base + por carácter. Piper es_AR (daniela, rate
    # 1,0) mide 50–70 ms por carácter. La guía no habla encima de su frase anterior: si lo
    # hiciera, las frases se encolarían y la dirección llegaría tarde.
    speech_base_s: float = 0.3
    speech_s_per_char: float = 0.065
    end_margin_s: float = 1.0  # cerrar con una frase antes de que venza la orden
    # Metros.
    range_m: tuple[float, float] = (0.5, 3.0)  # fuera de esto no se dice un número
    round_m: float = 0.1  # redondeo de lo que se dice
    arrive_m: float = 0.6  # al frente y a esta distancia o menos: "llegaste"
    # Distancia: un frame medido vale si llegó a lo sumo esto ANTES de la orden.
    accept_before_s: float = 0.5
    # Elección entre iguales: a lo sumo cuántos se enumeran.
    max_listed: int = 3

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ModeSettings:
        """Sección `modes` de la config; el rango viene de `depth.range_m` si no se pisa."""
        section = dict(cfg.get("modes") or {})
        if "range_m" not in section and (cfg.get("depth") or {}).get("range_m"):
            section["range_m"] = cfg["depth"]["range_m"]
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"modes: claves desconocidas {sorted(unknown)}")
        if "range_m" in section:
            section["range_m"] = tuple(float(x) for x in section["range_m"])
        return cls(**section)


@dataclass(frozen=True, slots=True)
class Candidate:
    """Una detección de la clase pedida en un frame."""

    label: str
    index: int  # posición en la Detection
    angle_h_deg: float
    angle_v_deg: float
    track_id: int | None
    score: float
    received_ns: int  # cuándo llegó su frame
    obj: SpatialObject | None = None  # ubicación del mismo frame, si la hay


def speech_s(text: str, s: ModeSettings) -> float:
    """Cuánto tarda en decirse una frase (estimado; los sonidos son cortos)."""
    return 0.15 if text.startswith("sonido:") else s.speech_base_s + s.speech_s_per_char * len(text)


def age_s(joint: Joint, now_ns: int) -> float:
    return max(0.0, (now_ns - joint.frame.received_ns) / 1e9)


def request_precision(view: View, command: Command, now_ns: int, s: ModeSettings) -> None:
    """Congelar una detección para este request_id; desactivado por defecto."""
    if view.precise is not None:
        view.precise.request(
            command, view, now_ns, end_margin_s=s.end_margin_s, accept_before_s=s.accept_before_s
        )


def precision_view(
    view: View,
    command: Command,
    now_ns: int,
    s: ModeSettings,
    geometry: StereoGeometry,
    label: str | None = None,
    guided_angle: float | None = None,
) -> View | None:
    """Esperar precisa sin bloquear; usar continua si vence/falla o no mide el objetivo.

    Reutiliza la política pública de metros del modo. Para varios objetos nombrados todos
    deben poder medirse; describir admite una escena con al menos un objeto medible.
    """
    if view.precise is None:
        return view

    def invalid(joint: Joint) -> str | None:
        if age_s(joint, now_ns) > s.max_measure_age_s:
            return "frame-viejo"
        if joint.frame.received_ns < command.issued_ns - int(s.accept_before_s * 1e9):
            return "frame-anterior-a-orden"
        selected = candidates(joint, label, geometry)
        if command.mode.value == "distance":
            if guided_angle is not None and selected:
                selected = [min(selected, key=lambda c: abs(c.angle_h_deg - guided_angle))]
            elif label is None:
                selected = [c for c in selected if abs(c.angle_h_deg) <= s.center_deg * 2]
                selected = [min(selected, key=lambda c: abs(c.angle_h_deg))] if selected else []
            selected = selected[: s.max_listed]
            if not selected:
                return "sin-objetivo"
            return next((m.reason for c in selected if not (m := measure(c.obj, s)).ok), None)
        return None if any(measure(c.obj, s).ok for c in selected) else "sin-metros"

    chosen, pending = view.precise.select(command, view, now_ns, invalid)
    return None if pending else replace(view, measured=chosen)


# Dos cajas de la misma clase que se cubren tanto son el mismo objeto detectado dos veces (YOLO
# devolvió una parte de la silla como otra silla: clip acercarse-1, BITACORA T09).
DUPLICATE_OVERLAP = 0.7


def _duplicates(det, label: str | None) -> set[int]:
    """Índices a descartar: la de menor score de cada par de la misma clase cuya intersección
    cubre más de DUPLICATE_OVERLAP de la caja más chica."""
    drop: set[int] = set()
    idx = [i for i, c in enumerate(det.classes) if label is None or c == label]
    for a in idx:
        for b in idx:
            if a >= b or det.classes[a] != det.classes[b] or a in drop or b in drop:
                continue
            ax1, ay1, ax2, ay2 = (float(x) for x in det.boxes[a])
            bx1, by1, bx2, by2 = (float(x) for x in det.boxes[b])
            inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0.0, min(ay2, by2) - max(ay1, by1)
            )
            smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
            if smaller > 0 and inter / smaller > DUPLICATE_OVERLAP:
                drop.add(a if det.scores[a] < det.scores[b] else b)
    return drop


def candidates(joint: Joint, label: str | None, geometry: StereoGeometry) -> list[Candidate]:
    """Detecciones de `label` (todas si None), ordenadas de izquierda a derecha.

    El ángulo sale del centro de la caja con la geometría del frame. Si la unión trae objetos
    ubicados, se adjuntan (mismo frame garantizado por T07). Los duplicados de la misma clase se
    descartan (ver DUPLICATE_OVERLAP).
    """
    det = joint.detection
    located = {o.detection_index: o for o in joint.objects}
    drop = _duplicates(det, label)
    out = []
    for i, cls in enumerate(det.classes):
        if (label is not None and cls != label) or i in drop:
            continue
        x1, y1, x2, y2 = (float(x) for x in det.boxes[i])
        h, vert = geometry.pixel_to_angles((x1 + x2) / 2, (y1 + y2) / 2)
        track = det.track_ids[i] if det.track_ids is not None else None
        out.append(
            Candidate(cls, i, h, vert, track, float(det.scores[i]), joint.frame.received_ns,
                      located.get(i))
        )  # fmt: skip
    return sorted(out, key=lambda c: c.angle_h_deg)


def horizontal(angle_h_deg: float, s: ModeSettings, prev: str | None = None) -> str:
    """Ángulo horizontal -> "al frente" / "un poco a la izquierda" / "a la derecha".

    Con `prev` (la zona dicha antes) se aplica histéresis: si el ángulo está a menos de
    `hysteresis_deg` de un límite, se conserva la zona anterior.
    """

    def zone(center: float, side_lim: float) -> str:
        a = abs(angle_h_deg)
        if a <= center:
            return "al frente"
        side = "derecha" if angle_h_deg > 0 else "izquierda"
        return f"un poco a la {side}" if a <= side_lim else f"a la {side}"

    raw = zone(s.center_deg, s.side_deg)
    if prev is None or prev == raw:
        return raw
    h = s.hysteresis_deg
    near_limits = {zone(s.center_deg + h, s.side_deg + h), zone(s.center_deg - h, s.side_deg - h)}
    return prev if prev in near_limits else raw


def vertical(angle_v_deg: float | None, s: ModeSettings, prev: str | None = None) -> str:
    """ "", ", arriba" o ", abajo"; con histéresis respecto de `prev`."""
    if angle_v_deg is None:
        return ""
    limit = s.vertical_deg - (s.hysteresis_deg if prev else 0.0)
    word = ""
    if abs(angle_v_deg) > limit:
        word = ", arriba" if angle_v_deg > 0 else ", abajo"
    if prev and word and word != prev and abs(angle_v_deg) <= s.vertical_deg:
        return ""  # cruzó el centro pero no pasó el umbral del otro lado
    return word


def direction(angle_h_deg: float, s: ModeSettings, angle_v_deg: float | None = None) -> str:
    """Frase de dirección: la horizontal y, solo si es clara, la vertical.

    "Taza a la izquierda, abajo." Lo vertical se agrega cuando |ángulo| > vertical_deg; cerca
    del centro no se dice nada, para no alargar la guía ni hacerla cambiar a cada paso.
    """
    return horizontal(angle_h_deg, s) + vertical(angle_v_deg, s)


def meters(z_m: float, s: ModeSettings) -> str:
    """Metros redondeados para decir: 1,07 -> "1,1 metros"; 1,0 -> "1 metro"."""
    r = round(round(z_m / s.round_m) * s.round_m, 2)
    text = f"{r:g}".replace(".", ",")
    return "1 metro" if r == 1 else f"{text} metros"


@dataclass(frozen=True, slots=True)
class Measure:
    """Qué se puede decir sobre la distancia de un objeto: un número o por qué no."""

    z_m: float | None
    reason: str | None  # None si z_m se puede decir

    @property
    def ok(self) -> bool:
        return self.reason is None


def measure(obj: SpatialObject | None, s: ModeSettings) -> Measure:
    """Respeta la política de quien ubicó el objeto (área A) y agrega solo el rango validado.

    Ningún metro sin profundidad válida: sin objeto ubicado, sin Z o con una política que no es
    SPEAK_METERS, no hay número.
    """
    if obj is None:
        return Measure(None, "sin-profundidad")
    if obj.policy is DistancePolicy.ASK_REFRAME:
        return Measure(None, "reencuadrar")
    if obj.policy is not DistancePolicy.SPEAK_METERS or obj.z_m is None:
        return Measure(None, "sin-medida")
    near, far = s.range_m
    if obj.z_m < near:
        return Measure(obj.z_m, "muy-cerca")
    if obj.z_m > far:
        return Measure(obj.z_m, "muy-lejos")
    return Measure(obj.z_m, None)


def measure_phrase(m: Measure, s: ModeSettings) -> str:
    """Final de frase para una medida: "a 1,1 metros" o la explicación de por qué no."""
    if m.ok:
        return f"a {meters(m.z_m, s)}"  # type: ignore[arg-type]
    near, far = s.range_m
    return {
        "muy-cerca": f"muy cerca, a menos de {meters(near, s)}",
        "muy-lejos": f"lejos, a más de {meters(far, s)}",
        "reencuadrar": "pero está cortado en el borde: girá un poco hacia ese lado",
        "sin-medida": "pero no puedo medir su distancia",
        "sin-profundidad": "pero todavía no pude medir su distancia",
    }[m.reason or ""]
