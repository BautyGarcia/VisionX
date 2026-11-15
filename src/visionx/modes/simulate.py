"""Simulación determinista de los modos sobre una grabación (T09, evaluación científica).

`app ejecutar` corre los modelos en hilos y a la velocidad de la CPU: dos corridas no dan lo
mismo y en esta notebook la profundidad llega con segundos de atraso. Para evaluar la LÓGICA de
buscar/distancia sobre clips reales hace falta otra cosa:

- El tiempo es el grabado (`ManualClock`): un clip de 70 s dura 70 s simulados aunque procesarlo
  tarde 10 minutos.
- Las latencias de detector y profundidad se EMULAN con valores dados (`det_latency_s`,
  `depth_latency_s`): un resultado se entrega recién cuando, en tiempo grabado, habría terminado,
  y mientras un modelo está "ocupado" no toma frames nuevos (como el buzón de T07). Con 0 s se
  obtiene el comportamiento ideal.
- Las órdenes se dan en segundos del clip, como en `ejecutar --ordenes`.

Lo que dice el sistema queda en una transcripción con el segundo del clip en que se pidió
decirlo. No mide latencia audible ni tiempos reales: eso es M1 con equipo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from visionx.app.controller import Controller
from visionx.app.runtime import Joint, View
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import AudioEvent
from visionx.shared.errors import EstimatorError
from visionx.shared.interfaces import DepthEstimator, Detector, FrameSource, SpatialEstimator


@dataclass(frozen=True, slots=True)
class Spoken:
    t_s: float  # segundo del clip en que se pidió decirlo
    request_id: int
    priority: int
    text: str


class RecordingSink:
    """AudioSink que no suena: anota qué se pidió decir y cuándo (reloj de la simulación)."""

    def __init__(self, clock: ManualClock, t0_ns: Callable[[], int]) -> None:
        self.clock = clock
        self.t0_ns = t0_ns
        self.spoken: list[Spoken] = []
        self.cancelled: list[tuple[float, int | None]] = []

    def _t(self) -> float:
        return round((self.clock.now_ns() - self.t0_ns()) / 1e9, 3)

    def submit(self, event: AudioEvent) -> None:
        self.spoken.append(Spoken(self._t(), event.request_id, int(event.priority),
                                  event.text_or_sound))  # fmt: skip

    def cancel(self, request_id: int) -> None:
        self.cancelled.append((self._t(), request_id))

    def cancel_all(self) -> None:
        self.cancelled.append((self._t(), None))

    def close(self) -> None:
        pass


@dataclass(slots=True)
class SimulationResult:
    spoken: list[Spoken]
    events: list[tuple[float, str, dict[str, Any]]]
    frames: int
    detections: int
    measurements: int
    duration_s: float
    errors: list[str] = field(default_factory=list)


def simulate(
    source: FrameSource,
    detector: Detector,
    make_controller: Callable[[Callable[[], View], RecordingSink, Callable], Controller],
    clock: ManualClock,
    orders: list[tuple[float, str]],
    *,
    depth: DepthEstimator | None = None,
    spatial: SpatialEstimator | None = None,
    det_latency_s: float = 0.0,
    depth_latency_s: float = 0.0,
    every: int = 1,
    tick_s: float | None = None,
    tail_s: float = 0.0,
) -> SimulationResult:
    """Recorre la fuente frame a frame con tiempo emulado.

    `make_controller(latest_view, sink, on_event)` arma el controlador con sus modos. `every`:
    el detector mira uno de cada N frames. `tick_s`: para fuentes sin tiempo propio (la
    sintética), cuánto avanza el reloj antes de cada frame. `tail_s`: segundos que se sigue
    simulando después del último frame (sin frames nuevos), para que los modos cierren.
    """
    t0: list[int | None] = [None]
    events: list[tuple[float, str, dict[str, Any]]] = []

    def t_s() -> float:
        return round((clock.now_ns() - (t0[0] or clock.now_ns())) / 1e9, 3)

    def on_event(kind: str, data: dict[str, Any]) -> None:
        events.append((t_s(), kind, data))

    state = {"view": View()}
    sink = RecordingSink(clock, lambda: t0[0] or clock.now_ns())
    controller = make_controller(lambda: state["view"], sink, on_event)
    pending = sorted(orders)
    det_free_ns = depth_free_ns = 0
    seen_queue: list[tuple[int, Joint]] = []  # (listo en ns, unión sin profundidad)
    measured_queue: list[tuple[int, Joint]] = []
    seen = measured = None
    n_frames = n_det = n_depth = 0
    errors: list[str] = []

    def deliver(now: int) -> None:
        nonlocal seen, measured, depth_free_ns, n_depth
        for ready, joint in [q for q in seen_queue if q[0] <= now]:
            seen_queue.remove((ready, joint))
            seen = joint
            # La profundidad toma el último frame ya detectado cuando se libera (T07).
            if depth is not None and ready >= depth_free_ns:
                try:
                    d = depth.predict(joint.frame)
                    objs = tuple(spatial.locate(joint.frame, d, joint.detection)) if spatial else ()
                except (EstimatorError, ValueError) as exc:
                    errors.append(f"profundidad frame {joint.frame.frame_id}: {exc}")
                    continue
                n_depth += 1
                done = ready + int(depth_latency_s * 1e9)
                depth_free_ns = done
                measured_queue.append((done, Joint(joint.frame, joint.detection, d, objs)))
        for ready, joint in [q for q in measured_queue if q[0] <= now]:
            measured_queue.remove((ready, joint))
            measured = joint
        state["view"] = View(seen, measured)

    def give_orders() -> None:
        while pending and pending[0][0] <= t_s():
            controller.handle_text(pending.pop(0)[1])

    index = 0
    while True:
        if tick_s:
            clock.advance(int(tick_s * 1e9))
        frame = source.read()
        if frame is None:
            break
        if t0[0] is None:
            t0[0] = clock.now_ns()
        n_frames += 1
        now = clock.now_ns()
        give_orders()
        if index % every == 0 and now >= det_free_ns:
            try:
                det = detector.predict(frame)
            except EstimatorError as exc:
                errors.append(f"detector frame {frame.frame_id}: {exc}")
            else:
                n_det += 1
                det_free_ns = now + int(det_latency_s * 1e9)
                seen_queue.append((det_free_ns, Joint(frame, det, None, ())))
        index += 1
        deliver(now)
        controller.tick()
    # Sin frames nuevos: el tiempo sigue para que los modos noten la ausencia y cierren.
    step = int(0.1 * 1e9)
    for _ in range(int(tail_s / 0.1)):
        clock.advance(step)
        give_orders()
        deliver(clock.now_ns())
        controller.tick()
    return SimulationResult(sink.spoken, events, n_frames, n_det, n_depth, t_s(), errors)
