"""Ejecución interactiva: captura, detector y profundidad a su propio ritmo (T07).

Cómo se evita esperar frames viejos o mezclar frames:
- La captura corre en su hilo y nunca espera a los modelos. Cada frame va a una caché acotada
  (por FrameKey) y a un buzón por modelo que guarda **solo el último**: si un modelo es más
  lento que la cámara, se saltea frames en vez de acumular una cola que crece.
- El detector toma el último frame capturado. La profundidad toma el último frame **ya
  detectado**: si cada modelo eligiera su propio último frame, con la cámara más rápida que
  ambos casi nunca coincidirían y no habría unión. Cada uno sigue a su ritmo (el detector puede
  avisar presencia sin esperar metros), pero la edad de la profundidad incluye la del detector.
- Cada modelo publica su resultado con la identidad del frame. La unión busca el frame más
  reciente que tenga detección **y** profundidad del mismo (session_id, frame_id); nunca junta
  "la última detección" con "la última profundidad" de frames distintos.
- Los estados de cada módulo (edad, latencia, FPS, errores, frames salteados) se exponen para
  el dashboard y la traza.

Esto es el recorrido interactivo. La comparación científica (todos los métodos sobre todos los
pares, sin descartes) está en science.py.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from visionx.app.precision import PrecisionWorker

from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import (
    DepthResult,
    Detection,
    FrameKey,
    SpatialObject,
    StereoFrame,
    check_same_frame,
)
from visionx.shared.errors import ContractError, EstimatorError, SourceError
from visionx.shared.interfaces import DepthEstimator, Detector, FrameSource, SpatialEstimator

T = TypeVar("T")


class LatestMailbox(Generic[T]):
    """Buzón de un lugar: put() reemplaza lo que no se leyó. take() espera algo más nuevo que
    lo último entregado. Tamaño fijo por construcción: no hay cola que pueda crecer."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._version = 0
        self._taken = 0
        self.overwritten = 0  # entregas que nadie llegó a leer (frames salteados)
        self._closed = False

    def put(self, item: T) -> None:
        with self._cond:
            if self._version > self._taken:
                self.overwritten += 1
            self._item = item
            self._version += 1
            self._cond.notify_all()

    def take(self, timeout_s: float | None = None) -> T | None:
        """El ítem más reciente no entregado; None si se cerró o venció el timeout."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._version > self._taken or self._closed, timeout=timeout_s
            )
            if not ok or (self._closed and self._version == self._taken):
                return None
            self._taken = self._version
            return self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class FrameCache:
    """Últimos `max_frames` frames por identidad, para unir resultados que llegan tarde."""

    def __init__(self, max_frames: int = 30) -> None:
        self.max_frames = max_frames
        self._frames: OrderedDict[FrameKey, StereoFrame] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, frame: StereoFrame) -> None:
        with self._lock:
            self._frames[frame.key] = frame
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)

    def get(self, key: FrameKey) -> StereoFrame | None:
        with self._lock:
            return self._frames.get(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


@dataclass(slots=True)
class ModuleStatus:
    """Estado de un módulo para el dashboard. Tiempos en ns del reloj de la ejecución."""

    name: str
    model_id: str = ""
    processed: int = 0
    errors: int = 0
    last_error: str | None = None
    last_key: FrameKey | None = None
    last_latency_ms: float | None = None
    last_done_ns: int | None = None
    last_frame_received_ns: int | None = None
    _done_times: list[int] = field(default_factory=list)

    def record(self, key: FrameKey, latency_ms: float, frame_received_ns: int, now: int) -> None:
        self.processed += 1
        self.last_key = key
        self.last_latency_ms = latency_ms
        self.last_done_ns = now
        self.last_frame_received_ns = frame_received_ns
        self._done_times.append(now)
        del self._done_times[:-30]

    def fps(self) -> float | None:
        if len(self._done_times) < 2:
            return None
        span = self._done_times[-1] - self._done_times[0]
        return (len(self._done_times) - 1) / (span / 1e9) if span > 0 else None

    def age_ms(self, now: int) -> float | None:
        """Antigüedad del frame del último resultado: lo que el usuario "ve" con retraso."""
        if self.last_frame_received_ns is None:
            return None
        # Puede dar apenas negativo si `now` se leyó antes que el último registro de otro hilo
        # (o por el reloj de 15,6 ms de Windows): eso es "recién llegado", no un valor físico.
        return max(0.0, (now - self.last_frame_received_ns) / 1e6)


@dataclass(frozen=True, slots=True)
class Joint:
    """Resultados del mismo frame, ya verificados como compatibles."""

    frame: StereoFrame
    detection: Detection
    depth: DepthResult | None
    objects: tuple[SpatialObject, ...]
    precise_request_id: int | None = None


@dataclass(frozen=True, slots=True)
class View:
    """Lo último observado, para los modos (T09–T11). Cualquiera puede faltar.

    `seen`: frame + detección más nuevos, sin profundidad (presencia y dirección no esperan
    metros). `measured`: la unión más nueva con profundidad y objetos del mismo frame.
    """

    seen: Joint | None = None
    measured: Joint | None = None
    # T10: captura sin esperar detección/profundidad. Original y rectificada mantienen
    # identidad propia; el OCR no usa sus cajas para medir metros.
    captured: StereoFrame | None = None
    original: StereoFrame | None = None
    precise: PrecisionWorker | None = None


class ResultBoard:
    """Guarda los últimos resultados de cada modelo por FrameKey y arma uniones del mismo frame."""

    def __init__(self, keep: int = 30) -> None:
        self.keep = keep
        self.detections: OrderedDict[FrameKey, Detection] = OrderedDict()
        # El frame de cada detección viaja con ella: la unión no depende de que siga en la
        # caché de captura, que a muchos FPS se renueva antes de que termine la profundidad.
        self.frames: OrderedDict[FrameKey, StereoFrame] = OrderedDict()
        self.depths: OrderedDict[FrameKey, DepthResult] = OrderedDict()
        self._lock = threading.Lock()

    def _put(self, store: OrderedDict, key: FrameKey, value: Any) -> None:
        with self._lock:
            store[key] = value
            while len(store) > self.keep:
                store.popitem(last=False)

    def put_detection(self, det: Detection, frame: StereoFrame | None = None) -> None:
        if frame is not None:
            self._put(self.frames, det.key, frame)
        self._put(self.detections, det.key, det)

    def frame(self, key: FrameKey) -> StereoFrame | None:
        with self._lock:
            return self.frames.get(key)

    def put_depth(self, depth: DepthResult) -> None:
        self._put(self.depths, depth.key, depth)

    def latest_detection_key(self) -> FrameKey | None:
        with self._lock:
            return next(reversed(self.detections), None)

    def latest_common_key(self) -> FrameKey | None:
        """El frame más nuevo con detección y profundidad. Las claves de una sesión crecen con
        frame_id; se busca de la detección más nueva hacia atrás."""
        with self._lock:
            for key in reversed(self.detections):
                if key in self.depths:
                    return key
        return None

    def get(self, key: FrameKey) -> tuple[Detection | None, DepthResult | None]:
        with self._lock:
            return self.detections.get(key), self.depths.get(key)


class Pipeline:
    """Arma captura + detector + profundidad (opcional) + ubicación espacial (opcional).

    Uso: `start()`, leer `latest_joint()` / `status()` desde el controlador o el dashboard y
    `stop()`. Si la fuente falla (`SourceError`) la ejecución se detiene y `error` lo indica;
    si un modelo falla en un frame, se cuenta y se sigue con el siguiente.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        depth: DepthEstimator | None = None,
        spatial: SpatialEstimator | None = None,
        *,
        clock: Clock | None = None,
        cache_frames: int = 30,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        precise: PrecisionWorker | None = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.depth = depth
        self.spatial = spatial
        self.precise = precise
        self.clock = clock or MonotonicClock()
        self.cache = FrameCache(cache_frames)
        self.board = ResultBoard(cache_frames)
        self._on_event = on_event or (lambda kind, data: None)
        self._det_box: LatestMailbox[StereoFrame] = LatestMailbox()
        self._depth_box: LatestMailbox[StereoFrame] = LatestMailbox()
        self.capture_status = ModuleStatus("captura", type(source).__name__)
        self.detector_status = ModuleStatus("detector", detector.model_id)
        self.depth_status = ModuleStatus("profundidad", depth.model_id if depth else "-")
        self.spatial_errors = 0
        self.latest_frame: StereoFrame | None = None
        self.error: str | None = None
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._joint_lock = threading.Lock()
        self._joint: Joint | None = None
        self._seen: Joint | None = None  # caché de latest_seen()
        self._captured: tuple[StereoFrame | None, StereoFrame | None] = (None, None)
        self._threads: list[threading.Thread] = []

    # -- control ---------------------------------------------------------------------------

    def start(self) -> None:
        targets = [("captura", self._capture_loop), ("detector", self._detector_loop)]
        if self.depth is not None:
            targets.append(("profundidad", self._depth_loop))
        for name, fn in targets:
            thread = threading.Thread(target=fn, name=f"visionx-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self.precise is not None:
            self.precise.close()
        self._det_box.close()
        self._depth_box.close()
        for thread in self._threads:
            thread.join(timeout=timeout_s)
        self.source.close()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def queue_sizes(self) -> dict[str, int]:
        """Todo lo que podría crecer. Debe mantenerse acotado aunque los modelos no den abasto."""
        return {
            "cache_frames": len(self.cache),
            "detecciones": len(self.board.detections),
            "frames_detectados": len(self.board.frames),
            "profundidades": len(self.board.depths),
        }

    def dropped(self) -> dict[str, int]:
        """Frames que cada modelo salteó por estar ocupado (esperable, no es un error)."""
        return {"detector": self._det_box.overwritten, "profundidad": self._depth_box.overwritten}

    def latest_joint(self) -> Joint | None:
        with self._joint_lock:
            return self._joint

    def joint_for(self, key: FrameKey) -> Joint | None:
        """Unión para un frame dado (por ejemplo, el congelado por una orden de Distancia)."""
        frame = self.board.frame(key) or self.cache.get(key)
        det, depth = self.board.get(key)
        if frame is None or det is None:
            return None
        return self._make_joint(frame, det, depth)

    # -- hilos -----------------------------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:  # fin normal de una grabación
                    break
                now = self.clock.now_ns()
                self.cache.put(frame)
                self.latest_frame = frame
                original = getattr(self.source, "original_frame", None)
                if not frame.rectified:
                    original = frame
                # Publicar juntos evita mezclar el original de otro frame entre hilos.
                with self._joint_lock:
                    self._captured = (frame, original)
                self.capture_status.record(frame.key, 0.0, frame.received_ns, now)
                self._det_box.put(frame)
                self._on_event("frame", {"frame_id": frame.frame_id, "session": frame.session_id})
        except SourceError as exc:
            self.error = f"fuente: {exc}"
            self._on_event("error", {"modulo": "captura", "error": str(exc)})
        finally:
            self.finished.set()
            # Al terminar la fuente, el detector procesa lo último que haya y se detiene.
            self._det_box.close()

    def _model_loop(
        self,
        box: LatestMailbox[StereoFrame],
        status: ModuleStatus,
        run: Callable[[StereoFrame], Any],
        publish: Callable[[Any], None],
    ) -> None:
        while not self._stop.is_set():
            frame = box.take(timeout_s=0.5)
            if frame is None:
                if box.closed:
                    return
                continue
            t0 = time.perf_counter()
            try:
                result = run(frame)
                # Un modelo no puede devolver otro frame (el detector devuelve frame + resultado).
                check_same_frame(frame, result[1] if isinstance(result, tuple) else result)
            except (EstimatorError, ContractError) as exc:
                status.errors += 1
                status.last_error = str(exc)
                self._on_event("error", {"modulo": status.name, "error": str(exc)})
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            status.record(frame.key, latency_ms, frame.received_ns, self.clock.now_ns())
            publish(result)
            self._on_event(
                status.name,
                {
                    "frame_id": frame.frame_id,
                    "model_id": status.model_id,
                    "latencia_ms": round(latency_ms, 2),
                    "edad_ms": round(status.age_ms(self.clock.now_ns()) or 0.0, 2),
                },
            )
            self._update_joint()

    def _detector_loop(self) -> None:
        def publish(item: tuple[StereoFrame, Detection]) -> None:
            frame, det = item
            self.board.put_detection(det, frame)
            # La profundidad trabaja sobre frames ya detectados (ver docstring del módulo).
            if self.depth is not None:
                self._depth_box.put(frame)

        def run(frame: StereoFrame) -> tuple[StereoFrame, Detection]:
            return frame, self.detector.predict(frame)

        try:
            self._model_loop(self._det_box, self.detector_status, run, publish)
        finally:
            # La profundidad depende del detector: termina cuando el detector ya no le pasa nada.
            self._depth_box.close()

    def _depth_loop(self) -> None:
        assert self.depth is not None
        self._model_loop(
            self._depth_box, self.depth_status, self.depth.predict, self.board.put_depth
        )

    # -- unión -----------------------------------------------------------------------------

    def _make_joint(
        self, frame: StereoFrame, det: Detection, depth: DepthResult | None
    ) -> Joint | None:
        objects: tuple[SpatialObject, ...] = ()
        if depth is not None and self.spatial is not None:
            try:
                objects = tuple(self.spatial.locate(frame, depth, det))
            except (EstimatorError, ValueError) as exc:
                self.spatial_errors += 1
                self._on_event("error", {"modulo": "espacial", "error": str(exc)})
        return Joint(frame, det, depth, objects)

    def latest_seen(self) -> Joint | None:
        """Frame + detección más nuevos, sin profundidad. Con profundidad configurada, la unión
        completa llega más tarde: buscar no la espera para avisar presencia."""
        key = self.board.latest_detection_key()
        if key is None:
            return None
        with self._joint_lock:
            cached = self._seen
        if cached is not None and cached.frame.key == key:
            return cached
        det, _ = self.board.get(key)
        frame = self.board.frame(key)
        if det is None or frame is None:
            return cached
        seen = Joint(frame, det, None, ())
        with self._joint_lock:
            if self._seen is None or _newer(key, self._seen.frame.key):
                self._seen = seen
        return seen

    def view(self) -> View:
        """Lo que reciben los modos en cada tick del controlador."""
        measured = self.latest_joint()
        with self._joint_lock:
            captured, original = self._captured
        return View(
            self.latest_seen(),
            measured if self.depth is not None else None,
            captured,
            original,
            self.precise,
        )

    def _update_joint(self) -> None:
        if self.depth is None:
            # Sin profundidad: la unión es frame + detección (alcanza para avisar presencia).
            key = self.board.latest_detection_key()
        else:
            key = self.board.latest_common_key()
        if key is None:
            return
        current = self.latest_joint()
        if current is not None and current.frame.key == key:
            return
        joint = self.joint_for(key)
        if joint is None:
            return
        with self._joint_lock:
            # Dos hilos pueden terminar casi juntos y armar la misma unión: solo la registra
            # el que la publica, y nunca se reemplaza por una más vieja.
            if self._joint is not None and not _newer(joint.frame.key, self._joint.frame.key):
                return
            self._joint = joint
        self._on_event(
            "union",
            {
                "frame_id": key.frame_id,
                "objetos": [
                    {"clase": o.label, "track_id": o.track_id, "z_m": o.z_m,
                     "dispersion_m": o.z_spread_m, "fraccion_valida": round(o.valid_fraction, 3),
                     "angulo_h": round(o.angle_h_deg, 1), "politica": str(o.policy)}
                    for o in joint.objects
                ],
            },
        )  # fmt: skip


def _newer(a: FrameKey, b: FrameKey) -> bool:
    return a.session_id != b.session_id or a.frame_id > b.frame_id
