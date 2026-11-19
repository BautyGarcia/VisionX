"""Profundidad a pedido: un cálculo activo y un buzón reemplazable, sin bloquear modos.

La cancelación es lógica: una inferencia nativa puede terminar, pero no publicar ni hablar
para una orden invalidada. El frame y su detección viajan juntos aunque salga de la caché.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from visionx.app.runtime import Joint, LatestMailbox, ModuleStatus, View
from visionx.shared.clock import Clock, MonotonicClock
from visionx.shared.contracts import Command, Mode, check_same_frame
from visionx.shared.errors import InvalidInput
from visionx.shared.interfaces import DepthEstimator, SpatialEstimator


@dataclass(slots=True)
class _Request:
    command: Command
    expires_ns: int
    earliest_ns: int
    frame: Joint | None = None
    result: Joint | None = None
    latency_ms: float | None = None
    reason: str | None = None
    reported: tuple | None = None


class PrecisionWorker:
    def __init__(
        self,
        estimator: DepthEstimator,
        spatial: SpatialEstimator,
        *,
        wait_s: float = 2.0,
        clock: Clock | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if not math.isfinite(wait_s) or wait_s < 0:
            raise InvalidInput("depth.precisa.espera_s debe ser finito y no negativo")
        self.estimator = estimator
        self.spatial = spatial
        self.wait_s = wait_s
        self.clock = clock or MonotonicClock()
        self.status = ModuleStatus("precisa", estimator.model_id)
        self.selection: dict[str, Any] = {}
        self._event = on_event or (lambda kind, data: None)
        self._lock = threading.RLock()
        self._box: LatestMailbox[_Request] = LatestMailbox()
        self._active: _Request | None = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def request(
        self,
        command: Command,
        view: View,
        now_ns: int,
        *,
        end_margin_s: float,
        accept_before_s: float = 0.5,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            expires = now_ns + int(self.wait_s * 1e9)
            if command.deadline_ns is not None:
                expires = min(expires, command.deadline_ns - int(end_margin_s * 1e9))
            earliest = command.issued_ns
            if command.mode is Mode.DISTANCE:
                earliest -= int(accept_before_s * 1e9)
            self._active = _Request(command, expires, earliest)
            self.selection = {}
            self._event(
                "precisa-pedido",
                {
                    "request_id": command.request_id,
                    "model_id": self.estimator.model_id,
                    "espera_s": max(0, (expires - now_ns) / 1e9),
                },
            )
            self._enqueue(view)

    def _enqueue(self, view: View) -> None:
        req = self._active
        if req is None or req.frame is not None or req.reason is not None:
            return
        joint = view.seen or view.measured
        if joint is None:
            return  # sin detección al recibir la orden: tomar la primera disponible
        if joint.frame.received_ns < req.earliest_ns:
            return  # esperar una detección vigente antes de gastar inferencia precisa
        req.frame = joint
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="visionx-precisa", daemon=True)
            self._thread.start()
        self._box.put(req)

    def select(
        self,
        command: Command,
        view: View,
        now_ns: int,
        invalid: Callable[[Joint], str | None],
    ) -> tuple[Joint | None, bool]:
        """(medición elegida, esperando). Un fallback conserva la continua, nunca mezcla frames."""
        with self._lock:
            req = self._active
            if req is None or req.command.request_id != command.request_id:
                return None, True
            if req.result is not None and req.reason is None:
                req.reason = invalid(req.result)
                if req.reason is not None:
                    req.reason = f"precisa-invalida: {req.reason}"
            if req.result is None and req.reason is None:
                if now_ns >= req.expires_ns:
                    req.reason = "espera-vencida"
                else:
                    self._enqueue(view)
                    return None, True
            chosen = req.result if req.reason is None else view.measured
            reason = req.reason or "precisa-valida"
            model = chosen.depth.model_id if chosen is not None and chosen.depth else None
            latency = (
                req.latency_ms
                if req.reason is None
                else chosen.depth.inference_ms
                if chosen is not None and chosen.depth
                else None
            )
            marker = (reason, chosen.frame.key if chosen else None, model)
            if req.reported != marker:
                req.reported = marker
                self.selection = {
                    "request_id": command.request_id,
                    "model_id": model,
                    "latencia_ms": latency,
                    "motivo": reason,
                    "frame_id": chosen.frame.frame_id if chosen else None,
                    "precisa_model_id": self.estimator.model_id,
                    "precisa_latencia_ms": req.latency_ms,
                }
                self._event("profundidad-elegida", self.selection)
            return chosen, False

    def cancel(self, request_id: int) -> None:
        with self._lock:
            if self._active is not None and self._active.command.request_id == request_id:
                self._active = None
                self._event("precisa-cancelada", {"request_id": request_id})

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active = None
            self._box.close()
        if self._thread is None:
            self.estimator.close()
        # Si hay inferencia nativa activa, el worker la cierra al terminar; cancelar no espera.

    def _run(self) -> None:
        try:
            while not self._closed:
                req = self._box.take(timeout_s=0.1)
                if req is None:
                    continue
                with self._lock:
                    if req is not self._active or self.clock.now_ns() >= req.expires_ns:
                        continue
                assert req.frame is not None
                start = time.perf_counter()
                result = None
                error = None
                try:
                    joint = req.frame
                    depth = self.estimator.predict(joint.frame)
                    check_same_frame(joint.frame, joint.detection, depth)
                    objects = tuple(self.spatial.locate(joint.frame, depth, joint.detection))
                    result = Joint(
                        joint.frame,
                        joint.detection,
                        depth,
                        objects,
                        precise_request_id=req.command.request_id,
                    )
                except Exception as exc:
                    # Una fábrica externa tampoco debe matar silenciosamente el worker.
                    error = f"error-precisa: {type(exc).__name__}: {exc}"
                latency = (time.perf_counter() - start) * 1000
                with self._lock:
                    now = self.clock.now_ns()
                    self.status.record(
                        req.frame.frame.key, latency, req.frame.frame.received_ns, now
                    )
                    if error:
                        self.status.errors += 1
                        self.status.last_error = error
                    discard = req is not self._active or now >= req.expires_ns or req.reason
                    self._event(
                        "precisa-descartada" if discard else "precisa-resultado",
                        {
                            "request_id": req.command.request_id,
                            "frame_id": req.frame.frame.frame_id,
                            "model_id": self.estimator.model_id,
                            "latencia_ms": latency,
                            "motivo": error or ("cancelada-o-vencida" if discard else None),
                        },
                    )
                    if not discard:
                        req.result, req.reason, req.latency_ms = result, error, latency
        finally:
            self.estimator.close()
