"""Comparación científica y traza (T07).

`compare` procesa **todos** los pares de una fuente con **todos** los métodos, en el mismo
orden y sin descartar frames: así cada método se mide sobre el mismo contenido y su tiempo es
su coste completo (no se reutiliza una caché de otro método). Es lo opuesto del recorrido
interactivo, que saltea frames para mantenerse actual.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from visionx.shared.contracts import DepthResult, Detection, check_same_frame
from visionx.shared.errors import ContractError, EstimatorError
from visionx.shared.interfaces import FrameSource


class TraceWriter:
    """JSONL con una línea por evento: {"t_ns", "tipo", ...}. Seguro entre hilos.

    t_ns es el reloj monotónico local: sirve para duraciones dentro de esta ejecución, no para
    comparar con otra máquina.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._lock = threading.Lock()
        self.lines = 0

    def __call__(self, kind: str, data: Mapping[str, Any]) -> None:
        record = {"t_ns": time.monotonic_ns(), "tipo": kind, **data}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._file.closed:
                return
            self._file.write(line + "\n")
            self.lines += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _summary(result: Any) -> dict[str, Any]:
    if isinstance(result, Detection):
        return {"objetos": len(result), "clases": "|".join(result.classes)}
    if isinstance(result, DepthResult):
        valid = result.valid_mask
        z = result.depth_m
        median = float(np.median(z[valid])) if z is not None and valid.any() else None
        return {
            "estado": result.status.value,
            "cobertura": round(float(valid.mean()), 4),
            "z_mediana_m": None if median is None else round(median, 3),
        }
    return {}


def compare(
    source: FrameSource,
    methods: Mapping[str, Any],
    out_csv: str | Path,
    *,
    max_frames: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Corre cada método (`predict(frame)`) sobre cada frame y escribe una fila por par
    frame-método: tiempo de pared, estado y un resumen. Devuelve medianas por método.

    Un fallo de un método en un frame queda registrado como fila con error, no se saltea.
    """
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    times: dict[str, list[float]] = {name: [] for name in methods}
    failures: dict[str, int] = dict.fromkeys(methods, 0)
    frames = 0
    fields = ["session_id", "frame_id", "metodo", "model_id", "ms", "error"]
    extra: list[str] = ["objetos", "clases", "estado", "cobertura", "z_mediana_m"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        try:
            while max_frames is None or frames < max_frames:
                frame = source.read()
                if frame is None:
                    break
                frames += 1
                for name, method in methods.items():
                    row: dict[str, Any] = {
                        "session_id": frame.session_id,
                        "frame_id": frame.frame_id,
                        "metodo": name,
                        "model_id": getattr(method, "model_id", name),
                    }
                    t0 = time.perf_counter()
                    try:
                        result = method.predict(frame)
                        check_same_frame(frame, result)
                        ms = (time.perf_counter() - t0) * 1000
                        times[name].append(ms)
                        row.update(ms=round(ms, 2), **_summary(result))
                    except (EstimatorError, ContractError) as exc:
                        failures[name] += 1
                        row.update(ms=round((time.perf_counter() - t0) * 1000, 2), error=str(exc))
                    writer.writerow(row)
        finally:
            source.close()
    return {
        name: {
            "frames": frames,
            "fallos": failures[name],
            "mediana_ms": round(float(np.median(t)), 2) if t else None,
            "p90_ms": round(float(np.percentile(t, 90)), 2) if t else None,
        }
        for name, t in times.items()
    }
