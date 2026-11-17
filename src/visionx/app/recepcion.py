"""Recepción offline de DepthEstimator: contrato, coste y ubicación real de la app.

No mide latencia audible ni certifica precisión: usa pocos pares de desarrollo.
Las advertencias de rango son heurísticas; la coherencia d/Z sí es contractual.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from visionx.app.depth_registry import build_estimator
from visionx.camera import RectifiedSource, ReplaySource, load_recording
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import DepthResult, check_depth_geometry, check_same_frame
from visionx.shared.errors import InvalidInput, SourceError
from visionx.shared.geometry import StereoGeometry

# Cámara quieta durante todo el clip (manifest.info.nota). Frames 152 y 240 de
# humo-distancias.jsonl, próximos a las consultas de taza/vaso y botella. Se usan
# tiempos GRABADOS del índice, nunca los t_ns de ejecución del script interactivo.
DISTANCIAS_SESSION = "elp-20261114-153207"
DISTANCIAS_MARKS = [
    {"t_s": 2.704669708, "objeto": "vaso rojo", "clase": "cup", "lado": "derecha", "cinta_m": 0.5},
    {
        "t_s": 2.704669708,
        "objeto": "taza blanca",
        "clase": "cup",
        "lado": "izquierda",
        "cinta_m": 1.0,
    },
    {"t_s": 4.260034250, "objeto": "botella", "clase": "bottle", "cinta_m": 1.0},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(name: str, cfg: dict) -> dict:
    """Verificar ANTES de invocar una fábrica externa; los baselines verifican su caché."""
    entry = ((cfg.get("depth") or {}).get("estimadores") or {}).get(name)
    if name in {"B0", "B1", "B2", "B3", "B4"} or not isinstance(entry, dict):
        return {}
    weights = next(
        (entry[k] for k in ("pesos", "weights", "weights_path", "checkpoint") if entry.get(k)), None
    )
    expected = entry.get("sha256")
    if expected is not None and (
        not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)
    ):
        raise InvalidInput(f"{name}: sha256 debe contener 64 dígitos hexadecimales")
    if expected is not None and not weights:
        raise InvalidInput(
            f"{name}: sha256 requiere una ruta pesos (o weights/weights_path/checkpoint)"
        )
    if weights is None:
        return {"verificacion": "la entrada no declara pesos ni sha256"}
    path = Path(weights)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise InvalidInput(f"{name}: pesos faltantes: {path}")
    actual = sha256(path)
    if expected is not None and actual != expected.lower():
        raise InvalidInput(
            f"{name}: SHA-256 distinto en {path}: esperado {expected}, recibido {actual}"
        )
    return {"ruta": str(path), "sha256": actual, "verificado": expected is not None}


def resident_bytes() -> int | None:
    """RSS del proceso sin dependencias extra. macOS informa máximo de vida del proceso."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (key, ctypes.c_size_t)
                for key in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize)
        return None
    if sys.platform.startswith("linux"):
        try:
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return None


class PeakMemory:
    """Pico RSS muestreado cada 10 ms, desde construcción hasta close (incluye nativos)."""

    def __init__(self):
        self.stop = threading.Event()
        self.baseline = resident_bytes()
        self.peak = self.baseline
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while True:
            value = resident_bytes()
            if value is not None:
                self.peak = max(self.peak or 0, value)
            if self.stop.wait(0.01):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {
            "base_mib": None if self.baseline is None else self.baseline / 2**20,
            "pico_mib": None if self.peak is None else self.peak / 2**20,
            "metodo": "ru_maxrss: máximo de vida del proceso"
            if sys.platform == "darwin"
            else "RSS proceso muestreado cada 10 ms",
            "alcance": (
                "construcción, warmup, inferencia, validación y close; RAM CPU, no VRAM; "
                "incluye bibliotecas y pares residentes"
            ),
        }


def validate_result(
    frame, result, estimator, capture_geometry, model_id
) -> tuple[DepthResult, dict]:
    if not isinstance(result, DepthResult):
        raise InvalidInput("predict debe devolver un DepthResult")
    # Revalidar también productores que modificaron una instancia después de construirla.
    result = replace(result)
    check_same_frame(frame, result)
    geometry = getattr(estimator, "geometry", capture_geometry)
    if not isinstance(geometry, StereoGeometry):
        raise InvalidInput("geometry debe ser StereoGeometry")
    if result.size != capture_geometry.size and not hasattr(estimator, "geometry"):
        raise InvalidInput("otra resolución sin declarar geometry de salida")
    expected_geometry = capture_geometry.resized(*geometry.size)
    if expected_geometry.geometry_id != geometry.geometry_id:
        raise InvalidInput(
            "geometry no es un reescalado de la captura; la app solo transforma máscaras por resize"
        )
    check_depth_geometry(result, geometry)
    if estimator.model_id != model_id or result.model_id != model_id:
        raise InvalidInput("model_id inestable o distinto entre estimador y DepthResult")
    z, valid = geometry.disparity_to_depth(result.disparity_px, result.valid_mask)
    values = z[valid]
    warnings = []
    if values.size and float(((values < 0.05) | (values > 100)).mean()) > 0.9:
        warnings.append(
            "rango Z mayormente fuera de 0,05–100 m: posible confusión metros/disparidad; "
            "revisar escala con cinta"
        )
    if (
        values.size
        and np.percentile(result.disparity_px[valid], 90) <= 3
        and np.median(values) > 10
    ):
        warnings.append(
            "disparidad <=3 px y Z mediana >10 m: posible confusión metros/disparidad; "
            "es una sospecha por rango, verificar con cinta"
        )
    if result.depth_m is None:
        warnings.append("depth_m ausente (permitido): app/spatial.py no dará metros por objeto")
    u = result.uncertainty
    details = {
        "frame_id": frame.frame_id,
        "session_id": frame.session_id,
        "geometry_id": geometry.geometry_id,
        "size": result.size,
        "status": result.status.value,
        "reason": result.reason,
        "model_id": model_id,
        "inference_ms_declarada": float(result.inference_ms),
        "cobertura": float(valid.mean()),
        "disparity_px": value_range(result.disparity_px[valid]),
        "depth_m": value_range(values),
        "depth_m_recibida": result.depth_m is not None,
        "incertidumbre": {
            "presente": u is not None,
            "tipo": result.uncertainty_kind.value,
            "rango_px": None if u is None else value_range(u[valid]),
        },
        "avisos": warnings,
    }
    # Conservar el comportamiento EXACTO de la app: si solo viene disparidad, el
    # contrato pasa pero spatial no informa metros. Z derivada solo sirve al diagnóstico.
    return result, details


def value_range(values):
    return None if values.size == 0 else [float(values.min()), float(values.max())]


def prepare_clip(clip: Path, cfg: dict, pairs: int):
    """Muestreo uniforme de N pares y marcas físicas; lectura pública con hashes de replay."""
    from visionx.calibration import StereoCalibration

    manifest, entries, truncated = load_recording(clip)
    if truncated:
        raise SourceError(
            "índice del clip truncado: completar/verificar la grabación antes de recibir"
        )
    if len(entries) < pairs:
        raise InvalidInput(f"se pidieron {pairs} pares, el clip contiene {len(entries)}")
    calibration_path = Path(cfg["camera"]["calibration"])
    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path
    calibration = StereoCalibration.load(calibration_path)
    # RectifiedSource rechaza doble rectificación: no aceptar geometría de origen ambigua.
    if manifest.get("rectificado"):
        raise InvalidInput(
            "recepción requiere clip crudo y camera.calibration para rectificar igual que la app"
        )
    indexes = set(np.linspace(0, len(entries) - 1, pairs, dtype=int).tolist())
    first_ns = entries[0]["received_ns"]
    times = [(entry["received_ns"] - first_ns) / 1e9 for entry in entries]
    marks = (cfg.get("recepcion") or {}).get("marcas")
    if marks is None:
        marks = DISTANCIAS_MARKS if manifest.get("session_id") == DISTANCIAS_SESSION else []
    selected_marks = {}
    for mark in marks:
        if not times[0] <= mark["t_s"] <= times[-1]:
            raise InvalidInput(f"marca fuera del clip: {mark}")
        index = min(range(len(times)), key=lambda i: abs(times[i] - mark["t_s"]))
        selected_marks.setdefault(index, []).append(dict(mark))
    all_indexes = indexes | selected_marks.keys()
    raw = ReplaySource(clip)

    # Un iterador mantiene la posición sin tocar atributos privados de ReplaySource.
    # Replay verifica los hashes de los pares salteados; solo rectificamos los elegidos.
    def selected_frames():
        for i in range(max(all_indexes) + 1):
            frame = raw.read()
            if i in all_indexes:
                yield i, frame

    iterator = selected_frames()
    yield_frame = []

    class IteratorSource:
        def read(self):
            item = next(iterator, None)
            if item is None:
                return None
            i, frame = item
            yield_frame.append(i)
            return frame

        def close(self):
            raw.close()

    source = RectifiedSource(IteratorSource(), calibration)
    frames, mark_frames = [], []
    try:
        while (frame := source.read()) is not None:
            i = yield_frame.pop()
            if i in indexes:
                frames.append(frame)
            if i in selected_marks:
                mark_frames.append((frame, times[i], selected_marks[i]))
    finally:
        source.close()
    return (
        frames,
        mark_frames,
        source.geometry,
        {
            "session_id": manifest.get("session_id"),
            "pares_clip": len(entries),
            "nota": (manifest.get("info") or {}).get("nota"),
            "manifest_sha256": sha256(clip / "manifest.yaml"),
            "index_sha256": sha256(clip / "frames.jsonl"),
            "calibration_sha256": sha256(calibration_path),
            "geometry": asdict(source.geometry),
            "marcas": marks,
        },
    )


def detect_marks(mark_frames, cfg):
    if not mark_frames:
        return []
    from visionx.detection import DetectorSettings, YoloDetector

    detector = YoloDetector(DetectorSettings.from_config(cfg))
    try:
        return [(frame, t_s, marks, detector.predict(frame)) for frame, t_s, marks in mark_frames]
    finally:
        detector.close()


def locate_marks(frame, t_s, marks, detection, result, cfg, geometry, depth_geometry):
    # Import diferido para compartir exactamente construcción y parámetros con ejecutar.
    from visionx.app.__main__ import build_spatial

    objects = build_spatial(cfg, geometry, depth_geometry).locate(frame, result, detection)
    rows = []
    for mark in marks:
        candidates = [obj for obj in objects if obj.label == mark["clase"]]
        if mark.get("lado") == "derecha":
            candidates = [obj for obj in candidates if obj.angle_h_deg > 0]
        elif mark.get("lado") == "izquierda":
            candidates = [obj for obj in candidates if obj.angle_h_deg < 0]
        # No elegir por cercanía a la cinta: sería sesgar la evaluación.
        obj = candidates[0] if len(candidates) == 1 else None
        z = obj.z_m if obj else None
        rows.append(
            {
                **mark,
                "frame_id": frame.frame_id,
                "t_clip_s": t_s,
                "z_m": z,
                "error_m": None if z is None else z - mark["cinta_m"],
                "politica": obj.policy.value if obj else "ausente" if not candidates else "ambiguo",
                "fraccion_valida": obj.valid_fraction if obj else None,
                "detector_model_id": detection.model_id,
                "aviso_unidades": "posible confusión metros/disparidad o escala (>4x contra cinta)"
                if z is not None and not 0.25 <= z / mark["cinta_m"] <= 4
                else None,
            }
        )
    return rows


def receive_method(name, cfg, geometry, frames, detections, warmup):
    report = {"nombre": name, "errores": [], "pares": [], "objetos": [], "close_ok": False}
    estimator = None
    times = []
    with PeakMemory() as memory:
        try:
            report["pesos"] = verify_weights(name, cfg)
            estimator = build_estimator(name, cfg, geometry)
            model_id = estimator.model_id
            report["model_id"] = model_id
            report["calentamiento"] = warmup
            for i in range(warmup):
                result = estimator.predict(frames[i % len(frames)])
                validate_result(frames[i % len(frames)], result, estimator, geometry, model_id)
            for frame in frames:
                start = time.perf_counter_ns()
                result = estimator.predict(frame)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                result, details = validate_result(frame, result, estimator, geometry, model_id)
                details["latencia_ms"] = elapsed
                report["pares"].append(details)
                times.append(elapsed)
            for frame, t_s, marks, detection in detections:
                result, _ = validate_result(
                    frame, estimator.predict(frame), estimator, geometry, model_id
                )
                report["objetos"].extend(
                    locate_marks(
                        frame,
                        t_s,
                        marks,
                        detection,
                        result,
                        cfg,
                        geometry,
                        getattr(estimator, "geometry", geometry),
                    )
                )
        except Exception as exc:
            # Una entrega puede fallar en código externo con cualquier excepción. Preservar
            # clase/mensaje y cerrar; continuar contra baselines para dejar evidencia útil.
            report["errores"].append(f"{type(exc).__name__}: {exc}")
        finally:
            if estimator is not None:
                try:
                    estimator.close()
                    report["close_ok"] = True
                except Exception as exc:
                    report["errores"].append(f"close(): {type(exc).__name__}: {exc}")
    report["memoria"] = memory.summary()
    report["latencia"] = {
        "n": len(times),
        "p50_ms": float(np.percentile(times, 50)) if times else None,
        "p95_ms": float(np.percentile(times, 95)) if times else None,
        "alcance": (
            "predict de pared; excluye warmup, lectura, detector y validación; percentil lineal"
        ),
    }
    report["ok"] = not report["errores"] and len(times) == len(frames)
    del estimator
    gc.collect()
    return report


def render_table(report):
    lines = [
        "Recepción de profundidad (desarrollo; sin audio ni dispositivos)",
        "",
        "| Método | Contrato | n | P50 ms | P95 ms | Pico RAM MiB | Incertidumbre |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for method in report["metodos"]:
        latency = method["latencia"]
        uncertainty = sorted({p["incertidumbre"]["tipo"] for p in method["pares"]})
        ranges = [
            p["incertidumbre"]["rango_px"]
            for p in method["pares"]
            if p["incertidumbre"]["rango_px"]
        ]
        u = (
            ", ".join("no trae" if kind == "none" else kind for kind in uncertainty)
            or "sin resultados"
        )
        if ranges:
            u += f" [{min(r[0] for r in ranges):.4g}, {max(r[1] for r in ranges):.4g}] px"
        lines.append(
            f"| {method['nombre']} | {'OK' if method['ok'] else 'FALLÓ'} | {latency['n']} | "
            f"{number(latency['p50_ms'], 1)} | {number(latency['p95_ms'], 1)} | "
            f"{number(method['memoria']['pico_mib'], 1)} | {u} |"
        )
        lines.extend(f"ERROR {method['nombre']}: {error}" for error in method["errores"])
        for pair in method["pares"]:
            lines.extend(
                f"AVISO {method['nombre']} frame {pair['frame_id']}: {warning}"
                for warning in pair["avisos"]
            )
        for obj in method["objetos"]:
            if obj["aviso_unidades"]:
                lines.append(f"AVISO {method['nombre']} {obj['objeto']}: {obj['aviso_unidades']}")
    keys = {}
    for method in report["metodos"]:
        for obj in method["objetos"]:
            keys[(obj["objeto"], obj["frame_id"])] = obj
    if keys:
        names = [method["nombre"] for method in report["metodos"]]
        lines += [
            "",
            "Metros con app/spatial.py (Z axial; política entre paréntesis)",
            "",
            "| Objeto | Frame / clip s | Cinta m | " + " | ".join(names) + " |",
            "|---|---:|---:|" + "---:|" * len(names),
        ]
        for (label, frame_id), obj in keys.items():
            cells = []
            for method in report["metodos"]:
                match = next(
                    (
                        o
                        for o in method["objetos"]
                        if o["objeto"] == label and o["frame_id"] == frame_id
                    ),
                    None,
                )
                cells.append(
                    "sin resultado"
                    if match is None
                    else f"{number(match['z_m'])} ({match['politica']})"
                )
            lines.append(
                f"| {label} | {frame_id} / {obj['t_clip_s']:.3f} | {obj['cinta_m']:.2f} | "
                + " | ".join(cells)
                + " |"
            )
    else:
        lines.append("\nSin paridad: clip sin marcas de cinta o detector no disponible (ver JSON).")
    lines.extend(f"ERROR: {error}" for error in report["errores"])
    return "\n".join(lines) + "\n"


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def cmd_receive(args):
    date = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = re.sub(r"[^\w.-]", "_", args.nombre)
    output = Path(args.salida) if args.salida else Path("outputs/recepcion") / f"{date}-{safe_name}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "fecha_utc": date,
        "nombre": args.nombre,
        "clip": str(args.clip),
        "equipo": platform.platform(),
        "python": platform.python_version(),
        "errores": [],
        "metodos": [],
    }
    try:
        if args.pares <= 0 or args.calentamiento < 1:
            raise InvalidInput("--pares debe ser > 0 y --calentamiento >= 1")
        cfg = load_config(args.config)
        report["config"] = cfg
        report["entorno"] = environment()
        frames, marks, geometry, evidence = prepare_clip(Path(args.clip), cfg, args.pares)
        report["clip_evidencia"] = evidence
        try:
            detections = detect_marks(marks, cfg)
        except Exception as exc:
            report["errores"].append(f"paridad/detector: {type(exc).__name__}: {exc}")
            detections = []
        for name in dict.fromkeys([args.nombre, *args.contra]):
            print(
                f"Recibiendo {name}: {len(frames)} pares + {args.calentamiento} warmup…", flush=True
            )
            report["metodos"].append(
                receive_method(name, cfg, geometry, frames, detections, args.calentamiento)
            )
    except Exception as exc:
        report["errores"].append(f"{type(exc).__name__}: {exc}")
    report["ok"] = (
        not report["errores"]
        and bool(report["metodos"])
        and all(m["ok"] for m in report["metodos"])
    )
    table = render_table(report)
    (output / "recepcion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "tabla.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Evidencia: {output / 'recepcion.json'}")
    return 0 if report["ok"] else 1


def environment():
    """Identificar código y librerías sin cargar redes opcionales para obtener su versión."""
    import cv2

    packages = {}
    for package in ("numpy", "opencv-contrib-python", "torch", "transformers", "ultralytics"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "commit_base": commit,
        "modulo_recepcion_sha256": sha256(Path(__file__)),
        "spatial_sha256": sha256(Path(__file__).with_name("spatial.py")),
        "versiones": packages,
        "opencv_threads": cv2.getNumThreads(),
        "cpu": platform.processor(),
    }


def register(sub):
    parser = sub.add_parser(
        "recibir-profundidad", help="validar entrega de un DepthEstimator sobre replay"
    )
    parser.add_argument("nombre", help="B0–B4 o nombre en depth.estimadores")
    parser.add_argument("--clip", default="recordings/dev-1114-distancias")
    parser.add_argument("--config")
    parser.add_argument("--pares", type=int, default=5, help="pares uniformes medidos (default 5)")
    parser.add_argument(
        "--contra", nargs="+", default=[], help="comparar sobre idénticos pares y máscaras"
    )
    parser.add_argument("--calentamiento", type=int, default=2)
    parser.add_argument(
        "--salida", help="carpeta de evidencia; default outputs/recepcion/FECHA-NOMBRE"
    )
    parser.set_defaults(fn=cmd_receive)
