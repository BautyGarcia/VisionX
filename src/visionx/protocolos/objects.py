"""M7: una muestra estadística por objeto/distancia/repetición, no por píxel/frame."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import number, percentile, read_csv, report, resolve


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["metodo"], r["referencia_z_m"]].append(r)
        groups[r["metodo"], "todas"].append(r)
    out = []
    for (method, distance), group in groups.items():
        errors = [r["z_pred_m"] - r["referencia_z_m"] for r in group if r["z_pred_m"] is not None]
        out.append(
            {
                "metodo": method,
                "distancia_z_m": distance,
                "n": len(group),
                "n_respuestas": len(errors),
                "mae_m": float(np.mean(np.abs(errors))) if errors else None,
                "sesgo_m": float(np.mean(errors)) if errors else None,
                "p95_error_absoluto_m": percentile(np.abs(errors), 95),
                "cobertura": len(errors) / len(group),
                "abstencion": 1 - len(errors) / len(group),
            }
        )
    return out


def evaluate(sheet: Path, out: Path, cfg: dict, methods, *, window_s=0.4, samples=3, backend=None):
    from visionx.detection.catalog import COCO_CLASSES
    from visionx.shared.contracts import DistancePolicy

    from .vision import OfflineVision

    annotations = read_csv(sheet)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("lista de métodos vacía o repetida")
    if any(m not in ("B0", "B1", "B2", "B3", "B4") for m in methods):
        raise ValueError("métodos válidos: B0 B1 B2 B3 B4")
    # Validar toda la planilla antes de cargar pesos/inferir.
    keys = set()
    for r in annotations:
        number(r["distancia_cinta_m"], "cinta", positive=True)
        number(r["t_s"], "t_s")
        if bool(r.get("instancia_x")) != bool(r.get("instancia_y")):
            raise ValueError("anotar ambas coordenadas de instancia o ninguna")
        if r.get("instancia_x"):
            if any(not 0 <= float(r[k]) <= 1 for k in ("instancia_x", "instancia_y")):
                raise ValueError("coordenadas normalizadas de instancia fuera de [0,1]")
        if not r["origen_cinta"].strip() or not np.isfinite(float(r["z_offset_m"])):
            raise ValueError("origen de cinta y corrección a Z explícitos requeridos")
        if float(r["z_offset_m"]) == 0 and r["origen_cinta"] != "vidrio_lente_izquierda":
            raise ValueError("otro origen de cinta requiere corrección a Z; no asumir offset 0")
        number(float(r["distancia_cinta_m"]) + float(r["z_offset_m"]), "Z", positive=True)
        if r["clase_coco"] not in COCO_CLASSES:
            raise ValueError(f"clase COCO desconocida: {r['clase_coco']}")
        key = (r.get("sesion", ""), r["objeto"], r["distancia_cinta_m"], r["repeticion"])
        if key in keys:
            raise ValueError(f"repetición duplicada: {key}")
        keys.add(key)
    simulated = backend is not None
    backend = backend or OfflineVision(cfg)
    rows = []
    try:
        for r in annotations:
            values = {m: [] for m in methods}
            detail = {m: [] for m in methods}
            frames = list(
                backend.frames(resolve(sheet, r["clip"]), float(r["t_s"]), window_s, samples)
            )
            if not frames:
                raise ValueError("ventana sin frames")
            for frame in frames:
                detection, predictions = backend.observe(frame, methods)
                matches = [
                    i for i, label in enumerate(detection.classes) if label == r["clase_coco"]
                ]
                if r.get("instancia_x"):
                    x = float(r["instancia_x"]) * frame.size[0]
                    y = float(r["instancia_y"]) * frame.size[1]
                    matches = [
                        i
                        for i in matches
                        if detection.boxes[i][0] <= x <= detection.boxes[i][2]
                        and detection.boxes[i][1] <= y <= detection.boxes[i][3]
                    ]
                for method in methods:
                    _, objects = predictions[method]
                    obj = next(
                        (
                            o
                            for o in objects
                            if len(matches) == 1 and o.detection_index == matches[0]
                        ),
                        None,
                    )
                    valid = obj is not None and obj.policy == DistancePolicy.SPEAK_METERS
                    z = obj.z_m if valid else None
                    if z is not None and np.isfinite(z) and z > 0:
                        values[method].append(z)
                    detail[method].append(
                        {
                            "frame_id": frame.frame_id,
                            "z_m": z,
                            "motivo": "ok"
                            if z is not None
                            else "deteccion_ausente_o_ambigua"
                            if len(matches) != 1
                            else "abstencion_espacial",
                        }
                    )
            for method in methods:
                rows.append(
                    {
                        **r,
                        "metodo": method,
                        "referencia_z_m": float(r["distancia_cinta_m"]) + float(r["z_offset_m"]),
                        "z_pred_m": float(np.median(values[method])) if values[method] else None,
                        "n_frames": len(frames),
                        "n_frames_con_respuesta": len(values[method]),
                        "frames": json.dumps(detail[method]),
                    }
                )
    finally:
        backend.close()
    return report(
        out,
        rows,
        {
            "simulado": simulated,
            "unidad_estadistica": "fila de planilla",
            "ventana_s": window_s,
            "frames_solicitados": samples,
            "agregacion": "mediana de respuestas válidas en la ventana",
            "tabla": summarize(rows),
        },
        [sheet, *backend.inputs],
        cfg,
    )
