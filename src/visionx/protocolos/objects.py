"""M7: una muestra estadística por objeto/distancia/repetición, no por píxel/frame."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import number, percentile, read_csv, report, resolve


def object_uncertainty(depth, detection, index):
    """Mediana sobre la máscara válida del objeto; menor significa mayor confianza.

    Ambos tipos actuales del contrato se expresan en píxeles de disparidad. No se
    interpretan como probabilidades ni como intervalos calibrados de Z.
    """
    from visionx.shared.geometry import resize_mask

    if depth.uncertainty is None or detection.masks is None:
        return None
    mask = resize_mask(detection.masks[index], *depth.size) & depth.valid_mask
    values = depth.uncertainty[mask]
    return float(np.median(values)) if values.size else None


def risk_coverage(group):
    scored = [
        r for r in group if r["z_pred_m"] is not None and r.get("incertidumbre_px") is not None
    ]
    kinds = {r["tipo_incertidumbre"] for r in scored}
    result = {"n_con_incertidumbre": len(scored), "tipo_incertidumbre": ",".join(sorted(kinds))}
    # No comparar escalas semánticamente distintas en una misma curva.
    ranked = sorted(scored, key=lambda r: r["incertidumbre_px"]) if len(kinds) == 1 else []
    for percent in (100, 90, 75, 50):
        chosen = ranked[: int(np.ceil(len(ranked) * percent / 100))]
        result[f"cobertura_confianza_{percent}"] = len(chosen) / len(group) if chosen else None
        result[f"mae_confianza_{percent}_m"] = (
            float(np.mean([abs(r["z_pred_m"] - r["referencia_z_m"]) for r in chosen]))
            if chosen
            else None
        )
    return result


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
                **risk_coverage(group),
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
            uncertainties = {m: [] for m in methods}
            kinds = {m: set() for m in methods}
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
                    depth, objects = predictions[method]
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
                    uncertainty = None
                    if z is not None and np.isfinite(z) and z > 0:
                        values[method].append(z)
                        uncertainty = object_uncertainty(depth, detection, matches[0])
                        if uncertainty is not None:
                            uncertainties[method].append(uncertainty)
                            kinds[method].add(depth.uncertainty_kind.value)
                    detail[method].append(
                        {
                            "frame_id": frame.frame_id,
                            "z_m": z,
                            "incertidumbre_px": uncertainty,
                            "tipo_incertidumbre": depth.uncertainty_kind.value,
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
                        "incertidumbre_px": float(np.median(uncertainties[method]))
                        if uncertainties[method] and len(kinds[method]) == 1
                        else None,
                        "tipo_incertidumbre": ",".join(sorted(kinds[method])),
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
            "riesgo_cobertura": (
                "mediana de incertidumbre en máscara válida, luego mediana entre frames; "
                "menor es más confiable; retener ceil(n*100/90/75/50%); empates por orden "
                "de planilla; cobertura sobre todas las filas; sin incertidumbre queda vacío; "
                "tipos distintos no se comparan; no es confianza probabilística calibrada"
            ),
            "tabla": summarize(rows),
        },
        [sheet, *backend.inputs],
        cfg,
    )
