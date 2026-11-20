"""Distancia por objeto contra el GT-sintético-ELP (docs/INCERTIDUMBRE-Y-GT.md, sección 2.3).

Cada objeto se mide con su máscara exacta del render (no con el detector): así el error es
el de la profundidad y del ubicador, no el de la detección. Una fila por objeto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np

from visionx.gtsynth.dataset import ObjetoGT, Vista, objetos_de, particion_de
from visionx.gtsynth.variantes import derecha_hermana, es_hermana

LN10 = math.log(10.0)  # Laplace: P(|e| > b·ln 10) = 0,1 -> intervalo del 90 %


@dataclass(frozen=True)
class Medidor:
    """Estimador + ubicador armados para una geometría supuesta (la de la ELP)."""

    estimador: Any
    ubicador: Any
    geometria: Any  # la que el sistema cree tener (captura)

    @classmethod
    def armar(cls, nombre: str, cfg: dict, geometria) -> Medidor:
        from visionx.app.depth_registry import build_estimator
        from visionx.depth.spatial import build_locator

        est = build_estimator(nombre, cfg, geometria)
        return cls(est, build_locator(cfg, geometria, getattr(est, "geometry", geometria)),
                   geometria)  # fmt: skip


def _frame(vista: Vista, izq: np.ndarray, der: np.ndarray, geometria, n: int):
    from visionx.shared.contracts import StereoFrame

    return StereoFrame(
        session_id="gtsynth", frame_id=n, left=izq, right=der, received_ns=0,
        rectified=True, calibration_id="gtsynth-elp", geometry_id=geometria.geometry_id,
        source_id="gtsynth",
    )  # fmt: skip


def _detecciones(frame, vista: Vista, objs: list[ObjetoGT], geometria):
    from visionx.shared.contracts import Detection

    inst = vista.instancias()
    return Detection(
        session_id=frame.session_id, frame_id=frame.frame_id, model_id="gt-mascaras",
        geometry_id=geometria.geometry_id, calibration_id=frame.calibration_id,
        image_size=geometria.size,
        boxes=np.array([o.caja for o in objs], dtype=np.float32).reshape(-1, 4),
        classes=tuple(o.coco or o.clase for o in objs),
        scores=np.ones(len(objs), dtype=np.float32), inference_ms=0.0,
        masks=np.stack([inst == o.id for o in objs]) if objs else None,
    )  # fmt: skip


def intervalo(d_frente_px: float | None, b_px: float | None, fb: float, doffs: float, k: float):
    """(z_bajo, z_alto) del 90 %; z_alto = inf si el intervalo cruza disparidad 0."""
    from visionx.shared.uncertainty import laplace_interval

    if d_frente_px is None or b_px is None:
        return None, None
    bajo, alto = laplace_interval(d_frente_px, b_px, fb, doffs, k)
    return bajo, math.inf if alto is None else alto


def medir_vista(m: Medidor, vista: Vista, variante: str, n: int, min_px: int) -> list[dict]:
    objs = objetos_de(vista, min_px=min_px)
    if not objs:
        return []
    der = derecha_hermana(vista, variante) if es_hermana(variante) else vista.der()
    frame = _frame(vista, vista.izq(), der, m.geometria, n)
    depth = m.estimador.predict(frame)
    dets = _detecciones(frame, vista, objs, m.geometria)
    dg = m.ubicador.depth_geometry
    fb = dg.fx * dg.baseline_m
    filas = []
    for o, (obj, diag) in zip(objs, m.ubicador.locate_detailed(frame, depth, dets), strict=True):
        z = obj.z_m
        bajo, alto = intervalo(diag.d_front_px, diag.b_median_px, fb, depth.doffs, 1.0)
        filas.append({
            "vista": vista.clave, "variante": variante, "particion": particion_de(vista.escena),
            "clase": o.clase, "coco": o.coco or "", "px": o.px, "cortado": o.cortado,
            "apuntado": o.apuntado, "z_gt_m": round(o.z_frente_m, 4),
            "z_gt_mediana_m": round(o.z_mediana_m, 4),
            "z_m": None if z is None else round(z, 4), "politica": obj.policy.value,
            "motivo": diag.reason, "fraccion_valida": round(obj.valid_fraction, 3),
            "error_m": None if z is None else round(z - o.z_frente_m, 4),
            "error_rel": None if z is None else round((z - o.z_frente_m) / o.z_frente_m, 4),
            "d_frente_px": diag.d_front_px, "b_px": diag.b_median_px,
            "d_gt_px": fb / o.z_frente_m - depth.doffs,
            "z_bajo_k1": bajo, "z_alto_k1": alto,
            "modelo": depth.model_id, "ms": round(depth.inference_ms, 1),
        })  # fmt: skip
    return filas


def medir(m: Medidor, vistas_, variante: str, maximo: int | None, min_px: int, progreso=None):
    filas: list[dict] = []
    for n, v in enumerate(islice(vistas_, maximo)):
        filas.extend(medir_vista(m, v, variante, n, min_px))
        if progreso:
            progreso(n + 1, v)
    return filas


def _rango(z: float) -> str:
    return "0,5–1" if z < 1 else "1–2" if z < 2 else "2–3" if z < 3 else "3–4" if z < 4 else "4+"


def factor_b(filas: list[dict], cobertura: float = 0.9) -> float | None:
    """k tal que el intervalo ±b·ln10·k contenga la distancia real en `cobertura` de los casos.

    Ajustarlo SOLO con val. Con k = 1 la b se usa tal cual sale de la red.
    """
    ks = [abs(f["d_gt_px"] - f["d_frente_px"]) / (LN10 * f["b_px"])
          for f in filas if f["b_px"] and f["d_frente_px"] is not None]  # fmt: skip
    return float(np.quantile(ks, cobertura)) if ks else None


def resumen(filas: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"objetos": len(filas)}
    pol: dict[str, int] = {}
    for f in filas:
        pol[f["politica"]] = pol.get(f["politica"], 0) + 1
    out["politicas"] = pol
    medidas = [f for f in filas if f["error_rel"] is not None]
    por_rango = {}
    for r in ("0,5–1", "1–2", "2–3", "3–4", "4+"):
        e = np.array([f["error_rel"] for f in medidas if _rango(f["z_gt_m"]) == r])
        em = np.array([f["error_m"] for f in medidas if _rango(f["z_gt_m"]) == r])
        if len(e):
            por_rango[r] = {
                "n": len(e), "mae_m": round(float(np.mean(np.abs(em))), 3),
                "sesgo_pct": round(100 * float(np.median(e)), 2),
                "mae_pct": round(100 * float(np.mean(np.abs(e))), 2),
                "p95_pct": round(100 * float(np.quantile(np.abs(e), 0.95)), 2),
            }  # fmt: skip
    out["por_rango"] = por_rango
    con_b = [f for f in filas if f["z_bajo_k1"] is not None]
    if con_b:
        dentro = [f["z_bajo_k1"] <= f["z_gt_m"] <= f["z_alto_k1"] for f in con_b]
        out["cobertura_90_k1"] = round(float(np.mean(dentro)), 3)
        out["factor_b_para_90"] = factor_b(con_b)
    return out
