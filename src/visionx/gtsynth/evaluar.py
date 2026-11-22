"""Distancia por objeto contra el GT-sintético-ELP (docs/INCERTIDUMBRE-Y-GT.md, sección 2.3).

Cada objeto se mide con su máscara exacta del render (no con el detector): así el error es
el de la profundidad y del ubicador, no el de la detección. Una fila por objeto.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np

from visionx.gtsynth.dataset import ObjetoGT, Vista, objetos_de, particion_de
from visionx.gtsynth.variantes import derecha_hermana, es_hermana, semilla_camara

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
    return _filas(m, vista, variante, frame, m.estimador.predict(frame), objs)


def _filas(m: Medidor, vista: Vista, variante: str, frame, depth, objs: list[ObjetoGT]):
    """Una fila por objeto, con la profundidad ya calculada."""
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


def medir(
    m: Medidor,
    vistas_,
    variante: str,
    maximo: int | None,
    min_px: int,
    progreso=None,
    alineador=None,
    vistas_por_camara: int = 1,
    distancia_conocida: bool | str = False,
):
    """Filas de todos los objetos de las vistas.

    Sin `alineador`, con una vista por cámara y sin distancia conocida, es la medición original:
    cada vista sola (y, en las hermanas, cada vista con su propia unidad). Si no, ver `_por_camara`.
    `distancia_conocida`: False, «corrimiento» (True) o «escala» (ver `MODOS_DISTANCIA`).
    """
    if alineador is None and vistas_por_camara == 1 and not distancia_conocida:
        filas: list[dict] = []
        for n, v in enumerate(islice(vistas_, maximo)):
            filas.extend(medir_vista(m, v, variante, n, min_px))
            if progreso:
                progreso(n + 1, v)
        return filas
    return _por_camara(
        m, list(islice(vistas_, maximo)), variante, min_px, progreso, alineador,
        vistas_por_camara, distancia_conocida,
    )  # fmt: skip


# Qué corrige un objeto a distancia conocida. Con un solo objeto hay una sola ecuación: se
# corrige una cosa. «corrimiento» resta un desplazamiento de columnas (yaw o centro óptico mal
# supuestos: el error que deja una unidad del mismo modelo). «escala» multiplica la distancia
# (focal × separación mal supuestas: el error de una cámara distinta o estimada, Z = f·B/d).
MODOS_DISTANCIA = ("corrimiento", "escala", "ambas")
CALIBRAN_AMBAS = 4  # vistas del principio del recorrido que miden el objeto (a varias distancias)
RAZON_MIN = 1.25  # «ambas» pide distancias distintas: la mayor ≥ 1,25 × la menor


def ajustar_ambas(medidas_d: list[float], reales_d: list[float]) -> tuple[float, float] | None:
    """Escala y corrimiento de la disparidad efectiva con un objeto a varias distancias.

    El objeto a distancias conocidas da pares (medida, real) de disparidad efectiva (con doffs).
    Si la cámara supuesta tiene mal `f·B` (escala) y la unidad un corrimiento de columnas, vale
    `medida = escala · real + corrimiento`: dos distancias distintas alcanzan para las dos cosas
    (con una sola, una u otra). Mínimos cuadrados; None si las distancias son casi iguales.
    """
    m, r = np.asarray(medidas_d, float), np.asarray(reales_d, float)
    ok = (m > 0) & (r > 0)
    m, r = m[ok], r[ok]
    if len(r) < 2 or r.max() < RAZON_MIN * r.min():
        return None
    escala, corrimiento = np.polyfit(r, m, 1)
    if escala <= 0:
        return None
    return float(escala), float(corrimiento)


def modo_distancia(valor: bool | str) -> str | None:
    """None (sin distancia conocida) o uno de `MODOS_DISTANCIA`; True es «corrimiento»."""
    if valor is False or valor is None or valor == "":
        return None
    modo = "corrimiento" if valor is True else str(valor)
    if modo not in MODOS_DISTANCIA:
        raise ValueError(f"distancia conocida: {modo!r} no es {' ni '.join(MODOS_DISTANCIA)}")
    return modo


def con_escala(depth, escala: float, fb: float):
    """La profundidad multiplicada por `escala` (distancia real / medida del objeto conocido).

    Equivale a corregir `f·B`: la disparidad efectiva se divide por `escala`, así disparidad y
    distancia siguen siendo coherentes con el mismo `fb`. La b de la U-Net (px de disparidad) se
    divide igual, para que el intervalo siga siendo el mismo en distancia relativa.
    """
    d_eff = (depth.disparity_px.astype(np.float32) + np.float32(depth.doffs)) / np.float32(escala)
    out = _con_disparidad(depth, d_eff - np.float32(depth.doffs), fb)
    if depth.uncertainty is not None:
        out = dataclasses.replace(out, uncertainty=depth.uncertainty / np.float32(escala))
    return out


def con_corrimiento(depth, delta_px: float, fb: float):
    """La profundidad después de restarle un corrimiento constante de disparidad (px).

    Es lo que hace una distancia conocida cuando el error es un corrimiento constante de columnas
    (centro óptico o yaw mal supuestos), que no se ve en la imagen y se mide con un objeto.
    """
    return _con_disparidad(depth, depth.disparity_px.astype(np.float32) - np.float32(delta_px), fb)


def _con_disparidad(depth, d: np.ndarray, fb: float):
    """La profundidad con la disparidad `d` corregida; «no medible» si no queda ningún píxel."""
    d = d.astype(np.float32)
    d_eff = d + depth.doffs
    ok = depth.valid_mask & np.isfinite(d_eff) & (d_eff > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ok, fb / d_eff, 0.0).astype(np.float32)
    if not ok.any():
        # Con un corrimiento grande (otra cámara supuesta ELP) puede no quedar ningún píxel con
        # disparidad positiva: el contrato pide marcarlo «no medible» con su motivo.
        from visionx.shared.contracts import DepthStatus

        return dataclasses.replace(
            depth, disparity_px=d, valid_mask=ok, depth_m=z, status=DepthStatus.UNMEASURABLE,
            reason="distancia conocida: ningún píxel queda con disparidad positiva",
        )  # fmt: skip
    return dataclasses.replace(depth, disparity_px=d, valid_mask=ok, depth_m=z)


def _por_camara(
    m: Medidor,
    lista: list[Vista],
    variante: str,
    min_px: int,
    progreso,
    alineador,
    vistas_por_camara: int,
    distancia_conocida: bool | str,
) -> list[dict]:
    """Las vistas de a grupos de `vistas_por_camara`, cada grupo visto por una misma cámara.

    Para cada grupo: (1) si es hermana, todas las vistas comparten la unidad (con 1 vista por
    cámara, la de siempre); (2) el alineador corrige todos los pares juntos, como lo haría la
    app acumulando cuadros; (3) con `distancia_conocida`, el objeto apuntado de la primera vista
    que lo tenga hace de «está a 2 metros»: según el modo, su error de disparidad se resta en
    todo el grupo («corrimiento») o su cociente de distancias multiplica todo el grupo
    («escala»); esa vista queda marcada `calibra` (no entra en el resumen).
    """
    modo_dc = modo_distancia(distancia_conocida)
    k_cam = max(int(vistas_por_camara), 1)
    dg = m.ubicador.depth_geometry
    fb = dg.fx * dg.baseline_m
    filas: list[dict] = []
    for g0 in range(0, len(lista), k_cam):
        grupo = lista[g0 : g0 + k_cam]
        semilla = semilla_camara(grupo[0], k_cam) if k_cam > 1 else None
        pares = [
            (v.izq(), derecha_hermana(v, variante, semilla) if es_hermana(variante) else v.der())
            for v in grupo
        ]
        if alineador is None:
            ders, modo, aplica = [p[1] for p in pares], "ninguna", ""
        else:
            ders, info = alineador.corregir(pares)
            modo, aplica = info.modo, ",".join(info.aplica)
        medidas = []
        for k, (v, (izq, _), der) in enumerate(zip(grupo, pares, ders, strict=True)):
            objs = objetos_de(v, min_px=min_px)
            frame = _frame(v, izq, der, m.geometria, g0 + k)
            medidas.append((v, frame, m.estimador.predict(frame) if objs else None, objs))
        delta, escala, calibra = 0.0, 1.0, None
        calibran: set[int] = set()
        if modo_dc == "ambas":  # el mismo objeto visto a varias distancias (la persona se mueve)
            ms, rs, ks = [], [], []
            for k, (v, frame, depth, objs) in enumerate(medidas):
                apuntado = [o for o in objs if o.apuntado]
                if depth is None or not apuntado or len(ks) >= CALIBRAN_AMBAS:
                    continue
                f0 = _filas(m, v, variante, frame, depth, apuntado)[0]
                if f0["d_frente_px"] is None:
                    continue
                ms.append(f0["d_frente_px"] + depth.doffs)
                rs.append(f0["d_gt_px"] + depth.doffs)
                ks.append(k)
            ajuste = ajustar_ambas(ms, rs)
            calibran = set(ks)
            if ajuste is not None:
                escala, delta = ajuste
            elif ms:  # todas a la misma distancia: solo el corrimiento, como con un objeto
                delta = float(np.median(np.subtract(ms, rs)))
        elif modo_dc:
            for k, (v, frame, depth, objs) in enumerate(medidas):
                apuntado = [o for o in objs if o.apuntado]
                if depth is None or not apuntado:
                    continue
                f0 = _filas(m, v, variante, frame, depth, apuntado)[0]
                if f0["d_frente_px"] is None:
                    continue
                if modo_dc == "corrimiento":
                    delta, calibra = float(f0["d_frente_px"] - f0["d_gt_px"]), k
                    break
                medida, real = f0["d_frente_px"] + depth.doffs, f0["d_gt_px"] + depth.doffs
                if medida > 0 and real > 0:  # Z = fb/d: real/medida en distancia = medida/real
                    escala, calibra = float(medida / real), k
                    break
        for k, (v, frame, depth, objs) in enumerate(medidas):
            if depth is None:
                continue
            if modo_dc == "ambas":  # d_real = (d − corrimiento) / escala
                if delta:
                    depth = con_corrimiento(depth, delta, fb)
                if escala != 1.0:
                    depth = con_escala(depth, escala, fb)
            elif delta:
                depth = con_corrimiento(depth, delta, fb)
            elif escala != 1.0:
                depth = con_escala(depth, escala, fb)
            for f in _filas(m, v, variante, frame, depth, objs):
                f.update({
                    "camara": g0 // k_cam, "alineacion": modo, "aplica": aplica,
                    "distancia_conocida": modo_dc or "", "delta_px": round(delta, 3),
                    "escala": round(escala, 4), "calibra": k == calibra or k in calibran,
                })  # fmt: skip
                filas.append(f)
        if progreso:
            progreso(min(g0 + k_cam, len(lista)), grupo[-1])
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
    # Las vistas que se usaron como «distancia conocida» no se evalúan (salen exactas).
    calibran = [f for f in filas if f.get("calibra")]
    filas = [f for f in filas if not f.get("calibra")]
    out: dict[str, Any] = {"objetos": len(filas)}
    if calibran:
        out["objetos_de_calibracion"] = len(calibran)
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
