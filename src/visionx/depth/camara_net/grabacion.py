"""De una grabación real (`camera grabar`) a recorridos con el formato de `gt-sintetico-elp`.

Para la parte real de D (PLAN-CIERRE): la geometría etiqueta video que nadie calibró y la red
se reentrena con esas etiquetas. `gtsynth autoetiquetas` y `camara_net entrenar` leen vistas
(`<raiz>/<variante>/<escena>/<indice>/` con `vista.json`, `izq.png` y `der.png`). Este módulo
corta la grabación en recorridos de `cuadros` cuadros, uno cada `paso`, y escribe cada par **tal
cual sale de la cámara** (sin rectificar ni sacar la lente: así lo ve una cámara desconocida),
achicado a `ancho` (la red ve 640 px y el movimiento trabaja a ≤ 960).

La cámara de `vista.json` sale de una calibración de tablero si se da (`--calibracion`). No se usa
para etiquetar: es solo el control, como la cámara real en la simulación. Sin calibración se
escribe una cámara de relleno (90°, 6 cm) para que el formato sea válido, con `control: false`:
ahí `fx_real` no significa nada.

    uv run python -m visionx.depth.camara_net desde-grabacion recordings/dev-1114-buscar-a \\
        --calibracion configs/calib/elp-2026-11-11-corregida.yaml
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from visionx.camera.replay import load_recording

VARIANTE = "elp-recorridos-real"


def camara_de_calibracion(ruta: str | Path, ancho: int, alto: int) -> dict[str, float]:
    """Focal, centro y separación del ojo izquierdo de una calibración, llevados a `ancho`×`alto`.

    Es la cámara cruda (antes de rectificar): `K` del ojo izquierdo, que es la que tiene que
    adivinar la red al ver la imagen sin tocar.
    """
    cal = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
    w0, h0 = cal["tamaño_imagen"]
    k = cal["izquierda"]["K"]
    sx, sy = ancho / w0, alto / h0
    fx = float(k[0][0]) * sx
    return {
        "fx": round(fx, 4),
        "fy": round(float(k[1][1]) * sy, 4),
        "cx": round((float(k[0][2]) + 0.5) * sx - 0.5, 4),
        "cy": round((float(k[1][2]) + 0.5) * sy - 0.5, 4),
        "base_m": round(float(cal["baseline_m"]), 5),
        "hfov_deg": round(math.degrees(2 * math.atan(ancho / (2 * fx))), 4),
    }


def _mapas_sin_lente(ruta, w0: int, h0: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mapas de cada ojo que sacan la distorsión y conservan su `K` (no rectifican)."""
    cal = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
    if tuple(cal["tamaño_imagen"]) != (w0, h0):
        raise ValueError(f"la calibración es de {cal['tamaño_imagen']}, la grabación de {w0}×{h0}")
    out = {}
    for ojo, lado in (("left", "izquierda"), ("right", "derecha")):
        k = np.asarray(cal[lado]["K"], np.float64)
        d = np.asarray(cal[lado]["D"], np.float64)
        out[ojo] = cv2.initUndistortRectifyMap(k, d, None, k, (w0, h0), cv2.CV_32FC1)
    return out


def ventanas(n: int, cuadros: int, paso: int) -> list[list[int]]:
    """Índices de cada recorrido: `cuadros` entradas una cada `paso`, sin solaparse."""
    largo = cuadros * paso
    return [list(range(i0, i0 + largo, paso)) for i0 in range(0, n - largo + 1, largo)]


def convertir(
    grabacion: str | Path,
    raiz: str | Path,
    variante: str = VARIANTE,
    cuadros: int = 60,
    paso: int = 2,
    ancho: int = 960,
    calibracion: str | Path | None = None,
    sin_lente: bool = False,
) -> dict[str, Any]:
    """Escribe los recorridos de una grabación. Devuelve cuántos y dónde.

    `sin_lente` (solo control, necesita `calibracion`): saca la distorsión de cada ojo con la
    calibración de tablero, con la misma `K`. Sirve para separar cuánto error de la etiqueta es
    la lente (`autoetiquetas` usa `focal_por_movimiento`, que supone pinhole) y cuánto lo
    demás.
    """
    grabacion = Path(grabacion)
    manifiesto, entradas, _ = load_recording(grabacion)
    if manifiesto.get("rectificado"):
        raise ValueError(f"{grabacion}: está rectificada; hace falta el par crudo")
    w0, h0 = (int(x) for x in manifiesto["ojo"])
    alto = int(round(h0 * ancho / w0))
    cam = (camara_de_calibracion(calibracion, ancho, alto) if calibracion
           else {"fx": ancho / 2, "fy": ancho / 2, "cx": (ancho - 1) / 2, "cy": (alto - 1) / 2,
                 "base_m": 0.06, "hfov_deg": 90.0})  # fmt: skip
    if sin_lente and not calibracion:
        raise ValueError("sin_lente necesita la calibración de tablero")
    mapas = _mapas_sin_lente(calibracion, w0, h0) if sin_lente else None
    escena = grabacion.name
    destino = Path(raiz) / variante / escena
    indice = 0
    recs = ventanas(len(entradas), cuadros, paso)
    for r, idx in enumerate(recs):
        for c, i in enumerate(idx):
            e = entradas[i]
            carpeta = destino / f"{indice:04d}"
            carpeta.mkdir(parents=True, exist_ok=True)
            for ojo, nombre in (("left", "izq.png"), ("right", "der.png")):
                img = cv2.imread(str(grabacion / e[ojo]), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(grabacion / e[ojo])
                if mapas is not None:
                    mx, my = mapas[ojo]
                    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
                img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(carpeta / nombre), img)
            meta = {
                "escena": escena, "variante": variante, "indice": indice,
                "camara": {"ancho": ancho, "alto": alto, **cam, "modelo": "real"},
                "fuente": "grabacion", "tipo_pose": "recorrido",
                "secuencia": {"recorrido": r, "cuadro": c},
                "grabacion": {"carpeta": grabacion.as_posix(), "frame_id": e["frame_id"],
                              "session_id": manifiesto.get("session_id")},
                "control": bool(calibracion), "sin_lente": bool(sin_lente),
                "calibracion_control": str(calibracion) if calibracion else None,
            }  # fmt: skip
            (carpeta / "vista.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                                encoding="utf-8")  # fmt: skip
            indice += 1
    return {"escena": escena, "recorridos": len(recs), "vistas": indice, "carpeta": str(destino)}
