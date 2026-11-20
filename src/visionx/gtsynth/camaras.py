"""Demo «Cámaras» del stand: la misma escena vista por cámaras distintas, con CREStereo ya corrido.

Para el público (y el jurado): ¿VisionX funciona con otra cámara estéreo? Se toman unas pocas
vistas de validación renderizadas con la ELP y con los perfiles del issue #53 (mismas poses), y
para cada cámara se corre CREStereo **una vez, antes** (en la app no corre ningún modelo para
esto). Por cámara queda:

    izq.jpg        lo que ve el ojo izquierdo (recorte nominal)
    real.jpg       profundidad del render (la verdad)
    estimada.jpg   profundidad de CREStereo con la calibración de esa cámara
    error.jpg      error relativo por píxel (verde < 3 %, ámbar < 10 %, coral más)
    datos.json     cámara (campo, base), error mediano y distancia al objeto apuntado

    python -m visionx.gtsynth camaras data/gt-sintetico-elp --salida data/stand-camaras

La pantalla (`app/stand_camaras.py`) solo lee esas carpetas.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.gtsynth.dataset import Vista, vistas

# Cámaras de la demo, en el orden en que se muestran: (variante, nombre para el público).
CAMARAS: tuple[tuple[str, str], ...] = (
    ("elp-margen", "ELP de VisionX · 116° · 6 cm"),
    ("angosta-60", "Angular normal · 60° · 6 cm"),
    ("base-corta-3", "Ojos juntos · 80° · 3 cm"),
    ("media-90", "Separación ancha · 90° · 12 cm"),
    ("cuatro-tercios-90", "Formato 4:3 · 90° · 6 cm"),
    ("base-larga-20", "Muy separada · 100° · 20 cm"),
    ("gran-angular-130", "Gran angular · 130° · 6 cm"),
)
ANCHO = 960  # ancho de las imágenes guardadas (la pantalla las escala)
Z_MIN, Z_MAX = 0.3, 6.0


def _color_z(z: np.ndarray, valid: np.ndarray, lo: float = Z_MIN, hi: float = Z_MAX):
    """Profundidad en color (rango lo–hi, escala log): cerca cálido, lejos frío; sin dato, gris."""
    t = np.clip((np.log(np.clip(z, lo, hi)) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1)
    img = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (40, 34, 30)
    return img


def _color_error(rel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    img = np.full(rel.shape + (3,), (40, 34, 30), np.uint8)
    a = np.abs(rel)
    img[valid & (a < 0.03)] = (140, 200, 90)
    img[valid & (a >= 0.03) & (a < 0.10)] = (60, 180, 235)
    img[valid & (a >= 0.10)] = (110, 105, 240)
    return img


def _chico(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (ANCHO, round(h * ANCHO / w)), interpolation=cv2.INTER_AREA)


def elegir_poses(raiz: str | Path, n: int = 4) -> list[tuple[str, int]]:
    """Vistas de val lindas para mostrar: sin huecos, con varios objetos, una por escena."""
    out, escenas = [], set()
    for v in vistas(raiz, "elp", "val"):
        if v.escena in escenas:
            continue
        z = v.z()
        if float((~(np.isfinite(z) & (z > 0))).mean()) > 0.01 or len(v.meta["instancias"]) < 10:
            continue
        if all((Path(raiz) / var / v.escena / f"{v.meta['indice']:04d}").is_dir()
               for var, _ in CAMARAS):  # fmt: skip
            out.append((v.escena, int(v.meta["indice"])))
            escenas.add(v.escena)
        if len(out) >= n:
            break
    return out


def preparar(
    raiz: str | Path,
    salida: str | Path,
    poses: list[tuple[str, int]] | None = None,
    profundidad: str = "crestereo-preciso",
    cfg: dict[str, Any] | None = None,
) -> list[dict]:
    from visionx.gtsynth.evaluar import Medidor, _frame
    from visionx.shared.config import load_config

    cfg = cfg or load_config()
    poses = poses or elegir_poses(raiz)
    resumen = []
    medidores: dict[str, Any] = {}
    for escena, indice in poses:
        for var, nombre in CAMARAS:
            v = Vista(Path(raiz) / var / escena / f"{indice:04d}")
            g = v.geometria_nominal
            # Múltiplos de 8 para la red: se recorta el borde derecho/inferior (no mueve cx, cy).
            w8, h8 = g.width // 8 * 8, g.height // 8 * 8
            izq = v.recortar(v.izq())[:h8, :w8]
            der = v.recortar(v.der())[:h8, :w8]
            z = v.recortar(v.z())[:h8, :w8]
            inst = v.recortar(v.instancias())[:h8, :w8]
            from dataclasses import replace

            g = replace(g, width=w8, height=h8)
            clave = f"{w8}x{h8}-{g.fx:.3f}-{g.baseline_m:.4f}"
            if clave not in medidores:
                medidores[clave] = Medidor.armar(profundidad, cfg, g)
            m = medidores[clave]
            t0 = time.perf_counter()
            depth = m.estimador.predict(_frame(v, izq, der, g, indice))
            ms = (time.perf_counter() - t0) * 1000
            # La disparidad viene en la geometría de trabajo de la red (el preciso usa otra
            # resolución): Z ahí y después a la del recorte.
            dg = getattr(m.estimador, "geometry", g)
            disp = depth.disparity_px + depth.doffs
            est = np.where(depth.valid_mask & (disp > 0),
                           dg.fx * dg.baseline_m / np.maximum(disp, 1e-6), np.nan)  # fmt: skip
            if est.shape != (h8, w8):
                est = cv2.resize(est.astype(np.float32), (w8, h8), interpolation=cv2.INTER_NEAREST)
            ok_gt = np.isfinite(z) & (z >= Z_MIN) & (z <= Z_MAX)
            ok = ok_gt & np.isfinite(est)
            rel = np.where(ok, (est - z) / np.where(ok_gt, z, 1), 0.0)
            apuntado = int(v.meta["apuntado"]["id"])
            mask = (inst == apuntado) & ok
            obj = None
            if mask.sum() > 50:
                z_gt = float(np.percentile(z[mask], 20))
                z_es = float(np.percentile(est[mask], 20))
                obj = {"clase": v.meta["apuntado"]["clase"], "z_real_m": round(z_gt, 3),
                       "z_estimada_m": round(z_es, 3),
                       "error_pct": round(100 * (z_es - z_gt) / z_gt, 2)}  # fmt: skip
            datos = {
                "escena": escena, "indice": indice, "variante": var, "nombre": nombre,
                "hfov_deg": round(math.degrees(2 * math.atan(g.width / (2 * g.fx))), 1),
                "base_cm": round(100 * g.baseline_m, 1), "ancho": w8, "alto": h8,
                "error_mediano_pct": round(100 * float(np.median(np.abs(rel[ok]))), 2)
                if ok.any() else None,
                "cobertura_pct": round(100 * float(ok.sum() / max(ok_gt.sum(), 1)), 1),
                "objeto": obj, "modelo": depth.model_id, "ms": round(ms),
            }  # fmt: skip
            d = Path(salida) / f"{escena}-{indice:04d}" / var
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "izq.jpg"), _chico(izq), [cv2.IMWRITE_JPEG_QUALITY, 88])
            # Mismo rango de colores para la real y la estimada, ajustado a esta vista.
            lo, hi = (float(x) for x in np.percentile(z[ok_gt], (2, 98)))
            lo, hi = max(lo, Z_MIN), max(hi, lo * 1.5)
            cv2.imwrite(str(d / "real.jpg"), _chico(_color_z(np.nan_to_num(z), ok_gt, lo, hi)))
            cv2.imwrite(str(d / "estimada.jpg"),
                        _chico(_color_z(np.nan_to_num(est), np.isfinite(est), lo, hi)))  # fmt: skip
            cv2.imwrite(str(d / "error.jpg"), _chico(_color_error(rel, ok)))
            (d / "datos.json").write_text(json.dumps(datos, ensure_ascii=False, indent=1),
                                          encoding="utf-8")  # fmt: skip
            resumen.append(datos)
            print(f"{escena}/{indice:04d} {var}: error {datos['error_mediano_pct']} %, "
                  f"objeto {obj and obj['error_pct']} %, {ms:.0f} ms", flush=True)  # fmt: skip
    (Path(salida) / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                                               encoding="utf-8")  # fmt: skip
    return resumen
