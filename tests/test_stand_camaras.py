"""Pestaña «Cámaras» del stand: lee la demo precalculada y dibuja sin correr modelos."""

from __future__ import annotations

import json

import cv2
import numpy as np

from visionx.app.stand import H, W
from visionx.app.stand_camaras import cargar, render_camaras
from visionx.gtsynth.camaras import CAMARAS


def _demo(tmp_path, n=2):
    for e in range(n):
        for k, (var, nombre) in enumerate(CAMARAS):
            d = tmp_path / f"escena_{e}-0000" / var
            d.mkdir(parents=True)
            for img in ("izq", "real", "estimada", "error"):
                cv2.imwrite(str(d / f"{img}.jpg"), np.full((270, 480, 3), 40 * k, np.uint8))
            obj = {"clase": "lamp", "z_real_m": 2.0, "z_estimada_m": 2.1, "error_pct": 5.0}
            datos = {"nombre": nombre, "hfov_deg": 60 + 10 * k, "base_cm": 6, "ancho": 1280,
                     "alto": 720, "error_mediano_pct": 0.2 * k,
                     "objeto": None if k == 1 else obj}  # fmt: skip
            (d / "datos.json").write_text(json.dumps(datos), encoding="utf-8")
    cargar.cache_clear()
    return tmp_path


def test_dibuja_y_pasa_de_camara(tmp_path):
    carpeta = _demo(tmp_path)
    escenas = cargar(str(carpeta))
    assert len(escenas) == 2 and len(escenas[0]) == len(CAMARAS)
    a = render_camaras(carpeta, ahora_s=0.5)
    b = render_camaras(carpeta, ahora_s=6.5)  # la siguiente cámara
    assert a.shape == (H, W, 3) and not np.array_equal(a, b)


def test_sin_demo_avisa_como_generarla(tmp_path):
    cargar.cache_clear()
    img = render_camaras(tmp_path)
    assert img.shape == (H, W, 3)
