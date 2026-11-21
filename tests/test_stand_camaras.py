"""Pestaña «Cámaras» del stand: lee la demo precalculada y dibuja sin correr modelos."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

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


def test_videos_pasa_de_cuadro_y_de_video(tmp_path):
    from visionx.app.stand_camaras import Videos

    # Una genérica (60°) que no va a la expo: no se reproduce.
    nombres = (
        "hotel_0-video-elp-full.mp4",
        "hotel_0-video-zed-2.mp4",
        "hotel_0-video-angosta-60.mp4",
    )
    for k, nombre in enumerate(nombres):
        w = cv2.VideoWriter(str(tmp_path / nombre), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
        for j in range(3):
            w.write(np.full((H, W, 3), 60 * k + 20 * j, np.uint8))
        w.release()
    v = Videos(tmp_path)
    assert [p.name for p in v.paths] == [
        "hotel_0-video-elp-full.mp4",
        "hotel_0-video-zed-2.mp4",
    ]
    a = v.cuadro(0.0)
    b = v.cuadro(0.25)  # tercer cuadro del primer video
    c = v.cuadro(0.45)  # ya en el segundo video
    assert a.shape == (H, W, 3) and int(b.mean()) > int(a.mean()) and int(c.mean()) >= 55
    img = render_camaras(tmp_path, ahora_s=0.0)
    assert img.shape == (H, W, 3)


def test_mosaico_junta_las_camaras_y_marca_las_que_faltan(tmp_path):
    from visionx.gtsynth.mosaico import CLASICO, armar

    lam = np.zeros((1080, 1920, 3), np.uint8)
    x, y, w, h = CLASICO
    lam[y : y + h, x : x + w] = (0, 200, 0)  # el panel «Clásico» en verde
    v = cv2.VideoWriter(str(tmp_path / "hotel_0-video-zed-2.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1080))  # fmt: skip
    for _ in range(3):
        v.write(lam)
    v.release()
    path = armar(tmp_path, "hotel_0")
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    ok, f = cap.read()
    assert ok and f.shape == (1080, 1920, 3)
    # ZED 2 es la cuarta celda (abajo a la izquierda): ahí aparece el verde del panel.
    celda = f[72 + (1080 - 72) // 2 :, : 1920 // 3]
    assert (celda[:, :, 1] > 150).mean() > 0.2
    assert armar(tmp_path, "otra_escena") is None


def test_distancia_conocida_se_controla_sobre_todos_los_cuadros():
    from types import SimpleNamespace

    from visionx.gtsynth.video import _delta_de_cuadros, _limite_delta

    d455 = SimpleNamespace(fx=343.2)  # focal del continuo de la D455
    # −3,7 px en todos los cuadros: un yaw real, aunque pase el tope viejo (3,4 px).
    assert _delta_de_cuadros([-3.7, -3.6, -3.85, None, -3.65], d455) == pytest.approx(-3.675)
    assert _delta_de_cuadros([-3.0, 1.0, 4.0, -2.0], d455) == 0.0  # salta: no sirve
    assert _delta_de_cuadros([None, None], d455) == 0.0
    assert _delta_de_cuadros([9.0, 9.1, 9.0], d455) == 0.0  # más que el tope de esa focal
    assert _limite_delta(SimpleNamespace(fx=50.0)) == 2.0  # mínimo
