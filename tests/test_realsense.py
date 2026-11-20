"""Fuente RealSense con un `pyrealsense2` falso: sin cámara ni librería instalada."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from visionx.camera.realsense import RealSenseSource
from visionx.shared.errors import SourceError


class _Frame:
    def __init__(self, img):
        self._img = img

    def get_data(self):
        return self._img


class _Frames:
    def __init__(self, k):
        self.k = k

    def get_infrared_frame(self, ojo):
        img = np.full((72, 128), 40 * ojo + self.k % 10, np.uint8)
        return _Frame(img)


class _Pipeline:
    def __init__(self, falla=False):
        self.falla, self.k, self.parado, self.emisor = falla, 0, False, None
        self.pedidos = []

    def start(self, cfg):
        if self.falla:
            raise RuntimeError("No device connected")
        self.pedidos = cfg.pedidos
        pipeline = self

        class Sensor:
            def supports(self, opt):
                return True

            def set_option(self, opt, valor):
                pipeline.emisor = valor

        dev = NS(get_info=lambda q: {"name": "Intel RealSense D435I", "serie": "123"}[q],
                 first_depth_sensor=lambda: Sensor())  # fmt: skip
        intr = NS(width=128, height=72, fx=67.4, fy=67.4, ppx=63.2, ppy=36.1)
        stream = NS(as_video_stream_profile=lambda: NS(
            get_intrinsics=lambda: intr,
            get_extrinsics_to=lambda otro: NS(translation=[-0.0501, 0.0, 0.0]),
        ))  # fmt: skip
        return NS(get_device=lambda: dev, get_stream=lambda s, i: stream)

    def wait_for_frames(self, timeout_ms):
        self.k += 1
        return _Frames(self.k)

    def stop(self):
        self.parado = True


def _rs(pipeline):
    class Config:
        def __init__(self):
            self.pedidos = []

        def enable_stream(self, *args):
            self.pedidos.append(args)

    return NS(
        pipeline=lambda: pipeline, config=Config,
        stream=NS(infrared="ir"), format=NS(y8="y8"),
        camera_info=NS(name="name", serial_number="serie"),
        option=NS(emitter_enabled="emisor"),
    )  # fmt: skip


def test_lee_los_dos_ojos_rectificados_con_la_geometria_de_fabrica():
    p = _Pipeline()
    src = RealSenseSource(128, 72, 30, rs_module=_rs(p), calentamiento=2)
    assert [a[1] for a in p.pedidos] == [1, 2]  # las dos transmisiones infrarrojas
    assert p.emisor == 0.0  # proyector apagado por defecto
    f = src.read()
    assert f.rectified and f.source_id == "realsense" and f.left.shape == (72, 128, 3)
    assert f.geometry_id == src.geometry.geometry_id
    assert int(f.left[0, 0, 0]) != int(f.right[0, 0, 0])  # ojo 1 y ojo 2
    assert src.geometry.baseline_m == pytest.approx(0.0501)
    assert src.read().frame_id == f.frame_id + 1
    perfil = src.perfil()
    assert perfil.nivel == "precisa" and perfil.separacion_m == pytest.approx(0.0501)
    assert perfil.campo_horizontal_deg == pytest.approx(87.0, abs=0.5)
    src.close()
    assert p.parado


def test_sin_camara_es_source_error():
    with pytest.raises(SourceError, match="RealSense"):
        RealSenseSource(rs_module=_rs(_Pipeline(falla=True)))
