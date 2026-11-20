"""Panel de audio (tecla «a»): teclas, estado, medidor de nivel y dibujo, sin audio real."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from visionx.app.audio_panel import (
    Dispositivo,
    MedidorNivel,
    PanelAudio,
    dibujar_panel,
    listar,
    posicion,
    tecla,
)

MIC = [Dispositivo(1, "AirPods Pro"), Dispositivo(3, "PRO X Headset"), Dispositivo(6, "MacBook")]
OUT = [Dispositivo(0, "Dell"), Dispositivo(3, "PRO X Headset")]


def test_teclas_de_flechas_en_mac_windows_y_linux_y_alternativas():
    for codigo in (63232, 2490368, 65362, ord("w")):
        assert tecla(codigo) == "arriba"
    for codigo in (63233, 2621440, 65364, ord("s")):
        assert tecla(codigo) == "abajo"
    assert [tecla(9), tecla(13), tecla(10), tecla(27)] == ["tab", "enter", "enter", "esc"]
    assert [tecla(ord("a")), tecla(32), tecla(-1)] == ["a", " ", None]


class _FakeSd:
    """Lo mínimo de sounddevice que usan `listar` y `MedidorNivel`."""

    def __init__(self, dispositivos, api_defecto=0):
        self.dispositivos, self.api = dispositivos, api_defecto
        self.callback = None

    def query_devices(self, kind=None):
        if kind == "input":
            return {"hostapi": self.api}
        return self.dispositivos

    def InputStream(self, channels, dtype, device, callback):  # noqa: N802 (API de sounddevice)
        if device == 99:
            raise RuntimeError("Error opening InputStream")
        self.callback = callback
        return _FakeStream()


class _FakeStream:
    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def test_listar_separa_entradas_y_salidas_de_la_api_por_defecto():
    sd = _FakeSd([
        {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
        {"name": "Auris", "max_input_channels": 1, "max_output_channels": 2, "hostapi": 0},
        {"name": "Auris", "max_input_channels": 1, "max_output_channels": 2, "hostapi": 1},
        {"name": "Parlante", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    ])  # fmt: skip
    entradas, salidas = listar(sd)
    assert [(d.indice, d.nombre) for d in entradas] == [(0, "Mic"), (1, "Auris")]
    assert [(d.indice, d.nombre) for d in salidas] == [(1, "Auris"), (3, "Parlante")]


def test_posicion_por_nombre_indice_o_defecto():
    assert posicion(MIC, "pro x", None) == 1
    assert posicion(MIC, 6, None) == 2
    assert posicion(MIC, None, 1) == 0
    assert posicion(MIC, "no existe", 6) == 2


def test_panel_navega_elige_y_cierra():
    panel = PanelAudio(MIC, OUT, mic=1, salida=0)
    panel.abrir()
    assert panel.bajo_cursor() == MIC[1]
    assert panel.manejar("abajo") == ("medir", MIC[2])
    assert panel.manejar("abajo") == ("medir", MIC[0])  # da la vuelta
    assert panel.manejar("enter") == ("mic", MIC[0])
    panel.usar("mic", MIC[0])
    assert panel.mic == 0
    assert panel.manejar("tab") is None and panel.columna == "salida"
    assert panel.manejar("abajo") is None  # en salidas no se mide nada
    assert panel.manejar("enter") == ("salida", OUT[1])
    panel.usar("salida", OUT[1])
    assert panel.salida == 1
    assert panel.manejar("q") is None  # q no cierra la app con el panel abierto
    assert panel.manejar("esc") == ("cerrar", None) and not panel.abierto


def test_panel_al_abrir_pone_el_cursor_en_lo_que_esta_en_uso():
    panel = PanelAudio(MIC, OUT, mic=2, salida=1)
    panel.abrir()
    assert panel.cursor == {"mic": 2, "salida": 1} and panel.columna == "mic"


def test_medidor_avisa_sin_senal_con_ceros_exactos_y_no_con_ruido():
    sd = _FakeSd([])
    m = MedidorNivel(sd)
    assert m.nivel() == (None, False)
    m.abrir(MIC[0])
    sd.callback(np.zeros((160, 1), np.float32), 160, None, None)
    m._abierto_en -= 1.0  # como si hubiera pasado un segundo
    m._ultimo_sonido -= 1.0
    db, sin_senal = m.nivel()
    assert sin_senal and db < -100
    ruido = np.random.default_rng(0).normal(0, 0.01, (160, 1)).astype(np.float32)
    sd.callback(ruido, 160, None, None)
    db, sin_senal = m.nivel()
    assert not sin_senal and -45 < db < -35  # 0,01 de desvío = -40 dBFS
    m.cerrar()
    assert m.nivel() == (None, False)


def test_medidor_propaga_el_error_si_el_dispositivo_no_abre():
    with pytest.raises(RuntimeError):
        MedidorNivel(_FakeSd([])).abrir(Dispositivo(99, "roto"))


def test_dibujar_panel_en_stand_y_diagnostico():
    pytest.importorskip("PIL")
    panel = PanelAudio(MIC, OUT, mic=1, salida=1)
    panel.abrir()
    panel.error = "PRO X no se puede usar"
    for size in ((1080, 1920), (900, 1600)):
        canvas = np.full((*size, 3), 200, np.uint8)
        out = dibujar_panel(canvas, panel, -30.0, True)
        assert out.shape == (*size, 3)
        assert out[5, 5, 0] < 200  # el fondo se oscurece


def test_la_salida_cambia_de_dispositivo_recien_al_empezar_la_frase():
    """set_device no cierra el stream que está sonando: el cambio se hace en `_open`."""
    pytest.importorskip("sounddevice")
    from visionx.voice.output import SoundDevicePlayer

    abiertos = []

    class Stream:
        def __init__(self, device, **_):
            self.device = device
            abiertos.append(device)

        def start(self):
            pass

        def write(self, data):
            return False

        def stop(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    player = SoundDevicePlayer(device=3)
    player._sd = type("Sd", (), {"OutputStream": staticmethod(Stream)})
    player.play(np.zeros(160, np.float32), 16000, threading.Event())
    player.set_device(0)
    assert player.device == 3 and abiertos == [3]  # todavía no cambió
    player.play(np.zeros(160, np.float32), 16000, threading.Event())
    assert player.device == 0 and abiertos == [3, 0]
    player.play(np.zeros(160, np.float32), 16000, threading.Event())
    assert abiertos == [3, 0]  # misma salida y frecuencia: no se reabre
