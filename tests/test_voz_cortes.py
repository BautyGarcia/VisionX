"""Voz entrecortada: búfer del parlante y conteo de cortes (prueba en vivo del 19/11)."""

from __future__ import annotations

import sys
import threading
import types

import numpy as np

from visionx.app.latency import summary
from visionx.voice.engines import VoiceSettings
from visionx.voice.output import SoundDevicePlayer


class _Stream:
    """OutputStream simulado: devuelve «se vació el búfer» en las escrituras pedidas."""

    abiertos: list[dict] = []

    def __init__(self, **kw):
        self.kw = kw
        self.writes = 0
        _Stream.abiertos.append(kw)

    def start(self):
        pass

    def write(self, data):
        self.writes += 1
        return self.writes in _Stream.vaciados

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


def _player(monkeypatch, vaciados, **kw):
    _Stream.abiertos, _Stream.vaciados = [], set(vaciados)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=_Stream))
    return SoundDevicePlayer(**kw)


def test_bufer_del_parlante_configurable(monkeypatch):
    p = _player(monkeypatch, [])
    assert p.play(np.zeros(16000, np.float32), 16000, threading.Event())
    assert _Stream.abiertos[-1]["latency"] == 0.2  # por defecto, no «low»
    p = _player(monkeypatch, [], latency_s=0.0)
    p.play(np.zeros(1600, np.float32), 16000, threading.Event())
    assert _Stream.abiertos[-1]["latency"] == "low"


def test_cuenta_los_cortes_sin_contar_el_arranque(monkeypatch):
    p = _player(monkeypatch, [1, 4, 7], block_ms=20.0)
    p.play(np.zeros(16000, np.float32), 16000, threading.Event())  # 50 bloques
    assert p.last_underflows == 2  # el vaciado del primer bloque es el arranque


def test_config_por_defecto_trae_el_bufer():
    from visionx.shared.config import load_config

    assert VoiceSettings.from_config(load_config()).output_latency_s == 0.2


def test_latencias_informa_frases_cortadas():
    rows = [
        {"tipo": "audio", "request_id": 1, "texto": "Está a 1,4 metros.", "started_ns": 1,
         "cortes": 3},
        {"tipo": "audio", "request_id": 1, "texto": "Listo.", "started_ns": 2, "cortes": 0},
    ]  # fmt: skip
    texto = summary(rows)
    assert "voz entrecortada: 1 de 2 frases con cortes (3 cortes en total)" in texto
    assert "«Está a 1,4 metros.»: 3 cortes" in texto
