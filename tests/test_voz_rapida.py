"""Voz más rápida: Whisper en la GPU de Apple (MLX) y la respuesta de Claude por tramos."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from visionx.assistant.core import split_sentences
from visionx.shared.errors import EstimatorError
from visionx.voice.engines import (
    MlxWhisperTranscriber,
    VoiceSettings,
    build_transcriber,
    mlx_available,
    mlx_repo,
)

LARGA = "La taza blanca está sobre el taburete de madera, a 1,4 metros, unos 20 cm a tu izquierda."


def test_sin_tramos_corta_solo_en_punto():
    ready, rest = split_sentences(LARGA + " Hay")
    assert ready == [LARGA] and rest == " Hay"


def test_con_tramos_arranca_con_el_primero():
    ready, rest = split_sentences(LARGA, clause_chars=30)
    assert ready[0] == "La taza blanca está sobre el taburete de madera,"
    # «1,4» no se corta (no hay espacio después de la coma) ni un tramo corto («a 1,4 metros,»).
    # El punto del final espera lo que sigue (como siempre): ese tramo sale con final=True.
    assert ready == ["La taza blanca está sobre el taburete de madera,"]
    assert rest.strip() == "a 1,4 metros, unos 20 cm a tu izquierda."
    ready, rest = split_sentences(LARGA, clause_chars=30, final=True)
    assert " ".join(ready) == LARGA and rest == ""


def test_en_streaming_espera_el_espacio_despues_de_la_coma():
    ready, rest = split_sentences(
        "La taza blanca está sobre el taburete de madera,", clause_chars=30
    )
    assert ready == []  # puede ser «1,4»: hasta que no llega el espacio no se sabe
    ready, rest = split_sentences(rest + " a", clause_chars=30)
    assert ready == ["La taza blanca está sobre el taburete de madera,"] and rest == " a"


def test_no_corta_dentro_de_una_cita():
    texto = "Dice «Ibuprofeno 400 mg, tomar cada 8 horas, con comida» y abajo hay una fecha."
    ready, _ = split_sentences(texto, clause_chars=10, final=True)
    assert ready == [texto]


def test_mlx_repo():
    assert mlx_repo("small") == "mlx-community/whisper-small-mlx"
    assert mlx_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_mlx_transcriber_mismas_opciones_y_precalienta():
    calls = []

    def fake(audio, **kw):
        calls.append((len(audio), kw))
        return {"text": " Buscá la taza. ", "language": "es"}

    s = VoiceSettings(stt_initial_prompt="Buscá la taza.")
    t = MlxWhisperTranscriber(s, transcribe=fake)
    assert len(calls) == 1  # precalentado al armarlo
    out = t.transcribe(np.zeros(16000, np.float32))
    assert out.text == "Buscá la taza." and out.audio_s == 1.0
    kw = calls[-1][1]
    assert kw["language"] == "es" and kw["condition_on_previous_text"] is False
    assert kw["initial_prompt"] == "Buscá la taza." and kw["path_or_hf_repo"].endswith("small")


def test_backend_invalido_o_mlx_sin_modelo(tmp_path):
    with pytest.raises(EstimatorError, match="auto, mlx o faster-whisper"):
        build_transcriber(VoiceSettings(stt_backend="gpu"))
    s = dataclasses.replace(VoiceSettings(), stt_backend="mlx", stt_mlx_models_dir=str(tmp_path))
    with pytest.raises(EstimatorError, match="--whisper-mlx small"):
        build_transcriber(s)


def test_fuera_de_una_mac_no_hay_mlx(monkeypatch):
    import platform
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    assert not mlx_available()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert not mlx_available()


def test_config_por_defecto():
    from visionx.assistant.core import AssistantSettings
    from visionx.shared.config import load_config

    cfg = load_config()
    assert VoiceSettings.from_config(cfg).stt_backend == "auto"
    assert AssistantSettings.from_config(cfg).clause_chars == 30
