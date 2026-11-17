"""Tests de voz (T08) sin micrófono, parlantes, modelos ni internet.

Las reglas se prueban con frases escritas; la salida de audio, con un reproductor falso que
tarda lo que duraría el audio. Esto no mide latencia audible ni reconocimiento de voz real:
eso queda para las pruebas con dispositivos (BITACORA).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import AudioEvent, Mode, Priority
from visionx.shared.errors import EstimatorError, InvalidInput
from visionx.shared.interfaces import AudioSink
from visionx.voice.__main__ import main as voice_main
from visionx.voice.__main__ import outcome, read_phrases, read_wav, write_wav
from visionx.voice.commands import (
    CANCEL,
    CommandParser,
    acknowledgement,
    clean,
    find_object,
)
from visionx.voice.engines import PiperSynth, VoiceSettings, WhisperTranscriber, resample
from visionx.voice.output import (
    SOUNDS,
    AudioOutput,
    CachedSynth,
    FakePlayer,
    Outcome,
    render_sound,
    tone,
)

PHRASES = ROOT / "tests" / "fixtures" / "voz-frases.csv"
CATALOG = ClassCatalog(COCO_CLASSES)


# --- reglas -----------------------------------------------------------------------------


def test_reference_phrases_are_interpreted_as_expected():
    parser = CommandParser(CATALOG)
    rows = list(read_phrases(PHRASES))
    assert len(rows) >= 50
    wrong = [
        (r["frase"], outcome(parser.parse(r["frase"])))
        for r in rows
        if outcome(parser.parse(r["frase"])) != (r["modo"], r["objetivo"])
    ]
    assert wrong == []


def test_clean_removes_punctuation_accents_and_fillers():
    assert clean("¿Che, buscá la taza, por favor?") == "busca la taza"
    assert clean("VisionX") == ""


def test_longest_object_name_wins():
    assert find_object("busca el control remoto", CATALOG) == "remote"
    assert find_object("busca algo", CATALOG) is None


def test_unknown_object_asks_instead_of_guessing():
    interp = CommandParser(CATALOG).parse("Buscá las llaves.")
    assert interp.mode is Mode.SEARCH and not interp.ok
    assert "llaves" in interp.reply
    with pytest.raises(ValueError, match="aclaración"):
        interp.to_command(1, 0)


@pytest.mark.parametrize(
    "phrase",
    [
        "¿A qué distancia está eso?",
        "¿A qué distancia está esa?",
        "¿A qué distancia está?",
        "¿Qué tan lejos está ese objeto?",
    ],
)
def test_distance_pronouns_are_implicit_references(phrase):
    interp = CommandParser(CATALOG).parse(phrase)
    assert interp.ok and interp.mode is Mode.DISTANCE and interp.target is None


def test_cancel_wins_even_with_more_words():
    interp = CommandParser(CATALOG).parse("Pará, pará, buscá la taza")
    assert (interp.mode, interp.target) == (Mode.CONTROL, CANCEL)


def test_interpretation_to_command():
    interp = CommandParser(CATALOG).parse("¿A qué distancia está la taza?")
    cmd = interp.to_command(request_id=7, issued_ns=100, deadline_ns=200)
    assert (cmd.mode, cmd.target, cmd.request_id, cmd.deadline_ns) == (
        Mode.DISTANCE,
        "cup",
        7,
        200,
    )
    assert cmd.utterance == "¿A qué distancia está la taza?"


def test_acknowledgements_use_spanish_names():
    parser = CommandParser(CATALOG)
    assert acknowledgement(parser.parse("buscá la taza"), CATALOG) == "Busco taza."
    assert acknowledgement(parser.parse("distancia"), CATALOG) == "Mido la distancia."
    assert acknowledgement(parser.parse("cancelar"), CATALOG) is None
    assert "No entendí" in acknowledgement(parser.parse("qué hora es"), CATALOG)


def test_cli_orden_and_frases(capsys):
    assert voice_main(["orden", "buscá la botella"]) == 0
    assert "search bottle" in capsys.readouterr().out
    assert voice_main(["orden", "hacé algo"]) == 1
    assert voice_main(["frases", str(PHRASES)]) == 0


# --- sonidos y remuestreo ---------------------------------------------------------------


def test_tones_are_bounded_and_start_and_end_silent():
    for name in SOUNDS:
        samples = render_sound(name, 16_000)
        assert samples.dtype == np.float32 and np.abs(samples).max() <= 0.41
        assert abs(samples[0]) < 1e-3 and abs(samples[-1]) < 0.05  # rampa: sin clic
    assert len(tone([440, 880], 100, 1000, gap_ms=50)) == 100 + 50 + 100
    with pytest.raises(InvalidInput):
        render_sound("trompeta", 16_000)


def test_resample_keeps_tone_and_removes_aliasing():
    rate_in, rate_out = 22_050, 16_000
    t = np.arange(rate_in) / rate_in
    voice = np.sin(2 * np.pi * 440 * t)
    high = np.sin(2 * np.pi * 10_000 * t)  # por encima del Nyquist de 16 kHz
    out = resample(voice + high, rate_in, rate_out)
    assert len(out) == rate_out
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / rate_out)
    assert abs(freqs[spectrum.argmax()] - 440) < 2
    # Sin filtrar, 10 kHz se doblaría a 6 kHz con amplitud comparable a la voz.
    assert spectrum[np.abs(freqs - 6000) < 50].max() < 0.01 * spectrum.max()
    assert resample(np.zeros(0), rate_in, rate_out).size == 0


# --- salida de audio --------------------------------------------------------------------


class FakeSynth:
    """10 ms de audio por carácter, a 1 kHz: duraciones previsibles."""

    sample_rate = 1000

    def __init__(self, delay_s: float = 0.0, fail_on: str | None = None):
        self.delay_s = delay_s
        self.fail_on = fail_on
        self.calls: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if text == self.fail_on:
            raise RuntimeError("no se pudo")
        time.sleep(self.delay_s)
        return np.full(10 * len(text), 0.1, np.float32)


def event(rid: int, text: str, priority=Priority.SHORT, *, created=0, deadline=None, **kw):
    return AudioEvent(rid, priority, text, created_ns=created, deadline_ns=deadline, **kw)


def outcomes(output: AudioOutput) -> dict[str, Outcome]:
    return {r.text_or_sound: r.outcome for r in output.records}


def test_output_plays_in_priority_then_arrival_order():
    player = FakePlayer()
    blocker = threading.Event()

    class SlowFirst(FakeSynth):
        def synthesize(self, text):
            if text == "primero":
                blocker.wait(2)  # mientras sintetiza el primero, llegan los demás
            return super().synthesize(text)

    synth = SlowFirst()
    output = AudioOutput(synth, player)
    assert isinstance(output, AudioSink)
    output.submit(event(1, "primero"))
    time.sleep(0.05)
    output.submit(event(2, "largo", Priority.LONG))
    output.submit(event(3, "corto-a"))
    output.submit(event(4, "corto-b"))
    blocker.set()
    assert output.wait_idle(5)
    output.close()
    # Orden de síntesis = orden de reproducción. No se ordena por started_ns: en Windows con
    # Python 3.11 el reloj monotónico avanza de a ~15,6 ms y empata eventos seguidos.
    assert synth.calls == ["primero", "corto-a", "corto-b", "largo"]
    assert set(outcomes(output).values()) == {Outcome.PLAYED}
    assert player.closed


def test_cancel_stops_current_and_drops_pending_of_that_request():
    player = FakePlayer(realtime=True, block_ms=5)
    output = AudioOutput(FakeSynth(), player)
    output.submit(event(1, "x" * 300))  # 3 s de audio
    output.submit(event(1, "pendiente"))
    output.submit(event(2, "otro pedido"))
    assert player.started.wait(2)
    t0 = time.perf_counter()
    output.cancel(1)
    assert output.wait_idle(5)
    output.close()
    result = outcomes(output)
    assert result["x" * 300] is Outcome.CANCELLED
    assert result["pendiente"] is Outcome.CANCELLED
    assert result["otro pedido"] is Outcome.PLAYED
    assert time.perf_counter() - t0 < 1.0  # no esperó los 3 s
    assert player.played[0][2] is False


def test_higher_priority_preempts_cancelable_but_not_the_reverse():
    player = FakePlayer(realtime=True, block_ms=5)
    output = AudioOutput(FakeSynth(), player)
    output.submit(event(1, "d" * 300, Priority.LONG))
    assert player.started.wait(2)
    output.submit(event(2, "alerta", Priority.ALERT, cancelable=False))
    assert output.wait_idle(5)
    result = outcomes(output)
    assert result["d" * 300] is Outcome.PREEMPTED
    assert result["alerta"] is Outcome.PLAYED

    player.started.clear()
    output.submit(event(3, "a" * 100, Priority.ALERT, cancelable=False))
    assert player.started.wait(2)
    output.cancel_all()  # una alerta no cancelable sigue sonando
    output.submit(event(4, "respuesta", Priority.SHORT))
    assert output.wait_idle(5)
    output.close()
    result = outcomes(output)
    assert result["a" * 100] is Outcome.PLAYED and result["respuesta"] is Outcome.PLAYED


def test_expired_events_are_not_played():
    clock = ManualClock(1_000)
    output = AudioOutput(FakeSynth(), FakePlayer(), clock=clock)
    output.submit(event(1, "viejo", deadline=500))
    assert output.wait_idle(2)
    output.close()
    assert outcomes(output)["viejo"] is Outcome.EXPIRED


def test_event_expiring_during_synthesis_is_not_played():
    clock = ManualClock(0)

    class AdvancingSynth(FakeSynth):
        def synthesize(self, text):
            clock.advance(2_000)  # la síntesis tardó más que la vigencia
            return super().synthesize(text)

    output = AudioOutput(AdvancingSynth(), FakePlayer(), clock=clock)
    output.submit(event(1, "tarde", deadline=1_000))
    assert output.wait_idle(2)
    output.close()
    assert outcomes(output)["tarde"] is Outcome.EXPIRED


def test_cancel_during_synthesis_prevents_playback():
    started = threading.Event()
    release = threading.Event()

    class Slow(FakeSynth):
        def synthesize(self, text):
            started.set()
            release.wait(2)
            return super().synthesize(text)

    player = FakePlayer()
    output = AudioOutput(Slow(), player)
    output.submit(event(1, "frase"))
    assert started.wait(2)
    output.cancel(1)
    release.set()
    assert output.wait_idle(2)
    output.close()
    assert outcomes(output)["frase"] is Outcome.CANCELLED and player.played == []


def test_synthesis_failure_is_recorded_and_output_keeps_working():
    output = AudioOutput(FakeSynth(fail_on="rota"), FakePlayer())
    output.submit(event(1, "rota"))
    output.submit(event(2, "sana"))
    assert output.wait_idle(2)
    output.close()
    rota = next(r for r in output.records if r.text_or_sound == "rota")
    assert rota.outcome is Outcome.FAILED and "no se pudo" in rota.error
    assert outcomes(output)["sana"] is Outcome.PLAYED


def test_full_queue_drops_lowest_priority():
    release = threading.Event()

    class Blocking(FakeSynth):
        def synthesize(self, text):
            release.wait(2)
            return super().synthesize(text)

    output = AudioOutput(Blocking(), FakePlayer(), max_queue=2)
    output.submit(event(1, "sonando"))
    time.sleep(0.05)
    output.submit(event(2, "larga", Priority.LONG))
    output.submit(event(3, "corta"))
    output.submit(event(4, "busqueda", Priority.SEARCH))
    release.set()
    assert output.wait_idle(5)
    output.close()
    result = outcomes(output)
    assert result["larga"] is Outcome.DROPPED
    assert result["corta"] is Outcome.PLAYED and result["busqueda"] is Outcome.PLAYED


def test_sounds_do_not_go_through_tts():
    synth = FakeSynth()
    player = FakePlayer()
    output = AudioOutput(synth, player)
    output.submit(event(1, "sonido:bip", Priority.SEARCH))
    assert output.wait_idle(2)
    output.close()
    assert synth.calls == [] and player.played[0][0] == 120  # 120 ms a 1 kHz


def test_records_measure_wait_until_sound():
    output = AudioOutput(FakeSynth(delay_s=0.05), FakePlayer())
    output.submit(event(1, "hola"))
    assert output.wait_idle(2)
    output.close()
    assert output.records[0].wait_ms >= 45


def test_cached_synth_reuses_frequent_phrases():
    synth = FakeSynth()
    cached = CachedSynth(synth, max_items=2)
    cached.preload(["a", "b"])
    a = cached.synthesize("a")
    cached.synthesize("c")  # expulsa "b", la menos usada
    cached.synthesize("b")
    assert synth.calls == ["a", "b", "c", "b"]
    assert not a.flags.writeable and (cached.hits, cached.misses) == (1, 4)


# --- motores sin modelos ----------------------------------------------------------------


def test_settings_from_default_config():
    s = VoiceSettings.from_config(load_config())
    assert s.tts_voice == "es_AR-daniela-high" and s.stt_model == "small"
    assert s.voice_path == ROOT / "weights" / "piper" / "es_AR-daniela-high.onnx"
    assert s.deadlines_s["control"] == 2.0


def test_missing_models_explain_how_to_download(tmp_path):
    s = VoiceSettings(voices_dir=str(tmp_path), stt_models_dir=str(tmp_path))
    with pytest.raises(EstimatorError, match="descargar"):
        PiperSynth(s)
    with pytest.raises(EstimatorError, match="descargar"):
        WhisperTranscriber(s)


def test_wav_round_trip(tmp_path: Path):
    samples = tone([440], 200, 16_000)
    write_wav(tmp_path / "x.wav", samples, 16_000)
    back, rate = read_wav(tmp_path / "x.wav")
    assert rate == 16_000 and np.allclose(back, samples, atol=1e-4)


def test_whisper_variants_of_search_and_describe():
    """Transcripciones de Whisper vistas en el bucle con Piper (BITACORA 2026-11-17)."""
    parser = CommandParser(ClassCatalog(COCO_CLASSES))
    cases = {
        "Encontre la botella.": ("search", "bottle"),
        "Encuéntrame la taza.": ("search", "cup"),
        "Descríme lo que hay.": ("describe", ""),
        "Cuéntame qué tengo enfrente.": ("describe", ""),
        "Describime lo que hay.": ("describe", ""),
        "Describí.": ("describe", ""),
    }
    for phrase, expected in cases.items():
        assert outcome(parser.parse(phrase)) == expected, phrase
