"""Regresiones R1: cierres y repetir con AudioOutput real y dispositivos falsos."""

import threading
from collections import deque

import numpy as np
import pytest
from test_modes import CATALOG, SEC, G, cmd, joint

from visionx.app.controller import EXPIRED_REPEAT, Controller, Say
from visionx.app.runtime import View
from visionx.modes.common import ModeSettings, speech_s
from visionx.modes.describe import DescribeHandler
from visionx.modes.distance import DistanceHandler
from visionx.modes.read import ReadHandler
from visionx.modes.search import SearchHandler
from visionx.reading.worker import ReadJob
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import AudioEvent, DistancePolicy, FrameKey, Mode, Priority
from visionx.shared.errors import InvalidInput
from visionx.voice.commands import CommandParser
from visionx.voice.output import AudioOutput, FakePlayer, Outcome


class TimedSynth:
    """Audio fake: usa exactamente la duración estimada pública de los modos."""

    sample_rate = 1000

    def synthesize(self, text):
        return np.zeros(round(speech_s(text, ModeSettings()) * self.sample_rate), np.float32)


class VirtualPlayer:
    """Reloj manual; termina el clip solo al avanzar su duración o al cancelarse."""

    def __init__(self, clock):
        self.clock = clock
        self.started = threading.Event()
        self.end_ns = None
        self.closed = False

    def play(self, samples, sample_rate, stop):
        self.end_ns = self.clock.now_ns() + round(len(samples) / sample_rate * SEC)
        self.started.set()
        while self.clock.now_ns() < self.end_ns:
            if stop.wait(0.001):
                return False
        return not stop.is_set()

    def close(self):
        self.closed = True


def test_distance_timeout_sentence_can_finish_before_audio_expiry():
    clock = ManualClock(0)
    player = VirtualPlayer(clock)
    output = AudioOutput(TimedSynth(), player, clock=clock)
    ctl = Controller(
        CommandParser(CATALOG),
        output,
        latest_view=View,
        handlers={Mode.DISTANCE: DistanceHandler(G, CATALOG.spoken)},
        deadlines_s={"distance": 5},
        clock=clock,
    )
    try:
        ctl.handle_text("a qué distancia está la taza")
        assert player.started.wait(1)  # sonido:ok
        clock.set(SEC)
        assert output.wait_idle(1)
        player.started.clear()
        for tenth in range(11, 41):
            clock.set(tenth * SEC // 10)
            ctl.tick()
            if output.records[-1].text_or_sound == "No pude medir a tiempo.":
                break
        assert player.started.wait(1)
        record = output.records[-1]
        assert record.text_or_sound == "No pude medir a tiempo."
        if player.end_ns < 5 * SEC:
            clock.set(player.end_ns)
            assert output.wait_idle(1)
        clock.set(5 * SEC)
        ctl.tick()
        assert output.wait_idle(1)
        assert record.outcome == Outcome.PLAYED, (record, player.end_ns)
    finally:
        ctl.close()
        output.close()


@pytest.mark.parametrize("next_order", ["leé", "cancelar", "close"])
def test_repeat_audio_is_cancelled_by_new_command(next_order):
    """Este camino ya existía en origin/main; reproducir el límite del arreglo del PR."""
    entered, release = threading.Event(), threading.Event()
    phrase = "Taza a la izquierda, a 1 metro."

    class BlockingRepeatSynth(TimedSynth):
        count = 0

        def synthesize(self, text):
            if text == phrase:
                self.count += 1
                if self.count == 2:
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("no se liberó la síntesis fake")
            return super().synthesize(text)

    class Answer:
        def start(self, *args):
            return [Say(phrase)]

        def tick(self, *args):
            return [], True

    output = AudioOutput(BlockingRepeatSynth(), FakePlayer())
    ctl = Controller(CommandParser(CATALOG), output, handlers={Mode.DISTANCE: Answer()})
    try:
        ctl.handle_text("a qué distancia está la taza")
        ctl.tick()
        assert output.wait_idle(1)
        ctl.handle_text("repetí")
        assert entered.wait(1)
        repeated = output.records[-1]
        ctl.close() if next_order == "close" else ctl.handle_text(next_order)
        release.set()
        assert output.wait_idle(1)
        assert repeated.outcome == Outcome.CANCELLED, repeated
    finally:
        release.set()
        if hasattr(ctl, "close"):
            ctl.close()
        output.close()


@pytest.mark.parametrize("kind", ["search-absent", "search-found", "describe", "read"])
def test_deadline_closings_finish_in_virtual_player(kind):
    clock = ManualClock(0)
    player = VirtualPlayer(clock)
    output = AudioOutput(TimedSynth(), player, clock=clock)
    if kind.startswith("search"):
        mode, text, handler = Mode.SEARCH, "buscá la taza", SearchHandler(G, CATALOG.spoken)
    elif kind == "describe":
        mode, text, handler = Mode.DESCRIBE, "describí", DescribeHandler(G, CATALOG.spoken)
    else:
        mode, text, handler = Mode.READ, "leé", ReadHandler()
    ctl = Controller(
        CommandParser(CATALOG),
        output,
        latest_view=View,
        handlers={mode: handler},
        deadlines_s={mode.value: 10},
        clock=clock,
    )
    try:
        ctl.handle_text(text)
        if kind == "search-found":
            handler.found_once = True  # búsqueda que ya tuvo una adquisición
        assert player.started.wait(1)
        clock.set(2 * SEC)
        assert output.wait_idle(1)
        player.started.clear()
        for tenth in range(21, 100):
            clock.set(tenth * SEC // 10)
            ctl.tick()
            if len(output.records) > 1:
                break
        assert player.started.wait(1)
        closing = output.records[-1]
        assert not closing.text_or_sound.startswith("sonido:")
        assert player.end_ns < 10 * SEC
        clock.set(player.end_ns)
        assert output.wait_idle(1)
        clock.set(10 * SEC)
        ctl.tick()
        assert closing.outcome == Outcome.PLAYED
        assert (
            closing.started_ns + int(speech_s(closing.text_or_sound, ModeSettings()) * SEC)
            < 10 * SEC
        )
    finally:
        ctl.close()
        output.close()


def test_distance_does_not_wait_for_precision_to_release_its_closing():
    class Pending:
        def request(self, *args, **kwargs):
            pass

        def select(self, *args):
            return None, True

    handler = DistanceHandler(G, CATALOG.spoken)
    command = cmd(Mode.DISTANCE, "cup", deadline_s=5)
    view = View(precise=Pending())
    handler.start(command, view, 0)
    says, done = handler.tick(command, view, int(2.3 * SEC))
    assert done and says[0].text_or_sound == "No pude medir a tiempo."


def test_late_tick_uses_short_closing_or_silence_without_overrunning_deadline():
    for now, expected in [(8.5, "Fin de búsqueda."), (9.9, None)]:
        handler = SearchHandler(G, CATALOG.spoken)
        command = cmd(Mode.SEARCH, "cup", deadline_s=10)
        handler.start(command, View(), 0)
        handler.found_once = True
        says, done = handler.tick(command, View(), int(now * SEC))
        assert done
        if expected is None:
            assert says == []
        else:
            assert says[0].text_or_sound == expected
            assert speech_s(expected, handler.s) <= 10 - now


def test_repeat_keeps_command_deadline_and_expires_while_synthesizing():
    entered, release = threading.Event(), threading.Event()

    class Blocking(TimedSynth):
        def synthesize(self, text):
            if text == "Respuesta.":
                entered.set()
                assert release.wait(2)
            return super().synthesize(text)

    clock = ManualClock(0)
    output = AudioOutput(Blocking(), FakePlayer(), clock=clock)
    ctl = Controller(CommandParser(CATALOG), output, clock=clock)
    try:
        ctl.last_spoken = "Respuesta."
        ctl._last_spoken_deadline_ns = 5 * SEC
        ctl.handle_text("repetí")
        assert entered.wait(1)
        clock.set(6 * SEC)  # sin tick: AudioOutput debe revisar el deadline tras TTS
        release.set()
        assert output.wait_idle(1)
        assert output.records[-1].outcome == Outcome.EXPIRED
        count = len(output.records)
        ctl.handle_text("repetí")
        assert output.wait_idle(1)
        # No renovar una orden vencida: avisar, sin repetir el hecho viejo.
        assert [r.text_or_sound for r in output.records[count:]] == [EXPIRED_REPEAT]
    finally:
        release.set()
        ctl.close()
        output.close()


def test_repeat_does_not_hide_audio_of_the_active_mode_from_cancellation():
    class Sink:
        def __init__(self):
            self.events, self.cancelled = [], []

        def submit(self, event):
            self.events.append(event)

        def cancel(self, request_id):
            self.cancelled.append(request_id)

    sink = Sink()
    ctl = Controller(CommandParser(CATALOG), sink, clock=ManualClock(0))
    command = ctl.handle_text("buscá la taza")
    mode_id = ctl.active.request_id
    ctl.handle_text("repetí")
    repeat_id = sink.events[-1].request_id
    ctl.handle_text("leé")
    assert mode_id != repeat_id
    assert mode_id in sink.cancelled and repeat_id in sink.cancelled
    assert command.ok


def test_trace_has_command_ids_player_times_and_fact_age_without_changing_legacy_fields():
    clock = ManualClock(SEC)
    output = AudioOutput(TimedSynth(), FakePlayer(), clock=clock)
    events = []
    view = View(joint(1, 10, [("cup", 0, 1)]))
    ctl = Controller(
        CommandParser(CATALOG),
        output,
        clock=clock,
        latest_view=lambda: view,
        handlers={Mode.SEARCH: SearchHandler(G, CATALOG.spoken, ModeSettings(confirm_hits=1))},
        deadlines_s={"search": 10},
        on_event=lambda k, d: events.append((k, d)),
    )
    try:
        ctl.handle_text("buscá la taza")
        assert output.wait_idle(1)
        order = next(d for k, d in events if k == "orden")
        assert order["request_id"] == 1 and order["issued_ns"] == SEC
        assert order["deadline_ns"] == 11 * SEC
        guide = next(r for r in output.records if r.text_or_sound == "Taza al frente.")
        data = guide.trace_data()
        assert data["resultado"] == "reproducido" and data["espera_ms"] == 0
        assert data["started_ns"] == data["finished_ns"] == SEC
        assert data["valid_until_ns"] == 2 * SEC
        assert data["fact_received_ns"] == SEC and data["fact_age_ms"] == 0
        ctl.handle_text("repetí")
        assert output.wait_idle(1)
        repeated = output.records[-1].trace_data()
        assert repeated["request_id"] != guide.request_id
        assert repeated["fact_received_ns"] == data["fact_received_ns"]
        assert repeated["deadline_ns"] == data["deadline_ns"]
    finally:
        ctl.close()
        output.close()


def test_audio_trace_distinguishes_discard_time_from_audible_end():
    clock = ManualClock(2 * SEC)
    output = AudioOutput(TimedSynth(), FakePlayer(), clock=clock)
    try:
        output.submit(
            AudioEvent(
                1,
                Priority.SHORT,
                "Guía.",
                SEC,
                valid_until_ns=SEC,
                fact_received_ns=0,
                frame=FrameKey("s", 7),
            )
        )
        data = output.records[-1].trace_data()
        assert data["started_ns"] is None and data["finished_ns"] is None
        assert data["ended_ns"] == 2 * SEC and data["fact_age_ms"] is None
        assert data["frame"] == {"session_id": "s", "frame_id": 7}
    finally:
        output.close()


@pytest.mark.parametrize("invalid", [-1, True, 0.5])
def test_audio_event_validates_optional_fact_timestamp(invalid):
    with pytest.raises(InvalidInput):
        AudioEvent(1, Priority.SHORT, "Texto.", SEC, fact_received_ns=invalid)


def test_combined_guide_age_uses_oldest_fact_and_keeps_independent_expiry():
    settings = ModeSettings(confirm_hits=1, guide_style="palabras")
    handler = SearchHandler(G, CATALOG.spoken, settings)
    command = cmd(Mode.SEARCH, "cup")
    handler.start(command, View(joint(0, 0, [("cup", 0, 1)])), 0)
    seen = joint(2, 20, [("cup", 0, 1)])
    measured = joint(1.8, 18, [("cup", 0, 1, 1.4, DistancePolicy.SPEAK_METERS)], measured=True)
    says, _ = handler.tick(command, View(seen, measured), 2 * SEC)
    guide = next(s for s in says if "1,4 metros" in s.text_or_sound)
    assert guide.fact_received_ns == int(1.8 * SEC)
    assert guide.valid_until_ns == 3 * SEC  # dirección vence antes que la Z


def test_read_does_not_start_a_block_that_would_be_cut_at_deadline():
    handler = ReadHandler(reader=object())
    handler._job = ReadJob(1, earliest_ns=0, end_ns=SEC)
    handler._loaded = True
    handler._chunks = deque(["Texto largo. " * 25])
    command = cmd(Mode.READ, None, deadline_s=5)
    says, done = handler.tick(command, View(), SEC)
    assert done and says[0].text_or_sound == "No terminé de leer a tiempo."
    assert speech_s(says[0].text_or_sound, ModeSettings()) + 1 <= 4
    assert handler._job.cancelled.is_set()


def test_search_leaves_room_after_ongoing_guides_for_its_closing():
    clock = ManualClock(0)

    class AdvancingPlayer:
        def play(self, samples, sample_rate, stop):
            if stop.is_set():
                return False
            clock.advance(round(len(samples) / sample_rate * SEC))
            return True

        def close(self):
            pass

    output = AudioOutput(TimedSynth(), AdvancingPlayer(), clock=clock)
    view = View()
    handler = SearchHandler(G, CATALOG.spoken, ModeSettings(confirm_hits=1))
    ctl = Controller(
        CommandParser(CATALOG),
        output,
        latest_view=lambda: view,
        handlers={Mode.SEARCH: handler},
        deadlines_s={"search": 10},
        clock=clock,
    )
    try:
        ctl.handle_text("buscá la taza")
        assert output.wait_idle(1)
        number = 0
        while ctl.active is not None and clock.now_ns() < 10 * SEC:
            number += 1
            view = View(joint(clock.now_ns() / SEC, number, [("cup", 0, 1)]))
            ctl.tick()
            assert output.wait_idle(1)
            clock.advance(SEC // 10)
        closing = output.records[-1]
        assert closing.text_or_sound == "Terminé de buscar taza."
        assert closing.outcome == Outcome.PLAYED and closing.ended_ns < 10 * SEC
        assert any(r.text_or_sound == "Taza al frente." for r in output.records)
        clock.set(10 * SEC)
        ctl.tick()
        assert closing.outcome == Outcome.PLAYED
    finally:
        ctl.close()
        output.close()
