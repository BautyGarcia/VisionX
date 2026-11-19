"""Tests de los modos buscar/guiar y distancia (T09), sin cámara ni pesos.

Las uniones se arman a mano (detecciones y objetos ubicados con ángulo, Z y política elegidos)
para forzar cada caso: presencia de un solo frame, pérdida, iguales, profundidad vieja, etc.
El reloj es manual: los tiempos son exactos y no dependen de la máquina.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visionx.app.controller import Controller
from visionx.app.fakes import FakeDepthEstimator, synthetic_geometry
from visionx.app.runtime import Joint, View
from visionx.app.spatial import ProvisionalSpatialEstimator
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.detection.convert import build_detection
from visionx.modes import build_handlers
from visionx.modes.__main__ import main as modes_main
from visionx.modes.__main__ import parse_order
from visionx.modes.common import (
    ModeSettings,
    candidates,
    direction,
    measure,
    measure_phrase,
    meters,
    speech_s,
)
from visionx.modes.describe import (
    DescribeHandler,
    DescribeSettings,
    MetricFact,
    PresenceFact,
    SceneFacts,
    TemplateNarrator,
)
from visionx.modes.distance import DistanceHandler
from visionx.modes.search import SearchHandler
from visionx.modes.simulate import RecordingSink
from visionx.modes.target import Target, TargetMemory, plural
from visionx.shared.clock import ManualClock
from visionx.shared.contracts import (
    Command,
    DistancePolicy,
    Mode,
    SpatialObject,
    StereoFrame,
)
from visionx.shared.errors import InvalidInput
from visionx.voice.commands import CommandParser

G = synthetic_geometry()  # 640x360, fx = 608
CATALOG = ClassCatalog(COCO_CLASSES)
S = ModeSettings(guide_style="palabras")  # frases de guía; el sonar se prueba aparte
SEC = 1_000_000_000


def u_for(angle_deg: float) -> float:
    """Columna del píxel que se ve a ese ángulo horizontal."""
    return G.cx + G.fx * math.tan(math.radians(angle_deg))


# Un objeto en una unión: (clase, ángulo, track, z, política). z None = sin ubicar.
Obj = tuple


def v_for(angle_deg: float) -> float:
    """Fila del píxel que se ve a ese ángulo vertical (positivo hacia arriba)."""
    return G.cy - G.fy * math.tan(math.radians(angle_deg))


def joint(
    t_s: float,
    frame_id: int,
    objs: list[Obj],
    *,
    measured: bool = False,
    scores=None,
    v_deg: float = 0.0,
) -> Joint:
    frame = StereoFrame(
        session_id="s",
        frame_id=frame_id,
        left=np.zeros((G.height, G.width, 3), np.uint8),
        right=np.zeros((G.height, G.width, 3), np.uint8),
        received_ns=int(t_s * SEC),
        rectified=True,
        calibration_id="test",
        geometry_id=G.geometry_id,
    )
    boxes = []
    for _, angle, *_rest in objs:
        u, v = u_for(angle), v_for(v_deg)
        boxes.append([u - 20, v - 30, u + 20, v + 30])
    det = build_detection(
        frame,
        model_id="test",
        boxes_xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        class_names=[o[0] for o in objs],
        scores=np.array(scores, np.float32) if scores else np.full(len(objs), 0.8, np.float32),
        inference_ms=1.0,
        track_ids=[o[2] for o in objs] if objs else None,
    )
    if not measured:
        return Joint(frame, det, None, ())
    depth = FakeDepthEstimator(G).predict(frame)
    located = tuple(
        SpatialObject(
            session_id="s", frame_id=frame_id, detection_index=i, track_id=o[2], label=o[0],
            depth_model_id=depth.model_id, angle_h_deg=o[1], angle_v_deg=0.0, z_m=o[3],
            z_spread_m=0.02 if o[3] else None, valid_fraction=1.0 if o[3] else 0.0,
            detector_score=0.8, policy=o[4],
        )
        for i, o in enumerate(objs)
    )  # fmt: skip
    return Joint(frame, det, depth, located)


def cmd(mode: Mode, target: str | None, t_s: float = 0.0, deadline_s: float | None = 30.0,
        request_id: int = 1) -> Command:  # fmt: skip
    deadline = int((t_s + deadline_s) * SEC) if deadline_s else None
    return Command(request_id, mode, "x", int(t_s * SEC), target=target, deadline_ns=deadline)


def texts(says) -> list[str]:
    return [s.text_or_sound for s in says]


def search(settings: ModeSettings = S, memory: TargetMemory | None = None) -> SearchHandler:
    return SearchHandler(G, CATALOG.spoken, settings, memory)


def run(handler, command, frames: list[Joint], *, measured: list[Joint] | None = None):
    """Tick por cada frame (en su tiempo). Devuelve [(t, texto)] y si terminó."""
    out, done = [], False
    m_by_t = {j.frame.received_ns: j for j in measured or []}
    current_m = None
    for j in frames:
        current_m = m_by_t.get(j.frame.received_ns, current_m)
        says, done = handler.tick(command, View(j, current_m), j.frame.received_ns)
        out += [(j.frame.received_ns / SEC, t) for t in texts(says)]
        if done:
            break
    return out, done


SPEAK = DistancePolicy.SPEAK_METERS


# --- piezas comunes ----------------------------------------------------------------------


def test_direction_and_meters_phrasing():
    assert direction(0, S) == "al frente" and direction(-8, S) == "al frente"
    assert direction(-15, S) == "un poco a la izquierda" and direction(30, S) == "a la derecha"
    assert meters(1.07, S) == "1,1 metros" and meters(1.0, S) == "1 metro"
    assert meters(0.54, S) == "0,5 metros" and meters(2.96, S) == "3 metros"


def test_vertical_is_added_only_when_clear():
    assert direction(-30, S, -25) == "a la izquierda, abajo"
    assert direction(0, S, 20) == "al frente, arriba"
    assert (
        direction(0, S, 10) == "al frente" and direction(-15, S, None) == "un poco a la izquierda"
    )
    # en la guía: abajo se dice, y "al frente" sigue decidiendo metros y llegada
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    out, _ = run(h, c, [joint(0.1 * i, i, [("cup", 0, None)], v_deg=-30) for i in (1, 2)])
    assert [x for _, x in out] == ["sonido:bip", "Taza al frente, abajo."]
    m = joint(0.3, 3, [("cup", 0, None, 0.55, SPEAK)], measured=True, v_deg=-30)
    says, done = h.tick(c, View(joint(0.4, 4, [("cup", 0, None)], v_deg=-30), m), int(0.4 * SEC))
    assert done and texts(says) == ["Llegaste: taza al frente, abajo, a 0,6 metros."]
    d = distance()
    dc = cmd(Mode.DISTANCE, "cup", t_s=0.2, request_id=5)
    assert texts(d.start(dc, View(None, m), int(0.4 * SEC))) == [
        "Taza al frente, abajo, a 0,6 metros."
    ]


def test_hysteresis_keeps_the_last_zone_near_a_limit():
    from visionx.modes.common import horizontal, vertical

    assert horizontal(9.5, S) == "un poco a la derecha"
    assert horizontal(9.5, S, prev="al frente") == "al frente"  # 1,5° pasado el límite: no cambia
    assert horizontal(12.0, S, prev="al frente") == "un poco a la derecha"
    assert vertical(-13.5, S) == "" and vertical(-13.5, S, prev=", abajo") == ", abajo"
    assert vertical(-11.0, S, prev=", abajo") == ""


def test_arrival_with_the_object_slightly_to_one_side():
    h, c = search(), cmd(Mode.SEARCH, "chair")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("chair", 15, None)]), joint(0.2, 2, [("chair", 15, None)])])
    m = joint(1.3, 13, [("chair", 15, None, 0.55, SPEAK)], measured=True)
    says, done = h.tick(c, View(joint(1.6, 16, [("chair", 15, None)]), m), int(1.6 * SEC))
    assert done and texts(says) == ["Llegaste: silla un poco a la derecha, a 0,6 metros."]


def test_measure_never_gives_meters_without_valid_depth_or_out_of_range():
    def obj(z, policy=SPEAK):
        return SpatialObject("s", 0, 0, None, "cup", "d", 0.0, 0.0, z, 0.01, 1.0, 0.9, policy)

    assert measure(obj(1.2), S).ok
    assert measure(None, S).reason == "sin-profundidad"
    assert measure(obj(1.2, DistancePolicy.DIRECTION_ONLY), S).reason == "sin-medida"
    assert measure(obj(1.2, DistancePolicy.ABSTAIN), S).reason == "sin-medida"
    assert measure(obj(1.2, DistancePolicy.ASK_REFRAME), S).reason == "reencuadrar"
    far = measure(obj(4.2), S)
    assert not far.ok and measure_phrase(far, S) == "lejos, a más de 3 metros"
    assert measure_phrase(measure(obj(0.3), S), S) == "muy cerca, a menos de 0,5 metros"
    assert "metro" not in measure_phrase(measure(obj(1.2, DistancePolicy.DIRECTION_ONLY), S), S)


def test_settings_from_config_and_plural():
    s = ModeSettings.from_config({"depth": {"range_m": [0.4, 2.5]}, "modes": {"lost_s": 2}})
    assert s.range_m == (0.4, 2.5) and s.lost_s == 2
    with pytest.raises(ValueError):
        ModeSettings.from_config({"modes": {"no_existe": 1}})
    assert plural("taza") == "tazas" and plural("televisor") == "televisores"
    assert plural("control remoto") == "controles remotos" and plural("ratón") == "ratones"


# --- buscar ------------------------------------------------------------------------------


def test_presence_needs_confirmation_in_time_not_one_frame():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    out, _ = run(h, c, [joint(0.1, 1, [("cup", -30, None)]), joint(0.2, 2, []),
                        joint(1.5, 3, [("cup", -30, None)])])  # fmt: skip
    assert out == []  # dos vistas separadas por más de confirm_s: no alcanza
    out, _ = run(h, c, [joint(1.6, 4, [("cup", -30, None)])])
    assert [t for _, t in out] == ["sonido:bip", "Taza a la izquierda."]


def test_guidance_and_presence_expire_from_observation_not_tick():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    h.tick(c, View(joint(0.1, 1, [("cup", -30, None)])), int(0.1 * SEC))
    says, _ = h.tick(c, View(joint(0.2, 2, [("cup", -30, None)])), int(0.8 * SEC))
    assert texts(says) == ["sonido:bip", "Taza a la izquierda."]
    assert [s.valid_until_ns for s in says] == [int(1.2 * SEC)] * 2
    # La pérdida narra historia: no afirma que la dirección antigua siga vigente.
    says, _ = h.tick(c, View(joint(3.0, 30, [])), 3 * SEC)
    assert texts(says) == ["Perdí taza: quedó a la izquierda."]
    assert says[0].valid_until_ns is None


@pytest.mark.parametrize("z_m", [0.55, 1.4])
@pytest.mark.parametrize("measured_at,expected_until", [(2.0, 5.0), (4.1, 5.2)])
def test_metric_guidance_expires_with_first_expiring_fact(z_m, measured_at, expected_until):
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])])
    seen = joint(4.2, 42, [("cup", 0, None)])
    measured = joint(measured_at, 40, [("cup", 0, None, z_m, SPEAK)], measured=True)
    says, done = h.tick(c, View(seen, measured), int(4.3 * SEC))
    assert len(says) == 1 and "metro" in says[0].text_or_sound
    assert done == (z_m == 0.55)
    assert says[0].valid_until_ns == int(expected_until * SEC)


def test_stale_observation_does_not_guide():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    old = [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])]
    for j in old:  # se ven recién a los 5 s: describen dónde ESTABA la taza
        says, _ = h.tick(c, View(j, None), 5 * SEC)
        assert texts(says) == []


def test_guidance_speaks_on_change_not_every_frame():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    # gira 10 grados por segundo durante 6 s: de la izquierda al frente
    frames = [joint(0.1 * i, i, [("cup", -40 + i, None)]) for i in range(1, 60)]
    out, _ = run(h, c, frames)
    said = [t for _, t in out]
    assert said[:2] == ["sonido:bip", "Taza a la izquierda."]
    # después de la primera, frases cortas sin el nombre (short_guide)
    assert "Un poco a la izquierda." in said and "Al frente." in said
    times = [t for t, x in out if not x.startswith("sonido")]
    assert all(b - a >= S.min_interval_s - 1e-9 for a, b in zip(times, times[1:], strict=False))
    assert len(out) < 8  # 59 frames, pocas frases


def test_repeats_direction_after_interval_if_nothing_changes():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    out, _ = run(h, c, [joint(0.1 * i, i, [("cup", -30, None)]) for i in range(1, 60)])
    phrases = [t for t, x in out if x in ("Taza a la izquierda.", "A la izquierda.")]
    assert len(phrases) == 2 and phrases[1] - phrases[0] == pytest.approx(S.repeat_s, abs=0.11)


def test_lost_target_is_announced_and_reappearance_is_a_new_acquisition():
    events = []
    h = SearchHandler(G, CATALOG.spoken, S, None, lambda k, d: events.append((k, d)))
    c = cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    seen = [joint(0.1 * i, i, [("cup", -30, 5)]) for i in range(1, 4)]
    gone = [joint(0.3 + 0.1 * i, 3 + i, []) for i in range(1, 31)]  # 3 s sin la taza
    back = [joint(3.4 + 0.1 * i, 40 + i, [("cup", 25, 9)]) for i in range(1, 4)]
    out, _ = run(h, c, seen + gone + back)
    said = [t for _, t in out]
    # salió por la izquierda: se dice hacia dónde quedó
    assert said.count("sonido:bip") == 2 and "Perdí taza: quedó a la izquierda." in said
    lost_t = next(t for t, x in out if x.startswith("Perdí"))
    assert lost_t == pytest.approx(0.3 + S.lost_s, abs=0.11)
    acquired = [d for k, d in events if k == "buscar-adquirido"]
    assert [a["reaparicion"] for a in acquired] == [False, True]
    assert said[-1] == "Taza a la derecha."


def test_no_guidance_with_the_last_known_direction_while_target_is_not_seen():
    from dataclasses import replace

    s = replace(S, repeat_s=1.0, min_interval_s=0.5, lost_s=3.0)
    h, c = search(s), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", -30, None)]), joint(0.2, 2, [("cup", -30, None)])])
    out, _ = run(h, c, [joint(0.2 + 0.1 * i, 2 + i, []) for i in range(1, 36)])
    said = [x for _, x in out]
    assert said == ["Perdí taza: quedó a la izquierda."]  # nada de "a la izquierda" repetido


def test_system_lag_is_not_reported_as_a_lost_object():
    events = []
    h = SearchHandler(G, CATALOG.spoken, S, None, lambda k, d: events.append(k))
    c = cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    last = joint(0.2, 2, [("cup", -30, None)])
    run(h, c, [joint(0.1, 1, [("cup", -30, None)]), last])
    # el detector se atrasa: durante 3 s la vista sigue siendo el frame de 0,2 s
    for i in range(1, 30):
        says, done = h.tick(c, View(last, None), int((0.2 + 0.1 * i) * SEC))
        assert texts(says) == [] and not done
    assert "buscar-demora" in events and h.state == "guiando"
    # vuelve con la taza: sigue guiando sin "Perdí" ni bip nuevo
    out, _ = run(h, c, [joint(3.3, 33, [("cup", -29, None)])])
    assert "Perdí taza." not in [x for _, x in out] and "sonido:bip" not in [x for _, x in out]


def test_follows_chosen_object_by_angle_even_without_track_ids():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", -30, None)]), joint(0.2, 2, [("cup", -30, None)])])
    # aparece otra taza a la derecha y la cámara gira un poco: se sigue a la de la izquierda
    run(h, c, [joint(0.3 + 0.1 * i, 2 + i, [("cup", -30 + 2 * i, None), ("cup", 30, None)])
               for i in range(1, 6)])  # fmt: skip
    assert h.target is not None and h.target.angle_h_deg == pytest.approx(-20, abs=0.5)


def make_controller(handlers=None, memory=None, clock=None):
    clock = clock or ManualClock(0)
    sink = RecordingSink(clock, lambda: 0)
    state = {"view": View()}
    ctl = Controller(
        CommandParser(CATALOG),
        sink,
        latest_view=lambda: state["view"],
        handlers=handlers or build_handlers({}, G, CATALOG.spoken),
        deadlines_s={"search": 30, "distance": 5, "choose": 10},
        clock=clock,
    )
    return ctl, sink, state, clock


def advance(ctl, state, clock, view: View) -> None:
    clock.set(max(clock.now_ns(), (view.seen or view.measured).frame.received_ns))
    state["view"] = view
    ctl.tick()


def test_several_equal_objects_ask_and_choice_completes_the_same_search():
    ctl, sink, state, clock = make_controller()
    ctl.handle_text("buscá la taza")
    search_id = ctl.active.request_id
    two = [("cup", -20, None), ("cup", 20, None)]
    advance(ctl, state, clock, View(joint(0.1, 1, two)))
    advance(ctl, state, clock, View(joint(0.2, 2, two)))
    said = [s.text for s in sink.spoken]
    assert said[-1].startswith("Veo 2 tazas. ¿Cuál?")
    ctl.handle_text("la de la derecha")
    assert ctl.active is not None and ctl.active.request_id == search_id  # no la reemplazó
    assert [s.text for s in sink.spoken][-2:] == ["sonido:bip", "Taza un poco a la derecha."]
    assert all(s.request_id == search_id for s in sink.spoken[-2:])


def test_unanswered_choice_is_asked_again_not_decided_for_the_user():
    h, c = search(), cmd(Mode.SEARCH, "bottle")
    h.start(c, View(), 0)
    two = [("bottle", -20, None), ("bottle", -10, None)]
    out, _ = run(h, c, [joint(0.1 * i, i, two) for i in range(1, 160)])
    asked = [t for t, x in out if "¿Cuál" in x]
    assert len(asked) == 2 and asked[1] - asked[0] == pytest.approx(S.choice_repeat_s, abs=0.11)
    assert h.state == "eligiendo" and h.target is None


def test_duplicate_box_of_the_same_object_is_not_a_second_object():
    # una caja de silla dentro de otra (YOLO detectó una parte como otra silla)
    objs = [("chair", 0, None), ("chair", 0.5, None), ("chair", 30, None)]
    j = joint(0.1, 1, objs, scores=[0.8, 0.3, 0.8])
    from visionx.modes.common import candidates

    cands = candidates(j, "chair", G)
    assert [c.index for c in cands] == [0, 2]


def test_one_stray_extra_detection_does_not_trigger_a_choice():
    h, c = search(), cmd(Mode.SEARCH, "backpack")
    h.start(c, View(), 0)
    one = [("backpack", -10, None)]
    two = [("backpack", -10, None), ("backpack", 30, None)]
    out, _ = run(
        h,
        c,
        [joint(0.1, 1, one), joint(0.2, 2, two)],
    )
    assert [x for _, x in out] == ["sonido:bip", "Mochila un poco a la izquierda."]


def test_choice_resolves_itself_when_only_one_remains():
    h, c = search(), cmd(Mode.SEARCH, "bottle")
    h.start(c, View(), 0)
    two = [("bottle", -20, None), ("bottle", 20, None)]
    run(h, c, [joint(0.1, 1, two), joint(0.2, 2, two)])
    assert h.state == "eligiendo"
    out, _ = run(h, c, [joint(0.2 + 0.1 * i, 2 + i, [("bottle", 20, None)]) for i in range(1, 13)])
    assert h.state == "guiando" and [x for _, x in out][:2] == [
        "sonido:bip",
        "Botella un poco a la derecha.",
    ]


def test_choose_nearest_needs_meters_and_choose_without_search_says_so():
    ctl, sink, state, clock = make_controller()
    ctl.handle_text("la de la izquierda")
    assert sink.spoken[-1].text == "No hay nada para elegir ahora." and ctl.active is None
    ctl.handle_text("buscá la taza")
    two = [("cup", -20, None), ("cup", 20, None)]
    advance(ctl, state, clock, View(joint(0.1, 1, two)))
    advance(ctl, state, clock, View(joint(0.2, 2, two)))
    ctl.handle_text("la más cercana")
    assert sink.spoken[-1].text.startswith("No puedo saber cuál está más cerca")
    measured = joint(0.25, 3, [("cup", -20, None, 2.0, SPEAK), ("cup", 20, None, 0.9, SPEAK)],
                     measured=True)  # fmt: skip
    state["view"] = View(joint(0.3, 4, two), measured)
    clock.set(int(0.3 * SEC))
    ctl.handle_text("la más cercana")
    assert [s.text for s in sink.spoken][-2:] == ["sonido:bip", "Taza un poco a la derecha."]


def test_nearest_choice_does_not_guide_to_an_instance_only_in_old_depth():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    two = [("cup", -20, 1), ("cup", 20, 2)]
    run(h, c, [joint(0.1, 1, two), joint(0.2, 2, two)])
    measured = joint(
        0.25, 3, [("cup", -20, 1, 0.8, SPEAK), ("cup", 20, 2, 1.5, SPEAK)], measured=True
    )
    seen = joint(0.3, 4, [("cup", 19, 2)])
    says = h.choose(c, "cerca", View(seen, measured), int(0.3 * SEC))
    assert texts(says)[-1] == "Taza un poco a la derecha."
    assert h.target.track_id == 2 and h.target.last_seen_ns == int(0.3 * SEC)


def test_meters_during_guidance_only_when_fresh_and_valid_and_arrival_ends():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    seen = [joint(0.1 * i, i, [("cup", 0, None)]) for i in range(1, 4)]
    run(h, c, seen)
    # medida sin política de metros: guía sin número
    m1 = joint(0.3, 3, [("cup", 0, None, 1.4, DistancePolicy.DIRECTION_ONLY)], measured=True)
    out, _ = run(h, c, [joint(1.6, 16, [("cup", 0, None)])], measured=[])
    says, _ = h.tick(c, View(joint(5.7, 57, [("cup", 0, None)]), m1), int(5.7 * SEC))
    assert not any("metro" in t for t in texts(says))
    m2 = joint(5.8, 58, [("cup", 1, None, 1.4, SPEAK)], measured=True)
    stale = View(joint(9.9, 99, [("cup", 0, None)]), m2)  # Z de hace 4,1 s: no se dice
    assert not any("metro" in t for t in texts(h.tick(c, stale, int(9.9 * SEC))[0]))
    m2b = joint(10.0, 100, [("cup", 1, None, 1.4, SPEAK)], measured=True)
    # la frase anterior ("Al frente.", 9,9 s) tarda ~1,15 s: la nueva espera a que termine
    says, _ = h.tick(c, View(joint(11.0, 110, [("cup", 0, None)]), m2b), int(11.0 * SEC))
    assert texts(says) == []
    says, _ = h.tick(c, View(joint(11.2, 112, [("cup", 0, None)]), m2b), int(11.2 * SEC))
    # misma dirección que la frase anterior: solo los metros nuevos, no la frase entera
    assert texts(says) == ["A 1,4 metros."]
    m3 = joint(11.6, 116, [("cup", 0, None, 0.55, SPEAK)], measured=True)
    says, done = h.tick(c, View(joint(11.7, 117, [("cup", 0, None)]), m3), int(11.7 * SEC))
    assert done and texts(says) == ["Llegaste: taza al frente, a 0,6 metros."]
    # más cerca que el rango validado: también llegó, sin un número que no se validó
    h2, c2 = search(), cmd(Mode.SEARCH, "cup", request_id=2)
    h2.start(c2, View(), 0)
    run(h2, c2, [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])])
    m4 = joint(1.3, 13, [("cup", 0, None, 0.3, SPEAK)], measured=True)
    says, done = h2.tick(c2, View(joint(1.4, 14, [("cup", 0, None)]), m4), int(1.4 * SEC))
    assert done and texts(says) == ["Llegaste: taza al frente, muy cerca, a menos de 0,5 metros."]


def test_search_closes_before_deadline():
    h, c = search(), cmd(Mode.SEARCH, "cup", deadline_s=10)
    h.start(c, View(), 0)
    closing_ns = c.deadline_ns - int((speech_s("No encontré taza.", S) + S.end_margin_s) * SEC)
    says, done = h.tick(c, View(), closing_ns)
    assert done and texts(says) == ["No encontré taza."]
    h2, c2 = search(), cmd(Mode.SEARCH, "cup", deadline_s=10, request_id=2)
    h2.start(c2, View(), 0)
    run(h2, c2, [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])])
    closing_ns = c2.deadline_ns - int(
        (speech_s("Terminé de buscar taza.", S) + S.end_margin_s) * SEC
    )
    says, done = h2.tick(c2, View(), closing_ns)
    assert done and texts(says) == ["Terminé de buscar taza."]


@pytest.mark.parametrize("explicit_choice", [True, False])
def test_search_remembers_selection_after_loss_with_two_new_tracks(explicit_choice):
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    two = [("cup", -20, 1), ("cup", 20, 2)]
    initial = two if explicit_choice else two[:1]
    run(h, c, [joint(0.1, 1, initial), joint(0.2, 2, initial)])
    if explicit_choice:
        h.choose(c, "izquierda", View(joint(0.2, 2, two)), int(0.2 * SEC))
    run(h, c, [joint(3, 30, [])])
    back = [("cup", -18, 11), ("cup", 18, 12)]
    out, _ = run(h, c, [joint(3.1, 31, back), joint(3.2, 32, back)])
    assert h.state == "guiando" and h.target.angle_h_deg == pytest.approx(-18, abs=0.1)
    assert not any("¿Cuál" in text for _, text in out)


def test_ambiguous_reassociation_asks_at_most_once_per_configured_interval():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    two = [("cup", -20, None), ("cup", 20, None)]
    run(h, c, [joint(0.1, 1, two), joint(0.2, 2, two)])
    h.choose(c, "izquierda", View(joint(0.2, 2, two)), int(0.2 * SEC))
    run(h, c, [joint(3, 30, [])])
    ambiguous = [("cup", -24, None), ("cup", -16, None)]
    out, _ = run(h, c, [joint(3 + i / 10, 30 + i, ambiguous) for i in range(1, 145)])
    asked = [t for t, text in out if "¿Cuál" in text]
    assert h.state == "eligiendo" and h.target is None
    assert len(asked) == 1 and asked[0] >= S.choice_repeat_s


def test_choice_cannot_acquire_an_object_from_stale_or_empty_video():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    two = [("cup", -20, None), ("cup", 20, None)]
    j = joint(0.2, 2, two)
    run(h, c, [joint(0.1, 1, two), j])
    out, _ = run(h, c, [joint(0.3, 3, []), joint(9, 90, [])])
    assert not any("¿Cuál" in text for _, text in out)
    says = h.choose(c, "izquierda", View(j), 10 * SEC)
    assert h.target is None and not any("sonido:bip" == t for t in texts(says))


def test_target_uses_surviving_id_then_position_and_rejects_a_tie():
    target = Target("cup", -20, 7, SEC)
    same_id = candidates(joint(4, 40, [("cup", -19, 8), ("cup", 20, 7)]), "cup", G)
    assert target.match(same_id, S).track_id == 7
    no_ids = candidates(joint(4, 40, [("cup", -19, None), ("cup", 20, None)]), "cup", G)
    assert target.match(no_ids, S).angle_h_deg == pytest.approx(-19, abs=0.1)
    tied = candidates(joint(4, 40, [("cup", -24, None), ("cup", -16, None)]), "cup", G)
    assert target.match(tied, S) is None


def test_target_depth_breaks_position_tie_without_inventing_depth():
    target = Target("cup", 0, None, SEC)
    original = candidates(joint(1, 10, [("cup", 0, None, 1.0, SPEAK)], measured=True), "cup", G)[0]
    target.update(original)
    cands = candidates(
        joint(4, 40, [("cup", -4, None, 1.1, SPEAK), ("cup", 4, None, 2.8, SPEAK)], measured=True),
        "cup",
        G,
    )
    assert target.match(cands, S).index == 0
    # Sin Z fiable, la posición empatada no alcanza para afirmar identidad.
    no_depth = [replace(c, obj=None) for c in cands]
    assert target.match(no_depth, S) is None


def test_new_search_replaces_implicit_reference_even_before_detection():
    memory = TargetMemory()
    h = search(memory=memory)
    h.start(cmd(Mode.SEARCH, "bottle"), View(), 0)
    h.start(cmd(Mode.SEARCH, "cup", t_s=10, request_id=2), View(), 10 * SEC)
    m = joint(15, 150, [("bottle", 0, None, 1.0, SPEAK)], measured=True)
    says = distance(memory).start(cmd(Mode.DISTANCE, None, t_s=15), View(m, m), 15 * SEC)
    assert texts(says) == ["No veo taza."]


def test_new_measurement_without_target_cannot_trigger_arrival_from_previous_z():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", -30, 1)]), joint(0.2, 2, [("cup", -30, 1)])])
    m1 = joint(0.3, 3, [("cup", -30, 1, 0.55, SPEAK)], measured=True)
    assert not h.tick(c, View(m1, m1), int(0.3 * SEC))[1]
    m2 = joint(0.4, 4, [], measured=True)
    seen = joint(0.5, 5, [("cup", 0, 1)])
    says, done = h.tick(c, View(seen, m2), int(0.5 * SEC))
    assert not done and not any("Llegaste" in text for text in texts(says))


def test_distance_after_arrival_keeps_reference_and_never_measures_other_class():
    memory = TargetMemory()
    h, c = search(memory=memory), cmd(Mode.SEARCH, "bottle")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("bottle", 0, 1)]), joint(0.2, 2, [("bottle", 0, 1)])])
    m = joint(0.3, 3, [("bottle", 0, 1, 0.55, SPEAK)], measured=True)
    assert h.tick(c, View(m, m), int(0.3 * SEC))[1]
    d, dc = distance(memory), cmd(Mode.DISTANCE, None, t_s=15)
    other = joint(15, 150, [("bicycle", 0, 2, 1.2, SPEAK)], measured=True)
    assert texts(d.start(dc, View(other, other), 15 * SEC)) == ["No veo botella."]


def test_implicit_distance_remembers_last_measurement_and_refreshes_reference():
    memory = TargetMemory()
    d = distance(memory)
    for t, label in [(1, "cup"), (40, None), (80, None)]:
        m = joint(
            t, t * 10, [("cup", -15, 1, 1.1, SPEAK), ("bicycle", 0, 2, 2.5, SPEAK)], measured=True
        )
        says = d.start(cmd(Mode.DISTANCE, label, t_s=t, request_id=t), View(m, m), t * SEC)
        assert texts(says) == ["Taza un poco a la izquierda, a 1,1 metros."]


def test_distance_reference_survives_loss_but_expires_with_configuration():
    memory = TargetMemory()
    h, c = search(memory=memory), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(
        h, c, [joint(0.1, 1, [("cup", -20, 1)]), joint(0.2, 2, [("cup", -20, 1)]), joint(3, 30, [])]
    )
    m = joint(10, 100, [("bottle", 0, 2, 1.0, SPEAK)], measured=True)
    assert texts(
        distance(memory).start(cmd(Mode.DISTANCE, None, t_s=10), View(m, m), 10 * SEC)
    ) == ["No veo taza."]
    settings = replace(S, target_memory_s=5)
    d = DistanceHandler(G, CATALOG.spoken, settings, memory)
    assert texts(d.start(cmd(Mode.DISTANCE, None, t_s=20), View(), 20 * SEC)) == []
    m = joint(20, 200, [("bottle", 0, 2, 1.0, SPEAK)], measured=True)
    assert texts(d.tick(cmd(Mode.DISTANCE, None, t_s=20), View(m, m), 20 * SEC)[0]) == [
        "Botella al frente, a 1 metro."
    ]


# --- distancia ---------------------------------------------------------------------------


def distance(memory: TargetMemory | None = None) -> DistanceHandler:
    return DistanceHandler(G, CATALOG.spoken, S, memory)


def test_distance_freezes_a_frame_after_the_order_never_an_older_one():
    h, c = distance(), cmd(Mode.DISTANCE, "cup", t_s=5.0, deadline_s=5)
    old = joint(4.0, 40, [("cup", 0, None, 2.0, SPEAK)], measured=True)  # antes de la orden
    assert texts(h.start(c, View(None, old), 5 * SEC)) == []
    says, done = h.tick(c, View(None, old), int(5.5 * SEC))
    assert (texts(says), done) == ([], False)
    new = joint(5.2, 52, [("cup", -15, None, 1.07, SPEAK)], measured=True)
    says, done = h.tick(c, View(None, new), int(6.0 * SEC))
    assert done and texts(says) == ["Taza un poco a la izquierda, a 1,1 metros."]
    says, done = h.tick(c, View(None, new), int(6.1 * SEC))
    assert (texts(says), done) == ([], True)  # una sola respuesta


def test_distance_without_valid_meters_explains_instead_of_guessing():
    h, c = distance(), cmd(Mode.DISTANCE, "bottle", t_s=1.0, deadline_s=5)
    m = joint(1.1, 11, [("bottle", 30, None, 1.3, DistancePolicy.DIRECTION_ONLY)], measured=True)
    says = h.start(c, View(None, m), int(1.2 * SEC))
    assert texts(says) == ["Botella a la derecha, pero no puedo medir su distancia."]
    h2, c2 = distance(), cmd(Mode.DISTANCE, "bottle", t_s=1.0, deadline_s=5, request_id=2)
    m2 = joint(1.1, 11, [("bottle", 0, None, 4.5, SPEAK)], measured=True)
    assert texts(h2.start(c2, View(None, m2), int(1.2 * SEC))) == [
        "Botella al frente, lejos, a más de 3 metros."
    ]


def test_distance_several_lists_them_unless_one_was_being_guided():
    two = [("cup", -20, None, 0.9, SPEAK), ("cup", 20, None, 2.0, SPEAK)]
    m = joint(1.1, 11, two, measured=True)
    h, c = distance(), cmd(Mode.DISTANCE, "cup", t_s=1.0)
    assert texts(h.start(c, View(None, m), int(1.2 * SEC))) == [
        "Veo 2 tazas: un poco a la izquierda, a 0,9 metros; un poco a la derecha, a 2 metros."
    ]
    memory = TargetMemory()
    sh = SearchHandler(G, CATALOG.spoken, S, memory)
    sc = cmd(Mode.SEARCH, "cup", request_id=7)
    sh.start(sc, View(), 0)
    run(sh, sc, [joint(0.5, 5, [("cup", 18, None)]), joint(0.6, 6, [("cup", 19, None)])])
    h2 = distance(memory)
    c2 = cmd(Mode.DISTANCE, None, t_s=1.0, request_id=8)
    assert texts(h2.start(c2, View(None, m), int(1.2 * SEC))) == [
        "Taza un poco a la derecha, a 2 metros."
    ]


def test_distance_without_object_measures_what_is_in_front():
    h, c = distance(), cmd(Mode.DISTANCE, None, t_s=1.0)
    m = joint(1.1, 11, [("chair", 40, None, 1.5, SPEAK), ("bottle", 3, None, 0.8, SPEAK)],
              measured=True)  # fmt: skip
    assert texts(h.start(c, View(None, m), int(1.2 * SEC))) == ["Botella al frente, a 0,8 metros."]
    h2, c2 = distance(), cmd(Mode.DISTANCE, None, t_s=1.0, request_id=2)
    m2 = joint(1.1, 11, [("chair", 40, None, 1.5, SPEAK)], measured=True)
    assert texts(h2.start(c2, View(None, m2), int(1.2 * SEC))) == [
        "No veo nada al frente que pueda medir."
    ]


def test_distance_timeout_and_absent_object():
    h, c = distance(), cmd(Mode.DISTANCE, "cup", t_s=1.0, deadline_s=5)
    h.start(c, View(), int(1.0 * SEC))
    seen = joint(3.2, 32, [("cup", -30, None)])
    says, done = h.tick(c, View(seen, None), int(3.3 * SEC))
    # La explicación larga no cabe en 5 s; el aviso breve conserva su sentido completo.
    assert done and texts(says) == ["No pude medir a tiempo."]
    h2, c2 = distance(), cmd(Mode.DISTANCE, "cup", t_s=1.0, deadline_s=5, request_id=2)
    m = joint(1.1, 11, [("bottle", 0, None, 1.0, SPEAK)], measured=True)
    assert texts(h2.start(c2, View(None, m), int(1.2 * SEC))) == ["No veo taza."]


# --- ubicación provisoria ----------------------------------------------------------------


def test_nearest_layer_ignores_sparse_noise_and_background_through_the_object():
    from visionx.app.spatial import nearest_layer

    rng = np.random.default_rng(0)
    chair = rng.normal(0.6, 0.02, 400)  # respaldo y asiento
    background = rng.normal(2.5, 0.05, 600)  # lo que se ve a través de la silla calada
    layer = nearest_layer(np.concatenate([chair, background]))
    assert layer is not None and abs(float(np.median(layer)) - 0.6) < 0.03
    noise = rng.normal(0.4, 0.01, 50)  # 5 % de píxeles sueltos delante: no es el objeto
    layer = nearest_layer(np.concatenate([noise, rng.normal(1.0, 0.02, 950)]))
    assert layer is not None and abs(float(np.median(layer)) - 1.0) < 0.03
    assert nearest_layer(np.array([], np.float32)) is None


def depth_from(frame: StereoFrame, z: np.ndarray):
    d, valid = G.depth_to_disparity(z)
    depth = FakeDepthEstimator(G).predict(frame)
    fields = {f: getattr(depth, f) for f in depth.__dataclass_fields__}
    fields.update(disparity_px=d, valid_mask=valid, depth_m=G.disparity_to_depth(d, valid)[0])
    return type(depth)(**fields)


def test_provisional_spatial_front_layer_border_and_diffuse():
    frame = joint(0.0, 0, []).frame
    z = np.full((G.height, G.width), 2.0, np.float32)
    z[:, 300:340] = 1.0  # la mitad izquierda de la máscara del medio está a 1 m
    z[:, 400:480] = np.linspace(1.0, 3.0, 80, dtype=np.float32)  # rampa: capa difusa
    masks = np.zeros((4, G.height, G.width), bool)
    masks[0, 100:200, 0:60] = True  # toca el borde izquierdo
    masks[1, 100:200, 300:380] = True  # frente a 1 m, fondo a 2 m (objeto calado)
    masks[2, 100:200, 400:480] = True  # superficies de 1 a 3 m sin separación
    masks[3, 100:200, 500:560] = True  # todo a 2 m
    boxes = [[0, 100, 60, 200], [300, 100, 380, 200], [400, 100, 480, 200], [500, 100, 560, 200]]
    det = build_detection(
        frame, model_id="t", inference_ms=1.0, class_names=["cup"] * 4,
        boxes_xyxy=np.array(boxes, np.float32), scores=np.full(4, 0.9, np.float32), masks=masks,
    )  # fmt: skip
    objs = ProvisionalSpatialEstimator(G).locate(frame, depth_from(frame, z), det)
    assert [o.policy for o in objs] == [
        DistancePolicy.ASK_REFRAME,
        DistancePolicy.SPEAK_METERS,
        DistancePolicy.DIRECTION_ONLY,
        DistancePolicy.SPEAK_METERS,
    ]
    assert objs[1].z_m == pytest.approx(1.0, abs=0.03)  # el frente, no el fondo
    assert objs[3].z_m == pytest.approx(2.0, abs=0.03)


def test_near_object_with_its_own_depth_still_gives_meters():
    # silla de cerca: superficies de 0,55 a 1,25 m (IQR ~0,35 m > 50 % de Z ~0,69 m) pero
    # dentro de la profundidad propia de un objeto (0,4 m): se informa su frente
    frame = joint(0.0, 0, []).frame
    z = np.full((G.height, G.width), 3.0, np.float32)
    z[:, 300:380] = np.linspace(0.55, 1.25, 80, dtype=np.float32)
    masks = np.zeros((1, G.height, G.width), bool)
    masks[0, 100:200, 300:380] = True
    det = build_detection(
        frame, model_id="t", inference_ms=1.0, class_names=["chair"],
        boxes_xyxy=np.array([[300, 100, 380, 200]], np.float32),
        scores=np.full(1, 0.9, np.float32), masks=masks,
    )  # fmt: skip
    (obj,) = ProvisionalSpatialEstimator(G).locate(frame, depth_from(frame, z), det)
    assert obj.policy is DistancePolicy.SPEAK_METERS
    assert obj.z_m == pytest.approx(0.69, abs=0.03)


# --- simulación y CLI --------------------------------------------------------------------


def test_parse_order():
    assert parse_order("12.5:buscá la taza") == (12.5, "buscá la taza")
    with pytest.raises(InvalidInput):
        parse_order("buscá la taza")


def test_simulation_cli_is_deterministic_and_answers(capsys, tmp_path):
    argv = ["simular", "--fuente", "sintetica", "--frames", "300", "--orden", "0.5:buscá la taza",
            "--orden", "6:a qué distancia está", "--latencia-detector", "0.2",
            "--latencia-profundidad", "1.0", "--traza", str(tmp_path / "t.jsonl")]  # fmt: skip
    assert modes_main(argv) == 0
    first = capsys.readouterr().out
    assert modes_main(argv) == 0
    second = capsys.readouterr().out
    assert first == second  # tiempo grabado: misma entrada, misma salida
    assert "sonido:bip" in first and "a 1,2 metros" in first
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"buscar-adquirido"' in x for x in lines)
    answers = [x for x in lines if '"distancia-respuesta"' in x]
    assert len(answers) == 1
    # el frame medido es posterior a la orden (t=6 s a 30 fps: frame >= 165 con 0,5 s de margen)
    assert int(answers[0].split('"frame_id": ')[1].split(",")[0]) >= 165


def test_demo_writes_a_video_without_voice(tmp_path, capsys):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")  # vienen con el extra de detección; CI puede no tenerlos
    out = tmp_path / "demo.mp4"
    argv = ["demo", "--fuente", "sintetica", "--frames", "60", "--orden", "0.3:buscá la taza",
            "--sin-voz", "--salida", str(out)]  # fmt: skip
    assert modes_main(argv) == 0
    produced = out if out.exists() else tmp_path / "demo-sin-audio.mp4"  # sin ffmpeg: mudo
    assert produced.stat().st_size > 0 and "Busco taza." in capsys.readouterr().out


def test_audio_schedule_queues_and_cancel_cuts():
    from visionx.modes.demo import AudioSchedule
    from visionx.modes.simulate import Spoken

    sched = AudioSchedule(None, 1000)
    sink = SimpleNamespace(spoken=[Spoken(0.0, 1, 1, "a" * 20), Spoken(0.1, 1, 1, "b" * 20)],
                           cancelled=[])  # fmt: skip
    sched.update(sink)
    first, second = sched.segments
    assert second.start_s == pytest.approx(first.end_s)  # en cola, no encima
    cut = first.end_s - 0.5
    sink.cancelled.append((cut, 1))  # una orden nueva cancela el pedido 1...
    sink.spoken.append(Spoken(cut, 2, 1, "nuevo"))
    sched.update(sink)
    # ...lo que sonaba se corta y lo pendiente ("b") se descarta
    assert [s.text for s in sched.segments] == ["a" * 20, "nuevo"]
    assert sched.segments[0].end_s == pytest.approx(cut)
    assert sched.segments[1].start_s == pytest.approx(cut)


# --- describir (T11): una escena congelada y números solo desde mediciones aceptadas -------


def describe(settings=S, options=None, **kwargs):
    # Estas pruebas aíslan composición y unión; la persistencia/default se prueba aparte.
    options = replace(
        options or DescribeSettings(wait_measure_s=3), persist_hits=1, max_speech_s=60
    )
    return DescribeHandler(G, CATALOG.spoken, settings, options, **kwargs)


def test_describe_groups_class_and_zone_with_catalog_plurals():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    m = joint(
        1.1,
        11,
        [
            ("cup", -20, 1, 0.54, SPEAK),
            ("cup", -10, 2, 0.8, SPEAK),
            ("chair", 0, 3, 1.63, SPEAK),
            ("person", 20, 4, None, DistancePolicy.DIRECTION_ONLY),
        ],
        measured=True,
    )
    assert texts(h.start(c, View(None, m), int(1.2 * SEC))) == [
        "Al frente, una silla a 1,6 metros.",
        "A la izquierda, dos tazas, la más cercana a 0,5 metros.",
        "A la derecha, una persona.",
    ]
    assert h.facts.frame_id == 11 and len(h.facts.measured) == 3
    assert h.tick(c, View(None, m), int(1.3 * SEC)) == ([], True)


@pytest.mark.parametrize(
    "policy,z",
    [
        (DistancePolicy.DIRECTION_ONLY, 1.0),
        (DistancePolicy.ABSTAIN, 1.0),
        (DistancePolicy.ASK_REFRAME, 1.0),
        (SPEAK, 0.49),
        (SPEAK, 3.01),
    ],
)
def test_describe_never_speaks_meters_without_policy_or_outside_range(policy, z):
    h = describe()
    m = joint(1.1, 11, [("cup", 0, None, z, policy)], measured=True)
    said = texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(None, m), int(1.2 * SEC)))
    assert said == ["Al frente, una taza."]
    assert not h.facts.measured


def test_describe_partial_group_preserves_meters_without_claiming_nearest():
    h = describe()
    m = joint(
        1.1,
        11,
        [("cup", -20, 1, 0.5, SPEAK), ("cup", -10, 2, None, DistancePolicy.DIRECTION_ONLY)],
        measured=True,
    )
    said = texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(None, m), int(1.2 * SEC)))
    assert said == ["A la izquierda, una taza a 0,5 metros y una taza."]


def test_describe_limits_individual_objects_and_counts_remaining():
    h = describe(options=DescribeSettings(max_objects=2))
    m = joint(
        1.1,
        11,
        [
            ("cup", -20, 1, 0.5, SPEAK),
            ("cup", -10, 2, 0.8, SPEAK),
            ("chair", 0, 3, 1.6, SPEAK),
            ("person", 20, 4, 2.0, SPEAK),
        ],
        measured=True,
    )
    said = texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(None, m), int(1.2 * SEC)))
    assert said == [
        "Al frente, una silla a 1,6 metros.",
        "A la izquierda, una taza a 0,5 metros, y 2 cosas más.",
    ]


def test_describe_prioritizes_people_and_large_boxes_when_no_distance():
    h = describe(options=DescribeSettings(max_objects=2))
    m = joint(
        1.1,
        11,
        [
            ("cup", -20, 1, None, DistancePolicy.DIRECTION_ONLY),
            ("chair", 0, 2, None, DistancePolicy.DIRECTION_ONLY),
            ("person", 20, 3, None, DistancePolicy.DIRECTION_ONLY),
        ],
        measured=True,
    )
    # Una silla con caja grande; el área es imagen ocupada, no una estimación de tamaño real.
    boxes = m.detection.boxes.copy()
    boxes[1] = [200, 60, 440, 300]
    det = build_detection(
        m.frame,
        model_id="test",
        boxes_xyxy=boxes,
        class_names=m.detection.classes,
        scores=m.detection.scores,
        inference_ms=1.0,
    )
    m = Joint(m.frame, det, m.depth, m.objects)
    said = texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(None, m), int(1.2 * SEC)))
    assert said == [
        "Al frente, una silla.",
        "A la derecha, una persona, y 1 cosa más.",
    ]


def test_describe_empty_scene():
    h = describe()
    m = joint(1.1, 11, [], measured=True)
    assert texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(None, m), int(1.2 * SEC))) == [
        "No reconozco objetos con suficiente confianza."
    ]


def test_describe_waits_for_post_order_joint_and_never_mixes_seen_and_depth():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    old = joint(0.99, 9, [("cup", 0, None, 0.5, SPEAK)], measured=True)
    assert h.start(c, View(None, old), SEC) == []  # aun dentro de accept_before_s: posterior
    seen = joint(1.2, 12, [("person", 20, None)])
    assert h.tick(c, View(seen, old), int(1.2 * SEC)) == ([], False)
    fresh = joint(1.1, 11, [("chair", -20, None, 1.63, SPEAK)], measured=True)
    says, done = h.tick(c, View(seen, fresh), int(1.3 * SEC))
    assert done and texts(says) == ["A la izquierda, una silla a 1,6 metros."]
    assert h.facts.frame_id == 11


def test_describe_timeout_uses_only_seen_even_if_it_has_objects():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(), SEC)
    seen = joint(3.9, 39, [("cup", -20, None, 0.54, SPEAK)], measured=True)
    says, done = h.tick(c, View(seen, None), 4 * SEC)
    assert done and texts(says) == [
        "No pude medir distancias.",
        "A la izquierda, una taza.",
    ]
    assert h.facts.measured == () and h.facts.depth_model_id is None


@pytest.mark.parametrize("frame_t", [0.9, 2.9, 4.1])
def test_describe_rejects_old_pre_order_or_future_seen(frame_t):
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(), SEC)
    seen = joint(frame_t, 1, [("cup", 0, None)])
    says, done = h.tick(c, View(seen, None), 4 * SEC)
    assert done and texts(says) == ["No pude obtener una imagen reciente para describir."]


def test_describe_hysteresis_keeps_center_while_waiting():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(joint(1, 10, [("cup", 7, 1)]), None), SEC)
    m = joint(1.2, 12, [("cup", 9.5, 1, 1.0, SPEAK)], measured=True)
    says, done = h.tick(c, View(m, m), int(1.3 * SEC))
    assert done and texts(says) == ["Al frente, una taza a 1 metro."]


def test_describe_deadline_and_old_request_never_speak():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1, deadline_s=2)
    h.start(c, View(), SEC)
    m = joint(2.9, 29, [("cup", 0, None, 0.5, SPEAK)], measured=True)
    assert h.tick(c, View(m, m), 3 * SEC) == ([], True)
    c2 = cmd(Mode.DESCRIBE, None, t_s=3, request_id=2)
    assert h.start(c2, View(), 3 * SEC) == []
    assert h.tick(c, View(m, m), 3 * SEC) == ([], True)


def test_describe_short_deadline_falls_back_before_expiring():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1, deadline_s=2.1)
    assert h.start(c, View(), SEC) == []
    says, done = h.tick(c, View(), int(1.1 * SEC))
    assert texts(says) == ["Sin imagen."]
    assert done and speech_s(texts(says)[0], S) <= 2
    assert h.tick(c, View(joint(2, 20, []), None), 2 * SEC) == ([], True)


def test_describe_controller_cancels_waiting_and_completed_narrations():
    clock = ManualClock(SEC)
    sink = RecordingSink(clock, lambda: SEC)
    state = [View()]
    controller = Controller(
        CommandParser(CATALOG),
        sink,
        latest_view=lambda: state[0],
        handlers=build_handlers({}, G, CATALOG.spoken),
        clock=clock,
        deadlines_s={"describe": 20},
    )
    controller.handle_text("describí")
    controller.handle_text("cancelar")
    clock.advance(SEC)
    m = joint(2, 20, [("cup", 0, None, 0.5, SPEAK)], measured=True)
    state[0] = View(m, m)
    controller.tick()
    assert [s.text for s in sink.spoken] == ["Describo.", "sonido:cancelado"]
    controller.handle_text("qué ves")
    clock.advance(int(0.2 * SEC))
    m = joint(2.2, 22, [("cup", 0, None, 0.5, SPEAK)], measured=True)
    state[0] = View(m, m)
    controller.tick()
    rid = sink.spoken[-1].request_id
    assert controller.active is None
    assert sink.spoken[-1].text == "Al frente, una taza a 0,5 metros."
    controller.handle_text("buscá la silla")
    assert sink.cancelled[-1] == (1.2, rid)  # cancela aunque el cálculo ya terminó


def test_describe_config_respects_depth_range_even_with_modes_override():
    h = build_handlers(
        {
            "depth": {"range_m": [1.0, 2.0]},
            "modes": {"range_m": [0.5, 3]},
            "describe": {"max_objects": 2},
        },
        G,
        CATALOG.spoken,
    )[Mode.DESCRIBE]
    assert h.s.range_m == (1.0, 2.0) and h.describe.max_objects == 2
    with pytest.raises(ValueError):
        DescribeSettings.from_config({"describe": {"typo": 3}})
    for options in (
        {"max_objects": 0},
        {"max_objects": 1.5},
        {"wait_measure_s": -1},
        {"wait_measure_s": float("nan")},
        {"large_area_fraction": 0},
    ):
        with pytest.raises(ValueError):
            DescribeSettings(**options)


def test_narrator_has_no_metric_text_channel_and_invalid_plans_are_rejected():
    assert TemplateNarrator().plan((PresenceFact(2, "cup", "al frente", 0.1),)) == ((2,),)

    class InvalidNarrator:
        def plan(self, ordered):
            return ((999,),)

    h = describe(narrator=InvalidNarrator())
    m = joint(1.1, 11, [("cup", 0, None, 0.5, SPEAK)], measured=True)
    with pytest.raises(ValueError, match="índices"):
        h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(m, m), int(1.2 * SEC))


def test_describe_stale_measurement_falls_back_to_current_presence():
    h, c = describe(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(), SEC)
    old = joint(1.01, 1, [("chair", 0, None, 1.0, SPEAK)], measured=True)
    seen = joint(4.1, 41, [("person", 20, None)])
    says, done = h.tick(c, View(seen, old), int(4.2 * SEC))
    assert done and texts(says) == [
        "No pude medir distancias.",
        "A la derecha, una persona.",
    ]
    assert not h.facts.measured


@pytest.mark.parametrize(
    "label,expected",
    [
        ("toothbrush", "cepillos de dientes"),
        ("baseball glove", "guantes de béisbol"),
        ("skis", "esquís"),
        ("broccoli", "brócolis"),
        ("snowboard", "snowboards"),
        ("laptop", "computadoras portátiles"),
        ("remote", "controles remotos"),
    ],
)
def test_describe_catalog_compound_plurals(label, expected):
    h = describe()
    m = joint(
        1.1,
        11,
        [
            (label, -20, 1, None, DistancePolicy.DIRECTION_ONLY),
            (label, -10, 2, None, DistancePolicy.DIRECTION_ONLY),
        ],
        measured=True,
    )
    said = texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(m, m), int(1.2 * SEC)))
    assert said == [f"A la izquierda, dos {expected}."]


def test_describe_simulation_records_frozen_facts_and_snapshots(tmp_path):
    import json

    trace = tmp_path / "trace.jsonl"
    captures = tmp_path / "frames"
    assert (
        modes_main(
            [
                "simular",
                "--fuente",
                "sintetica",
                "--frames",
                "100",
                "--orden",
                "0.5:describí",
                "--traza",
                str(trace),
                "--capturas",
                str(captures),
            ]
        )
        == 0
    )
    events = [json.loads(line) for line in trace.read_text(encoding="utf8").splitlines()]
    answers = [e for e in events if e["tipo"] == "describir-respuesta"]
    assert len(answers) == 1 and not answers[0]["solo_presencia"]
    frame_id = answers[0]["hechos"]["frame_id"]
    assert list(captures.glob(f"*-measured-{frame_id}.jpg"))
    assert "Esa función todavía no está disponible." not in trace.read_text(encoding="utf8")


def test_describe_lists_each_zone_once_after_selecting_relevant_objects():
    h = describe(options=DescribeSettings(max_objects=5))
    m = joint(
        1.1,
        11,
        [
            ("chair", 0, 1, 1.4, SPEAK),
            ("chair", 12, 2, 0.6, SPEAK),
            ("chair", 18, 3, 0.9, SPEAK),
            ("bottle", 24, 4, 2.1, SPEAK),
            ("laptop", -20, 5, None, DistancePolicy.DIRECTION_ONLY),
            ("cup", -10, 6, None, DistancePolicy.DIRECTION_ONLY),
        ],
        measured=True,
    )
    assert texts(h.start(cmd(Mode.DESCRIBE, None, t_s=1), View(m, m), int(1.2 * SEC))) == [
        "Al frente, una silla a 1,4 metros.",
        "A la izquierda, una computadora portátil.",
        "A la derecha, dos sillas, la más cercana a 0,6 metros y una botella a 2,1 metros, "
        "y 1 cosa más.",
    ]


@pytest.mark.parametrize(
    "labels,expected",
    [
        (["cup"], "una taza"),
        (["cup", "bottle"], "una taza y una botella"),
        (["cup", "bottle", "chair"], "una taza, una botella y una silla"),
        (["mouse", "mouse", "cup"], "dos mouses y una taza"),
        (["tv"], "una pantalla"),
        (["tv", "tv"], "dos pantallas"),
    ],
)
def test_describe_natural_lists_without_heading(labels, expected):
    h = describe()
    scene = SceneFacts(
        "s",
        11,
        tuple(PresenceFact(i, label, "a la derecha", 0.1) for i, label in enumerate(labels)),
        (),
        None,
    )
    assert h._render(scene) == [f"A la derecha, {expected}."]


def persistent(options=None):
    # Los casos temporales de R2 fijan tres segundos; la espera de producción va aparte.
    return DescribeHandler(
        G, CATALOG.spoken, describe=options or DescribeSettings(wait_measure_s=3)
    )


def test_describe_score_filter_applies_to_current_and_past_without_counting_doubtful_extras():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    objs = [
        ("cup", -20, 1, None, DistancePolicy.DIRECTION_ONLY),
        ("bottle", 0, 2, None, DistancePolicy.DIRECTION_ONLY),
        ("chair", 20, 3, None, DistancePolicy.DIRECTION_ONLY),
    ]
    first = joint(1, 10, objs, scores=[0.8, 0.26, 0.25])
    assert h.start(c, View(first), SEC) == []
    second = joint(1.2, 12, objs, measured=True, scores=[0.8, 0.8, 0.25])
    says, done = h.tick(c, View(second, second), int(1.3 * SEC))
    assert done and texts(says) == ["A la izquierda, una taza."]
    assert [f.label for f in h.facts.presence] == ["cup"]
    # El mismo frame débil sigue disponible para buscar; no se cambia detección ni candidatos.
    from visionx.modes.common import candidates

    assert len(candidates(second, None, G)) == 3


def test_describe_repeated_ticks_and_seen_measured_count_as_one_frame():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    m = joint(1, 10, [("cup", 0, 1, 0.8, SPEAK)], measured=True)
    assert h.start(c, View(m, m), SEC) == []
    for t in (1.1, 1.2, 1.3):
        assert h.tick(c, View(m, m), int(t * SEC)) == ([], False)
    assert h.facts is None
    newer = joint(1.5, 15, [("cup", 0, 1, 0.9, SPEAK)], measured=True)
    says, done = h.tick(c, View(newer, newer), int(1.6 * SEC))
    assert done and texts(says) == ["Al frente, una taza a 0,9 metros."]


@pytest.mark.parametrize("kind", ["old", "future", "other-session", "other-class", "other-id"])
def test_describe_persistence_excludes_unrelated_frames(kind):
    previous = joint(1, 10, [("cup", 0, 1)])
    if kind == "other-class":
        previous = joint(1, 10, [("bottle", 0, 1)])
    if kind == "other-id":
        previous = joint(1, 10, [("cup", 0, 2)])
    if kind == "other-session":
        previous = replace(previous, frame=replace(previous.frame, session_id="another"))
    if kind == "future":
        previous = joint(3, 30, [("cup", 0, 1)])
    c = cmd(Mode.DESCRIBE, None, t_s=1.1)
    h = persistent(DescribeSettings(persist_s=1))
    h.start(c, View(previous), int(1.1 * SEC))
    t = 2.2 if kind == "old" else 1.2
    current = joint(t, 22, [("cup", 0, 1, 0.8, SPEAK)], measured=True)
    assert h.tick(c, View(current, current), int((t + 0.1) * SEC)) == ([], False)
    assert not h._confirmed(current)


def test_describe_future_observation_cannot_confirm_older_measurement():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    old = joint(1.1, 11, [("cup", 0, 1, 0.8, SPEAK)], measured=True)
    future = joint(1.2, 12, [("cup", 0, 1)])
    assert h.start(c, View(future, old), int(1.3 * SEC)) == []
    assert not h._confirmed(old)


def test_describe_persistence_without_tracking_matches_instances_one_to_one():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    first = joint(1, 10, [("cup", 11, None)])
    h.start(c, View(first), SEC)
    current = joint(
        1.2,
        12,
        [
            ("cup", 10, None, None, DistancePolicy.DIRECTION_ONLY),
            ("cup", 12, None, None, DistancePolicy.DIRECTION_ONLY),
        ],
        measured=True,
    )
    # Las cajas casi se tocan; ambas solapan con la previa, pero no son duplicados entre sí.
    says, done = h.tick(c, View(current, current), int(1.3 * SEC))
    assert done and texts(says) == ["A la derecha, una taza."]
    assert len(h.facts.presence) == 1


def test_describe_three_hits_and_new_order_resets_persistence():
    h = persistent(DescribeSettings(persist_hits=3))
    c = cmd(Mode.DESCRIBE, None, t_s=1)
    first = joint(1, 10, [("cup", 0, 1)])
    h.start(c, View(first), SEC)
    second = joint(1.2, 12, [("cup", 0, 1, None, DistancePolicy.DIRECTION_ONLY)], measured=True)
    assert h.tick(c, View(second, second), int(1.3 * SEC)) == ([], False)
    third = joint(1.4, 14, [("cup", 0, 1, None, DistancePolicy.DIRECTION_ONLY)], measured=True)
    assert h.tick(c, View(third, third), int(1.5 * SEC))[1]
    assert (
        h.start(cmd(Mode.DESCRIBE, None, t_s=1.5, request_id=2), View(third), int(1.5 * SEC)) == []
    )
    assert len(h._history) == 1


def test_describe_default_duration_preserves_nearest_and_omission_count():
    from visionx.modes.common import speech_s

    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    objs = [
        ("cup", -20, 1, 0.5, SPEAK),
        ("bottle", -10, 2, 1.0, SPEAK),
        ("laptop", 0, 3, 1.5, SPEAK),
        ("chair", 12, 4, 2.0, SPEAK),
        ("person", 20, 5, 2.5, SPEAK),
        ("banana", 24, 6, 0.6, SPEAK),
    ]
    h.start(c, View(joint(1, 10, objs, scores=[0.8] * 5 + [0.3])), SEC)
    m = joint(1.2, 12, objs, measured=True, scores=[0.8] * 5 + [0.3])
    says, done = h.tick(c, View(m, m), int(1.3 * SEC))
    result = texts(says)
    assert done and sum(speech_s(t, S) for t in result) <= 8
    assert len(h.facts.presence) == 5
    assert "una taza a 0,5 metros" in " ".join(result)
    assert "cosas más" in result[-1]
    assert "banana" not in " ".join(result) and "Veo" not in " ".join(result)


def test_describe_timeout_abstains_when_score_never_reaches_threshold():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(joint(1, 10, [("cup", 0, 1)], scores=[0.26])), SEC)
    m = joint(3.9, 39, [("cup", 0, 1, 0.8, SPEAK)], measured=True, scores=[0.26])
    says, done = h.tick(c, View(m, m), 4 * SEC)
    assert done and texts(says) == ["No reconozco objetos con suficiente confianza."]
    assert not h.facts.presence and not h.facts.measured


def test_describe_presence_fallback_includes_warning_in_duration_budget():
    from visionx.modes.common import speech_s

    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    objs = [("cup", 0, 1), ("chair", 12, 2), ("bottle", -12, 3)]
    h.start(c, View(), SEC)
    assert h.tick(c, View(joint(3.5, 35, objs)), int(3.5 * SEC)) == ([], False)
    says, done = h.tick(c, View(joint(3.9, 39, objs)), 4 * SEC)
    result = texts(says)
    assert done and result[0] == "No pude medir distancias."
    assert sum(speech_s(t, S) for t in result) <= 8
    assert not h.facts.measured and "metros" not in " ".join(result)


@pytest.mark.parametrize(
    "options",
    [
        {"min_score": -0.1},
        {"min_score": 1.1},
        {"min_score": float("nan")},
        {"persist_hits": 0},
        {"persist_hits": True},
        {"persist_hits": 1.5},
        {"persist_hits": 257},
        {"persist_s": 0},
        {"persist_s": float("inf")},
        {"max_speech_s": -1},
        {"max_speech_s": float("nan")},
    ],
)
def test_describe_filter_config_validation(options):
    with pytest.raises(ValueError):
        DescribeSettings(**options)


@pytest.mark.parametrize(
    "scores,confidence,phrase",
    [
        ([0.8, 0.8], "high", "Al frente, una taza a 0,8 metros."),
        ([0.4, 0.4], "cautious", "Al frente, parece haber una taza."),
        ([0.4, 0.8], "cautious", "Al frente, parece haber una taza."),
        ([0.8, 0.4], "cautious", "Al frente, parece haber una taza."),
    ],
)
def test_describe_each_level_requires_current_and_previous_score(scores, confidence, phrase):
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    obj = [("cup", 0, 1, 0.8, SPEAK)]
    assert h.start(c, View(joint(1, 10, obj, scores=[scores[0]])), SEC) == []
    m = joint(1.2, 12, obj, scores=[scores[1]], measured=True)
    says, done = h.tick(c, View(m, m), int(1.3 * SEC))
    assert done and texts(says) == [phrase]
    assert h.facts.presence[0].confidence == confidence
    assert bool(h.facts.measured) == (confidence == "high")


def test_describe_never_abstains_with_only_persistent_medium_candidates():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    objects = [("sink", -20, None, 1.5, SPEAK), ("refrigerator", 20, 2, 2.0, SPEAK)]
    h.start(c, View(joint(1, 10, objects, scores=[0.3, 0.4])), SEC)
    m = joint(1.2, 12, objects, measured=True, scores=[0.3, 0.4])
    says, done = h.tick(c, View(m, m), int(1.3 * SEC))
    assert done and texts(says) == [
        "A la izquierda, parece haber una pileta.",
        "A la derecha, parece haber una heladera.",
    ]
    assert not h.facts.measured


def test_describe_medium_candidates_still_require_two_distinct_frames_at_timeout():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    m = joint(3.8, 38, [("sink", 0, 1, 1.5, SPEAK)], scores=[0.3], measured=True)
    h.start(c, View(m, m), int(3.8 * SEC))
    says, done = h.tick(c, View(m, m), 4 * SEC)
    assert done and texts(says) == ["No reconozco objetos con suficiente confianza."]


def test_describe_mixed_confidence_never_groups_or_assigns_meters_to_medium():
    from visionx.modes.describe import MetricFact

    h = describe()
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "cup", "al frente", 0.1, "high"),
            PresenceFact(1, "cup", "al frente", 0.1, "cautious"),
            PresenceFact(2, "sink", "al frente", 0.1, "cautious"),
        ),
        (MetricFact(0, 0.8), MetricFact(1, 0.5)),
        "fake",
    )
    assert h._render(scene) == [
        "Al frente, una taza a 0,8 metros; parece haber una taza y una pileta."
    ]


def test_describe_rejects_narrator_merging_high_and_medium_instances():
    class MixedNarrator:
        def plan(self, ordered):
            return ((0, 1),)

    h = describe(narrator=MixedNarrator())
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "cup", "al frente", 0.1, "high"),
            PresenceFact(1, "cup", "al frente", 0.1, "cautious"),
        ),
        (),
        None,
    )
    with pytest.raises(ValueError, match="confianza"):
        h._render(scene)


@pytest.mark.parametrize(
    "label,score,accepted",
    [
        ("sink", 0.3, True),
        ("refrigerator", 0.3, True),
        ("oven", 0.6, False),
        ("oven", 0.7, True),
        ("bed", 0.32, False),
        ("bed", 0.4, True),
        ("laptop", 0.45, False),
        ("laptop", 0.55, True),
    ],
)
def test_describe_calibrated_class_floors_apply_to_both_observations(label, score, accepted):
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    obj = [(label, 0, 1, 1.5, SPEAK)]
    h.start(c, View(joint(1, 10, obj, scores=[score])), SEC)
    m = joint(1.2, 12, obj, scores=[score], measured=True)
    says, done = h.tick(c, View(m, m), 4 * SEC)
    assert done and bool(h.facts.presence) == accepted
    assert ("No reconozco" not in " ".join(texts(says))) == accepted


@pytest.mark.parametrize(
    "options",
    [
        {"cautious_score": -0.1},
        {"cautious_score": 0.7},
        {"cautious_score": float("nan")},
        {"cautious_min_scores": []},
        {"cautious_min_scores": {"sink": float("nan")}},
        {"cautious_min_scores": {"sink": 1.1}},
    ],
)
def test_describe_confidence_settings_are_validated(options):
    with pytest.raises(ValueError):
        DescribeSettings(**options)


def test_describe_default_wait_allows_second_observation_after_three_seconds():
    h = DescribeHandler(G, CATALOG.spoken)
    c = cmd(Mode.DESCRIBE, None, t_s=1)
    assert h.start(c, View(), SEC) == []
    first = joint(3.5, 35, [("tv", -20, 1)], scores=[0.45])
    assert h.tick(c, View(first), int(3.8 * SEC)) == ([], False)
    second = joint(
        4.2, 42, [("tv", -20, 1, None, DistancePolicy.DIRECTION_ONLY)], measured=True, scores=[0.7]
    )
    says, done = h.tick(c, View(second, second), int(4.4 * SEC))
    assert done and texts(says) == ["A la izquierda, parece haber una pantalla."]
    assert not h.facts.measured


def test_describe_budget_keeps_anonymous_count_when_removing_a_name():
    from visionx.modes.common import speech_s

    h = DescribeHandler(G, CATALOG.spoken, describe=DescribeSettings(max_objects=2, max_speech_s=3))
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "cup", "al frente", 0.1),
            PresenceFact(1, "chair", "al frente", 0.1),
            PresenceFact(2, "bottle", "al frente", 0.01),
        ),
        (),
        None,
    )
    result = h._render(scene)
    assert result == ["Al frente, una taza, y 2 cosas más."]
    assert sum(speech_s(p, S) for p in result) <= 3


@pytest.mark.parametrize(
    "budget, expected",
    [
        (
            8,
            "una taza a 0,5 metros, una botella a 1 metro y una silla a 1,5 metros; "
            "parece haber un gato.",
        ),
        (
            6.5,
            "una taza a 0,5 metros, una botella a 1 metro y una silla a 1,5 metros, y 1 cosa más.",
        ),
        (6, "una taza a 0,5 metros, una botella a 1 metro y una silla, y 1 cosa más."),
        (5, "una taza a 0,5 metros, una botella y una silla, y 1 cosa más."),
        (4.5, "una taza a 0,5 metros y una botella, y 2 cosas más."),
        (4, "una taza a 0,5 metros, y 3 cosas más."),
    ],
)
def test_describe_budget_cuts_cautious_then_secondary_meters_then_names(budget, expected):
    from visionx.modes.common import speech_s

    h = DescribeHandler(G, CATALOG.spoken, describe=DescribeSettings(max_speech_s=budget))
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "cup", "al frente", 0.1),
            PresenceFact(1, "bottle", "al frente", 0.1),
            PresenceFact(2, "chair", "al frente", 0.1),
            PresenceFact(3, "cat", "al frente", 0.1, "cautious"),
        ),
        (MetricFact(0, 0.5), MetricFact(1, 1.0), MetricFact(2, 1.5)),
        "B0",
    )
    result = h._render(scene)
    assert result == ["Al frente, " + expected]
    assert sum(speech_s(p, S) for p in result) <= budget


@pytest.mark.parametrize("budget", [9, 10, 12])
def test_describe_budget_keeps_nearest_meters_in_each_zone_while_they_fit(budget):
    from visionx.modes.common import speech_s

    h = DescribeHandler(G, CATALOG.spoken, describe=DescribeSettings(max_speech_s=budget))
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "cup", "al frente", 0.1),
            PresenceFact(1, "bottle", "a la izquierda", 0.1),
            PresenceFact(2, "chair", "a la derecha", 0.1),
            PresenceFact(3, "keyboard", "al frente", 0.1),
            PresenceFact(4, "tv", "al frente", 0.1),
        ),
        tuple(MetricFact(i, z) for i, z in enumerate((0.5, 1.0, 2.0, 0.7, 0.9))),
        "B0",
    )
    result = h._render(scene)
    said = " ".join(result)
    assert "una taza a 0,5 metros" in said
    assert "una botella a 1 metro" in said
    assert "una silla a 2 metros" in said
    assert sum(speech_s(p, S) for p in result) <= budget
    if budget == 9:
        assert "teclado" not in said and "pantalla" not in said and "y 2 cosas más" in said


@pytest.mark.parametrize("max_objects", [1, 2, 8])
def test_describe_nearest_uses_exact_z_before_rounded_relevance_or_area(max_objects):
    h = DescribeHandler(
        G, CATALOG.spoken, describe=DescribeSettings(max_objects=max_objects, max_speech_s=4)
    )
    scene = SceneFacts(
        "s",
        10,
        (
            PresenceFact(0, "person", "al frente", 0.8),
            PresenceFact(1, "cup", "al frente", 0.2),
            PresenceFact(2, "bottle", "al frente", 0.01),
        ),
        (MetricFact(0, 0.54), MetricFact(1, 0.53), MetricFact(2, 0.51)),
        "B0",
    )
    assert h._render(scene) == ["Al frente, una botella a 0,5 metros, y 2 cosas más."]


def test_describe_impossibly_small_override_never_silences_nearest_meters():
    h = DescribeHandler(G, CATALOG.spoken, describe=DescribeSettings(max_speech_s=1))
    scene = SceneFacts(
        "s", 10, (PresenceFact(0, "cup", "al frente", 0.1),), (MetricFact(0, 0.5),), "B0"
    )
    assert h._render(scene) == ["Al frente, una taza a 0,5 metros."]


def test_describe_medium_match_cannot_leave_orphan_high_metrics():
    h, c = persistent(), cmd(Mode.DESCRIBE, None, t_s=1)
    h.start(c, View(joint(1, 10, [("cup", 11, None)], scores=[0.8])), SEC)
    m = joint(
        1.2,
        12,
        [
            ("cup", 10, None, 0.8, SPEAK),
            ("cup", 11.2, None, 0.9, SPEAK),
        ],
        measured=True,
        scores=[0.8, 0.4],
    )
    says, done = h.tick(c, View(m, m), int(1.3 * SEC))
    assert done and texts(says) == ["A la derecha, parece haber una taza."]
    assert len(h.facts.presence) == 1 and not h.facts.measured


def test_guidance_does_not_repeat_small_distance_changes():
    """Metros nuevos se dicen solos; un cambio chico (o el borde del rango) no se repite y
    perder la medida no es noticia. Un cambio de dirección sí vuelve a la frase entera."""
    h, c = search(), cmd(Mode.SEARCH, "cup", deadline_s=60)
    h.start(c, View(), 0)
    first, _ = run(h, c, [joint(0.1 * i, i, [("cup", 0, None)]) for i in range(1, 4)])
    assert "Taza al frente." in [t for _, t in first]

    def at(t, z=None, h_deg=0):
        m = None
        if z is not None:
            m = joint(t - 0.1, int(t * 10) - 1, [("cup", h_deg, None, z, SPEAK)], measured=True)
        live = joint(t, int(t * 10), [("cup", h_deg, None)])
        return texts(h.tick(c, View(live, m), int(t * SEC))[0])

    assert at(2.0, 2.9) == ["A 2,9 metros."]
    assert at(3.4, 3.1) == []  # borde del rango (2,9 → 3,1 «más de 3»): no es noticia
    assert at(4.8) == []  # sin medida nueva: nada
    assert at(5.0, 2.2) == ["A 2,2 metros."]  # cambió 0,85 m
    assert at(6.4, 2.1) == []  # 0,1 m: redondeo, no se repite
    assert at(7.8, 2.1, h_deg=-30) == ["A la izquierda."]  # otra dirección: frase entera


def test_far_meters_are_approximate_up_to_the_range_end():
    """Entre approx_from_m y range_m[1] los metros se dicen aproximados («unos 4,5 metros»);
    más allá, «lejos, a más de 5 metros» (el límite no se aproxima)."""
    from visionx.modes.common import Measure, measure_phrase, meters

    far = dataclasses.replace(S, range_m=(0.5, 5.0), approx_from_m=3.0, approx_round_m=0.5)
    assert meters(2.94, far) == "2,9 metros"
    assert meters(3.0, far) == "3 metros"
    assert meters(4.3, far) == "unos 4,5 metros"
    assert meters(3.2, far) == "unos 3 metros"
    assert measure_phrase(Measure(4.0, None), far) == "a unos 4 metros"
    assert measure_phrase(Measure(6.2, "muy-lejos"), far) == "lejos, a más de 5 metros"


def test_direction_change_while_speaking_cuts_and_replaces_the_old_phrase():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", -30, 7)]), joint(0.2, 2, [("cup", -30, 7)])])
    # la cámara gira rápido: 0,6 s después la taza ya está a la derecha
    says, _ = h.tick(c, View(joint(0.8, 8, [("cup", 30, 7)])), int(0.8 * SEC))
    assert texts(says) == ["A la derecha."] and says[0].replace


def test_no_replacement_before_preempt_min_or_without_a_change():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", -30, 7)]), joint(0.2, 2, [("cup", -30, 7)])])
    # cambió, pero hace menos de preempt_min_s que empezó la frase: no se corta todavía
    assert texts(h.tick(c, View(joint(0.3, 3, [("cup", 30, 7)])), int(0.3 * SEC))[0]) == []
    h2, c2 = search(), cmd(Mode.SEARCH, "cup", request_id=2)
    h2.start(c2, View(), 0)
    run(h2, c2, [joint(0.1, 1, [("cup", -30, 7)]), joint(0.2, 2, [("cup", -30, 7)])])
    # misma dirección mientras suena: nada
    assert texts(h2.tick(c2, View(joint(0.9, 9, [("cup", -31, 7)])), int(0.9 * SEC))[0]) == []


class FakeAssistant:
    """Asistente sin red: guarda los pedidos y responde con frases ya listas."""

    def __init__(self, sentences=("Adelante tenés una silla.",), error=None, available=True):
        from visionx.assistant import AssistantJob

        self.calls, self.available = [], available
        self._job_cls, self._sentences, self._error = AssistantJob, sentences, error

    def submit(self, request_id, kind, image, **kw):
        job = self._job_cls(request_id, kind)
        job.sentences = [(0.5 * (i + 1), s) for i, s in enumerate(self._sentences)]
        job.text, job.error = " ".join(self._sentences), self._error
        if self._error:
            job.sentences = []
        job.done.set()
        self.calls.append((kind, kw))
        return job


def test_describe_with_assistant_uses_measured_meters_and_the_mode_clock():
    a = FakeAssistant(("Adelante tenés una silla a 1,6 metros.", "A tu derecha, una persona."))
    h, c = describe(assistant=a), cmd(Mode.DESCRIBE, None, t_s=1)
    m = joint(1.1, 11, [("chair", 0, 3, 1.63, SPEAK), ("person", 30, 4, None,
                                                       DistancePolicy.DIRECTION_ONLY)],
              measured=True)  # fmt: skip
    assert texts(h.start(c, View(None, m), int(1.2 * SEC))) == []  # consultando
    kind, kw = a.calls[0]
    assert kind == "describir" and "silla al frente, a 1,6 metros" in kw["objects"]
    assert texts(h.tick(c, View(None, m), int(1.6 * SEC))[0]) == []  # todavía no pasó 0,5 s
    says, done = h.tick(c, View(None, m), int(2.3 * SEC))
    assert texts(says) == ["Adelante tenés una silla a 1,6 metros.", "A tu derecha, una persona."]
    assert done


def test_describe_falls_back_to_template_when_the_assistant_fails():
    a = FakeAssistant(error="ConnectionError: sin red")
    h, c = describe(assistant=a), cmd(Mode.DESCRIBE, None, t_s=1)
    m = joint(1.1, 11, [("chair", 0, 3, 1.63, SPEAK)], measured=True)
    h.start(c, View(None, m), int(1.2 * SEC))
    says, done = h.tick(c, View(None, m), int(1.3 * SEC))
    assert texts(says) == ["Al frente, una silla a 1,6 metros."] and done


def test_free_question_goes_to_the_assistant_with_the_question():
    a = FakeAssistant(("Sí, la silla de adelante está libre.",))
    h = describe(assistant=a)
    c = cmd(Mode.DESCRIBE, "¿Hay algún lugar para sentarme?", t_s=1)
    seen = joint(1.05, 10, [("chair", 0, 3)])
    assert texts(h.start(c, View(seen, None), int(1.1 * SEC))) == []
    kind, kw = a.calls[0]
    assert kind == "pregunta" and kw["request"] == "¿Hay algún lugar para sentarme?"
    says, done = h.tick(c, View(seen, None), int(1.7 * SEC))
    assert texts(says) == ["Sí, la silla de adelante está libre."] and done
    off = describe(assistant=FakeAssistant(available=False))
    said = texts(off.start(cmd(Mode.DESCRIBE, "¿Qué es?", t_s=1), View(seen, None), int(1.1 * SEC)))
    assert said == ["Ahora no puedo responder preguntas: no tengo conexión."]


def test_lateral_and_body_height_phrases():
    from visionx.modes.common import body_height_phrase, lateral_phrase

    assert lateral_phrase(1.1, -15, S) == "unos 30 centímetros a tu izquierda"
    assert lateral_phrase(1.1, 3, S) == "al frente"  # menos de side_min_m: no se dice
    assert lateral_phrase(2.0, -35, S) == "1,4 metros a tu izquierda"
    assert body_height_phrase(1.1, 0) == "a la altura de tu pecho"
    assert body_height_phrase(1.1, -25) == "a la altura de tu cintura"
    assert body_height_phrase(3.0, -25) == "cerca del piso"


def test_distance_with_assistant_passes_measured_numbers_and_waits_one_sentence():
    a = FakeAssistant(("Tenés una taza blanca sobre un taburete, a 1,1 metros, al frente.",))
    h = DistanceHandler(G, CATALOG.spoken, S, assistant=a)
    c = cmd(Mode.DISTANCE, "cup", t_s=1)
    m = joint(1.1, 11, [("cup", 0, 3, 1.07, SPEAK)], measured=True)
    assert texts(h.start(c, View(None, m), int(1.2 * SEC))) == []  # consultando
    kind, kw = a.calls[0]
    assert kind == "distancia"
    assert kw["objects"] == ["taza: a 1,1 metros; al frente; a la altura de tu pecho"]
    says, done = h.tick(c, View(None, m), int(1.8 * SEC))
    assert texts(says) == ["Tenés una taza blanca sobre un taburete, a 1,1 metros, al frente."]
    assert done


def test_distance_falls_back_to_the_plain_answer_when_the_assistant_fails():
    h = DistanceHandler(G, CATALOG.spoken, S, assistant=FakeAssistant(error="sin red"))
    c = cmd(Mode.DISTANCE, "cup", t_s=1)
    m = joint(1.1, 11, [("cup", 0, 3, 1.07, SPEAK)], measured=True)
    h.start(c, View(None, m), int(1.2 * SEC))
    says, done = h.tick(c, View(None, m), int(1.3 * SEC))
    assert texts(says) == ["Taza al frente, a 1,1 metros."] and done


def test_arrival_cuts_the_guidance_that_is_still_playing():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    run(h, c, [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])])
    near = joint(0.3, 3, [("cup", 0, None, 0.55, SPEAK)], measured=True)
    says, done = h.tick(c, View(joint(0.4, 4, [("cup", 0, None)]), near), int(0.4 * SEC))
    assert done and says[0].text_or_sound.startswith("Llegaste") and says[0].replace


def test_sonar_cues_pan_to_the_side_and_speed_up_when_centered():
    from visionx.modes.sonar import Sonar, SonarSettings, beep, cue_for

    s = SonarSettings()
    left, right, ahead = cue_for(-40, s), cue_for(40, s), cue_for(2, s)
    assert left[0] < 0 < right[0] and ahead[1] > left[1]  # lado y tono
    assert ahead[2] < left[2]  # al frente, más seguido
    stereo = beep(-1.0, 880, s, 24_000)
    assert stereo.shape[1] == 2 and abs(stereo[:, 1]).max() < 1e-6  # todo a la izquierda
    sonar = Sonar(s)
    cues = [sonar.step(0.05 * i, (2.0, 0.1)) for i in range(40)]
    assert sum(c is not None for c in cues) >= 12  # ~cada 0,12 s al frente
    assert sonar.step(3.0, (2.0, 2.0)) is None  # no se lo ve hace 2 s: silencio


def test_sonar_style_speaks_only_key_moments():
    sonar = ModeSettings()  # por defecto: sonar
    assert sonar.guide_style == "sonar"
    h, c = search(sonar), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    # gira de la izquierda al frente: con palabras habría varias frases de dirección
    frames = [joint(0.1 * i, i, [("cup", -40 + i, None)]) for i in range(1, 60)]
    out, _ = run(h, c, frames)
    said = [t for _, t in out]
    assert said == ["sonido:bip", "Taza a la izquierda."]  # solo encontrarla
    assert h.sonar_target(int(5.9 * SEC))[0] > -1  # el sonar sabe dónde está
