"""Tests de los modos buscar/guiar y distancia (T09), sin cámara ni pesos.

Las uniones se arman a mano (detecciones y objetos ubicados con ángulo, Z y política elegidos)
para forzar cada caso: presencia de un solo frame, pérdida, iguales, profundidad vieja, etc.
El reloj es manual: los tiempos son exactos y no dependen de la máquina.
"""

from __future__ import annotations

import math

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
from visionx.modes.common import ModeSettings, direction, measure, measure_phrase, meters
from visionx.modes.distance import DistanceHandler
from visionx.modes.search import SearchHandler
from visionx.modes.simulate import RecordingSink
from visionx.modes.target import TargetMemory, plural
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
S = ModeSettings()
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
    assert "Taza un poco a la izquierda." in said and "Taza al frente." in said
    times = [t for t, x in out if not x.startswith("sonido")]
    assert all(b - a >= S.min_interval_s - 1e-9 for a, b in zip(times, times[1:], strict=False))
    assert len(out) < 8  # 59 frames, pocas frases


def test_repeats_direction_after_interval_if_nothing_changes():
    h, c = search(), cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    out, _ = run(h, c, [joint(0.1 * i, i, [("cup", -30, None)]) for i in range(1, 60)])
    phrases = [t for t, x in out if x == "Taza a la izquierda."]
    assert len(phrases) == 2 and phrases[1] - phrases[0] == pytest.approx(S.repeat_s, abs=0.11)


def test_lost_target_is_announced_and_reappearance_is_a_new_acquisition():
    events = []
    h = SearchHandler(G, CATALOG.spoken, S, None, lambda k, d: events.append((k, d)))
    c = cmd(Mode.SEARCH, "cup")
    h.start(c, View(), 0)
    seen = [joint(0.1 * i, i, [("cup", -30, 5)]) for i in range(1, 4)]
    gone = [joint(0.3 + 0.1 * i, 3 + i, []) for i in range(1, 25)]  # 2,4 s sin la taza
    back = [joint(2.8 + 0.1 * i, 30 + i, [("cup", 25, 9)]) for i in range(1, 4)]
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
    out, _ = run(h, c, [joint(0.1 * i, i, two) for i in range(1, 100)])
    asked = [t for t, x in out if "¿Cuál" in x]
    assert len(asked) == 2 and asked[1] - asked[0] == pytest.approx(2 * S.repeat_s, abs=0.11)
    assert h.state == "eligiendo" and h.target is None


def test_duplicate_box_of_the_same_object_is_not_a_second_object():
    # una caja de silla dentro de otra (YOLO detectó una parte como otra silla)
    objs = [("chair", 0, None), ("chair", 0.5, None), ("chair", 30, None)]
    j = joint(0.1, 1, objs, scores=[0.8, 0.3, 0.8])
    from visionx.modes.common import candidates

    cands = candidates(j, "chair", G)
    assert [c.index for c in cands] == [0, 2]


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
    says, _ = h.tick(c, View(joint(11.2, 112, [("cup", 0, None)]), m2b), int(11.2 * SEC))
    assert texts(says) == ["Taza al frente, a 1,4 metros."]
    m3 = joint(11.3, 113, [("cup", 0, None, 0.55, SPEAK)], measured=True)
    says, done = h.tick(c, View(joint(11.4, 114, [("cup", 0, None)]), m3), int(11.4 * SEC))
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
    says, done = h.tick(c, View(), int(9.5 * SEC))
    assert done and texts(says) == ["No encontré taza."]
    h2, c2 = search(), cmd(Mode.SEARCH, "cup", deadline_s=10, request_id=2)
    h2.start(c2, View(), 0)
    run(h2, c2, [joint(0.1, 1, [("cup", 0, None)]), joint(0.2, 2, [("cup", 0, None)])])
    says, done = h2.tick(c2, View(), int(9.5 * SEC))
    assert done and texts(says) == ["Terminé de buscar taza."]


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
    seen = joint(4.9, 49, [("cup", -30, None)])
    says, done = h.tick(c, View(seen, None), int(5.0 * SEC))
    assert done and texts(says) == [
        "Veo taza a la izquierda, pero no pude medir la distancia a tiempo."
    ]
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
