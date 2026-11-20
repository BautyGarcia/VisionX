"""Pantalla del stand: conversación, tiempos, rótulos e incertidumbre."""

from __future__ import annotations

import numpy as np

from visionx.app.stand import (
    LiveLog,
    StandState,
    is_ack,
    level_of,
    model_name,
    render_stand,
    uncertainty_colormap,
)
from visionx.modes.common import ModeSettings
from visionx.shared.contracts import DistancePolicy, SpatialObject

S = ModeSettings()


def obj(z=1.4, bajo=None, alto=None):
    return SpatialObject(
        session_id="s", frame_id=1, detection_index=0, track_id=None, label="cup",
        depth_model_id="m", angle_h_deg=0.0, angle_v_deg=0.0, z_m=z, z_spread_m=0.0,
        valid_fraction=1.0, detector_score=0.9, policy=DistancePolicy.SPEAK_METERS,
        z_bajo_m=bajo, z_alto_m=alto,
    )  # fmt: skip


def test_conversacion_agrupa_tramos_y_no_repite_la_guia():
    log = LiveLog()
    log.feed("orden", {"request_id": 1, "texto": "Buscá la botella."})
    for texto in ("Busco botella.", "Veo botella.", "Girá a la izquierda.", "Girá a la izquierda."):
        log.feed("audio", {"request_id": 1, "texto": texto, "started_ns": 5})
    turns, _ = log.snapshot()
    assert [(t.who, t.text) for t in turns] == [
        ("vos", "Buscá la botella."),
        ("visionx", "Veo botella. Girá a la izquierda."),
    ]


def test_tiempos_desde_soltar_el_boton():
    log = LiveLog()
    log.feed("boton-soltado", {})
    log.feed("stt", {"texto": "¿A qué distancia está la taza?", "ms": 640})
    log.feed("orden", {"request_id": 7, "texto": "¿A qué distancia está la taza?"})
    log.feed("asistente-respuesta", {"request_id": 7, "primer_texto_s": 0.6})
    log.feed("audio", {"request_id": 7, "texto": "sonido:ok", "started_ns": 10})
    import time

    log.feed("audio", {"request_id": 7, "texto": "La taza está a 1,4 metros.",
                       "started_ns": time.monotonic_ns()})  # fmt: skip
    turns, t = log.snapshot()
    assert t.stt_ms == 640 and t.claude_ms == 600
    assert t.first_word_ms is not None and t.first_word_ms >= 0
    assert [x.who for x in turns] == ["vos", "visionx"]  # la orden dicha no se duplica


def test_confirmaciones_y_nombres():
    assert is_ack("Busco taza.") and is_ack("Leo.") and not is_ack("Veo taza.")
    assert model_name("crestereo-2271ab61-1280x720-i4p2") == "CREStereo"
    assert model_name("sgbm-b0") == "SGBM B0"


def test_rotulos_por_nivel():
    assert level_of(obj(1.43), S) == ("precisa", "1,4 m")
    assert level_of(obj(2.3, 1.9, 2.75), S) == ("aproximada", "~2,5 m")
    assert level_of(obj(2.3, 1.6, None), S) == ("no-segura", "¿?")


def test_mapa_de_incertidumbre():
    from visionx.depth.stereo import build_result
    from visionx.shared.contracts import StereoFrame, UncertaintyKind
    from visionx.shared.geometry import StereoGeometry

    g = StereoGeometry(64, 36, 40.0, 40.0, 31.5, 17.5, 0.06)
    frame = StereoFrame(session_id="s", frame_id=1, left=np.zeros((36, 64, 3), np.uint8),
                        right=np.zeros((36, 64, 3), np.uint8), received_ns=0, rectified=True,
                        calibration_id="c", geometry_id=g.geometry_id)  # fmt: skip
    d = np.full((36, 64), 4.0, np.float32)
    b = np.full_like(d, 0.05)
    b[:, 32:] = 1.0  # mitad derecha muy incierta
    depth = build_result(frame, "m", g, d, np.ones_like(d, bool), 1.0, 0.0, uncertainty=b,
                         uncertainty_kind=UncertaintyKind.LAPLACE_B)  # fmt: skip
    img = uncertainty_colormap(depth, S)
    assert img.shape == (36, 64, 3)
    left, right = img[:, :32].mean(axis=(0, 1)), img[:, 32:].mean(axis=(0, 1))
    assert left[1] > left[2] and right[2] > right[1]  # verde a la izquierda, rojo a la derecha


def test_pantalla_sin_camara_no_falla():
    from visionx.app.stand import Turn

    st = StandState(joint=None, statuses=(), geometry=None, frame_geometry=None,
                    range_m=(0.5, 3.0), turns=[Turn("vos", "¿Qué hay adelante?", 0)])  # fmt: skip
    img = render_stand(st)
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
