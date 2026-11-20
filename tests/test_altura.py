"""La altura de la persona y de la cámara: «a la altura de tu cintura» para cada cuerpo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from visionx.app.floor import FloorTracker, fit_floor
from visionx.modes import body
from visionx.modes.common import body_height_phrase
from visionx.shared.geometry import StereoGeometry

G = StereoGeometry(640, 360, 300.0, 300.0, 319.5, 179.5, 0.06)


@pytest.fixture(autouse=True)
def _limpio():
    body.reset()
    yield
    body.reset()


def piso(cam_h: float, pitch_deg: float, g: StereoGeometry = G) -> np.ndarray:
    """Z axial de un piso plano visto desde una cámara a cam_h, inclinada pitch_deg."""
    v, u = np.mgrid[0 : g.height, 0 : g.width].astype(np.float64)
    dy = -(v - g.cy) / g.fy  # rayo con z = 1, y hacia arriba
    p = math.radians(pitch_deg)
    up = dy * math.cos(p) + math.sin(p)
    with np.errstate(divide="ignore"):
        z = np.where(up < -1e-3, -cam_h / up, np.inf)
    return np.where(z < 8, z, np.nan).astype(np.float32)


@pytest.mark.parametrize("cam_h,pitch", [(1.25, -15.0), (1.10, 0.0), (1.45, -25.0)])
def test_piso_da_altura_e_inclinacion_de_la_camara(cam_h, pitch):
    z = piso(cam_h, pitch)
    fit = fit_floor(z, np.isfinite(z), G.fx, G.fy, G.cx, G.cy)
    assert fit is not None
    assert fit.camera_height_m == pytest.approx(cam_h, abs=0.01)
    assert fit.pitch_deg == pytest.approx(pitch, abs=0.5)


def test_sin_piso_no_inventa():
    pared = np.full((G.height, G.width), 2.0, np.float32)  # todo a 2 m: una pared
    assert fit_floor(pared, np.ones_like(pared, bool), G.fx, G.fy, G.cx, G.cy) is None


def test_tracker_aplica_solo_si_es_estable_y_olvida_sin_piso():
    t = FloorTracker(keep_s=2.0)
    z = piso(1.30, -10.0)
    t.update(z, np.isfinite(z), G, 0.0)
    assert not body.current().floor_seen  # un solo ajuste no alcanza
    for k in (1, 2):
        t.update(z, np.isfinite(z), G, 0.1 * k)
    b = body.current()
    assert b.floor_seen and b.camera_m == pytest.approx(1.30, abs=0.02)
    pared = np.full_like(z, 2.0)
    t.update(pared, np.ones_like(pared, bool), G, 1.0)
    assert body.current().floor_seen  # se conserva un rato
    t.update(pared, np.ones_like(pared, bool), G, 3.5)
    assert not body.current().floor_seen and body.current().camera_height_m is None


def test_tracker_no_aplica_ajustes_que_saltan():
    t = FloorTracker()
    for k, (h, p) in enumerate([(0.9, 0.0), (1.4, -25.0), (0.7, 5.0), (1.3, -20.0)]):
        z = piso(h, p)
        t.update(z, np.isfinite(z), G, 0.1 * k)
    assert not t.stable and not body.current().floor_seen  # vuelve a la proporción


def test_las_bandas_dependen_de_la_estatura():
    # Un objeto a 1,1 m del piso, a 2 m adelante, cámara nivelada en el pecho.
    for h, esperado in [(1.50, "a la altura de tu pecho"), (1.90, "a la altura de tu cintura")]:
        body.set_height(h, persist=False)
        b = body.current()
        v = math.degrees(math.atan2(1.1 - b.camera_m, 2.0))
        assert body_height_phrase(2.0, v) == esperado


def test_la_inclinacion_cambia_la_altura_del_objeto():
    body.set_height(1.75, persist=False)
    body.set_camera(1.26, -20.0, True)  # mirando 20° hacia abajo
    # Al centro de la imagen, a 2 m: está abajo de la cámara (la cámara apunta al piso).
    assert body.current().object_height_m(2.0, 0.0) == pytest.approx(
        1.26 - 2.0 * math.sin(math.radians(20)), abs=1e-6
    )
    assert body_height_phrase(2.0, 0.0) == "a la altura de tus rodillas"  # 0,58 m del piso
    body.set_camera(1.26, 0.0, True)  # la misma cámara nivelada diría otra cosa
    assert body_height_phrase(2.0, 0.0) == "a la altura de tu pecho"


@pytest.mark.parametrize(
    "frase,altura",
    [("Mido 1,80.", 1.80), ("mido 175", 1.75), ("Mi altura es 1.62", 1.62),
     ("Mido un metro ochenta y cinco", 1.85), ("Yo mido un metro setenta", 1.70)],
)  # fmt: skip
def test_parsear_la_altura(frase, altura):
    assert body.is_height_statement(frase)
    assert body.parse_height(frase) == pytest.approx(altura)


def test_no_confunde_otras_ordenes():
    assert not body.is_height_statement("¿A qué distancia está la taza?")
    assert body.parse_height("mido 40") is None  # fuera de rango


def test_orden_de_voz_guarda_la_altura_y_confirma(tmp_path, monkeypatch):
    from visionx.app.controller import Controller
    from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
    from visionx.voice.commands import CommandParser
    from visionx.voice.output import AudioOutput, FakePlayer

    class Synth:
        sample_rate = 1000

        def synthesize(self, text):
            return np.zeros(len(text), np.float32)

    monkeypatch.setattr(body.Path, "home", lambda: tmp_path)
    out = AudioOutput(Synth(), FakePlayer())
    ctl = Controller(CommandParser(ClassCatalog(COCO_CLASSES)), out)
    ctl.handle_text("Mido 1,80.")
    out.wait_idle(2)
    assert [r.text_or_sound for r in out.records] == [
        "Listo, medís 1,8 metros. Lo uso para decirte a qué altura están las cosas."
    ]
    assert body.current().height_m == 1.80 and body.current().height_known
    body.reset()
    assert body.load().height_m == 1.80  # quedó guardada
    out.close()
