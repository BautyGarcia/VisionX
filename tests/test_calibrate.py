"""«Calibrar distancia» (app/calibrate.py): intérprete, controlador y método opcional."""

from types import SimpleNamespace

import numpy as np

from visionx.app.calibrate import build_calibrator, parse_meters
from visionx.app.controller import Controller
from visionx.detection.catalog import COCO_CLASSES, ClassCatalog
from visionx.shared.clock import ManualClock
from visionx.voice.commands import CommandParser

CATALOG = ClassCatalog(COCO_CLASSES)


def test_parse_meters_in_digits_words_and_centimeters():
    assert parse_meters("Calibrá, está a 1,5 metros.") == 1.5
    assert parse_meters("calibrar a 150 centímetros") == 1.5
    assert parse_meters("Calibrá, está a dos metros y medio") == 2.5
    assert parse_meters("calibrá a un metro y medio") == 1.5
    assert parse_meters("calibrá a medio metro") == 0.5
    assert parse_meters("calibrá") is None


def test_parser_turns_calibrate_into_a_control_order():
    p = CommandParser(CATALOG)
    i = p.parse("Calibrá, está a 2 metros.")
    assert (i.mode.value, i.target, i.rule) == ("control", "calibrar:2", "calibrar")
    assert p.parse("Calibrá.").reply.startswith("Decime a cuántos metros")


class Sink:
    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)

    def cancel(self, request_id):
        pass

    def cancel_all(self):
        pass


def _view(angle_ok=True):
    det = SimpleNamespace(masks=np.ones((1, 4, 4), bool))
    frame = SimpleNamespace(received_ns=10**9)
    joint = SimpleNamespace(frame=frame, detection=det)
    return SimpleNamespace(measured=joint)


def test_controller_calls_the_estimator_method_when_it_exists(monkeypatch):
    import visionx.app.calibrate as cal

    cand = SimpleNamespace(label="cup", angle_h_deg=2.0, index=0)
    monkeypatch.setattr(cal, "candidates", lambda joint, label, geometry: [cand])
    monkeypatch.setattr(cal, "age_s", lambda joint, now: 0.1)
    calls = []
    depth = SimpleNamespace(calibrar_distancia=lambda frame, mask, z: calls.append(z) or True)
    view = _view()
    sink = Sink()
    ctl = Controller(
        CommandParser(CATALOG), sink, latest_view=lambda: view, clock=ManualClock(10**9),
        calibrate=build_calibrator(depth, None, CATALOG.spoken),
    )  # fmt: skip
    ctl.handle_text("Calibrá, está a 1,5 metros.")
    assert calls == [1.5]
    assert [e.text_or_sound for e in sink.events] == ["Listo, calibré con taza a 1,5 metros."]


def test_without_the_method_the_app_says_so():
    sink = Sink()
    ctl = Controller(
        CommandParser(CATALOG), sink, latest_view=lambda: None, clock=ManualClock(0),
        calibrate=build_calibrator(SimpleNamespace(), None, CATALOG.spoken),
    )  # fmt: skip
    ctl.handle_text("Calibrá a 2 metros.")
    assert "no se calibra" in sink.events[0].text_or_sound


def test_dos_distancias_corrigen_escala_y_corrimiento_en_la_app():
    """«Calibrá, está a 2 metros» y después a 4: queda exacta a cualquier distancia."""
    import numpy as np

    from visionx.app.distancia_viva import CorreccionDistancia, ProfundidadCorregida
    from visionx.shared.contracts import DepthResult, DepthStatus

    class Est:
        geometry = SimpleNamespace(fx=100.0, baseline_m=0.1)  # f·B = 10

        def predict(self, frame):  # cámara con f·B 4 % corrido y la unidad 0,5 px corrida
            d = np.full((60, 90), 1.04 * 10.0 / frame.z + 0.5, np.float32)
            return DepthResult(session_id="s", frame_id=frame.frame_id, model_id="m",
                               geometry_id="g", disparity_px=d, valid_mask=np.ones(d.shape, bool),
                               status=DepthStatus.VALID, inference_ms=1.0,
                               depth_m=(10.0 / d).astype(np.float32))  # fmt: skip

    guardadas = []
    prof = ProfundidadCorregida(Est(), CorreccionDistancia(), guardadas.append)

    def z_medida(z, fid):
        return float(np.median(prof.predict(SimpleNamespace(frame_id=fid, z=z)).depth_m))

    assert abs(z_medida(2.0, 1) - 2.0) > 0.05  # sin calibrar, errada
    assert prof.calibrar_distancia(SimpleNamespace(frame_id=1), None, 2.0)
    assert prof.correccion.modo == "corrimiento" and "un paso" in prof.sugerencia
    assert abs(z_medida(2.0, 2) - 2.0) < 1e-3  # a esa distancia ya está
    assert abs(z_medida(4.0, 3) - 4.0) > 0.02  # a otra, todavía no
    assert prof.calibrar_distancia(SimpleNamespace(frame_id=3), None, 4.0)
    assert prof.correccion.modo == "ambas" and len(guardadas) == 2
    for k, z in enumerate((0.8, 1.5, 3.0, 5.0)):
        assert abs(z_medida(z, 10 + k) - z) < 1e-3 * z
    assert not prof.calibrar_distancia(SimpleNamespace(frame_id=999), None, 2.0)  # sin cuadro
    otra = CorreccionDistancia.de_dict(prof.correccion.a_dict())
    assert otra.modo == "ambas" and abs(otra.escala - 1.04) < 1e-3
