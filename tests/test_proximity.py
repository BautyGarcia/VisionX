"""Alerta de cercanía (app/proximity.py) con mapas de profundidad sintéticos."""

from types import SimpleNamespace

import numpy as np

from visionx.app.proximity import ProximityAlert, ProximitySettings
from visionx.shared.contracts import DepthStatus
from visionx.shared.geometry import StereoGeometry

SEC = 1_000_000_000
G = StereoGeometry(640, 360, 300.0, 300.0, 320.0, 180.0, 0.06)


def view(z_center, t_s, frame_id, *, floor=False, status=DepthStatus.VALID):
    depth = np.full((360, 640), 3.0, np.float32)
    if z_center is not None:
        depth[60:270, 200:440] = z_center
    if floor:  # piso cercano en las filas de abajo de la zona: no es obstáculo
        rows = np.arange(360, dtype=np.float32)[:, None]
        depth[:] = np.where(rows > 200, 1.3 * 300 / np.maximum(rows - 180, 1), 3.0)
    frame = SimpleNamespace(received_ns=int(t_s * SEC), key=("s", frame_id))
    d = SimpleNamespace(depth_m=depth, status=status)
    return SimpleNamespace(measured=SimpleNamespace(frame=frame, depth=d))


def run(alert, frames):
    said = []
    for z, t, i in frames:
        said += [s.text_or_sound for s in alert.observe(view(z, t, i), int(t * SEC))]
    return said


def test_wall_close_ahead_alerts_once_after_hold():
    a = ProximityAlert(ProximitySettings(enabled=True), geometry=G)
    said = run(a, [(0.6, 0.1 * i, i) for i in range(1, 12)])
    assert said == ["sonido:bip-grave", "Cuidado, algo muy cerca adelante."]
    assert abs(a.last_z_m - 0.6) < 1e-6


def test_far_or_floor_does_not_alert():
    a = ProximityAlert(ProximitySettings(enabled=True), geometry=G)
    assert run(a, [(1.5, 0.1 * i, i) for i in range(1, 12)]) == []
    floor = ProximityAlert(ProximitySettings(enabled=True), geometry=G)
    said = [s for i in range(1, 12) for s in floor.observe(view(None, 0.1 * i, i, floor=True),
                                                           int(0.1 * i * SEC))]  # fmt: skip
    assert said == []


def test_stale_depth_and_silent_modes_suspend_the_alert():
    a = ProximityAlert(ProximitySettings(enabled=True), geometry=G)
    stale = [a.observe(view(0.5, 0.1 * i, i), int((0.1 * i + 2) * SEC)) for i in range(1, 12)]
    assert not any(stale)
    reading = ProximityAlert(ProximitySettings(enabled=True), geometry=G, active_mode="read")
    assert run(reading, [(0.4, 0.1 * i, i) for i in range(1, 12)]) == []


def test_rearms_only_after_clearing_and_respects_cooldown():
    a = ProximityAlert(ProximitySettings(enabled=True, cooldown_s=2.0), geometry=G)
    first = run(a, [(0.6, 0.1 * i, i) for i in range(1, 8)])
    still = run(a, [(0.6, 1 + 0.1 * i, 10 + i) for i in range(1, 20)])  # sigue cerca: no repite
    cleared = run(a, [(2.0, 3 + 0.1 * i, 40 + i) for i in range(1, 5)])
    again = run(a, [(0.6, 4 + 0.1 * i, 60 + i) for i in range(1, 8)])
    assert len(first) == 2 and still == [] and cleared == [] and len(again) == 2


def test_disabled_by_default_and_quiet_after_an_order_ends():
    assert not ProximitySettings().enabled
    a = ProximityAlert(ProximitySettings(enabled=True), geometry=G, active_mode="search")
    run(a, [(2.0, 0.1, 1)])
    a.active_mode = None  # terminó la búsqueda («Llegaste»)
    assert run(a, [(0.5, 0.2 + 0.1 * i, 2 + i) for i in range(1, 20)]) == []  # < quiet_after_s
    assert run(a, [(0.5, 6 + 0.1 * i, 40 + i) for i in range(1, 8)]) != []
