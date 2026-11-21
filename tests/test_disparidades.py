"""`gtsynth disparidades`: guarda una vez por vista y saltea lo que ya está."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from visionx.gtsynth import disparidades as D


class _Matcher:
    def __init__(self):
        self.llamados = 0
        self.last_extras = {}

    def reset(self):
        pass

    def compute(self, izq, der):
        self.llamados += 1
        h, w = izq.shape[:2]
        self.last_extras = {"vertical": np.zeros((h, w), np.float32), "half": None}
        return np.full((h, w), 12.5, np.float32)


def _vista(k):
    img = np.zeros((40, 64, 3), np.uint8)
    g = SimpleNamespace(fx=50.0, baseline_m=0.06, width=64, height=40)
    return SimpleNamespace(escena="casa", meta={"indice": k}, geometria_nominal=g,
                           izq=lambda: img, der=lambda: img, recortar=lambda x: x)  # fmt: skip


def test_calcula_y_despues_saltea(tmp_path, monkeypatch):
    m = _Matcher()
    params = SimpleNamespace(tag="i4p2")
    monkeypatch.setattr(D, "_matcher", lambda cfg: (m, params, "abcd1234"))
    import visionx.gtsynth.dataset as ds

    monkeypatch.setattr(ds, "vistas", lambda raiz, var: [_vista(0), _vista(1)])
    r = D.calcular("raiz", ["var"], tmp_path, cfg={}, log=lambda *_: None)
    assert r == {"calculadas": 2, "salteadas": 0} and m.llamados == 2
    with np.load(tmp_path / "var" / "casa" / "0001.npz") as d:
        assert d["disparidad"].dtype == np.float16 and float(d["disparidad"][0, 0]) == 12.5
        meta = json.loads(str(d["meta"]))
        assert meta["fx"] == 50.0 and meta["pesos"] == "abcd1234" and "mitad" not in d
    r = D.calcular("raiz", ["var"], tmp_path, cfg={}, log=lambda *_: None)
    assert r == {"calculadas": 0, "salteadas": 2} and m.llamados == 2
