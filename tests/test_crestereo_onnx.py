"""Modo ONNX + DirectML de CREStereo: nombres de los grafos y respaldo en PyTorch sin grafo."""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from visionx.depth.crestereo_adapt.onnx_dml import CreStereoOnnx, nombre  # noqa: E402


def test_nombres_de_los_tres_grafos():
    assert nombre(4, 192, 320, None) == "i4_192x320_sin.onnx"
    assert nombre(4, 384, 640, (192, 320)) == "i4_384x640_192x320.onnx"
    assert nombre(4, 384, 640, (384, 640)) == "i4_384x640_384x640.onnx"


def test_sin_grafo_usa_pytorch_y_avisa_una_vez(tmp_path):
    llamados = []

    def respaldo(izq, der, iters=4, flow_init=None):
        llamados.append((tuple(izq.shape), iters, flow_init is None))
        return torch.zeros(1, 2, *izq.shape[-2:])

    m = CreStereoOnnx(tmp_path, respaldo)
    x = torch.zeros(1, 3, 64, 96)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        out = m(x, x, iters=4, flow_init=None)
        m(x, x, iters=4, flow_init=None)
    assert out.shape == (1, 2, 64, 96)
    assert llamados == [((1, 3, 64, 96), 4, True)] * 2
    assert sum("i4_64x96_sin.onnx" in str(a.message) for a in avisos) == 1
    assert m.to("cpu") is m and m.eval() is m
