"""CREStereo en una placa sin CUDA ni MPS: ONNX Runtime con DirectML (Windows, AMD o Intel).

PyTorch en Windows solo acelera NVIDIA, y `torch-directml` se cae dentro de CREStereo. La red
exportada a ONNX sí corre con DirectML: en una RX 5700 XT, 0,10 s por cuadro en caliente
(640×384) contra 1,8 s en la CPU, y 0,51 s el modo preciso contra 9,4 s; la diferencia con
PyTorch es de 0,002–0,004 px (BITACORA 20/11).

ONNX necesita **tamaños fijos**: un grafo por combinación de alto × ancho y tamaño del flujo
previo. Para el modo de dos pasadas hacen falta tres:

    i{iters}_{H}x{W}_sin           primera pasada, sin flujo previo (a media resolución)
    i{iters}_{H}x{W}_{h}x{w}       segunda pasada, con el flujo de la primera
    i{iters}_{H}x{W}_{H}x{W}       en caliente, con el flujo del cuadro anterior

Se exportan una vez (`python -m visionx.depth.crestereo_adapt.onnx_dml exportar`, con un
PyTorch que tenga el exportador clásico, por ejemplo el venv de `torch-directml`) a
`outputs/onnx/crestereo/<pesos>/`. `CreStereoOnnx` se usa como el modelo de PyTorch: si falta
el grafo de un tamaño, ese llamado corre en PyTorch (CPU) y lo avisa una vez.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

from visionx.shared.config import ROOT

CARPETA = ROOT / "outputs" / "onnx" / "crestereo"


def dml_disponible() -> bool:
    """ONNX Runtime con DirectML instalado (paquete `onnxruntime-directml`)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


def nombre(iters: int, alto: int, ancho: int, init: tuple[int, int] | None) -> str:
    previo = "sin" if init is None else f"{init[0]}x{init[1]}"
    return f"i{iters}_{alto}x{ancho}_{previo}.onnx"


class CreStereoOnnx:
    """Hace de `nets.Model`: `modelo(izq, der, iters=, flow_init=)` -> tensor (1, 2, H, W)."""

    def __init__(self, carpeta: Path, respaldo: Any) -> None:
        self.carpeta = Path(carpeta)
        self.respaldo = respaldo  # el modelo de PyTorch en CPU, para los tamaños sin grafo
        self._sesiones: dict[str, Any] = {}
        self._avisados: set[str] = set()

    def _sesion(self, archivo: str):
        if archivo not in self._sesiones:
            path = self.carpeta / archivo
            if not path.is_file():
                self._sesiones[archivo] = None
            else:
                import onnxruntime as ort

                opciones = ort.SessionOptions()
                opciones.log_severity_level = 3
                provs = ["DmlExecutionProvider", "CPUExecutionProvider"]
                self._sesiones[archivo] = ort.InferenceSession(str(path), opciones, providers=provs)
        return self._sesiones[archivo]

    def __call__(self, izq, der, iters: int = 4, flow_init=None, **_):
        import torch

        alto, ancho = int(izq.shape[-2]), int(izq.shape[-1])
        init = None if flow_init is None else (int(flow_init.shape[-2]), int(flow_init.shape[-1]))
        archivo = nombre(iters, alto, ancho, init)
        sesion = self._sesion(archivo)
        if sesion is None:
            if archivo not in self._avisados:
                self._avisados.add(archivo)
                warnings.warn(f"CREStereo ONNX: falta {archivo}; ese tamaño corre en la CPU",
                              stacklevel=2)  # fmt: skip
            return self.respaldo(izq, der, iters=iters, flow_init=flow_init)
        entradas = {"izq": izq.detach().cpu().numpy(), "der": der.detach().cpu().numpy()}
        if flow_init is not None:
            entradas["flujo"] = flow_init.detach().cpu().numpy()
        return torch.from_numpy(sesion.run(None, entradas)[0])

    # Lo que el resto del código le pide al modelo de PyTorch.
    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self


def exportar(modelo: Any, carpeta: Path, alto: int, ancho: int, iters: int = 4) -> list[Path]:
    """Los tres grafos de un tamaño de trabajo (alto × ancho ya rellenado a múltiplo de 32)."""
    import torch

    class Envoltura(torch.nn.Module):
        def __init__(self, m, con_init: bool) -> None:
            super().__init__()
            self.m, self.con_init = m, con_init

        def forward(self, izq, der, flujo=None):
            return self.m(izq, der, iters=iters, flow_init=flujo if self.con_init else None)

    carpeta.mkdir(parents=True, exist_ok=True)
    h, w = alto // 2, ancho // 2
    casos = [
        ((h, w), None),  # primera pasada (media resolución)
        ((alto, ancho), (h, w)),  # segunda pasada
        ((alto, ancho), (alto, ancho)),  # en caliente
    ]
    hechos = []
    for (ah, aw), init in casos:
        path = carpeta / nombre(iters, ah, aw, init)
        if path.is_file():
            hechos.append(path)
            continue
        args = [torch.rand(1, 3, ah, aw) * 255, torch.rand(1, 3, ah, aw) * 255]
        nombres = ["izq", "der"]
        if init is not None:
            args.append(torch.zeros(1, 2, *init))
            nombres.append("flujo")
        with torch.no_grad():
            torch.onnx.export(Envoltura(modelo, init is not None), tuple(args), str(path),
                              opset_version=17, input_names=nombres, output_names=["salida"],
                              do_constant_folding=True)  # fmt: skip
        hechos.append(path)
        print(f"exportado {path.name}")
    return hechos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exportar CREStereo a ONNX para DirectML")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("exportar", help="los grafos de los modos de la config")
    ex.add_argument("--tamanos", nargs="*", default=None,
                    help="ALTOxANCHO ya rellenados (por defecto, continuo y preciso)")  # fmt: skip
    args = ap.parse_args(argv)

    from visionx.depth.crestereo_adapt.matcher import CreStereoParams, padded_size
    from visionx.depth.crestereo_adapt.port import PortConfig, load_model
    from visionx.shared.config import load_config

    cfg = load_config()["depth"]["crestereo"]
    port = PortConfig.from_config({**cfg, "device": "cpu"})
    modelo, _ = load_model(port)
    carpeta = CARPETA / port.weights_sha256[:8]
    tamanos: list[tuple[int, int, int]] = []
    if args.tamanos:
        for t in args.tamanos:
            a, b = (int(x) for x in t.lower().split("x"))
            tamanos.append((a, b, 4))
    else:
        for modo in ("continuo", "preciso"):
            p = CreStereoParams.from_config(cfg[modo])
            w, h = (int(x) for x in cfg[modo]["compute"])
            ph, pw = padded_size(h, w, p.multiple)
            tamanos.append((ph, pw, p.iters))
    for alto, ancho, iters in tamanos:
        exportar(modelo, carpeta, alto, ancho, iters)
    print(f"grafos en {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CARPETA", "CreStereoOnnx", "dml_disponible", "exportar", "nombre"]
