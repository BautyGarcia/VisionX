"""En qué corre cada modelo, para mostrarlo en el stand («CREStereo en Radeon RX 5700 XT»).

Se calcula una vez al arrancar: mirar la placa de video cuesta un llamado al sistema.
"""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def nombre_gpu() -> str | None:
    """Nombre de la placa de video según el sistema (o None)."""
    try:
        if platform.system() == "Windows":
            consulta = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"
            cmd = ["powershell", "-NoProfile", "-Command", consulta]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout
            nombres = [x.strip() for x in out.splitlines() if x.strip()]
            # Hay adaptadores virtuales (pantallas remotas, capturadoras): la placa real es la
            # de un fabricante conocido.
            reales = [x for x in nombres if any(m in x for m in ("Radeon", "NVIDIA", "GeForce",
                                                                  "Arc", "Intel"))]  # fmt: skip
            elegido = (reales or nombres or [""])[0]
            return elegido.replace("AMD ", "").replace("NVIDIA ", "") or None
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout.strip()
            return out or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _crestereo(cfg: dict[str, Any]) -> str:
    sec = (cfg.get("depth") or {}).get("crestereo") or {}
    pedido = str(sec.get("device", "auto"))
    try:
        import torch

        if pedido in ("auto", "mps") and torch.backends.mps.is_available():
            return f"{nombre_gpu() or 'GPU Apple'} (MPS)"
        if pedido in ("auto", "cuda") and torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} (CUDA)"
    except ImportError:
        return "CPU"
    if pedido in ("auto", "directml"):
        from visionx.depth.crestereo_adapt.onnx_dml import CARPETA, dml_disponible

        sha = str(sec.get("weights_sha256", ""))[:8]
        if dml_disponible() and any((CARPETA / sha).glob("*.onnx")):
            return f"{nombre_gpu() or 'GPU'} (DirectML)"
    return "CPU"


def _yolo(cfg: dict[str, Any]) -> str:
    pedido = str((cfg.get("detection") or {}).get("device", "cpu"))
    if pedido.startswith("mps"):
        return "GPU Apple (MPS)"
    if pedido.startswith("cuda"):
        return "GPU (CUDA)"
    return "CPU"


def describir(
    cfg: dict[str, Any], profundidad: str | None = None, detector: str | None = None
) -> str:
    """«CREStereo en Radeon RX 5700 XT (DirectML) · YOLO en CPU», con lo que corre de verdad."""
    if profundidad is None or "crestereo" in profundidad.lower():
        prof = f"CREStereo en {_crestereo(cfg)}"
    else:
        prof = f"profundidad «{profundidad}» en CPU"
    det = f"YOLO en {_yolo(cfg)}" if detector in (None, "yolo") else f"detector «{detector}»"
    return f"{prof} · {det}"
