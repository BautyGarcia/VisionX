"""Ubicar y cargar CREStereo: el código del port y los pesos publicados.

Nada de esto viaja en Git. El port a PyTorch (ibaiGorordo/CREStereo-Pytorch) no declara
licencia, así que se descarga en `third_party/crestereo/` con un commit fijo; los pesos van a
`weights/crestereo/`. Los dos se controlan antes de usarlos: el commit por el archivo `COMMIT`
que deja el script de descarga, los pesos por SHA-256.

La red queda **congelada**: modo evaluación y sin gradientes. Nunca se entrena desde acá.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionx.shared.config import ROOT
from visionx.shared.errors import EstimatorError

HINT = "corré `uv run python scripts/profundidad/descargar_crestereo.py`"


@dataclass(frozen=True)
class PortConfig:
    """Dónde están el código y los pesos, y contra qué se verifican."""

    port_dir: Path
    port_commit: str
    weights: Path
    weights_sha256: str
    device: str = "auto"
    max_disp: int = 256

    @classmethod
    def from_config(cls, section: dict[str, Any], root: Path = ROOT) -> PortConfig:
        return cls(
            port_dir=root / section["port_dir"],
            port_commit=str(section["port_commit"]),
            weights=root / section["weights"],
            weights_sha256=str(section["weights_sha256"]),
            device=str(section.get("device", "auto")),
            max_disp=int(section.get("max_disp", 256)),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(cfg: PortConfig) -> None:
    """Falla con un mensaje útil si falta algo o no es la versión fijada."""
    marker = cfg.port_dir / "COMMIT"
    if not (cfg.port_dir / "nets" / "crestereo.py").exists() or not marker.exists():
        raise EstimatorError(f"falta el código de CREStereo en {cfg.port_dir}: {HINT}")
    found = marker.read_text(encoding="utf-8").strip()
    if found != cfg.port_commit:
        raise EstimatorError(
            f"el port está en {found[:10]} y la config fija {cfg.port_commit[:10]}: {HINT}"
        )
    if not cfg.weights.exists():
        raise EstimatorError(f"faltan los pesos {cfg.weights}: {HINT}")
    digest = sha256_file(cfg.weights)
    if digest != cfg.weights_sha256:
        raise EstimatorError(f"los pesos {cfg.weights} no son los fijados (SHA-256 {digest[:12]}…)")


def _import_nets(port_dir: Path) -> Any:
    """Importa el paquete `nets` del port.

    El port no es instalable: sus módulos se importan como `nets.*` desde su carpeta. Se agrega
    esa carpeta al camino de búsqueda solo durante la importación. Si ya hay otro `nets`
    cargado, se rechaza: mezclar dos paquetes con el mismo nombre daría errores confusos.
    """
    loaded = sys.modules.get("nets")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", "") or "").resolve()
        if port_dir.resolve() not in origin.parents:
            raise EstimatorError(f"ya hay un módulo `nets` cargado desde {origin}")
        return loaded
    sys.path.insert(0, str(port_dir))
    try:
        return importlib.import_module("nets")
    finally:
        sys.path.remove(str(port_dir))


def load_model(cfg: PortConfig) -> tuple[Any, str]:
    """Carga CREStereo congelado. Devuelve (modelo, dispositivo resuelto)."""
    import torch

    from visionx.depth.mono import pick_device

    check_files(cfg)
    nets = _import_nets(cfg.port_dir)
    device = pick_device(cfg.device)
    # test_mode=True: la red devuelve solo la última predicción y no la lista de todas.
    model = nets.Model(max_disp=cfg.max_disp, mixed_precision=False, test_mode=True)
    # weights_only=True: el archivo se lee como tensores, sin ejecutar código guardado adentro.
    state = torch.load(cfg.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, device
