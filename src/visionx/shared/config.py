"""Carga de configuración YAML."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee la config por defecto y, si se pasa otra, la superpone encima."""
    # En desarrollo hay una única fuente: configs/default.yaml. El wheel incluye ese
    # mismo archivo como recurso del paquete; no depende del directorio donde se ejecute.
    is_checkout = (ROOT / "pyproject.toml").is_file() and (ROOT / "src" / "visionx").is_dir()
    if is_checkout:
        defaults = DEFAULT_CONFIG
    else:
        defaults = files("visionx") / "default.yaml"
    with defaults.open("r", encoding="utf-8") as f:
        cfg = _mapping(yaml.safe_load(f), str(defaults))
    # Nunca buscar local.yaml automáticamente: el llamador debe elegir su override.
    if path is not None:
        _merge_file(cfg, Path(path), set())
    return cfg


def _merge_file(cfg: dict[str, Any], path: Path, seen: set[Path]) -> None:
    """Superpone `path`; si trae `incluir:` (un archivo o una lista), esos van antes.

    Sirve para pruebas que suman unas claves encima de la config local de cada máquina
    (p. ej. configs/velocidad/C1.yaml incluye configs/local.yaml). Rutas relativas al repo.
    """
    path = path if path.is_absolute() or path.exists() else ROOT / path
    real = path.resolve()
    if real in seen:
        raise ValueError(f"{path}: `incluir` en ciclo")
    seen.add(real)
    with open(path, encoding="utf-8") as f:
        over = _mapping(yaml.safe_load(f), str(path))
    included = over.pop("incluir", None) or []
    for item in [included] if isinstance(included, str) else included:
        _merge_file(cfg, Path(item), seen)
    _merge(cfg, over)


def _mapping(value: object, source: str) -> dict[str, Any]:
    # Un archivo vacío es un override sin cambios. Una lista o un escalar no lo es:
    # rechazarlo aquí evita errores confusos cuando otro módulo acceda a sus secciones.
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{source}: se esperaba un mapping YAML con claves de texto")
    return value


def _merge(base: dict[str, Any], over: dict[str, Any]) -> None:
    """Superpone claves explícitas y conserva el resto de cada sección por defecto."""
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
