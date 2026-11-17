"""Modos de uso (T09–T11): buscar/guiar, distancia y leer.

`build_handlers` arma los modos para el controlador de T07 con la config y la geometría del
frame (donde están las cajas del detector).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from visionx.modes.common import ModeSettings, View
from visionx.modes.distance import DistanceHandler
from visionx.modes.read import ReadHandler
from visionx.modes.search import SearchHandler
from visionx.modes.target import TargetMemory
from visionx.reading import ReadingSettings
from visionx.shared.contracts import Mode
from visionx.shared.geometry import StereoGeometry

__all__ = [
    "DistanceHandler", "ModeSettings", "ReadHandler", "SearchHandler", "TargetMemory", "View",
    "build_handlers",
]  # fmt: skip


def build_handlers(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    spoken: Callable[[str], str],
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[Mode, Any]:
    """Buscar y distancia comparten la memoria del objetivo guiado."""
    settings = ModeSettings.from_config(cfg)
    memory = TargetMemory()
    return {
        Mode.SEARCH: SearchHandler(geometry, spoken, settings, memory, on_event),
        Mode.DISTANCE: DistanceHandler(geometry, spoken, settings, memory, on_event),
        Mode.READ: ReadHandler(settings=ReadingSettings.from_config(cfg), on_event=on_event),
    }
