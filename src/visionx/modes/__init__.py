"""Modos de uso (T09–T11): buscar/guiar, distancia y leer.

`build_handlers` arma los modos para el controlador de T07 con la config y la geometría del
frame (donde están las cajas del detector).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from visionx.modes.common import ModeSettings, View
from visionx.modes.describe import DescribeHandler, DescribeSettings
from visionx.modes.distance import DistanceHandler
from visionx.modes.read import ReadHandler
from visionx.modes.search import SearchHandler
from visionx.modes.target import TargetMemory
from visionx.reading import ReadingSettings
from visionx.shared.contracts import Mode
from visionx.shared.geometry import StereoGeometry

__all__ = [
    "DescribeHandler", "DescribeSettings", "DistanceHandler", "ModeSettings", "ReadHandler",
    "SearchHandler", "TargetMemory", "View", "build_handlers",
]  # fmt: skip


def build_handlers(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    spoken: Callable[[str], str],
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    *,
    prewarm_reading: bool = False,
    assistant: Any = None,
) -> dict[Mode, Any]:
    """Buscar y distancia comparten la memoria del objetivo guiado."""
    settings = ModeSettings.from_config(cfg)
    memory = TargetMemory()
    return {
        Mode.DESCRIBE: DescribeHandler(
            geometry,
            spoken,
            # DESCRIBIR siempre respeta depth.range_m, aun con un override de modes.range_m.
            settings=replace(
                settings,
                range_m=tuple((cfg.get("depth") or {}).get("range_m") or settings.range_m),
            ),
            describe=DescribeSettings.from_config(cfg),
            on_event=on_event,
            assistant=assistant,
        ),
        Mode.SEARCH: SearchHandler(
            geometry, spoken, settings, memory, on_event, assistant=assistant
        ),
        Mode.DISTANCE: DistanceHandler(
            geometry, spoken, settings, memory, on_event, assistant=assistant
        ),
        Mode.READ: ReadHandler(
            settings=ReadingSettings.from_config(cfg),
            on_event=on_event,
            prewarm=prewarm_reading,
            mode_settings=settings,
            assistant=assistant,
        ),
    }
