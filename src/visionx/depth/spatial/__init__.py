"""Ubicación por objeto de A (T09; política calibrada: T13): máscara + profundidad -> objeto.

`ObjectLocator` cumple el Protocol `SpatialEstimator` (`shared/interfaces.py`) y `build_locator`
lo arma desde la sección `spatial` de la config. Ver `locator.py` para el método y la política.
"""

from visionx.depth.spatial.locator import (
    LocatorParams,
    ObjectDiagnostics,
    ObjectLocator,
    build_locator,
)

__all__ = ["LocatorParams", "ObjectDiagnostics", "ObjectLocator", "build_locator"]
