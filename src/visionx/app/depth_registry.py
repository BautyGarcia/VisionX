"""Construcción única de profundidad para la app, con imports neuronales diferidos."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry
from visionx.shared.interfaces import DepthEstimator


def build_estimator(name: str, cfg: dict[str, Any], geometry: StereoGeometry) -> DepthEstimator:
    """Armar B0–B5, Fusion o una entrada de ``depth.estimadores``.

    ``fusion`` arma el paquete de ``depth.fusion.paquete`` con ``load_estimator`` de A
    (config completa). Una entrada ``{tipo: fusion, paquete: carpeta}`` arma otro paquete
    con la misma config (p. ej. dos candidatos lado a lado en ``recibir-profundidad``).
    Una entrada tiene ``fabrica: paquete.modulo:funcion`` y parámetros libres. La fábrica
    pública debe tener firma ``(cfg: dict, geometry: StereoGeometry) -> DepthEstimator``;
    recibe SOLO los parámetros de su entrada, sin ``fabrica``, y la geometría de captura.
    Si devuelve mapas reescalados, debe exponer ``geometry`` con su StereoGeometry de salida
    para que la ubicación espacial transforme las máscaras. No se descargan pesos aquí.
    """
    dcfg = cfg.get("depth") or {}
    if name in {"B0", "B1", "B2", "B3", "B4", "B5"}:
        from visionx.depth.stereo import SgbmEstimator, SgbmParams

        params = SgbmParams.from_config(dcfg.get("sgbm") or {})
        compute = dcfg.get("compute")
        options = {
            "compute_size": tuple(compute) if compute else None,
            "min_coverage": float(dcfg.get("min_coverage", 0.0)),
        }
        if name in {"B0", "B1", "B2"}:
            return SgbmEstimator(name, params, geometry, **options)
        from visionx.depth.align import AlignParams
        from visionx.depth.hybrid import HybridEstimator, HybridMatcher
        from visionx.depth.mono import DepthAnythingV2, MonoConfig

        stereo = (dcfg.get("b4") or {}).get("estereo", "B2")  # B5 usa las anclas de B4
        mono = DepthAnythingV2(MonoConfig.from_config(dcfg.get("mono") or {}))
        if name == "B5":
            from visionx.depth.hybrid import B5Params

            b5 = B5Params.from_config(dcfg)
            matcher = HybridMatcher(name, stereo, params, mono, b5.align, b5.local)
        else:
            align = AlignParams.from_config(dcfg.get("alineacion") or {})
            matcher = HybridMatcher(name, stereo, params, mono, align)
        return HybridEstimator(matcher, geometry, **options)

    registry = dcfg.get("estimadores") or {}
    if name == "fusion" and name not in registry:
        return _fusion(name, cfg, geometry, None)
    if name not in registry:
        raise InvalidInput(
            f"estimador de profundidad desconocido {name!r}; use B0–B5, fusion o depth.estimadores"
        )
    entry = registry[name]
    if not isinstance(entry, dict):
        raise InvalidInput(f"depth.estimadores.{name} debe ser un mapa con fabrica")
    if entry.get("tipo") == "fusion":
        return _fusion(name, cfg, geometry, entry.get("paquete"))
    factory_path = entry.get("fabrica")
    try:
        module, attribute = factory_path.split(":")
        factory = getattr(import_module(module), attribute)
        if not callable(factory):
            raise TypeError("la fábrica no es invocable")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise InvalidInput(
            f"depth.estimadores.{name}: no se puede importar fabrica {factory_path!r}: {exc}"
        ) from exc
    try:
        estimator = factory({k: v for k, v in entry.items() if k != "fabrica"}, geometry)
    except (ImportError, TypeError, ValueError) as exc:
        raise InvalidInput(
            f"depth.estimadores.{name}: falló fabrica {factory_path!r}: {exc}"
        ) from exc
    if not isinstance(estimator, DepthEstimator):
        raise InvalidInput(f"depth.estimadores.{name}: la fábrica no devolvió un DepthEstimator")
    return estimator


def _fusion(
    name: str, cfg: dict[str, Any], geometry: StereoGeometry, package: str | None
) -> DepthEstimator:
    """Fusion de A con la config completa: base, SGBM, alineación, B5 y monocular de ``depth``."""
    try:
        from visionx.depth.fusion.estimator import load_estimator
    except ImportError as exc:
        raise InvalidInput(f"{name}: Fusion necesita el extra ml ({exc})") from exc
    try:
        return load_estimator(cfg, geometry, package=package or None)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise InvalidInput(f"{name}: no se pudo cargar el paquete de Fusion: {exc}") from exc
