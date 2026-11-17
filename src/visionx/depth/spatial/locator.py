"""`ObjectLocator`: de máscara + profundidad del mismo frame a `SpatialObject` (T09, área A).

Reemplaza al provisorio de B (`app/spatial.py`), del que conserva la idea medida en clips
(capa más cercana de la máscara y percentil del frente) y agrega lo que es del área de
profundidad. Pasos por objeto:

1. **Máscara en la grilla de la profundidad** (vecino más cercano, `surface.mask_to_depth_grid`).
   Sin máscara no hay segmentación: una caja no equivale al objeto (TECNICO T09/T13), así que
   se informa solo la dirección (centro de la caja) y la política es ABSTAIN.
2. **Interior adaptativo** (`surface.interior_mask`): erosión proporcional al grosor del objeto,
   con tope, para no medir el fondo que la máscara toma en el borde. Apagado por defecto: en
   los clips no mejoró el error y en la silla calada alejó el frente (config `spatial`).
3. **Capa más cercana en disparidad efectiva** `d + doffs` (`surface.nearest_layer`): el error del
   estéreo es ~constante en px de disparidad. Z sale de la geometría: `Z = fx·B / (d + doffs)`.
   Con intervalos relativos (3 %) es lo mismo que agrupar en Z con intervalos logarítmicos,
   como el provisorio; trabajar en disparidad permite además un piso en px
   (`layer_bin_min_px`), que en los clips no ayudó y queda en 0.
4. **Superficie del frente**: percentil `front_percentile` de Z de la capa (en disparidad es el
   percentil `100 - front_percentile`, porque Z = fx·B/d es decreciente; un percentil se
   conserva por una transformación monótona). `z_spread_m` = IQR de Z de la capa, en metros:
   dispersión robusta de superficies, no un intervalo de confianza.
5. **Dirección desde el centro representativo**: centroide de los píxeles de la capa elegida;
   si no hay capa, centroide de la máscara; sin máscara, centro de la caja.
6. **Política** (`DistancePolicy`), en este orden:
   - ABSTAIN: profundidad UNMEASURABLE, sin máscara, máscara vacía en la grilla de la
     profundidad o ningún píxel válido dentro del objeto;
   - ASK_REFRAME: la caja toca el borde izquierdo o derecho (objeto cortado; el modo pide girar
     hacia ese lado) o, si `reframe_bottom`, el borde inferior **y** la franja de la capa que
     toca ese borde está al menos tan cerca como el frente (lo más cercano sigue fuera de
     cuadro). El borde superior no pide reencuadre: con la cámara a la altura del pecho lo
     que se corta arriba es lo más alto, no lo más cercano;
   - DIRECTION_ONLY: menos de `min_valid` de la máscara con profundidad, menos de
     `min_valid_px` píxeles válidos en el interior, sin capa, o capa difusa (IQR mayor que
     max(`max_rel_spread`·Z, `spread_allowance_m`));
   - SPEAK_METERS en otro caso. El rango que se anuncia lo decide el modo (`depth.range_m`).

`locate_detailed` devuelve además un `ObjectDiagnostics` por objeto (píxeles, erosión, capa,
fuente estéreo/monocular, mediana de b si el DepthResult trae b de Laplace). Esa b es solo un
dato de diagnóstico: los píxeles vecinos están correlacionados y promediar b no da la
incertidumbre del objeto (TECNICO, T13). No cambia la política ni el contrato.

Los valores por defecto (`configs/default.yaml`, sección `spatial`) salen de clips de
desarrollo de una sola sesión; los definitivos se fijan con el banco ELP (T04/T13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from visionx.depth.spatial.surface import interior_mask, mask_to_depth_grid, nearest_layer
from visionx.shared import _validate as v
from visionx.shared.contracts import (
    DepthResult,
    DepthStatus,
    Detection,
    DistancePolicy,
    SpatialObject,
    StereoFrame,
    UncertaintyKind,
    check_depth_geometry,
    check_detection_frame,
    check_same_frame,
)
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

SOURCE_STEREO, SOURCE_MONO = 1, 2  # mismos valores que el mapa `source` de depth.hybrid


@dataclass(frozen=True)
class LocatorParams:
    """Sección `spatial` de la config. Unidades y razones en `configs/default.yaml`."""

    min_valid: float = 0.5
    min_valid_px: int = 30
    layer_min_frac: float = 0.25
    layer_bin_rel: float = 0.03
    layer_bin_min_px: float = 0.0
    layer_floor: float = 0.01
    front_percentile: float = 20.0
    max_rel_spread: float = 0.5
    spread_allowance_m: float = 0.4
    erode_frac: float = 0.0
    erode_max_px: float = 6.0
    min_interior_px: int = 50
    border_px: float = 2.0
    reframe_bottom: bool = False
    bottom_band_px: int = 2

    def __post_init__(self) -> None:
        v.finite("min_valid", self.min_valid, 0.0, 1.0)
        v.non_negative_int("min_valid_px", self.min_valid_px)
        v.finite("layer_min_frac", self.layer_min_frac, 0.0, 1.0)
        v.finite("layer_bin_rel", self.layer_bin_rel, low=0.0)
        v.finite("layer_bin_min_px", self.layer_bin_min_px, low=0.0)
        if self.layer_bin_rel <= 0 and self.layer_bin_min_px <= 0:
            raise InvalidInput("layer_bin_rel o layer_bin_min_px tiene que ser > 0")
        v.finite("layer_floor", self.layer_floor, 0.0, 1.0)
        v.finite("front_percentile", self.front_percentile, 0.0, 100.0)
        v.finite("max_rel_spread", self.max_rel_spread, low=0.0)
        v.finite("spread_allowance_m", self.spread_allowance_m, low=0.0)
        v.finite("erode_frac", self.erode_frac, 0.0, 1.0)
        v.finite("erode_max_px", self.erode_max_px, low=0.0)
        v.non_negative_int("min_interior_px", self.min_interior_px)
        v.finite("border_px", self.border_px, low=0.0)
        if not isinstance(self.reframe_bottom, bool):
            raise InvalidInput("reframe_bottom tiene que ser true o false")
        v.non_negative_int("bottom_band_px", self.bottom_band_px)

    @classmethod
    def from_config(cls, section: dict[str, Any] | None) -> LocatorParams:
        """Lee `spatial`; las claves que falten toman el valor por defecto de la clase."""
        section = section or {}
        known = {f.name for f in fields(cls)}
        unknown = set(section) - set(known)
        if unknown:
            raise InvalidInput(f"spatial: claves desconocidas {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, value in section.items():
            default = getattr(cls, name)
            if isinstance(default, bool):
                kwargs[name] = value
            elif isinstance(default, int):
                kwargs[name] = int(value)
            else:
                kwargs[name] = float(value)
        return cls(**kwargs)


@dataclass(frozen=True)
class ObjectDiagnostics:
    """Cómo se midió un objeto. Para trazas y evaluación; no viaja en el contrato."""

    detection_index: int
    reason: str  # por qué se eligió la política ("ok", "borde-izquierdo", "capa-difusa", ...)
    mask_source: str  # "mascara" | "caja" (sin máscara del detector)
    direction_source: str  # "capa" | "mascara" | "caja"
    mask_px: int  # píxeles de la máscara en la grilla de la profundidad
    interior_px: int  # después de la erosión adaptativa
    erosion_px: float  # radio de erosión aplicado, px de la grilla de profundidad
    valid_px: int  # píxeles del interior con profundidad válida
    layer_px: int  # píxeles de la capa elegida
    cut_edges: tuple[str, ...]  # bordes de la imagen que toca la caja
    d_front_px: float | None  # disparidad (cruda, sin doffs) de la superficie del frente
    z_median_m: float | None  # mediana de Z de la capa
    box_angle_h_deg: float  # ángulos desde el centro de la caja, para comparar
    box_angle_v_deg: float
    stereo_fraction: float | None  # de la capa: fracción con fuente estéreo (si hay mapa)
    b_median_px: float | None  # mediana de b de Laplace en la capa: SOLO diagnóstico

    @property
    def coverage(self) -> float:
        """Fracción de la máscara que sobrevive a la erosión."""
        return self.interior_px / self.mask_px if self.mask_px else 0.0


@dataclass
class _Surface:
    """Resultado intermedio de medir la superficie de un objeto."""

    mask_px: int = 0
    interior_px: int = 0
    erosion_px: float = 0.0
    valid_px: int = 0
    valid_fraction: float = 0.0
    layer_px: int = 0
    z_m: float | None = None
    spread_m: float | None = None
    z_median_m: float | None = None
    d_front_px: float | None = None
    centroid: tuple[float, float] | None = None  # (u, v) en la grilla de la profundidad
    front_at_bottom: bool = False  # la franja inferior de la capa está tan cerca como el frente
    stereo_fraction: float | None = None
    b_median_px: float | None = None


class ObjectLocator:
    """SpatialEstimator de A. Misma construcción que el provisorio: `geometry` es la del frame
    (cajas y máscaras); si la profundidad está en otra resolución, va en `depth_geometry`."""

    def __init__(
        self,
        geometry: StereoGeometry,
        params: LocatorParams | None = None,
        depth_geometry: StereoGeometry | None = None,
    ) -> None:
        self.geometry = geometry
        self.depth_geometry = depth_geometry or geometry
        self.params = params or LocatorParams()

    def locate(
        self, frame: StereoFrame, depth: DepthResult, detections: Detection
    ) -> list[SpatialObject]:
        return [obj for obj, _ in self.locate_detailed(frame, depth, detections)]

    def locate_detailed(
        self,
        frame: StereoFrame,
        depth: DepthResult,
        detections: Detection,
        source: np.ndarray | None = None,
    ) -> list[tuple[SpatialObject, ObjectDiagnostics]]:
        """Como `locate`, más el diagnóstico de cada objeto.

        `source` (opcional, H x W uint8 en la grilla de la profundidad): de dónde salió cada
        píxel (1 estéreo, 2 monocular), como `HybridOutput.source`. El DepthResult no lo trae.
        """
        check_same_frame(frame, depth, detections)
        check_detection_frame(frame, detections)
        check_depth_geometry(depth, self.depth_geometry)
        if source is not None:
            v.array("source", source, np.uint8, 2, depth.disparity_px.shape)
        return [self._one(i, frame, depth, detections, source) for i in range(len(detections))]

    # --- un objeto ------------------------------------------------------------------------

    def _one(
        self,
        i: int,
        frame: StereoFrame,
        depth: DepthResult,
        det: Detection,
        source: np.ndarray | None,
    ) -> tuple[SpatialObject, ObjectDiagnostics]:
        p = self.params
        x1, y1, x2, y2 = (float(c) for c in det.boxes[i])
        box_h, box_v = self.geometry.pixel_to_angles((x1 + x2) / 2, (y1 + y2) / 2)
        width, height = self.geometry.size
        cut = []
        if x1 <= p.border_px:
            cut.append("izquierda")
        if x2 >= width - 1 - p.border_px:
            cut.append("derecha")
        if y1 <= p.border_px:
            cut.append("arriba")
        if y2 >= height - 1 - p.border_px:
            cut.append("abajo")

        mask = det.masks[i] if det.masks is not None else None
        surface = _Surface()
        if mask is not None:
            surface = self._measure(mask, depth, source)

        # Dirección: capa -> máscara -> caja.
        if surface.centroid is not None:
            h, vert = self.depth_geometry.pixel_to_angles(*surface.centroid)
            direction = "capa"
        elif mask is not None and mask.any():
            rows, cols = np.nonzero(mask)
            h, vert = self.geometry.pixel_to_angles(float(cols.mean()), float(rows.mean()))
            direction = "mascara"
        else:
            h, vert, direction = box_h, box_v, "caja"

        policy, reason = self._policy(depth, mask, surface, cut)
        z = surface.z_m if policy is not DistancePolicy.ABSTAIN else None
        spread = surface.spread_m if z is not None else None
        track = det.track_ids[i] if det.track_ids is not None else None
        obj = SpatialObject(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            detection_index=i,
            track_id=track,
            label=det.classes[i],
            depth_model_id=depth.model_id,
            angle_h_deg=h,
            angle_v_deg=vert,
            z_m=z,
            z_spread_m=spread,
            valid_fraction=surface.valid_fraction,
            detector_score=float(det.scores[i]),
            policy=policy,
        )
        diag = ObjectDiagnostics(
            detection_index=i,
            reason=reason,
            mask_source="mascara" if mask is not None else "caja",
            direction_source=direction,
            mask_px=surface.mask_px,
            interior_px=surface.interior_px,
            erosion_px=surface.erosion_px,
            valid_px=surface.valid_px,
            layer_px=surface.layer_px,
            cut_edges=tuple(cut),
            d_front_px=surface.d_front_px,
            z_median_m=surface.z_median_m,
            box_angle_h_deg=box_h,
            box_angle_v_deg=box_v,
            stereo_fraction=surface.stereo_fraction,
            b_median_px=surface.b_median_px,
        )
        return obj, diag

    def _measure(self, mask: np.ndarray, depth: DepthResult, source: np.ndarray | None) -> _Surface:
        """Interior, capa cercana, frente, dispersión y centroide de un objeto."""
        p = self.params
        g = self.depth_geometry
        s = _Surface()
        crop, (c0, r0) = mask_to_depth_grid(mask, g.size)
        s.mask_px = int(crop.sum())
        if s.mask_px == 0:
            return s
        ch, cw = crop.shape
        edges = (c0 == 0, r0 == 0, c0 + cw >= g.width, r0 + ch >= g.height)
        inner, s.erosion_px = interior_mask(
            crop, p.erode_frac, p.erode_max_px, p.min_interior_px, edges
        )
        s.interior_px = int(inner.sum())
        window = (slice(r0, r0 + ch), slice(c0, c0 + cw))
        valid = depth.valid_mask[window]
        # valid_fraction del contrato: sobre la máscara entera (no el interior), como el provisorio.
        s.valid_fraction = float((crop & valid).sum()) / s.mask_px
        sel = inner & valid
        s.valid_px = int(sel.sum())
        if s.valid_px == 0:
            return s
        rows, cols = np.nonzero(sel)
        # Disparidad efectiva d + doffs (px): la magnitud física, positiva en píxeles válidos.
        d_eff = depth.disparity_px[window][rows, cols].astype(np.float64) + depth.doffs
        layer = nearest_layer(
            d_eff, p.layer_min_frac, p.layer_bin_rel, p.layer_bin_min_px, p.layer_floor
        )
        if layer is None or not layer.any():
            return s
        d_layer = d_eff[layer]
        s.layer_px = int(d_layer.size)
        fb = g.fx * g.baseline_m  # Z = fx·B / (d + doffs), metros·px
        # Frente = percentil `front_percentile` en Z = percentil (100 - p) en disparidad.
        d_front = float(np.percentile(d_layer, 100.0 - p.front_percentile))
        d_q25, d_med, d_q75 = np.percentile(d_layer, [25.0, 50.0, 75.0])
        s.z_m = fb / d_front
        s.z_median_m = fb / float(d_med)
        s.spread_m = fb / float(d_q25) - fb / float(d_q75)  # IQR de Z, en metros
        s.d_front_px = d_front - depth.doffs
        lr, lc = rows[layer], cols[layer]
        s.centroid = (float(lc.mean()) + c0, float(lr.mean()) + r0)
        # ¿El frente sigue por debajo del cuadro? Píxeles de la capa en la franja inferior de la
        # imagen cuya disparidad mediana es al menos la del frente informado: lo más cercano
        # del objeto llega al borde, y lo que se cortó puede estar todavía más cerca.
        band = (lr + r0) >= g.height - 1 - p.bottom_band_px
        if band.any():
            s.front_at_bottom = bool(np.median(d_layer[band]) >= d_front)
        if source is not None:
            src = source[window][lr, lc]
            s.stereo_fraction = float((src == SOURCE_STEREO).mean())
        if depth.uncertainty is not None and depth.uncertainty_kind is UncertaintyKind.LAPLACE_B:
            s.b_median_px = float(np.median(depth.uncertainty[window][lr, lc]))
        if not math.isfinite(s.z_m) or s.z_m <= 0:  # no debería pasar: d + doffs > 0
            s.z_m = s.spread_m = None
        return s

    def _policy(
        self,
        depth: DepthResult,
        mask: np.ndarray | None,
        s: _Surface,
        cut: list[str],
    ) -> tuple[DistancePolicy, str]:
        p = self.params
        if depth.status is DepthStatus.UNMEASURABLE:
            return DistancePolicy.ABSTAIN, "profundidad-inmedible"
        if mask is None:
            return DistancePolicy.ABSTAIN, "sin-mascara"
        if s.mask_px == 0:
            return DistancePolicy.ABSTAIN, "mascara-vacia"
        if s.valid_px == 0:
            return DistancePolicy.ABSTAIN, "sin-pixeles-validos"
        if "izquierda" in cut or "derecha" in cut:
            side = "izquierdo" if "izquierda" in cut else "derecho"
            return DistancePolicy.ASK_REFRAME, f"borde-{side}"
        if p.reframe_bottom and "abajo" in cut and s.front_at_bottom:
            return DistancePolicy.ASK_REFRAME, "borde-inferior"
        if s.valid_fraction < p.min_valid:
            return DistancePolicy.DIRECTION_ONLY, "poca-fraccion-valida"
        if s.valid_px < p.min_valid_px:
            return DistancePolicy.DIRECTION_ONLY, "pocos-pixeles"
        if s.z_m is None:
            return DistancePolicy.DIRECTION_ONLY, "sin-capa"
        if s.spread_m is not None and s.spread_m > max(
            p.max_rel_spread * s.z_m, p.spread_allowance_m
        ):
            return DistancePolicy.DIRECTION_ONLY, "capa-difusa"
        return DistancePolicy.SPEAK_METERS, "ok"


def build_locator(
    cfg: dict[str, Any],
    geometry: StereoGeometry,
    depth_geometry: StereoGeometry | None = None,
) -> ObjectLocator:
    """`ObjectLocator` con la sección `spatial` de la config completa (la de `load_config`).

    Misma firma que `app.__main__.build_spatial`, para que la app cambie de ubicador en una
    línea.
    """
    return ObjectLocator(geometry, LocatorParams.from_config(cfg.get("spatial")), depth_geometry)
