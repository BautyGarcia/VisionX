"""Ubicación por objeto PROVISORIA (área B) hasta que A entregue la suya (T09/T13).

De una máscara del detector y un DepthResult del mismo frame a un SpatialObject: ángulos,
distancia y política (decir metros, solo dirección o reencuadrar). Es un candidato medido en
clips de desarrollo, no la política calibrada de T13; se reemplaza en `app/__main__.py::
build_depth` sin tocar los modos, que solo leen `policy` y `z_m`.

**Qué distancia se informa.** La de la **capa más cercana** de la máscara: dentro de la
máscara, los valores de Z se agrupan en capas (histograma logarítmico, bins de 3 %, porque el
error del estéreo crece con Z) y se toma la capa más cercana que tenga al menos
`layer_min_frac` de los píxeles. De esa capa se informa el percentil `front_percentile` (la
superficie del frente, que es lo que se mide con cinta y con lo que uno choca).

Por qué no la mediana de toda la máscara (versión anterior): en objetos calados (una silla)
la máscara incluye el fondo que se ve a través, y la mediana se va hacia el fondo; en
`dev-1114-acercarse-1` daba 0,84 m con la silla a ~0,5 m. La capa cercana daba 0,59 m. En
objetos compactos (vaso, botella) capa y máscara coinciden y el percentil 20 bajó el error
contra la cinta en ~2–3 cm (BITACORA T09 mejoras).
"""

from __future__ import annotations

import numpy as np

from visionx.shared.contracts import (
    DepthResult,
    Detection,
    DistancePolicy,
    SpatialObject,
    StereoFrame,
    check_depth_geometry,
    check_detection_frame,
    check_same_frame,
)
from visionx.shared.geometry import StereoGeometry


def nearest_layer(
    z: np.ndarray, min_frac: float = 0.25, ratio: float = 1.03, floor: float = 0.01
) -> np.ndarray | None:
    """Valores de Z de la capa más cercana con al menos `min_frac` de los píxeles, o None.

    Una capa es una racha de bins (de ancho relativo `ratio`) con al menos `floor` de los
    píxeles cada uno. Los bins casi vacíos separan capas: así un poco de ruido delante no
    se toma como objeto, y el fondo visto a través de un objeto calado queda en otra capa.
    """
    if z.size == 0:
        return None
    lo = max(float(z.min()), 0.05)
    n = int(np.ceil(np.log(max(float(z.max()), lo * ratio) / lo) / np.log(ratio))) + 1
    edges = lo * ratio ** np.arange(n + 1)
    counts, _ = np.histogram(z, bins=edges)
    total = counts.sum()
    occupied = counts >= max(1.0, floor * total)
    i = 0
    while i < len(counts):
        if not occupied[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(counts) and occupied[j + 1]:
            j += 1
        if counts[i : j + 1].sum() / total >= min_frac:
            return z[(z >= edges[i]) & (z <= edges[j + 1])]
        i = j + 1
    return None


class ProvisionalSpatialEstimator:
    """SpatialEstimator provisorio. Política (candidata, ver docstring del módulo):

    - caja que toca el borde izquierdo o derecho -> ASK_REFRAME (el objeto está cortado);
    - menos de `min_valid` de la máscara con profundidad, o sin capa -> DIRECTION_ONLY;
    - IQR de la capa mayor que `max_rel_spread`·Z -> DIRECTION_ONLY (capa demasiado difusa);
    - si no, SPEAK_METERS. El rango que se anuncia lo controla el modo (`modes.range_m`).

    `geometry` es la del frame (cajas y máscaras). Si la profundidad está en una versión
    reescalada, se pasa como `depth_geometry` y cada máscara se reduce por vecino más cercano.
    """

    def __init__(
        self,
        geometry: StereoGeometry,
        min_valid: float = 0.5,
        depth_geometry: StereoGeometry | None = None,
        max_rel_spread: float = 0.5,
        border_px: float = 2.0,
        layer_min_frac: float = 0.25,
        front_percentile: float = 20.0,
    ) -> None:
        self.geometry = geometry
        self.depth_geometry = depth_geometry or geometry
        self.min_valid = min_valid
        self.max_rel_spread = max_rel_spread
        self.border_px = border_px
        self.layer_min_frac = layer_min_frac
        self.front_percentile = front_percentile

    def locate(
        self, frame: StereoFrame, depth: DepthResult, detections: Detection
    ) -> list[SpatialObject]:
        check_same_frame(frame, depth, detections)
        check_detection_frame(frame, detections)
        check_depth_geometry(depth, self.depth_geometry)
        out = []
        z_map = depth.depth_m
        for i, label in enumerate(detections.classes):
            x1, y1, x2, y2 = detections.boxes[i]
            mask = (
                self._to_depth_size(detections.masks[i])
                if detections.masks is not None
                else np.zeros(depth.valid_mask.shape, bool)
            )
            inside = mask & depth.valid_mask
            frac = float(inside.sum() / mask.sum()) if mask.any() else 0.0
            z = spread = None
            layer = nearest_layer(z_map[inside], self.layer_min_frac) if z_map is not None else None
            if layer is not None and layer.size:
                z = float(np.percentile(layer, self.front_percentile))
                spread = float(np.percentile(layer, 75) - np.percentile(layer, 25))
            h, vert = self.geometry.pixel_to_angles(float(x1 + x2) / 2, float(y1 + y2) / 2)
            width = self.geometry.width
            if x1 <= self.border_px or x2 >= width - 1 - self.border_px:
                policy = DistancePolicy.ASK_REFRAME
            elif z is None or frac < self.min_valid:
                policy = DistancePolicy.DIRECTION_ONLY
            elif spread is not None and spread > self.max_rel_spread * z:
                policy = DistancePolicy.DIRECTION_ONLY
            else:
                policy = DistancePolicy.SPEAK_METERS
            track = detections.track_ids[i] if detections.track_ids is not None else None
            out.append(
                SpatialObject(
                    session_id=frame.session_id,
                    frame_id=frame.frame_id,
                    detection_index=i,
                    track_id=track,
                    label=label,
                    depth_model_id=depth.model_id,
                    angle_h_deg=h,
                    angle_v_deg=vert,
                    z_m=z,
                    z_spread_m=spread,
                    valid_fraction=frac,
                    detector_score=float(detections.scores[i]),
                    policy=policy,
                )
            )
        return out

    def _to_depth_size(self, mask: np.ndarray) -> np.ndarray:
        """Máscara del detector llevada a la resolución de la profundidad (vecino más cercano:
        no inventa bordes intermedios)."""
        if mask.shape == self.depth_geometry.size[::-1]:
            return mask
        import cv2

        small = cv2.resize(
            mask.astype(np.uint8), self.depth_geometry.size, interpolation=cv2.INTER_NEAREST
        )
        return small.astype(bool)
