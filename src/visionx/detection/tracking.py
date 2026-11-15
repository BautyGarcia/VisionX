"""IDs temporales con ByteTrack, sin modificar lo que entregó el detector (T06).

`model.track()` de Ultralytics hace dos cosas que no queremos:
- Oculta un objeto nuevo hasta que ByteTrack lo confirma en el frame siguiente. Para VisionX eso
  sería informar "no hay taza" cuando el detector sí la vio (TECNICO: fallo no es ausencia).
- Reemplaza la caja del detector por la caja suavizada del filtro de Kalman, pero deja la máscara
  del detector: caja y máscara quedan desfasadas cuando el objeto se mueve.

Por eso el detector predice normalmente y este módulo solo agrega un ID a cada detección que
ByteTrack confirmó. Las demás se informan igual, con track None. Un ID es temporal: identifica
"el mismo objeto visto hace un momento", no una identidad que sobreviva a perderlo de vista
más de `track_buffer` llamadas.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ByteTrackIds:
    """Envoltorio del BYTETracker de Ultralytics que devuelve un ID (o None) por detección.

    `track_buffer` y los umbrales vienen del YAML de Ultralytics. El búfer cuenta llamadas, no
    segundos: con el detector a 7 FPS, 30 llamadas son ~4 s; a 30 FPS, 1 s. Hay que medirlo
    con clips reales antes de fijarlo (T06, pérdidas y reapariciones).
    """

    def __init__(self, tracker_yaml: str = "bytetrack.yaml") -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
        if cfg.tracker_type != "bytetrack":
            # Otros trackers de Ultralytics (BoT-SORT con ReID) bajan un modelo extra al vuelo.
            raise ValueError(f"{tracker_yaml}: solo se admite tracker_type bytetrack")
        self.config: dict[str, Any] = dict(vars(cfg))
        self._tracker = BYTETracker(args=cfg)

    def reset(self) -> None:
        """Olvidar todas las trayectorias y volver a numerar desde 1."""
        self._tracker.reset()

    def assign(self, boxes: Any, image: np.ndarray | None = None) -> list[int | None]:
        """Un ID por detección, en el mismo orden; None si ByteTrack todavía no la confirmó.

        boxes: `result.boxes.cpu().numpy()` de Ultralytics (conf, cls, xywh e indexado bool).
        Hay que llamarlo también en frames sin detecciones: así cuenta el tiempo de pérdida.
        """
        tracks = self._tracker.update(boxes, image)
        ids: list[int | None] = [None] * len(boxes)
        # Cada fila es [x1, y1, x2, y2, track_id, score, cls, idx]; idx es la detección original.
        for row in tracks:
            ids[int(row[-1])] = int(row[4])
        return ids
