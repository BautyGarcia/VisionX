"""Armar el mensaje Detection a partir de arrays de un modelo (T06).

Separado del adaptador de Ultralytics para poder probarlo sin torch ni pesos: el adaptador solo
pasa los tensores a NumPy y llama a `build_detection`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visionx.shared.contracts import Detection, StereoFrame
from visionx.shared.errors import EstimatorError, InvalidInput


def require_geometry(frame: StereoFrame) -> str:
    """geometry_id del frame; sin él no se pueden combinar las máscaras con la profundidad.

    Un par rectificado lo trae siempre. Para imágenes sueltas, quien arma el frame tiene que
    declarar en qué geometría están (por ejemplo `imagen-1920x1080`): el detector no la inventa.
    """
    if frame.geometry_id is None:
        raise InvalidInput(
            "el detector necesita frame.geometry_id: rectificar el par o declarar la geometría"
        )
    return frame.geometry_id


def build_detection(
    frame: StereoFrame,
    *,
    model_id: str,
    boxes_xyxy: np.ndarray,
    class_names: Sequence[str],
    scores: np.ndarray,
    inference_ms: float,
    masks: np.ndarray | None = None,
    track_ids: Sequence[int | None] | None = None,
) -> Detection:
    """Detection del ojo izquierdo de `frame`, con cajas y máscaras en sus píxeles.

    boxes_xyxy: N x 4 en píxeles de la imagen izquierda. masks: N x H x W con el tamaño de esa
    imagen (bool o probabilidades, se umbraliza en 0,5). Si el modelo devuelve máscaras de otro
    tamaño es un error del adaptador: reescalarlas acá escondería un desplazamiento de letterbox.
    """
    geometry_id = require_geometry(frame)
    w, h = frame.size
    n = len(class_names)
    # Copia: el recorte de abajo no debe modificar el array de quien llama.
    boxes = np.array(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != n or len(scores) != n:
        raise EstimatorError(
            f"el modelo devolvió {len(boxes)} cajas, {len(scores)} scores, {n} clases"
        )
    # Las cajas pueden salirse unos píxeles de la imagen; recortarlas mantiene xyxy válido.
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    # Algunos modelos dan scores apenas por encima de 1 por redondeo en media precisión.
    scores = np.clip(scores, 0.0, 1.0)

    bool_masks = None
    if masks is not None:
        masks = np.asarray(masks)
        if masks.ndim != 3 or masks.shape[0] != n or masks.shape[1:] != (h, w):
            raise EstimatorError(
                f"máscaras {masks.shape}, se esperaba ({n}, {h}, {w}): "
                "el adaptador debe entregarlas en la resolución de la imagen"
            )
        bool_masks = masks if masks.dtype == np.bool_ else masks > 0.5

    ids: tuple[int | None, ...] | None = None
    if track_ids is not None:
        ids = tuple(None if t is None else int(t) for t in track_ids)

    return Detection(
        session_id=frame.session_id,
        frame_id=frame.frame_id,
        model_id=model_id,
        geometry_id=geometry_id,
        calibration_id=frame.calibration_id,
        image_size=(w, h),
        boxes=boxes,
        classes=tuple(class_names),
        scores=scores,
        inference_ms=float(inference_ms),
        track_ids=ids,
        masks=bool_masks,
    )


def empty_detection(frame: StereoFrame, *, model_id: str, inference_ms: float) -> Detection:
    """Cero objetos: un resultado válido, distinto de un fallo del modelo."""
    w, h = frame.size
    return build_detection(
        frame,
        model_id=model_id,
        boxes_xyxy=np.zeros((0, 4), np.float32),
        class_names=(),
        scores=np.zeros(0, np.float32),
        inference_ms=inference_ms,
        masks=np.zeros((0, h, w), np.bool_),
    )
