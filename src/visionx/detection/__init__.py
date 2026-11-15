"""Detección, segmentación y seguimiento de objetos (T06).

Desde la terminal (ver `python -m visionx.detection --help`; requiere `uv sync --extra deteccion`):

    uv run python -m visionx.detection descargar --modelo yolo26n-seg yolo11n-seg
    uv run python -m visionx.detection clases "las tazas"
    uv run python -m visionx.detection detectar recordings/prueba --imagenes --rectificar
    uv run python -m visionx.detection medir carpeta_de_imagenes --modelos yolo26n-seg yolo11n-seg

Desde código, el real y el falso cumplen la interfaz Detector y se intercambian:

    detector = YoloDetector(DetectorSettings.from_config(load_config()))
    detector = FakeDetector({0: [FakeObject("cup", (0.4, 0.4, 0.6, 0.7), track_id=1)]})
    detection = detector.predict(frame)   # frame con geometry_id (rectificado o declarado)

YoloDetector importa torch/ultralytics recién al construirse: este paquete se puede importar
sin el extra instalado.
"""

from visionx.detection.catalog import COCO_CLASSES, COCO_ES, ClassCatalog, normalize
from visionx.detection.convert import build_detection, empty_detection, require_geometry
from visionx.detection.fake import FakeDetector, FakeObject
from visionx.detection.yolo import DetectorSettings, YoloDetector, download_weights

__all__ = [
    "COCO_CLASSES",
    "COCO_ES",
    "ClassCatalog",
    "DetectorSettings",
    "FakeDetector",
    "FakeObject",
    "YoloDetector",
    "build_detection",
    "download_weights",
    "empty_detection",
    "normalize",
    "require_geometry",
]
