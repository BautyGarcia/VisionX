"""Detector YOLO-seg de Ultralytics detrás de la interfaz Detector (T06).

Corre sobre el ojo izquierdo del par (la imagen de referencia de la profundidad) y entrega cajas,
clases, scores, máscaras en la resolución de esa imagen y, si se pide, IDs de ByteTrack.

Dependencia opcional: `uv sync --extra deteccion`. Los pesos se bajan una vez con
`python -m visionx.detection descargar`; el detector nunca descarga solo, para que una corrida
no dependa de internet ni cambie de pesos sin que quede registrado.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionx.detection.catalog import ClassCatalog
from visionx.detection.convert import build_detection, empty_detection, require_geometry
from visionx.detection.tracking import ByteTrackIds
from visionx.shared.config import ROOT
from visionx.shared.contracts import Detection, StereoFrame
from visionx.shared.errors import EstimatorError

# Release de ultralytics/assets de donde salen los pesos oficiales (el mismo que usa
# ultralytics 8.4 internamente). El hash de cada archivo bajado se registra en BITACORA.
WEIGHTS_RELEASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    """Sección `detection` de la config. Unidades en los comentarios de configs/default.yaml."""

    model: str = "yolo26n-seg"
    weights_dir: str = "weights"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    threads: int | None = None
    track: bool = True
    tracker: str = "bytetrack.yaml"
    classes: tuple[str, ...] | None = None
    extra_model: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DetectorSettings:
        d = cfg.get("detection") or {}
        classes = d.get("classes")
        return cls(
            model=str(d.get("model", cls.model)),
            weights_dir=str(d.get("weights_dir", cls.weights_dir)),
            device=str(d.get("device", cls.device)),
            imgsz=int(d.get("imgsz", cls.imgsz)),
            conf=float(d.get("conf", cls.conf)),
            iou=float(d.get("iou", cls.iou)),
            threads=None if d.get("threads") is None else int(d["threads"]),
            track=bool(d.get("track", cls.track)),
            tracker=str(d.get("tracker", cls.tracker)),
            classes=None if classes is None else tuple(str(c) for c in classes),
            extra_model=None if not d.get("extra_model") else str(d["extra_model"]),
        )

    @property
    def weights_path(self) -> Path:
        folder = Path(self.weights_dir)
        if not folder.is_absolute():
            folder = ROOT / folder
        return folder / f"{self.model}.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights(model: str, dest_dir: Path) -> Path:
    """Baja `<model>.pt` del release oficial a dest_dir. No pisa un archivo existente.

    Escribe primero un temporal y lo renombra al final: un corte de red no deja un .pt a medias
    que después se cargue como si fuera el modelo.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{model}.pt"
    if target.exists():
        return target
    partial = target.with_suffix(".pt.parcial")
    url = f"{WEIGHTS_RELEASE_URL}/{model}.pt"
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if partial.stat().st_size < 100_000:  # una página de error, no un modelo
        partial.unlink()
        raise OSError(f"{url}: la descarga no parece un modelo")
    partial.replace(target)
    return target


def build_detector(settings: DetectorSettings) -> Any:
    """El detector de la app: COCO, y YOLOE para lo que COCO no conoce si sus pesos están."""
    import dataclasses

    base = YoloDetector(dataclasses.replace(settings, extra_model=None))
    if not settings.extra_model:
        return base
    extra_settings = dataclasses.replace(settings, model=settings.extra_model, extra_model=None,
                                         classes=None)  # fmt: skip
    if not extra_settings.weights_path.is_file():
        print(f"aviso: no está {extra_settings.weights_path}; solo clases COCO "
              "(armarlo con `python -m visionx.detection vocabulario`)")  # fmt: skip
        return base
    from visionx.detection.switch import VocabularySwitch

    return VocabularySwitch(base, YoloDetector(extra_settings))


def build_vocabulary_weights(
    base: str, out: str, weights_dir: Path, classes: tuple[str, ...] | None = None
) -> Path:
    """Arma pesos YOLOE con el vocabulario de VisionX incorporado (una vez, con internet).

    YOLOE es de vocabulario abierto: cada clase es un embedding de texto (MobileCLIP). Acá se
    calculan una sola vez —promediando los sinónimos de `PROMPTS`— y quedan guardados en
    `<out>.pt`. Ese archivo se carga después con el YOLO de siempre, sin internet ni el
    codificador de texto: la app no cambia de vocabulario sin que quede registrado (sha256).
    """
    from visionx.detection.catalog import VISIONX_CLASSES, prompts_de

    classes = tuple(classes or VISIONX_CLASSES)
    base_path = download_weights(base, weights_dir)
    target = weights_dir / f"{out}.pt"
    try:
        import torch
        from ultralytics import YOLOE
    except ImportError as exc:
        raise EstimatorError("falta ultralytics: instalar con `uv sync --extra deteccion`") from exc
    here = os.getcwd()
    os.chdir(weights_dir)  # Ultralytics baja el codificador de texto a la carpeta actual
    try:
        model = YOLOE(str(base_path))
        rows = []
        for cls in classes:
            pe = model.get_text_pe(list(prompts_de(cls)))[0]  # (n_textos, D), normalizados
            mean = pe.mean(0)
            rows.append(mean / mean.norm())
        model.set_classes(list(classes), torch.stack(rows)[None])
        partial = target.with_suffix(".parcial.pt")
        model.save(str(partial))
        partial.replace(target)
    finally:
        os.chdir(here)
    return target


def _import_ultralytics() -> tuple[Any, Any]:
    # YOLO_OFFLINE antes del primer import: Ultralytics manda estadísticas de uso y consulta
    # internet al arrancar; VisionX no comparte datos de quien lo usa sin pedirlo.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise EstimatorError("falta ultralytics: instalar con `uv sync --extra deteccion`") from exc
    return YOLO, torch


def _resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    """Detector + segmentador YOLO: COCO con los pesos oficiales, o el vocabulario de VisionX
    con YOLOE (`<base>-visionx.pt`, armado por `build_vocabulary_weights`)."""

    def __init__(self, settings: DetectorSettings) -> None:
        self.settings = settings
        path = settings.weights_path
        if not path.is_file():
            raise EstimatorError(
                f"no están los pesos {path}: bajarlos con "
                f"`python -m visionx.detection descargar --modelo {settings.model}`"
            )
        YOLO, torch = _import_ultralytics()
        self._torch = torch
        self.device = _resolve_device(settings.device, torch)
        self.weights_sha256 = sha256_file(path)
        # El ID incluye el hash: dos archivos con el mismo nombre no se confunden en resultados.
        self.model_id = f"{settings.model}@{self.weights_sha256[:12]}"
        try:
            self._model = YOLO(str(path))
        except Exception as exc:  # pesos corruptos o de otra versión
            raise EstimatorError(f"no se pudo cargar {path}: {exc}") from exc
        names = self._model.names
        self.class_names: tuple[str, ...] = tuple(names[i] for i in sorted(names))
        self.catalog = ClassCatalog(self.class_names)
        self._class_filter: list[int] | None = None
        if settings.classes is not None:
            unknown = [c for c in settings.classes if c not in self.class_names]
            if unknown:
                raise ValueError(f"detection.classes: {unknown} no están en {settings.model}")
            self._class_filter = [self.class_names.index(c) for c in settings.classes]
        # IDs propios en vez de model.track(): ver tracking.py. Una instancia por detector.
        self._ids = ByteTrackIds(settings.tracker) if settings.track else None
        self._session: str | None = None
        self.last_speed_ms: dict[str, float] = {}

    def predict(self, frame: StereoFrame) -> Detection:
        require_geometry(frame)
        s = self.settings
        t0 = time.perf_counter()
        try:
            results = self._model.predict(
                frame.left,
                imgsz=s.imgsz,
                conf=s.conf,
                iou=s.iou,
                device=self.device,
                classes=self._class_filter,
                # Máscaras en la resolución de la imagen original. Sin esto vienen a la
                # resolución de la red con el relleno del letterbox, y reescalarlas a mano
                # corre el contorno.
                retina_masks=True,
                verbose=False,
            )
            # En su primera predicción Ultralytics fija los hilos de torch en núcleos-1 (deja uno
            # libre, útil para la captura). Si la config pide otro número, se impone después.
            if s.threads and self._torch.get_num_threads() != s.threads:
                self._torch.set_num_threads(s.threads)
            result = results[0]
            boxes = result.boxes.cpu().numpy()
            track_ids = None
            if self._ids is not None:
                # Otra sesión (otra grabación, o una reconexión con session nuevo) no continúa
                # la anterior: sus IDs no deben heredarse.
                if frame.session_id != self._session:
                    self._ids.reset()
                    self._session = frame.session_id
                track_ids = self._ids.assign(boxes, frame.left)
        except Exception as exc:
            raise EstimatorError(
                f"{self.model_id} falló en el frame {frame.frame_id}: {exc}"
            ) from exc
        # Tiempo de pared completo (preproceso + red + postproceso + máscaras a resolución
        # original + tracking): es lo que espera la aplicación, no solo el forward de la red.
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_speed_ms = dict(getattr(result, "speed", {}) or {})
        if len(boxes) == 0:
            return empty_detection(frame, model_id=self.model_id, inference_ms=elapsed_ms)
        masks = None if result.masks is None else result.masks.data.cpu().numpy()
        return build_detection(
            frame,
            model_id=self.model_id,
            boxes_xyxy=boxes.xyxy,
            class_names=[self.class_names[int(i)] for i in boxes.cls],
            scores=boxes.conf,
            inference_ms=elapsed_ms,
            masks=masks,
            track_ids=track_ids,
        )

    def close(self) -> None:
        self._model = None
