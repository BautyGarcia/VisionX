"""RapidOCR 3.10 + PP-OCRv5 latino, ONNX Runtime CPU, sin descargas implícitas.

Los tres hashes vienen del catálogo del tag v3.10.0 de RapidAI/RapidOCR.
El import opcional y la carga viven en el worker (warmup/read), nunca en el controlador.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from visionx.reading.core import ReadingSettings, TextBlock
from visionx.shared.config import ROOT
from visionx.shared.errors import EstimatorError

MODEL_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5"
# etapa: (nombre, SHA-256)
MODELS = {
    "det": (
        "ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    "rec": (
        "latin_PP-OCRv5_rec_mobile.onnx",
        "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a",
    ),
    "cls": (
        "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
        "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7",
    ),
}


def weights_folder(s: ReadingSettings) -> Path:
    path = Path(s.weights_dir)
    return path if path.is_absolute() else ROOT / path


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download_models(s: ReadingSettings) -> dict[str, str]:
    """Solo este comando usa red; archivo temporal + hash antes del renombrado."""
    folder = weights_folder(s)
    folder.mkdir(parents=True, exist_ok=True)
    for stage, (name, digest) in MODELS.items():
        path = folder / name
        if path.exists():
            if file_hash(path) != digest:
                raise EstimatorError(
                    f"hash OCR incorrecto: {path}; retirar el archivo y reintentar"
                )
            continue
        partial = path.with_suffix(".onnx.parcial")
        try:
            with urllib.request.urlopen(f"{MODEL_BASE}/{stage}/{name}", timeout=60) as response:
                with partial.open("wb") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
            if file_hash(partial) != digest:
                raise EstimatorError(f"hash OCR incorrecto en descarga de {name}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return {name: file_hash(folder / name) for name, _ in MODELS.values()}


class RapidReader:
    model_id = "rapidocr-3.10.0-ppocrv5-latin-cpu"

    def __init__(self, settings: ReadingSettings | None = None) -> None:
        self.settings = settings or ReadingSettings()
        self._engine = None

    def _load(self) -> None:
        folder = weights_folder(self.settings)
        for name, digest in MODELS.values():
            path = folder / name
            if not path.is_file():
                raise EstimatorError("faltan modelos OCR: `python -m visionx.reading descargar`")
            if file_hash(path) != digest:
                raise EstimatorError(f"hash OCR incorrecto: {path}")
        try:
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise EstimatorError("falta OCR: instalar el extra `ocr` con uv sync") from exc
        params = {
            "Global.text_score": 0.0,  # el filtrado es público y compartido con la evaluación
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": self.settings.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.use_coreml": False,
            "Det.limit_side_len": self.settings.det_limit_px,
            "Det.limit_type": "max",  # acotar CPU manteniendo 1280 px para etiquetas pequeñas
            "Rec.lang_type": LangRec.LATIN,
        }
        for stage, (name, _) in MODELS.items():
            prefix = stage.capitalize()
            params.update(
                {
                    f"{prefix}.engine_type": EngineType.ONNXRUNTIME,
                    f"{prefix}.ocr_version": OCRVersion.PPOCRV5,
                    f"{prefix}.model_type": ModelType.MOBILE,
                    f"{prefix}.model_path": str(folder / name),
                }
            )
        self._engine = RapidOCR(params=params)

    def read(self, image: np.ndarray) -> list[TextBlock]:
        return self.read_region(image)

    def warmup(self) -> None:
        """Crea las sesiones ONNX una sola vez, serializado por ReadingWorker."""
        if self._engine is None:
            self._load()

    def read_region(
        self, image: np.ndarray, object_box: tuple[float, float, float, float] | None = None
    ) -> list[TextBlock]:
        """Recorta/escala solo en el worker; devuelve cajas en la imagen de entrada.

        object_box debe pertenecer al mismo frame/geometría (lo valida el consumidor).
        Sin caja válida, object usa text; sin cajas de texto, text usa imagen completa.
        """
        if self._engine is None:
            self._load()
        h, w = image.shape[:2]
        roi = (0, 0, w, h)
        mode = self.settings.crop
        if mode == "center":
            f = self.settings.center_fraction
            roi = (
                int(w * (1 - f) / 2),
                int(h * (1 - f) / 2),
                int(w * (1 + f) / 2),
                int(h * (1 + f) / 2),
            )
        elif mode in ("text", "object"):
            box = object_box if mode == "object" else None
            if box is None:
                scale = min(1.0, self.settings.text_probe_px / max(h, w))
                probe = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
                # RapidOCR conserva use_rec/use_cls entre llamadas: fijarlos en AMBAS pasadas.
                detected = self._engine(probe, use_det=True, use_cls=False, use_rec=False)
                if detected.boxes is not None and len(detected.boxes):
                    points = np.asarray(detected.boxes).reshape(-1, 2)
                    low, high = points.min(axis=0), points.max(axis=0)
                    box = (
                        low[0] * w / probe.shape[1],
                        low[1] * h / probe.shape[0],
                        high[0] * w / probe.shape[1],
                        high[1] * h / probe.shape[0],
                    )
            if box is not None and all(math.isfinite(v) for v in box):
                x1, y1, x2, y2 = box
                mx, my = (
                    (x2 - x1) * self.settings.crop_margin,
                    (y2 - y1) * self.settings.crop_margin,
                )
                proposed = (
                    max(0, math.floor(x1 - mx)),
                    max(0, math.floor(y1 - my)),
                    min(w, math.ceil(x2 + mx)),
                    min(h, math.ceil(y2 + my)),
                )
                if proposed[2] > proposed[0] and proposed[3] > proposed[1]:
                    roi = proposed
        x, y, right, bottom = roi
        cropped = image[y:bottom, x:right]
        if mode != "none":
            scale = self.settings.det_limit_px / max(cropped.shape[:2])
            cropped = cv2.resize(
                cropped,
                (max(1, round(cropped.shape[1] * scale)), max(1, round(cropped.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        self.last_crop = {"mode": mode, "roi": roi, "ocr_size": cropped.shape[:2][::-1]}
        result = self._engine(cropped, use_det=True, use_cls=True, use_rec=True)
        if result.boxes is None or result.txts is None:
            return []
        sx, sy = (right - x) / cropped.shape[1], (bottom - y) / cropped.shape[0]
        blocks = []
        for polygon, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            xy = np.asarray(polygon) * (sx, sy) + (x, y)
            x1, y1 = np.maximum(xy.min(axis=0), (0, 0))
            x2, y2 = np.minimum(xy.max(axis=0), (w, h))
            if x2 > x1 and y2 > y1:
                blocks.append(
                    TextBlock(str(text), float(score), (float(x1), float(y1), float(x2), float(y2)))
                )
        return blocks
