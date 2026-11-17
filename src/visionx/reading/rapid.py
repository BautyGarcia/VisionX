"""RapidOCR 3.10 + PP-OCRv5 latino, ONNX Runtime CPU, sin descargas implícitas.

Los tres hashes vienen del catálogo del tag v3.10.0 de RapidAI/RapidOCR.
El import opcional y la carga viven en read(), que el modo llama desde su worker.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

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
        if self._engine is None:
            self._load()
        result = self._engine(image)
        if result.boxes is None or result.txts is None:
            return []
        h, w = image.shape[:2]
        blocks = []
        for polygon, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            xy = np.asarray(polygon)
            x1, y1 = np.maximum(xy.min(axis=0), (0, 0))
            x2, y2 = np.minimum(xy.max(axis=0), (w, h))
            if x2 > x1 and y2 > y1:
                blocks.append(
                    TextBlock(str(text), float(score), (float(x1), float(y1), float(x2), float(y2)))
                )
        return blocks
