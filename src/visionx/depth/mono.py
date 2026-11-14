"""Profundidad monocular preentrenada y congelada (T05): Depth Anything V2 Small.

Qué devuelve: un mapa **relativo** float32 del tamaño de la imagen, donde más grande es más
cerca (es una disparidad relativa, no metros). No tiene escala: se la da la alineación con
anclas estéreo (align.py). No usar nunca la imagen coloreada del mapa.

Cómo se carga: con `transformers` (port oficial en Hugging Face) y una **revisión fija** del
repositorio de pesos, para que el preprocesamiento y los pesos no cambien sin que lo sepamos.
Solo la versión Small (Apache 2.0): Base y Large no permiten uso comercial.

torch y transformers son del extra `ml` (`uv sync --locked --extra ml`). Se importan recién al
crear el modelo, así el resto de VisionX y los tests funcionan sin ellos. Los pesos se bajan
una vez a la caché de Hugging Face (fuera de Git) y se controla su SHA-256 antes de usarlos.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from visionx.shared import _validate as v
from visionx.shared.errors import EstimatorError, InvalidInput


@runtime_checkable
class MonoModel(Protocol):
    """Modelo monocular: imagen BGR uint8 H x W x 3 -> mapa relativo H x W float32."""

    model_id: str

    def predict(self, image_bgr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MonoConfig:
    repo: str = "depth-anything/Depth-Anything-V2-Small-hf"
    revision: str = "5426e4f0f36572d16453bbda7a8389317b1bef99"  # commit del repo de pesos
    weights_sha256: str = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
    device: str = "auto"  # auto | mps | cuda | cpu

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> MonoConfig:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: val for k, val in section.items() if k in names})


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DepthAnythingV2:
    """Depth Anything V2 Small vía `transformers`, en modo evaluación y sin gradientes."""

    def __init__(self, config: MonoConfig | None = None, allow_download: bool = False) -> None:
        self.config = config or MonoConfig()
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise EstimatorError(
                "Depth Anything necesita el extra ml: uv sync --locked --extra ml"
            ) from exc
        c = self.config
        # Sin allow_download solo se usa la caché local: la demo no baja pesos al recibir
        # una orden (TECNICO, "Monocular preentrenada").
        try:
            weights = hf_hub_download(
                c.repo,
                "model.safetensors",
                revision=c.revision,
                local_files_only=not allow_download,
            )
        except Exception as exc:  # huggingface_hub lanza varios tipos según el caso
            raise EstimatorError(
                f"pesos de {c.repo}@{c.revision[:8]} no disponibles en la caché local; "
                "bajarlos una vez con `python -m visionx.depth bajar-mono`"
            ) from exc
        got = sha256_file(weights)
        if got != c.weights_sha256:
            raise EstimatorError(f"SHA-256 de los pesos {got[:12]}… no coincide con la config")
        local = not allow_download
        self._processor = AutoImageProcessor.from_pretrained(
            c.repo, revision=c.revision, local_files_only=local
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            c.repo, revision=c.revision, local_files_only=local
        )
        self.device = pick_device(c.device)
        self._model = model.eval().to(self.device)
        self._torch = torch
        self.model_id = f"da-v2-small@{c.revision[:8]}"
        self.weights_path = weights

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        v.array("image_bgr", image_bgr, np.uint8, 3, (None, None, 3))
        h, w = image_bgr.shape[:2]
        # El contrato usa BGR (OpenCV); la red se entrenó con RGB. La conversión es acá.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        torch = self._torch
        # Preprocesamiento oficial del port: lado corto a 518, múltiplo de 14, normalizado.
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # Vuelve al tamaño original con el post-proceso del propio processor (una sola ruta
        # de remuestreo; el repo oficial advierte diferencias entre implementaciones).
        post = self._processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
        relative = post[0]["predicted_depth"].float().cpu().numpy().astype(np.float32)
        if relative.shape != (h, w) or not np.isfinite(relative).all():
            raise EstimatorError(f"{self.model_id}: salida con forma o valores inválidos")
        return relative

    def timed_predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """predict + tiempo en ms, sincronizando la GPU para no medir solo el encolado."""
        start = time.perf_counter()
        out = self.predict(image_bgr)  # .cpu() ya sincroniza
        return out, (time.perf_counter() - start) * 1000.0


def download(config: MonoConfig | None = None) -> str:
    """Baja (una vez) los pesos fijados y devuelve la ruta local, con el hash ya controlado."""
    model = DepthAnythingV2(config, allow_download=True)
    return model.weights_path


def check_relative(relative: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Valida un mapa relativo recibido de cualquier MonoModel (también los fakes de tests)."""
    m = v.array("mono", relative, np.float32, 2, shape)
    if not np.isfinite(m).all():
        raise InvalidInput("mapa monocular con NaN o infinitos")
    return m
