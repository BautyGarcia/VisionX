"""¿Qué detector encuentra más objetos, y con cuántos errores? Medido contra el GT sintético.

Cada vista del render sabe qué objeto hay en cada píxel (`inst.png` + `vista.json`), así que se
puede medir un detector sin anotar a mano:

- **recall por clase:** de los objetos visibles (≥ `min_px` píxeles, a ≤ 6 m), qué fracción tiene
  una detección de su clase cuya caja se superpone con la del objeto (IoU ≥ `iou`);
- **precisión (cota inferior):** de las detecciones de clases que el GT etiqueta, cuántas caen
  sobre un objeto de esa clase. Es una cota: Replica no etiqueta todo (llaves, cables…), así que
  una detección correcta de algo sin etiqueta cuenta como error.

Las clases del render se traducen a las del detector con `REPLICA_A_VISIONX`; las que no tienen
equivalente no cuentan. Con un detector COCO, las clases que no conoce quedan sin medir (su
recall no se informa en vez de contar como cero).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from visionx.gtsynth.dataset import Vista

# Clase de Replica (y de HSSD, con sus nombres) -> clase del vocabulario de VisionX.
REPLICA_A_VISIONX: dict[str, str] = {
    "lamp": "lamp", "table": "dining table", "chair": "chair", "indoor-plant": "potted plant",
    "bottle": "bottle", "tv-screen": "tv", "box": "box", "camera": "camera", "plate": "plate",
    "bin": "trash can", "cabinet": "cabinet", "vase": "vase", "rack": "shelf", "shelf": "shelf",
    "sofa": "couch", "cushion": "pillow", "bowl": "bowl", "cup": "cup", "mat": "rug",
    "rug": "rug", "sink": "sink", "chopping-board": "cutting board", "clothing": "clothes",
    "basket": "basket", "blanket": "blanket", "comforter": "blanket", "tablet": "tablet",
    "refrigerator": "refrigerator", "bike": "bicycle", "scarf": "scarf", "pillow": "pillow",
    "shoe": "shoe", "book": "book", "handbag": "handbag", "bag": "handbag", "clock": "clock",
    "pot": "cooking pot", "tissue-paper": "tissue box", "bed": "bed", "handrail": "handrail",
    "picture": "picture frame", "nightstand": "nightstand", "umbrella": "umbrella",
    "faucet": "faucet", "desk": "dining table", "stool": "chair", "bench": "bench",
    "candle": "candle",
    "monitor": "computer monitor", "pan": "frying pan", "remote-control": "remote",
    "toilet": "toilet", "towel": "towel", "bathtub": "bathtub", "door": "door",
    "window": "window", "blinds": "window blinds", "curtain": "curtain", "stair": "stairs",
    "switch": "light switch", "wall-plug": "power outlet", "ceiling-lamp": "ceiling light",
    "laptop": "laptop", "mouse": "mouse", "computer-keyboard": "keyboard", "phone": "cell phone",
    "microwave": "microwave", "hair-dryer": "hair drier", "toothbrush": "toothbrush",
    "backpack": "backpack", "beanbag": "pillow", "cooktop": "stove",
}  # fmt: skip
# Categorías de HSSD (metadata/fpmodels-with-decomposed.csv) -> clase de VisionX. Las que
# coinciden con un nombre de Replica (chair, table, cabinet…) ya están arriba.
REPLICA_A_VISIONX.update({
    "carpet": "rug", "potted_plant": "potted plant", "shelves": "shelf",
    "ceiling_lamp": "ceiling light", "chest_of_drawers": "dresser", "fridge": "refrigerator",
    "table_lamp": "lamp", "floor_lamp": "lamp", "wall_lamp": "lamp", "stove": "stove",
    "drinkware": "cup", "wall_clock": "clock", "alarm_clock": "clock", "mantel_clock": "clock",
    "tray": "tray", "stairway": "stairs", "trashcan": "trash can", "car": "car",
    "washer_dryer": "washing machine", "shower": "shower", "picture_frame": "picture frame",
    "clothing": "clothes", "plate": "plate", "paper_towel": "paper towel", "oven": "oven",
    "wardrobe": "wardrobe", "plant": "potted plant", "shoes": "shoe",
    "computer_screen": "computer monitor", "quilt": "blanket", "throw": "blanket",
    "phone": "cell phone", "railing": "handrail", "kitchen_scale": "scale",
    "bathroom_scale": "scale", "hamper": "basket", "storage_box": "box", "gift_box": "box",
    "crate": "box", "coffee_maker": "coffee maker", "plush_toy": "teddy bear",
    "mirror": "mirror",
})  # fmt: skip
# Con un detector COCO, la misma clase de Replica se compara con la COCO más cercana.
VISIONX_A_COCO: dict[str, str] = {"computer monitor": "tv"}


@dataclass
class Conteo:
    gt: int = 0
    encontrados: int = 0
    det: int = 0
    det_ok: int = 0

    @property
    def recall(self) -> float | None:
        return self.encontrados / self.gt if self.gt else None

    @property
    def precision(self) -> float | None:
        return self.det_ok / self.det if self.det else None


@dataclass
class Resultado:
    modelo: str
    vistas: int = 0
    por_clase: dict[str, Conteo] = field(default_factory=lambda: defaultdict(Conteo))
    ms: list[float] = field(default_factory=list)

    def total(self, clases: Iterable[str] | None = None) -> Conteo:
        out = Conteo()
        for c, k in self.por_clase.items():
            if clases is not None and c not in clases:
                continue
            out.gt += k.gt
            out.encontrados += k.encontrados
            out.det += k.det
            out.det_ok += k.det_ok
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas a (n, 4) y b (m, 4), formato x0 y0 x1 y1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def objetos_gt(
    vista: Vista, conocidas: set[str], min_px: int = 400, z_max_m: float = 6.0
) -> tuple[np.ndarray, list[str]]:
    """Cajas y clases (del detector) de los objetos visibles que el detector podría nombrar."""
    inst, z = vista.recortar(vista.instancias()), vista.recortar(vista.z())
    cajas, clases = [], []
    for idx in np.unique(inst):
        if idx == 0:
            continue
        cls = REPLICA_A_VISIONX.get(vista.clase(int(idx)))
        if cls is None:
            continue
        cls = cls if cls in conocidas else VISIONX_A_COCO.get(cls, cls)
        if cls not in conocidas:
            continue
        mask = (inst == idx) & (z > 0)
        if mask.sum() < min_px or float(np.percentile(z[mask], 20)) > z_max_m:
            continue
        rows, cols = np.nonzero(mask)
        cajas.append((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
        clases.append(cls)
    return np.asarray(cajas, np.float64).reshape(-1, 4), clases


def evaluar(
    vistas: Iterable[Vista],
    predecir,  # (imagen BGR) -> (cajas (n, 4), clases list[str], ms)
    modelo: str,
    conocidas: set[str],
    iou: float = 0.4,
    min_px: int = 400,
) -> Resultado:
    res = Resultado(modelo)
    for v in vistas:
        cajas_gt, clases_gt = objetos_gt(v, conocidas, min_px)
        cajas_det, clases_det, ms = predecir(v.recortar(v.izq()))
        res.vistas += 1
        res.ms.append(ms)
        m = _iou(cajas_gt, np.asarray(cajas_det, np.float64).reshape(-1, 4))
        etiquetadas = set(clases_gt) | set(REPLICA_A_VISIONX.values())
        for i, c in enumerate(clases_gt):
            k = res.por_clase[c]
            k.gt += 1
            if any(m[i, j] >= iou and clases_det[j] == c for j in range(len(clases_det))):
                k.encontrados += 1
        for j, c in enumerate(clases_det):
            if c not in etiquetadas:  # el GT no etiqueta esa clase: no se puede juzgar
                continue
            k = res.por_clase[c]
            k.det += 1
            if any(m[i, j] >= iou and clases_gt[i] == c for i in range(len(clases_gt))):
                k.det_ok += 1
    return res


def predictor_yolo(path: str, imgsz: int = 960, conf: float = 0.25, device: str = "cpu"):
    """Predictor sobre los pesos de un YOLO de Ultralytics (COCO o YOLOE con vocabulario)."""
    import os
    import time

    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    model = YOLO(path)
    names = model.names

    def predecir(img: np.ndarray):
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        b = r.boxes.cpu().numpy()
        return b.xyxy, [names[int(c)] for c in b.cls], ms

    return predecir, set(names.values())


def escribir_csv(resultados: list[Resultado], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "clase", "gt", "encontrados", "recall", "det", "det_ok", "precision"])
        for r in resultados:
            for c, k in sorted(r.por_clase.items()):
                w.writerow([r.modelo, c, k.gt, k.encontrados,
                            "" if k.recall is None else f"{k.recall:.3f}", k.det, k.det_ok,
                            "" if k.precision is None else f"{k.precision:.3f}"])  # fmt: skip
