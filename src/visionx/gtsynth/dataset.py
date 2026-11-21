"""Lectura de vistas y tabla de objetos con distancia de referencia."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from visionx.shared.geometry import StereoGeometry

# Partición por escena (no por vista), fija. `test` no se mira hasta la evaluación final;
# `val` sirve para ajustar umbrales y el factor de b; `train` queda para entrenar (p. ej. la red
# de entrada) sin tocar las otras dos.
PARTICION: dict[str, tuple[str, ...]] = {
    "train": ("apartment_0", "apartment_1", "frl_apartment_0", "frl_apartment_1",
              "frl_apartment_2", "office_0", "office_1", "room_0"),
    "val": ("frl_apartment_3", "office_2", "room_1", "hotel_0"),
    "test": ("apartment_2", "frl_apartment_4", "frl_apartment_5", "office_3", "office_4",
             "room_2"),
}  # fmt: skip

# Clase de Replica -> clase COCO del detector (detection/catalog.py). None = sin equivalente:
# el objeto tiene distancia de referencia pero el detector no lo puede nombrar.
REPLICA_A_COCO: dict[str, str] = {
    "backpack": "backpack",
    "bag": "handbag",
    "handbag": "handbag",
    "bed": "bed",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "chair": "chair",
    "clock": "clock",
    "computer-keyboard": "keyboard",
    "cup": "cup",
    "hair-dryer": "hair drier",
    "indoor-plant": "potted plant",
    "laptop": "laptop",
    "microwave": "microwave",
    "monitor": "tv",
    "tv-screen": "tv",
    "mouse": "mouse",
    "phone": "cell phone",
    "refrigerator": "refrigerator",
    "remote-control": "remote",
    "sink": "sink",
    "sofa": "couch",
    "table": "dining table",
    "toilet": "toilet",
    "toothbrush": "toothbrush",
    "umbrella": "umbrella",
    "vase": "vase",
}

# Estructura y cosas a las que nadie pregunta la distancia: no entran en la tabla de objetos.
# La misma lista está en scripts/gt_sintetico/render_habitat.py (corre en otro entorno).
NO_OBJETO = frozenset({
    "wall", "floor", "ceiling", "undefined", "wall-plug", "wall-cabinet", "beam", "pipe",
    "vent", "panel", "rug", "base-cabinet", "blinds", "curtain", "window", "door",
    "anonymize_text", "anonymize_picture", "pillar", "stair", "non-plane", "major-appliance",
    "switch", "cloth", "ceiling-lamp", "wall-plaque", "shelf",
})  # fmt: skip

FRENTE_PERCENTIL = 20.0

# Geometría con la que se renderizó la variante `elp` (scripts/gt_sintetico/render_habitat.py):
# la de configs/calib/elp-2026-11-11-corregida.yaml con el centro en el medio de la imagen.
# Es la que el sistema «cree tener» al medir cualquier variante.
ELP_RENDER = StereoGeometry(
    1920, 1080, 601.0069353622214, 601.0069353622214, 959.5, 539.5, 0.05980835164566523
)  # igual que spatial.front_percentile: la superficie del frente


def _hssd() -> dict[str, tuple[str, ...]]:
    """Partición de HSSD (125 train / 30 val / 13 test), de su `scene_splits.yaml`."""
    global _HSSD
    if _HSSD is None:
        path = Path(__file__).with_name("hssd_particion.json")
        datos = json.loads(path.read_text(encoding="utf-8"))
        _HSSD = {k: tuple(datos[k]) for k in ("train", "val", "test")}
    return _HSSD


_HSSD: dict[str, tuple[str, ...]] | None = None


def escenas_de(particion: str) -> tuple[str, ...]:
    """Escenas de una partición: las de Replica más las de HSSD."""
    return PARTICION[particion] + _hssd()[particion]


def particion_de(escena: str) -> str | None:
    for nombre in PARTICION:
        if escena in escenas_de(nombre):
            return nombre
    return None


@dataclass(frozen=True)
class Vista:
    """Una vista renderizada: par rectificado, Z e instancias del ojo izquierdo."""

    carpeta: Path

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.carpeta / "vista.json").read_text(encoding="utf-8"))

    @property
    def escena(self) -> str:
        return self.meta["escena"]

    @property
    def variante(self) -> str:
        return self.meta["variante"]

    @property
    def clave(self) -> str:
        return f"{self.variante}/{self.escena}/{self.meta['indice']:04d}"

    @property
    def geometria(self) -> StereoGeometry:
        c = self.meta["camara"]
        return StereoGeometry(c["ancho"], c["alto"], c["fx"], c["fy"], c["cx"], c["cy"],
                              c["base_m"])  # fmt: skip

    @property
    def modelo(self) -> str:
        """«pinhole» u «ojo-de-pez-doble-esfera» (solo para evaluar: no es un par rectificado)."""
        return self.meta["camara"].get("modelo", "pinhole")

    @property
    def nominal(self) -> dict | None:
        """Dónde está la cámara nominal dentro del lienzo con margen (tanda #53), o None.

        Las variantes con margen (`elp-margen`, perfiles, `camara-sorteada`…) se renderizan en un
        lienzo más grande con la misma focal y el centro en el medio: así se puede desalinear el
        ojo derecho sin que entren bordes negros y después recortar la parte nominal.
        """
        n = self.meta.get("nominal")
        c = self.meta["camara"]
        if n is None or (n["ancho"], n["alto"]) == (c["ancho"], c["alto"]):
            return None
        return n

    @property
    def geometria_nominal(self) -> StereoGeometry:
        """La cámara que se simula (el recorte nominal); sin margen, la misma que `geometria`."""
        n, c = self.nominal, self.meta["camara"]
        if n is None:
            return self.geometria
        return StereoGeometry(n["ancho"], n["alto"], c["fx"], c["fy"], n["cx"], n["cy"],
                              c["base_m"])  # fmt: skip

    def recortar(self, img: np.ndarray) -> np.ndarray:
        """Del lienzo al recorte nominal (imagen, Z o instancias); sin margen, la misma imagen."""
        n = self.nominal
        if n is None:
            return img
        return img[n["y0"] : n["y0"] + n["alto"], n["x0"] : n["x0"] + n["ancho"]]

    def izq(self) -> np.ndarray:
        """BGR uint8, como entrega la cámara."""
        return _leer_bgr(self.carpeta / "izq.png")

    def der(self) -> np.ndarray:
        return _leer_bgr(self.carpeta / "der.png")

    def z(self) -> np.ndarray:
        with np.load(self.carpeta / "z.npz") as f:
            return f["z"].astype(np.float32)

    def instancias(self) -> np.ndarray:
        import cv2

        inst = cv2.imread(str(self.carpeta / "inst.png"), cv2.IMREAD_UNCHANGED)
        if inst is None:
            raise FileNotFoundError(self.carpeta / "inst.png")
        return inst.astype(np.int32)

    def clase(self, idx: int) -> str:
        return self.meta["instancias"].get(str(idx), "otro")


@dataclass(frozen=True)
class VistaNominal(Vista):
    """La misma vista vista por la cámara nominal: imágenes, Z e instancias recortadas.

    Para medir una cámara con margen (`gt-video`, perfiles) con su propia geometría: la
    profundidad y el ubicador trabajan sobre el recorte, como lo haría la cámara real.
    """

    @property
    def geometria(self) -> StereoGeometry:
        return Vista(self.carpeta).geometria_nominal

    @property
    def nominal(self) -> dict | None:
        return None

    def _recorte(self, img: np.ndarray) -> np.ndarray:
        return Vista(self.carpeta).recortar(img)

    def izq(self) -> np.ndarray:
        return np.ascontiguousarray(self._recorte(super().izq()))

    def der(self) -> np.ndarray:
        return np.ascontiguousarray(self._recorte(super().der()))

    def z(self) -> np.ndarray:
        return np.ascontiguousarray(self._recorte(super().z()))

    def instancias(self) -> np.ndarray:
        return np.ascontiguousarray(self._recorte(super().instancias()))


def _leer_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def vistas(
    raiz: str | Path, variante: str = "elp", particion: str | None = None
) -> Iterator[Vista]:
    """Vistas de una variante, ordenadas; opcionalmente solo las de una partición."""
    base = Path(raiz) / variante
    escenas = set(escenas_de(particion)) if particion else None
    for json_path in sorted(base.glob("*/*/vista.json")):
        if escenas is not None and json_path.parent.parent.name not in escenas:
            continue
        yield Vista(json_path.parent)


@dataclass(frozen=True)
class ObjetoGT:
    """Un objeto visible en el ojo izquierdo, con su distancia de referencia."""

    vista: str
    id: int
    clase: str  # de Replica
    coco: str | None  # equivalente del detector, si hay
    px: int  # píxeles visibles
    z_frente_m: float  # percentil 20 de Z en la parte visible (la definición del ubicador)
    z_mediana_m: float
    angulo_h_deg: float  # del centroide visible; positivo a la derecha
    angulo_v_deg: float  # positivo hacia arriba
    caja: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    cortado: bool  # toca un borde de la imagen
    apuntado: bool  # es el objeto al que se apuntó la cámara


def objetos_de(vista: Vista, min_px: int = 400, z_max_m: float = 6.0) -> list[ObjetoGT]:
    """Objetos visibles de una vista (sin paredes, piso ni techo), de cerca a lejos."""
    inst = vista.instancias()
    z = vista.z()
    g = vista.geometria
    h, w = inst.shape
    apuntado = int(vista.meta["apuntado"]["id"])
    out = []
    for idx in np.unique(inst):
        idx = int(idx)
        clase = vista.clase(idx)
        if idx == 0 or clase == "otro" or clase in NO_OBJETO:
            continue
        mask = (inst == idx) & np.isfinite(z) & (z > 0)
        n = int(mask.sum())
        if n < min_px:
            continue
        zz = z[mask]
        z_frente = float(np.percentile(zz, FRENTE_PERCENTIL))
        if z_frente > z_max_m:
            continue
        rows, cols = np.nonzero(mask)
        u, v_ = float(cols.mean()), float(rows.mean())
        ang_h = math.degrees(math.atan2(u - g.cx, g.fx))
        ang_v = math.degrees(math.atan2(g.cy - v_, g.fy))
        x0, x1, y0, y1 = int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())
        out.append(ObjetoGT(
            vista=vista.clave, id=idx, clase=clase, coco=REPLICA_A_COCO.get(clase), px=n,
            z_frente_m=z_frente, z_mediana_m=float(np.median(zz)),
            angulo_h_deg=ang_h, angulo_v_deg=ang_v, caja=(x0, y0, x1, y1),
            cortado=x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1,
            apuntado=idx == apuntado,
        ))  # fmt: skip
    return sorted(out, key=lambda o: o.z_frente_m)


def consistencia(vista: Vista, max_z_m: float = 8.0) -> dict[str, float]:
    """¿La Z del render coincide con el par? Reproyecta la derecha con la disparidad exacta.

    Con Z axial y la geometría correcta, la izquierda reconstruida desde la derecha se parece a
    la real salvo en oclusiones. Se compara contra no mover nada (disparidad 0) y contra una
    disparidad 10 % mayor y menor: la exacta tiene que ganar. Se mide el error medio en los
    píxeles con textura horizontal (en una pared lisa cualquier disparidad da lo mismo).
    """
    import cv2

    g = vista.geometria
    izq = cv2.cvtColor(vista.izq(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    der = cv2.cvtColor(vista.der(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    z = vista.z()
    ok = np.isfinite(z) & (z > 0.3) & (z < max_z_m)
    d = np.where(ok, g.fx * g.baseline_m / np.where(ok, z, 1.0), 0.0).astype(np.float32)
    h, w = z.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Solo donde hay textura horizontal: en una pared lisa cualquier disparidad da error 0.
    textura = np.abs(cv2.Sobel(izq, cv2.CV_32F, 1, 0, ksize=3)) > 20
    usar = ok & textura

    def error(escala: float) -> float:
        rec = cv2.remap(der, uu - d * escala, vv, cv2.INTER_LINEAR, borderValue=np.nan)
        diff = np.abs(rec - izq)[usar & np.isfinite(rec)]
        return float(np.mean(diff)) if diff.size else float("nan")

    return {"exacta": error(1.0), "sin_mover": error(0.0), "mas_10pct": error(1.1),
            "menos_10pct": error(0.9)}  # fmt: skip
