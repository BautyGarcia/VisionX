"""Datos para la red que estima la cámara (E2 de PLUG-AND-PLAY).

Cada muestra es un par de una vista renderizada con lienzo de margen (`camara-sorteada`,
`hssd-camara-sorteada`, recorridos…), visto a través de una cámara simulada:

1. Se sortea la lente (`k1, k2`) y un corrimiento del centro (±3 % del ancho/alto), y se descarta
   lo que no cabe en el lienzo o pliega la imagen (`lente.cabe`).
2. Los dos ojos se remuestrean con esa cámara directo a 640 px de ancho, **sin recortar**: el
   campo de visión tiene que llegar intacto a la red.
3. Aumentos de sensor: ruido, desenfoque, JPEG, viñeteo, exposición distinta entre ojos y gris.

La etiqueta es la de la cámara simulada (`lente.Camara.etiqueta`). En evaluación no se sortea
nada salvo que se pida (`lente_eval`), con una semilla fija por vista.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.depth.camara_net.lente import Camara, cabe, grilla_rayos, mapa_lente

ANCHO_RED = 640  # ancho de la imagen que ve la red (el alto sigue la proporción)
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, RGB
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Aumento:
    """Qué se sortea al armar una muestra. `Aumento.ninguno()` para evaluar tal cual."""

    p_lente: float = 0.8  # probabilidad de simular una lente con distorsión
    k1: tuple[float, float] = (-0.35, 0.10)
    k2: tuple[float, float] = (-0.10, 0.10)
    centro: float = 0.03  # corrimiento máximo del centro, fracción del ancho / alto
    fotometrico: bool = True

    @staticmethod
    def ninguno() -> Aumento:
        return Aumento(p_lente=0.0, centro=0.0, fotometrico=False)


def alto_red(ancho: int, alto: int) -> int:
    """Alto de la imagen para la red: misma proporción, múltiplo de 8."""
    return max(8, int(round(ANCHO_RED * alto / ancho / 8)) * 8)


@dataclass(frozen=True)
class Item:
    """Una vista con lo necesario para muestrearla sin abrir las imágenes."""

    carpeta: str
    lienzo: tuple[int, int]
    centro_lienzo: tuple[float, float]
    nominal: tuple[int, int]  # (ancho, alto)
    centro_nominal: tuple[float, float]
    fx: float
    grupo: str  # cámara (para juntar cuadros de un recorrido) o la vista misma

    @property
    def alto_red(self) -> int:
        return alto_red(*self.nominal)


def item_de(vista: Any) -> Item:
    """De una `visionx.gtsynth.dataset.Vista` con lienzo de margen a un `Item`."""
    c, n = vista.meta["camara"], vista.nominal
    if n is None:  # sin margen: el lienzo es la propia imagen (solo sirve sin lente ni corrimiento)
        n = {"ancho": c["ancho"], "alto": c["alto"], "cx": c["cx"], "cy": c["cy"]}
    sec = vista.meta.get("secuencia")
    grupo = (f"{vista.variante}/{vista.escena}/r{sec['recorrido']}" if sec
             else vista.clave)  # fmt: skip
    return Item(str(vista.carpeta), (int(c["ancho"]), int(c["alto"])),
                (float(c["cx"]), float(c["cy"])), (int(n["ancho"]), int(n["alto"])),
                (float(n["cx"]), float(n["cy"])), float(c["fx"]), grupo)  # fmt: skip


def sortear_camara(item: Item, aum: Aumento, rng: np.random.Generator) -> Camara:
    """Cámara simulada para este ítem: centro corrido y lente, siempre dentro del lienzo."""
    w, h = item.nominal
    base = Camara(w, h, item.fx, *item.centro_nominal)
    for _ in range(30):
        dx = rng.uniform(-aum.centro, aum.centro) * w if aum.centro else 0.0
        dy = rng.uniform(-aum.centro, aum.centro) * h if aum.centro else 0.0
        k1 = k2 = 0.0
        if aum.p_lente and rng.random() < aum.p_lente:
            k1, k2 = rng.uniform(*aum.k1), rng.uniform(*aum.k2)
        cam = Camara(w, h, item.fx, base.cx - dx, base.cy - dy, k1, k2)
        if cabe(cam, item.lienzo, item.centro_lienzo):
            return cam
    return base


def _ruido_sensor(img: np.ndarray, rng: np.random.Generator, gris: bool) -> np.ndarray:
    """Aumentos de un ojo (float32 RGB en [0, 1]): exposición, color, viñeteo, desenfoque,
    ruido y JPEG. `gris` lo deciden los dos ojos juntos (RealSense y OAK-D ven en gris)."""
    h, w = img.shape[:2]
    img = img * rng.uniform(0.7, 1.3) * rng.uniform(0.95, 1.05, 3).astype(np.float32)
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:  # viñeteo: oscurece hacia los bordes
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        img = img * (1 - rng.uniform(0, 0.4) * r2)[..., None]
    if gris:
        img = np.repeat(img.mean(-1, keepdims=True), 3, -1)
    if rng.random() < 0.3:
        s = rng.uniform(0.3, 1.2)
        img = cv2.GaussianBlur(img, (0, 0), s)
    img = img + rng.normal(0, rng.uniform(0, 0.025), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", (img * 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])  # fmt: skip
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, -1)
    return img.astype(np.float32)


def _ver(path: Path, cam: Camara, item: Item, salida: tuple[int, int]) -> np.ndarray:
    """Un ojo visto con `cam`, a tamaño `salida`, RGB float32 en [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    # Achicar el lienzo antes de remuestrear (≈ 1,5 veces la salida) evita el aliasing.
    q = min(1.0, 1.5 * salida[0] / cam.ancho)
    chico = cv2.resize(img, (round(img.shape[1] * q), round(img.shape[0] * q)),
                       interpolation=cv2.INTER_AREA) if q < 1 else img  # fmt: skip
    qx, qy = chico.shape[1] / img.shape[1], chico.shape[0] / img.shape[0]
    mx, my = mapa_lente(cam, salida, item.centro_lienzo, 1.0)
    mx = (mx + 0.5) * qx - 0.5
    my = (my + 0.5) * qy - 0.5
    out = cv2.remap(chico, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def muestra(item: Item, aum: Aumento, rng: np.random.Generator) -> dict[str, Any]:
    """Arrays de una muestra: ojos (2, 3, H, W) normalizados, etiqueta, rayos y píxeles."""
    cam = sortear_camara(item, aum, rng)
    salida = (ANCHO_RED, item.alto_red)
    ojos = []
    gris = aum.fotometrico and rng.random() < 0.2
    for nombre in ("izq.png", "der.png"):
        img = _ver(Path(item.carpeta) / nombre, cam, item, salida)
        if aum.fotometrico:
            img = _ruido_sensor(img, rng, gris)
        ojos.append(((img - MEDIA) / DESVIO).transpose(2, 0, 1))
    rayos, pix = grilla_rayos(cam)
    return {
        "ojos": np.stack(ojos).astype(np.float32),
        "y": cam.etiqueta(),
        "rayos": rayos,
        "pix": pix / np.float32(cam.ancho),  # en anchos de imagen
        "aspecto": np.float32(cam.alto / cam.ancho),
    }


def semilla_de(clave: str, epoca: int = 0) -> int:
    return zlib.crc32(f"{clave}/{epoca}".encode())


class Conjunto:
    """`torch.utils.data.Dataset` sin heredar (así este módulo no importa torch)."""

    def __init__(self, items: list[Item], aum: Aumento, semilla_fija: bool = False) -> None:
        self.items, self.aum, self.semilla_fija = items, aum, semilla_fija

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        it = self.items[i]
        # Evaluar: misma cámara simulada siempre para cada vista. Entrenar: una nueva cada vez.
        rng = np.random.default_rng(semilla_de(it.carpeta) if self.semilla_fija else None)
        out = muestra(it, self.aum, rng)
        out["indice"] = i
        return out


class LotesPorAlto:
    """`batch_sampler` que vuelve a mezclar en cada época (`lotes_por_alto` con otra semilla)."""

    def __init__(self, items: list[Item], tam: int, semilla: int) -> None:
        self.items, self.tam, self.semilla, self.epoca = items, tam, semilla, 0
        self.n = len(lotes_por_alto(items, tam, None))

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.semilla * 1000 + self.epoca)
        self.epoca += 1
        return iter(lotes_por_alto(self.items, self.tam, rng))


def lotes_por_alto(items: list[Item], tam: int, rng: np.random.Generator | None,
                   ) -> list[list[int]]:  # fmt: skip
    """Índices agrupados en lotes del mismo alto de imagen (16:9, 4:3… no se mezclan)."""
    grupos: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        grupos.setdefault(it.alto_red, []).append(i)
    lotes = []
    for idx in grupos.values():
        idx = list(idx)
        if rng is not None:
            rng.shuffle(idx)
        lotes += [idx[k : k + tam] for k in range(0, len(idx), tam)]
    if rng is not None:
        rng.shuffle(lotes)
    return lotes


def juntar(muestras: list[dict[str, Any]]) -> dict[str, Any]:
    """collate_fn: apila arrays de muestras del mismo alto."""
    import torch

    out = {}
    for k in ("ojos", "y", "rayos", "pix"):
        out[k] = torch.from_numpy(np.stack([m[k] for m in muestras]))
    out["aspecto"] = torch.tensor([float(m["aspecto"]) for m in muestras])
    out["indice"] = torch.tensor([m["indice"] for m in muestras])
    return out


def submuestra_recorridos(items: list[Item], por_grupo: int, semilla: int = 0) -> list[Item]:
    """De cada recorrido (misma cámara) se quedan `por_grupo` cuadros: si no, las ~120 cámaras de
    los recorridos pesarían como miles de vistas sueltas."""
    grupos: dict[str, list[Item]] = {}
    for it in items:
        grupos.setdefault(it.grupo, []).append(it)
    rng = np.random.default_rng(semilla)
    out = []
    for g in sorted(grupos):
        lista = grupos[g]
        k = min(por_grupo, len(lista))
        out += [lista[j] for j in sorted(rng.choice(len(lista), k, replace=False))]
    return out


def hfov_de(y: np.ndarray) -> float:
    """Campo horizontal (°) del pinhole de una etiqueta."""
    return math.degrees(2 * math.atan(0.5 / math.exp(float(y[0]))))
