"""Medir la superficie de un objeto dentro de su máscara (T09, área A).

Tres piezas puras (sin estado, sin geometría de cámara), que usa `locator.ObjectLocator`:

1. `mask_to_depth_grid`: la máscara del detector (resolución del frame) llevada a la grilla de
   la profundidad por vecino más cercano, sin promediar bordes.
2. `interior_mask`: una máscara interior **adaptativa**. El borde de la máscara de YOLO suele
   caer un poco afuera del objeto y toma fondo; erosionar un número fijo de píxeles borraría
   objetos finos (cables, el asa de una taza). Acá se erosiona una fracción del radio del
   mayor círculo inscripto (transformada de distancia), con tope, y nunca por debajo de un
   mínimo de píxeles.
3. `nearest_layer`: agrupa la disparidad efectiva en capas y devuelve la más cercana con
   suficiente peso. Se trabaja en disparidad porque el error del estéreo es aproximadamente
   constante en píxeles de disparidad, no en metros: el ancho de cada intervalo es relativo
   (como el 3 % en Z del provisorio) pero no más angosto que `min_px` px, el ruido típico del
   estéreo. Así a lo lejos (d chica) el ruido no parte una superficie en capas falsas.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from visionx.shared.errors import InvalidInput


@lru_cache(maxsize=32)
def _nearest_indices(src_n: int, dst_n: int) -> np.ndarray:
    """Índice de origen de cada píxel de destino al reescalar `src_n` -> `dst_n` por vecino
    más cercano, exactamente como `cv2.INTER_NEAREST_EXACT` (la regla de
    `geometry.resize_mask`): se obtiene reescalando con OpenCV una fila de índices. La regla
    es separable, así que filas y columnas se resuelven por separado."""
    ramp = np.arange(src_n, dtype=np.uint16)[None, :]
    out = cv2.resize(ramp, (dst_n, 1), interpolation=cv2.INTER_NEAREST_EXACT)
    out = out.reshape(-1).astype(np.int64)
    out.flags.writeable = False
    return out


def mask_to_depth_grid(
    mask: np.ndarray, depth_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Máscara bool (H x W, resolución del frame) -> (recorte en la grilla de profundidad, origen).

    Vecino más cercano, el mismo que `geometry.resize_mask` (`cv2.INTER_NEAREST_EXACT`): nunca
    promedia ni inventa bordes. Pero en vez de reescalar la imagen entera (1920x1080 por cada
    detección), solo se muestrea el rectángulo de la grilla chica cuyos píxeles caen sobre la
    máscara.

    Devuelve el recorte (bool) y el origen (columna, fila) del recorte en la grilla de la
    profundidad. Una máscara vacía (o que desaparece al reducir) da un recorte 0 x 0.
    """
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise InvalidInput("se esperaba una máscara bool H x W")
    if max(mask.shape) > np.iinfo(np.uint16).max:
        raise InvalidInput("máscara demasiado grande")
    big_h, big_w = mask.shape
    w, h = depth_size
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return np.zeros((0, 0), bool), (0, 0)
    src_r = _nearest_indices(big_h, h)
    src_c = _nearest_indices(big_w, w)
    # Los índices de origen crecen con el destino: buscar el tramo que cae sobre la máscara.
    r0, r1 = np.searchsorted(src_r, rows[0], "left"), np.searchsorted(src_r, rows[-1], "right")
    c0, c1 = np.searchsorted(src_c, cols[0], "left"), np.searchsorted(src_c, cols[-1], "right")
    if r0 >= r1 or c0 >= c1:
        return np.zeros((0, 0), bool), (0, 0)
    crop = mask[np.ix_(src_r[r0:r1], src_c[c0:c1])]
    return crop, (int(c0), int(r0))


def interior_mask(
    mask: np.ndarray,
    erode_frac: float,
    erode_max_px: float,
    min_px: int,
    image_edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> tuple[np.ndarray, float]:
    """Interior adaptativo de una máscara: (máscara interior, radio de erosión aplicado en px).

    - `dist` = distancia de cada píxel de la máscara al fondo más cercano (L2, en px de la
      grilla de profundidad). Su máximo `r` es el radio del mayor círculo inscripto: mide
      qué tan "gordo" es el objeto, sin depender de su largo.
    - Se conservan los píxeles con `dist > k`, con `k = min(erode_frac · r, erode_max_px)`.
      Como `k < r`, nunca se vacía: un objeto de 1–2 px de ancho (`r` = 1) no se toca y uno
      de 3 px conserva al menos su línea central. El tope evita comerse medio objeto cuando
      está muy cerca.
    - Si quedan menos de `min_px` píxeles (o menos que la máscara entera, si es más chica),
      se vuelve a la máscara sin erosionar: la erosión es para limpiar, no para decidir.
    - `mask` puede ser un recorte. Alrededor se agrega un marco de fondo (ceros), salvo en los
      lados indicados en `image_edges` (izquierda, arriba, derecha, abajo) que coinciden con
      el borde de la imagen: ahí se replica la máscara. Lo que queda fuera de cuadro no
      contamina; un objeto cortado ya se trata con el reencuadre.
    """
    if erode_frac <= 0 or erode_max_px <= 0 or not mask.any():
        return mask, 0.0
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    left, top, right, bottom = image_edges
    if left:
        padded[:, 0] = padded[:, 1]
    if right:
        padded[:, -1] = padded[:, -2]
    if top:
        padded[0, :] = padded[1, :]
    if bottom:
        padded[-1, :] = padded[-2, :]
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    radius = float(dist.max())
    k = min(erode_frac * radius, erode_max_px)
    inner = mask & (dist > k)
    if int(inner.sum()) < min(min_px, int(mask.sum())):
        return mask, 0.0
    return inner, k


def layer_edges(lo: float, hi: float, rel: float, min_px: float) -> np.ndarray:
    """Bordes de intervalos de disparidad efectiva de `lo` a `hi` (px).

    Cada intervalo mide `max(rel · d, min_px)`: relativo (un 3 % en d es un 3 % en Z, igual que
    los intervalos logarítmicos en Z del provisorio) pero nunca más angosto que el ruido del
    estéreo en px. Con `min_px` = 0 es exactamente un histograma logarítmico.
    """
    if rel <= 0 and min_px <= 0:
        raise InvalidInput("layer_bin_rel o layer_bin_min_px tiene que ser > 0")
    edges = [lo]
    while edges[-1] <= hi:
        e = edges[-1]
        edges.append(e + max(rel * e, min_px, 1e-6))
    return np.asarray(edges, np.float64)


def nearest_layer(
    d_eff: np.ndarray,
    min_frac: float = 0.25,
    rel: float = 0.03,
    min_px: float = 0.0,
    floor: float = 0.01,
) -> np.ndarray | None:
    """Selección (bool, misma forma que `d_eff`) de la capa más cercana, o None si no hay.

    `d_eff` = disparidad + doffs (px, > 0) de los píxeles válidos del objeto. Más disparidad
    es más cerca. Una capa es una racha de intervalos con al menos `floor` de los píxeles
    cada uno; los intervalos casi vacíos separan capas, así un poco de ruido delante no se
    toma como objeto y el fondo visto a través de un objeto calado (silla) queda en otra capa.
    Se recorre desde la disparidad mayor y se devuelve la primera capa con al menos
    `min_frac` de los píxeles.
    """
    if d_eff.size == 0:
        return None
    lo, hi = float(d_eff.min()), float(d_eff.max())
    edges = layer_edges(lo, hi, rel, min_px)
    counts, _ = np.histogram(d_eff, bins=edges)
    total = counts.sum()
    occupied = counts >= max(1.0, floor * total)
    j = len(counts) - 1
    while j >= 0:
        if not occupied[j]:
            j -= 1
            continue
        i = j
        while i - 1 >= 0 and occupied[i - 1]:
            i -= 1
        if counts[i : j + 1].sum() / total >= min_frac:
            return (d_eff >= edges[i]) & (d_eff <= edges[j + 1])
        j = i - 1
    return None
