"""Geometría de la cámara que estima la red: etiquetas, distorsión sorteada y remuestreo.

La cámara se describe con cinco números (una sola cámara para los dos ojos, PLUG-AND-PLAY §4):

    log(fx / ancho)        focal relativa al ancho de la imagen (la meta de E2 va en la focal)
    cx / ancho − 0,5       centro óptico, relativo
    cy / alto − 0,5
    k1, k2                 distorsión radial de OpenCV, sobre coordenadas normalizadas por fx

El campo de visión es el del pinhole **sin distorsión**: `hfov = 2·atan(ancho / (2·fx))`.

Los renders son pinhole con un lienzo 25 % más grande (12,5 % de margen por lado). Para que la
red vea lentes reales sin renderizar cada una, `mapa_lente` arma, para cada píxel de la imagen
que verá la red, de qué punto del lienzo sale si la cámara tuviera esa distorsión y ese centro.
Las combinaciones que pliegan la imagen (dos radios distintos al mismo píxel) o que piden
píxeles fuera del lienzo se descartan (revisión #85, punto 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOMBRES = ("log_f", "cx", "cy", "k1", "k2")
# Escala típica de cada etiqueta: la pérdida trabaja con los residuos divididos por esto, para
# que ninguna domine por sus unidades (≈ 1 % de focal, 0,5 % del ancho, 0,05 de k1).
ESCALA = np.array([0.01, 0.005, 0.005, 0.05, 0.03], np.float32)


@dataclass(frozen=True)
class Camara:
    """Una cámara pinhole más distorsión radial, en píxeles de la imagen nominal."""

    ancho: int
    alto: int
    fx: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0

    def etiqueta(self) -> np.ndarray:
        return np.array([math.log(self.fx / self.ancho), self.cx / self.ancho - 0.5,
                         self.cy / self.alto - 0.5, self.k1, self.k2], np.float32)  # fmt: skip

    @staticmethod
    def de_etiqueta(y: np.ndarray, ancho: int, alto: int) -> Camara:
        return Camara(ancho, alto, ancho * math.exp(float(y[0])), (float(y[1]) + 0.5) * ancho,
                      (float(y[2]) + 0.5) * alto, float(y[3]), float(y[4]))  # fmt: skip

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.ancho / (2 * self.fx)))


def hfov_a_fx(hfov_deg: float, ancho: int) -> float:
    return ancho / (2 * math.tan(math.radians(hfov_deg) / 2))


def distorsionar(xu: np.ndarray, yu: np.ndarray, k1: float, k2: float):
    """Coordenadas normalizadas sin distorsión → con distorsión (modelo radial de OpenCV)."""
    r2 = xu * xu + yu * yu
    g = 1 + k1 * r2 + k2 * r2 * r2
    return xu * g, yu * g


def sin_distorsion(xd: np.ndarray, yd: np.ndarray, k1: float, k2: float):
    """Inversa de `distorsionar`. Como la distorsión es radial, alcanza con despejar el radio:
    r·(1 + k1·r² + k2·r⁴) = r_d, por Newton desde r = r_d. Si no converge (lente plegada) da
    NaN, y `cabe` la descarta."""
    if k1 == 0 and k2 == 0:
        return np.array(xd, np.float64), np.array(yd, np.float64)
    xd, yd = np.asarray(xd, np.float64), np.asarray(yd, np.float64)
    rd = np.sqrt(xd * xd + yd * yd)
    r = rd.copy()
    for _ in range(40):
        r2 = r * r
        f = r * (1 + k1 * r2 + k2 * r2 * r2) - rd
        df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
        r = r - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
    r2 = r * r
    ok = (np.abs(r * (1 + k1 * r2 + k2 * r2 * r2) - rd) < 1e-7) & (r >= 0)
    escala = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
    escala = np.where(ok, escala, np.nan)
    return xd * escala, yd * escala


def sin_pliegue(k1: float, k2: float, r_max: float, margen: float = 0.1) -> bool:
    """True si `r·(1 + k1·r² + k2·r⁴)` crece en todo [0, r_max]: d/dr = 1 + 3·k1·r² + 5·k2·r⁴."""
    r = np.linspace(0.0, r_max, 256)
    return bool(np.all(1 + 3 * k1 * r**2 + 5 * k2 * r**4 > margen))


def _borde(ancho: int, alto: int, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Píxeles del borde de la imagen (donde la distorsión pide más del lienzo)."""
    xs, ys = np.linspace(-0.5, ancho - 0.5, n), np.linspace(-0.5, alto - 0.5, n)
    u = np.concatenate([xs, xs, np.full(n, -0.5), np.full(n, ancho - 0.5)])
    v = np.concatenate([np.full(n, -0.5), np.full(n, alto - 0.5), ys, ys])
    return u, v


def cabe(cam: Camara, lienzo: tuple[int, int], centro_lienzo: tuple[float, float]) -> bool:
    """¿Se puede simular `cam` desde el lienzo pinhole sin pliegue ni bordes negros?

    `lienzo` = (ancho, alto) del render y `centro_lienzo` su centro óptico (misma focal
    `cam.fx`). El centro de `cam` está en píxeles de la imagen nominal: correrlo equivale a
    recortar la nominal más a un lado del lienzo.
    """
    u, v = _borde(cam.ancho, cam.alto)
    xd, yd = (u - cam.cx) / cam.fx, (v - cam.cy) / cam.fx
    xu, yu = sin_distorsion(xd, yd, cam.k1, cam.k2)
    r_u = float(np.sqrt(xu * xu + yu * yu).max())
    if not np.all(np.isfinite(xu)) or not sin_pliegue(cam.k1, cam.k2, 1.05 * r_u):
        return False
    xr, yr = distorsionar(xu, yu, cam.k1, cam.k2)  # la inversa tiene que haber convergido
    if max(np.abs(xr - xd).max(), np.abs(yr - yd).max()) > 1e-4:
        return False
    X, Y = cam.fx * xu + centro_lienzo[0], cam.fx * yu + centro_lienzo[1]
    ok_x = X.min() >= -0.5 and X.max() <= lienzo[0] - 0.5
    ok_y = Y.min() >= -0.5 and Y.max() <= lienzo[1] - 0.5
    return bool(ok_x and ok_y)


def mapa_lente(
    cam: Camara,
    salida: tuple[int, int],
    centro_lienzo: tuple[float, float],
    escala_lienzo: float = 1.0,
    paso: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Mapas de `cv2.remap` para ver el lienzo pinhole a través de `cam`, a tamaño `salida`.

    `salida` = (ancho, alto) de la imagen para la red (la nominal escalada, sin recortar).
    `escala_lienzo` es por cuánto se achicó el lienzo antes de remuestrear (para no hacer
    aliasing al bajar de 1600 a 640 px). La inversa de la distorsión se calcula en una grilla
    cada `paso` píxeles y se interpola: la distorsión es suave.
    """
    wo, ho = salida
    sx, sy = cam.ancho / wo, cam.alto / ho
    gx = np.arange(0, wo + paso, paso, dtype=np.float64)
    gy = np.arange(0, ho + paso, paso, dtype=np.float64)
    uo, vo = np.meshgrid(gx, gy)
    u, v = (uo + 0.5) * sx - 0.5, (vo + 0.5) * sy - 0.5  # píxel de la imagen nominal
    xu, yu = sin_distorsion((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fx, cam.k1, cam.k2)
    X = (cam.fx * xu + centro_lienzo[0] + 0.5) * escala_lienzo - 0.5
    Y = (cam.fx * yu + centro_lienzo[1] + 0.5) * escala_lienzo - 0.5
    # De la grilla gruesa (un punto cada `paso` px desde 0) a cada píxel de la salida.
    return _alinear(X, paso, wo, ho), _alinear(Y, paso, wo, ho)


def _alinear(g: np.ndarray, paso: int, wo: int, ho: int) -> np.ndarray:
    """Interpola bilinealmente una grilla con un punto cada `paso` px (esquinas alineadas)."""
    xs, ys = np.arange(wo) / paso, np.arange(ho) / paso
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = (xs - x0)[None, :], (ys - y0)[:, None]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)


def grilla_rayos(cam: Camara, nx: int = 7, ny: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Rayos (coordenadas sin distorsión, tan del ángulo) y los píxeles donde caen en `cam`.

    Sirven para la pérdida geométrica: se proyectan los mismos rayos con la cámara estimada y se
    mide en píxeles cuánto se aleja de donde cayeron de verdad (un error de focal que no cambia
    la imagen no se castiga de más).
    """
    u = np.linspace(0.05, 0.95, nx) * cam.ancho
    v = np.linspace(0.05, 0.95, ny) * cam.alto
    uu, vv = np.meshgrid(u, v)
    xu, yu = sin_distorsion((uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fx, cam.k1, cam.k2)
    rayos = np.stack([xu.ravel(), yu.ravel()], -1).astype(np.float32)
    pix = np.stack([uu.ravel(), vv.ravel()], -1).astype(np.float32)
    return rayos, pix
