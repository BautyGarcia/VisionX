"""La corrección de un ojo respecto del otro, como cuenta geométrica (sin pesos).

Después de rectificar con una calibración que no describe del todo a la cámara, el ojo derecho
queda un poco girado o con otro aumento. Un mismo punto ya no cae en la misma fila de las dos
imágenes. Esa diferencia se describe con cuatro números:

    roll   giro sobre el eje óptico (la imagen rota como un volante)
    pitch  giro hacia arriba o abajo
    yaw    giro hacia los costados
    focal  cuánto más (o menos) agranda el ojo derecho, como logaritmo del cociente

Para una cámara ideal (sin distorsión, ya rectificada), girarla o cambiarle el aumento mueve
los píxeles con una **homografía**: una matriz 3x3 G que se arma con la matriz K de la cámara.

    G = K · S(focal) · R(roll, pitch, yaw) · K⁻¹        q = G · p

`p` es un píxel de la derecha tal como salió de la rectificación base y `q` es dónde debería
estar. Aplicar G a la imagen derecha la deja alineada con la izquierda.

**Solo se corrige el ojo derecho.** El izquierdo no se toca: su grilla es la referencia donde
se mide la disparidad y donde el detector dibuja las máscaras.

Convención de ejes de cámara: x a la derecha, y hacia abajo, z hacia adelante. Para giros
chicos, con x̂ = (u - cx)/fx e ŷ = (v - cy)/fy, el corrimiento vertical que produce cada uno es

    pitch: -fy · pitch · (1 + ŷ²)     casi el mismo en toda la imagen
    roll:  +fy · roll · x̂             crece hacia los costados
    yaw:   +fy · yaw · x̂ · ŷ          solo se nota en las esquinas
    focal: +fy · focal · ŷ            crece hacia arriba y abajo

Por eso el yaw se estima mal mirando filas: casi no deja huella vertical. Su efecto fuerte es
horizontal (corre toda la disparidad ~fx · yaw), y eso se confunde con profundidad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from visionx.shared.geometry import StereoGeometry

PARAMETERS = ("pitch", "yaw", "roll", "log_focal")


@dataclass(frozen=True)
class Correction:
    """Corrección del ojo derecho. Ángulos en grados; `log_focal` sin unidades."""

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    log_focal: float = 0.0  # ln(aumento real / aumento que supone la calibración)

    def as_vector(self) -> np.ndarray:
        """[pitch, yaw, roll] en radianes y log_focal: el orden de `PARAMETERS`."""
        return np.array(
            [
                math.radians(self.pitch_deg),
                math.radians(self.yaw_deg),
                math.radians(self.roll_deg),
                self.log_focal,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> Correction:
        pitch, yaw, roll, log_focal = (float(x) for x in vector)
        return cls(
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
            log_focal=log_focal,
        )

    @property
    def is_identity(self) -> bool:
        return not np.any(self.as_vector())


def camera_matrix(geometry: StereoGeometry) -> np.ndarray:
    """K del ojo derecho rectificado.

    Los dos ojos rectificados comparten focal y fila del centro. La columna del centro del
    derecho es `cx + doffs` (así define `doffs` la geometría). En la ELP `doffs` vale menos de
    un píxel y casi no cambia nada; en cámaras con los centros muy corridos sí importa.
    """
    return np.array(
        [
            [geometry.fx, 0.0, geometry.cx + geometry.doffs],
            [0.0, geometry.fy, geometry.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def homography(correction: Correction, geometry: StereoGeometry) -> np.ndarray:
    """G: lleva un píxel de la derecha base a donde debería estar (q = G · p)."""
    pitch, yaw, roll, log_focal = correction.as_vector()
    # Vector de rotación (eje por ángulo): x es pitch, y es yaw, z es roll.
    rotation, _ = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float64))
    scale = np.diag([math.exp(log_focal), math.exp(log_focal), 1.0])
    k = camera_matrix(geometry)
    g = k @ scale @ rotation @ np.linalg.inv(k)
    return g / g[2, 2]


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Aplica una homografía a puntos (N, 2) en píxeles."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))]) @ matrix.T
    return hom[:, :2] / hom[:, 2:3]


def displacement_field(matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cuánto mueve una homografía cada píxel: (alto, ancho, 2) con (dx, dy) en px."""
    width, height = size
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    moved = apply_homography(matrix, np.column_stack([u.ravel(), v.ravel()]))
    field = moved - np.column_stack([u.ravel(), v.ravel()])
    return field.reshape(height, width, 2).astype(np.float32)


def warp_image(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Imagen movida por la homografía: el contenido del píxel p pasa a q = matrix · p.

    Lo que queda sin datos se rellena en negro. Sirve para pruebas y para el generador; en la
    app la corrección no se aplica así, sino compuesta con los mapas (`compose_maps`), para
    remuestrear cada cuadro una sola vez.
    """
    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compose_maps(
    map_x: np.ndarray, map_y: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mete la corrección dentro de los mapas de rectificación del ojo derecho.

    Los mapas base dicen, para cada píxel rectificado p, de qué lugar de la foto cruda sale.
    El píxel corregido q = matrix · p tiene que salir del mismo lugar que p, así que el mapa
    nuevo en q vale lo que el mapa base en p. Mover los mapas (que son suaves) y remuestrear
    la foto una vez evita interpolar la imagen dos veces. Fuera de la foto queda -1, que
    `cv2.remap` pinta de negro.
    """
    height, width = map_x.shape[:2]
    out = []
    for m in (map_x, map_y):
        out.append(
            cv2.warpPerspective(
                m.astype(np.float32),
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=-1.0,
            )
        )
    return out[0], out[1]
