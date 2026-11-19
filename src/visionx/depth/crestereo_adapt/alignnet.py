"""Red de alineación: cuánto hay que mover cada píxel del ojo derecho para alinearlo.

Es la primera red propia de CREStereo adaptable (TECNICO, «CREStereo adaptable»). Recibe un par
ya rectificado con la calibración nominal (que puede no describir del todo a la cámara) y
devuelve un **campo de corrección**: para cada píxel de la derecha corregida, de dónde tomarlo en
la derecha que llegó. Es lo mismo que hace `cv2.remap` con unos mapas.

Cómo está armada, de la entrada a la salida:

1. **Rasgos.** Un mismo codificador chico procesa los dos ojos y los reduce 8 veces.
2. **Correlación.** Para cada lugar del ojo derecho se compara su rasgo con los del izquierdo
   corridos a muchas posiciones: hacia la derecha (la disparidad, que no se conoce) y unas pocas
   filas arriba y abajo. De ahí sale cuántas filas está corrido y cuán seguro es eso.
3. **Cabeza.** Unas convoluciones afinan esa medición y dicen cuánto confiar en cada lugar. Acá
   entra la calibración: por un vector (FiLM) y por el mapa de rayos.
4. **Ajuste geométrico, sin pesos.** Las mediciones de filas se convierten en seis números
   (pitch, yaw, roll, aumento y dos términos radiales) con cuadrados mínimos pesados por la
   confianza. Es la misma cuenta del estimador clásico (`classic.py`, `lens.py`) pero dentro de
   la red, así el gradiente pasa por ella. La geometría la hace la cuenta; la red solo aprende
   a medir.
5. **Resto denso.** Una salida chica para lo que esos seis números no explican (por ejemplo, una
   distorsión de otra forma). Arranca en cero y se lleva a resolución completa interpolando,
   así que es suave.

Lo que **no** intenta: el corrimiento constante de columnas (centro óptico y yaw corren todas
las columnas casi igual y eso no deja huella en las filas). Eso lo arregla la distancia conocida.

Convención de signos, la de todo el paquete: el campo es de dónde **tomar** cada píxel. Un
`LensCorrection` mueve un punto p de la derecha que llegó a C(p); el campo correspondiente es,
para pequeños errores, −D(q) con D(q) = C(q) − q. Las columnas de `jacobian_maps` son D por
unidad de cada parámetro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from visionx.depth.crestereo_adapt.lens import LENS_PARAMETERS, LensCorrection, correct_points
from visionx.shared.geometry import StereoGeometry

STRIDE = 8  # la red mide en una grilla 8 veces más chica que la imagen
CALIB_SIZE = 13  # largo del vector de calibración (ver `calibration_vector`)
# Cuánto se espera que valga cada parámetro antes de mirar las imágenes (desvío del "prior").
# Entra en el ajuste como regularización: sin datos, la respuesta tiende a cero.
# Los giros con margen amplio: el prior no tiene que ser lo que limita un corrimiento real (con
# focal chica, 8 px de filas ya son 3° de pitch).
PRIOR_SIGMA = (math.radians(4.0), math.radians(2.0), math.radians(4.0), 0.05, 0.05, 0.05)


def grid_centers(size: tuple[int, int], stride: int = STRIDE) -> np.ndarray:
    """Centros de las celdas de la grilla reducida, en píxeles de la imagen: (h, w, 2)."""
    width, height = size
    xs = np.arange(width // stride) * stride + (stride - 1) / 2.0
    ys = np.arange(height // stride) * stride + (stride - 1) / 2.0
    u, v = np.meshgrid(xs, ys)
    return np.stack([u, v], axis=-1)


def jacobian_maps(geometry: StereoGeometry, stride: int = STRIDE) -> np.ndarray:
    """Cuánto mueve cada parámetro cada punto de la grilla: (12, h, w) float32.

    Canales 2i y 2i+1: desplazamiento en columnas y en filas (px) por unidad del parámetro i
    (radianes para los giros, sin unidades para el resto). Se calcula con la cuenta exacta de
    `lens.correct_points` y un paso chico, una vez por cámara.
    """
    centers = grid_centers(geometry.size, stride)
    flat = centers.reshape(-1, 2)
    step = 1e-4
    out = []
    for index in range(len(LENS_PARAMETERS)):
        vector = np.zeros(len(LENS_PARAMETERS))
        vector[index] = step
        moved = correct_points(LensCorrection.from_vector(vector), geometry, flat)
        delta = (moved - flat) / step
        out.extend([delta[:, 0], delta[:, 1]])
    h, w = centers.shape[:2]
    return np.stack(out).reshape(2 * len(LENS_PARAMETERS), h, w).astype(np.float32)


def ray_maps(geometry: StereoGeometry, stride: int = STRIDE) -> np.ndarray:
    """Dirección en el espacio que mira cada celda (rayo de largo 1): (3, h, w) float32."""
    centers = grid_centers(geometry.size, stride)
    x = (centers[..., 0] - geometry.cx) / geometry.fx
    y = (centers[..., 1] - geometry.cy) / geometry.fy
    norm = np.sqrt(x * x + y * y + 1.0)
    return np.stack([x / norm, y / norm, 1.0 / norm]).astype(np.float32)


def calibration_vector(
    geometry: StereoGeometry, distortion: np.ndarray | None, known: bool = True
) -> np.ndarray:
    """Lo que la red sabe de la cámara, en 13 números sin unidades.

    focal y centro relativos al tamaño, los primeros 8 coeficientes de distorsión y una marca de
    si la calibración se conoce. Sin calibración todo va en cero salvo la marca: la red aprende
    que eso significa «cámara tipo ELP».
    """
    out = np.zeros(CALIB_SIZE, dtype=np.float32)
    if not known:
        return out
    out[0] = geometry.fx / geometry.width
    out[1] = geometry.fy / geometry.width
    out[2] = geometry.cx / geometry.width - 0.5
    out[3] = geometry.cy / geometry.height - 0.5
    if distortion is not None:
        d = np.asarray(distortion, dtype=np.float32).ravel()[:8]
        out[4 : 4 + len(d)] = np.clip(d, -5.0, 5.0)
    out[12] = 1.0
    return out


@dataclass(frozen=True)
class AlignConfig:
    width: int = 64  # canales de los rasgos que se correlacionan
    head_width: int = 96
    rows: int = 3  # filas buscadas hacia arriba y abajo, en celdas (±24 px a esta escala)
    columns: int = 16  # columnas buscadas hacia la derecha (disparidad), en celdas
    b_floor_px: float = 0.1  # incertidumbre mínima de una medición de filas
    residual_scale_px: float = 2.0  # una unidad de la salida densa son 2 px


def _build_torch() -> dict:
    """Define las clases de PyTorch recién cuando se usan: el módulo se importa sin torch."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )

    class Encoder(nn.Module):
        """Imagen (N, 3, H, W) en 0-1 -> rasgos (N, C, H/8, W/8) de largo 1 por posición."""

        def __init__(self, width: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                block(3, 32, 2),
                block(32, 32),
                block(32, 64, 2),
                block(64, 64),
                block(64, 96, 2),
                block(96, 96),
            )
            self.out = nn.Conv2d(96, width, 1)

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.net(image * 2.0 - 1.0)
            # Normalizados: la correlación es un coseno entre -1 y 1, sin depender del brillo.
            return F.normalize(self.out(features), dim=1), features

    class FiLM(nn.Module):
        """La calibración regula cada canal: x * (1 + γ) + β. Arranca sin efecto (γ = β = 0)."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(CALIB_SIZE, 64), nn.SiLU(), nn.Linear(64, 2 * channels)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, x: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
            gamma, beta = self.mlp(calib)[..., None, None].chunk(2, dim=1)
            return x * (1.0 + gamma) + beta

    class Net(nn.Module):
        def __init__(self, config: AlignConfig) -> None:
            super().__init__()
            self.config = c = config
            self.encoder = Encoder(c.width)
            n_rows, n_cols = 2 * c.rows + 1, c.columns + 1
            # Entradas de la cabeza: la correlación, la probabilidad de cada fila, los rasgos
            # del ojo derecho y los rayos.
            c_in = n_rows * n_cols + n_rows + 96 + 3
            self.head_in = block(c_in, c.head_width)
            self.head = nn.ModuleList([block(c.head_width, c.head_width) for _ in range(3)])
            self.films = nn.ModuleList([FiLM(c.head_width) for _ in range(3)])
            self.dy_out = nn.Conv2d(c.head_width, 1, 3, padding=1)
            self.b_out = nn.Conv2d(c.head_width, 1, 3, padding=1)
            self.residual_out = nn.Conv2d(c.head_width + 2, 2, 3, padding=1)
            for layer in (self.dy_out, self.residual_out):  # arrancan sin aportar nada
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.b_out.weight)
            nn.init.constant_(self.b_out.bias, math.log(math.expm1(1.0)))  # b = 1 px + piso
            self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
            self.register_buffer("prior", torch.tensor([1.0 / s**2 for s in PRIOR_SIGMA]))
            self.register_buffer(
                "row_offsets", torch.arange(-c.rows, c.rows + 1, dtype=torch.float32)
            )

        def correlation(self, right: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
            """(N, filas, columnas, h, w): coseno entre el derecho y el izquierdo corrido."""
            c = self.config
            h, w = right.shape[-2:]
            padded = F.pad(left, (0, c.columns, c.rows, c.rows))
            out = []
            for dy in range(2 * c.rows + 1):
                row = []
                for dx in range(c.columns + 1):
                    shifted = padded[:, :, dy : dy + h, dx : dx + w]
                    row.append((right * shifted).sum(dim=1))
                out.append(torch.stack(row, dim=1))
            return torch.stack(out, dim=1)

        def forward(
            self,
            left: torch.Tensor,
            right: torch.Tensor,
            jacobian: torch.Tensor,
            rays: torch.Tensor,
            calib: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            c = self.config
            f_left, _ = self.encoder(left)
            f_right, raw_right = self.encoder(right)
            corr = self.correlation(f_right, f_left)  # N, R, D, h, w
            n, n_rows, n_cols, h, w = corr.shape
            temperature = self.log_temperature.exp()
            # Probabilidad de cada corrimiento de filas, sumando sobre todas las disparidades.
            row_score = torch.logsumexp(corr / temperature, dim=2)
            row_prob = torch.softmax(row_score, dim=1)
            # Si el izquierdo aparece s celdas más abajo, el derecho está s celdas más arriba.
            dy_soft = -STRIDE * (row_prob * self.row_offsets[None, :, None, None]).sum(1, True)
            x = torch.cat([corr.reshape(n, n_rows * n_cols, h, w), row_prob, raw_right, rays], 1)
            x = self.head_in(x)
            for layer, film in zip(self.head, self.films, strict=True):
                x = x + film(layer(x), calib)
            dy = dy_soft + self.dy_out(x)
            b = F.softplus(self.b_out(x)) + c.b_floor_px

            # Ajuste geométrico: J_y θ ≈ −dy, pesado por 1/b² y regularizado por el prior.
            # Las celdas a menos de `rows` del borde de arriba o de abajo no entran: su pareja
            # puede caer fuera de la imagen izquierda y la medición no vale.
            jx, jy = jacobian[:, 0::2], jacobian[:, 1::2]  # N, 6, h, w
            inside = torch.ones_like(b)
            inside[:, :, : c.rows] = 0.0
            inside[:, :, h - c.rows :] = 0.0
            weight = (inside / b**2).reshape(n, 1, -1)
            a = jy.reshape(n, 6, -1)
            normal = (a * weight) @ a.transpose(1, 2) + torch.diag_embed(self.prior.expand(n, -1))
            rhs = -(a * weight) @ dy.reshape(n, -1, 1)
            theta = torch.linalg.solve(normal, rhs)[..., 0]  # N, 6
            parametric = -torch.stack(
                [(jx * theta[:, :, None, None]).sum(1), (jy * theta[:, :, None, None]).sum(1)], 1
            )
            residual = c.residual_scale_px * self.residual_out(
                torch.cat([x, dy - parametric[:, 1:2], parametric[:, 1:2]], 1)
            )
            return {
                "dy": dy,  # filas medidas por celda (px de la imagen de entrada)
                "b": b,  # incertidumbre de esa medición (px)
                "theta": theta,  # pitch, yaw, roll (rad), aumento y radiales
                "parametric": parametric,  # campo de esos seis números (N, 2, h, w)
                "field": parametric + residual,  # campo total en la grilla reducida
            }

    return {"Net": Net}


def build_model(config: AlignConfig | None = None):
    """La red, lista para entrenar. Importa PyTorch recién acá."""
    return _build_torch()["Net"](config or AlignConfig())


def upsample_field(field, size: tuple[int, int]):
    """Campo de la grilla reducida a resolución completa (N, 2, H, W), interpolando.

    Con `align_corners=False` cada valor queda en el centro de su celda, que es donde se midió.
    """
    import torch.nn.functional as F

    width, height = size
    return F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)
