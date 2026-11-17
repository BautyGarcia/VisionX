"""Canales de entrada de Fusion (T12): un único preprocesamiento para entrenar y para aplicar.

Fusion recibe una pila de **canales** (imágenes numéricas de una propiedad, todas del mismo
tamaño) y corrige una **base**: B4 o B5 (`hybrid.py`). El orden y el significado de los
canales forman parte del checkpoint: si cambian, cambia `CHANNELS_VERSION` y los pesos viejos
dejan de servir. Versión 2 agregó F3 sin tocar F0 ni F1: un paquete "fusion-canales-1" con
F0/F1 se sigue cargando (`check_channels`).

Variantes: F0 (10 canales), F1 (+ máscara del monocular) y F3 (F1 + 3 canales de la
corrección local de B5: cuánto corrige, distancia al ancla y densidad de anclas). Los canales
de F3 se calculan sobre el **frame completo**, antes de recortar: así llevan información de
anclas a cientos de px aunque la red vea solo un recorte (BITACORA 2026-11-16).

Unidad de disparidad dentro de Fusion: **d/W_ref**, la disparidad en px dividida por el ancho
en px del frame preparado completo. GT, base B4, residual y b usan esa misma escala; para
volver a px se multiplica por W_ref. Al recortar parches para entrenar se conserva W_ref del
frame entero (`FusionInputs.ref_width`): recortar no cambia la escala de la disparidad, porque
la disparidad de un punto no depende de cuánto de la imagen se mire. Redimensionar sí la
cambia, y eso se hace antes, al preparar el par.

Reglas numéricas (TECNICO, "Entradas y CNN Fusion"):
- Todo canal sale finito. Las convoluciones no saben de máscaras: un NaN contamina a todos
  sus vecinos. Donde una fuente no existe se escribe un valor fijo (0, o el tope del error
  L/R) y la ausencia se comunica con las máscaras de validez, nunca con `NaN * 0`.
- Los errores se expresan en d/W multiplicados por una ganancia fija y con tope, para que
  queden en un rango parecido al de los demás canales (~0–1) sin depender de la resolución.

Solo numpy y OpenCV: no importa torch, así se puede preparar y auditar sin el extra `ml`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import cv2
import numpy as np

from visionx.depth.align import Alignment, AlignParams
from visionx.depth.hybrid import SOURCE_STEREO, B5Params, HybridOutput, combine
from visionx.depth.mono import check_relative
from visionx.depth.stereo import StereoOutput
from visionx.shared import _validate as v
from visionx.shared.errors import InvalidInput

CHANNELS_VERSION = "fusion-canales-2"
F0_CHANNELS = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "disp_estereo",
    "valid_estereo",
    "error_lr",
    "error_foto",
    "gradiente",
    "disp_mono",
    "dif_estereo_mono",
)
F1_CHANNELS = F0_CHANNELS + ("valid_mono",)
F3_CHANNELS = F1_CHANNELS + ("corr_local", "dist_ancla", "densidad_anclas")
VARIANTS = {"F0": F0_CHANNELS, "F1": F1_CHANNELS, "F3": F3_CHANNELS}
# Qué variantes de cada versión vieja siguen valiendo con este código (misma definición).
COMPATIBLE_VERSIONS = {"fusion-canales-1": ("F0", "F1"), CHANNELS_VERSION: tuple(VARIANTS)}
BASES = ("B4", "B5")


def check_channels(version: Any, variant: Any, names: Any = None) -> None:
    """Rechaza con InvalidInput una versión o variante de canales que este código no arma igual.

    `names`: el orden registrado (checkpoint o paquete); si viene, tiene que ser el de la
    variante.
    """
    if version not in COMPATIBLE_VERSIONS:
        raise InvalidInput(f"canales {version!r}, este código usa {CHANNELS_VERSION!r}")
    if variant not in COMPATIBLE_VERSIONS[version]:
        raise InvalidInput(f"variante {variant!r} no existe en los canales {version!r}")
    if names is not None and tuple(names) != VARIANTS[variant]:
        raise InvalidInput(f"orden de canales distinto del de la variante {variant}")


def uses_b5(variant: str, base: str) -> bool:
    """¿Hace falta B5 (y depth.b5) para esta variante y esta base? F3 la usa en sus canales."""
    return base == "B5" or variant == "F3"


@dataclass(frozen=True)
class ChannelParams:
    """Escalas fijas de los canales. Son parte del preprocesamiento: van al checkpoint."""

    error_gain: float = 32.0  # ganancia de los canales de error, en unidades d/W
    lr_cap: float = 1.0  # tope tras la ganancia (y valor cuando L/R no se puede calcular)
    gradient_gain: float = 0.25  # Sobel 3x3 sobre gris/255, por esta ganancia, tope 1
    # F3. corr_local = (d_B5 − d_B4)/W · corr_gain, en [−1, 1]: con 16 y W = 1080 satura a
    # ±67 px; las correcciones típicas (1–10 px) quedan en 0,015–0,15.
    corr_gain: float = 16.0
    # F3. dist_ancla = distancia/W · distance_gain, tope 1: con 4 y W = 1080 satura a 270 px,
    # del orden de 2σ de la corrección local (más lejos ya no hay anclas que pesen).
    distance_gain: float = 4.0

    def __post_init__(self) -> None:
        for f in fields(self):
            if v.finite(f.name, getattr(self, f.name)) <= 0:
                raise InvalidInput(f"{f.name} tiene que ser > 0")

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> ChannelParams:
        """Toma de la sección de config solo las claves que son parámetros de canales."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: val for k, val in section.items() if k in names})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FusionInputs:
    """Entradas de Fusion para un par. Disparidades en d/W_ref."""

    channels: np.ndarray  # C x H x W float32, todo finito
    names: tuple[str, ...]  # orden de canales (forma parte del checkpoint)
    base_norm: np.ndarray  # H x W float32: disparidad de la base / W_ref; 0 donde no es válida
    base_valid: np.ndarray  # H x W bool: dónde la base (B4 o B5) respondió
    ref_width: int  # W_ref, en px del frame preparado completo
    variant: str


def describe_channels(variant: str, params: ChannelParams | None = None) -> list[dict[str, str]]:
    """Nombre, unidad y tratamiento de cada canal, en orden, para la model card."""
    names = _channel_names(variant)
    p = params or ChannelParams()
    gain, cap, ggain = p.error_gain, p.lr_cap, p.gradient_gain
    table = {
        "rgb_r": ("adimensional", "BGR→RGB; x/255 − 0,5"),
        "rgb_g": ("adimensional", "BGR→RGB; x/255 − 0,5"),
        "rgb_b": ("adimensional", "BGR→RGB; x/255 − 0,5"),
        "disp_estereo": ("d/W_ref", "disparidad estéreo (B2) / W_ref; 0 fuera de su máscara"),
        "valid_estereo": ("0/1", "máscara de validez del estéreo"),
        "error_lr": (
            f"d/W_ref × {gain:g}",
            f"|dL(x) − dR(x − dL)| / W_ref × {gain:g}, tope {cap:g}; "
            f"vale {cap:g} donde L/R no se puede calcular. No es b",
        ),
        "error_foto": (
            "adimensional (0–1)",
            "media RGB de |izq(x) − der(x − d_B4)| / 255, bilineal; "
            "0 donde B4 no es válida o x − d cae fuera de la imagen",
        ),
        "gradiente": ("adimensional (0–1)", f"|Sobel 3x3| del gris/255 × {ggain:g}, tope 1"),
        "disp_mono": (
            "d/W_ref",
            "monocular alineado con la recta global de la base (s·m + t; B4: RANSAC, B5: la de "
            "depth.b5) / W_ref donde la alineación se aceptó y d + doffs > 0; 0 en el resto",
        ),
        "dif_estereo_mono": (
            f"d/W_ref × {gain:g}",
            f"(d_estéreo − d_mono) / W_ref × {gain:g}, recortado a [−1, 1], "
            "solo donde existen ambas; 0 en el resto. Signo: estéreo menos monocular",
        ),
        "valid_mono": ("0/1", "máscara del canal disp_mono"),
        "corr_local": (
            f"d/W_ref × {p.corr_gain:g}",
            f"(d_B5 − d_B4) / W_ref × {p.corr_gain:g}, recortado a [−1, 1], solo en huecos del "
            "estéreo donde existen las dos; 0 en el resto. Frame completo, antes de recortar",
        ),
        "dist_ancla": (
            f"px / W_ref × {p.distance_gain:g}",
            f"distancia al ancla de B5 más cercana (tras la exclusión) / W_ref × "
            f"{p.distance_gain:g}, tope 1 (también sin anclas). Frame completo",
        ),
        "densidad_anclas": (
            "0–1",
            "fracción de anclas de B5 en su ventana gaussiana σ (Σw normalizada). Frame completo",
        ),
    }
    return [{"nombre": n, "unidad": table[n][0], "tratamiento": table[n][1]} for n in names]


def _channel_names(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise InvalidInput(f"variante {variant!r}: se esperaba una de {sorted(VARIANTS)}")
    return VARIANTS[variant]


def _check_map(name: str, disparity: np.ndarray, valid: np.ndarray, shape) -> None:
    """Disparidad float32 y máscara bool del tamaño del par; finita dentro de la máscara."""
    v.array(f"{name}.disparity_px", disparity, np.float32, 2, shape)
    v.array(f"{name}.valid", valid, np.bool_, 2, shape)
    if not np.isfinite(disparity[valid]).all():
        raise InvalidInput(f"{name}: hay NaN o infinitos dentro de la máscara válida")


def fusion_bases(
    stereo: StereoOutput,
    mono_relative: np.ndarray,
    doffs: float,
    align_params: AlignParams,
    b5_params: B5Params | None,
    need_b4: bool,
    need_b5: bool,
    b4_alignment: Alignment | None = None,
) -> tuple[HybridOutput | None, HybridOutput | None]:
    """B4 y/o B5 de un par, armadas igual al entrenar (caché) y al aplicar (app).

    - `b4_alignment`: la recta de B4 ya calculada (la guarda la caché); si no, se calcula.
    - B5 recalcula su recta con `b5_params.align`, salvo que sean exactamente los parámetros de
      B4: entonces reusa la de B4 (es la misma cuenta, determinista).
    """
    b4 = b5 = None
    if need_b4:
        b4 = combine("B4", stereo, mono_relative, doffs, align_params, alignment=b4_alignment)
    if need_b5:
        if b5_params is None:
            raise InvalidInput("B5 necesita depth.b5 (B5Params)")
        reuse = None
        if b5_params.align == align_params:
            reuse = b4.alignment if b4 is not None else b4_alignment
        b5 = combine("B5", stereo, mono_relative, doffs, b5_params.align, alignment=reuse,
                     local=b5_params.local)  # fmt: skip
    return b4, b5


def build_inputs(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    stereo: StereoOutput,
    base: HybridOutput,
    mono_relative: np.ndarray,
    doffs: float = 0.0,
    variant: str = "F1",
    params: ChannelParams | None = None,
    *,
    b4: HybridOutput | None = None,
    b5: HybridOutput | None = None,
    b5_params: B5Params | None = None,
) -> FusionInputs:
    """Arma los canales de Fusion para un par rectificado y ya preparado (tamaño final).

    - `stereo`: salida B2 de SgbmMatcher, con `lr_error_px`.
    - `base`: la base que corrige la red, B4 o B5, calculada con ESE estéreo y ESE monocular
      (`hybrid.combine` o `fusion_bases`).
    - `mono_relative`: el mapa relativo del monocular (sin alinear); se alinea con
      `base.alignment`, la misma recta global que usó la base.
    - `doffs`: en px; decide qué disparidades monoculares son físicas (d + doffs > 0).
    - Solo F3: necesita B4 y B5 del mismo par. Si la base es una, la otra va en `b4` o `b5`;
      B5 también se puede calcular acá si se pasa `b5_params` (B4 no: su recta es la de la
      caché o la del par, y la tiene quien llama).

    W_ref es el ancho de `left_bgr`: hay que llamar a esta función con el frame completo y
    recortar después (ver el docstring del módulo).
    """
    names = _channel_names(variant)
    p = params or ChannelParams()
    if not isinstance(p, ChannelParams):
        raise InvalidInput("params: se esperaba ChannelParams")
    left = v.array("left_bgr", left_bgr, np.uint8, 3, (None, None, 3))
    right = v.array("right_bgr", right_bgr, np.uint8, 3, left.shape)
    shape = left.shape[:2]
    doffs = v.finite("doffs", doffs)
    if not isinstance(stereo, StereoOutput):
        raise InvalidInput("stereo: se esperaba StereoOutput")
    if not isinstance(base, HybridOutput):
        raise InvalidInput("base: se esperaba HybridOutput (B4)")
    _check_map("stereo", stereo.disparity_px, stereo.valid, shape)
    _check_map("base", base.disparity_px, base.valid, shape)
    if stereo.lr_error_px is None:
        raise InvalidInput("stereo sin lr_error_px: Fusion necesita la salida B1/B2, no B0")
    v.array("stereo.lr_error_px", stereo.lr_error_px, np.float32, 2, shape)
    # B4 y B5 copian el estéreo en toda su máscara: si no, la base salió de otro estéreo (o
    # es B3).
    if not (base.source[stereo.valid] == SOURCE_STEREO).all():
        raise InvalidInput("base no es B4/B5 de este estéreo: hay píxeles estéreo con otra fuente")
    mono = check_relative(mono_relative, shape)

    width = int(shape[1])  # W_ref
    gain = np.float32(p.error_gain)

    # 1–3. RGB de la izquierda, centrado en 0.
    rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5

    # 4–5. Estéreo en d/W_ref y su máscara. np.where descarta los NaN de afuera.
    stereo_ok = stereo.valid
    d_stereo = np.where(stereo_ok, stereo.disparity_px, 0.0).astype(np.float32)
    disp_stereo = d_stereo / width

    # 6. Error L/R: px -> d/W_ref -> ganancia, con tope. Sin dato = tope ("sin evidencia de
    # consistencia"), no 0: un 0 diría "perfectamente consistente".
    lr = stereo.lr_error_px
    lr_ok = np.isfinite(lr)
    error_lr = np.where(lr_ok, np.minimum(np.where(lr_ok, lr, 0.0) / width * gain, p.lr_cap),
                        p.lr_cap).astype(np.float32)  # fmt: skip

    # 7. Error fotométrico de la base: ¿el punto se parece a su correspondencia según la base?
    d_base = np.where(base.valid, base.disparity_px, 0.0).astype(np.float32)
    error_photo = _photometric_error(left, right, d_base, base.valid)

    # 8. Gradiente de la izquierda: bordes, donde estéreo y monocular suelen fallar distinto.
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.minimum(np.sqrt(gx * gx + gy * gy) * np.float32(p.gradient_gain), 1.0)

    # 9. Monocular alineado con la misma recta que usó B4; solo si la alineación se aceptó.
    if base.alignment.accepted:
        aligned = base.alignment.apply(mono)
        with np.errstate(invalid="ignore"):
            mono_ok = np.isfinite(aligned) & ((aligned.astype(np.float64) + doffs) > 0)
        d_mono = np.where(mono_ok, aligned, 0.0).astype(np.float32)
    else:
        mono_ok = np.zeros(shape, bool)
        d_mono = np.zeros(shape, np.float32)
    disp_mono = d_mono / width

    # 10. Desacuerdo entre fuentes, solo donde existen las dos. Signo: estéreo − monocular.
    both = stereo_ok & mono_ok
    diff = np.clip((d_stereo - d_mono) / width * gain, -1.0, 1.0)
    diff_stereo_mono = np.where(both, diff, 0.0).astype(np.float32)

    planes: dict[str, np.ndarray] = {}
    if variant == "F3":
        planes.update(_local_planes(stereo, base, mono, doffs, width, p, b4, b5, b5_params))

    planes.update({
        "rgb_r": rgb[..., 0],
        "rgb_g": rgb[..., 1],
        "rgb_b": rgb[..., 2],
        "disp_estereo": disp_stereo,
        "valid_estereo": stereo_ok.astype(np.float32),
        "error_lr": error_lr,
        "error_foto": error_photo,
        "gradiente": gradient,
        "disp_mono": disp_mono,
        "dif_estereo_mono": diff_stereo_mono,
        "valid_mono": mono_ok.astype(np.float32),  # 11 (F1 y F3)
    })  # fmt: skip
    channels = np.stack([planes[n] for n in names]).astype(np.float32)
    base_valid = base.valid.copy()
    base_norm = (d_base / width).astype(np.float32)
    return FusionInputs(channels, names, base_norm, base_valid, width, variant)


def _local_planes(
    stereo: StereoOutput,
    base: HybridOutput,
    mono: np.ndarray,
    doffs: float,
    width: int,
    p: ChannelParams,
    b4: HybridOutput | None,
    b5: HybridOutput | None,
    b5_params: B5Params | None,
) -> dict[str, np.ndarray]:
    """Canales 12–14 (F3): la corrección local de B5 y su apoyo, sobre el frame completo."""
    b4 = b4 if b4 is not None else (base if base.variant == "B4" else None)
    b5 = b5 if b5 is not None else (base if base.variant == "B5" else None)
    if b5 is None and b5_params is not None:
        _, b5 = fusion_bases(stereo, mono, doffs, b5_params.align, b5_params, False, True)
    if b4 is None or b4.variant != "B4":
        raise InvalidInput("F3 necesita B4 del mismo par (argumento b4 o base B4)")
    if b5 is None or b5.variant != "B5" or b5.local is None:
        raise InvalidInput("F3 necesita B5 del mismo par (argumento b5, base B5 o b5_params)")
    for name, other in (("b4", b4), ("b5", b5)):
        _check_map(name, other.disparity_px, other.valid, stereo.valid.shape)
        if not (other.source[stereo.valid] == SOURCE_STEREO).all():
            raise InvalidInput(f"{name} no se armó con este estéreo")

    # 12. Cuánto mueve la corrección local a B4, solo en los huecos (donde el estéreo vale,
    # B4 y B5 son el mismo estéreo). np.where descarta los NaN de afuera.
    both = ~stereo.valid & b4.valid & b5.valid
    d4 = np.where(both, b4.disparity_px, 0.0).astype(np.float32)
    d5 = np.where(both, b5.disparity_px, 0.0).astype(np.float32)
    corr = np.clip((d5 - d4) / np.float32(width) * np.float32(p.corr_gain), -1.0, 1.0)
    # 13. Distancia al ancla usada más cercana: inf (sin anclas) satura en 1, como lo lejano.
    dist = b5.local.support.distance_px.astype(np.float32)
    dist = np.minimum(np.where(np.isfinite(dist), dist, np.inf) / np.float32(width)
                      * np.float32(p.distance_gain), 1.0)  # fmt: skip
    # 14. Densidad de anclas en la ventana gaussiana (ya en [0, 1]).
    density = np.clip(b5.local.support.density, 0.0, 1.0)
    return {
        "corr_local": np.where(both, corr, 0.0).astype(np.float32),
        "dist_ancla": dist.astype(np.float32),
        "densidad_anclas": density.astype(np.float32),
    }


def _photometric_error(
    left: np.ndarray, right: np.ndarray, d_base: np.ndarray, base_valid: np.ndarray
) -> np.ndarray:
    """Media sobre RGB de |izq(x, y) − der(x − d, y)| / 255, en 0–1.

    El punto (x, y) de la izquierda con disparidad d está en (x − d, y) de la derecha
    (d = x_izq − x_der). Como d tiene subpíxel, la derecha se muestrea con interpolación
    bilineal. Donde la base no es válida o x − d cae fuera de la imagen no hay con qué
    comparar: 0 (la máscara de la base ya dice que ahí no hay medida).
    """
    h, w = d_base.shape
    cols = np.arange(w, dtype=np.float32)[None, :]
    map_x = (cols - d_base).astype(np.float32)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    # En float32 para no redondear la interpolación a enteros de uint8.
    warped = cv2.remap(right.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
    error = np.abs(left.astype(np.float32) - warped).mean(axis=2) / 255.0
    inside = (map_x >= 0) & (map_x <= w - 1)
    return np.where(base_valid & inside, error, 0.0).astype(np.float32)


# --- salida de la red -> px (mismo postproceso para evaluar y para la app) -------------------


def disparity_from_residual(
    base_px: np.ndarray, base_valid: np.ndarray, delta_norm: np.ndarray, ref_width: float
) -> np.ndarray:
    """Disparidad de Fusion en px: d = d_B4 (px originales) + delta (d/W_ref) · W_ref.

    La base se suma en px, sin pasar por d/W_ref: la ida y vuelta px -> d/W -> px en float32
    pierde hasta ~3e-5 px con d ~ 272 y W ~ 1080, y M5 pide que una red recién creada (delta 0)
    reproduzca B4 a 1e-5 px. Solo el residual, que es chico, viaja normalizado.
    NaN donde la base no es válida.
    """
    shape = base_px.shape
    v.array("base_valid", base_valid, np.bool_, 2, shape)
    delta = np.asarray(delta_norm, np.float32)
    if delta.shape != shape:
        raise InvalidInput(f"delta {delta.shape} y base {shape} difieren")
    d = np.where(base_valid, base_px, 0.0).astype(np.float32) + delta * np.float32(ref_width)
    return np.where(base_valid, d, np.nan).astype(np.float32)


def fusion_validity(
    disparity_px: np.ndarray, b_px: np.ndarray, base_valid: np.ndarray, doffs: float
) -> np.ndarray:
    """Dónde vale la salida de Fusion: base válida, d finita con d + doffs > 0 y b finita > 0.

    Un residual que lleva d a un valor no físico invalida el píxel; no se recorta (eso
    inventaría una medida). La usan el estimador de la app y la evaluación.
    """
    with np.errstate(invalid="ignore"):
        physical = np.isfinite(disparity_px) & ((disparity_px.astype(np.float64) + doffs) > 0)
        b_ok = np.isfinite(b_px) & (b_px > 0)
    return base_valid & physical & b_ok
