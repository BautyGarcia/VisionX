"""Lector del formato Middlebury, que usan MiddEval3 y ETH3D two-view.

Cada par vive en una carpeta con:
    im0.png, im1.png   ojo izquierdo y derecho, ya rectificados
    disp0GT.pfm        disparidad GT del ojo izquierdo, en píxeles; infinito = sin GT
    mask0nocc.png      0 = sin GT, 128 = GT pero ocluido en el derecho, 255 = visible en ambos
    calib.txt          cam0/cam1 = [f 0 cx; 0 f cy; 0 0 1], doffs (px), baseline (mm), ...

Lo comprobamos en los archivos descargados (BITACORA 2026-11-13): los 0 de la máscara
coinciden exactamente con los infinitos del PFM. ETH3D guarda imágenes en grises como
tres canales iguales.

Z = baseline * f / (d + doffs). Middlebury da el baseline en milímetros: lo pasamos a metros
para usar la misma StereoGeometry que la ELP.

**Escenas sueltas (2014 y 2021, validación "dev" de otro dominio).** Las páginas de escenas
publican otro juego de archivos por par: `disp0.pfm` y `disp1.pfm` (GT de cada ojo, infinito =
sin GT) y **no** publican máscara de oclusión. `load_scene_folder` deriva `nonoccluded` con un
chequeo de ida y vuelta entre las dos GT (`stereo.lr_consistency`, tolerancia
`GT_OCCLUSION_TOL_PX`): es nuestra regla, no la máscara oficial de MiddEval3.

**Media resolución (H).** MiddEval3-H es la resolución completa reducida a la mitad; la página no
documenta el procedimiento exacto. `write_half_resolution` lo hace con las reglas de
`shared/geometry.py`, una sola vez al preparar: imágenes por promedio de bloques 2x2
(`cv2.INTER_AREA`), GT por vecino más cercano y x 0,5 (`resize_disparity`: nunca promedia a
través de un borde ni de un inválido) y calibración con `StereoGeometry.resized` (fx, cx, doffs
y ndisp a la mitad; el baseline no cambia). Con eso Z = fx * B / (d + doffs) da lo mismo antes y
después: los tres factores se reducen a la mitad.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from visionx.depth.datasets.pfm import read_pfm
from visionx.depth.datasets.sample import StereoSample
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry, resize_disparity

MASK_NO_GT, MASK_OCCLUDED, MASK_VISIBLE = 0, 128, 255
FILES = ("im0.png", "im1.png", "disp0GT.pfm", "mask0nocc.png", "calib.txt")
# Páginas de escenas 2014/2021: GT de los dos ojos y sin máscara publicada.
SCENE_FILES = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")
# Un píxel de la izquierda es "visible en ambos ojos" si la GT del derecho, en x - d, da la
# misma disparidad con esta tolerancia (px de la resolución guardada). 1 px es el umbral
# habitual del chequeo L/R; las GT de Middlebury tienen error subpíxel.
GT_OCCLUSION_TOL_PX = 1.0


def read_calib(path: str | Path) -> dict[str, object]:
    """calib.txt a diccionario: matrices como arrays 3x3, el resto como float."""
    values: dict[str, object] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, raw = (part.strip() for part in line.split("=", 1))
        if raw.startswith("["):
            rows = raw.strip("[]").split(";")
            values[key] = np.array([[float(x) for x in r.split()] for r in rows], np.float64)
        else:
            values[key] = float(raw)
    for key in ("cam0", "cam1", "doffs", "baseline", "width", "height"):
        if key not in values:
            raise InvalidInput(f"{path}: falta {key}")
    return values


def geometry_from_calib(calib: dict[str, object]) -> StereoGeometry:
    cam0, cam1 = calib["cam0"], calib["cam1"]
    doffs = float(calib["doffs"])
    # Control de coherencia: doffs es la diferencia de centros ópticos de los dos ojos.
    if not math.isclose(doffs, cam1[0, 2] - cam0[0, 2], abs_tol=0.01):
        raise InvalidInput(f"doffs {doffs} no coincide con cx1 - cx0 = {cam1[0, 2] - cam0[0, 2]}")
    return StereoGeometry(
        width=int(calib["width"]),
        height=int(calib["height"]),
        fx=float(cam0[0, 0]),
        fy=float(cam0[1, 1]),
        cx=float(cam0[0, 2]),
        cy=float(cam0[1, 2]),
        baseline_m=float(calib["baseline"]) / 1000.0,  # mm -> m
        doffs=doffs,
    )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)  # siempre 3 canales BGR uint8
    if image is None:
        raise InvalidInput(f"no se pudo leer {path}")
    return image


def load_folder(
    folder: str | Path, dataset: str, sample_id: str, scene_id: str, root: str | Path
) -> StereoSample:
    """Lee una carpeta en formato Middlebury. `root` sirve para guardar rutas relativas."""
    folder, root = Path(folder), Path(root)
    left = _read_image(folder / "im0.png")
    right = _read_image(folder / "im1.png")
    raw = read_pfm(folder / "disp0GT.pfm")
    mask = cv2.imread(str(folder / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise InvalidInput(f"no se pudo leer {folder / 'mask0nocc.png'}")
    unknown = set(np.unique(mask).tolist()) - {MASK_NO_GT, MASK_OCCLUDED, MASK_VISIBLE}
    if unknown:
        raise InvalidInput(f"{folder}: valores de máscara inesperados {sorted(unknown)}")
    valid = np.isfinite(raw)
    if not np.array_equal(valid, mask != MASK_NO_GT):
        raise InvalidInput(f"{folder}: la máscara y los infinitos del PFM no coinciden")
    disparity = np.where(valid, raw, np.nan).astype(np.float32)
    gray = bool(
        np.array_equal(left[..., 0], left[..., 1]) and np.array_equal(left[..., 1], left[..., 2])
    )
    return StereoSample(
        dataset=dataset,
        sample_id=sample_id,
        scene_id=scene_id,
        left=left,
        right=right,
        disparity=disparity,
        valid=valid,
        nonoccluded=mask == MASK_VISIBLE,
        geometry=geometry_from_calib(read_calib(folder / "calib.txt")),
        grayscale=gray,
        files=tuple((folder / n).relative_to(root).as_posix() for n in FILES),
    )


# --- Escenas 2014/2021 (sin máscara publicada) ---------------------------------------------


def _read_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """PFM de GT -> (disparidad float32 con NaN, válida). Infinito = sin GT (verificado en los
    archivos descargados el 17/11: no hay NaN ni ceros; los huecos son +inf)."""
    raw = read_pfm(path)
    valid = np.isfinite(raw)
    return np.where(valid, raw, np.nan).astype(np.float32), valid


def gt_nonoccluded(
    disp0: np.ndarray,
    valid0: np.ndarray,
    disp1: np.ndarray,
    valid1: np.ndarray,
    tol_px: float = GT_OCCLUSION_TOL_PX,
) -> np.ndarray:
    """Píxeles con GT en la izquierda cuya correspondencia también tiene la misma GT en el ojo
    derecho: |d0(x) - d1(x - d0)| <= tol. Lo que falla es oclusión (o GT del derecho ausente)."""
    from visionx.depth.stereo import lr_consistency  # diferido: datasets no depende de depth

    return lr_consistency(disp0, valid0, disp1, valid1, tol_px) & valid0


def load_scene_folder(
    folder: str | Path, dataset: str, sample_id: str, scene_id: str, root: str | Path
) -> StereoSample:
    """Lee un par de las páginas de escenas de Middlebury (2014 -perfect o 2021)."""
    folder, root = Path(folder), Path(root)
    left = _read_image(folder / "im0.png")
    right = _read_image(folder / "im1.png")
    disparity, valid = _read_gt(folder / "disp0.pfm")
    disp1, valid1 = _read_gt(folder / "disp1.pfm")
    if disparity.shape != left.shape[:2] or disp1.shape != left.shape[:2]:
        raise InvalidInput(f"{folder}: GT e imagen de distinto tamaño")
    calib = read_calib(folder / "calib.txt")
    h, w = left.shape[:2]
    if (int(calib["width"]), int(calib["height"])) == (h, w) != (w, h):
        # Middlebury 2021, ladder1/ladder2: fotos verticales (1080 de ancho x 1920 de alto)
        # con width/height de calib.txt intercambiados. Las focales y centros sí son los de la
        # imagen vertical (cy ≈ 957 ≈ 1920 / 2) y la GT es 1080 x 1920: solo se corrige el
        # tamaño declarado. La auditoría census confirma que la GT corre sobre las filas.
        calib = {**calib, "width": float(w), "height": float(h)}
    return StereoSample(
        dataset=dataset,
        sample_id=sample_id,
        scene_id=scene_id,
        left=left,
        right=right,
        disparity=disparity,
        valid=valid,
        nonoccluded=gt_nonoccluded(disparity, valid, disp1, valid1),
        geometry=geometry_from_calib(calib),
        grayscale=False,
        files=tuple((folder / n).relative_to(root).as_posix() for n in SCENE_FILES),
    )


def _matrix(fx: float, fy: float, cx: float, cy: float) -> str:
    return f"[{fx:.6f} 0 {cx:.6f}; 0 {fy:.6f} {cy:.6f}; 0 0 1]"


def half_calib(calib: dict[str, object], width: int, height: int) -> str:
    """calib.txt a la mitad: la cámara de `StereoGeometry.resized` y ndisp/vmin/vmax a la mitad.

    `width`/`height` son los de la imagen reducida (si la completa tenía un lado impar, se
    recortó una fila/columna del final antes: recortar a la derecha o abajo no mueve cx/cy).
    cam1 se reduce con la misma regla, así doffs = cx1 - cx0 sigue valiendo.
    """
    full = geometry_from_calib(calib)
    cropped = full.cropped(0, 0, 2 * width, 2 * height)
    half = cropped.resized(width, height)
    cam1 = calib["cam1"]
    cx1 = (float(cam1[0, 2]) + 0.5) * 0.5 - 0.5  # misma regla de centros que resized()
    lines = [
        f"cam0={_matrix(half.fx, half.fy, half.cx, half.cy)}",
        f"cam1={_matrix(half.fx, half.fy, cx1, half.cy)}",
        f"doffs={half.doffs:.6f}",
        f"baseline={float(calib['baseline']):.6f}",  # mm; no depende de la resolución
        f"width={width}",
        f"height={height}",
    ]
    # Rangos de disparidad: se reducen como la disparidad. ndisp hacia arriba (es una cota).
    if "ndisp" in calib:
        lines.append(f"ndisp={math.ceil(float(calib['ndisp']) / 2)}")
    for key in ("vmin", "vmax"):
        if key in calib:
            lines.append(f"{key}={float(calib[key]) / 2:g}")
    for key, value in calib.items():  # isint, dyavg, dymax...: se copian como están
        if key not in {"cam0", "cam1", "doffs", "baseline", "width", "height", "ndisp", "vmin",
                       "vmax"}:  # fmt: skip
            lines.append(f"{key}={float(value):g}")
    return "\n".join(lines) + "\n"


def write_half_resolution(src: str | Path, dst: str | Path) -> Path:
    """Escribe en `dst` el par de `src` (formato de escenas) a media resolución.

    Imágenes: promedio de bloques 2x2 (INTER_AREA con factor exacto 1/2). GT de los dos ojos:
    `resize_disparity` (vecino más cercano y x 0,5; inválido sigue inválido). Calibración:
    `half_calib`. Un lado impar pierde su última fila/columna para que el factor sea exacto.
    """
    from visionx.depth.datasets.pfm import write_pfm

    src, dst = Path(src), Path(dst)
    calib = read_calib(src / "calib.txt")
    left = _read_image(src / "im0.png")
    full_h, full_w = left.shape[:2]
    if (int(calib["width"]), int(calib["height"])) != (full_w, full_h):
        raise InvalidInput(f"{src}: calib.txt no coincide con el tamaño de im0.png")
    width, height = full_w // 2, full_h // 2
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("im0.png", "im1.png"):
        image = left if name == "im0.png" else _read_image(src / name)
        image = image[: 2 * height, : 2 * width]
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(dst / name), small):
            raise InvalidInput(f"no se pudo escribir {dst / name}")
    for name in ("disp0.pfm", "disp1.pfm"):
        d, ok = _read_gt(src / name)
        d, ok = d[: 2 * height, : 2 * width], ok[: 2 * height, : 2 * width]
        small, small_ok = resize_disparity(d, ok, width, height)
        write_pfm(dst / name, np.where(small_ok, small, np.inf).astype(np.float32))
    (dst / "calib.txt").write_text(half_calib(calib, width, height), encoding="utf-8")
    return dst
