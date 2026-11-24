"""Banco ELP: evaluar la profundidad contra puntos medidos con cinta (T04/T15).

El banco son escenas grabadas con la cámara quieta y stickers numerados cuya profundidad se
midió con cinta (protocolo en TECNICO, «Datos y GT»). No hay un mapa de referencia: hay unos
pocos puntos por escena, medidos con cuidado. Este módulo compara, en esos puntos, lo que dice
un método de profundidad con lo que dijo la cinta.

Dos archivos por escena, dentro de `recordings/banco-sN-eMM/`:

    escena.yaml   la planilla que se completa al grabar: id de cada sticker y su lectura de cinta
    puntos.yaml   dónde está cada sticker en la imagen izquierda rectificada (lo escribe `marcar`)

    # 1. marcar los stickers a mano (necesita pantalla; clic sobre cada uno en orden)
    uv run python -m visionx.depth.bench marcar recordings/banco-s1-e01

    # 2. evaluar métodos sobre una o más escenas
    uv run python -m visionx.depth.bench evaluar recordings/banco-s1-e0*

    # 3. lo mismo con la cámara que dio de alta «Cámara nueva», en vez del tablero
    uv run python -m visionx.depth.bench evaluar recordings/banco-s1-e0* --camara camara-desconocida

**Con un perfil de cámara** (`--camara`, PLAN-CIERRE §6, prueba A). Los stickers se marcan una
sola vez, sobre la imagen rectificada con la calibración de tablero (la referencia). Para
evaluar con el perfil, cada sticker se lleva a la foto cruda con la calibración de tablero (que
es la verdad) y de ahí a la imagen rectificada con el perfil, que es donde lo ve la app. Como en
la app, el ojo derecho se alinea con el clásico sobre los pares de la escena (`--sin-alinear`
lo apaga). El cero de la cinta sale siempre de la calibración de tablero: es la distancia del
vidrio al centro óptico, no depende de cómo se estime la cámara. La corrección «calibrá, está a
N metros» que pueda tener el perfil no se aplica: se mide la cámara tal como la dio de alta.

**Sesiones de test.** `s3`, `s4` y `s5` están reservadas (decisión del 17/11): `evaluar` las
rechaza salvo que se pase `--test-congelado`, igual que el test público. `marcar` sí se puede
usar: mirar la imagen para ubicar los stickers no corre ningún método.

Cómo se mide un punto: se toma una ventana chica alrededor del sticker y la **mediana** de la
profundidad válida adentro. La mediana no se deja arrastrar por un borde ni por un píxel malo.
La cinta se leyó desde el vidrio de la lente; el centro óptico queda unos milímetros atrás, así
que a la lectura se le suma ese cero (`cero_cinta_mm`, guardado en la calibración).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import DepthResult, UncertaintyKind
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError

SCENE_FILE = "escena.yaml"
POINTS_FILE = "puntos.yaml"
SCENE_FORMAT = "visionx-banco-elp"
POINTS_FORMAT = "visionx-banco-puntos"
TEST_SESSIONS = frozenset({"s3", "s4", "s5"})
RANGES_M = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5))  # la prueba A llega a 4 m
TOLERANCE_PCT = 5.0  # métrica principal en la ELP: % de puntos con error <= 5 % (13/11)


@dataclass(frozen=True)
class BenchPoint:
    id: int
    u: float  # columna en la imagen izquierda rectificada, a resolución de captura
    v: float  # fila
    tape_m: float  # lectura de cinta, sin sumar el cero
    surface: str
    app_class: str | None


@dataclass(frozen=True)
class BenchScene:
    name: str
    session: str
    folder: Path
    size: tuple[int, int]  # (ancho, alto) de la imagen donde se marcaron los puntos
    points: tuple[BenchPoint, ...]

    @property
    def is_test(self) -> bool:
        return self.session in TEST_SESSIONS


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SourceError(f"falta {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise InvalidInput(f"{path}: se esperaba un mapa YAML")
    return data


def load_scene(folder: str | Path) -> BenchScene:
    """Lee planilla y puntos de una escena y controla que hablen de los mismos stickers."""
    folder = Path(folder)
    sheet = _read_yaml(folder / SCENE_FILE)
    marks = _read_yaml(folder / POINTS_FILE)
    if sheet.get("formato") != SCENE_FORMAT:
        raise InvalidInput(f"{folder / SCENE_FILE}: formato inesperado {sheet.get('formato')!r}")
    if marks.get("formato") != POINTS_FORMAT:
        raise InvalidInput(f"{folder / POINTS_FILE}: formato inesperado {marks.get('formato')!r}")
    measured = {int(p["id"]): p for p in sheet.get("puntos") or []}
    located = {int(p["id"]): p for p in marks.get("puntos") or []}
    if not measured:
        raise InvalidInput(f"{folder / SCENE_FILE}: no tiene puntos")
    if set(measured) != set(located):
        only = sorted(set(measured) ^ set(located))
        raise InvalidInput(f"{folder}: la planilla y los puntos marcados difieren en {only}")
    points = []
    for number in sorted(measured):
        tape = float(measured[number]["cinta_m"])
        if not tape > 0:
            raise InvalidInput(f"{folder}: el punto {number} no tiene lectura de cinta")
        points.append(
            BenchPoint(
                id=number,
                u=float(located[number]["u"]),
                v=float(located[number]["v"]),
                tape_m=tape,
                surface=str(measured[number].get("superficie", "")),
                app_class=measured[number].get("clase_app"),
            )
        )
    width, height = (int(x) for x in marks["tamano"])
    return BenchScene(
        name=str(sheet.get("escena", folder.name)),
        session=str(sheet.get("sesion", "")),
        folder=folder,
        size=(width, height),
        points=tuple(points),
    )


def trasladar_puntos(
    points: tuple[BenchPoint, ...], desde: Any, hacia: Any
) -> tuple[BenchPoint, ...]:
    """Los stickers marcados en la izquierda rectificada con `desde`, en la rectificada con `hacia`.

    Ida a la foto cruda con `desde` (la calibración de tablero, la verdad): el píxel rectificado
    es un rayo en la cámara rectificada; `R1` lo devuelve a la cámara izquierda y la lente lo
    proyecta. Vuelta con `hacia` (el perfil): `undistort_points` hace lo mismo que la
    rectificación de la imagen, así que el sticker cae donde lo ve el método.
    """
    from visionx.calibration.models import project, undistort_points

    if tuple(desde.image_size) != tuple(hacia.image_size):
        raise InvalidInput(
            f"la calibración de referencia es de {desde.image_size} y la de la cámara de "
            f"{hacia.image_size}: tienen que ser de la misma foto"
        )
    uv = np.array([[p.u, p.v] for p in points], np.float64)
    P = desde.rect.P1
    rays = np.column_stack(
        [(uv[:, 0] - P[0, 2]) / P[0, 0], (uv[:, 1] - P[1, 2]) / P[1, 1], np.ones(len(uv))]
    )
    rays = rays @ desde.rect.R1  # filas: R1ᵀ · rayo, de la cámara rectificada a la izquierda
    raw = project(desde.model, rays, np.zeros(3), np.zeros(3), desde.left)
    moved = undistort_points(hacia.model, raw, hacia.left, hacia.rect.R1, hacia.rect.P1)
    pairs = zip(points, moved, strict=True)
    return tuple(dataclasses.replace(p, u=float(u), v=float(v)) for p, (u, v) in pairs)


def save_points(
    folder: str | Path, size: tuple[int, int], points: dict[int, tuple[float, float]], frame_id: int
) -> Path:
    """Escribe `puntos.yaml`: posición de cada sticker en la izquierda rectificada."""
    path = Path(folder) / POINTS_FILE
    data = {
        "formato": POINTS_FORMAT,
        "version": 1,
        "imagen": "izquierda rectificada, resolución de captura",
        "frame_id": int(frame_id),
        "tamano": [int(size[0]), int(size[1])],
        "puntos": [
            {"id": int(i), "u": round(float(u), 1), "v": round(float(v), 1)}
            for i, (u, v) in sorted(points.items())
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def _window(array: np.ndarray, u: float, v: float, half: int) -> np.ndarray:
    row, col = int(round(v)), int(round(u))
    return array[max(row - half, 0) : row + half + 1, max(col - half, 0) : col + half + 1]


def measure_point(
    result: DepthResult,
    point: BenchPoint,
    capture_size: tuple[int, int],
    zero_m: float,
    window_px: int = 9,
) -> dict[str, Any]:
    """Profundidad del método en un sticker y su error contra la cinta.

    `window_px` es el lado de la ventana a resolución de captura; se achica si el método
    calcula a menor resolución. `cubierto` dice qué parte de la ventana tenía dato.
    """
    scale_x = result.size[0] / capture_size[0]
    scale_y = result.size[1] / capture_size[1]
    half = max(int(round(window_px * scale_x / 2)), 1)
    u, v = point.u * scale_x, point.v * scale_y
    valid = _window(result.valid_mask, u, v, half)
    depth = _window(result.depth_m, u, v, half)
    reference = point.tape_m + zero_m
    row: dict[str, Any] = {
        "punto": point.id,
        "superficie": point.surface,
        "clase_app": point.app_class,
        "referencia_m": round(reference, 4),
        "cubierto": float(valid.mean()) if valid.size else 0.0,
        "medido_m": math.nan,
        "error_pct": math.nan,
    }
    if valid.size == 0 or not valid.any():
        return row
    measured = float(np.median(depth[valid]))
    row["medido_m"] = round(measured, 4)
    row["error_pct"] = round((measured - reference) / reference * 100.0, 3)
    if result.uncertainty is not None and result.uncertainty_kind is UncertaintyKind.LAPLACE_B:
        b = float(np.median(_window(result.uncertainty, u, v, half)[valid]))
        disparity = float(np.median(_window(result.disparity_px, u, v, half)[valid]))
        row["b_px"] = round(b, 4)
        # Error en px de disparidad, para poder comparar b con lo que de verdad se erró.
        reference_disparity = measured * (disparity + result.doffs) / reference - result.doffs
        row["error_px"] = round(disparity - reference_disparity, 4)
        # Intervalo del 90 % de una Laplace, en disparidad: ±b·ln(10). Se pasa a metros
        # convirtiendo los dos extremos (el intervalo en metros no es simétrico).
        spread = b * math.log(10.0)
        factor = measured * (disparity + result.doffs)  # = f·B a la resolución de cómputo
        low = factor / (disparity + result.doffs + spread)
        high_d = disparity + result.doffs - spread
        high = factor / high_d if high_d > 0 else math.inf
        row["intervalo90_m"] = [round(low, 4), round(high, 4) if math.isfinite(high) else None]
        row["dentro_intervalo90"] = bool(low <= reference <= high)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen de puntos: cuántos midió el método y cuántos quedaron dentro de la tolerancia."""
    total = len(rows)
    errors = np.array([r["error_pct"] for r in rows], dtype=np.float64)
    measured = np.isfinite(errors)
    out: dict[str, Any] = {
        "puntos": total,
        "medidos": int(measured.sum()),
        "cobertura": round(float(measured.mean()), 3) if total else math.nan,
    }
    if measured.any():
        absolute = np.abs(errors[measured])
        out |= {
            # Sobre todos los puntos: no medir un punto cuenta como no acertarlo.
            "dentro_5pct": round(float(np.sum(absolute <= TOLERANCE_PCT) / total * 100.0), 1),
            "error_abs_mediana_pct": round(float(np.median(absolute)), 2),
            "error_abs_p90_pct": round(float(np.percentile(absolute, 90)), 2),
            "sesgo_mediana_pct": round(float(np.median(errors[measured])), 2),
        }
    inside = [r["dentro_intervalo90"] for r in rows if "dentro_intervalo90" in r]
    if inside:
        out["cobertura_intervalo90"] = round(float(np.mean(inside)), 3)
    ratios = [abs(r["error_px"]) / r["b_px"] for r in rows if r.get("b_px")]
    if ratios:
        # Si b estuviera bien calibrada, |error| / b promediaría 1 (es la escala de una
        # Laplace). Este promedio es el factor por el que habría que multiplicar b para que
        # lo esté en estos puntos: menor que 1, b es prudente de más; mayor, confía de más.
        out["escala_b_sugerida"] = round(float(np.mean(ratios)), 3)
    return out


def summarize_by_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"todos": summarize(rows)}
    for low, high in RANGES_M:
        subset = [r for r in rows if low <= r["referencia_m"] < high]
        if subset:
            out[f"{low:g}-{high:g} m"] = summarize(subset)
    return out


# --- comandos -----------------------------------------------------------------------------


def _rectified_frames(folder: Path, cal: Any, count: int) -> list[Any]:
    from visionx.camera.rectified import RectifiedSource
    from visionx.camera.replay import ReplaySource

    source = RectifiedSource(ReplaySource(folder), cal)
    frames = []
    while (frame := source.read()) is not None and len(frames) < count:
        frames.append(frame)
    source.close()
    if not frames:
        raise SourceError(f"{folder} no tiene pares")
    return frames


def _mark(args: argparse.Namespace) -> int:
    """Marca a mano los stickers de una escena. Abre una ventana: necesita pantalla."""
    import cv2

    from visionx.calibration import StereoCalibration

    cfg = load_config(args.config)
    cal = StereoCalibration.load(ROOT / cfg["camera"]["calibration"])
    folder = Path(args.carpeta)
    sheet = _read_yaml(folder / SCENE_FILE)
    ids = [int(p["id"]) for p in sheet.get("puntos") or []]
    if not ids:
        raise InvalidInput(f"{folder / SCENE_FILE}: cargá primero los puntos de la planilla")
    frame = _rectified_frames(folder, cal, 1)[0]
    height, width = frame.left.shape[:2]
    scale = min(args.ancho_ventana / width, 1.0)
    shown = cv2.resize(frame.left, (int(width * scale), int(height * scale)))
    marks: dict[int, tuple[float, float]] = {}

    def click(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(marks) < len(ids):
            marks[ids[len(marks)]] = (x / scale, y / scale)

    title = "banco ELP: marcar stickers"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, click)
    print("Clic sobre cada sticker en el orden de la planilla. u: deshacer, g: guardar, q: salir.")
    while True:
        canvas = shown.copy()
        for number, (u, v) in marks.items():
            center = (int(u * scale), int(v * scale))
            cv2.circle(canvas, center, 6, (0, 255, 0), 1)
            cv2.putText(
                canvas,
                str(number),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        pending = ids[len(marks)] if len(marks) < len(ids) else None
        text = f"falta el sticker {pending}" if pending is not None else "listo: g para guardar"
        cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and marks:
            marks.pop(ids[len(marks) - 1])
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("sin guardar")
            return 1
        elif key == ord("g") and pending is None:
            break
    cv2.destroyAllWindows()
    print(f"guardado {save_points(folder, (width, height), marks, frame.frame_id)}")
    return 0


def _estimators(names: list[str], cfg: dict[str, Any], geometry: Any, args: Any) -> dict[str, Any]:
    """Arma los métodos pedidos. Los de CREStereo comparten una sola red cargada."""
    from visionx.depth.stereo import (
        SgbmEstimator,
        SgbmParams,
        compute_geometry,
        num_disparities_for,
    )

    depth_cfg = cfg["depth"]
    out: dict[str, Any] = {}
    shared: tuple[Any, str] | None = None
    for name in names:
        if name.startswith("crestereo-"):
            from visionx.depth.crestereo_adapt.estimator import load_estimator
            from visionx.depth.crestereo_adapt.port import PortConfig, load_model

            if shared is None:
                shared = load_model(PortConfig.from_config(depth_cfg["crestereo"]))
            head = None
            if args.incertidumbre:
                from visionx.depth.crestereo_adapt.uncertainty import UncertaintyHead

                head = UncertaintyHead.load(args.incertidumbre, shared[1])
            out[name] = load_estimator(
                cfg,
                geometry,
                name.removeprefix("crestereo-"),
                model=shared[0],
                device=shared[1],
                uncertainty=head,
            )
        elif name.startswith("sgbm-"):
            size = tuple(int(x) for x in depth_cfg["compute"])
            ndisp = num_disparities_for(
                compute_geometry(geometry, size), float(depth_cfg["range_m"][0])
            )
            params = SgbmParams.from_config(depth_cfg["sgbm"], num_disparities=ndisp)
            out[name] = SgbmEstimator(name.removeprefix("sgbm-").upper(), params, geometry, size)
        else:
            raise InvalidInput(f"método {name!r}: se esperaba crestereo-<modo> o sgbm-<variante>")
    return out


def _camera(name: str) -> tuple[Any, Any]:
    """(perfil, calibración) de una cámara de `configs/camaras` o dada de alta en la Mac."""
    from visionx.camera.perfiles import calibracion_de, cargar_perfiles

    profiles = {p.id: p for p in cargar_perfiles()}
    if name not in profiles:
        raise InvalidInput(f"--camara {name!r}: no hay perfil; hay {sorted(profiles)}")
    return profiles[name], calibracion_de(profiles[name])


def _align_right(frames: list[Any], geometry: Any) -> tuple[list[Any], tuple[str, ...]]:
    """El ojo derecho corregido con el clásico sobre los pares de la escena, como en la app."""
    from visionx.depth.crestereo_adapt.align_apply import Alineador

    rights, info = Alineador("clasico", geometry, np.zeros(5)).corregir(
        [(f.left, f.right) for f in frames]
    )
    out = [dataclasses.replace(f, right=r) for f, r in zip(frames, rights, strict=True)]
    return out, tuple(info.aplica)


def _evaluate(args: argparse.Namespace) -> int:
    from visionx.calibration import StereoCalibration

    cfg = load_config(args.config)
    reference = StereoCalibration.load(ROOT / cfg["camera"]["calibration"])
    profile, cal = (None, reference) if not args.camara else _camera(args.camara)
    # Una cámara dada de alta (sin calibración propia) se alinea, como en la app.
    align = profile is not None and not profile.calibracion and not args.sin_alinear
    geometry = cal.geometry()
    zero_mm = args.cero_mm
    if zero_mm is None:
        zero_mm = float(reference.disparity_offset_source.get("cero_cinta_mm", 0.0))
    scenes = [load_scene(folder) for folder in args.carpetas]
    reserved = [s.name for s in scenes if s.is_test]
    if reserved and not args.test_congelado:
        raise InvalidInput(
            f"escenas de test reservadas {reserved}: solo con todo congelado y --test-congelado"
        )
    estimators = _estimators(args.metodos, cfg, geometry, args)
    rows: list[dict[str, Any]] = []
    aligned: dict[str, list[str]] = {}
    for scene in scenes:
        marked_size = reference.geometry().size
        if scene.size != marked_size:
            raise InvalidInput(
                f"{scene.name}: puntos marcados a {scene.size}, calibración a {marked_size}"
            )
        points = scene.points
        if cal is not reference:
            points = trasladar_puntos(points, reference, cal)
        frames = _rectified_frames(scene.folder, cal, args.pares)
        if align:
            frames, applied = _align_right(frames, geometry)
            aligned[scene.name] = list(applied)
        for name, est in estimators.items():
            for frame in frames:
                result = est.predict(frame)
                for point in points:
                    row = measure_point(
                        result, point, geometry.size, zero_mm / 1000.0, args.ventana
                    )
                    rows.append(
                        {
                            "escena": scene.name,
                            "sesion": scene.session,
                            "metodo": name,
                            "frame_id": frame.frame_id,
                            **row,
                        }
                    )
            est.close()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-banco-elp"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with (out_dir / "puntos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    camera = None
    if profile is not None:
        camera = {
            "perfil": profile.id,
            "nivel": profile.nivel,
            "calibration_id": cal.calibration_id,
            "focal_px": round(float(geometry.fx), 2),
            "separacion_m": profile.separacion_m,
            "alineacion": aligned if align else "no",
        }
    report: dict[str, Any] = {
        "calibracion": cfg["camera"]["calibration"],
        "camara": camera,
        "cero_cinta_mm": zero_mm,
        "escenas": [s.name for s in scenes],
        "pares_por_escena": args.pares,
        "ventana_px": args.ventana,
        "comando": " ".join(sys.argv),
        "metodos": {},
    }
    for name, est in estimators.items():
        mine = [r for r in rows if r["metodo"] == name]
        summary = summarize_by_range(mine)
        report["metodos"][name] = {"model_id": est.model_id, "resumen": summary}
        total = summary["todos"]
        print(
            f"{name}: {total['medidos']}/{total['puntos']} puntos medidos; dentro de 5 %: "
            f"{total.get('dentro_5pct', math.nan)} %; error mediano "
            f"{total.get('error_abs_mediana_pct', math.nan)} %, P90 "
            f"{total.get('error_abs_p90_pct', math.nan)} %"
        )
    with (out_dir / "resumen.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(f"resultados en {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.depth.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    ma = sub.add_parser("marcar", help="ubicar a mano los stickers de una escena (con pantalla)")
    ma.add_argument("carpeta")
    ma.add_argument("--ancho-ventana", type=int, default=1440)
    ma.add_argument("--config", default=None, help="override YAML explícito")
    ev = sub.add_parser("evaluar", help="medir métodos de profundidad contra la cinta")
    ev.add_argument("carpetas", nargs="+")
    ev.add_argument("--metodos", nargs="+", default=["crestereo-preciso", "sgbm-b0"])
    ev.add_argument("--incertidumbre", default=None, help="checkpoint de la U-Net de incertidumbre")
    ev.add_argument("--pares", type=int, default=5, help="cuadros de cada escena que se evalúan")
    ev.add_argument("--ventana", type=int, default=9, help="lado de la ventana, px de captura")
    ev.add_argument("--cero-mm", type=float, default=None)
    ev.add_argument("--camara", default=None,
                    help="perfil de cámara (p. ej. el de «Cámara nueva») en vez de la calibración "
                         "del config; los puntos se marcan igual con la del config")  # fmt: skip
    ev.add_argument("--sin-alinear", action="store_true",
                    help="con --camara, no alinear el ojo derecho con el clásico")  # fmt: skip
    ev.add_argument(
        "--test-congelado",
        action="store_true",
        help="confirma que todo está congelado (habilita s3-s5)",
    )
    ev.add_argument("--config", default=None, help="override YAML explícito")
    ev.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    args = parser.parse_args(argv)
    try:
        return _mark(args) if args.comando == "marcar" else _evaluate(args)
    except (InvalidInput, EstimatorError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
