"""Mediciones de la compuerta de T12 sobre CREStereo congelado.

    # una vez por máquina: código y pesos, fuera de Git
    uv run python scripts/profundidad/descargar_crestereo.py

    # tabla de latencia M9 en la Mac (resolución x iteraciones x pasadas x arranque)
    uv run --extra ml python -m visionx.depth.crestereo_adapt latencia recordings/CLIP

    # distancias contra la cinta, con los pares de cinta.py
    uv run --extra ml python -m visionx.depth.crestereo_adapt cinta RUTA/cinta-s7

Las salidas van a outputs/profundidad/<fecha>-.../ (fuera de Git), con commit, comando y equipo.
Medir no cambia la config: los modos `continuo` y `preciso` se fijan a mano después de mirar.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from visionx.depth.crestereo_adapt.estimator import CreStereoEstimator, load_estimator
from visionx.depth.crestereo_adapt.matcher import CreStereoParams
from visionx.depth.crestereo_adapt.port import PortConfig, load_model
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import StereoFrame
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return f"{sha}{'-con-cambios' if dirty else ''}"


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise InvalidInput(f"resolución {text!r}: se esperaba ANCHOxALTO") from exc
    return w, h


def _equipment(device: str) -> dict[str, Any]:
    import torch

    return {
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "dispositivo": device,
    }


def _write_report(out_dir: Path, name: str, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)


# --- latencia -----------------------------------------------------------------------------


def _latency(args: argparse.Namespace) -> int:
    from visionx.calibration import StereoCalibration
    from visionx.camera.rectified import RectifiedSource
    from visionx.camera.replay import ReplaySource
    from visionx.depth import latency

    cfg = load_config(args.config)
    calib_path = ROOT / cfg["camera"]["calibration"]
    source = RectifiedSource(ReplaySource(args.clip), StereoCalibration.load(calib_path))
    frames = []
    while (frame := source.read()) is not None and len(frames) < args.pares:
        frames.append(frame)
    source.close()
    capture = source.geometry
    print(f"{len(frames)} pares rectificados de {args.clip} ({capture.size[0]}x{capture.size[1]})")

    model, device = load_model(PortConfig.from_config(cfg["depth"]["crestereo"]))
    print(f"CREStereo congelado en {device}")
    rows: list[dict[str, Any]] = []
    for text in args.resoluciones:
        size = _parse_size(text)
        for passes in args.pasadas:
            for iters in args.iters:
                for warm in args.arranque:
                    params = CreStereoParams(
                        iters=iters,
                        passes=passes,
                        warm_start=warm == "caliente",
                        warm_reset_every=args.reinicio,
                    )
                    est = load_estimator(
                        cfg, capture, model=model, device=device, params=params, compute_size=size
                    )

                    def run(frame: StereoFrame, est: CreStereoEstimator = est) -> dict[str, float]:
                        result = est.predict(frame)
                        was_warm = est.matcher.last_stats["caliente"] > 0
                        # Con arranque en caliente, cada tanto hay un cuadro en frío: se informa
                        # aparte porque es el peor caso que ve el usuario.
                        key = "caliente_ms" if was_warm else "frio_ms"
                        return {"total_ms": result.inference_ms, key: result.inference_ms}

                    stats = latency.measure(frames, run, args.calentamiento, args.medidos)
                    est.close()
                    total = stats["total_ms"]
                    row = {
                        "resolucion": text,
                        "red": [
                            int(est.matcher.last_stats["ancho_red"]),
                            int(est.matcher.last_stats["alto_red"]),
                        ],
                        "pasadas": passes,
                        "iters": iters,
                        "arranque": warm,
                        **{k: {m: round(x, 2) for m, x in s.items()} for k, s in stats.items()},
                    }
                    rows.append(row)
                    cold = stats.get("frio_ms", {}).get("p50")
                    extra = f", cuadro en frío {cold:.0f} ms" if warm == "caliente" and cold else ""
                    print(
                        f"{text} pasadas={passes} iters={iters:2d} {warm:8s}: "
                        f"P50 {total['p50']:.0f} ms, P95 {total['p95']:.0f} ms{extra}"
                    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-latencia-crestereo-{Path(args.clip).name}"
    _write_report(
        out_dir,
        "latencia.yaml",
        {
            "metodo": "crestereo",
            "clip": str(args.clip),
            "session_id": frames[0].session_id if frames else None,
            "pares_distintos": len(frames),
            "calibracion": cfg["camera"]["calibration"],
            "commit": _git_commit(),
            "comando": " ".join(sys.argv),
            "equipo": _equipment(device),
            "protocolo": {
                "calentamiento": args.calentamiento,
                "medidos": args.medidos,
                "reinicio_caliente_cada": args.reinicio,
            },
            "pesos_sha256": cfg["depth"]["crestereo"]["weights_sha256"],
            "port_commit": cfg["depth"]["crestereo"]["port_commit"],
            "memoria_pico_mb": latency.peak_rss_mb(),
            "filas": rows,
        },
    )
    print(f"resultados en {out_dir}")
    return 0


# --- cinta --------------------------------------------------------------------------------


def _sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Valor bilineal de un mapa en puntos (u, v) con decimales. NaN si cae en un hueco."""
    values = cv2.remap(
        image.astype(np.float32),
        u.astype(np.float32).reshape(-1, 1),
        v.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return values.reshape(-1)


def _distance(u: np.ndarray, v: np.ndarray, z: np.ndarray, g: Any) -> float:
    """Distancia del punto medio entre lentes al centro del damero (igual que cinta.py)."""
    x = (u - g.cx) * z / g.fx
    y = (v - g.cy) * z / g.fy
    center = np.array([x.mean(), y.mean(), z.mean()])
    return float(np.linalg.norm(center - np.array([g.baseline_m / 2, 0.0, 0.0])))


def _tape(args: argparse.Namespace) -> int:
    from visionx.calibration import StereoCalibration
    from visionx.calibration.board import Checkerboard
    from visionx.calibration.detection import detect_corners, match_order
    from visionx.calibration.models import undistort_points
    from visionx.depth.stereo import (
        SgbmEstimator,
        SgbmParams,
        compute_geometry,
        num_disparities_for,
    )

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    cal = StereoCalibration.load(ROOT / cfg["camera"]["calibration"])
    g = cal.geometry()  # con la corrección de disparidad: el camino real de VisionX
    maps = cal.rectify_maps()
    cols, rows = (int(x) for x in args.tablero.split("x"))
    board = Checkerboard(cols, rows, args.cuadrado_mm / 1000)
    folder = Path(args.carpeta)
    # c: el cero de la cinta (vidrio) queda c metros delante del centro óptico. Es del
    # montaje y se midió al ajustar la corrección de disparidad (offset_cinta.py).
    zero_mm = args.cero_mm
    if zero_mm is None:
        zero_mm = float(cal.disparity_offset_source.get("cero_cinta_mm", 0.0))
    zero_m = zero_mm / 1000.0

    model, device = load_model(PortConfig.from_config(depth_cfg["crestereo"]))
    estimators: dict[str, Any] = {}
    for mode in args.modos:
        estimators[f"crestereo-{mode}"] = load_estimator(cfg, g, mode, model=model, device=device)
    for text in args.variantes:  # p. ej. 640x360:i10p1 -> resolución, iteraciones y pasadas
        size_text, _, run_text = text.partition(":")
        try:
            iters, passes = (int(x) for x in run_text.lower().lstrip("i").split("p"))
        except ValueError as exc:
            raise InvalidInput(f"variante {text!r}: se esperaba ANCHOxALTO:iNpM") from exc
        estimators[text] = load_estimator(
            cfg,
            g,
            model=model,
            device=device,
            compute_size=_parse_size(size_text),
            params=CreStereoParams(iters=iters, passes=passes),
        )
    sgbm_size = tuple(int(x) for x in depth_cfg["compute"])
    ndisp = num_disparities_for(compute_geometry(g, sgbm_size), float(depth_cfg["range_m"][0]))
    sgbm = SgbmParams.from_config(depth_cfg["sgbm"], num_disparities=ndisp)
    estimators["sgbm-b0"] = SgbmEstimator("B0", sgbm, g, sgbm_size)
    for name, est in estimators.items():
        print(f"{name}: {est.model_id}")

    measurements = []
    with (folder / "resultados.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            measurements.append((float(row["cinta_m"]), [int(n) for n in row["pares"].split()]))
    measurements.sort(key=lambda m: m[0])

    lines: list[dict[str, Any]] = []
    for tape_m, pairs in measurements:
        per_method: dict[str, list[float]] = {"esquinas": []}
        used: dict[str, list[int]] = {}
        for n in pairs:
            left = cv2.imread(str(folder / f"left_{n}.jpg"))
            right = cv2.imread(str(folder / f"right_{n}.jpg"))
            if left is None or right is None:
                raise SourceError(f"falta el par {n} en {folder}")
            cl, cr = detect_corners(left, board), detect_corners(right, board)
            if cl is None or cr is None:
                print(f"  par {n}: no se detectó el damero, se saltea")
                continue
            pr, _ = match_order(cl.points, cr.points)
            ul = undistort_points(cal.model, cl.points, cal.left, cal.rect.R1, cal.rect.P1)
            ur = undistort_points(cal.model, pr, cal.right, cal.rect.R2, cal.rect.P2)
            u, v = ul[:, 0], ul[:, 1]
            # Referencia: la disparidad de las mismas esquinas detectadas en los dos ojos.
            z_corners = np.array([g.depth_at(float(d)) for d in u - ur[:, 0]], dtype=np.float64)
            per_method["esquinas"].append(_distance(u, v, z_corners, g))

            rect_left, rect_right = cal.rectify(left, right, maps)
            frame = StereoFrame(
                session_id=f"cinta-{folder.name}",
                frame_id=n,
                left=rect_left,
                right=rect_right,
                received_ns=0,
                rectified=True,
                calibration_id=cal.calibration_id,
                geometry_id=g.geometry_id,
            )
            for name, est in estimators.items():
                result = est.predict(frame)
                est.close()  # cada par es una escena distinta: sin arranque en caliente
                sx = result.size[0] / g.width
                sy = result.size[1] / g.height
                z = _sample(result.depth_m, u * sx, v * sy).astype(np.float64)
                ok = np.isfinite(z)
                used.setdefault(name, []).append(int(ok.sum()))
                if ok.sum() >= 0.5 * len(z):
                    per_method.setdefault(name, []).append(_distance(u[ok], v[ok], z[ok], g))
        for name, dists in per_method.items():
            if not dists:
                continue
            mean = float(np.mean(dists)) - zero_m
            lines.append(
                {
                    "cinta_m": tape_m,
                    "metodo": name,
                    "pares": len(dists),
                    "distancia_m": round(mean, 4),
                    "error_mm": round((mean - tape_m) * 1000, 1),
                    "error_pct": round((mean - tape_m) / tape_m * 100, 2),
                    "esquinas_usadas": used.get(name),
                }
            )

    names = ["esquinas", *estimators]
    print(f"\ncero de cinta c = {zero_mm:.1f} mm; error = distancia - c - cinta")
    print(f"{'cinta':>6} " + " ".join(f"{n[-20:]:>20}" for n in names))
    for tape_m, _ in measurements:
        cells = []
        for n in names:
            hit = [x for x in lines if x["cinta_m"] == tape_m and x["metodo"] == n]
            cells.append(
                f"{hit[0]['error_mm']:+8.1f} mm {hit[0]['error_pct']:+6.2f} %" if hit else "—"
            )
        print(f"{tape_m:6.2f} " + " ".join(f"{c:>20}" for c in cells))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-cinta-crestereo-{folder.name}"
    _write_report(
        out_dir,
        "cinta.yaml",
        {
            "carpeta": str(folder),
            "calibracion": cfg["camera"]["calibration"],
            "calibration_id": cal.calibration_id,
            "cero_cinta_mm": zero_mm,
            "commit": _git_commit(),
            "comando": " ".join(sys.argv),
            "equipo": _equipment(device),
            "metodos": {name: est.model_id for name, est in estimators.items()},
            "nota": "la corrección de disparidad de la calibración se ajustó con estas mismas "
            "mediciones y con la disparidad de esquinas: no es una prueba independiente",
            "filas": lines,
        },
    )
    print(f"resultados en {out_dir}")
    return 0


# --- robustez -----------------------------------------------------------------------------


def _robustness(args: argparse.Namespace) -> int:
    from visionx.calibration import StereoCalibration
    from visionx.camera.rectified import RectifiedSource
    from visionx.camera.replay import ReplaySource
    from visionx.depth.crestereo_adapt import robustness
    from visionx.depth.crestereo_adapt.classic import ClassicParams, estimate_alignment
    from visionx.depth.crestereo_adapt.correction import (
        displacement_field,
        homography,
        warp_image,
    )

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    calib_path = ROOT / cfg["camera"]["calibration"]
    source = RectifiedSource(ReplaySource(args.clip), StereoCalibration.load(calib_path))
    everything = []
    while (frame := source.read()) is not None:
        everything.append(frame)
    source.close()
    if not everything:
        raise SourceError(f"{args.clip} no tiene pares")
    # Pares repartidos a lo largo del clip: escenas distintas, no cuadros casi iguales.
    picks = np.linspace(0, len(everything) - 1, min(args.pares, len(everything))).astype(int)
    frames = [everything[i] for i in picks]
    capture = source.geometry
    range_m = (float(depth_cfg["range_m"][0]), float(depth_cfg["range_m"][1]))

    model, device = load_model(PortConfig.from_config(depth_cfg["crestereo"]))
    est = load_estimator(cfg, capture, args.modo, model=model, device=device)
    compute = est.geometry
    # El clásico busca puntos a media resolución de captura: alcanza y es mucho más rápido.
    align_geom = capture.resized(capture.width // 2, capture.height // 2)
    classic = ClassicParams(estimate_yaw=not args.sin_yaw)
    print(f"{len(frames)} pares de {args.clip}; red: {est.model_id}; rango {range_m} m")

    def run(frame: StereoFrame, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probe = StereoFrame(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            left=frame.left,
            right=right,
            received_ns=frame.received_ns,
            rectified=True,
            calibration_id=frame.calibration_id,
            geometry_id=frame.geometry_id,
        )
        result = est.predict(probe)
        est.close()  # cada corrida arranca en frío: no se mezclan niveles
        return result.depth_m, result.valid_mask

    references = [run(f, f.right) for f in frames]
    rows: list[dict[str, Any]] = []
    for parameter, levels in robustness.SWEEPS.items():
        if parameter in args.omitir:
            continue
        for level in levels:
            error = robustness.error_for(parameter, level)
            undo = homography(error, capture)  # arregla
            do = np.linalg.inv(undo)  # desalinea
            vertical = np.abs(displacement_field(do, compute.size)[:, :, 1])
            per_system: dict[str, list[dict[str, float]]] = {
                "sin_corregir": [],
                "clasico": [],
                "oraculo": [],
            }
            recovered: list[float] = []
            accepted = 0
            for frame, (ref, ref_ok) in zip(frames, references, strict=True):
                perturbed = warp_image(frame.right, do)
                depth, ok = run(frame, perturbed)
                per_system["sin_corregir"].append(
                    robustness.depth_change(depth, ok, ref, ref_ok, range_m)
                )
                # Oráculo: deshacer y volver a hacer, en una sola interpolación extra, para
                # que pague el mismo borde perdido que cualquier corrección real.
                restored = warp_image(perturbed, undo)
                depth, ok = run(frame, restored)
                per_system["oraculo"].append(
                    robustness.depth_change(depth, ok, ref, ref_ok, range_m)
                )
                small = (align_geom.width, align_geom.height)
                guess = estimate_alignment(
                    cv2.resize(frame.left, small, interpolation=cv2.INTER_AREA),
                    cv2.resize(perturbed, small, interpolation=cv2.INTER_AREA),
                    align_geom,
                    classic,
                )
                if guess.ok:
                    accepted += 1
                    fixed = warp_image(perturbed, homography(guess.correction, capture))
                    recovered.append(
                        abs(getattr(guess.correction, parameter) - level) / level * 100.0
                    )
                else:
                    fixed = perturbed  # sin corrección confiable, se deja como está
                depth, ok = run(frame, fixed)
                per_system["clasico"].append(
                    robustness.depth_change(depth, ok, ref, ref_ok, range_m)
                )
            row: dict[str, Any] = {
                "parametro": parameter,
                "nivel": level,
                "vertical_px_mediana": round(float(np.median(vertical)), 2),
                "vertical_px_max": round(float(vertical.max()), 2),
                "clasico_aceptado": f"{accepted}/{len(frames)}",
                "clasico_error_parametro_pct": (
                    round(float(np.median(recovered)), 1) if recovered else None
                ),
            }
            for name, values in per_system.items():
                row[name] = {k: round(x, 3) for k, x in robustness.summarize(values).items()}
            rows.append(row)
            print(
                f"{parameter:>9} {level:<6g} vertical {row['vertical_px_mediana']:5.2f} px | "
                + " | ".join(
                    f"{name} {row[name]['mediana_pct']:5.2f} % (>5 %: "
                    f"{row[name]['mas_de_5_pct']:4.1f} %)"
                    for name in per_system
                )
                + f" | clásico aceptado {row['clasico_aceptado']}"
            )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-robustez-crestereo-{Path(args.clip).name}"
    _write_report(
        out_dir,
        "robustez.yaml",
        {
            "clip": str(args.clip),
            "pares": [int(f.frame_id) for f in frames],
            "calibracion": cfg["camera"]["calibration"],
            "metodo": est.model_id,
            "rango_m": list(range_m),
            "referencia": "la misma red sobre el par bien rectificado (no es GT)",
            "commit": _git_commit(),
            "comando": " ".join(sys.argv),
            "equipo": _equipment(device),
            "filas": rows,
        },
    )
    print(f"resultados en {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.depth.crestereo_adapt",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    la = sub.add_parser("latencia", help="tabla de latencia M9 de CREStereo sobre un clip ELP")
    la.add_argument("clip", help="carpeta de una grabación sin rectificar (T02)")
    la.add_argument("--resoluciones", nargs="+", default=["640x360", "960x540", "1280x720"])
    la.add_argument("--iters", nargs="+", type=int, default=[2, 4, 10])
    la.add_argument("--pasadas", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    la.add_argument(
        "--arranque", nargs="+", default=["frio", "caliente"], choices=["frio", "caliente"]
    )
    la.add_argument(
        "--reinicio",
        type=int,
        default=30,
        help="con arranque en caliente, cada cuántos cuadros se arranca en frío",
    )
    la.add_argument("--calentamiento", type=int, default=50)
    la.add_argument("--medidos", type=int, default=300)
    la.add_argument("--pares", type=int, default=100, help="pares distintos en memoria")
    la.add_argument("--config", default=None, help="override YAML explícito")
    la.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    ci = sub.add_parser("cinta", help="distancias contra la cinta con los pares de cinta.py")
    ci.add_argument("carpeta", help="salida de cinta.py: left_N/right_N.jpg y resultados.csv")
    ci.add_argument(
        "--modos", nargs="+", default=["preciso", "continuo"], choices=["preciso", "continuo"]
    )
    ci.add_argument(
        "--variantes",
        nargs="*",
        default=[],
        help="otras formas de correr la red, como 640x360:i10p1",
    )
    ci.add_argument("--tablero", default="9x6")
    ci.add_argument("--cuadrado-mm", type=float, default=38.0)
    ci.add_argument(
        "--cero-mm",
        type=float,
        default=None,
        help="cero de la cinta; por defecto, el guardado en la calibración",
    )
    ci.add_argument("--config", default=None, help="override YAML explícito")
    ci.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    ro = sub.add_parser("robustez", help="cuánto cambia la profundidad al desalinear el par")
    ro.add_argument("clip", help="carpeta de una grabación sin rectificar (T02)")
    ro.add_argument("--modo", default="preciso", choices=["preciso", "continuo"])
    ro.add_argument("--pares", type=int, default=8, help="pares repartidos a lo largo del clip")
    ro.add_argument("--omitir", nargs="*", default=[], help="parámetros que no se barren")
    ro.add_argument("--sin-yaw", action="store_true", help="el clásico no estima el yaw")
    ro.add_argument("--config", default=None, help="override YAML explícito")
    ro.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    args = parser.parse_args(argv)
    try:
        if args.comando == "latencia":
            return _latency(args)
        if args.comando == "robustez":
            return _robustness(args)
        return _tape(args)
    except (InvalidInput, EstimatorError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
