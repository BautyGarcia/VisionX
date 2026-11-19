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


def _spread_rectified(
    clip: str, cal: Any, count: int, run_length: int = 1
) -> tuple[list[StereoFrame], Any]:
    """Pares rectificados repartidos a lo largo del clip (escenas distintas).

    Con `run_length` = 1 son `count` pares sueltos. Con más, son `count` tramos de esa cantidad
    de cuadros seguidos, para medir cosas que dependen del cuadro anterior. Recorre el clip
    sin guardarlo: un clip largo no entra en memoria a 1920x1080.
    """
    from visionx.camera.replay import ReplaySource

    source = ReplaySource(clip, verify_hashes=False)
    total = len(source)
    if total == 0:
        raise SourceError(f"{clip} no tiene pares")
    last_start = max(total - run_length, 0)
    starts = np.linspace(0, last_start, min(count, last_start + 1)).astype(int).tolist()
    picks = {s + k for s in starts for k in range(run_length) if s + k < total}
    geometry = cal.geometry()
    maps = cal.rectify_maps()
    frames: list[StereoFrame] = []
    for index in range(total):
        raw = source.read()
        if raw is None:
            break
        if index not in picks:
            continue
        left, right = cal.rectify(raw.left, raw.right, maps)
        frames.append(
            StereoFrame(
                session_id=raw.session_id,
                frame_id=raw.frame_id,
                left=left,
                right=right,
                received_ns=raw.received_ns,
                rectified=True,
                calibration_id=cal.calibration_id,
                geometry_id=geometry.geometry_id,
            )
        )
    source.close()
    return frames, geometry


# --- robustez -----------------------------------------------------------------------------


def _robustness(args: argparse.Namespace) -> int:
    from visionx.calibration import StereoCalibration
    from visionx.depth.crestereo_adapt import robustness
    from visionx.depth.crestereo_adapt.classic import ClassicParams, fit_correction, match_points
    from visionx.depth.crestereo_adapt.correction import (
        displacement_field,
        homography,
        warp_image,
    )

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    cal = StereoCalibration.load(ROOT / cfg["camera"]["calibration"])
    frames, capture = _spread_rectified(args.clip, cal, args.pares)
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
                "clasico_acumulado": [],
                "oraculo": [],
            }
            small = (align_geom.width, align_geom.height)
            perturbed_all, matches_all = [], []
            for frame in frames:
                perturbed = warp_image(frame.right, do)
                perturbed_all.append(perturbed)
                matches_all.append(
                    match_points(
                        cv2.resize(frame.left, small, interpolation=cv2.INTER_AREA),
                        cv2.resize(perturbed, small, interpolation=cv2.INTER_AREA),
                        classic,
                    )
                )
            # Acumulado: el error es de la cámara, no de la escena. Juntar los puntos de todos
            # los pares y ajustar una sola vez es lo que haría el sistema a lo largo del tiempo.
            pooled = fit_correction(
                np.vstack([m[0] for m in matches_all]),
                np.vstack([m[1] for m in matches_all]),
                align_geom,
                classic,
            )
            pooled_fix = homography(pooled.correction, capture) if pooled.ok else None
            recovered: list[float] = []
            accepted = 0
            for frame, perturbed, (pts_l, pts_r), (ref, ref_ok) in zip(
                frames, perturbed_all, matches_all, references, strict=True
            ):

                def change(right: np.ndarray, frame=frame, ref=ref, ref_ok=ref_ok) -> dict:
                    depth, ok = run(frame, right)
                    return robustness.depth_change(depth, ok, ref, ref_ok, range_m)

                per_system["sin_corregir"].append(change(perturbed))
                # Oráculo: la corrección exacta, con la misma interpolación extra y el mismo
                # borde perdido que paga cualquier corrección real.
                per_system["oraculo"].append(change(warp_image(perturbed, undo)))
                guess = fit_correction(pts_l, pts_r, align_geom, classic)
                fixed = perturbed  # sin corrección confiable, se deja como está
                if guess.ok:
                    accepted += 1
                    fixed = warp_image(perturbed, homography(guess.correction, capture))
                    recovered.append(
                        abs(getattr(guess.correction, parameter) - level) / level * 100.0
                    )
                per_system["clasico"].append(change(fixed))
                fixed = perturbed if pooled_fix is None else warp_image(perturbed, pooled_fix)
                per_system["clasico_acumulado"].append(change(fixed))
            row: dict[str, Any] = {
                "parametro": parameter,
                "nivel": level,
                "vertical_px_mediana": round(float(np.median(vertical)), 2),
                "vertical_px_max": round(float(vertical.max()), 2),
                "clasico_aceptado": f"{accepted}/{len(frames)}",
                "clasico_error_parametro_pct": (
                    round(float(np.median(recovered)), 1) if recovered else None
                ),
                "acumulado": {
                    "ok": pooled.ok,
                    "pares_de_puntos": pooled.inliers,
                    "roll_deg": round(pooled.correction.roll_deg, 4),
                    "pitch_deg": round(pooled.correction.pitch_deg, 4),
                    "yaw_deg": round(pooled.correction.yaw_deg, 4),
                    "log_focal": round(pooled.correction.log_focal, 5),
                    "sigma": {k: round(float(x), 5) for k, x in pooled.sigma.items()},
                },
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


# --- descalibración real ------------------------------------------------------------------


def _real(args: argparse.Namespace) -> int:
    """Pares reales tomados con una calibración que ya no describe a la cámara.

    No usa la red: mide si el estimador clásico encuentra la corrección y cuánto arregla. Si el
    par tiene un damero, sus esquinas dan dos controles independientes de los puntos SIFT: la
    diferencia de filas y la escala del tablero triangulado contra su tamaño real.
    """
    from visionx.calibration import StereoCalibration
    from visionx.calibration.board import Checkerboard
    from visionx.calibration.calibrate import similarity_scale
    from visionx.calibration.detection import detect_corners, match_order
    from visionx.calibration.models import undistort_points
    from visionx.depth.crestereo_adapt.classic import ClassicParams, fit_correction, match_points
    from visionx.depth.crestereo_adapt.correction import apply_homography, homography

    cal = StereoCalibration.load(args.calibracion)
    g = cal.geometry()
    maps = cal.rectify_maps()
    p1, p2 = cal.rect.P1, cal.rect.P2
    cols, rows_ = (int(x) for x in args.tablero.split("x"))
    board = Checkerboard(cols, rows_, args.cuadrado_mm / 1000)
    align_geom = g.resized(g.width // 2, g.height // 2)
    small = (align_geom.width, align_geom.height)
    variants = {
        "con_yaw": ClassicParams(),
        "sin_yaw": ClassicParams(estimate_yaw=False),
        "solo_giros": ClassicParams(estimate_yaw=False, estimate_focal=False),
    }

    def board_check(ul: np.ndarray, ur: np.ndarray) -> tuple[float, float]:
        """(mediana de |diferencia de filas| en px, error de escala del tablero en %)."""
        points = cv2.triangulatePoints(p1, p2, ul.T, ur.T)
        points = (points[:3] / points[3]).T
        scale = (similarity_scale(board.object_points().astype(np.float64), points) - 1) * 100
        return float(np.median(np.abs(ul[:, 1] - ur[:, 1]))), float(scale)

    report: list[dict[str, Any]] = []
    for folder_text in args.carpetas:
        folder = Path(folder_text)
        numbers = sorted(int(f.stem.split("_")[1]) for f in folder.glob("left_*.jpg"))
        pairs = []
        for n in numbers:
            left = cv2.imread(str(folder / f"left_{n}.jpg"))
            right = cv2.imread(str(folder / f"right_{n}.jpg"))
            if left is None or right is None:
                continue
            rect_l, rect_r = cal.rectify(left, right, maps)
            matches = match_points(
                cv2.resize(rect_l, small, interpolation=cv2.INTER_AREA),
                cv2.resize(rect_r, small, interpolation=cv2.INTER_AREA),
            )
            corners = None
            cl, cr = detect_corners(left, board), detect_corners(right, board)
            if cl is not None and cr is not None:
                pr, _ = match_order(cl.points, cr.points)
                corners = (
                    undistort_points(cal.model, cl.points, cal.left, cal.rect.R1, p1),
                    undistort_points(cal.model, pr, cal.right, cal.rect.R2, p2),
                )
            pairs.append((n, matches, corners))
        if not pairs:
            raise SourceError(f"{folder} no tiene pares left_N/right_N.jpg")
        name = f"{folder.parent.name}/{folder.name}"
        print(f"\n== {name}: {len(pairs)} pares, calibración {cal.calibration_id}")
        for n, _, corners in pairs:
            if corners is not None:
                dy, scale = board_check(*corners)
                print(f"  par {n}: damero sin corregir: filas {dy:.2f} px, escala {scale:+.2f} %")
        for label, params in variants.items():
            pooled = fit_correction(
                np.vstack([m[0] for _, m, _ in pairs]),
                np.vstack([m[1] for _, m, _ in pairs]),
                align_geom,
                params,
            )
            c = pooled.correction
            entry: dict[str, Any] = {
                "carpeta": name,
                "variante": label,
                "ok": pooled.ok,
                "motivo": pooled.reason,
                "pares_de_puntos": pooled.inliers,
                "roll_deg": round(c.roll_deg, 4),
                "pitch_deg": round(c.pitch_deg, 4),
                "yaw_deg": round(c.yaw_deg, 4),
                "log_focal": round(c.log_focal, 5),
                "sigma": {k: round(float(x), 5) for k, x in pooled.sigma.items()},
                "filas_sift_antes_px": round(pooled.residual_before_px * 2, 3),
                "filas_sift_despues_px": round(pooled.residual_after_px * 2, 3),
                "damero": [],
            }
            print(
                f"  {label:>10}: roll {c.roll_deg:+.3f}° pitch {c.pitch_deg:+.3f}° "
                f"yaw {c.yaw_deg:+.3f}° (±{pooled.sigma['yaw']:.3f}) focal "
                f"{c.log_focal * 100:+.3f} % | filas SIFT {entry['filas_sift_antes_px']:.2f} -> "
                f"{entry['filas_sift_despues_px']:.2f} px, {pooled.inliers} puntos"
                + ("" if pooled.ok else f" | NO CONFIABLE: {pooled.reason}")
            )
            fix = homography(c, g)
            for n, _, corners in pairs:
                if corners is None:
                    continue
                ul, ur = corners
                dy, scale = board_check(ul, apply_homography(fix, ur))
                entry["damero"].append(
                    {"par": n, "filas_px": round(dy, 3), "escala_pct": round(scale, 3)}
                )
                print(f"      par {n}: damero corregido: filas {dy:.2f} px, escala {scale:+.2f} %")
            report.append(entry)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-descalibracion-real"
    _write_report(
        out_dir,
        "descalibracion.yaml",
        {
            "calibracion": str(args.calibracion),
            "calibration_id": cal.calibration_id,
            "commit": _git_commit(),
            "comando": " ".join(sys.argv),
            "nota": "filas SIFT a resolución de captura (medidas a la mitad y escaladas)",
            "filas": report,
        },
    )
    print(f"resultados en {out_dir}")
    return 0


# --- coherencia del modo continuo ---------------------------------------------------------


def _coherence(args: argparse.Namespace) -> int:
    """Cuánto se aparta el modo continuo del preciso, en escenas reales y por distancia.

    No hay profundidad de referencia en los clips, así que se compara contra el modo preciso,
    que es el que se midió con cinta. Dice cuánto se pierde por correr rápido; no dice si el
    preciso acierta. Los cuadros van en tramos seguidos para que el arranque en caliente
    trabaje como en la app.
    """
    from visionx.calibration import StereoCalibration
    from visionx.depth.crestereo_adapt import robustness

    cfg = load_config(args.config)
    depth_cfg = cfg["depth"]
    cal = StereoCalibration.load(ROOT / cfg["camera"]["calibration"])
    frames, capture = _spread_rectified(args.clip, cal, args.tramos, args.largo)
    model, device = load_model(PortConfig.from_config(depth_cfg["crestereo"]))
    reference = load_estimator(cfg, capture, "preciso", model=model, device=device)
    systems: dict[str, CreStereoEstimator] = {
        "continuo": load_estimator(cfg, capture, "continuo", model=model, device=device)
    }
    for text in args.variantes:
        size_text, _, run_text = text.partition(":")
        try:
            iters, passes = (int(x) for x in run_text.lower().lstrip("i").split("p"))
        except ValueError as exc:
            raise InvalidInput(f"variante {text!r}: se esperaba ANCHOxALTO:iNpM") from exc
        systems[text] = load_estimator(
            cfg,
            capture,
            model=model,
            device=device,
            compute_size=_parse_size(size_text),
            params=CreStereoParams(iters=iters, passes=passes),
        )
    bins = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 6.0)]
    print(
        f"{len(frames)} pares en {args.tramos} tramos de {args.largo}; referencia "
        f"{reference.model_id}"
    )
    samples: dict[tuple[str, str], list[dict[str, float]]] = {}
    for index, frame in enumerate(frames):
        ref = reference.predict(frame)
        if index % args.largo == 0:
            for est in systems.values():
                est.close()  # tramo nuevo: otra escena, se arranca en frío
        for name, est in systems.items():
            result = est.predict(frame)
            warm = est.matcher.last_stats["caliente"] > 0
            # La referencia se lleva al tamaño del sistema tomando el píxel más cercano:
            # promediar profundidades mezclaría el objeto con el fondo en los bordes.
            size = result.size
            ref_depth = cv2.resize(ref.depth_m, size, interpolation=cv2.INTER_NEAREST)
            ref_ok = cv2.resize(
                ref.valid_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for low, high in bins:
                label = f"{low:g}-{high:g} m"
                stats = robustness.depth_change(
                    result.depth_m, result.valid_mask, ref_depth, ref_ok, (low, high)
                )
                if stats["cobertura"] > 0:
                    samples.setdefault((name, label), []).append(stats)
                    if name == "continuo":
                        key = (
                            "continuo, cuadros en frío"
                            if not warm
                            else "continuo, cuadros en caliente",
                            label,
                        )
                        samples.setdefault(key, []).append(stats)

    rows = []
    for (name, label), values in sorted(samples.items()):
        summary = robustness.summarize(values)
        rows.append(
            {
                "sistema": name,
                "rango": label,
                "pares": len(values),
                **{k: round(x, 3) for k, x in summary.items()},
            }
        )
    current = None
    for row in rows:
        if row["sistema"] != current:
            current = row["sistema"]
            print(f"\n{current}")
        print(
            f"  {row['rango']:>8}: mediana {row['mediana_pct']:5.2f} %, "
            f"P90 {row['p90_pct']:6.2f} %, "
            f"más de 5 %: {row['mas_de_5_pct']:5.1f} % de los píxeles ({row['pares']} pares)"
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.salida) / f"{stamp}-coherencia-crestereo-{Path(args.clip).name}"
    _write_report(
        out_dir,
        "coherencia.yaml",
        {
            "clip": str(args.clip),
            "tramos": args.tramos,
            "largo": args.largo,
            "referencia": reference.model_id,
            "sistemas": {name: est.model_id for name, est in systems.items()},
            "nota": "diferencia relativa de profundidad contra el modo preciso; no es GT",
            "calibracion": cfg["camera"]["calibration"],
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
    re = sub.add_parser(
        "descalibracion", help="estimador clásico sobre pares reales mal calibrados"
    )
    re.add_argument("calibracion", help="la calibración que ya no describe a la cámara")
    re.add_argument("carpetas", nargs="+", help="carpetas con left_N.jpg y right_N.jpg crudos")
    re.add_argument("--tablero", default="9x6")
    re.add_argument("--cuadrado-mm", type=float, default=38.0)
    re.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    co = sub.add_parser("coherencia", help="modo continuo contra el preciso, por distancia")
    co.add_argument("clip", help="carpeta de una grabación sin rectificar (T02)")
    co.add_argument("--tramos", type=int, default=6, help="tramos repartidos en el clip")
    co.add_argument("--largo", type=int, default=12, help="cuadros seguidos por tramo")
    co.add_argument("--variantes", nargs="*", default=[], help="como 960x540:i4p2 (en frío)")
    co.add_argument("--config", default=None, help="override YAML explícito")
    co.add_argument("--salida", default=str(ROOT / "outputs" / "profundidad"))
    args = parser.parse_args(argv)
    try:
        if args.comando == "latencia":
            return _latency(args)
        if args.comando == "robustez":
            return _robustness(args)
        if args.comando == "descalibracion":
            return _real(args)
        if args.comando == "coherencia":
            return _coherence(args)
        return _tape(args)
    except (InvalidInput, EstimatorError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
