"""Comandos de detección (T06).

    uv run python -m visionx.detection descargar [--modelo yolo26n-seg yolo11n-seg]
    uv run python -m visionx.detection clases [TEXTO ...]
    uv run python -m visionx.detection detectar ENTRADA [--salida DIR] [--imagenes] [--rectificar]
    uv run python -m visionx.detection medir ENTRADA [--modelos A B] [--repeticiones 3] [--csv X]
    uv run python -m visionx.detection presencia ENTRADA --clases cup bottle [--csv X]

ENTRADA es una imagen, una carpeta de imágenes o una grabación de `python -m visionx.camera
grabar`. El detector se configura en la sección `detection` de configs/default.yaml.
Requiere `uv sync --extra deteccion`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from visionx.detection.catalog import COCO_CLASSES, VISIONX_CLASSES, VISIONX_ES, ClassCatalog
from visionx.detection.draw import draw_detection
from visionx.detection.yolo import DetectorSettings, YoloDetector, download_weights, sha256_file
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import Detection, StereoFrame
from visionx.shared.errors import EstimatorError, InvalidInput, SourceError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# --- entradas ---------------------------------------------------------------------------


def image_frames(paths: list[Path], session_id: str) -> Iterator[StereoFrame]:
    """Imágenes sueltas como frames. El detector solo mira el ojo izquierdo, así que el derecho
    es la misma imagen. La geometría se declara como "imagen-WxH": no hay calibración."""
    for frame_id, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SourceError(f"no se pudo leer la imagen {path}")
        h, w = image.shape[:2]
        yield StereoFrame(
            session_id=session_id,
            frame_id=frame_id,
            left=image,
            right=image,
            received_ns=time.monotonic_ns(),
            geometry_id=f"imagen-{w}x{h}",
            source_id=f"imagen:{path.name}",
        )


def recording_frames(folder: Path, cfg: dict[str, Any], rectify: bool) -> Iterator[StereoFrame]:
    """Pares de una grabación T02. Sin --rectificar quedan en la imagen cruda y se rotulan así:
    sus cajas no se pueden combinar con una profundidad rectificada."""
    from visionx.camera import RectifiedSource, ReplaySource

    source = ReplaySource(folder)
    if rectify:
        from visionx.calibration import StereoCalibration

        path = Path((cfg.get("camera") or {}).get("calibration") or "")
        if not str(path):
            raise InvalidInput("--rectificar necesita camera.calibration en la config")
        source = RectifiedSource(
            source, StereoCalibration.load(path if path.is_absolute() else ROOT / path)
        )
    try:
        while (frame := source.read()) is not None:
            if frame.geometry_id is None:
                w, h = frame.size
                frame = dataclasses.replace(frame, geometry_id=f"sin-rectificar-{w}x{h}")
            yield frame
    finally:
        source.close()


def open_input(entrada: str, cfg: dict[str, Any], rectify: bool = False) -> Iterator[StereoFrame]:
    path = Path(entrada)
    if path.is_dir() and (path / "manifest.yaml").is_file():
        return recording_frames(path, cfg, rectify)
    if rectify:
        raise InvalidInput("--rectificar solo aplica a grabaciones, no a imágenes sueltas")
    if path.is_file():
        return image_frames([path], f"imagen-{path.stem}")
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise SourceError(f"{path}: no hay imágenes ni manifest.yaml de grabación")
        return image_frames(images, f"imagenes-{path.name}")
    raise SourceError(f"{entrada}: no existe")


# --- resúmenes --------------------------------------------------------------------------


def latency_summary(ms: list[float]) -> dict[str, float | int]:
    """Mediana y p90 en ms. La media sola esconde los picos que el usuario sí nota."""
    if not ms:
        return {"n": 0}
    a = np.asarray(ms, dtype=np.float64)
    return {
        "n": len(ms),
        "mediana_ms": round(float(np.median(a)), 1),
        "p90_ms": round(float(np.percentile(a, 90)), 1),
        "media_ms": round(float(a.mean()), 1),
        "max_ms": round(float(a.max()), 1),
    }


def detection_record(det: Detection, catalog: ClassCatalog) -> dict[str, Any]:
    """Una línea JSONL por frame: lo justo para auditar o anotar después, sin las máscaras."""
    objects = []
    for i, cls in enumerate(det.classes):
        obj: dict[str, Any] = {
            "clase": cls,
            "nombre": catalog.spoken(cls),
            "score": round(float(det.scores[i]), 4),
            "caja_xyxy": [round(float(c), 1) for c in det.boxes[i]],
        }
        if det.track_ids is not None:
            obj["track"] = det.track_ids[i]
        if det.masks is not None:
            obj["mascara_px"] = int(det.masks[i].sum())
        objects.append(obj)
    return {
        "session_id": det.session_id,
        "frame_id": det.frame_id,
        "geometry_id": det.geometry_id,
        "model_id": det.model_id,
        "inference_ms": round(det.inference_ms, 2),
        "objetos": objects,
    }


def equipo() -> dict[str, Any]:
    import torch

    return {
        # version() y no release(): en Windows 11, release() todavía dice "10".
        "sistema": f"{platform.system()} {platform.version()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # TorchVersion: subclase de str que YAML no serializa
        "hilos_torch": torch.get_num_threads(),
    }


# --- comandos ---------------------------------------------------------------------------


def cmd_descargar(args: argparse.Namespace) -> int:
    settings = DetectorSettings.from_config(load_config(args.config))
    folder = settings.weights_path.parent
    for model in args.modelo or [settings.model]:
        path = download_weights(model, folder)
        print(f"{model}: {path}  {path.stat().st_size} bytes  sha256 {sha256_file(path)}")
    print("Registrar los sha256 en BITACORA: identifican los pesos usados en cada resultado.")
    return 0


def cmd_vocabulario(args: argparse.Namespace) -> int:
    from visionx.detection.yolo import build_vocabulary_weights

    settings = DetectorSettings.from_config(load_config(args.config))
    folder = settings.weights_path.parent
    path = build_vocabulary_weights(args.base, args.salida or f"{args.base}-visionx", folder)
    print(f"{path}: {len(VISIONX_CLASSES)} clases  sha256 {sha256_file(path)}")
    print("Registrar el sha256 en BITACORA; la config elige el modelo con detection.model.")
    return 0


def cmd_clases(args: argparse.Namespace) -> int:
    catalog = ClassCatalog(VISIONX_CLASSES)
    if not args.texto:
        for cls in VISIONX_CLASSES:
            print(f"{cls:22} {', '.join(VISIONX_ES[cls])}")
        print(f"{len(VISIONX_CLASSES)} clases de VisionX ({len(COCO_CLASSES)} COCO con los pesos "
              "oficiales). Estar en la lista no prueba que se detecte bien.")  # fmt: skip
        return 0
    status = 0
    for text in args.texto:
        cls = catalog.resolve(text)
        print(f"{text!r} -> {cls if cls else 'no está en el catálogo'}")
        status |= cls is None
    return status


def cmd_detectar(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    settings = DetectorSettings.from_config(cfg)
    if args.modelo:
        settings = dataclasses.replace(settings, model=args.modelo)
    if args.sin_seguimiento:
        settings = dataclasses.replace(settings, track=False)
    out = Path(args.salida or ROOT / "outputs" / "deteccion" / time.strftime("%Y%m%d-%H%M%S"))
    detector = YoloDetector(settings)  # primero: sin pesos no queda una carpeta vacía
    out.mkdir(parents=True, exist_ok=True)
    frames = open_input(args.entrada, cfg, args.rectificar)
    ms: list[float] = []
    counts: Counter[str] = Counter()
    try:
        with open(out / "detecciones.jsonl", "w", encoding="utf-8", newline="\n") as jsonl:
            for frame in frames:
                det = detector.predict(frame)
                ms.append(det.inference_ms)
                counts.update(det.classes)
                jsonl.write(json.dumps(detection_record(det, detector.catalog), ensure_ascii=False))
                jsonl.write("\n")
                if args.imagenes or args.ver:
                    view = draw_detection(frame.left, det, detector.catalog.spoken)
                    if args.imagenes:
                        cv2.imwrite(str(out / f"{frame.frame_id:06d}.jpg"), view)
                    if args.ver:
                        cv2.imshow("VisionX - deteccion", view)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if args.max and len(ms) >= args.max:
                    break
    finally:
        detector.close()
        if args.ver:
            cv2.destroyAllWindows()
    summary = {
        "entrada": str(args.entrada),
        "model_id": detector.model_id,
        "pesos_sha256": detector.weights_sha256,
        # Listas, no tuplas: safe_dump solo escribe tipos YAML estándar.
        "config": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(settings).items()
        },
        "dispositivo": detector.device,
        "equipo": equipo(),
        "frames": len(ms),
        # El primer frame incluye inicializar torch/la red: se informa aparte para no inflar p90.
        "primer_frame_ms": round(ms[0], 1) if ms else None,
        "latencia_sin_primero": latency_summary(ms[1:]),
        "objetos_por_clase": dict(counts.most_common()),
    }
    (out / "resumen.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"{len(ms)} frames; primero {summary['primer_frame_ms']} ms (incluye inicialización)")
    print(f"latencia del resto: {summary['latencia_sin_primero']}")
    print(f"objetos: {dict(counts.most_common(10))}")
    print(f"resultados en {out}")
    return 0


def cmd_presencia(args: argparse.Namespace) -> int:
    """En qué % de los cuadros aparece cada clase pedida (calidad de detección por config).

    Una fila por clase, más la latencia del detector sobre esos mismos cuadros. Sirve para
    comparar dispositivo e imgsz sobre los clips de buscar/distancias (docs/VELOCIDAD.md).
    """
    from collections import Counter

    cfg = load_config(args.config)
    settings = dataclasses.replace(DetectorSettings.from_config(cfg), track=False)
    if args.imgsz:
        settings = dataclasses.replace(settings, imgsz=args.imgsz)
    if args.dispositivo:
        settings = dataclasses.replace(settings, device=args.dispositivo)
    detector = YoloDetector(settings)
    wanted = set(args.clases)
    counts: Counter = Counter()
    best: dict[str, list[float]] = {c: [] for c in wanted}
    latencies: list[float] = []
    n = 0
    try:
        for frame in islice(open_input(args.entrada, cfg), args.max or None):
            det = detector.predict(frame)
            n += 1
            if n > args.calentamiento:
                latencies.append(det.inference_ms)
            seen: dict[str, float] = {}
            for cls, score in zip(det.classes, det.scores, strict=True):
                if cls in wanted:
                    seen[cls] = max(seen.get(cls, 0.0), float(score))
            for cls, score in seen.items():
                counts[cls] += 1
                best[cls].append(score)
    finally:
        detector.close()
    if n == 0:
        print("no hay frames", file=sys.stderr)
        return 1
    lat = latency_summary(latencies or [np.nan])
    rows = []
    for cls in sorted(wanted):
        rows.append({
            "entrada": str(args.entrada),
            "clase": cls,
            "frames": n,
            "presencia_pct": round(100 * counts[cls] / n, 1),
            "score_mediano": round(float(np.median(best[cls])), 3) if best[cls] else None,
            "modelo": settings.model,
            "dispositivo": detector.device,
            "imgsz": settings.imgsz,
            "det_mediana_ms": lat["mediana_ms"],
            "det_p90_ms": lat["p90_ms"],
        })  # fmt: skip
        print(f"{cls:12} presente en {rows[-1]['presencia_pct']:5.1f} % de {n} cuadros  "
              f"(score mediano {rows[-1]['score_mediano']})")  # fmt: skip
    print(f"detector {settings.model} en {detector.device} a {settings.imgsz}: "
          f"mediana {lat['mediana_ms']} ms, p90 {lat['p90_ms']} ms")  # fmt: skip
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
        print(f"filas agregadas a {path}")
    return 0


def cmd_medir(args: argparse.Namespace) -> int:
    """Latencia de varios modelos sobre los mismos frames, sin tracking (comparación justa).

    Se cargan los frames antes de medir para no contar la lectura de disco, y se descartan las
    primeras corridas: la primera inferencia incluye inicialización y no representa el uso.
    """
    cfg = load_config(args.config)
    base = DetectorSettings.from_config(cfg)
    frames = list(islice(open_input(args.entrada, cfg), args.max or None))
    if not frames:
        print("no hay frames para medir", file=sys.stderr)
        return 1
    rows = []
    for model in args.modelos or [base.model]:
        settings = dataclasses.replace(base, model=model, track=False)
        if args.imgsz:
            settings = dataclasses.replace(settings, imgsz=args.imgsz)
        if args.dispositivo:
            settings = dataclasses.replace(settings, device=args.dispositivo)
        detector = YoloDetector(settings)
        try:
            for _ in range(args.calentamiento):
                detector.predict(frames[0])
            total: list[float] = []
            network: list[float] = []
            objects: list[int] = []
            for _ in range(args.repeticiones):
                for frame in frames:
                    det = detector.predict(frame)
                    total.append(det.inference_ms)
                    network.append(float(detector.last_speed_ms.get("inference", np.nan)))
                    objects.append(len(det))
        finally:
            detector.close()
        w, h = frames[0].size
        row = {
            "modelo": model,
            "model_id": detector.model_id,
            "dispositivo": detector.device,
            "imgsz": settings.imgsz,
            "imagen": f"{w}x{h}",
            "frames": len(frames),
            "repeticiones": args.repeticiones,
            **latency_summary(total),
            "red_mediana_ms": round(float(np.nanmedian(network)), 1),
            "objetos_medio": round(float(np.mean(objects)), 2),
        }
        rows.append(row)
        print(
            f"{model:14} total mediana {row['mediana_ms']:7.1f} ms  p90 {row['p90_ms']:7.1f} ms  "
            f"red {row['red_mediana_ms']:7.1f} ms  objetos {row['objetos_medio']}"
        )
    info = equipo()
    print(f"equipo: {info}")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[*rows[0], *info])
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **info})
        print(f"tabla en {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visionx.detection",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("descargar", help="bajar pesos oficiales a detection.weights_dir")
    p.add_argument("--modelo", nargs="+", help="por defecto, detection.model de la config")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_descargar)

    p = sub.add_parser("vocabulario", help="armar los pesos YOLOE con el vocabulario de VisionX")
    p.add_argument("--base", default="yoloe-26s-seg", help="pesos YOLOE oficiales de partida")
    p.add_argument("--salida", help="nombre del modelo armado (por defecto <base>-visionx)")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_vocabulario)

    p = sub.add_parser("clases", help="catálogo en español, o qué clase corresponde a un texto")
    p.add_argument("texto", nargs="*")
    p.set_defaults(fn=cmd_clases)

    p = sub.add_parser("detectar", help="correr el detector y guardar JSONL, resumen e imágenes")
    p.add_argument("entrada")
    p.add_argument("--salida", help="carpeta de resultados (por defecto outputs/deteccion/...)")
    p.add_argument("--modelo", help="pisa detection.model")
    p.add_argument("--imagenes", action="store_true", help="guardar cada frame anotado")
    p.add_argument("--ver", action="store_true", help="mostrar en una ventana (q para salir)")
    p.add_argument("--rectificar", action="store_true", help="rectificar la grabación primero")
    p.add_argument("--sin-seguimiento", action="store_true", help="sin ByteTrack")
    p.add_argument("--max", type=int, default=0, help="procesar como máximo N frames")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_detectar)

    p = sub.add_parser("medir", help="comparar latencia de modelos sobre los mismos frames")
    p.add_argument("entrada")
    p.add_argument("--modelos", nargs="+", help="por defecto, detection.model")
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--calentamiento", type=int, default=3)
    p.add_argument("--imgsz", type=int, help="pisa detection.imgsz")
    p.add_argument("--max", type=int, default=0, help="usar como máximo N frames")
    p.add_argument("--csv", help="guardar la tabla en este CSV")
    p.add_argument("--dispositivo", help="pisa detection.device: cpu | mps | cuda:0 | auto")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_medir)

    p = sub.add_parser("presencia", help="% de cuadros con cada clase pedida (calidad por config)")
    p.add_argument("entrada")
    p.add_argument("--clases", nargs="+", required=True, help="clases COCO, ej. cup bottle")
    p.add_argument("--imgsz", type=int, help="pisa detection.imgsz")
    p.add_argument("--dispositivo", help="pisa detection.device: cpu | mps | cuda:0 | auto")
    p.add_argument("--max", type=int, default=0, help="usar como máximo N frames")
    p.add_argument("--calentamiento", type=int, default=3, help="cuadros sin contar en la latencia")
    p.add_argument("--csv", help="agregar las filas a este CSV")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_presencia)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (EstimatorError, SourceError, InvalidInput) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
