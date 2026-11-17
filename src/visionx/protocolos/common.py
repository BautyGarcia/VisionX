"""Archivos auditables; las rutas de las planillas son relativas a la propia planilla."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: planilla vacía")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no hay filas para escribir")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


def resolve(sheet: Path, value: str) -> Path:
    if not value.strip():
        raise ValueError(f"{sheet}: falta ruta")
    p = Path(value)
    return p if p.is_absolute() else sheet.resolve().parent / p


def number(value, name: str, *, positive: bool = False) -> float:
    n = float(value)
    if not math.isfinite(n) or n < 0 or (positive and n == 0):
        raise ValueError(f"{name}: se esperaba número {'positivo' if positive else 'no negativo'}")
    return n


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def report(out: Path, rows: list[dict], summary, inputs: list[Path], cfg=None) -> dict:
    """Detalle CSV + resumen y procedencia JSON; los fakes se rotulan en summary."""
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "detalle.csv", rows)
    table = summary.get("tabla", summary.get("modelos"))
    if table:
        write_csv(out / "tabla.csv", table)
    else:
        write_csv(
            out / "tabla.csv",
            [{k: v for k, v in summary.items() if not isinstance(v, (list, dict))}],
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "desconocido"
    result = {
        "resumen": summary,
        "comando": sys.argv,
        "commit_base": commit,
        "codigo_protocolos_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "equipo": {"sistema": platform.platform(), "python": platform.python_version()},
        "config": cfg,
        "sha256": {str(p): sha(p) for p in dict.fromkeys(inputs)},
    }
    write_json(out / "resumen.json", result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result
