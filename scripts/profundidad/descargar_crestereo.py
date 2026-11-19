"""Baja el código y los pesos de CREStereo que usa `visionx.depth.crestereo_adapt` (T12).

Uso (desde la raíz del repo, una sola vez por máquina):
    uv run python scripts/profundidad/descargar_crestereo.py

Deja dos cosas fuera de Git:
    third_party/crestereo/                    port a PyTorch, en un commit fijo
    weights/crestereo/crestereo_eth3d.pth     pesos publicados, verificados por SHA-256

Por qué no están en el repo: el port (ibaiGorordo/CREStereo-Pytorch) no declara licencia
(su issue #22 sigue abierto), así que no se redistribuye; y los pesos nunca se commitean.
El original de Megvii (megvii-research/CREStereo) es Apache-2.0.

El commit y el hash salen de `depth.crestereo` en configs/default.yaml: si cambian ahí,
este script baja la versión nueva. No se ejecuta en tests ni en CI.
"""

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from visionx.shared.config import ROOT, load_config

PORT_URL = "https://github.com/ibaiGorordo/CREStereo-Pytorch.git"
# Id de Google Drive que publica el README del port para crestereo_eth3d.pth (~21 MB).
WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1D2s1v4VhJlNz98FQpFxf_kBAKQVN_7xo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_port(folder: Path, commit: str) -> None:
    marker = folder / "COMMIT"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == commit:
        print(f"port ya presente en {folder} ({commit[:10]})")
        return
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", PORT_URL, str(folder)], check=True)
    subprocess.run(["git", "-C", str(folder), "checkout", "--quiet", commit], check=True)
    # Sin .git no queda un repositorio anidado dentro del nuestro; el commit va en un archivo.
    shutil.rmtree(folder / ".git")
    marker.write_text(commit + "\n", encoding="utf-8")
    print(f"port en {folder} ({commit[:10]})")


def fetch_weights(path: Path, expected: str) -> None:
    if path.exists() and sha256(path) == expected:
        print(f"pesos ya presentes y verificados: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".descarga")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310 - URL fija, https
    found = sha256(tmp)
    if found != expected:
        tmp.unlink()
        raise SystemExit(f"SHA-256 inesperado en los pesos: {found} (se esperaba {expected})")
    tmp.replace(path)
    print(f"pesos verificados (SHA-256 {expected[:12]}…): {path}")


def main() -> int:
    cfg = load_config(None)["depth"]["crestereo"]
    fetch_port(ROOT / cfg["port_dir"], cfg["port_commit"])
    fetch_weights(ROOT / cfg["weights"], cfg["weights_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
