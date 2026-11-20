"""Perfiles de cámaras estéreo: reconocer la cámara enchufada y armar su calibración nominal.

Primer paso del plug and play (docs/PLUG-AND-PLAY.md, sección 3). Cada cámara conocida tiene un
YAML en `configs/camaras/` con su geometría de la hoja de datos; las que se calibran con la
pantalla «Cámara nueva» se guardan en `configs/camaras/locales/` (fuera de Git) y pisan a las
de la carpeta general si tienen el mismo `id`.

Con un perfil sin calibración propia se arma una **calibración nominal**: pinhole con el campo y
la separación del perfil, centro en el medio, ojos paralelos y, si el perfil la trae, la
distorsión radial estimada (`k1`, `k2`). Es el punto de partida; la alineación clásica y la
distancia conocida corrigen lo demás.

    perfiles = cargar_perfiles()
    perfil = reconocer("Intel(R) RealSense(TM) Depth Camera 435i", perfiles)
    calib = calibracion_de(perfil)  # StereoCalibration, la misma que usa RectifiedSource
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visionx.shared.config import ROOT
from visionx.shared.errors import InvalidInput

CARPETA = ROOT / "configs" / "camaras"
FORMATOS = ("sbs", "realsense", "dos-dispositivos")
# precisa: calibración propia o de fábrica. aproximada: hoja de datos o estimada con buena
# confianza. sin-estimar: valores por defecto, todavía no se estimó la cámara.
NIVELES = ("precisa", "aproximada", "sin-estimar")


@dataclass(frozen=True, slots=True)
class PerfilCamara:
    """Lo que VisionX sabe de una cámara estéreo."""

    id: str
    nombre: str
    formato: str
    resolucion: tuple[int, int]  # (ancho, alto) de un ojo
    campo_horizontal_deg: float
    separacion_m: float
    nivel: str
    fabricante: str = ""
    usb: tuple[str, ...] = ()
    calibracion: str | None = None  # ruta (relativa a la raíz) a una calibración propia
    fuente: str = ""
    centro_rel: tuple[float, float] = (0.0, 0.0)  # (cx/ancho - 0,5, cy/alto - 0,5)
    k1: float = 0.0
    k2: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)  # lo que agregue la pantalla de alta

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.id):
            raise InvalidInput(f"id de cámara inválido {self.id!r}: minúsculas, números y guiones")
        if self.formato not in FORMATOS:
            raise InvalidInput(f"formato {self.formato!r}: opciones {FORMATOS}")
        if self.nivel not in NIVELES:
            raise InvalidInput(f"nivel {self.nivel!r}: opciones {NIVELES}")
        w, h = self.resolucion
        if w <= 0 or h <= 0:
            raise InvalidInput(f"resolución inválida {self.resolucion}")
        if not 20.0 <= self.campo_horizontal_deg <= 170.0:
            raise InvalidInput(f"campo horizontal fuera de rango: {self.campo_horizontal_deg}°")
        if not 0.005 <= self.separacion_m <= 1.0:
            raise InvalidInput(f"separación fuera de rango: {self.separacion_m} m")

    def fx(self, ancho: int | None = None) -> float:
        """Focal en píxeles para un ojo de `ancho` píxeles (por defecto, el del perfil)."""
        w = ancho or self.resolucion[0]
        return (w / 2.0) / math.tan(math.radians(self.campo_horizontal_deg) / 2.0)

    def a_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["resolucion"] = list(self.resolucion)
        d["usb"] = list(self.usb)
        d["centro_rel"] = list(self.centro_rel)
        if not d["extra"]:
            d.pop("extra")
        return d

    @classmethod
    def desde_dict(cls, d: dict[str, Any]) -> PerfilCamara:
        conocidos = {f.name for f in dataclasses.fields(cls)}
        sobra = set(d) - conocidos
        if sobra:
            raise InvalidInput(f"campos desconocidos en el perfil {d.get('id')!r}: {sorted(sobra)}")
        d = dict(d)
        d["resolucion"] = tuple(int(x) for x in d["resolucion"])
        d["usb"] = tuple(str(x) for x in d.get("usb") or ())
        d["centro_rel"] = tuple(float(x) for x in d.get("centro_rel") or (0.0, 0.0))
        for k in ("campo_horizontal_deg", "separacion_m", "k1", "k2"):
            if k in d:
                d[k] = float(d[k])
        return cls(**d)


def cargar_perfil(path: str | Path) -> PerfilCamara:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InvalidInput(f"{path}: no es un perfil de cámara")
    return PerfilCamara.desde_dict(data)


def cargar_perfiles(carpeta: str | Path = CARPETA) -> list[PerfilCamara]:
    """Los perfiles generales y los locales; un local pisa al general con el mismo `id`."""
    carpeta = Path(carpeta)
    por_id: dict[str, PerfilCamara] = {}
    for sub in (carpeta, carpeta / "locales"):
        for path in sorted(sub.glob("*.yaml")):
            perfil = cargar_perfil(path)
            por_id[perfil.id] = perfil
    return list(por_id.values())


def reconocer(nombre_dispositivo: str, perfiles: list[PerfilCamara]) -> PerfilCamara | None:
    """El perfil cuyo patrón `usb` aparece en el nombre del dispositivo (el más largo gana)."""
    nombre = nombre_dispositivo.lower()
    mejor: tuple[int, PerfilCamara] | None = None
    for perfil in perfiles:
        for patron in perfil.usb:
            if patron.lower() in nombre and (mejor is None or len(patron) > mejor[0]):
                mejor = (len(patron), perfil)
    return mejor[1] if mejor else None


def calibracion_nominal(perfil: PerfilCamara, image_size: tuple[int, int] | None = None):
    """StereoCalibration pinhole con la geometría del perfil (ojos paralelos, misma lente)."""
    from visionx.calibration.models import Intrinsics, LensModel
    from visionx.calibration.storage import from_parameters

    w, h = image_size or perfil.resolucion
    f = perfil.fx(w)
    cx = (0.5 + perfil.centro_rel[0]) * w - 0.5
    cy = (0.5 + perfil.centro_rel[1]) * h - 0.5
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    D = np.array([perfil.k1, perfil.k2, 0.0, 0.0, 0.0]).reshape(-1, 1)
    ojo = Intrinsics(K, D)
    T = np.array([-perfil.separacion_m, 0.0, 0.0])
    info = {"perfil": perfil.id, "fuente": perfil.fuente, "nivel": perfil.nivel, "nominal": True}
    return from_parameters(LensModel.PINHOLE5, (w, h), ojo, ojo, np.eye(3), T, info=info)


def calibracion_de(perfil: PerfilCamara):
    """La calibración propia del perfil si la tiene; si no, la nominal."""
    from visionx.calibration.storage import StereoCalibration

    if perfil.calibracion:
        return StereoCalibration.load(ROOT / perfil.calibracion)
    return calibracion_nominal(perfil)


def guardar_local(perfil: PerfilCamara, carpeta: str | Path = CARPETA / "locales") -> Path:
    """Guarda el perfil de una cámara calibrada con la pantalla «Cámara nueva»."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    path = carpeta / f"{perfil.id}.yaml"
    texto = yaml.safe_dump(perfil.a_dict(), allow_unicode=True, sort_keys=False)
    path.write_text(texto, encoding="utf-8")
    return path


def listar_dispositivos(timeout_s: float = 8.0) -> list[str]:
    """Nombres de las cámaras conectadas, según el sistema (lo mejor posible; [] si no se puede).

    El orden no siempre coincide con el índice de OpenCV: sirve para reconocer el modelo, no
    para elegir el `device`.
    """
    sistema = platform.system()
    try:
        if sistema == "Windows":
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction SilentlyContinue"
                " | Select-Object -ExpandProperty FriendlyName",
            ]  # fmt: skip
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s).stdout
            return [x.strip() for x in out.splitlines() if x.strip()]
        if sistema == "Darwin":
            cmd = ["system_profiler", "SPCameraDataType", "-json"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s).stdout
            return [c.get("_name", "") for c in json.loads(out or "{}").get("SPCameraDataType", [])]
        if sistema == "Linux":
            return [
                p.read_text().strip() for p in sorted(Path("/sys/class/video4linux").glob("*/name"))
            ]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return []
