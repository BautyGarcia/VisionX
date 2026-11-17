"""Paquete de Fusion (T12): lo mínimo para aplicar una red entrenada, con su procedencia.

Un run de entrenamiento (`outputs/fusion/<run>/`) guarda mucho más de lo que la app necesita
(optimizador, scheduler, estados de azar). `empaquetar` copia solo esto a
`outputs/fusion/paquetes/<nombre>/`:

- `modelo.pt`: el state_dict de la red y nada más (se lee con `weights_only=True`: un .pt
  arbitrario puede ejecutar código al abrirse con pickle completo).
- `paquete.yaml`: formato, arquitectura (`ModelConfig`), variante, orden y versión de canales,
  `ChannelParams`, la unidad d/W_ref, la base (B4 o B5) y las secciones de config con las que
  se calculó (SGBM, alineación, monocular, estéreo de B4 y, si la base es B5 o la variante es
  F3, `depth.b5`), datos, métricas de val del checkpoint, commit, semilla, backend, precisión
  y el sha256 de `modelo.pt`.
- `MODEL_CARD.md`: uso, entrenamiento, resultados y límites, llenada con los datos del run.
  Lo que el run no midió dice "no medido": la tarjeta no inventa números.

`load_package` verifica sha256, formato y versión de canales antes de cargar los pesos, y
rechaza con `InvalidInput` si algo no coincide. Paquetes "fusion-canales-1" con F0/F1 (antes
de F3) se siguen cargando: esas variantes no cambiaron; su base es B4.

**La base es parte del modelo.** Fusion aprende un residual sobre su base; si cambia SGBM, la
alineación, el monocular, el estéreo de B4 o (con B5 o F3) `depth.b5`, la base o los canales
cambian y el residual aprendido ya no les corresponde. `base_differences` compara la base en
uso con la del entrenamiento. El rango de búsqueda (`num_disparities`) no entra en la
comparación: no es un parámetro elegido sino una regla por dataset o por geometría
(InStereo2K usó 272 px; la ELP lo calcula de su geometría).
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from visionx.depth.align import AlignParams
from visionx.depth.fusion.cache import mono_identity
from visionx.depth.fusion.channels import (
    VARIANTS,
    ChannelParams,
    check_channels,
    describe_channels,
    uses_b5,
)
from visionx.depth.fusion.model import FusionNet, ModelConfig, count_parameters
from visionx.depth.fusion.train import load_checkpoint
from visionx.depth.hybrid import B5Params
from visionx.depth.mono import MonoConfig, pick_device, sha256_file
from visionx.depth.stereo import SgbmParams
from visionx.shared.config import ROOT
from visionx.shared.errors import InvalidInput

PACKAGE_FORMAT = "fusion-paquete-1"  # cambiarlo invalida los paquetes escritos antes
WEIGHTS_FILE = "modelo.pt"
META_FILE = "paquete.yaml"
CARD_FILE = "MODEL_CARD.md"
UNIT = "d/W_ref"
NOT_MEASURED = "no medido"
BASE_SECTIONS = ("sgbm", "alineacion", "mono", "b4")

# Métricas de val del checkpoint: nombre en el paquete -> columnas de curvas.csv (train.py),
# la nueva primero y la de runs de antes de la base configurable (base siempre B4) después.
_VAL_FIELDS = {
    "epe_fusion_px": ("val_epe_fusion",),
    "epe_base_px": ("val_epe_base", "val_epe_b4"),
    "bad2_fusion_pct": ("val_bad2_fusion",),
    "bad2_base_pct": ("val_bad2_base", "val_bad2_b4"),
    "acierto2_fusion_pct": ("val_acierto2_fusion",),
    "acierto2_base_pct": ("val_acierto2_base", "val_acierto2_b4"),
    "nll_px": ("val_nll_px",),
    "b_mediana_px": ("val_b_mediana_px",),
    "epe_fusion_escena_px": ("val_epe_fusion_escena",),
    "epe_base_escena_px": ("val_epe_base_escena", "val_epe_b4_escena"),
    "epe_comun_fusion_px": ("val_epe_comun_fusion",),
    "epe_comun_b4_px": ("val_epe_comun_b4",),
    "epe_comun_b5_px": ("val_epe_comun_b5",),
    "acierto2_comun_fusion_pct": ("val_acierto2_comun_fusion",),
    "acierto2_comun_b4_pct": ("val_acierto2_comun_b4",),
    "acierto2_comun_b5_pct": ("val_acierto2_comun_b5",),
}


# --- base B4 -------------------------------------------------------------------------------


def mono_model_id(section: dict[str, Any]) -> str:
    """`model_id` que tendría el monocular de una sección `depth.mono` (repo + revisión)."""
    return mono_identity(MonoConfig.from_config(section))["model_id"]


def mono_identity_of(mono: Any) -> dict[str, str]:
    """Identidad de un monocular en uso: completa (repo, revisión, sha de pesos) si trae su
    `MonoConfig` (DepthAnythingV2); si no (un falso de tests), solo su `model_id`."""
    config = getattr(mono, "config", None)
    if isinstance(config, MonoConfig):
        return {**mono_identity(config), "model_id": mono.model_id}
    return {"model_id": mono.model_id}


def _field_diffs(prefix: str, trained: Any, in_use: Any) -> list[str]:
    out = []
    for f in fields(trained):
        old, new = getattr(trained, f.name), getattr(in_use, f.name)
        if old != new:
            out.append(f"{prefix}.{f.name}: entrenado {old!r}, en uso {new!r}")
    return out


def base_differences(
    trained: dict[str, Any],
    stereo_variant: str,
    sgbm: SgbmParams,
    align: AlignParams,
    mono: dict[str, str] | str,
    b5: B5Params | None = None,
    uses_b5_params: bool | None = None,
) -> list[str]:
    """Diferencias entre la base del entrenamiento (`trained`: secciones `depth` registradas)
    y la que se va a usar. Lista vacía = misma base.

    `b5`: los parámetros de B5 en uso. Se comparan si el modelo los usa (`uses_b5_params`;
    None = si `trained` registra `b5`, como los paquetes, que solo la registran cuando hace
    falta). Se comparan ya combinados con la alineación heredada (`B5Params.from_config`).

    Se comparan los parámetros ya interpretados (dataclasses), no los textos del YAML: 4000 y
    4000.0 son lo mismo. `num_disparities` se toma del uso (ver docstring del módulo).
    `mono`: identidad del monocular en uso (`mono_identity`, o solo el `model_id`). Se comparan
    todas las claves que trae: `model_id`, repo y revisión completos y sha256 de los pesos (el
    `model_id` solo mira 8 caracteres de la revisión).
    """
    missing = [k for k in BASE_SECTIONS if k not in trained]
    if missing:
        raise InvalidInput(f"no está registrada la base del entrenamiento: faltan {missing}")
    t_sgbm = SgbmParams.from_config(trained["sgbm"], num_disparities=sgbm.num_disparities)
    t_align = AlignParams.from_config(trained["alineacion"])
    diffs = []
    t_stereo = trained["b4"].get("estereo")
    if t_stereo != stereo_variant:
        diffs.append(f"b4.estereo: entrenado {t_stereo!r}, en uso {stereo_variant!r}")
    diffs += _field_diffs("sgbm", t_sgbm, sgbm)
    diffs += _field_diffs("alineacion", t_align, align)
    t_mono = mono_identity(MonoConfig.from_config(trained["mono"]))
    in_use = {"model_id": mono} if isinstance(mono, str) else mono
    for key, value in in_use.items():
        if key in t_mono and t_mono[key] != value:
            name = "mono" if key == "model_id" else f"mono.{key}"
            diffs.append(f"{name}: entrenado {t_mono[key]!r}, en uso {value!r}")
    if uses_b5_params is None:
        uses_b5_params = "b5" in trained
    if uses_b5_params:
        if "b5" not in trained:
            raise InvalidInput("el modelo usa B5 y no está registrada depth.b5 del entrenamiento")
        t_b5 = B5Params.from_config(trained)
        if b5 is None:
            diffs.append("b5: registrada en el entrenamiento, ausente en la config en uso")
        else:
            diffs += _field_diffs("b5.alineacion", t_b5.align, b5.align)
            diffs += _field_diffs("b5", t_b5.local, b5.local)
    return diffs


def config_base_differences(
    trained: dict[str, Any], depth_cfg: dict[str, Any], uses_b5_params: bool | None = None
) -> list[str]:
    """`base_differences` contra la sección `depth` de una config (sin cargar modelos)."""
    return base_differences(
        trained,
        depth_cfg["b4"]["estereo"],
        SgbmParams.from_config(depth_cfg["sgbm"]),
        AlignParams.from_config(depth_cfg["alineacion"]),
        mono_identity(MonoConfig.from_config(depth_cfg["mono"])),
        B5Params.from_config(depth_cfg) if "b5" in depth_cfg else None,
        uses_b5_params,
    )


# --- empaquetar ----------------------------------------------------------------------------


def _number(text: Any) -> float | str:
    """Celda de curvas.csv -> float; vacía o no finita -> "no medido"."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return NOT_MEASURED
    return value if math.isfinite(value) else NOT_MEASURED


def _val_metrics(run_dir: Path, epoch: int) -> dict[str, Any] | str:
    """Métricas de val de la época del checkpoint, tal como las escribió el entrenamiento."""
    path = run_dir / "curvas.csv"
    if not path.is_file():
        return NOT_MEASURED
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("epoca") == str(epoch)]
    if not rows or _number(rows[-1].get("val_epe_fusion")) == NOT_MEASURED:
        return NOT_MEASURED  # esa época no se validó
    row = rows[-1]
    out = {}
    for name, columns in _VAL_FIELDS.items():
        column = next((c for c in columns if c in row), columns[0])
        out[name] = _number(row.get(column))
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise InvalidInput(f"{path}: se esperaba un mapping YAML")
    return data


def _plain(value: Any) -> Any:
    """Tuplas -> listas y NaN -> "no medido", para un YAML portable."""
    if isinstance(value, dict):
        return {k: _plain(x) for k, x in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return NOT_MEASURED
    return value


def build_meta(run_dir: Path, checkpoint: str, name: str, ckpt: dict[str, Any]) -> dict[str, Any]:
    """Metadatos del paquete a partir del checkpoint y de los archivos del run.

    El sha256 de `modelo.pt` se completa después de escribirlo (ver `empaquetar`).
    """
    record = _read_yaml(run_dir / "config.yaml")
    depth = record.get("depth") or {}
    missing = [k for k in BASE_SECTIONS if k not in depth]
    if missing:
        raise InvalidInput(
            f"{run_dir}/config.yaml no registra la base B4 (faltan {missing}): sin eso el "
            "paquete no puede comprobar que se aplique sobre la misma base"
        )
    meta_path = run_dir / "meta.json"
    run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    train_cfg = record.get("entrenamiento", {})
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])

    def got(key: str) -> Any:
        return run_meta.get(key, NOT_MEASURED)

    variant = ckpt["variant"]
    base_variant = ckpt.get("base", "B4")
    needs_b5 = uses_b5(variant, base_variant)
    if needs_b5 and "b5" not in depth:
        raise InvalidInput(f"{run_dir}/config.yaml no registra depth.b5 y el modelo la usa")
    return _plain({
        "formato": PACKAGE_FORMAT,
        "nombre": name,
        "creado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": {
            "archivo": WEIGHTS_FILE,
            "sha256": None,
            "arquitectura": f"FusionNet: U-Net residual sobre {base_variant} con cabezas delta y "
                            "b (model.py)",
            "config": model_cfg.to_dict(),
            "parametros": count_parameters(FusionNet(model_cfg)),
            "precision": "float32",
        },
        "unidades": {
            "disparidad": UNIT,
            "descripcion": "disparidad en px / W_ref (ancho en px del frame preparado "
                           "completo); multiplicar por W_ref para volver a px",
            "residual_scale": model_cfg.residual_scale,
            "b": f"escala de Laplace en {UNIT}; b_px = b · W_ref",
        },
        "canales": {
            "version": ckpt["channels_version"],
            "variante": variant,
            "nombres": list(ckpt["channel_names"]),
            "params": dict(ckpt["channel_params"]),
        },
        "base": {
            "metodo": base_variant,
            **{k: depth[k] for k in BASE_SECTIONS},
            # depth.b5 solo si el modelo la usa: si no, cambiarla no tiene por qué invalidarlo.
            **({"b5": depth["b5"]} if needs_b5 else {}),
            "monocular_id": mono_model_id(depth["mono"]),
            "rango_busqueda": depth.get("rango_busqueda", NOT_MEASURED),
        },
        "datos": {
            "dataset": "instereo2k",
            "train": "grupo train (aprende)",
            "val": "grupo val (elige el checkpoint por EPE media por muestra)",
            "n_train": got("n_train"),
            "n_val": got("n_val"),
            "limite_train": train_cfg.get("limit_train", NOT_MEASURED),
            "limite_val": train_cfg.get("limit_val", NOT_MEASURED),
            "manifiesto_train_sha256": got("manifiesto_train_sha256"),
            "manifiesto_val_sha256": got("manifiesto_val_sha256"),
            "cache": _portable(record.get("cache", NOT_MEASURED)),
        },
        "checkpoint": {
            "run": run_dir.name,
            "archivo": checkpoint,
            "sha256": sha256_file(run_dir / checkpoint),
            "epoca": int(ckpt["epoch"]),
            "paso": int(ckpt["step"]),
            "mejor": dict(ckpt["best"]),
        },
        "metricas_val": _val_metrics(run_dir, int(ckpt["epoch"])),
        "origen": {
            "commit": got("commit"),
            "seed": int(ckpt["seed"]),
            "torch": got("torch"),
            "backend_entrenamiento": got("dispositivo"),
            "equipo": got("equipo"),
            "estado_run": got("estado"),
            "inicio": got("inicio"),
            "fin": got("fin"),
        },
        "entrenamiento": train_cfg or NOT_MEASURED,
    })  # fmt: skip


def _portable(cache: Any) -> Any:
    """La raíz de la caché, relativa al repo si está adentro (no guardar rutas de una Mac)."""
    if isinstance(cache, dict) and "raiz" in cache:
        root = Path(cache["raiz"])
        if root.is_relative_to(ROOT):
            cache = {**cache, "raiz": root.relative_to(ROOT).as_posix()}
    return cache


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name or ""):
        raise InvalidInput(f"nombre de paquete {name!r}: solo letras, números, '.', '_' y '-'")
    return name


def empaquetar(
    run_dir: str | Path,
    out_dir: str | Path,
    nombre: str,
    checkpoint: str = "mejor.pt",
    overwrite: bool = False,
) -> Path:
    """Copia pesos y metadatos de un run a `<out_dir>/<nombre>/` y devuelve esa carpeta.

    Se escribe en una carpeta temporal al lado y se renombra al final: un corte a mitad de
    camino no deja un paquete a medias con nombre válido.
    """
    run_dir = Path(run_dir)
    ckpt_path = run_dir / checkpoint
    if not ckpt_path.is_file():
        raise InvalidInput(f"{ckpt_path}: no existe el checkpoint a empaquetar")
    dest = Path(out_dir) / _safe_name(nombre)
    if dest.exists() and not overwrite:
        raise InvalidInput(f"{dest} ya existe: elegir otro nombre o reemplazar a propósito")
    # load_checkpoint ya rechaza otro formato, otra versión u otro orden de canales.
    _, ckpt = load_checkpoint(ckpt_path, "cpu")
    meta = build_meta(run_dir, checkpoint, dest.name, ckpt)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{dest.name}-", dir=dest.parent))
    try:
        torch.save(ckpt["model"], tmp / WEIGHTS_FILE)  # state_dict y nada más
        meta["modelo"]["sha256"] = sha256_file(tmp / WEIGHTS_FILE)
        with (tmp / META_FILE).open("w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
        (tmp / CARD_FILE).write_text(model_card(meta), encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


# --- cargar --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionPackage:
    """Red lista para inferir (en `eval()`) más los metadatos de `paquete.yaml`."""

    path: Path
    model: FusionNet
    meta: dict[str, Any]
    device: str

    @property
    def variant(self) -> str:
        return self.meta["canales"]["variante"]

    @property
    def channel_params(self) -> ChannelParams:
        return ChannelParams(**self.meta["canales"]["params"])

    @property
    def weights_sha256(self) -> str:
        return self.meta["modelo"]["sha256"]

    @property
    def model_id(self) -> str:
        """`fusion-f1-<8 hex del sha256 de los pesos>`: otro entrenamiento, otro id."""
        return f"fusion-{self.variant.lower()}-{self.weights_sha256[:8]}"

    @property
    def base(self) -> dict[str, Any]:
        """Secciones `depth` con las que se calculó la base del entrenamiento."""
        return self.meta["base"]

    @property
    def base_variant(self) -> str:
        """B4 o B5. Paquetes de antes de la base configurable: B4."""
        return self.meta["base"].get("metodo", "B4")

    @property
    def uses_b5(self) -> bool:
        """¿La red necesita B5 (base B5 o canales F3) y por lo tanto `depth.b5`?"""
        return uses_b5(self.variant, self.base_variant)


def load_package(path: str | Path, device: str = "auto") -> FusionPackage:
    """Lee un paquete y verifica sha256, formato, unidad y versión/orden de canales.

    `device`: "auto" elige con `mono.pick_device` (MPS, CUDA o CPU).
    """
    path = Path(path)
    meta_path, weights = path / META_FILE, path / WEIGHTS_FILE
    for p in (meta_path, weights):
        if not p.is_file():
            raise InvalidInput(f"{path}: no es un paquete de Fusion (falta {p.name})")
    meta = _read_yaml(meta_path)
    if meta.get("formato") != PACKAGE_FORMAT:
        raise InvalidInput(
            f"{path}: formato {meta.get('formato')!r}, se esperaba {PACKAGE_FORMAT!r}"
        )
    try:
        expected = meta["modelo"]["sha256"]
        channels = meta["canales"]
        unit = meta["unidades"]["disparidad"]
        model_cfg = ModelConfig.from_dict(meta["modelo"]["config"])
    except (KeyError, TypeError) as exc:
        raise InvalidInput(f"{path}/{META_FILE}: falta {exc}") from exc
    got = sha256_file(weights)
    if got != expected:
        raise InvalidInput(
            f"{weights}: sha256 {got[:12]}… no coincide con el del paquete ({str(expected)[:12]}…)"
        )
    variant = channels.get("variante")
    try:
        check_channels(channels.get("version"), variant, channels.get("nombres", ()))
    except InvalidInput as exc:
        raise InvalidInput(f"{path}: {exc}") from exc
    if model_cfg.in_channels != len(VARIANTS[variant]):
        raise InvalidInput(f"{path}: la red espera {model_cfg.in_channels} canales, no {variant}")
    if unit != UNIT:
        raise InvalidInput(f"{path}: unidad de disparidad {unit!r}, se esperaba {UNIT!r}")
    base = meta.get("base", {})
    missing = [k for k in BASE_SECTIONS if k not in base]
    if missing:
        raise InvalidInput(f"{path}: el paquete no registra la base (faltan {missing})")
    if base.get("metodo", "B4") not in ("B4", "B5"):
        raise InvalidInput(f"{path}: base {base.get('metodo')!r} desconocida")
    if uses_b5(variant, base.get("metodo", "B4")):
        if "b5" not in base:
            raise InvalidInput(f"{path}: el modelo usa B5 pero el paquete no registra depth.b5")
        B5Params.from_config(base)  # valida los parámetros registrados
    ChannelParams(**channels.get("params", {}))  # valida los parámetros de canales

    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = FusionNet(model_cfg)
    try:
        model.load_state_dict(state)
    except (RuntimeError, TypeError, AttributeError) as exc:
        raise InvalidInput(f"{weights}: los pesos no corresponden a la arquitectura") from exc
    device = pick_device(device)
    return FusionPackage(path, model.to(device).eval(), meta, device)


# --- model card ----------------------------------------------------------------------------


def _fmt(value: Any, unit: str = "", digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return NOT_MEASURED if value in (None, "", NOT_MEASURED) else str(value)
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"{text} {unit}".rstrip()


def _cell(text: str) -> str:
    """Una barra vertical (valor absoluto, |x|) cortaría la celda de una tabla Markdown."""
    return text.replace("|", "\\|")


def _ranges(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{k}: {x}" + (" px" if isinstance(x, int) else "") for k, x in value.items()
        )
    return _fmt(value)


def _kv(section: Any) -> str:
    if not isinstance(section, dict):
        return f"- {_fmt(section)}"
    return "\n".join(f"- `{k}`: {_fmt(v) if not isinstance(v, dict) else v}"
                     for k, v in section.items())  # fmt: skip


def _results(meta: dict[str, Any]) -> str:
    """Tabla de val del checkpoint, o "no medido"."""
    m, ck = meta.get("metricas_val"), meta["checkpoint"]
    if not isinstance(m, dict):
        return f"Métricas de val del checkpoint: {NOT_MEASURED}."
    base = meta["base"].get("metodo", "B4")
    rows = []
    for name, key in ((f"Base {base}", "base"), ("Fusion", "fusion")):
        cells = (
            _fmt(m[f"epe_{key}_px"]),
            _fmt(m[f"bad2_{key}_pct"], digits=2),
            _fmt(m[f"acierto2_{key}_pct"], digits=2),
            _fmt(m[f"epe_{key}_escena_px"]),
        )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    common = [f"| {name} | {_fmt(m.get(f'epe_comun_{key}_px'))} | "
              f"{_fmt(m.get(f'acierto2_comun_{key}_pct'), digits=2)} |"
              for name, key in (("B4", "b4"), ("B5", "b5"), ("Fusion", "fusion"))]  # fmt: skip
    return "\n".join([
        f"Checkpoint `{ck['archivo']}` (época {ck['epoca']}). Medias por muestra sobre InStereo2K "
        f"val, frames completos, soporte = GT válida ∩ base {base} válida (misma cobertura para "
        "la base y Fusion).",
        "",
        "| Método | EPE (px) | bad-2 (%) | acierto-2 (%) | EPE por escena (px) |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "Los dos controles fijos y Fusion en el soporte común de los tres (GT ∩ B4 ∩ B5 ∩ Fusion):",
        "",
        "| Método | EPE (px) | acierto-2 (%) |",
        "|---|---|---|",
        *common,
        "",
        f"NLL de Laplace (px): {_fmt(m['nll_px'])}; b mediana: {_fmt(m['b_mediana_px'], 'px', 2)}.",
        "",
        "**Val eligió este checkpoint:** estos números no son una medida independiente.",
    ])  # fmt: skip


def model_card(meta: dict[str, Any]) -> str:
    """MODEL_CARD.md a partir de `paquete.yaml`. Todo número sale de los metadatos."""
    model, ch, base = meta["modelo"], meta["canales"], meta["base"]
    data, ck, origin = meta["datos"], meta["checkpoint"], meta["origen"]
    train = meta.get("entrenamiento")
    train = train if isinstance(train, dict) else {}
    variant = ch["variante"]
    stereo = base["b4"].get("estereo")
    base_name = base.get("metodo", "B4")
    model_id = f"fusion-{variant.lower()}-{str(model['sha256'])[:8]}"
    arch = model["config"]
    b5_text = (f"\n\nB5 (`depth.b5`, sobre la alineación de arriba):\n\n{_kv(base['b5'])}"
               if "b5" in base else "")  # fmt: skip
    params = ChannelParams(**ch["params"])
    channel_rows = "\n".join(
        f"| {i} | `{c['nombre']}` | {_cell(c['unidad'])} | {_cell(c['tratamiento'])} |"
        for i, c in enumerate(describe_channels(variant, params), 1)
    )
    t = {k: _fmt(train.get(k)) for k in ("loss", "l1_weight", "crop", "crops_per_image",
                                          "batch_images", "epochs")}  # fmt: skip
    lr, wd = _fmt(train.get("lr"), digits=6), _fmt(train.get("weight_decay"), digits=6)
    o = {k: _fmt(x) for k, x in origin.items()}
    unit, scale = meta["unidades"]["disparidad"], meta["unidades"]["residual_scale"]

    return f"""# Model card — Fusion {variant} (`{meta["nombre"]}`)

Generada por `python -m visionx.depth.fusion empaquetar` el {meta["creado"]} a partir del run
`{ck["run"]}`. Los números salen de `paquete.yaml` (y este, de los archivos del run); lo que el
run no midió dice "{NOT_MEASURED}".

## Qué es

SpatialDepth-Fusion (T12): U-Net residual compacta que parte de la base {base_name} (SGBM
{stereo} + Depth Anything V2 Small alineado; B5 agrega la corrección afín local) y le suma una
corrección `delta`; además estima `b`, la escala de una distribución de Laplace sobre el error
de disparidad. Una red recién creada reproduce la base exactamente (la cabeza delta arranca en
cero).

- `model_id`: `{model_id}`
- Pesos: `{model["archivo"]}`, sha256 `{model["sha256"]}`, {_fmt(model["parametros"])}
  parámetros, {model["precision"]}.
- Canales: `{ch["version"]}`, variante {variant} ({len(ch["nombres"])} canales).
- Arquitectura: niveles {arch.get("widths")}, normalización {arch.get("norm", "group")},
  bloques dilatados {arch.get("bottleneck_dilations") or "ninguno"}.
- Unidad interna: {unit} (disparidad en px dividida por el ancho del frame preparado
  completo); `residual_scale` = {scale}.

## Uso

- Config: `depth.fusion.paquete: <esta carpeta>`; la app arma el estimador con
  `visionx.depth.fusion.estimator.load_estimator(cfg, geometry)`. Necesita el extra `ml`.
- Entrada: par rectificado BGR uint8 (`StereoFrame`). Salida: `DepthResult` con disparidad en
  px a la resolución de cómputo, máscara, metros y `uncertainty` = b en px
  (`UncertaintyKind.LAPLACE_B`, b > 0 dentro de la máscara).
- Validez: donde la base respondió y `d + doffs > 0` después de la corrección (se invalida, no se
  recorta). Sin alineación aceptada corre sobre la base solo estéreo y sale PARTIAL con el
  motivo; si fallan las dos fuentes, UNMEASURABLE.
- Evaluar: `python -m visionx.depth.fusion evaluar <esta carpeta>`.

## Base {base_name} con la que se entrenó

El residual aprendido corresponde a esta base. Si la config en uso difiere (salvo el rango de
búsqueda `num_disparities`, que sale del dataset o de la geometría), el estimador y `evaluar`
se niegan salvo un permiso explícito.

- Estéreo de B4: {stereo}; monocular: `{base["monocular_id"]}`.
- Rango de búsqueda en el entrenamiento: {_ranges(base.get("rango_busqueda"))}. El
  `num_disparities` de la sección SGBM no se usó para entrenar (el rango sale de esta regla
  por dataset; en la app, de la geometría).

SGBM (`depth.sgbm`):

{_kv(base["sgbm"])}

Alineación (`depth.alineacion`):

{_kv(base["alineacion"])}{b5_text}

## Canales de entrada

| # | Canal | Unidad | Tratamiento |
|---|---|---|---|
{channel_rows}

## Entrenamiento

- Datos: InStereo2K; {data["train"]}: {_fmt(data["n_train"])} pares; {data["val"]}:
  {_fmt(data["n_val"])} pares. Límites del run: train {_fmt(data["limite_train"])}, val
  {_fmt(data["limite_val"])} (0 = todos).
- Checkpoint: `{ck["archivo"]}`, época {ck["epoca"]}, paso {ck["paso"]}; sha256 del
  checkpoint `{ck["sha256"]}`.
- Pérdida: {t["loss"]} (L1 extra {t["l1_weight"]}); recorte {t["crop"]};
  {t["crops_per_image"]} recortes por imagen, {t["batch_images"]} imágenes por paso;
  {t["epochs"]} épocas planeadas; AdamW lr {lr}, weight decay {wd}.
- Semilla {o["seed"]}; torch {o["torch"]}; backend {o["backend_entrenamiento"]}; commit
  `{o["commit"]}`; estado del run: {o["estado_run"]} ({o["inicio"]} → {o["fin"]}).

## Resultados

{_results(meta)}

- Test público (MiddEval3, ETH3D): {NOT_MEASURED} en este paquete.
- Cámara ELP (banco físico): {NOT_MEASURED}.
- Latencia en la Mac: {NOT_MEASURED}.
- Calibración de b (¿la fracción de errores dentro de k·b es la esperada?): {NOT_MEASURED}.

## Límites

- Entrenada solo con InStereo2K (interiores, ~1080x860). En otra cámara o resolución la unidad
  d/W_ref hace comparable la escala, pero el dominio cambia: sin medir en la ELP, no hay
  garantía de que mejore la base ahí.
- El residual depende de la base: cambiar SGBM, la alineación, el monocular, el estéreo de B4
  o (con B5 o F3) `depth.b5` exige reentrenar (o un permiso explícito, solo para experimentar).
- Sin aumentos de datos (un espejo rompe la geometría estéreo; el jitter de color desincroniza
  el canal fotométrico).
- La red no decide validez: fuera de la máscara de la base no hay salida. `b` solo se informa dentro
  de la máscara.
"""
