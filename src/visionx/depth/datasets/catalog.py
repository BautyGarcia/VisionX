"""Catálogo de datasets públicos: de dónde se bajan, con qué hash, licencia y cómo se leen.

Los hashes se fijaron con la primera descarga (13/11/2026): las fuentes no publican sumas
de verificación. Si una descarga posterior da otro hash, el archivo cambió en el servidor
o llegó roto, y no se usa sin revisar.

Roles (TECNICO T04, decidido 13/11):
- middeval3: test público principal. Entrenamiento de MiddEval3 a media resolución (H),
  15 pares con GT; es el protocolo de FoundationStereo (bad-2 en píxeles visibles).
- eth3d: test público secundario. Imágenes en grises replicadas a 3 canales.
- instereo2k: entrenamiento y validación. Se baja a mano (OneDrive no deja bajar por
  script); `descargar` controla el hash de los zip y los extrae.
- midd2021 y midd2014extra (agregados 17/11): **segunda validación de otro dominio ("dev")**
  para elegir el candidato de Fusion antes del test. Su único grupo se llama `dev`, no `test`:
  se puede mirar y comparar cuantas veces haga falta. Ninguna de sus escenas está en MiddEval3
  (`check_not_in_middeval3` lo exige en código).

Cada dataset se describe con un **índice liviano** (SampleRef: id, escena, archivos) y se
carga muestra por muestra. InStereo2K tiene ~2000 pares: cargarlos juntos ocuparía GB.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from visionx.depth.datasets import instereo2k, middlebury
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.datasets.splits import check_no_leak, split_scenes
from visionx.shared.errors import InvalidInput

SPLIT_SEED = "visionx-T04"  # fija la partición train/val; cambiarla es cambiar el protocolo
VAL_FRACTION = 0.10


@dataclass(frozen=True)
class Download:
    """Archivo comprimido con su hash. `url` vacía = se baja a mano y se deja en
    data/descargas/<dataset>/<name>; `extract_to` es la subcarpeta de data/raw/<dataset>/."""

    url: str
    sha256: str
    bytes: int
    name: str = ""  # por defecto, el final de la url
    extract_to: str = ""

    @property
    def filename(self) -> str:
        return self.name or self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class SampleRef:
    """Una muestra sin cargar: lo necesario para particionar, hashear y leerla después."""

    sample_id: str
    scene_id: str
    group: str  # grupo oficial del dataset: "training", "train", "test"
    files: tuple[str, ...]  # rutas relativas a la raíz extraída
    load: Callable[[], StereoSample] = field(compare=False, repr=False)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    role: str
    license: str
    citation: str
    source: str
    downloads: tuple[Download, ...]
    index: Callable[[Path], list[SampleRef]]  # recibe la carpeta raíz extraída
    gt_rules: tuple[str, ...]
    partition: Callable[[list[SampleRef]], dict[str, list[str]]]  # grupo -> escenas
    notes: tuple[str, ...] = ()
    manual_download: str = ""  # instrucciones si no se puede bajar por script
    # Paso determinista después de extraer (recibe data/raw/<nombre>/), p. ej. reducir a H.
    prepare: Callable[[Path], None] | None = None


def _all_test(refs: list[SampleRef]) -> dict[str, list[str]]:
    """Dataset de evaluación: todas sus escenas son test; no entrena ni elige nada."""
    return {"test": sorted({r.scene_id for r in refs})}


# --- MiddEval3 y ETH3D (formato Middlebury) --------------------------------------------------

MIDDLEBURY_RULES = (
    "disparidad en px, d = x_izq - x_der; infinito del PFM = sin GT (NaN en StereoSample)",
    "máscara: 0 sin GT, 128 ocluido, 255 visible en ambos; métricas noc usan 255",
    "metros: Z = fx * baseline / (d + doffs), baseline de calib.txt en mm -> m",
)

# Variantes de la misma escena (otra exposición/iluminación) comparten escena: no son
# evidencia independiente. ArtL viene de la escena Art de 2005.
_MIDDEVAL3_SCENE = {"MotorcycleE": "Motorcycle", "PianoL": "Piano", "PlaytableP": "Playtable",
                    "ArtL": "Art"}  # fmt: skip


def _middlebury_refs(
    folders: list[Path], dataset: str, group: str, scene_of: Callable[[str], str], root: Path
) -> list[SampleRef]:
    refs = []
    for folder in folders:
        scene = scene_of(folder.name)
        files = tuple((folder / n).relative_to(root).as_posix() for n in middlebury.FILES)
        load = lambda f=folder, s=scene: middlebury.load_folder(f, dataset, f.name, s, root)  # noqa: E731
        refs.append(SampleRef(folder.name, scene, group, files, load))
    return refs


def _middeval3(root: Path) -> list[SampleRef]:
    base = root / "MiddEval3" / "trainingH"
    folders = sorted(p for p in base.iterdir() if p.is_dir())
    return _middlebury_refs(folders, "middeval3", "training",
                            lambda n: _MIDDEVAL3_SCENE.get(n, n), root)  # fmt: skip


def _eth3d(root: Path) -> list[SampleRef]:
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    # delivery_area_1l / delivery_area_1s: mismo lugar, otro punto de vista o lente.
    return _middlebury_refs(folders, "eth3d", "training",
                            lambda n: re.sub(r"_\d+[ls]$", "", n), root)  # fmt: skip


# --- Validación de otro dominio: escenas de Middlebury 2021 y 2014 fuera de MiddEval3 --------

# Escenas del test público MiddEval3 (trainingH): las 10 de 2014, Art (2005) y Teddy (2003).
# Ninguna puede aparecer en un dataset de desarrollo: sería mirar el test por la ventana.
MIDDEVAL3_SCENES = frozenset({
    "Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable",
    "Recycle", "Shelves", "Vintage", "Art", "Teddy",
})  # fmt: skip

SCENES_RULES = (
    "disparidad en px, d = x_izq - x_der; infinito del PFM = sin GT (NaN en StereoSample)",
    "sin máscara publicada: nonoccluded = GT con |d0(x) - d1(x - d0)| <= 1 px (derivada por "
    "nosotros con disp1.pfm, no es la máscara oficial de MiddEval3)",
    "metros: Z = fx * baseline / (d + doffs), baseline de calib.txt en mm -> m",
)


def _scene_name(dataset_name: str) -> str:
    """artroom1 -> artroom, Sword2 -> Sword: los números son tomas de la misma escena/sesión."""
    return re.sub(r"\d+$", "", dataset_name)


def check_not_in_middeval3(scenes: list[str] | set[str]) -> None:
    """Falla si alguna escena (o su nombre sin número/variante) es una escena de MiddEval3."""
    test = {s.lower() for s in MIDDEVAL3_SCENES}
    for scene in scenes:
        names = {scene.lower(), _scene_name(scene).lower(), re.sub(r"[ELP]$", "", scene).lower()}
        if names & test:
            raise InvalidInput(f"la escena {scene!r} es de MiddEval3 (test): no puede ser dev")


def _all_dev(refs: list[SampleRef]) -> dict[str, list[str]]:
    """Dataset de desarrollo: todas sus escenas en `dev` (no `test`), sin escenas de MiddEval3."""
    scenes = sorted({r.scene_id for r in refs})
    check_not_in_middeval3(scenes + [r.sample_id for r in refs])
    return {"dev": scenes}


def _scene_folders(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "calib.txt").is_file())


def _scene_refs(folders: list[Path], dataset: str, root: Path) -> list[SampleRef]:
    refs = []
    for folder in folders:
        scene = _scene_name(folder.name)
        files = tuple((folder / n).relative_to(root).as_posix() for n in middlebury.SCENE_FILES)
        load = lambda f=folder, s=scene: middlebury.load_scene_folder(f, dataset, f.name, s, root)  # noqa: E731
        refs.append(SampleRef(folder.name, scene, "dev", files, load))
    return refs


def _midd2021(root: Path) -> list[SampleRef]:
    return _scene_refs(_scene_folders(root / "data"), "midd2021", root)


MIDD2014_EXTRA = ("Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask",
                  "Shopvac", "Sticks", "Storage", "Sword1", "Sword2", "Umbrella")  # fmt: skip
HALF_FOLDER, FULL_FOLDER = "H", "full"


def _midd2014extra(root: Path) -> list[SampleRef]:
    return _scene_refs(_scene_folders(root / HALF_FOLDER), "midd2014extra", root)


def _halve_midd2014(root: Path) -> None:
    """full/<Escena>-perfect/ -> H/<Escena>/ (media resolución, como MiddEval3-H).

    Se rehace siempre: es determinista y barato (unos segundos por escena)."""
    for folder in _scene_folders(root / FULL_FOLDER):
        name = folder.name.removesuffix("-perfect")
        middlebury.write_half_resolution(folder, root / HALF_FOLDER / name)
        print(f"  {folder.name} -> {HALF_FOLDER}/{name}")


# Hashes de la primera descarga (17/11/2026); las páginas no publican sumas.
_MIDD2014_ZIPS = {
    "Backpack": ("c1af6865490ad6a18040e6c9875fa88594b6833d3cf38f284d9c1c53c77171b2", 108647761),
    "Bicycle1": ("9bff3d11d042e46617b16615519ab9743f7f1d615fbea1182935d1eba76998e0", 100820342),
    "Cable": ("6805d8e330bdc4a44410bfa0e0326f0c40072228a892239872c062b56cb83a41", 94056860),
    "Classroom1": ("138314871346c927346558e7b829f4e859e3413f787546b40472615d850812d9", 97738965),
    "Couch": ("f1d26f8775dbae7c2ce02a6373d95ac56e05c19c88ee8ac9850bcbd3bdf8e0ab", 80541864),
    "Flowers": ("2bbd4ed034e0dfa82d47fa54bda4121731d3abc63117130e2ed32d599d0f1db1", 97000110),
    "Mask": ("100fcecd3f184044ae7094e0701757845ef031c7a5b9b3f38a3a030790583942", 92685882),
    "Shopvac": ("6393753d132b5c3171e085d65b45bee63c03df248d46a15a87260854ac7779d7", 67733626),
    "Sticks": ("411a6781a49dd0d9884f6b19bd6c9ceebf48d49584a307825688f37c712a4e2b", 105987989),
    "Storage": ("f54f369edcd094b82e4ae8bdd05072d11bf5095098036bab5a74b2c03acadd7e", 90867951),
    "Sword1": ("b9e12e3ccf515014851ccfd0851a861c819ea2c7948c10894cf35ce22cec84ce", 105157228),
    "Sword2": ("5aa3acab7fb10d49c1a1351ffcb9191f591fc2176f97fec67611a1564c61fc6f", 90120847),
    "Umbrella": ("b10767d94f596b30e0ab4b8a607c6ea2dadc2b3276e9e2df301ef126e8d30f3a", 96603088),
}
check_not_in_middeval3(set(_MIDD2014_ZIPS))  # al importar: una escena de test no entra ni bajada

MIDDLEBURY_CITATION_2014 = (
    "Scharstein et al., High-resolution stereo datasets with subpixel-accurate ground truth, "
    "GCPR 2014"
)


# --- InStereo2K -----------------------------------------------------------------------------


def _instereo2k(root: Path) -> list[SampleRef]:
    base = instereo2k.find_base(root)
    refs = []
    for group, folder in instereo2k.frame_folders(base):
        sample_id = folder.name  # número de cuadro, único entre train y test
        scene = instereo2k.block_of(instereo2k.frame_number(folder))
        files = tuple((folder / n).relative_to(root).as_posix() for n in instereo2k.FILES)
        load = lambda f=folder, i=sample_id, s=scene: instereo2k.load_pair(f, i, s, root)  # noqa: E731
        refs.append(SampleRef(sample_id, scene, group, files, load))
    return refs


def _instereo2k_partition(refs: list[SampleRef]) -> dict[str, list[str]]:
    """Tramos -> train/val. El test oficial no forma grupo propio: sus cuadros están
    intercalados en los mismos tramos que train, así que siguen al tramo como cualquier otro.

    La partición se elige sobre todos los tramos del dataset (no solo los descargados) para
    que completar la descarga no mueva tramos de grupo; acá se listan los presentes.
    """
    train, val = split_scenes(instereo2k.all_blocks(), VAL_FRACTION, SPLIT_SEED)
    present = {r.scene_id for r in refs}
    groups = {"train": sorted(present & set(train)), "val": sorted(present & set(val))}
    check_no_leak(groups)
    return groups


CATALOG: dict[str, DatasetInfo] = {
    "middeval3": DatasetInfo(
        name="middeval3",
        role="test público principal",
        license="Sin licencia declarada en la página; uso académico citando el paper",
        citation="Scharstein et al., High-resolution stereo datasets with subpixel-accurate "
        "ground truth, GCPR 2014",
        source="https://vision.middlebury.edu/stereo/submit3/",
        downloads=(
            Download(
                "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip",
                "2fc24f494a1e62066d0519cf86b759f344fc7bf0f49785632b582763fb0770a8",
                109900544,
            ),
            Download(
                "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip",
                "c6537f6ee9debcf2a2ec605e69950b5feecb3c5df369f313bb22761acf74025c",
                53097428,
            ),
        ),
        index=_middeval3,
        gt_rules=MIDDLEBURY_RULES,
        partition=_all_test,
        notes=(
            "Solo trainingH tiene GT; testH no se usa.",
            "Comparadores: no usar pesos entrenados con Middlebury (TECNICO T14).",
        ),
    ),
    "eth3d": DatasetInfo(
        name="eth3d",
        role="test público secundario",
        license="Sin licencia declarada en la página; uso académico citando el paper",
        citation="Schöps et al., A multi-view stereo benchmark with high-resolution images "
        "and multi-camera videos, CVPR 2017",
        source="https://www.eth3d.net/datasets",
        downloads=(
            Download(
                "https://www.eth3d.net/data/two_view_training.7z",
                "cb7683a01ba037759f30b5a4f561d302453e20f1f7095cbdc51c8564b218224b",
                14283045,
            ),
            Download(
                "https://www.eth3d.net/data/two_view_training_gt.7z",
                "e440f7ec4444dfd2af33a44495f325a3aed06762432e2419277596ba7dcb9b40",
                14863383,
            ),
        ),
        index=_eth3d,
        gt_rules=MIDDLEBURY_RULES,
        partition=_all_test,
        notes=("Imágenes en grises: DA V2 y Fusion las reciben replicadas a 3 canales.",),
    ),
    "midd2021": DatasetInfo(
        name="midd2021",
        role="desarrollo (segunda validación de otro dominio, grupo dev)",
        license="Sin licencia declarada en la página; uso académico citando el paper",
        citation="Middlebury 2021 Mobile stereo datasets (Scharstein, Guan, Rouke, Zelenka); "
        "citar Scharstein et al., GCPR 2014, como pide la página",
        source="https://vision.middlebury.edu/stereo/data/scenes2021/",
        downloads=(
            Download(
                "https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip",
                "1b60af93fb33811236470c398ed705bd726f8eacb11507ac5d3b66ad8aea42bb",
                423330044,
            ),
        ),
        index=_midd2021,
        gt_rules=SCENES_RULES,
        partition=_all_dev,
        notes=(
            "Resolución nativa 1920x1080 (celular), sin reducir. doffs = 0 en todos los pares.",
            "ladder1/ladder2 son verticales (1080x1920) y su calib.txt trae width/height "
            "intercambiados: el adaptador toma el tamaño de la imagen (focales y centros ya "
            "corresponden a la vertical).",
            "24 pares de 11 escenas: el número final es otra toma de la misma escena "
            "(chess1-3 = chess).",
            "Sin escenas de MiddEval3. Rol dev: elige candidato antes del test, no es test.",
            "Auditoría T04 del 17/11: NO pasa. skiboots2 y skiboots3 desalineadas (mínimo census "
            "en -0,77 y +0,86 px, curvas marcadas) y podium1 no concluyente. No se usa para "
            "elegir nada hasta entender el corrimiento.",
        ),
    ),
    "midd2014extra": DatasetInfo(
        name="midd2014extra",
        role="desarrollo (segunda validación de otro dominio, grupo dev)",
        license="Sin licencia declarada en la página; uso académico citando el paper",
        citation=MIDDLEBURY_CITATION_2014,
        source="https://vision.middlebury.edu/stereo/data/scenes2014/",
        downloads=tuple(
            Download(
                f"https://vision.middlebury.edu/stereo/data/scenes2014/zip/{name}-perfect.zip",
                sha,
                size,
                extract_to=FULL_FOLDER,
            )
            for name, (sha, size) in _MIDD2014_ZIPS.items()
        ),
        index=_midd2014extra,
        gt_rules=SCENES_RULES
        + (
            "media resolución (H) preparada al descargar: imágenes promedio 2x2, GT vecino más "
            "cercano x 0,5 (inválidos siguen inválidos), calib con StereoGeometry.resized",
        ),
        partition=_all_dev,
        notes=(
            "Las 13 escenas de 2014 con GT que no están en MiddEval3, versión -perfect "
            "(rectificación perfecta). MiddEval3 usa la -imperfect: el dev es algo más fácil.",
            "Se evalúa en H (como MiddEval3-H): data/raw/midd2014extra/H/, derivado de full/ "
            "por `descargar`; los hashes de H dependen de la versión de OpenCV.",
            "Escena = nombre sin el número: Sword1 y Sword2 comparten escena (Sword).",
            "Sin escenas de MiddEval3 (chequeo en código). Rol dev, no test.",
        ),
        prepare=_halve_midd2014,
    ),
    "instereo2k": DatasetInfo(
        name="instereo2k",
        role="entrenamiento y validación",
        license="Uso no comercial (investigación, docencia, publicaciones), según el README",
        citation="Bao et al., InStereo2K: a large real dataset for stereo matching in indoor "
        "scenes, Science China Information Sciences 63, 212101, 2020",
        source="https://github.com/YuhuaXu/StereoDataset",
        # Hashes de la descarga manual del 13/11/2026. part1, part5 y part6 no están
        # disponibles (OneDrive: "This file cannot be downloaded"): trabajamos sin ellas.
        downloads=(
            Download(
                url="",
                name="part2.zip",
                extract_to="train",
                bytes=1777751123,
                sha256="9f21fcb1195a1467f1fa485f0310780f332d49ca5b618efab3828983d17e7302",
            ),
            Download(
                url="",
                name="part3.zip",
                extract_to="train",
                bytes=1773314372,
                sha256="64b46042f765504bb4f6f2cd4d01d32df1a7c34b73c2ca337fa1a8e27e65d678",
            ),
            Download(
                url="",
                name="part4.zip",
                extract_to="train",
                bytes=1782540810,
                sha256="2e7e5b9815904ec1a1c230cf9e67238012859415d34e63884540e4c0b6da3a68",
            ),
            Download(
                url="",
                name="test.zip",
                extract_to="",
                bytes=225603067,
                sha256="d68314e2869b850fdd6fec0118c4c15b7abbd62761bd4437d8eb3904dce780da",
            ),
        ),
        index=_instereo2k,
        gt_rules=(
            "disparidad en px = PNG uint16 / 100; 0 = sin GT (NaN en StereoSample)",
            "sin calibración publicada: sin metros",
        ),
        partition=_instereo2k_partition,
        notes=(
            "Escena = tramo de 100 cuadros consecutivos (los cuadros vecinos son el mismo "
            "armado); el dataset no publica escenas.",
            "El test oficial (cuadros 000040, 000080, ...) comparte tramos con train: no es "
            "independiente y no se usa como test.",
            "Subconjunto disponible: part2-4 y test, 1221 de 2050 pares. part1, part5 y part6 "
            "no se pueden bajar de OneDrive (13/11/2026).",
            "Comparadores FoundationStereo/FFS entrenaron con InStereo2K: no es test.",
        ),
        manual_download=(
            "Los links de OneDrive del README (https://github.com/YuhuaXu/StereoDataset) no se "
            "pueden bajar por script. Bajar a mano los zip (partN.zip y test.zip), dejarlos en "
            "data/descargas/instereo2k/ y correr `descargar instereo2k`: controla el hash y "
            "los extrae."
        ),
    ),
}
