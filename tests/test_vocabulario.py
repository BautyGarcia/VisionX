"""Vocabulario de VisionX (YOLOE) y selector COCO / YOLOE por cuadro."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from visionx.detection.catalog import (
    COCO_CLASSES,
    COCO_FUERA,
    EXTRA_ES,
    PROMPTS,
    VISIONX_CLASSES,
    ClassCatalog,
    normalize,
    prompts_de,
)
from visionx.detection.switch import VocabularySwitch
from visionx.detection.yolo import DetectorSettings
from visionx.shared.contracts import Mode
from visionx.voice.commands import CommandParser

CAT = ClassCatalog(VISIONX_CLASSES)


def test_tamano_y_sin_choques():
    assert 200 <= len(VISIONX_CLASSES) <= 300
    assert not set(COCO_FUERA) & set(VISIONX_CLASSES)
    assert not CAT.missing_aliases()  # todas se pueden pedir en castellano
    # ClassCatalog rechaza alias ambiguos al construirse; además, ningún nombre inglés de una
    # clase puede ser el alias español de otra («pan» es el pan, no la sartén).
    for cls in VISIONX_CLASSES:
        assert CAT.resolve(cls) == cls


def test_sinonimos_solo_en_clases_nuevas():
    # Promediar sinónimos empeora las COCO (silla 53 % -> 8 % de recall): solo en las nuevas.
    assert not set(PROMPTS) & set(COCO_CLASSES)
    assert set(PROMPTS) <= set(EXTRA_ES)
    assert prompts_de("chair") == ("chair",)
    assert "chalkboard" in prompts_de("whiteboard")


@pytest.mark.parametrize(
    "dicho,clase",
    [
        ("el pizarrón", "whiteboard"), ("la pizarra", "whiteboard"), ("las llaves", "keys"),
        ("mi billetera", "wallet"), ("los anteojos", "eyeglasses"), ("la puerta", "door"),
        ("las escaleras", "stairs"), ("el tacho de basura", "trash can"), ("la pava", "kettle"),
        ("el termo", "thermos"), ("el mate", "mate gourd"), ("el bidet", "bidet"),
        ("la canilla", "faucet"), ("el papel higiénico", "toilet paper"),
        ("el matafuegos", "fire extinguisher"), ("el ascensor", "elevator"),
        ("la mesa de luz", "nightstand"), ("el escritorio", "dining table"),
        ("la banqueta", "chair"), ("el almohadón", "pillow"), ("el bastón", "white cane"),
        ("la salida", "exit sign"), ("los auriculares", "headphones"), ("la taza", "cup"),
        ("el celular", "cell phone"), ("la silla", "chair"),
    ],
)  # fmt: skip
def test_nombres_en_castellano(dicho, clase):
    assert CAT.resolve(dicho) == clase


def test_palabras_comunes_no_disparan_busquedas():
    for palabra in ("aire", "luz", "toma", "sube", "nota", "sobre", "media"):
        assert CAT.resolve(palabra) is None, palabra


def test_el_parser_entiende_los_objetos_nuevos():
    parser = CommandParser(CAT, questions=True)
    for frase, clase in (("Guiame al pizarrón.", "whiteboard"), ("Buscá mis llaves.", "keys"),
                         ("¿A qué distancia está la puerta?", "door")):  # fmt: skip
        i = parser.parse(frase)
        assert i.ok and i.target == clase and i.mode in (Mode.SEARCH, Mode.DISTANCE), (frase, i)


def test_con_un_modelo_coco_los_nuevos_no_se_entienden():
    coco = ClassCatalog(COCO_CLASSES)
    assert coco.resolve("pizarrón") is None and coco.resolve("silla") == "chair"


class _Det:
    def __init__(self, nombre, clases):
        self.model_id, self.class_names, self.vistos = nombre, tuple(clases), 0
        self.last_speed_ms = {"inference": 1.0}

    def predict(self, frame):
        self.vistos += 1
        return SimpleNamespace(model_id=self.model_id)


def test_selector_usa_yoloe_solo_para_lo_que_coco_no_conoce():
    base = _Det("coco", COCO_CLASSES)
    extra = _Det("yoloe", VISIONX_CLASSES)
    s = VocabularySwitch(base, extra)
    assert set(s.class_names) == set(COCO_CLASSES) | set(VISIONX_CLASSES)
    assert s.predict(None).model_id == "coco"  # sin búsqueda
    s.active_target = "cup"
    assert s.predict(None).model_id == "coco"  # algo COCO
    s.active_target = "whiteboard"
    assert s.predict(None).model_id == "yoloe" and s.last_speed_ms == {"inference": 1.0}
    s.active_target = None
    assert s.predict(None).model_id == "coco"
    assert s.observe(None, 0) == [] and (base.vistos, extra.vistos) == (3, 1)


def test_config_por_defecto_trae_yoloe():
    from visionx.shared.config import load_config

    s = DetectorSettings.from_config(load_config())
    assert s.extra_model == "yoloe-26s-seg-visionx"
    cfg = load_config()
    cfg["detection"] = {**cfg["detection"], "extra_model": None}
    assert DetectorSettings.from_config(cfg).extra_model is None


def test_normaliza_tildes_de_los_nombres_nuevos():
    assert normalize("Matafuegos") == "matafuegos" and CAT.resolve("PIZARRON") == "whiteboard"
