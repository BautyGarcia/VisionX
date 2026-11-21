"""Perfiles de cámaras y flujo «Cámara nueva» (plug and play), sin hardware."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from visionx.app.camara_nueva import AltaCamara, EstimacionCamara, Paso, medir_filas
from visionx.camera.perfiles import (
    PerfilCamara,
    calibracion_de,
    calibracion_nominal,
    cargar_perfiles,
    guardar_local,
    reconocer,
)
from visionx.shared.errors import InvalidInput


def _textura(w: int = 320, h: int = 180, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    chica = rng.integers(0, 255, (h // 6, w // 6, 3), dtype=np.uint8)
    return cv2.resize(chica, (w, h), interpolation=cv2.INTER_NEAREST)


def _par(semilla: int, dy: float = 0.0, dx: float = 6.0):
    izq = _textura(semilla=semilla)
    m = np.float32([[1, 0, -dx], [0, 1, dy]])
    der = cv2.warpAffine(izq, m, (izq.shape[1], izq.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return izq, der


def test_los_perfiles_de_la_carpeta_cargan_y_se_reconocen():
    perfiles = cargar_perfiles()
    ids = {p.id for p in perfiles}
    assert {"elp", "realsense-d435", "realsense-d455", "zed-2", "zed-mini", "oak-d"} <= ids
    nombre = "Intel(R) RealSense(TM) Depth Camera 435i"
    assert reconocer(nombre, perfiles).id == "realsense-d435"
    assert reconocer("ZED-M", perfiles).id == "zed-mini"
    assert reconocer("c922 Pro Stream Webcam", perfiles) is None


def test_calibracion_nominal_respeta_campo_y_separacion():
    p = PerfilCamara(
        id="prueba", nombre="Prueba", formato="sbs", resolucion=(1280, 720),
        campo_horizontal_deg=90.0, separacion_m=0.08, nivel="aproximada",
    )  # fmt: skip
    g = calibracion_nominal(p).geometry()
    assert g.fx == pytest.approx(640.0, rel=1e-3)
    assert g.baseline_m == pytest.approx(0.08, rel=1e-6)
    assert calibracion_de(p).image_size == (1280, 720)


def test_perfil_invalido_y_guardado_local(tmp_path):
    with pytest.raises(InvalidInput):
        PerfilCamara(
            id="Mal Id", nombre="x", formato="sbs", resolucion=(640, 480),
            campo_horizontal_deg=90, separacion_m=0.06, nivel="precisa",
        )  # fmt: skip
    p = PerfilCamara(
        id="mia", nombre="Mía", formato="sbs", resolucion=(640, 480),
        campo_horizontal_deg=75, separacion_m=0.1, nivel="aproximada",
        extra={"escala": "separacion"},
    )  # fmt: skip
    guardar_local(p, tmp_path / "locales")
    assert cargar_perfiles(tmp_path) == [p]


def test_medir_filas_ve_el_desfase_vertical():
    assert medir_filas(*_par(1, dy=0.0)).mediana_px < 0.3
    assert medir_filas(*_par(1, dy=3.0)).mediana_px == pytest.approx(3.0, abs=0.5)


def test_una_camara_conocida_queda_lista_al_conectarla():
    alta = AltaCamara(cargar_perfiles())
    frases = alta.conectar("Intel(R) RealSense(TM) Depth Camera 435i", "realsense", (1280, 720))
    assert alta.estado.paso is Paso.LISTO and alta.estado.conocida
    assert "RealSense D435" in frases[0]


class _Red:
    def estimar(self, pares):
        return EstimacionCamara(70.0, confianza=0.8, origen="red de prueba")


def _alinear_falso(pares, geometria):
    """Corre el ojo derecho 2 px: deshace el desfase de `_par(dy=2)`."""
    h, w = pares[0][0].shape[:2]
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return (mx, my + 2.0), ("pitch",)


def test_flujo_completo_de_una_camara_nueva():
    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), estimador=_Red(), pares_necesarios=4, alinear=_alinear_falso,
        reloj=lambda: next(t),
    )  # fmt: skip
    alta.conectar("Cámara de prueba", "sbs", (320, 180))
    assert alta.estado.paso is Paso.MIRAR
    # Un par repetido no cuenta: hay que mirar alrededor.
    izq, der = _par(10, dy=2.0)
    alta.agregar_par(izq, der)
    alta.agregar_par(izq, der)
    assert alta.estado.pares == 1
    frases = []
    for s in (11, 12, 13):
        frases += alta.agregar_par(*_par(s, dy=2.0))
    e = alta.estado
    assert e.paso is Paso.ESCALA and e.aplica == ("pitch",)
    assert e.filas_antes.mediana_px > 1.5 and e.filas_despues.mediana_px < 0.75
    assert any("70 grados" in f for f in frases)
    assert any("," in f and "píxeles" in f for f in frases)  # decimales con coma
    assert alta.separacion(5.0) and e.paso is Paso.ESCALA  # 5 m: fuera de rango
    alta.separacion(0.07)
    assert e.paso is Paso.LISTO and e.nivel == "precisa"
    assert e.perfil.separacion_m == 0.07 and e.perfil.campo_horizontal_deg == 70.0


def test_sin_red_ni_escala_no_promete_metros():
    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), pares_necesarios=2, alinear=lambda p, g: (None, ()),
        reloj=lambda: next(t),
    )  # fmt: skip
    alta.conectar("Otra", "sbs", (320, 180))
    alta.agregar_par(*_par(20))
    alta.agregar_par(*_par(21))
    frases = alta.saltear_escala()
    assert alta.estado.nivel == "sin-estimar" and "no metros" in frases[0]


def test_la_pantalla_se_dibuja_en_cada_paso():
    from visionx.app.camara_nueva_demo import H, W, render_alta

    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), pares_necesarios=2, alinear=_alinear_falso, reloj=lambda: next(t)
    )
    izq, der = _par(30, dy=2.0)
    assert render_alta(alta, None).shape == (H, W, 3)  # antes de conectar
    alta.conectar("Cámara de prueba", "sbs", (320, 180))
    assert render_alta(alta, izq).shape == (H, W, 3)
    alta.agregar_par(izq, der)
    alta.agregar_par(*_par(31, dy=2.0))
    assert alta.estado.paso is Paso.ESCALA and alta.muestra is not None
    assert render_alta(alta, izq).shape == (H, W, 3)
    alta.saltear_escala()
    assert render_alta(alta, izq).shape == (H, W, 3)


def test_la_escala_se_puede_decir():
    from visionx.app.camara_nueva import interpretar_escala

    assert interpretar_escala("Seis centímetros") == pytest.approx(0.06)
    assert interpretar_escala("Los lentes están a 9,5 centímetros") == pytest.approx(0.095)
    assert interpretar_escala("doce centímetros y medio") == pytest.approx(0.125)
    assert interpretar_escala("sesenta milímetros") == pytest.approx(0.06)
    assert interpretar_escala("No sé") == "saltear"
    assert interpretar_escala("hola") is None
    assert interpretar_escala("150 centímetros") is None  # fuera de rango


def test_responder_la_escala_por_voz():
    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), pares_necesarios=2, alinear=_alinear_falso, reloj=lambda: next(t)
    )
    alta.conectar("Cámara de prueba", "sbs", (320, 180))
    alta.agregar_par(*_par(40, dy=2.0))
    alta.agregar_par(*_par(41, dy=2.0))
    alta.responder("eh, no sé cuánto")  # «no sé» = saltear
    assert alta.estado.paso is Paso.LISTO and alta.estado.escala == "salteada"
    alta2 = AltaCamara(
        cargar_perfiles(), pares_necesarios=2, alinear=_alinear_falso, reloj=lambda: next(t)
    )
    alta2.conectar("Otra", "sbs", (320, 180))
    alta2.agregar_par(*_par(42, dy=2.0))
    alta2.agregar_par(*_par(43, dy=2.0))
    assert alta2.responder("hola")[0].startswith("No entendí")
    alta2.responder("ocho centímetros")
    assert alta2.estado.paso is Paso.LISTO and alta2.estado.perfil.separacion_m == pytest.approx(
        0.08
    )
