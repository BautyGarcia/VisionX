"""Perfiles de cámaras y flujo «Cámara nueva» (plug and play), sin hardware."""

from __future__ import annotations

import itertools
import math

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


def test_el_alta_mide_la_lente_con_el_movimiento_y_pide_girar_si_no_alcanza():
    from visionx.app.autocalibracion import FocalMedida

    llamadas = []

    def mide(cuadros, cx, cy, f0):
        llamadas.append((len(cuadros), f0))
        if len(llamadas) == 1:  # la primera vez no alcanza: tiene que pedir girar más
            return FocalMedida(None, None, 2, "poco movimiento o poca textura")
        return FocalMedida(160.0, 0.01, 40)  # 320 px de ancho y f = 160: 90°

    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), estimador=_Red(), pares_necesarios=2, alinear=_alinear_falso,
        reloj=lambda: next(t), autocalibrar=mide, cuadros_movimiento=4,
    )  # fmt: skip
    frases = alta.conectar("Cámara de prueba", "sbs", (320, 180))
    assert "Girá la cámara" in frases[0]
    for s in range(50, 54):
        frases = alta.agregar_par(*_par(s, dy=2.0))
    # Primero avisa que mide (la medición frena la ventana unos segundos) y mide en el siguiente.
    assert alta.estado.paso is Paso.MEDIR and "midiendo la lente" in frases[0] and not llamadas
    frases = alta.agregar_par(*_par(54, dy=2.0))
    assert alta.estado.paso is Paso.MIRAR and "un poco más" in frases[0]
    assert alta.estado.cuadros_mov == 0
    for s in range(55, 59):
        frases = alta.agregar_par(*_par(s, dy=2.0))
    assert alta.estado.paso is Paso.MEDIR
    frases = alta.agregar_par(*_par(59, dy=2.0))
    e = alta.estado
    assert e.paso is Paso.ESCALA and len(llamadas) == 2
    assert any("Medí la lente" in f and "90 grados" in f for f in frases)
    assert e.perfil.campo_horizontal_deg == pytest.approx(90.0, abs=1.0)  # manda el movimiento
    alta.separacion(0.06)
    assert e.nivel == "precisa" and e.perfil.extra["movimiento"]["ok"]


def test_la_lente_medida_queda_en_el_perfil():
    from visionx.app.autocalibracion import FocalMedida

    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), estimador=_Red(), pares_necesarios=2, alinear=_alinear_falso,
        reloj=lambda: next(t), cuadros_movimiento=3,
        autocalibrar=lambda c, cx, cy, f0: FocalMedida(160.0, 0.01, 40, k1=-0.18),
    )  # fmt: skip
    alta.conectar("Cámara de prueba", "sbs", (320, 180))
    for s in range(80, 84):
        alta.agregar_par(*_par(s, dy=2.0))
    assert alta.estado.paso is Paso.ESCALA
    assert alta.estado.perfil.k1 == pytest.approx(-0.18) and alta.estado.perfil.k2 == 0.0


def test_un_objeto_a_distancia_conocida_da_la_separacion():
    from visionx.app.camara_nueva import disparidad_central

    izq, der = _par(60, dx=12.0)
    assert disparidad_central(izq, der) == pytest.approx(12.0, abs=0.6)
    t = itertools.count(0.0, 1.0)
    alta = AltaCamara(
        cargar_perfiles(), estimador=_Red(), pares_necesarios=2,
        alinear=lambda p, g: (None, ()), reloj=lambda: next(t),
    )  # fmt: skip
    alta.conectar("Cámara de prueba", "sbs", (320, 180))
    alta.agregar_par(*_par(61, dx=12.0))
    alta.agregar_par(*_par(62, dx=12.0))
    assert alta.estado.paso is Paso.ESCALA
    alta.agregar_par(*_par(63, dx=12.0))  # lo que ve ahora, apuntando al objeto
    frases = alta.responder("está a dos metros")
    fx = 160 / math.tan(math.radians(35))  # 70° en 320 px
    b = alta.estado.perfil.separacion_m
    assert alta.estado.escala == "distancia" and b == pytest.approx(2.0 * 12.0 / fx, rel=0.08)
    assert any("centímetros" in f for f in frases)


def test_alineacion_en_vivo_alinea_al_arrancar_y_se_realinea_si_se_golpea():
    from visionx.app.alineacion_viva import FuenteAlineada
    from visionx.shared.contracts import StereoFrame

    class Fuente:
        geometry = None

        def __init__(self):
            self.k, self.dy = 0, 2.0

        def read(self):
            self.k += 1
            izq, der = _par(100 + self.k, dy=self.dy)
            return StereoFrame(
                "s",
                self.k,
                izq,
                der,
                received_ns=self.k,
                rectified=True,
                calibration_id="c",
                geometry_id="g",
            )

        def close(self):
            pass

    def alinear(pares, geometria):  # «clásico»: mide el desfase y lo deshace
        dy = float(np.median(medir_filas(*pares[0]).dy))
        h, w = pares[0][0].shape[:2]
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return (mx, my + dy), ("pitch",)

    t = itertools.count(0.0, 1.0)
    f = Fuente()
    viva = FuenteAlineada(f, pares=3, cada_s=2.0, seguidas=2, en_hilo=False,
                          reloj=lambda: next(t), alinear=alinear)  # fmt: skip
    for _ in range(3):
        viva.read()
    assert viva.estado == "alineada" and viva.ultimo_desfase_px < 0.75
    for _ in range(4):  # en uso: mide y no pasa nada
        viva.read()
    assert viva.realineaciones == 0
    f.dy = 5.0  # golpe: el ojo derecho se corrió 3 px más
    for _ in range(6):
        viva.read()
    assert viva.realineaciones == 1 and viva.estado in ("realineando", "alineada")
    for _ in range(6):
        viva.read()
    assert viva.estado == "alineada" and viva.ultimo_desfase_px < 0.75
    eventos = [h["evento"] for h in viva.historial]
    assert eventos.count("alineada") == 2 and "desalineada" in eventos


def test_la_separacion_se_tipea_o_se_elige_el_modelo():
    def lista():
        t = itertools.count(0.0, 1.0)
        alta = AltaCamara(
            cargar_perfiles(), estimador=_Red(), pares_necesarios=2, alinear=_alinear_falso,
            reloj=lambda: next(t),
        )  # fmt: skip
        alta.conectar("Cámara de prueba", "sbs", (320, 180))
        alta.agregar_par(*_par(70, dy=2.0))
        alta.agregar_par(*_par(71, dy=2.0))
        assert alta.estado.paso is Paso.ESCALA
        return alta

    a = lista()  # tipear «6,3» + Enter
    for c in "6.3":
        a.tipear(c)
    a.tipear(".")  # una sola coma
    assert a.estado.entrada == "6,3"
    a.borrar()
    a.tipear("5")
    a.confirmar()
    assert a.estado.paso is Paso.LISTO and a.estado.perfil.separacion_m == pytest.approx(0.065)

    b = lista()  # Tab hasta un modelo + Enter
    modelos = b.modelos()
    assert len(modelos) >= 2 and all(m.separacion_m > 0 for m in modelos)
    b.siguiente_modelo()
    b.siguiente_modelo()
    elegido = b.estado.modelo
    frases = b.confirmar()
    assert b.estado.paso is Paso.LISTO
    assert b.estado.perfil.separacion_m == pytest.approx(elegido.separacion_m)
    assert b.estado.perfil.extra["modelo"] == elegido.id and elegido.nombre in frases[0]

    c = lista()  # vacío: explica qué tipear y sigue en la escala
    assert "Tab" in c.confirmar()[0] and c.estado.paso is Paso.ESCALA
