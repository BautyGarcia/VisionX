"""Red que estima la cámara (E2): geometría de la lente, muestras, modelo y métricas.

Sin GPU, sin internet y sin datos reales: una vista falsa con lienzo de margen en tmp_path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visionx.depth.camara_net.datos import (  # noqa: E402
    Aumento,
    Item,
    alto_red,
    item_de,
    lotes_por_alto,
    muestra,
    sortear_camara,
    submuestra_recorridos,
)
from visionx.depth.camara_net.entrenar import cobertura, combinar, errores  # noqa: E402
from visionx.depth.camara_net.lente import (  # noqa: E402
    Camara,
    cabe,
    distorsionar,
    hfov_a_fx,
    mapa_lente,
    sin_distorsion,
    sin_pliegue,
)

LIENZO, CENTRO = (400, 225), (199.5, 112.0)
NOMINAL = (320, 180)


def test_etiqueta_ida_y_vuelta_y_campo_del_pinhole():
    cam = Camara(1280, 720, hfov_a_fx(90, 1280), 650.0, 350.0, -0.1, 0.02)
    otra = Camara.de_etiqueta(cam.etiqueta(), 1280, 720)
    assert otra.fx == pytest.approx(cam.fx) and otra.cx == pytest.approx(650.0)
    assert otra.cy == pytest.approx(350.0) and otra.k1 == pytest.approx(-0.1)
    assert cam.hfov_deg == pytest.approx(90.0)


def test_sin_distorsion_invierte_a_distorsionar():
    x = np.linspace(-0.7, 0.7, 15)
    y = -0.4 * x
    xd, yd = distorsionar(x, y, -0.25, 0.05)
    xu, yu = sin_distorsion(xd, yd, -0.25, 0.05)
    assert np.abs(xu - x).max() < 1e-9 and np.abs(yu - y).max() < 1e-9


def test_pliegue_y_lienzo_descartan_lentes_imposibles():
    # Revisión #85: a 130° con k1 = -0,35 y k2 = 0 la imagen se pliega cerca de r = 1.
    assert not sin_pliegue(-0.35, 0.0, 2.2)
    assert sin_pliegue(-0.35, 0.0, 0.8)
    f130 = hfov_a_fx(130, 1280)
    assert not cabe(Camara(1280, 720, f130, 639.5, 359.5, -0.35, 0.0), (1600, 900), (799.5, 449.5))
    # Un cojín suave pide menos lienzo que el pinhole: siempre entra.
    assert cabe(Camara(1280, 720, f130, 639.5, 359.5, 0.05, 0.0), (1600, 900), (799.5, 449.5))


def test_mapa_sin_lente_es_el_recorte_nominal_escalado():
    cam = Camara(*NOMINAL, 250.0, 159.5, 89.5)
    mx, my = mapa_lente(cam, (160, 90), CENTRO)
    # Píxel (0, 0) de la salida = centro del bloque 2×2 en (0..1) de la nominal → lienzo +40, +22,5.
    assert mx[0, 0] == pytest.approx(0.5 + 40, abs=1e-3)
    assert my[0, 0] == pytest.approx(0.5 + 22.5, abs=1e-3)
    assert np.allclose(np.diff(mx[0]), 2.0, atol=1e-4)


def _vista_falsa(tmp_path, nombre="0000", recorrido=None):
    carpeta = tmp_path / "camara-sorteada" / "room_0" / nombre
    carpeta.mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = (rng.random((LIENZO[1], LIENZO[0], 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(carpeta / "izq.png"), img)
    cv2.imwrite(str(carpeta / "der.png"), img)
    meta = {
        "escena": "room_0", "variante": "camara-sorteada", "indice": int(nombre),
        "camara": {"ancho": LIENZO[0], "alto": LIENZO[1], "fx": 250.0, "fy": 250.0,
                   "cx": CENTRO[0], "cy": CENTRO[1], "base_m": 0.06},
        "nominal": {"ancho": NOMINAL[0], "alto": NOMINAL[1], "x0": 40, "y0": 22,
                    "cx": 159.5, "cy": 90.0, "hfov_deg": 65.2},
        "secuencia": recorrido,
    }  # fmt: skip
    (carpeta / "vista.json").write_text(json.dumps(meta))
    from visionx.gtsynth.dataset import Vista

    return Vista(carpeta)


def test_muestra_sin_aumento_tiene_la_etiqueta_de_la_vista(tmp_path):
    it = item_de(_vista_falsa(tmp_path))
    assert it.nominal == NOMINAL and it.alto_red == alto_red(*NOMINAL) == 360
    m = muestra(it, Aumento.ninguno(), np.random.default_rng(0))
    assert m["ojos"].shape == (2, 3, 360, 640)
    assert m["y"][0] == pytest.approx(math.log(250.0 / 320))
    assert np.allclose(m["y"][1:], [159.5 / 320 - 0.5, 90 / 180 - 0.5, 0, 0], atol=1e-6)
    assert m["rayos"].shape == m["pix"].shape == (35, 2)


def test_sorteo_respeta_el_lienzo_y_corre_el_centro(tmp_path):
    it = item_de(_vista_falsa(tmp_path))
    rng = np.random.default_rng(1)
    camaras = [sortear_camara(it, Aumento(), rng) for _ in range(60)]
    assert all(cabe(c, it.lienzo, it.centro_lienzo) for c in camaras)
    assert any(c.k1 != 0 for c in camaras) and any(abs(c.cx - 159.5) > 1 for c in camaras)
    assert max(abs(c.cx - 159.5) for c in camaras) <= 0.03 * NOMINAL[0] + 1e-6


def test_lotes_no_mezclan_altos_y_recorridos_se_submuestrean():
    def item(alto, grupo):
        return Item("x", (400, 300), (199.5, 149.5), (320, alto), (159.5, alto / 2), 250.0, grupo)

    items = [item(180, f"v{i}") for i in range(7)] + [item(240, f"w{i}") for i in range(5)]
    lotes = lotes_por_alto(items, 4, np.random.default_rng(0))
    assert sorted(i for b in lotes for i in b) == list(range(12))
    assert all(len({items[i].alto_red for i in b}) == 1 for b in lotes)
    rec = [item(180, "r0") for _ in range(30)] + [item(180, "r1") for _ in range(30)]
    assert len(submuestra_recorridos(rec, 6)) == 12


def test_errores_cobertura_y_combinar():
    y = np.array([[math.log(0.5), 0.0, 0.0, -0.1, 0.0]], np.float32)
    mu = y.copy()
    mu[0, 0] += math.log(1.02)  # 2 % de focal
    mu[0, 1] += 0.01  # 1 % del ancho
    e = errores(mu, y, np.array([[1000, 500]], np.float32))
    assert e["focal_pct"][0] == pytest.approx(2.0, rel=1e-3)
    assert e["cx_px"][0] == pytest.approx(10.0, rel=1e-3)
    assert e["reproy_px"][0] > 5
    assert errores(y, y, np.array([[1000, 500]], np.float32))["reproy_px"][0] < 1e-3
    b = np.full((1, 5), 0.01, np.float32)
    assert cobertura(mu, b, y)["cob90_cx"] == 1.0  # 0,01 ≤ 0,01·ln 10
    m, bb = combinar(np.array([[0.0] * 5, [1.0] * 5]), np.array([[1.0] * 5, [1.0] * 5]))
    assert np.allclose(m, 0.5) and np.allclose(bb, 1 / math.sqrt(2))


def test_modelo_y_perdida_en_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from visionx.depth.camara_net.entrenar import perdida, proyectar
    from visionx.depth.camara_net.modelo import RedCamara

    red = RedCamara(preentrenada=False)
    red.fijar_centro(torch.tensor([math.log(0.6), 0, 0, -0.05, 0]))
    ojos = torch.randn(2, 2, 3, 64, 128)
    out = red(ojos, torch.tensor([0.5, 0.5]))
    assert out["mu"].shape == out["log_b"].shape == (2, 5)
    assert torch.allclose(out["mu"][0], red.centro)  # cabeza en cero: arranca en el centro
    cam = Camara(1280, 720, 768.0, 639.5, 359.5, -0.05, 0.0)
    from visionx.depth.camara_net.lente import grilla_rayos

    rayos, pix = grilla_rayos(cam)
    lote = {"y": torch.tensor(cam.etiqueta())[None].repeat(2, 1),
            "rayos": torch.tensor(rayos)[None].repeat(2, 1, 1),
            "pix": torch.tensor(pix / 1280)[None].repeat(2, 1, 1),
            "aspecto": torch.tensor([720 / 1280] * 2)}  # fmt: skip
    # Con la cámara exacta, la reproyección da cero.
    p = proyectar(lote["y"], lote["rayos"], lote["aspecto"])
    assert float((p - lote["pix"]).abs().max()) < 1e-4
    total, partes = perdida(out, lote, 0.5)
    total.backward()
    assert math.isfinite(float(total)) and partes["reproy_pct"] > 0


def test_zoom_es_otra_camara_con_mas_focal_relativa(tmp_path):
    it = item_de(_vista_falsa(tmp_path))
    rng = np.random.default_rng(3)
    aum = Aumento(p_lente=0.0, centro=0.0, fotometrico=False, zoom=(0.6, 0.6))
    cam = sortear_camara(it, aum, rng)
    assert cam.ancho == 192 and cam.alto == 108  # 0,6 × 320 × 180
    assert cam.etiqueta()[0] == pytest.approx(math.log(250.0 / 192))
    assert cam.cx == pytest.approx(159.5 - (320 - 192) / 2)  # sigue en el medio
    m = muestra(it, aum, np.random.default_rng(0))
    assert m["ojos"].shape == (2, 3, 360, 640)  # misma proporción, mismo lote
    # Un campo más angosto que `hfov_min` no se sortea: queda la cámara sin zoom.
    angosto = Aumento(p_lente=0.0, centro=0.0, fotometrico=False, zoom=(0.1, 0.1))
    assert sortear_camara(it, angosto, rng).ancho == NOMINAL[0]
