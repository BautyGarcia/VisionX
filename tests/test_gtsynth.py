"""GT-sintético-ELP: lectura de vistas, tabla de objetos y verificación de consistencia."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from visionx.gtsynth import PARTICION, REPLICA_A_COCO, Vista, objetos_de, vistas
from visionx.gtsynth.__main__ import main
from visionx.gtsynth.dataset import consistencia, particion_de

W, H, FX, B = 320, 180, 160.0, 0.06


def _vista(tmp_path, escena="room_1", z_fondo=3.0, z_obj=1.5, indice=0):
    """Fondo plano texturado a z_fondo y una caja (instancia 7, «cup») a z_obj."""
    rng = np.random.default_rng(indice)
    textura = cv2.GaussianBlur(rng.integers(0, 255, (H, W * 2)).astype(np.uint8), (3, 3), 0)
    z = np.full((H, W), z_fondo, np.float32)
    inst = np.full((H, W), 3, np.uint16)  # 3 = pared
    z[60:120, 100:160] = z_obj
    inst[60:120, 100:160] = 7
    # Izquierda: cada píxel ve la textura en u; derecha: en u - d (d = fx·B/Z).
    cx = W / 2 - 0.5
    uu = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    vv = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    d = FX * B / z

    # Textura en coordenadas de mundo x = (u - cx)·Z/fx, muestreada en una grilla fija.
    def ver(desde_u):
        return cv2.remap(textura, desde_u + W / 2, vv, cv2.INTER_LINEAR)

    izq = ver(uu)
    der = cv2.remap(izq, uu + d, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    carpeta = tmp_path / "elp" / escena / f"{indice:04d}"
    carpeta.mkdir(parents=True)
    cv2.imwrite(str(carpeta / "izq.png"), cv2.cvtColor(izq, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(carpeta / "der.png"), cv2.cvtColor(der, cv2.COLOR_GRAY2BGR))
    np.savez_compressed(carpeta / "z.npz", z=z)
    cv2.imwrite(str(carpeta / "inst.png"), inst)
    meta = {
        "escena": escena, "variante": "elp", "indice": indice, "semilla": 0,
        "camara": {"ancho": W, "alto": H, "fx": FX, "fy": FX, "cx": cx, "cy": H / 2 - 0.5,
                   "base_m": B, "hfov_deg": 90.0},
        "pose": {}, "apuntado": {"id": 7, "clase": "cup"},
        "instancias": {"3": "wall", "7": "cup"},
    }  # fmt: skip
    (carpeta / "vista.json").write_text(json.dumps(meta), encoding="utf-8")
    return Vista(carpeta)


def test_particion_sin_solapamientos_y_18_escenas():
    todas = [e for es in PARTICION.values() for e in es]
    assert len(todas) == len(set(todas)) == 18
    assert particion_de("room_2") == "test" and particion_de("nada") is None


def test_mapeo_a_coco_usa_clases_del_catalogo():
    from visionx.detection.catalog import COCO_ES

    assert set(REPLICA_A_COCO.values()) <= set(COCO_ES)


def test_objetos_de_mide_el_frente_y_descarta_paredes(tmp_path):
    v = _vista(tmp_path)
    objs = objetos_de(v, min_px=100)
    assert [o.clase for o in objs] == ["cup"]  # la pared se descarta por no ser objeto
    o = objs[0]
    assert o.coco == "cup" and o.apuntado and not o.cortado
    assert o.z_frente_m == pytest.approx(1.5) and o.px == 60 * 60
    assert o.angulo_h_deg == pytest.approx(np.degrees(np.arctan2(129.5 - (W / 2 - 0.5), FX)))


def test_geometria_y_clave(tmp_path):
    v = _vista(tmp_path)
    g = v.geometria
    assert (g.width, g.height, g.fx, g.baseline_m) == (W, H, FX, B)
    assert v.clave == "elp/room_1/0000"


def test_consistencia_prefiere_la_disparidad_exacta(tmp_path):
    e = consistencia(_vista(tmp_path))
    assert e["exacta"] < min(e["sin_mover"], e["mas_10pct"], e["menos_10pct"])


def test_cli_resumen_objetos_y_verificar(tmp_path, capsys):
    _vista(tmp_path, "room_1", indice=0)
    _vista(tmp_path, "room_2", indice=1)
    assert [v.escena for v in vistas(tmp_path, particion="test")] == ["room_2"]
    assert main(["resumen", str(tmp_path)]) == 0
    assert "objetos: 2" in capsys.readouterr().out
    out = tmp_path / "o.csv"
    assert main(["objetos", str(tmp_path), "--csv", str(out), "--min-px", "100"]) == 0
    assert out.read_text(encoding="utf-8").count("\n") == 3
    assert main(["verificar", str(tmp_path)]) == 0


class _Exacto:
    """Estimador de prueba: la disparidad exacta del render y b constante."""

    def __init__(self, vista, b_px=0.5):
        self.vista, self.b_px = vista, b_px
        self.geometry = vista.geometria

    def predict(self, frame):
        from visionx.depth.stereo import build_result
        from visionx.shared.contracts import UncertaintyKind

        g = self.geometry
        d = (g.fx * g.baseline_m / self.vista.z()).astype(np.float32)
        b = np.full_like(d, self.b_px)
        return build_result(
            frame,
            "exacto",
            g,
            d,
            np.ones_like(d, bool),
            1.0,
            0.0,
            uncertainty=b,
            uncertainty_kind=UncertaintyKind.LAPLACE_B,
        )


def test_evaluar_con_disparidad_exacta_da_error_cero_y_cobertura(tmp_path):
    from visionx.depth.spatial import build_locator
    from visionx.gtsynth.evaluar import Medidor, factor_b, medir, resumen

    v = _vista(tmp_path)
    m = Medidor(_Exacto(v), build_locator({}, v.geometria), v.geometria)
    filas = medir(m, [v], "elp", None, min_px=100)
    assert len(filas) == 1
    f = filas[0]
    assert f["politica"] == "speak_meters"
    assert abs(f["error_rel"]) < 0.01
    assert f["z_bajo_k1"] < f["z_gt_m"] < f["z_alto_k1"]
    r = resumen(filas)
    assert r["cobertura_90_k1"] == 1.0 and r["por_rango"]["1–2"]["n"] == 1
    assert factor_b(filas) < 0.1  # error ~0: alcanza con achicar mucho la b
