"""Tests de datos y GT (T04) sin descargar nada: escenas sintéticas de disparidad conocida."""

import numpy as np
import pytest

from visionx.depth.datasets.audit import (
    SHIFTS_PX,
    alignment_curve,
    audit_sample,
    contact_sheet,
    dataset_verdict,
    subpixel_minimum,
    verdict,
)
from visionx.depth.datasets.manifest import file_entry, sha256_file, verify_files
from visionx.depth.datasets.pfm import read_pfm, write_pfm
from visionx.depth.datasets.sample import StereoSample
from visionx.depth.datasets.splits import check_no_leak, split_scenes
from visionx.shared.errors import InvalidInput
from visionx.shared.geometry import StereoGeometry

D_TRUE = 6  # píxeles


def synthetic_sample(d: float = D_TRUE, **kwargs) -> StereoSample:
    """Textura aleatoria; el ojo derecho ve cada punto D_TRUE px más a la izquierda."""
    rng = np.random.default_rng(0)
    h, w = 160, 240  # suficientes píxeles para que el ruido no decida la auditoría
    left = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    left = np.repeat(np.repeat(left[::2, ::2], 2, 0), 2, 1)  # manchas de 2 px: interpolable
    right = np.roll(left, -D_TRUE, axis=1)  # right[x - d] = left[x]
    valid = np.ones((h, w), bool)
    valid[:, :D_TRUE] = False  # esas columnas no tienen correspondencia en el ojo derecho
    disparity = np.where(valid, np.float32(d), np.nan).astype(np.float32)
    args = dict(dataset="sintetico", sample_id="s0", scene_id="escena0", left=left,
                right=right, disparity=disparity, valid=valid)  # fmt: skip
    args.update(kwargs)
    return StereoSample(**args)


# --- PFM -------------------------------------------------------------------------------


def test_pfm_roundtrip_keeps_row_order_and_inf(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    values[0, 0] = np.inf  # Middlebury marca así los píxeles sin GT
    write_pfm(tmp_path / "d.pfm", values)
    back = read_pfm(tmp_path / "d.pfm")
    assert back.dtype == np.float32
    assert np.array_equal(back, values)  # fila 0 sigue arriba


def test_pfm_big_endian_and_truncated(tmp_path):
    values = np.array([[1.5, 2.5]], np.float32)
    path = tmp_path / "be.pfm"
    path.write_bytes(b"Pf\n2 1\n1.0\n" + values.astype(">f4").tobytes())
    assert np.array_equal(read_pfm(path), values)
    path.write_bytes(b"Pf\n2 2\n-1.0\n" + values.astype("<f4").tobytes())
    with pytest.raises(InvalidInput, match="truncado"):
        read_pfm(path)
    path.write_bytes(b"P6\n2 1\n255\n")
    with pytest.raises(InvalidInput, match="no es PFM"):
        read_pfm(path)


# --- Muestra -----------------------------------------------------------------------------


def test_sample_rejects_zero_as_missing_gt():
    s = synthetic_sample()
    bad = s.disparity.copy()
    bad[~s.valid] = 0.0  # un 0 fuera de valid parecería una medición
    with pytest.raises(InvalidInput, match="NaN"):
        synthetic_sample(disparity=bad)


def test_sample_checks_masks_and_geometry():
    s = synthetic_sample()
    with pytest.raises(InvalidInput, match="nonoccluded"):
        synthetic_sample(nonoccluded=~s.valid)
    with pytest.raises(InvalidInput, match="geometría"):
        synthetic_sample(geometry=StereoGeometry(100, 60, 500.0, 500.0, 50.0, 30.0, 0.1))
    with pytest.raises(InvalidInput, match="sin calibración"):
        s.depth()
    g = StereoGeometry(240, 160, 600.0, 600.0, 120.0, 80.0, 0.06)
    z, ok = synthetic_sample(geometry=g).depth()
    assert np.allclose(z[ok], 600.0 * 0.06 / D_TRUE)


# --- Auditoría fotométrica -----------------------------------------------------------------


def test_alignment_minimum_at_zero_for_correct_gt(tmp_path):
    s = synthetic_sample()
    errors, n = alignment_curve(s)
    assert n > 0
    # Sin corrimiento el census coincide salvo en el borde que np.roll mezcla.
    assert errors[len(errors) // 2] < 0.01 < min(errors[0], errors[-1])
    report = audit_sample(s)
    assert report["fotometrico"]["gt_alineada"] is True
    assert report["fotometrico"]["minimo_en_px"] == 0.0
    assert report["disparidad_px"]["p50"] == D_TRUE
    assert contact_sheet(s, tmp_path / "hoja.png").exists()


def test_alignment_ignores_exposure_change():
    """Ojo derecho más oscuro (como MotorcycleE): census no cambia, la GT sigue alineada."""
    s = synthetic_sample()
    darker = (s.right.astype(np.float32) * 0.6 + 10).astype(np.uint8)
    assert audit_sample(synthetic_sample(right=darker))["fotometrico"]["gt_alineada"] is True


@pytest.mark.parametrize("wrong", [D_TRUE * 100.0, -D_TRUE, D_TRUE / 2])
def test_audit_flags_badly_read_gt(wrong):
    """Escala x100 olvidada, signo invertido o escala mal: la prueba lo detecta."""
    report = audit_sample(synthetic_sample(d=wrong))
    assert report["fotometrico"]["gt_alineada"] is False
    # Un dataset entero leído así no pasa, aunque alguna muestra quedara «no concluyente».
    assert dataset_verdict([report] * 20)["ok"] is False


def test_audit_verdicts_and_subpixel_minimum():
    flat = [0.5] * len(SHIFTS_PX)  # pared lisa: corrimientos indistinguibles
    assert verdict(flat, 100_000) == "no concluyente"
    v = [abs(s - 0.25) + 0.1 for s in SHIFTS_PX]  # mínimo real en +0,25 px
    assert verdict(v, 100_000) == "alineada"  # el discreto cae en 0 o +0,5: un paso
    assert subpixel_minimum([(s - 0.25) ** 2 for s in SHIFTS_PX]) == pytest.approx(0.25)
    far = [abs(s - 2.0) + 0.1 for s in SHIFTS_PX]
    assert verdict(far, 100_000) == "desalineada"
    assert verdict(v, 5_000) == "no concluyente"  # pocos píxeles


def _report(veredicto, sub=0.0, muestra="m"):
    return {"muestra": muestra, "fotometrico": {"veredicto": veredicto, "minimo_subpx": sub}}


def test_dataset_verdict_rules():
    good = [_report("alineada", 0.1)] * 99
    assert dataset_verdict([*good, _report("no concluyente")])["ok"] is True  # 1 %
    assert dataset_verdict([*good[:90], *[_report("no concluyente")] * 10])["ok"] is False
    assert dataset_verdict([*good, _report("desalineada", 2.0, "x")])["desalineadas"] == ["x"]
    shifted = [_report("alineada", 0.4)] * 50  # todo corrido ~0,4 px: no es ruido
    assert dataset_verdict(shifted)["ok"] is False


# --- Particiones y manifiesto ----------------------------------------------------------------


def test_split_is_deterministic_and_stable():
    scenes = [f"s{i:02d}" for i in range(20)]
    rest, held = split_scenes(scenes * 3, 0.2, "visionx")  # pares repetidos por escena
    assert len(held) == 4 and len(rest) == 16
    assert split_scenes(reversed(scenes), 0.2, "visionx") == (rest, held)
    check_no_leak({"train": rest, "val": held})
    with pytest.raises(InvalidInput, match="fuga"):
        check_no_leak({"train": rest, "val": held + [rest[0]]})
    with pytest.raises(InvalidInput):
        split_scenes(["a"], 0.5, "x")


def test_manifest_detects_changed_files(tmp_path):
    f = tmp_path / "a" / "x.bin"
    f.parent.mkdir()
    f.write_bytes(b"hola")
    entry = file_entry(f, tmp_path)
    assert entry["ruta"] == "a/x.bin"
    assert entry["sha256"] == sha256_file(f)
    assert verify_files([entry], tmp_path) == []
    f.write_bytes(b"chau")
    assert verify_files([entry], tmp_path) == ["hash distinto: a/x.bin"]
    f.unlink()
    assert verify_files([entry], tmp_path) == ["falta a/x.bin"]


# --- Formato Middlebury / ETH3D --------------------------------------------------------------


def _write_middlebury(folder, sample, doffs=10.0, mask_override=None):
    """Escribe una muestra sintética como la publica Middlebury (baseline en mm)."""
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    w, h = sample.size
    cv2.imwrite(str(folder / "im0.png"), sample.left)
    cv2.imwrite(str(folder / "im1.png"), sample.right)
    # El PFM guarda la disparidad de imagen (x_izq - x_der); doffs solo entra en Z. Antes se
    # escribía d - doffs y la auditoría lo daba por alineado de casualidad: la textura en
    # bloques de 2 px tiene mínimos census cada 2 px (10 px de error = 5 bloques).
    write_pfm(folder / "disp0GT.pfm", np.where(sample.valid, sample.disparity, np.inf))
    mask = np.where(sample.valid, 255, 0).astype(np.uint8)
    mask[:, -10:][sample.valid[:, -10:]] = 128  # algunas ocluidas
    cv2.imwrite(str(folder / "mask0nocc.png"), mask if mask_override is None else mask_override)
    (folder / "calib.txt").write_text(
        f"cam0=[600 0 120; 0 600 80; 0 0 1]\ncam1=[600 0 {120 + doffs}; 0 600 80; 0 0 1]\n"
        f"doffs={doffs}\nbaseline=60.0\nwidth={w}\nheight={h}\nndisp=64\n",
        encoding="utf-8",
    )


def test_middlebury_folder_reads_units_masks_and_doffs(tmp_path):
    from visionx.depth.datasets.middlebury import load_folder

    src = synthetic_sample()
    _write_middlebury(tmp_path / "Escena", src)
    s = load_folder(tmp_path / "Escena", "mb", "Escena", "Escena", tmp_path)
    assert s.geometry.baseline_m == pytest.approx(0.060)  # mm -> m
    assert s.geometry.doffs == pytest.approx(10.0)
    assert np.array_equal(s.valid, src.valid)
    assert s.nonoccluded.sum() < s.valid.sum()  # las de 128 quedan como ocluidas
    assert np.allclose(s.disparity[s.valid], D_TRUE)
    z, ok = s.depth()  # Z = 600 * 0.06 / (d + doffs)
    assert np.allclose(z[ok], 600 * 0.06 / (D_TRUE + 10.0))
    assert s.files[0] == "Escena/im0.png"
    assert audit_sample(s)["fotometrico"]["gt_alineada"] is True


def test_middlebury_folder_rejects_inconsistent_files(tmp_path):
    from visionx.depth.datasets.middlebury import load_folder

    src = synthetic_sample()
    bad_mask = np.full(src.valid.shape, 255, np.uint8)  # dice "hay GT" donde el PFM tiene inf
    _write_middlebury(tmp_path / "A", src, mask_override=bad_mask)
    with pytest.raises(InvalidInput, match="no coinciden"):
        load_folder(tmp_path / "A", "mb", "A", "A", tmp_path)
    _write_middlebury(tmp_path / "B", src)
    calib = (tmp_path / "B" / "calib.txt").read_text().replace("doffs=10.0", "doffs=3.0")
    (tmp_path / "B" / "calib.txt").write_text(calib)
    with pytest.raises(InvalidInput, match="doffs"):
        load_folder(tmp_path / "B", "mb", "B", "B", tmp_path)


def test_catalog_groups_variants_into_scenes(tmp_path):
    from visionx.depth.datasets.catalog import CATALOG

    src = synthetic_sample()
    for name in ("Motorcycle", "MotorcycleE", "Teddy"):
        _write_middlebury(tmp_path / "mb" / "MiddEval3" / "trainingH" / name, src)
    scenes = {r.sample_id: r.scene_id for r in CATALOG["middeval3"].index(tmp_path / "mb")}
    assert scenes == {"Motorcycle": "Motorcycle", "MotorcycleE": "Motorcycle", "Teddy": "Teddy"}
    for name in ("electro_1l", "electro_1s", "forest_2s"):
        _write_middlebury(tmp_path / "eth" / name, src)
    scenes = {r.sample_id: r.scene_id for r in CATALOG["eth3d"].index(tmp_path / "eth")}
    assert scenes == {"electro_1l": "electro", "electro_1s": "electro", "forest_2s": "forest"}


# --- InStereo2K y CLI ------------------------------------------------------------------------


def _write_instereo2k(folder, sample, scale=100.0):
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "left.png"), sample.left)
    cv2.imwrite(str(folder / "right.png"), sample.right)
    raw = np.where(sample.valid, np.rint(np.nan_to_num(sample.disparity) * scale), 0)
    cv2.imwrite(str(folder / "left_disp.png"), raw.astype(np.uint16))
    cv2.imwrite(str(folder / "right_disp.png"), raw.astype(np.uint16))


def test_instereo2k_divides_by_100_and_partitions_by_block(tmp_path):
    from visionx.depth.datasets.catalog import CATALOG

    src = synthetic_sample()
    root = tmp_path / "raw"
    # Estructura real: train/part<N>/<cuadro>/ y test/<cuadro>/, cuadros numerados en común.
    for frame in (1, 2, 101, 102, 401):
        _write_instereo2k(root / "train" / f"part{(frame - 1) // 400 + 1}" / f"{frame:06d}", src)
    _write_instereo2k(root / "test" / "000040", src)
    info = CATALOG["instereo2k"]
    refs = info.index(root)
    assert [r.sample_id for r in refs] == ["000001", "000002", "000040", "000101", "000102",
                                           "000401"]  # fmt: skip
    scenes = {r.sample_id: r.scene_id for r in refs}
    assert scenes["000001"] == scenes["000040"] == "tramo-00"  # el test oficial sigue al tramo
    assert scenes["000401"] == "tramo-04"
    sample = refs[0].load()
    assert sample.geometry is None  # sin calibración publicada
    assert np.allclose(sample.disparity[sample.valid], D_TRUE)  # 600 / 100
    assert np.array_equal(sample.valid, src.valid)  # 0 = sin GT
    assert audit_sample(sample)["fotometrico"]["gt_alineada"] is True
    groups = info.partition(refs)
    # La semilla fija aparta tramo-00 y tramo-04 de los 21 tramos del dataset completo.
    assert groups == {"train": ["tramo-01"], "val": ["tramo-00", "tramo-04"]}
    assert info.partition(refs[:4]) == {"train": ["tramo-01"], "val": ["tramo-00"]}  # estable


def test_instereo2k_rejects_repeated_frames(tmp_path):
    from visionx.depth.datasets.catalog import CATALOG

    for part in ("part1", "part2"):
        _write_instereo2k(tmp_path / "train" / part / "000007", synthetic_sample())
    (tmp_path / "test").mkdir()
    with pytest.raises(InvalidInput, match="repetidos"):
        CATALOG["instereo2k"].index(tmp_path)


def test_cli_extracts_manual_archives_after_checking_hash(tmp_path, monkeypatch):
    import dataclasses
    import zipfile

    from visionx.depth.datasets import __main__ as cli
    from visionx.depth.datasets.catalog import CATALOG, Download

    data = tmp_path / "data"
    zpath = data / "descargas" / "instereo2k" / "part9.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("part9/003001/left.png", b"x")
    item = Download("", sha256_file(zpath), zpath.stat().st_size, "part9.zip", "train")
    info = dataclasses.replace(CATALOG["instereo2k"], downloads=(item,))
    monkeypatch.setitem(cli.CATALOG, "instereo2k", info)
    assert cli.main(["descargar", "instereo2k", "--datos", str(data)]) == 0
    assert (data / "raw" / "instereo2k" / "train" / "part9" / "003001" / "left.png").exists()
    zpath.write_bytes(b"otro contenido")  # el zip cambió: no se extrae
    assert cli.main(["descargar", "instereo2k", "--datos", str(data)]) == 1


def test_instereo2k_wrong_scale_is_caught_by_audit(tmp_path):
    """Si la escala real fuera /1024 (como asume torchvision), leer /100 se detecta."""
    from visionx.depth.datasets.instereo2k import load_pair

    _write_instereo2k(tmp_path / "train" / "a", synthetic_sample(), scale=1024.0)
    sample = load_pair(tmp_path / "train" / "a", "a", "a", tmp_path)
    assert audit_sample(sample)["fotometrico"]["gt_alineada"] is False


def test_cli_audit_manifest_verify(tmp_path, capsys):
    from visionx.depth.datasets.__main__ import main

    src = synthetic_sample()
    for name in ("electro_1l", "forest_2s"):
        _write_middlebury(tmp_path / "data" / "raw" / "eth3d" / name, src)
    data, out, man = str(tmp_path / "data"), str(tmp_path / "out"), str(tmp_path / "man")
    assert main(["auditar", "eth3d", "--datos", data, "--salida", out, "--hojas", "1"]) == 0
    assert len(list((tmp_path / "out" / "eth3d").glob("*.png"))) == 1
    out0 = str(tmp_path / "out0")
    assert main(["auditar", "eth3d", "--datos", data, "--salida", out0, "--hojas", "0"]) == 0
    assert not list((tmp_path / "out0" / "eth3d").glob("*.png"))  # 0 hojas = ninguna
    assert main(["manifiesto", "eth3d", "--datos", data, "--manifiestos", man]) == 0
    assert main(["verificar", "eth3d", "--datos", data, "--manifiestos", man]) == 0
    (tmp_path / "data" / "raw" / "eth3d" / "forest_2s" / "im1.png").write_bytes(b"roto")
    assert main(["verificar", "eth3d", "--datos", data, "--manifiestos", man]) == 1
    assert main(["descargar", "instereo2k", "--datos", data]) == 1
    assert "a mano" in capsys.readouterr().err


# --- Validación dev de otro dominio: escenas de Middlebury 2014/2021 --------------------------


def _write_scene(folder, sample, doffs=10.0, occluded_cols=8, swap_size=False, ndisp=64):
    """Par en formato de las páginas de escenas: disp0/disp1, sin máscara publicada.

    disp1 coincide con disp0 salvo en `occluded_cols` columnas, donde el ojo derecho "ve otra
    cosa" (disparidad distinta): esas columnas tienen que salir como ocluidas.
    """
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    w, h = sample.size
    cv2.imwrite(str(folder / "im0.png"), sample.left)
    cv2.imwrite(str(folder / "im1.png"), sample.right)
    # Disparidad de imagen (x_izq - x_der), como Middlebury; doffs solo entra en Z.
    write_pfm(folder / "disp0.pfm", np.where(sample.valid, sample.disparity, np.inf))
    d1 = np.full((h, w), np.float32(D_TRUE))  # convención positiva, ojo derecho
    # x - d de las columnas 100..100+occ cae en 100-D..: ahí el derecho ve otra disparidad.
    d1[:, 100 - D_TRUE : 100 - D_TRUE + occluded_cols] += 5.0
    write_pfm(folder / "disp1.pfm", d1)
    size = (h, w) if swap_size else (w, h)
    (folder / "calib.txt").write_text(
        f"cam0=[600 0 120; 0 600 80; 0 0 1]\ncam1=[600 0 {120 + doffs}; 0 600 80; 0 0 1]\n"
        f"doffs={doffs}\nbaseline=60.0\nwidth={size[0]}\nheight={size[1]}\nndisp={ndisp}\n"
        "isint=0\nvmin=1\nvmax=7\n",
        encoding="utf-8",
    )


def test_scene_folder_reads_both_gt_and_derives_nonoccluded(tmp_path):
    from visionx.depth.datasets.middlebury import load_scene_folder

    src = synthetic_sample()
    _write_scene(tmp_path / "chess1", src)
    s = load_scene_folder(tmp_path / "chess1", "d", "chess1", "chess", tmp_path)
    assert s.geometry.doffs == pytest.approx(10.0)
    assert np.array_equal(s.valid, src.valid)  # inf = sin GT
    assert np.allclose(s.disparity[s.valid], D_TRUE)
    z, ok = s.depth()  # Z = fx·B / (d + doffs) = 600 · 0,06 / 16
    assert np.allclose(z[ok], 600 * 0.06 / (D_TRUE + 10.0))
    # Las 8 columnas que el ojo derecho no confirma quedan ocluidas; el resto, visible.
    assert not s.nonoccluded[:, 100:108].any()
    assert s.nonoccluded[:, 120:200].all()
    assert (s.nonoccluded <= s.valid).all()
    names = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")
    assert s.files == tuple(f"chess1/{n}" for n in names)
    assert audit_sample(s)["fotometrico"]["gt_alineada"] is True


def test_scene_folder_fixes_swapped_size_in_calib(tmp_path):
    """Middlebury 2021 ladder1/2: foto vertical con width/height intercambiados en calib.txt."""
    from visionx.depth.datasets.middlebury import load_scene_folder

    src = synthetic_sample()
    _write_scene(tmp_path / "ladder1", src, swap_size=True)
    s = load_scene_folder(tmp_path / "ladder1", "d", "ladder1", "ladder", tmp_path)
    assert s.geometry.size == s.size


def test_half_resolution_halves_disparity_keeps_invalid_and_depth(tmp_path):
    from visionx.depth.datasets.middlebury import (
        geometry_from_calib,
        load_scene_folder,
        read_calib,
        write_half_resolution,
    )

    src = synthetic_sample()
    rng = np.random.default_rng(1)
    # GT variada (para comprobar Z punto a punto) y un bloque inválido alineado a 2x2.
    d = (src.disparity + rng.uniform(-0.4, 0.4, src.disparity.shape)).astype(np.float32)
    valid = src.valid.copy()
    valid[40:60, 80:120] = False
    d[~valid] = np.nan
    # Ancho impar: la última columna se descarta para que el factor sea exactamente 1/2.
    left, right = src.left[:, :-1], src.right[:, :-1]
    full = StereoSample(dataset="d", sample_id="X", scene_id="X", left=left.copy(),
                        right=right.copy(), disparity=d[:, :-1].copy(),
                        valid=valid[:, :-1].copy())  # fmt: skip
    _write_scene(tmp_path / "full" / "X-perfect", full, ndisp=63)
    out = write_half_resolution(tmp_path / "full" / "X-perfect", tmp_path / "H" / "X")
    half = load_scene_folder(out, "d", "X", "X", tmp_path)
    w, h = full.size
    assert half.size == (w // 2, h // 2) == (119, 80)
    big = load_scene_folder(tmp_path / "full" / "X-perfect", "d", "X", "X", tmp_path)
    # Vecino más cercano: cada valor de H es un valor de la completa (sin mezclas) / 2.
    sub = big.disparity[1::2, 1 : 2 * half.size[0] : 2]
    sub_ok = big.valid[1::2, 1 : 2 * half.size[0] : 2]
    assert np.array_equal(half.valid, sub_ok)
    assert np.allclose(half.disparity[half.valid], sub[sub_ok] / 2, atol=1e-5)
    assert not half.valid[20:30, 40:60].any()  # el bloque inválido sigue inválido
    # Geometría: fx, cx (centros alineados), doffs y ndisp a la mitad; el baseline igual.
    calib = read_calib(out / "calib.txt")
    g = geometry_from_calib(calib)  # además verifica doffs = cx1 - cx0
    assert g.fx == pytest.approx(300.0) and g.doffs == pytest.approx(5.0)
    assert g.cx == pytest.approx((120 + 0.5) / 2 - 0.5)
    assert g.baseline_m == pytest.approx(0.060)
    assert calib["ndisp"] == 32.0  # ceil(63 / 2)
    assert calib["vmax"] == pytest.approx(3.5)
    # d -> Z igual antes y después: Z = fx·B/(d + doffs) con los tres a la mitad.
    g_full = big.geometry
    z_full = g_full.fx * g_full.baseline_m / (sub[sub_ok].astype(np.float64) + g_full.doffs)
    z_half, ok = half.depth()
    assert np.array_equal(ok, sub_ok)
    assert np.allclose(z_half[ok], z_full, rtol=1e-5)


def test_dev_catalogs_index_scenes_and_partition_only_dev(tmp_path):
    from visionx.depth.__main__ import TEST_GROUPS
    from visionx.depth.datasets.catalog import CATALOG

    src = synthetic_sample()
    for name in ("chess1", "chess2", "ladder1"):
        _write_scene(tmp_path / "m21" / "data" / name, src)
    info = CATALOG["midd2021"]
    refs = info.index(tmp_path / "m21")
    assert {r.sample_id: r.scene_id for r in refs} == {"chess1": "chess", "chess2": "chess",
                                                        "ladder1": "ladder"}  # fmt: skip
    groups = info.partition(refs)
    assert groups == {"dev": ["chess", "ladder"]}
    assert not set(groups) & TEST_GROUPS  # el evaluador solo bloquea "test": esto no lo es
    assert refs[0].files[0] == "data/chess1/im0.png"

    for name in ("Sword1", "Sword2", "Couch"):
        _write_scene(tmp_path / "m14" / "full" / f"{name}-perfect", src)
    info = CATALOG["midd2014extra"]
    info.prepare(tmp_path / "m14")  # full/<X>-perfect -> H/<X>
    refs = info.index(tmp_path / "m14")
    assert {r.sample_id: r.scene_id for r in refs} == {"Couch": "Couch", "Sword1": "Sword",
                                                        "Sword2": "Sword"}  # fmt: skip
    assert info.partition(refs) == {"dev": ["Couch", "Sword"]}
    assert refs[0].load().size == (120, 80)  # se lee la versión H
    assert refs[0].files[0] == "H/Couch/im0.png"


@pytest.mark.parametrize("name", ["Adirondack", "Playtable2", "PianoL", "teddy", "Art"])
def test_dev_partition_rejects_middeval3_scenes(tmp_path, name):
    from visionx.depth.datasets.catalog import CATALOG

    _write_scene(tmp_path / "data" / name, synthetic_sample())
    refs = CATALOG["midd2021"].index(tmp_path)
    with pytest.raises(InvalidInput, match="MiddEval3"):
        CATALOG["midd2021"].partition(refs)


def test_dev_catalog_downloads_exclude_middeval3():
    from visionx.depth.datasets.catalog import CATALOG, MIDDEVAL3_SCENES, check_not_in_middeval3

    names = [d.filename.removesuffix("-perfect.zip") for d in CATALOG["midd2014extra"].downloads]
    assert len(names) == 13
    check_not_in_middeval3(names)  # no levanta
    assert not {n.rstrip("12") for n in names} & MIDDEVAL3_SCENES
    for scene in ("Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes", "Playroom",
                  "Playtable", "Recycle", "Shelves", "Vintage"):  # fmt: skip
        assert scene in MIDDEVAL3_SCENES
        with pytest.raises(InvalidInput):
            check_not_in_middeval3([scene])
    for name in ("midd2021", "midd2014extra"):
        assert "dev" in CATALOG[name].role


def test_cli_download_runs_prepare_after_extracting(tmp_path, monkeypatch):
    import dataclasses
    import zipfile

    from visionx.depth.datasets import __main__ as cli
    from visionx.depth.datasets.catalog import CATALOG, Download

    data = tmp_path / "data"
    zpath = data / "descargas" / "midd2014extra" / "Couch-perfect.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Couch-perfect/calib.txt", b"x")
    item = Download("", sha256_file(zpath), zpath.stat().st_size, zpath.name, "full")
    seen = []
    info = dataclasses.replace(CATALOG["midd2014extra"], downloads=(item,),
                               prepare=lambda root: seen.append(root))  # fmt: skip
    monkeypatch.setitem(cli.CATALOG, "midd2014extra", info)
    assert cli.main(["descargar", "midd2014extra", "--datos", str(data)]) == 0
    assert (data / "raw" / "midd2014extra" / "full" / "Couch-perfect" / "calib.txt").exists()
    assert [p.as_posix() for p in seen] == [(data / "raw" / "midd2014extra").as_posix()]


def test_audit_rejects_gt_shifted_by_doffs():
    """Escribir d - doffs (confundir disparidad de imagen con d + doffs) corre la GT 10 px: con
    una textura sin periodo la curva census no tiene mínimo en 0 y no puede salir alineada."""
    rng = np.random.default_rng(3)
    left = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
    left = cv2_blur(left)
    right = np.roll(left, -D_TRUE, axis=1)
    valid = np.ones((160, 240), bool)
    valid[:, :16] = False
    exact = np.where(valid, np.float32(D_TRUE), np.nan).astype(np.float32)
    good = synthetic_sample(left=left, right=right, valid=valid, disparity=exact)
    assert audit_sample(good)["fotometrico"]["veredicto"] == "alineada"
    shifted = np.where(valid, np.float32(D_TRUE - 10.0), np.nan).astype(np.float32)
    bad = synthetic_sample(left=left, right=right, valid=valid, disparity=shifted)
    assert audit_sample(bad)["fotometrico"]["veredicto"] != "alineada"


def cv2_blur(image):
    """Suaviza 1 px la textura aleatoria (interpolable) sin crear un periodo espacial."""
    import cv2

    return cv2.GaussianBlur(image, (3, 3), 0.8)
