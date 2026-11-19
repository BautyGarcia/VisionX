"""`incluir:` en los overrides de configuración (configs/velocidad/*.yaml)."""

import pytest

from visionx.shared.config import load_config


def test_include_applies_the_base_first_and_the_test_keys_on_top(tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text("camera:\n  device: 3\ndetection:\n  imgsz: 640\n", encoding="utf-8")
    test = tmp_path / "C1.yaml"
    test.write_text(f"incluir: {local.as_posix()}\ndetection:\n  device: mps\n  imgsz: 960\n",
                    encoding="utf-8")  # fmt: skip
    cfg = load_config(test)
    assert cfg["camera"]["device"] == 3  # de la local
    assert cfg["detection"]["device"] == "mps" and cfg["detection"]["imgsz"] == 960  # la prueba
    assert "incluir" not in cfg


def test_include_cycle_is_an_error(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(f"incluir: {b.as_posix()}\n", encoding="utf-8")
    b.write_text(f"incluir: {a.as_posix()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ciclo"):
        load_config(a)


def test_speed_configs_change_only_their_keys(tmp_path, monkeypatch):
    import yaml

    from visionx.shared.config import ROOT

    for name in ("C0", "C1", "C2", "C3", "C4"):
        data = yaml.safe_load((ROOT / "configs" / "velocidad" / f"{name}.yaml").read_text("utf-8"))
        assert data["incluir"] == "configs/local.yaml"
        assert set(data) == {"incluir", "detection", "depth"}
        assert data["depth"]["precisa"]["espera_s"] == 3.0
