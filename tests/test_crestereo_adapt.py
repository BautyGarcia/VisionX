"""CREStereo adaptable (T12): wrapper y estimador, con una red falsa.

Ningún test baja ni carga los pesos reales: la red se reemplaza por una que devuelve una
disparidad conocida y anota cómo la llamaron. Así se prueba todo lo que rodea a la red.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from visionx.depth.crestereo_adapt.estimator import CreStereoEstimator, load_estimator  # noqa: E402
from visionx.depth.crestereo_adapt.matcher import (  # noqa: E402
    CreStereoMatcher,
    CreStereoParams,
    padded_size,
)
from visionx.depth.crestereo_adapt.port import PortConfig, check_files, sha256_file  # noqa: E402
from visionx.shared.config import load_config  # noqa: E402
from visionx.shared.contracts import DepthStatus, StereoFrame, check_depth_geometry  # noqa: E402
from visionx.shared.errors import EstimatorError, IdentityMismatch, InvalidInput  # noqa: E402
from visionx.shared.geometry import StereoGeometry  # noqa: E402

DISPARITY = 7.5


class FakeNet:
    """Imita la interfaz del port: (izquierda, derecha, iters, flow_init) -> flujo de 2 canales."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, left, right, iters=10, flow_init=None):
        self.calls.append(
            {
                "shape": tuple(left.shape),
                "iters": iters,
                "init": flow_init is not None,
                "max": float(left.max()),
            }
        )
        flow = torch.zeros(1, 2, left.shape[2], left.shape[3])
        # La disparidad está en píxeles de la imagen que entró: a media resolución, la mitad.
        flow[:, 0] = DISPARITY
        return flow


def geometry() -> StereoGeometry:
    return StereoGeometry(
        width=200, height=120, fx=400.0, fy=400.0, cx=100.0, cy=60.0, baseline_m=0.06
    )


def pair(height: int = 120, width: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    left = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return left, left.copy()


def frame(geom: StereoGeometry, **changes) -> StereoFrame:
    left, right = pair(geom.height, geom.width)
    fields = dict(
        session_id="s",
        frame_id=3,
        left=left,
        right=right,
        received_ns=1,
        rectified=True,
        calibration_id="calib-test",
        geometry_id=geom.geometry_id,
    )
    fields.update(changes)
    return StereoFrame(**fields)


def test_padded_size_sube_al_multiplo() -> None:
    assert padded_size(360, 640, 16) == (368, 640)
    assert padded_size(360, 640, 32) == (384, 640)
    assert padded_size(352, 640, 16) == (352, 640)


def test_params_validan_y_tienen_etiqueta_estable() -> None:
    assert CreStereoParams(iters=4, passes=1).tag == "i4p1"
    assert CreStereoParams(iters=10, passes=2, warm_start=True, warm_reset_every=30).tag == (
        "i10p2w30"
    )
    assert CreStereoParams(passes=2).multiple == 32
    with pytest.raises(InvalidInput):
        CreStereoParams(iters=0)
    with pytest.raises(InvalidInput):
        CreStereoParams(passes=3)


def test_compute_rellena_recorta_y_devuelve_la_disparidad() -> None:
    net = FakeNet()
    matcher = CreStereoMatcher(net, "cpu", CreStereoParams(iters=4))
    left, right = pair()
    disparity = matcher.compute(left, right)
    assert disparity.shape == (120, 200) and disparity.dtype == np.float32
    assert np.allclose(disparity, DISPARITY)
    # 120 no es múltiplo de 16: la red vio 128 filas. La imagen entra en 0-255, sin normalizar.
    assert net.calls == [
        {"shape": (1, 3, 128, 208), "iters": 4, "init": False, "max": float(left.max())}
    ]


def test_dos_pasadas_media_resolucion_y_despues_completa() -> None:
    net = FakeNet()
    matcher = CreStereoMatcher(net, "cpu", CreStereoParams(iters=10, passes=2))
    matcher.compute(*pair())
    assert [c["shape"] for c in net.calls] == [(1, 3, 64, 112), (1, 3, 128, 224)]
    assert [c["init"] for c in net.calls] == [False, True]


def test_arranque_en_caliente_y_reinicio() -> None:
    net = FakeNet()
    params = CreStereoParams(iters=4, warm_start=True, warm_reset_every=2)
    matcher = CreStereoMatcher(net, "cpu", params)
    left, right = pair()
    warm = []
    for _ in range(5):
        matcher.compute(left, right)
        warm.append(matcher.last_stats["caliente"] > 0)
    # frío, dos en caliente, y al tercero vuelve a arrancar en frío
    assert warm == [False, True, True, False, True]
    assert [c["init"] for c in net.calls] == warm
    matcher.reset()
    matcher.compute(left, right)
    assert matcher.last_stats["caliente"] == 0


def test_arranque_en_caliente_no_reusa_otro_tamano() -> None:
    net = FakeNet()
    matcher = CreStereoMatcher(net, "cpu", CreStereoParams(warm_start=True, warm_reset_every=0))
    matcher.compute(*pair(120, 200))
    matcher.compute(*pair(64, 96))
    assert net.calls[-1]["init"] is False


def test_compute_rechaza_entradas_mal_formadas() -> None:
    matcher = CreStereoMatcher(FakeNet(), "cpu", CreStereoParams())
    left, right = pair()
    with pytest.raises(InvalidInput):
        matcher.compute(left, right[:-1])
    with pytest.raises(InvalidInput):
        matcher.compute(left.astype(np.float32), right.astype(np.float32))


def test_estimador_cumple_el_contrato() -> None:
    geom = geometry()
    est = CreStereoEstimator(
        CreStereoMatcher(FakeNet(), "cpu", CreStereoParams(iters=4)), geom, weights_id="abc"
    )
    result = est.predict(frame(geom))
    assert est.model_id == "crestereo-abc-200x120-i4p1"
    assert result.model_id == est.model_id
    assert result.status is DepthStatus.VALID
    assert result.valid_mask.all()
    check_depth_geometry(result, geom)  # metros = fx * B / (d + doffs)
    assert np.allclose(result.depth_m, 400.0 * 0.06 / DISPARITY)
    est.close()


def test_estimador_calcula_a_otra_resolucion() -> None:
    geom = geometry()
    est = CreStereoEstimator(
        CreStereoMatcher(FakeNet(), "cpu", CreStereoParams()), geom, compute_size=(100, 60)
    )
    result = est.predict(frame(geom))
    assert result.size == (100, 60)
    check_depth_geometry(result, geom.resized(100, 60))


def test_estimador_rechaza_par_sin_rectificar_u_otra_geometria() -> None:
    geom = geometry()
    est = CreStereoEstimator(CreStereoMatcher(FakeNet(), "cpu", CreStereoParams()), geom)
    with pytest.raises(InvalidInput):
        est.predict(frame(geom, rectified=False, calibration_id=None, geometry_id=None))
    with pytest.raises(IdentityMismatch):
        est.predict(frame(geom, geometry_id="otra"))


def test_error_de_la_red_sale_como_estimator_error() -> None:
    def broken(left, right, iters=10, flow_init=None):
        raise RuntimeError("sin memoria")

    geom = geometry()
    est = CreStereoEstimator(CreStereoMatcher(broken, "cpu", CreStereoParams()), geom)
    with pytest.raises(EstimatorError):
        est.predict(frame(geom))


def test_config_trae_los_dos_modos() -> None:
    cfg = load_config(None)
    section = cfg["depth"]["crestereo"]
    port = PortConfig.from_config(section)
    assert len(port.port_commit) == 40 and len(port.weights_sha256) == 64
    geom = geometry()
    for mode in ("continuo", "preciso"):
        est = load_estimator(cfg, geom, mode, model=FakeNet(), device="cpu")
        assert est.geometry.size == tuple(section[mode]["compute"])
    with pytest.raises(InvalidInput):
        load_estimator(cfg, geom, "otro", model=FakeNet(), device="cpu")


def test_check_files_explica_que_falta(tmp_path) -> None:
    weights = tmp_path / "pesos.pth"
    cfg = PortConfig(tmp_path / "port", "a" * 40, weights, "0" * 64)
    with pytest.raises(EstimatorError, match="descargar_crestereo"):
        check_files(cfg)  # no hay código
    (tmp_path / "port" / "nets").mkdir(parents=True)
    (tmp_path / "port" / "nets" / "crestereo.py").write_text("", encoding="utf-8")
    (tmp_path / "port" / "COMMIT").write_text("b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(EstimatorError, match="la config fija"):
        check_files(cfg)  # otro commit
    (tmp_path / "port" / "COMMIT").write_text("a" * 40 + "\n", encoding="utf-8")
    with pytest.raises(EstimatorError, match="faltan los pesos"):
        check_files(cfg)
    weights.write_bytes(b"no son los pesos")
    with pytest.raises(EstimatorError, match="no son los fijados"):
        check_files(cfg)
    ok = PortConfig(cfg.port_dir, cfg.port_commit, weights, sha256_file(weights))
    check_files(ok)
