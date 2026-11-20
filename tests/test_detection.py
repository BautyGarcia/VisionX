"""Tests de detección (T06) sin pesos, sin cámara y sin internet.

El adaptador YOLO se prueba con un modelo sustituto que imita la salida de Ultralytics; así se
comprueba la conversión a Detection, los errores y los IDs sin bajar pesos. Los tests de
ByteTrack necesitan el extra `deteccion` y se saltean si no está instalado (como en CI).
Ninguno de estos tests mide la calidad del detector: eso requiere clips anotados.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from visionx.camera import Recorder
from visionx.detection import (
    COCO_CLASSES,
    COCO_ES,
    ClassCatalog,
    DetectorSettings,
    FakeDetector,
    FakeObject,
    YoloDetector,
    build_detection,
    empty_detection,
    normalize,
)
from visionx.detection import yolo as yolo_module
from visionx.detection.__main__ import (
    detection_record,
    latency_summary,
    open_input,
)
from visionx.detection.__main__ import (
    main as detection_main,
)
from visionx.detection.draw import draw_detection
from visionx.shared.config import ROOT, load_config
from visionx.shared.contracts import StereoFrame, check_detection_frame
from visionx.shared.errors import EstimatorError, InvalidInput
from visionx.shared.interfaces import Detector

W, H = 64, 48


def make_frame(frame_id: int = 0, session: str = "s", *, rectified: bool = True) -> StereoFrame:
    image = np.full((H, W, 3), 90, np.uint8)
    extra = (
        dict(rectified=True, calibration_id="calib-test", geometry_id="geo-test")
        if rectified
        else {}
    )
    return StereoFrame(session, frame_id, image, image, received_ns=frame_id, **extra)


# --- catálogo ---------------------------------------------------------------------------


def test_coco_catalog_has_80_classes_with_unambiguous_spanish_names():
    assert len(COCO_CLASSES) == 80
    catalog = ClassCatalog(COCO_CLASSES)  # un alias repetido entre clases lanzaría ValueError
    assert catalog.missing_aliases() == ()
    assert all(names for names in COCO_ES.values())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("taza", "cup"),
        ("las tazas", "cup"),
        ("  Teléfono ", "cell phone"),
        ("telefono", "cell phone"),
        ("Controles remotos", "remote"),
        ("relojes", "clock"),
        ("bates", "baseball bat"),
        ("mi celular", "cell phone"),
        ("cup", "cup"),  # el nombre propio del modelo también sirve
        ("llaves", None),  # COCO no tiene llaves: no se finge que se pueden buscar
        ("buscá la taza que está arriba de la mesa", None),  # frase, no nombre de objeto
        ("", None),
    ],
)
def test_resolve_spanish_requests(text, expected):
    assert ClassCatalog(COCO_CLASSES).resolve(text) == expected


def test_spoken_name_and_unknown_class():
    catalog = ClassCatalog(COCO_CLASSES)
    assert catalog.spoken("cup") == "taza"
    assert catalog.spoken("dining table") == "mesa"
    assert catalog.spoken("cosa-rara") == "cosa-rara"


def test_aliases_only_for_classes_the_model_has():
    catalog = ClassCatalog(["cup", "bottle"])
    assert catalog.resolve("taza") == "cup"
    assert catalog.resolve("silla") is None  # alias de una clase que este modelo no tiene
    assert ClassCatalog(["cup", "widget"]).missing_aliases() == ("widget",)


def test_ambiguous_alias_is_rejected():
    with pytest.raises(ValueError, match="apunta"):
        ClassCatalog(["cup", "bowl"], {"cup": ("vaso",), "bowl": ("Vasó",)})


def test_normalize():
    assert normalize("  Árbol   Grande ") == "arbol grande"


# --- construir Detection ----------------------------------------------------------------


def test_build_detection_keeps_frame_context_and_passes_consumer_checks():
    frame = make_frame(3)
    boxes = np.array([[-5, 2, 30, 60]], np.float32)  # se sale de la imagen
    original = boxes.copy()
    det = build_detection(
        frame,
        model_id="m",
        boxes_xyxy=boxes,
        class_names=["cup"],
        scores=np.array([1.0000001], np.float32),
        inference_ms=5.0,
        masks=np.full((1, H, W), 0.7, np.float32),
        track_ids=[4],
    )
    check_detection_frame(frame, det)
    assert det.calibration_id == "calib-test" and det.geometry_id == "geo-test"
    assert det.image_size == (W, H)
    np.testing.assert_array_equal(det.boxes[0], [0, 2, 30, H])
    np.testing.assert_array_equal(boxes, original)  # no modificó el array de quien llama
    assert det.scores[0] == 1.0
    assert det.masks.dtype == np.bool_ and det.masks.all()
    assert det.track_ids == (4,)


def test_build_detection_requires_declared_geometry():
    with pytest.raises(InvalidInput, match="geometry_id"):
        empty_detection(make_frame(rectified=False), model_id="m", inference_ms=1.0)


def test_masks_at_network_resolution_are_an_adapter_error():
    with pytest.raises(EstimatorError, match="máscaras"):
        build_detection(
            make_frame(),
            model_id="m",
            boxes_xyxy=np.zeros((1, 4), np.float32),
            class_names=["cup"],
            scores=np.ones(1, np.float32),
            inference_ms=1.0,
            masks=np.zeros((1, H // 2, W // 2), np.bool_),
        )


def test_inconsistent_counts_are_an_adapter_error():
    with pytest.raises(EstimatorError, match="cajas"):
        build_detection(
            make_frame(),
            model_id="m",
            boxes_xyxy=np.zeros((2, 4), np.float32),
            class_names=["cup"],
            scores=np.ones(1, np.float32),
            inference_ms=1.0,
        )


def test_zero_objects_is_a_valid_result():
    det = empty_detection(make_frame(), model_id="m", inference_ms=1.0)
    assert len(det) == 0 and det.masks.shape == (0, H, W)


# --- detector falso ---------------------------------------------------------------------


def test_fake_detector_follows_script_and_interface():
    fake = FakeDetector(
        {
            0: [FakeObject("cup", (0.25, 0.25, 0.75, 0.75), 0.8, track_id=1)],
            2: [FakeObject("bottle", (0.0, 0.0, 0.5, 0.5))],
        }
    )
    assert isinstance(fake, Detector)
    d0, d1, d2 = (fake.predict(make_frame(i)) for i in range(3))
    assert d0.classes == ("cup",) and d0.track_ids == (1,)
    np.testing.assert_allclose(d0.boxes[0], [16, 12, 48, 36])
    # Elipse inscripta: dentro de la caja, pero sin las esquinas.
    mask = d0.masks[0]
    assert mask[24, 32] and not mask[12, 16] and not mask[:, :16].any()
    assert len(d1) == 0
    assert d2.track_ids is None
    assert fake.calls == 3


def test_fake_detector_simulates_model_failure_and_close():
    fake = FakeDetector(lambda i: [FakeObject("cup", (0.1, 0.1, 0.2, 0.2))], fail_on=[1])
    assert fake.predict(make_frame(0)).classes == ("cup",)
    with pytest.raises(EstimatorError, match="simulada"):
        fake.predict(make_frame(1))
    fake.close()
    with pytest.raises(EstimatorError):
        fake.predict(make_frame(2))


def test_draw_detection_does_not_touch_input():
    frame = make_frame()
    det = FakeDetector({0: [FakeObject("cup", (0.2, 0.2, 0.8, 0.8), track_id=2)]}).predict(frame)
    image = frame.left.copy()
    view = draw_detection(image, det, ClassCatalog(COCO_CLASSES).spoken)
    assert view.shape == image.shape and not np.array_equal(view, image)
    np.testing.assert_array_equal(image, frame.left)
    with pytest.raises(ValueError):
        draw_detection(np.zeros((10, 10, 3), np.uint8), det)


# --- adaptador YOLO con un modelo sustituto ---------------------------------------------


class _Arr(np.ndarray):
    """Array con .cpu()/.numpy(), como un tensor de torch ya en CPU."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32)
        self.cls = np.asarray(cls, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.conf)


class FakeYOLO:
    """Imita ultralytics.YOLO: names y predict() con una salida fija por llamada."""

    names = {0: "person", 1: "cup"}
    outputs: list = []
    fail = False

    def __init__(self, path):
        self.path = path
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        if FakeYOLO.fail:
            raise RuntimeError("explotó")
        xyxy, conf, cls, masks = FakeYOLO.outputs.pop(0)
        boxes = FakeBoxes(xyxy, conf, cls)
        mask_obj = None if masks is None else SimpleNamespace(data=np.asarray(masks).view(_Arr))
        return [SimpleNamespace(boxes=boxes, masks=mask_obj, speed={"inference": 3.0})]


@pytest.fixture
def fake_yolo(monkeypatch, tmp_path):
    weights = tmp_path / "pesos"
    weights.mkdir()
    (weights / "yolo-prueba.pt").write_bytes(b"no es un modelo de verdad")
    fake_torch = SimpleNamespace(
        set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(yolo_module, "_import_ultralytics", lambda: (FakeYOLO, fake_torch))
    FakeYOLO.outputs = []
    FakeYOLO.fail = False
    return DetectorSettings(model="yolo-prueba", weights_dir=str(weights), track=False)


def test_yolo_adapter_converts_model_output(fake_yolo):
    detector = YoloDetector(fake_yolo)
    assert isinstance(detector, Detector)
    assert detector.model_id == f"yolo-prueba@{detector.weights_sha256[:12]}"
    assert detector.class_names == ("person", "cup")
    masks = np.zeros((2, H, W), np.float32)
    masks[1, 10:20, 10:20] = 1.0
    FakeYOLO.outputs = [([[1, 2, 10, 20], [5, 5, 30, 30]], [0.9, 0.4], [1, 0], masks)]
    det = detector.predict(make_frame(7))
    assert det.frame_id == 7 and det.classes == ("cup", "person")
    assert det.track_ids is None  # sin tracking no se inventan IDs
    assert det.masks[1].sum() == 100 and not det.masks[0].any()
    call = detector._model.calls[0]
    assert call["retina_masks"] is True and call["device"] == "cpu"


def test_yolo_adapter_zero_objects_and_failures(fake_yolo):
    detector = YoloDetector(fake_yolo)
    FakeYOLO.outputs = [(np.zeros((0, 4)), [], [], None)]
    assert len(detector.predict(make_frame())) == 0
    FakeYOLO.fail = True
    with pytest.raises(EstimatorError, match="explotó"):
        detector.predict(make_frame(1))  # un fallo no se disfraza de "cero objetos"
    with pytest.raises(InvalidInput):
        detector.predict(make_frame(2, rectified=False))


def test_yolo_adapter_class_filter(fake_yolo):
    import dataclasses

    detector = YoloDetector(dataclasses.replace(fake_yolo, classes=("cup",)))
    FakeYOLO.outputs = [(np.zeros((0, 4)), [], [], None)]
    detector.predict(make_frame())
    assert detector._model.calls[0]["classes"] == [1]
    with pytest.raises(ValueError, match="llaves"):
        YoloDetector(dataclasses.replace(fake_yolo, classes=("llaves",)))


def test_missing_weights_explain_how_to_download(tmp_path):
    settings = DetectorSettings(model="yolo26n-seg", weights_dir=str(tmp_path))
    with pytest.raises(EstimatorError, match="descargar"):
        YoloDetector(settings)


def test_settings_from_default_config():
    settings = DetectorSettings.from_config(load_config())
    # default elegido con clips ELP (BITACORA T09 mejoras): yolo26s a 960 px
    assert settings.model == "yolo26s-seg" and settings.imgsz == 960 and settings.track is True
    assert settings.weights_path == ROOT / "weights" / "yolo26s-seg.pt"


def test_download_rejects_error_pages_and_keeps_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        yolo_module.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"<html>404</html>")
    )
    with pytest.raises(OSError, match="no parece"):
        yolo_module.download_weights("yolo-x", tmp_path)
    assert list(tmp_path.iterdir()) == []  # ni el .pt ni el temporal
    (tmp_path / "yolo-x.pt").write_bytes(b"ya estaba")
    assert yolo_module.download_weights("yolo-x", tmp_path).read_bytes() == b"ya estaba"


# --- ByteTrack (necesita el extra deteccion) --------------------------------------------


def _boxes(rows):
    from ultralytics.engine.results import Boxes

    data = np.array(rows, np.float32).reshape(-1, 6)  # x1, y1, x2, y2, conf, cls
    return Boxes(data, (480, 640)).cpu().numpy()


def test_bytetrack_ids_report_unconfirmed_detections_as_none():
    pytest.importorskip("ultralytics")
    from visionx.detection.tracking import ByteTrackIds

    ids = ByteTrackIds()
    a = [100, 100, 200, 220, 0.9, 41]
    b = [400, 300, 450, 380, 0.8, 39]
    assert ids.assign(_boxes([a])) == [1]  # el primer frame activa directamente
    # Objeto nuevo: ByteTrack lo confirma recién en el frame siguiente, pero no se oculta.
    assert ids.assign(_boxes([a, b])) == [1, None]
    assert ids.assign(_boxes([a, b])) == [1, 2]
    assert ids.assign(_boxes([])) == []
    ids.reset()
    assert ids.assign(_boxes([b])) == [1]


def test_bytetrack_rejects_trackers_that_download_models():
    pytest.importorskip("ultralytics")
    from visionx.detection.tracking import ByteTrackIds

    with pytest.raises(ValueError, match="bytetrack"):
        ByteTrackIds("botsort.yaml")


# --- CLI --------------------------------------------------------------------------------


def test_cli_clases(capsys):
    assert detection_main(["clases", "tazas"]) == 0
    assert "cup" in capsys.readouterr().out
    assert detection_main(["clases", "llaves"]) == 0  # vocabulario de VisionX (YOLOE)
    assert "keys" in capsys.readouterr().out
    assert detection_main(["clases", "dragón"]) == 1
    assert detection_main(["clases"]) == 0
    assert "215 clases de VisionX (80 COCO" in capsys.readouterr().out


def test_open_input_images_declare_their_geometry(tmp_path):
    for name, size in (("a.png", (40, 30)), ("b.jpg", (20, 10))):
        cv2.imwrite(str(tmp_path / name), np.zeros((size[1], size[0], 3), np.uint8))
    (tmp_path / "notas.txt").write_text("no es imagen")
    frames = list(open_input(str(tmp_path), {}))
    assert [f.geometry_id for f in frames] == ["imagen-40x30", "imagen-20x10"]
    assert [f.frame_id for f in frames] == [0, 1]
    assert not frames[0].rectified
    with pytest.raises(InvalidInput, match="grabaciones"):
        open_input(str(tmp_path), {}, rectify=True)


def test_open_input_recording_marks_unrectified_geometry(tmp_path):
    with Recorder(tmp_path / "rec", queue_size=0) as recorder:
        for i in range(2):
            recorder.write(make_frame(i, rectified=False))
    frames = list(open_input(str(tmp_path / "rec"), {}))
    assert [f.geometry_id for f in frames] == [f"sin-rectificar-{W}x{H}"] * 2


def test_detection_record_is_json_and_latency_summary():
    det = FakeDetector({0: [FakeObject("cup", (0.1, 0.1, 0.5, 0.5), track_id=3)]}).predict(
        make_frame()
    )
    record = json.loads(json.dumps(detection_record(det, ClassCatalog(COCO_CLASSES))))
    obj = record["objetos"][0]
    assert obj["nombre"] == "taza" and obj["track"] == 3 and obj["mascara_px"] > 0
    assert latency_summary([]) == {"n": 0}
    summary = latency_summary([10.0, 20.0, 30.0])
    assert summary["mediana_ms"] == 20.0 and summary["max_ms"] == 30.0


def test_cli_reports_missing_weights_without_traceback(tmp_path, capsys):
    override = tmp_path / "local.yaml"
    override.write_text(f"detection:\n  weights_dir: {Path(tmp_path).as_posix()}\n")
    image = tmp_path / "x.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    code = detection_main(["detectar", str(image), "--config", str(override)])
    assert code == 1 and "descargar" in capsys.readouterr().err
