import sys
import types

import numpy as np
import pytest

from multimodalcv.core.models import ObjectClass
from multimodalcv.detection.yolo import YOLODependencyError, YOLODetector, _map_yolo_class
from multimodalcv.video.reader import VideoFrame


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, index):
        return self._value


class FakeBox:
    def __init__(self, *, class_index: int, confidence: float, xyxy):
        self.cls = FakeScalar(class_index)
        self.conf = FakeScalar(confidence)
        self.xyxy = [xyxy]


class FakeResult:
    names = {0: "person", 1: "car", 2: "dog"}

    def __init__(self):
        self.boxes = [
            FakeBox(class_index=0, confidence=0.9, xyxy=[1, 2, 10, 20]),
            FakeBox(class_index=1, confidence=0.8, xyxy=[20, 2, 40, 20]),
            FakeBox(class_index=2, confidence=0.99, xyxy=[50, 2, 70, 20]),
            FakeBox(class_index=0, confidence=0.1, xyxy=[80, 2, 90, 20]),
        ]


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, image, verbose):
        return [FakeResult()]


def make_frame() -> VideoFrame:
    return VideoFrame(
        frame_index=3,
        timestamp_sec=0.3,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
    )


def test_maps_supported_yolo_classes() -> None:
    assert _map_yolo_class("person") == ObjectClass.PERSON
    assert _map_yolo_class("car") == ObjectClass.CAR
    assert _map_yolo_class("dog") is None


def test_yolo_detector_raises_clear_error_when_dependency_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(YOLODependencyError):
        YOLODetector()


def test_yolo_detector_converts_supported_boxes(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    detector = YOLODetector(model_path="fake.pt", confidence_threshold=0.25)
    detections = detector.detect(make_frame())

    assert len(detections) == 2
    assert [detection.object_class for detection in detections] == [
        ObjectClass.PERSON,
        ObjectClass.CAR,
    ]
    assert detections[0].frame_index == 3
    assert detections[0].timestamp_sec == 0.3
    assert detections[0].bbox.x1 == 1.0

