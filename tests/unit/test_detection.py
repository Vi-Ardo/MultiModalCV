import numpy as np

from multimodalcv.core.models import BoundingBox, Detection, ObjectClass
from multimodalcv.detection.base import image_shape
from multimodalcv.detection.fake import FakeDetector
from multimodalcv.video.reader import VideoFrame


def make_frame(frame_index: int = 1) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
    )


def make_detection(frame_index: int = 1) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=ObjectClass.PERSON,
        confidence=0.9,
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
    )


def test_image_shape_returns_width_and_height() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    assert image_shape(image) == (32, 24)


def test_fake_detector_returns_detections_for_frame() -> None:
    detection = make_detection(frame_index=3)
    detector = FakeDetector({3: [detection]})

    detections = detector.detect(make_frame(frame_index=3))

    assert detections == [detection]


def test_fake_detector_returns_empty_list_for_unknown_frame() -> None:
    detector = FakeDetector({3: [make_detection(frame_index=3)]})

    detections = detector.detect(make_frame(frame_index=4))

    assert detections == []


def test_fake_detector_returns_copy_of_detection_list() -> None:
    detection = make_detection(frame_index=3)
    detector = FakeDetector({3: [detection]})

    detections = detector.detect(make_frame(frame_index=3))
    detections.clear()

    assert detector.detect(make_frame(frame_index=3)) == [detection]

