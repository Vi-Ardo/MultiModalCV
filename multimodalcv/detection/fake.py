"""Fake object detector for deterministic tests and demos."""

from collections.abc import Mapping

from multimodalcv.core.models import Detection
from multimodalcv.detection.base import ObjectDetector
from multimodalcv.video.reader import VideoFrame


class FakeDetector(ObjectDetector):
    """Detector that returns predefined detections by frame index."""

    def __init__(self, detections_by_frame: Mapping[int, list[Detection]]) -> None:
        self._detections_by_frame = {
            frame_index: list(detections)
            for frame_index, detections in detections_by_frame.items()
        }

    def detect(self, frame: VideoFrame) -> list[Detection]:
        return list(self._detections_by_frame.get(frame.frame_index, []))

