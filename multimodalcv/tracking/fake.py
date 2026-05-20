"""Fake object trackers for deterministic tests and demos."""

from collections.abc import Mapping

from multimodalcv.core.models import Detection, Track
from multimodalcv.tracking.base import ObjectTracker


class FakeTracker(ObjectTracker):
    """Tracker that returns predefined tracks by frame index."""

    def __init__(self, tracks_by_frame: Mapping[int, list[Track]]) -> None:
        self._tracks_by_frame = {
            frame_index: list(tracks)
            for frame_index, tracks in tracks_by_frame.items()
        }

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            return []

        frame_index = detections[0].frame_index
        return list(self._tracks_by_frame.get(frame_index, []))


class SequentialTracker(ObjectTracker):
    """Simple tracker that assigns detection-order IDs per frame.

    This is not a real multi-frame tracker. It is only useful for tests and
    early demos where detections are already arranged in deterministic order.
    """

    def update(self, detections: list[Detection]) -> list[Track]:
        return [
            Track(
                track_id=index + 1,
                frame_index=detection.frame_index,
                timestamp_sec=detection.timestamp_sec,
                object_class=detection.object_class,
                bbox=detection.bbox,
            )
            for index, detection in enumerate(detections)
        ]

