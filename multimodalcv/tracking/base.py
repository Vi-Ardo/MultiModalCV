"""Object tracker interfaces."""

from typing import Protocol

from multimodalcv.core.models import Detection, Track


class ObjectTracker(Protocol):
    """Interface implemented by object trackers."""

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker state from detections and return active tracks."""

