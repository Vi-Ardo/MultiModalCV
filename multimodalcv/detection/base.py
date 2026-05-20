"""Object detector interfaces."""

from typing import Protocol

import numpy as np

from multimodalcv.core.models import Detection
from multimodalcv.video.reader import VideoFrame


class ObjectDetector(Protocol):
    """Interface implemented by object detectors."""

    def detect(self, frame: VideoFrame) -> list[Detection]:
        """Detect objects in a video frame."""


def image_shape(image: np.ndarray) -> tuple[int, int]:
    """Return image shape as width and height."""
    height, width = image.shape[:2]
    return width, height

