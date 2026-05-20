"""Video file reading utilities."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for an archived video file."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class VideoFrame:
    """A decoded video frame with timing metadata."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


class VideoOpenError(ValueError):
    """Raised when a video file cannot be opened."""


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read metadata from a video file."""
    path = Path(video_path)
    capture = _open_capture(path)

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        path=path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def iter_video_frames(
    video_path: str | Path,
    *,
    every_n_frames: int = 1,
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Yield decoded video frames with frame index and timestamp."""
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    path = Path(video_path)
    capture = _open_capture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    yielded = 0

    try:
        while True:
            success, image = capture.read()
            if not success:
                break

            if frame_index % every_n_frames == 0:
                timestamp_sec = frame_index / fps if fps > 0 else 0.0
                yield VideoFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image=image,
                )
                yielded += 1

                if max_frames is not None and yielded >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"Cannot open video file: {path}")
    return capture

