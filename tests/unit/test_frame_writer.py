import cv2
import numpy as np
import pytest

from multimodalcv.reporting.frame_writer import (
    FrameWriteError,
    frame_filename,
    write_frame_image,
    write_frame_images,
)
from multimodalcv.video.reader import VideoFrame


def make_frame(frame_index: int = 3) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        image=np.full((24, 32, 3), 128, dtype=np.uint8),
    )


def test_builds_stable_frame_filename() -> None:
    frame = make_frame(frame_index=42)

    assert frame_filename(frame) == "frame_000042.jpg"
    assert frame_filename(frame, prefix="event", extension=".png") == "event_000042.png"


def test_write_frame_image_creates_parent_directory(tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "frames"

    output_path = write_frame_image(make_frame(), output_dir)

    assert output_path == output_dir / "frame_000003.jpg"
    assert output_path.exists()

    image = cv2.imread(str(output_path))
    assert image.shape == (24, 32, 3)


def test_write_frame_images_writes_multiple_files(tmp_path) -> None:
    frames = [make_frame(frame_index=1), make_frame(frame_index=2)]

    output_paths = write_frame_images(frames, tmp_path, prefix="sample", extension="png")

    assert output_paths == [
        tmp_path / "sample_000001.png",
        tmp_path / "sample_000002.png",
    ]
    assert all(path.exists() for path in output_paths)


def test_write_frame_image_raises_for_invalid_extension(tmp_path) -> None:
    with pytest.raises(FrameWriteError):
        write_frame_image(make_frame(), tmp_path, extension="invalid")

