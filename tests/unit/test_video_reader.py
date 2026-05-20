import cv2
import numpy as np
import pytest

from multimodalcv.video.reader import (
    VideoOpenError,
    iter_video_frames,
    read_video_metadata,
)


def write_sample_video(path, *, frame_count: int = 5, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 24),
    )
    assert writer.isOpened()

    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_reads_video_metadata(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path, frame_count=5, fps=10.0)

    metadata = read_video_metadata(video_path)

    assert metadata.path == video_path
    assert metadata.frame_count == 5
    assert metadata.fps == 10.0
    assert metadata.width == 32
    assert metadata.height == 24
    assert metadata.duration_sec == 0.5


def test_iterates_video_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path, frame_count=3, fps=10.0)

    frames = list(iter_video_frames(video_path))

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_sec for frame in frames] == [0.0, 0.1, 0.2]
    assert frames[0].image.shape == (24, 32, 3)


def test_iterates_every_n_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path, frame_count=5, fps=10.0)

    frames = list(iter_video_frames(video_path, every_n_frames=2))

    assert [frame.frame_index for frame in frames] == [0, 2, 4]


def test_iterates_max_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path, frame_count=5, fps=10.0)

    frames = list(iter_video_frames(video_path, max_frames=2))

    assert [frame.frame_index for frame in frames] == [0, 1]


def test_rejects_invalid_frame_step(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path)

    with pytest.raises(ValueError):
        list(iter_video_frames(video_path, every_n_frames=0))


def test_raises_error_for_missing_video(tmp_path) -> None:
    with pytest.raises(VideoOpenError):
        read_video_metadata(tmp_path / "missing.mp4")

