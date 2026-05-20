import json

import cv2
import numpy as np

from multimodalcv.cli.video import inspect_video, main


def write_sample_video(path, *, frame_count: int = 4, fps: float = 10.0) -> None:
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


def test_inspect_video_writes_metadata_and_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_dir = tmp_path / "outputs"
    write_sample_video(video_path, frame_count=4)

    result = inspect_video(video_path=video_path, output_dir=output_dir, max_frames=2)

    assert result.metadata_path == output_dir / "metadata.json"
    assert result.frames_dir == output_dir / "frames"
    assert len(result.frame_paths) == 2
    assert all(path.exists() for path in result.frame_paths)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 4
    assert metadata["width"] == 32
    assert metadata["height"] == 24


def test_inspect_video_respects_every_n_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_dir = tmp_path / "outputs"
    write_sample_video(video_path, frame_count=5)

    result = inspect_video(
        video_path=video_path,
        output_dir=output_dir,
        max_frames=2,
        every_n_frames=2,
    )

    assert [path.name for path in result.frame_paths] == [
        "frame_000000.jpg",
        "frame_000002.jpg",
    ]


def test_video_cli_main_returns_zero_for_valid_video(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    output_dir = tmp_path / "outputs"
    write_sample_video(video_path)

    exit_code = main([str(video_path), "--output-dir", str(output_dir), "--max-frames", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Video metadata written" in captured.out
    assert "Saved 1 frame image" in captured.out


def test_video_cli_main_returns_error_for_missing_video(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing.mp4")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Cannot open video file" in captured.out

