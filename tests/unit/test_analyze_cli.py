import json

import cv2
import numpy as np

from multimodalcv.cli.analyze import analyze_video, main
from multimodalcv.core.models import EventType


def write_sample_video(path, *, frame_count: int = 3, fps: float = 10.0) -> None:
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


def test_analyze_video_writes_enter_event_json(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    events = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ENTER_ZONE

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["event_type"] == "enter_zone"
    assert data[0]["object_class"] == "person"


def test_analyze_video_writes_empty_zone_event_json(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    events = analyze_video(
        video_path=video_path,
        command="Сообщи, когда все люди покинут зону",
        output_path=output_path,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.EMPTY_ZONE


def test_analyze_main_returns_zero_for_supported_command(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    exit_code = main(
        [
            str(video_path),
            "Посчитай людей в зоне",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote 2 event(s)" in captured.out
    assert output_path.exists()


def test_analyze_main_returns_error_for_unsupported_command(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path)

    exit_code = main([str(video_path), "Определи подозрительное поведение"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unsupported command" in captured.out
    assert "Supported commands" in captured.out


def test_analyze_main_returns_error_for_missing_video(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing.mp4"), "Посчитай людей в зоне"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Cannot open video file" in captured.out
