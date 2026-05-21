import json

import cv2
import numpy as np
import pytest

from multimodalcv.cli.analyze import (
    analyze_video,
    format_event_summary,
    format_timestamp,
    main,
    print_event_summary,
)
from multimodalcv.core.models import BoundingBox, Detection, Event, EventType, ObjectClass


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

    result = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
    )

    assert len(result.events) == 1
    assert result.events[0].event_type == EventType.ENTER_ZONE
    assert result.processed_frames == 2
    assert result.detector_name == "fake"

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["event_type"] == "enter_zone"
    assert data[0]["object_class"] == "person"


def test_analyze_video_uses_custom_zone_rect(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    result = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
        zone_rect="200,200,300,300",
    )

    assert result.events == []


def test_analyze_video_writes_empty_zone_event_json(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    result = analyze_video(
        video_path=video_path,
        command="Сообщи, когда все люди покинут зону",
        output_path=output_path,
    )

    assert len(result.events) == 1
    assert result.events[0].event_type == EventType.EMPTY_ZONE


def test_analyze_video_writes_count_in_frame_event_json(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    result = analyze_video(
        video_path=video_path,
        command="Посчитай людей в кадре",
        output_path=output_path,
    )

    assert len(result.events) == 2
    assert result.events[0].event_type == EventType.COUNT_IN_FRAME
    assert result.events[0].metadata["count"] == 1


def test_analyze_video_can_smooth_count_events(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    result = analyze_video(
        video_path=video_path,
        command="Посчитай людей в кадре",
        output_path=output_path,
        count_window_size=3,
    )

    assert "raw_count" in result.events[0].metadata


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
    assert "Processed frame(s): 2" in captured.out
    assert "Detector: fake" in captured.out
    assert "Wrote 2 event(s)" in captured.out
    assert "Events:" in captured.out
    assert "count_in_zone person zone=main count=1" in captured.out
    assert output_path.exists()


def test_analyze_main_can_save_annotated_frames(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    frames_dir = tmp_path / "frames"
    write_sample_video(video_path)

    exit_code = main(
        [
            str(video_path),
            "Сообщи, когда человек войдет в зону",
            "--output",
            str(output_path),
            "--save-frames",
            "--frames-dir",
            str(frames_dir),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    captured = capsys.readouterr()
    assert "Saved 2 annotated frame(s)" in captured.out
    assert (frames_dir / "annotated_000000.jpg").exists()
    assert (frames_dir / "annotated_000001.jpg").exists()


def test_analyze_main_can_save_only_event_frames(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    frames_dir = tmp_path / "frames"
    write_sample_video(video_path)

    exit_code = main(
        [
            str(video_path),
            "Сообщи, когда человек войдет в зону",
            "--output",
            str(output_path),
            "--save-frames",
            "--frames-dir",
            str(frames_dir),
            "--frame-mode",
            "events",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Saved 1 annotated frame(s)" in captured.out
    assert not (frames_dir / "annotated_000000.jpg").exists()
    assert (frames_dir / "annotated_000001.jpg").exists()


def test_analyze_main_accepts_event_cooldown(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    exit_code = main(
        [
            str(video_path),
            "Сообщи, когда человек войдет в зону",
            "--output",
            str(output_path),
            "--event-cooldown-sec",
            "2.0",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "00:00.00"
    assert format_timestamp(1.23) == "00:01.23"
    assert format_timestamp(61.05) == "01:01.05"


def test_format_event_summary() -> None:
    event = Event(
        event_type=EventType.ENTER_ZONE,
        frame_index=12,
        timestamp_sec=1.2,
        zone_name="main",
        track_id=7,
        object_class=ObjectClass.PERSON,
    )

    assert format_event_summary(event) == "00:01.20 enter_zone person track=#7 zone=main"


def test_format_event_summary_includes_count() -> None:
    event = Event(
        event_type=EventType.COUNT_IN_FRAME,
        frame_index=12,
        timestamp_sec=1.2,
        object_class=ObjectClass.PERSON,
        metadata={"count": 3},
    )

    assert format_event_summary(event) == "00:01.20 count_in_frame person count=3"


def test_print_event_summary_handles_empty_list(capsys) -> None:
    print_event_summary([])

    captured = capsys.readouterr()
    assert "Events: none" in captured.out


def test_print_event_summary_limits_long_lists(capsys) -> None:
    events = [
        Event(event_type=EventType.COUNT_IN_FRAME, frame_index=index, timestamp_sec=index)
        for index in range(12)
    ]

    print_event_summary(events, max_events=10)

    captured = capsys.readouterr()
    assert captured.out.count("count_in_frame") == 10
    assert "... 2 more event(s)" in captured.out


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


def test_analyze_main_returns_error_for_invalid_zone_rect(tmp_path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    write_sample_video(video_path)

    exit_code = main(
        [
            str(video_path),
            "Посчитай людей в зоне",
            "--zone-rect",
            "10,20,5,30",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "zone rectangle" in captured.out


class StubYOLODetector:
    def __init__(self, model_path, confidence_threshold, supported_classes):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.supported_classes = supported_classes

    def detect(self, frame):
        center_x = 105 if frame.frame_index == 0 else 95
        return [
            Detection(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                object_class=ObjectClass.PERSON,
                confidence=0.9,
                bbox=BoundingBox(
                    x1=center_x - 10,
                    y1=40,
                    x2=center_x + 10,
                    y2=60,
                ),
            )
        ]


def test_analyze_video_can_use_yolo_backend_with_stub(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)
    monkeypatch.setattr("multimodalcv.cli.analyze.YOLODetector", StubYOLODetector)

    result = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
        detector_name="yolo",
        model_path=tmp_path / "fake.pt",
        confidence_threshold=0.5,
    )

    assert len(result.events) == 1
    assert result.events[0].event_type == EventType.ENTER_ZONE
    assert result.detector_name == "yolo"


def test_analyze_video_rejects_unknown_detector_backend(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    with pytest.raises(ValueError):
        analyze_video(
            video_path=video_path,
            command="Посчитай людей в зоне",
            output_path=output_path,
            detector_name="unknown",
        )
