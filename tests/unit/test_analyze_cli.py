import json

import cv2
import numpy as np
import pytest

from multimodalcv.cli.analyze import analyze_video, main
from multimodalcv.core.models import BoundingBox, Detection, EventType, ObjectClass


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


def test_analyze_video_uses_custom_zone_rect(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    events = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
        zone_rect="200,200,300,300",
    )

    assert events == []


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


def test_analyze_video_writes_count_in_frame_event_json(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "events.json"
    write_sample_video(video_path)

    events = analyze_video(
        video_path=video_path,
        command="Посчитай людей в кадре",
        output_path=output_path,
    )

    assert len(events) == 2
    assert events[0].event_type == EventType.COUNT_IN_FRAME
    assert events[0].metadata["count"] == 1


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
    assert (frames_dir / "annotated_000000.jpg").exists()
    assert (frames_dir / "annotated_000001.jpg").exists()


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

    events = analyze_video(
        video_path=video_path,
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
        detector_name="yolo",
        model_path=tmp_path / "fake.pt",
        confidence_threshold=0.5,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ENTER_ZONE


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
