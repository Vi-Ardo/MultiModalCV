import json

from multimodalcv.core.models import Event, EventType, ObjectClass
from multimodalcv.reporting.summary_report import (
    build_analysis_summary,
    default_summary_path,
    write_analysis_summary_json,
)


def make_count_event(frame_index: int, count: int) -> Event:
    return Event(
        event_type=EventType.COUNT_IN_FRAME,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=ObjectClass.PERSON,
        metadata={"count": count},
    )


def test_build_analysis_summary_with_count_events() -> None:
    events = [
        make_count_event(1, 1),
        make_count_event(2, 2),
        make_count_event(3, 2),
        Event(
            event_type=EventType.ENTER_ZONE,
            frame_index=4,
            timestamp_sec=0.4,
            zone_name="main",
            track_id=1,
            object_class=ObjectClass.PERSON,
        ),
    ]

    summary = build_analysis_summary(
        events=events,
        processed_frames=4,
        detector_name="yolo",
        annotated_frame_count=2,
    )

    assert summary["processed_frames"] == 4
    assert summary["detector"] == "yolo"
    assert summary["event_count"] == 4
    assert summary["annotated_frame_count"] == 2
    assert summary["events_by_type"] == {
        "count_in_frame": 3,
        "enter_zone": 1,
    }
    assert summary["counts"]["max_count"] == 2
    assert summary["counts"]["most_common_count"] == 2
    assert summary["counts"]["count_distribution"] == {"1": 1, "2": 2}
    assert summary["counts"]["frames_with_count_gt_zero"] == 3
    assert summary["counts"]["frames_with_count_gte_two"] == 2


def test_build_analysis_summary_without_count_events() -> None:
    summary = build_analysis_summary(
        events=[
            Event(
                event_type=EventType.ENTER_ZONE,
                frame_index=1,
                timestamp_sec=0.1,
            )
        ],
        processed_frames=1,
        detector_name="fake",
    )

    assert "counts" not in summary


def test_write_analysis_summary_json(tmp_path) -> None:
    output_path = tmp_path / "reports" / "summary.json"
    summary = {"processed_frames": 3}

    written_path = write_analysis_summary_json(summary, output_path)

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_default_summary_path() -> None:
    assert str(default_summary_path("outputs/run/events.json")).endswith("outputs\\run\\summary.json")

