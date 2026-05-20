import json

from multimodalcv.core.models import Event, EventType, ObjectClass
from multimodalcv.reporting.json_report import event_to_dict, events_to_dicts, write_events_json


def make_event() -> Event:
    return Event(
        event_type=EventType.ENTER_ZONE,
        frame_index=12,
        timestamp_sec=1.2,
        zone_name="main",
        track_id=7,
        object_class=ObjectClass.PERSON,
        metadata={"confidence": 0.91},
    )


def test_event_to_dict_serializes_enums() -> None:
    data = event_to_dict(make_event())

    assert data == {
        "event_type": "enter_zone",
        "frame_index": 12,
        "timestamp_sec": 1.2,
        "zone_name": "main",
        "track_id": 7,
        "object_class": "person",
        "metadata": {"confidence": 0.91},
    }


def test_event_to_dict_allows_missing_optional_fields() -> None:
    event = Event(
        event_type=EventType.EMPTY_ZONE,
        frame_index=20,
        timestamp_sec=2.0,
    )

    data = event_to_dict(event)

    assert data["zone_name"] is None
    assert data["track_id"] is None
    assert data["object_class"] is None


def test_events_to_dicts_serializes_multiple_events() -> None:
    events = [make_event(), make_event()]

    data = events_to_dicts(events)

    assert len(data) == 2
    assert data[0]["event_type"] == "enter_zone"
    assert data[1]["track_id"] == 7


def test_write_events_json_creates_parent_directories(tmp_path) -> None:
    output_path = tmp_path / "reports" / "events.json"

    written_path = write_events_json([make_event()], output_path)

    assert written_path == output_path
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["event_type"] == "enter_zone"
    assert data[0]["object_class"] == "person"

