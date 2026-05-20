from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import BoundingBox, EventType, ObjectClass, Track, Zone
from multimodalcv.rules.engine import evaluate_rule


def make_track(
    *,
    track_id: int = 1,
    object_class: ObjectClass = ObjectClass.PERSON,
    center_x: float,
    center_y: float,
    frame_index: int,
) -> Track:
    return Track(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=object_class,
        bbox=BoundingBox(
            x1=center_x - 5,
            y1=center_y - 5,
            x2=center_x + 5,
            y2=center_y + 5,
        ),
    )


def make_zone() -> Zone:
    return Zone(name="main", points=((0, 0), (100, 0), (100, 100), (0, 100)))


def make_rule(event_type: EventType, object_class: ObjectClass = ObjectClass.PERSON) -> CommandRule:
    return CommandRule(
        event_type=event_type,
        object_class=object_class,
        raw_text="test command",
        zone_name="main",
    )


def test_evaluates_enter_zone_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
        previous_tracks=[make_track(center_x=150, center_y=50, frame_index=1)],
        current_tracks=[make_track(center_x=50, center_y=50, frame_index=2)],
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ENTER_ZONE


def test_evaluates_leave_zone_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.LEAVE_ZONE),
        zone=make_zone(),
        previous_tracks=[make_track(center_x=50, center_y=50, frame_index=1)],
        current_tracks=[make_track(center_x=150, center_y=50, frame_index=2)],
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.LEAVE_ZONE


def test_enter_rule_does_not_return_leave_events() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
        previous_tracks=[make_track(center_x=50, center_y=50, frame_index=1)],
        current_tracks=[make_track(center_x=150, center_y=50, frame_index=2)],
    )

    assert events == []


def test_evaluates_empty_zone_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.EMPTY_ZONE),
        zone=make_zone(),
        previous_tracks=[make_track(center_x=50, center_y=50, frame_index=1)],
        current_tracks=[make_track(center_x=150, center_y=50, frame_index=2)],
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.EMPTY_ZONE


def test_evaluates_count_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.COUNT_IN_ZONE),
        zone=make_zone(),
        previous_tracks=[],
        current_tracks=[
            make_track(track_id=1, center_x=50, center_y=50, frame_index=2),
            make_track(track_id=2, center_x=150, center_y=50, frame_index=2),
        ],
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.COUNT_IN_ZONE
    assert events[0].metadata["count"] == 1


def test_evaluates_frame_count_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.COUNT_IN_FRAME),
        zone=make_zone(),
        previous_tracks=[],
        current_tracks=[
            make_track(track_id=1, center_x=50, center_y=50, frame_index=2),
            make_track(track_id=2, object_class=ObjectClass.CAR, center_x=150, center_y=50, frame_index=2),
        ],
        frame_index=2,
        timestamp_sec=0.2,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.COUNT_IN_FRAME
    assert events[0].metadata["count"] == 1
    assert events[0].zone_name is None


def test_evaluates_track_object_rule() -> None:
    events = evaluate_rule(
        rule=make_rule(EventType.TRACK_OBJECT, object_class=ObjectClass.CAR),
        zone=make_zone(),
        previous_tracks=[],
        current_tracks=[
            make_track(track_id=1, object_class=ObjectClass.PERSON, center_x=50, center_y=50, frame_index=2),
            make_track(track_id=2, object_class=ObjectClass.CAR, center_x=150, center_y=50, frame_index=2),
        ],
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.TRACK_OBJECT
    assert events[0].track_id == 2
    assert events[0].object_class == ObjectClass.CAR
