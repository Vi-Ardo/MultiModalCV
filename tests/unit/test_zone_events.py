from multimodalcv.core.models import BoundingBox, EventType, ObjectClass, Track, Zone
from multimodalcv.rules.zone_events import detect_zone_transitions


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
    return Zone(name="room", points=((0, 0), (100, 0), (100, 100), (0, 100)))


def test_detects_enter_zone_event() -> None:
    previous = [make_track(center_x=150, center_y=50, frame_index=1)]
    current = [make_track(center_x=50, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ENTER_ZONE
    assert events[0].track_id == 1
    assert events[0].zone_name == "room"
    assert events[0].object_class == ObjectClass.PERSON


def test_detects_leave_zone_event() -> None:
    previous = [make_track(center_x=50, center_y=50, frame_index=1)]
    current = [make_track(center_x=150, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.LEAVE_ZONE


def test_ignores_track_that_stays_inside_zone() -> None:
    previous = [make_track(center_x=40, center_y=50, frame_index=1)]
    current = [make_track(center_x=60, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert events == []


def test_ignores_track_that_stays_outside_zone() -> None:
    previous = [make_track(center_x=140, center_y=50, frame_index=1)]
    current = [make_track(center_x=160, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert events == []


def test_filters_by_object_class() -> None:
    previous = [make_track(object_class=ObjectClass.CAR, center_x=150, center_y=50, frame_index=1)]
    current = [make_track(object_class=ObjectClass.CAR, center_x=50, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert events == []


def test_detects_multiple_transitions() -> None:
    previous = [
        make_track(track_id=1, center_x=150, center_y=50, frame_index=1),
        make_track(track_id=2, center_x=50, center_y=50, frame_index=1),
        make_track(track_id=3, center_x=150, center_y=150, frame_index=1),
    ]
    current = [
        make_track(track_id=1, center_x=50, center_y=50, frame_index=2),
        make_track(track_id=2, center_x=150, center_y=50, frame_index=2),
        make_track(track_id=3, center_x=160, center_y=160, frame_index=2),
    ]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert [event.event_type for event in events] == [
        EventType.ENTER_ZONE,
        EventType.LEAVE_ZONE,
    ]
    assert [event.track_id for event in events] == [1, 2]


def test_ignores_new_track_without_previous_state() -> None:
    current = [make_track(center_x=50, center_y=50, frame_index=2)]

    events = detect_zone_transitions(
        zone=make_zone(),
        previous_tracks=[],
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert events == []

