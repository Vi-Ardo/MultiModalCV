from multimodalcv.core.models import BoundingBox, EventType, ObjectClass, Track, Zone
from multimodalcv.rules.occupancy import (
    build_count_event,
    count_tracks_in_zone,
    detect_empty_zone_event,
)


def make_track(
    *,
    track_id: int = 1,
    object_class: ObjectClass = ObjectClass.PERSON,
    center_x: float,
    center_y: float,
    frame_index: int = 1,
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


def test_counts_tracks_inside_zone() -> None:
    tracks = [
        make_track(track_id=1, center_x=50, center_y=50),
        make_track(track_id=2, center_x=80, center_y=80),
        make_track(track_id=3, center_x=150, center_y=50),
    ]

    count = count_tracks_in_zone(zone=make_zone(), tracks=tracks, object_class=ObjectClass.PERSON)

    assert count == 2


def test_count_filters_by_object_class() -> None:
    tracks = [
        make_track(track_id=1, object_class=ObjectClass.PERSON, center_x=50, center_y=50),
        make_track(track_id=2, object_class=ObjectClass.CAR, center_x=80, center_y=80),
    ]

    count = count_tracks_in_zone(zone=make_zone(), tracks=tracks, object_class=ObjectClass.PERSON)

    assert count == 1


def test_build_count_event() -> None:
    tracks = [
        make_track(track_id=1, center_x=50, center_y=50, frame_index=5),
        make_track(track_id=2, center_x=150, center_y=50, frame_index=5),
    ]

    event = build_count_event(
        zone=make_zone(),
        tracks=tracks,
        frame_index=5,
        timestamp_sec=0.5,
        object_class=ObjectClass.PERSON,
    )

    assert event.event_type == EventType.COUNT_IN_ZONE
    assert event.frame_index == 5
    assert event.timestamp_sec == 0.5
    assert event.zone_name == "room"
    assert event.object_class == ObjectClass.PERSON
    assert event.metadata["count"] == 1


def test_detects_empty_zone_event() -> None:
    previous = [make_track(track_id=1, center_x=50, center_y=50, frame_index=1)]
    current = [make_track(track_id=1, center_x=150, center_y=50, frame_index=2)]

    event = detect_empty_zone_event(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert event is not None
    assert event.event_type == EventType.EMPTY_ZONE
    assert event.frame_index == 2
    assert event.timestamp_sec == 0.2
    assert event.metadata["previous_count"] == 1
    assert event.metadata["current_count"] == 0


def test_empty_zone_does_not_fire_when_zone_was_already_empty() -> None:
    previous = [make_track(track_id=1, center_x=150, center_y=50, frame_index=1)]
    current = [make_track(track_id=1, center_x=160, center_y=50, frame_index=2)]

    event = detect_empty_zone_event(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert event is None


def test_empty_zone_does_not_fire_when_someone_remains_inside() -> None:
    previous = [
        make_track(track_id=1, center_x=50, center_y=50, frame_index=1),
        make_track(track_id=2, center_x=70, center_y=70, frame_index=1),
    ]
    current = [
        make_track(track_id=1, center_x=150, center_y=50, frame_index=2),
        make_track(track_id=2, center_x=70, center_y=70, frame_index=2),
    ]

    event = detect_empty_zone_event(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert event is None


def test_empty_zone_filter_respects_object_class() -> None:
    previous = [
        make_track(track_id=1, object_class=ObjectClass.CAR, center_x=50, center_y=50, frame_index=1),
        make_track(track_id=2, object_class=ObjectClass.PERSON, center_x=150, center_y=50, frame_index=1),
    ]
    current = [
        make_track(track_id=1, object_class=ObjectClass.CAR, center_x=150, center_y=50, frame_index=2),
        make_track(track_id=2, object_class=ObjectClass.PERSON, center_x=160, center_y=50, frame_index=2),
    ]

    event = detect_empty_zone_event(
        zone=make_zone(),
        previous_tracks=previous,
        current_tracks=current,
        object_class=ObjectClass.PERSON,
    )

    assert event is None

