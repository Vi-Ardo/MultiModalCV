from multimodalcv.core.models import BoundingBox, EventType, ObjectClass, Track
from multimodalcv.rules.frame_counts import build_frame_count_event, count_tracks_in_frame


def make_track(
    *,
    track_id: int = 1,
    object_class: ObjectClass = ObjectClass.PERSON,
    frame_index: int = 5,
) -> Track:
    return Track(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=object_class,
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
    )


def test_counts_tracks_in_frame() -> None:
    tracks = [
        make_track(track_id=1),
        make_track(track_id=2),
        make_track(track_id=3, object_class=ObjectClass.CAR),
    ]

    assert count_tracks_in_frame(tracks=tracks, object_class=ObjectClass.PERSON) == 2


def test_build_frame_count_event() -> None:
    tracks = [make_track(track_id=1), make_track(track_id=2, object_class=ObjectClass.CAR)]

    event = build_frame_count_event(
        tracks=tracks,
        frame_index=5,
        timestamp_sec=0.5,
        object_class=ObjectClass.PERSON,
    )

    assert event.event_type == EventType.COUNT_IN_FRAME
    assert event.frame_index == 5
    assert event.timestamp_sec == 0.5
    assert event.object_class == ObjectClass.PERSON
    assert event.zone_name is None
    assert event.metadata["count"] == 1

