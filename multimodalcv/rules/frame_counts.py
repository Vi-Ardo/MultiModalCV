"""Frame-level counting rules."""

from collections.abc import Iterable

from multimodalcv.core.models import Event, EventType, ObjectClass, Track


def count_tracks_in_frame(
    *,
    tracks: Iterable[Track],
    object_class: ObjectClass | None = None,
) -> int:
    """Count tracks in the current frame."""
    count = 0

    for track in tracks:
        if object_class is not None and track.object_class != object_class:
            continue

        count += 1

    return count


def build_frame_count_event(
    *,
    tracks: Iterable[Track],
    frame_index: int,
    timestamp_sec: float,
    object_class: ObjectClass | None = None,
) -> Event:
    """Create an event that stores a frame-level object count."""
    count = count_tracks_in_frame(tracks=tracks, object_class=object_class)

    return Event(
        event_type=EventType.COUNT_IN_FRAME,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        object_class=object_class,
        metadata={"count": count},
    )
