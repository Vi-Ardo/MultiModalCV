"""Zone occupancy rules."""

from collections.abc import Iterable

from multimodalcv.core.models import Event, EventType, ObjectClass, Track, Zone


def count_tracks_in_zone(
    *,
    zone: Zone,
    tracks: Iterable[Track],
    object_class: ObjectClass | None = None,
) -> int:
    """Count tracks whose bounding-box center is inside the zone."""
    count = 0

    for track in tracks:
        if object_class is not None and track.object_class != object_class:
            continue

        if zone.contains_bbox_center(track.bbox):
            count += 1

    return count


def build_count_event(
    *,
    zone: Zone,
    tracks: Iterable[Track],
    frame_index: int,
    timestamp_sec: float,
    object_class: ObjectClass | None = None,
) -> Event:
    """Create an event that stores the current zone occupancy count."""
    count = count_tracks_in_zone(zone=zone, tracks=tracks, object_class=object_class)

    return Event(
        event_type=EventType.COUNT_IN_ZONE,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        zone_name=zone.name,
        object_class=object_class,
        metadata={"count": count},
    )


def detect_empty_zone_event(
    *,
    zone: Zone,
    previous_tracks: Iterable[Track],
    current_tracks: Iterable[Track],
    object_class: ObjectClass | None = None,
) -> Event | None:
    """Return an empty-zone event when occupancy changes from non-empty to empty."""
    previous_tracks_list = list(previous_tracks)
    current_tracks_list = list(current_tracks)

    previous_count = count_tracks_in_zone(
        zone=zone,
        tracks=previous_tracks_list,
        object_class=object_class,
    )
    current_count = count_tracks_in_zone(
        zone=zone,
        tracks=current_tracks_list,
        object_class=object_class,
    )

    if previous_count == 0 or current_count != 0:
        return None

    frame_index, timestamp_sec = _event_time_from_tracks(current_tracks_list, previous_tracks_list)

    return Event(
        event_type=EventType.EMPTY_ZONE,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        zone_name=zone.name,
        object_class=object_class,
        metadata={
            "previous_count": previous_count,
            "current_count": current_count,
        },
    )


def _event_time_from_tracks(
    current_tracks: list[Track],
    previous_tracks: list[Track],
) -> tuple[int, float]:
    if current_tracks:
        track = current_tracks[0]
        return track.frame_index, track.timestamp_sec

    if previous_tracks:
        track = previous_tracks[0]
        return track.frame_index, track.timestamp_sec

    return 0, 0.0

