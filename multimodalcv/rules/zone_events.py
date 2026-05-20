"""Zone entry and exit event detection."""

from collections.abc import Iterable

from multimodalcv.core.models import Event, EventType, ObjectClass, Track, Zone


def detect_zone_transitions(
    *,
    zone: Zone,
    previous_tracks: Iterable[Track],
    current_tracks: Iterable[Track],
    object_class: ObjectClass | None = None,
    new_tracks_enter_zone: bool = False,
    known_track_ids: set[int] | None = None,
) -> list[Event]:
    """Detect object entry and exit events between two frame states."""
    previous_by_id = {track.track_id: track for track in previous_tracks}
    events: list[Event] = []

    for current_track in current_tracks:
        if object_class is not None and current_track.object_class != object_class:
            continue

        previous_track = previous_by_id.get(current_track.track_id)
        if previous_track is None:
            is_known_track = known_track_ids is not None and current_track.track_id in known_track_ids
            if new_tracks_enter_zone and not is_known_track and zone.contains_bbox_center(current_track.bbox):
                events.append(_build_transition_event(EventType.ENTER_ZONE, zone, current_track))
            continue

        if previous_track.object_class != current_track.object_class:
            continue

        was_inside = zone.contains_bbox_center(previous_track.bbox)
        is_inside = zone.contains_bbox_center(current_track.bbox)

        if not was_inside and is_inside:
            events.append(_build_transition_event(EventType.ENTER_ZONE, zone, current_track))
        elif was_inside and not is_inside:
            events.append(_build_transition_event(EventType.LEAVE_ZONE, zone, current_track))

    return events


def _build_transition_event(event_type: EventType, zone: Zone, track: Track) -> Event:
    return Event(
        event_type=event_type,
        frame_index=track.frame_index,
        timestamp_sec=track.timestamp_sec,
        zone_name=zone.name,
        track_id=track.track_id,
        object_class=track.object_class,
    )
