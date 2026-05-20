"""Rule orchestration for command-driven event evaluation."""

from collections.abc import Iterable

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import Event, EventType, Track, Zone
from multimodalcv.rules.occupancy import build_count_event, detect_empty_zone_event
from multimodalcv.rules.zone_events import detect_zone_transitions


def evaluate_rule(
    *,
    rule: CommandRule,
    zone: Zone,
    previous_tracks: Iterable[Track],
    current_tracks: Iterable[Track],
) -> list[Event]:
    """Evaluate a command rule against two scene states."""
    previous_tracks_list = list(previous_tracks)
    current_tracks_list = list(current_tracks)

    if rule.event_type in {EventType.ENTER_ZONE, EventType.LEAVE_ZONE}:
        transition_events = detect_zone_transitions(
            zone=zone,
            previous_tracks=previous_tracks_list,
            current_tracks=current_tracks_list,
            object_class=rule.object_class,
        )
        return [
            event
            for event in transition_events
            if event.event_type == rule.event_type
        ]

    if rule.event_type == EventType.EMPTY_ZONE:
        event = detect_empty_zone_event(
            zone=zone,
            previous_tracks=previous_tracks_list,
            current_tracks=current_tracks_list,
            object_class=rule.object_class,
        )
        return [] if event is None else [event]

    if rule.event_type == EventType.COUNT_IN_ZONE:
        frame_index, timestamp_sec = _current_time(current_tracks_list, previous_tracks_list)
        return [
            build_count_event(
                zone=zone,
                tracks=current_tracks_list,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                object_class=rule.object_class,
            )
        ]

    if rule.event_type == EventType.TRACK_OBJECT:
        return [
            Event(
                event_type=EventType.TRACK_OBJECT,
                frame_index=track.frame_index,
                timestamp_sec=track.timestamp_sec,
                zone_name=zone.name,
                track_id=track.track_id,
                object_class=track.object_class,
            )
            for track in current_tracks_list
            if track.object_class == rule.object_class
        ]

    return []


def _current_time(
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

