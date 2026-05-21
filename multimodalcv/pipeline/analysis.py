"""Command-driven frame analysis pipeline."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import median

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import Detection, Event, EventType, ObjectClass, Track, Zone
from multimodalcv.detection.base import ObjectDetector
from multimodalcv.rules.engine import evaluate_rule
from multimodalcv.tracking.base import ObjectTracker
from multimodalcv.video.reader import VideoFrame


@dataclass(frozen=True)
class FrameAnalysis:
    """Analysis output for a single frame."""

    frame: VideoFrame
    detections: list[Detection]
    tracks: list[Track]
    events: list[Event]


@dataclass(frozen=True)
class AnalysisResult:
    """Output of a complete frame sequence analysis."""

    frames: list[FrameAnalysis] = field(default_factory=list)

    @property
    def events(self) -> list[Event]:
        return [event for frame in self.frames for event in frame.events]


def analyze_frames(
    *,
    frames: Iterable[VideoFrame],
    detector: ObjectDetector,
    tracker: ObjectTracker,
    rule: CommandRule,
    zone: Zone,
    count_window_size: int = 1,
    event_cooldown_sec: float = 0.0,
) -> AnalysisResult:
    """Analyze decoded frames with detector, tracker, and rule engine."""
    if count_window_size < 1:
        raise ValueError("count_window_size must be >= 1")
    if event_cooldown_sec < 0:
        raise ValueError("event_cooldown_sec must be >= 0")

    previous_tracks: list[Track] = []
    known_track_ids: set[int] = set()
    count_history: dict[tuple[EventType, ObjectClass | None, str | None], list[int]] = {}
    last_event_times: dict[tuple[EventType, ObjectClass | None, str | None], float] = {}
    frame_results: list[FrameAnalysis] = []

    for frame in frames:
        detections = detector.detect(frame)
        current_tracks = tracker.update(detections)
        events = evaluate_rule(
            rule=rule,
            zone=zone,
            previous_tracks=previous_tracks,
            current_tracks=current_tracks,
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            known_track_ids=known_track_ids,
        )
        processed_events = process_frame_events(
            events,
            count_history=count_history,
            count_window_size=count_window_size,
            last_event_times=last_event_times,
            event_cooldown_sec=event_cooldown_sec,
        )
        frame_results.append(
            FrameAnalysis(
                frame=frame,
                detections=detections,
                tracks=current_tracks,
                events=processed_events,
            )
        )
        previous_tracks = current_tracks
        known_track_ids.update(track.track_id for track in current_tracks)

    return AnalysisResult(frames=frame_results)


def process_frame_events(
    events: list[Event],
    *,
    count_history: dict[tuple[EventType, ObjectClass | None, str | None], list[int]],
    count_window_size: int,
    last_event_times: dict[tuple[EventType, ObjectClass | None, str | None], float],
    event_cooldown_sec: float,
) -> list[Event]:
    processed_events: list[Event] = []

    for event in events:
        smoothed_event = smooth_count_event(
            event,
            count_history=count_history,
            count_window_size=count_window_size,
        )

        if is_suppressed_by_cooldown(
            smoothed_event,
            last_event_times=last_event_times,
            event_cooldown_sec=event_cooldown_sec,
        ):
            continue

        processed_events.append(smoothed_event)

    return processed_events


def smooth_count_event(
    event: Event,
    *,
    count_history: dict[tuple[EventType, ObjectClass | None, str | None], list[int]],
    count_window_size: int,
) -> Event:
    if event.event_type not in {EventType.COUNT_IN_FRAME, EventType.COUNT_IN_ZONE}:
        return event

    raw_count = int(event.metadata.get("count", 0))
    key = event_key(event)
    history = count_history.setdefault(key, [])
    history.append(raw_count)
    if len(history) > count_window_size:
        del history[:-count_window_size]

    if count_window_size == 1:
        return event

    smoothed_count = int(round(median(history)))
    metadata = dict(event.metadata)
    metadata["raw_count"] = raw_count
    metadata["count"] = smoothed_count

    return Event(
        event_type=event.event_type,
        frame_index=event.frame_index,
        timestamp_sec=event.timestamp_sec,
        zone_name=event.zone_name,
        track_id=event.track_id,
        object_class=event.object_class,
        metadata=metadata,
    )


def is_suppressed_by_cooldown(
    event: Event,
    *,
    last_event_times: dict[tuple[EventType, ObjectClass | None, str | None], float],
    event_cooldown_sec: float,
) -> bool:
    if event_cooldown_sec <= 0 or event.event_type in {EventType.COUNT_IN_FRAME, EventType.COUNT_IN_ZONE}:
        return False

    key = event_key(event)
    last_timestamp = last_event_times.get(key)
    if last_timestamp is not None and event.timestamp_sec - last_timestamp < event_cooldown_sec:
        return True

    last_event_times[key] = event.timestamp_sec
    return False


def event_key(event: Event) -> tuple[EventType, ObjectClass | None, str | None]:
    return event.event_type, event.object_class, event.zone_name
