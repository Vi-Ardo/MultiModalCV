"""Command-driven frame analysis pipeline."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import Detection, Event, Track, Zone
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
) -> AnalysisResult:
    """Analyze decoded frames with detector, tracker, and rule engine."""
    previous_tracks: list[Track] = []
    known_track_ids: set[int] = set()
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
        frame_results.append(
            FrameAnalysis(
                frame=frame,
                detections=detections,
                tracks=current_tracks,
                events=events,
            )
        )
        previous_tracks = current_tracks
        known_track_ids.update(track.track_id for track in current_tracks)

    return AnalysisResult(frames=frame_results)
