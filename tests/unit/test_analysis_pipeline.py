import numpy as np

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import BoundingBox, Detection, EventType, ObjectClass, Track, Zone
from multimodalcv.detection.fake import FakeDetector
from multimodalcv.pipeline.analysis import analyze_frames
from multimodalcv.tracking.fake import FakeTracker
from multimodalcv.video.reader import VideoFrame


def make_frame(frame_index: int) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
    )


def make_detection(*, frame_index: int, center_x: float, center_y: float) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=ObjectClass.PERSON,
        confidence=0.9,
        bbox=BoundingBox(
            x1=center_x - 5,
            y1=center_y - 5,
            x2=center_x + 5,
            y2=center_y + 5,
        ),
    )


def make_track(*, frame_index: int, center_x: float, center_y: float) -> Track:
    return Track(
        track_id=1,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=ObjectClass.PERSON,
        bbox=BoundingBox(
            x1=center_x - 5,
            y1=center_y - 5,
            x2=center_x + 5,
            y2=center_y + 5,
        ),
    )


def make_zone() -> Zone:
    return Zone(name="main", points=((0, 0), (100, 0), (100, 100), (0, 100)))


def make_rule(event_type: EventType) -> CommandRule:
    return CommandRule(
        event_type=event_type,
        object_class=ObjectClass.PERSON,
        raw_text="test command",
        zone_name="main",
    )


def test_analyze_frames_detects_enter_zone_event() -> None:
    frames = [make_frame(1), make_frame(2)]
    detector = FakeDetector(
        {
            1: [make_detection(frame_index=1, center_x=150, center_y=50)],
            2: [make_detection(frame_index=2, center_x=50, center_y=50)],
        }
    )
    tracker = FakeTracker(
        {
            1: [make_track(frame_index=1, center_x=150, center_y=50)],
            2: [make_track(frame_index=2, center_x=50, center_y=50)],
        }
    )

    result = analyze_frames(
        frames=frames,
        detector=detector,
        tracker=tracker,
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
    )

    assert len(result.frames) == 2
    assert result.frames[0].events == []
    assert len(result.frames[1].events) == 1
    assert result.events[0].event_type == EventType.ENTER_ZONE
    assert result.events[0].track_id == 1


def test_analyze_frames_detects_empty_zone_event() -> None:
    frames = [make_frame(1), make_frame(2)]
    detector = FakeDetector(
        {
            1: [make_detection(frame_index=1, center_x=50, center_y=50)],
            2: [make_detection(frame_index=2, center_x=150, center_y=50)],
        }
    )
    tracker = FakeTracker(
        {
            1: [make_track(frame_index=1, center_x=50, center_y=50)],
            2: [make_track(frame_index=2, center_x=150, center_y=50)],
        }
    )

    result = analyze_frames(
        frames=frames,
        detector=detector,
        tracker=tracker,
        rule=make_rule(EventType.EMPTY_ZONE),
        zone=make_zone(),
    )

    assert len(result.events) == 1
    assert result.events[0].event_type == EventType.EMPTY_ZONE


def test_analyze_frames_keeps_detections_and_tracks_per_frame() -> None:
    frames = [make_frame(1)]
    detection = make_detection(frame_index=1, center_x=50, center_y=50)
    track = make_track(frame_index=1, center_x=50, center_y=50)

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector({1: [detection]}),
        tracker=FakeTracker({1: [track]}),
        rule=make_rule(EventType.COUNT_IN_ZONE),
        zone=make_zone(),
    )

    assert result.frames[0].frame == frames[0]
    assert result.frames[0].detections == [detection]
    assert result.frames[0].tracks == [track]
    assert result.frames[0].events[0].metadata["count"] == 1


def test_analyze_frames_handles_empty_input() -> None:
    result = analyze_frames(
        frames=[],
        detector=FakeDetector({}),
        tracker=FakeTracker({}),
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
    )

    assert result.frames == []
    assert result.events == []

