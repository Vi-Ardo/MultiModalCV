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


def make_track(*, frame_index: int, center_x: float, center_y: float, track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
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


def test_analyze_frames_count_event_uses_frame_time_without_tracks() -> None:
    frames = [make_frame(3)]

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector({}),
        tracker=FakeTracker({}),
        rule=make_rule(EventType.COUNT_IN_ZONE),
        zone=make_zone(),
    )

    assert len(result.events) == 1
    assert result.events[0].frame_index == 3
    assert result.events[0].timestamp_sec == 0.3
    assert result.events[0].metadata["count"] == 0


def test_analyze_frames_counts_tracks_in_frame_without_zone_membership() -> None:
    frames = [make_frame(1)]
    detection = make_detection(frame_index=1, center_x=150, center_y=150)
    track = make_track(frame_index=1, center_x=150, center_y=150)

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector({1: [detection]}),
        tracker=FakeTracker({1: [track]}),
        rule=make_rule(EventType.COUNT_IN_FRAME),
        zone=make_zone(),
    )

    assert len(result.events) == 1
    assert result.events[0].event_type == EventType.COUNT_IN_FRAME
    assert result.events[0].metadata["count"] == 1


def test_analyze_frames_smooths_count_events_with_median_window() -> None:
    frames = [make_frame(1), make_frame(2), make_frame(3)]
    person_detection = make_detection(frame_index=1, center_x=50, center_y=50)
    person_track = make_track(frame_index=1, center_x=50, center_y=50)
    car_detection = Detection(
        frame_index=2,
        timestamp_sec=0.2,
        object_class=ObjectClass.CAR,
        confidence=0.9,
        bbox=BoundingBox(x1=10, y1=10, x2=20, y2=20),
    )
    car_track = Track(
        track_id=2,
        frame_index=2,
        timestamp_sec=0.2,
        object_class=ObjectClass.CAR,
        bbox=car_detection.bbox,
    )

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector({1: [person_detection], 2: [car_detection], 3: [person_detection]}),
        tracker=FakeTracker({1: [person_track], 2: [car_track], 3: [person_track]}),
        rule=make_rule(EventType.COUNT_IN_FRAME),
        zone=make_zone(),
        count_window_size=3,
    )

    assert [event.metadata["raw_count"] for event in result.events] == [1, 0, 1]
    assert [event.metadata["count"] for event in result.events] == [1, 0, 1]


def test_analyze_frames_suppresses_repeated_enter_events_with_cooldown() -> None:
    frames = [make_frame(1), make_frame(2), make_frame(3)]

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector(
            {
                1: [make_detection(frame_index=1, center_x=50, center_y=50)],
                2: [make_detection(frame_index=2, center_x=150, center_y=50)],
                3: [make_detection(frame_index=3, center_x=50, center_y=50)],
            }
        ),
        tracker=FakeTracker(
            {
                1: [make_track(track_id=1, frame_index=1, center_x=50, center_y=50)],
                2: [make_track(track_id=2, frame_index=2, center_x=150, center_y=50)],
                3: [make_track(track_id=3, frame_index=3, center_x=50, center_y=50)],
            }
        ),
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
        event_cooldown_sec=1.0,
    )

    assert len(result.events) == 1
    assert result.events[0].frame_index == 1


def test_analyze_frames_does_not_reenter_known_track_after_missed_detection() -> None:
    frames = [make_frame(1), make_frame(2), make_frame(3)]
    detection = make_detection(frame_index=1, center_x=50, center_y=50)
    returning_detection = make_detection(frame_index=3, center_x=50, center_y=50)
    track = make_track(frame_index=1, center_x=50, center_y=50)
    returning_track = make_track(frame_index=3, center_x=50, center_y=50)

    result = analyze_frames(
        frames=frames,
        detector=FakeDetector({1: [detection], 3: [returning_detection]}),
        tracker=FakeTracker({1: [track], 3: [returning_track]}),
        rule=make_rule(EventType.ENTER_ZONE),
        zone=make_zone(),
    )

    assert len(result.events) == 1
    assert result.events[0].frame_index == 1


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
