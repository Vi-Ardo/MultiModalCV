from multimodalcv.core.models import BoundingBox, Detection, ObjectClass, Track
from multimodalcv.tracking.fake import FakeTracker, SequentialTracker


def make_detection(
    *,
    frame_index: int = 1,
    object_class: ObjectClass = ObjectClass.PERSON,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=object_class,
        confidence=0.9,
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
    )


def make_track(track_id: int = 1, frame_index: int = 1) -> Track:
    return Track(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=ObjectClass.PERSON,
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=20),
    )


def test_fake_tracker_returns_tracks_for_detection_frame() -> None:
    track = make_track(track_id=7, frame_index=3)
    tracker = FakeTracker({3: [track]})

    tracks = tracker.update([make_detection(frame_index=3)])

    assert tracks == [track]


def test_fake_tracker_returns_empty_list_without_detections() -> None:
    tracker = FakeTracker({3: [make_track(frame_index=3)]})

    assert tracker.update([]) == []


def test_fake_tracker_returns_empty_list_for_unknown_frame() -> None:
    tracker = FakeTracker({3: [make_track(frame_index=3)]})

    assert tracker.update([make_detection(frame_index=4)]) == []


def test_fake_tracker_returns_copy_of_track_list() -> None:
    track = make_track(track_id=7, frame_index=3)
    tracker = FakeTracker({3: [track]})

    tracks = tracker.update([make_detection(frame_index=3)])
    tracks.clear()

    assert tracker.update([make_detection(frame_index=3)]) == [track]


def test_sequential_tracker_converts_detections_to_tracks() -> None:
    tracker = SequentialTracker()
    detections = [
        make_detection(frame_index=5, object_class=ObjectClass.PERSON),
        make_detection(frame_index=5, object_class=ObjectClass.CAR),
    ]

    tracks = tracker.update(detections)

    assert [track.track_id for track in tracks] == [1, 2]
    assert [track.object_class for track in tracks] == [ObjectClass.PERSON, ObjectClass.CAR]
    assert [track.frame_index for track in tracks] == [5, 5]
    assert tracks[0].bbox == detections[0].bbox

