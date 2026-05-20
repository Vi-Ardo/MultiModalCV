from multimodalcv.core.models import BoundingBox, Detection, ObjectClass
from multimodalcv.tracking.iou import IoUTracker, bbox_center_distance, bbox_iou


def make_bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def make_detection(
    *,
    frame_index: int,
    bbox: BoundingBox,
    object_class: ObjectClass = ObjectClass.PERSON,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=object_class,
        confidence=0.9,
        bbox=bbox,
    )


def test_bbox_iou_for_overlapping_boxes() -> None:
    first = make_bbox(0, 0, 10, 10)
    second = make_bbox(5, 5, 15, 15)

    assert round(bbox_iou(first, second), 2) == 0.14


def test_bbox_iou_for_non_overlapping_boxes() -> None:
    first = make_bbox(0, 0, 10, 10)
    second = make_bbox(20, 20, 30, 30)

    assert bbox_iou(first, second) == 0.0


def test_bbox_center_distance() -> None:
    first = make_bbox(0, 0, 10, 10)
    second = make_bbox(3, 4, 13, 14)

    assert bbox_center_distance(first, second) == 5.0


def test_iou_tracker_keeps_track_id_for_overlapping_detection() -> None:
    tracker = IoUTracker(iou_threshold=0.2)

    first_tracks = tracker.update([make_detection(frame_index=1, bbox=make_bbox(0, 0, 20, 20))])
    second_tracks = tracker.update([make_detection(frame_index=2, bbox=make_bbox(2, 2, 22, 22))])

    assert first_tracks[0].track_id == 1
    assert second_tracks[0].track_id == 1


def test_iou_tracker_creates_new_track_for_distant_detection() -> None:
    tracker = IoUTracker(iou_threshold=0.2, center_distance_threshold=30)

    first_tracks = tracker.update([make_detection(frame_index=1, bbox=make_bbox(0, 0, 20, 20))])
    second_tracks = tracker.update([make_detection(frame_index=2, bbox=make_bbox(100, 100, 120, 120))])

    assert first_tracks[0].track_id == 1
    assert second_tracks[0].track_id == 2


def test_iou_tracker_matches_by_center_distance_when_iou_is_low() -> None:
    tracker = IoUTracker(iou_threshold=0.9, center_distance_threshold=30)

    first_tracks = tracker.update([make_detection(frame_index=1, bbox=make_bbox(0, 0, 20, 20))])
    second_tracks = tracker.update([make_detection(frame_index=2, bbox=make_bbox(15, 0, 35, 20))])

    assert first_tracks[0].track_id == 1
    assert second_tracks[0].track_id == 1


def test_iou_tracker_does_not_match_different_object_classes() -> None:
    tracker = IoUTracker(iou_threshold=0.2)

    person_tracks = tracker.update(
        [make_detection(frame_index=1, bbox=make_bbox(0, 0, 20, 20), object_class=ObjectClass.PERSON)]
    )
    car_tracks = tracker.update(
        [make_detection(frame_index=2, bbox=make_bbox(2, 2, 22, 22), object_class=ObjectClass.CAR)]
    )

    assert person_tracks[0].track_id == 1
    assert car_tracks[0].track_id == 2


def test_iou_tracker_removes_stale_tracks_after_missed_frames() -> None:
    tracker = IoUTracker(iou_threshold=0.2, max_missed_frames=1)

    first_tracks = tracker.update([make_detection(frame_index=1, bbox=make_bbox(0, 0, 20, 20))])
    tracker.update([])
    tracker.update([])
    new_tracks = tracker.update([make_detection(frame_index=4, bbox=make_bbox(2, 2, 22, 22))])

    assert first_tracks[0].track_id == 1
    assert new_tracks[0].track_id == 2
