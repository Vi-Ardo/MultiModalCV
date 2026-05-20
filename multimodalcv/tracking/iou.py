"""Simple IoU-based object tracker."""

from dataclasses import dataclass

from multimodalcv.core.models import BoundingBox, Detection, ObjectClass, Track
from multimodalcv.tracking.base import ObjectTracker


@dataclass
class _ActiveTrack:
    track_id: int
    object_class: ObjectClass
    bbox: BoundingBox
    missed_frames: int = 0


class IoUTracker(ObjectTracker):
    """Small deterministic tracker that matches detections by IoU."""

    def __init__(self, *, iou_threshold: float = 0.3, max_missed_frames: int = 2) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: list[_ActiveTrack] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        matched_track_ids: set[int] = set()
        output_tracks: list[Track] = []

        for detection in detections:
            active_track = self._best_match(detection, matched_track_ids)

            if active_track is None:
                active_track = _ActiveTrack(
                    track_id=self._next_track_id,
                    object_class=detection.object_class,
                    bbox=detection.bbox,
                )
                self._next_track_id += 1
                self._tracks.append(active_track)
            else:
                active_track.bbox = detection.bbox
                active_track.missed_frames = 0

            matched_track_ids.add(active_track.track_id)
            output_tracks.append(
                Track(
                    track_id=active_track.track_id,
                    frame_index=detection.frame_index,
                    timestamp_sec=detection.timestamp_sec,
                    object_class=detection.object_class,
                    bbox=detection.bbox,
                )
            )

        self._age_unmatched_tracks(matched_track_ids)
        return output_tracks

    def _best_match(
        self,
        detection: Detection,
        matched_track_ids: set[int],
    ) -> _ActiveTrack | None:
        candidates = [
            (bbox_iou(track.bbox, detection.bbox), track)
            for track in self._tracks
            if track.object_class == detection.object_class and track.track_id not in matched_track_ids
        ]

        if not candidates:
            return None

        best_iou, best_track = max(candidates, key=lambda item: item[0])
        if best_iou < self._iou_threshold:
            return None

        return best_track

    def _age_unmatched_tracks(self, matched_track_ids: set[int]) -> None:
        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.missed_frames += 1

        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self._max_missed_frames
        ]


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection over union for two bounding boxes."""
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

