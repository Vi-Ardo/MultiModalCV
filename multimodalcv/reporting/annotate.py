"""Frame annotation helpers for visual diagnostics."""

from pathlib import Path

import cv2
import numpy as np

from multimodalcv.core.models import BoundingBox, Detection, Track, Zone
from multimodalcv.pipeline.analysis import FrameAnalysis
from multimodalcv.reporting.frame_writer import write_frame_image
from multimodalcv.video.reader import VideoFrame


ZONE_COLOR = (0, 215, 255)
DETECTION_COLOR = (70, 170, 255)
TRACK_COLOR = (80, 220, 80)
TEXT_COLOR = (255, 255, 255)
CENTER_COLOR = (0, 0, 255)


def annotate_frame(
    analysis: FrameAnalysis,
    *,
    zone: Zone | None = None,
) -> VideoFrame:
    """Return a copy of the frame image with visual annotations."""
    image = analysis.frame.image.copy()

    if zone is not None:
        draw_zone(image, zone)

    for detection in analysis.detections:
        draw_detection(image, detection)

    for track in analysis.tracks:
        draw_track(image, track)

    return VideoFrame(
        frame_index=analysis.frame.frame_index,
        timestamp_sec=analysis.frame.timestamp_sec,
        image=image,
    )


def write_annotated_frames(
    analyses: list[FrameAnalysis],
    output_dir: str | Path,
    *,
    zone: Zone | None = None,
) -> list[Path]:
    """Write annotated frame images for a sequence of frame analyses."""
    return [
        write_frame_image(
            annotate_frame(analysis, zone=zone),
            output_dir,
            prefix="annotated",
        )
        for analysis in analyses
    ]


def draw_zone(image: np.ndarray, zone: Zone) -> None:
    points = np.array(zone.points, dtype=np.int32)
    if len(points) < 2:
        return

    cv2.polylines(image, [points], isClosed=True, color=ZONE_COLOR, thickness=2)
    first_x, first_y = points[0]
    draw_label(image, zone.name, int(first_x), int(first_y) - 6, ZONE_COLOR)


def draw_detection(image: np.ndarray, detection: Detection) -> None:
    draw_bbox(image, detection.bbox, DETECTION_COLOR, thickness=1)
    label = f"{detection.object_class.value} {detection.confidence:.2f}"
    draw_label(image, label, int(detection.bbox.x1), int(detection.bbox.y1) - 6, DETECTION_COLOR)


def draw_track(image: np.ndarray, track: Track) -> None:
    draw_bbox(image, track.bbox, TRACK_COLOR, thickness=2)
    center_x, center_y = track.bbox.center
    cv2.circle(image, (int(center_x), int(center_y)), radius=3, color=CENTER_COLOR, thickness=-1)
    label = f"#{track.track_id} {track.object_class.value}"
    draw_label(image, label, int(track.bbox.x1), int(track.bbox.y2) + 14, TRACK_COLOR)


def draw_bbox(image: np.ndarray, bbox: BoundingBox, color: tuple[int, int, int], *, thickness: int) -> None:
    cv2.rectangle(
        image,
        (int(bbox.x1), int(bbox.y1)),
        (int(bbox.x2), int(bbox.y2)),
        color,
        thickness,
    )


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(12, y)
    cv2.putText(
        image,
        text,
        (max(0, x), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

