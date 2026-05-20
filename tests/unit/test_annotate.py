import cv2
import numpy as np

from multimodalcv.core.models import BoundingBox, Detection, ObjectClass, Track, Zone
from multimodalcv.pipeline.analysis import FrameAnalysis
from multimodalcv.reporting.annotate import annotate_frame, write_annotated_frames
from multimodalcv.video.reader import VideoFrame


def make_analysis() -> FrameAnalysis:
    frame = VideoFrame(
        frame_index=3,
        timestamp_sec=0.3,
        image=np.zeros((120, 160, 3), dtype=np.uint8),
    )
    bbox = BoundingBox(x1=40, y1=30, x2=80, y2=90)
    detection = Detection(
        frame_index=3,
        timestamp_sec=0.3,
        object_class=ObjectClass.PERSON,
        confidence=0.9,
        bbox=bbox,
    )
    track = Track(
        track_id=7,
        frame_index=3,
        timestamp_sec=0.3,
        object_class=ObjectClass.PERSON,
        bbox=bbox,
    )
    return FrameAnalysis(frame=frame, detections=[detection], tracks=[track], events=[])


def make_zone() -> Zone:
    return Zone(name="main", points=((10, 10), (100, 10), (100, 100), (10, 100)))


def test_annotate_frame_returns_copy_with_drawn_pixels() -> None:
    analysis = make_analysis()

    annotated = annotate_frame(analysis, zone=make_zone())

    assert annotated.frame_index == 3
    assert annotated.timestamp_sec == 0.3
    assert annotated.image.shape == analysis.frame.image.shape
    assert int(annotated.image.sum()) > 0
    assert int(analysis.frame.image.sum()) == 0


def test_write_annotated_frames_creates_images(tmp_path) -> None:
    output_paths = write_annotated_frames([make_analysis()], tmp_path, zone=make_zone())

    assert output_paths == [tmp_path / "annotated_000003.jpg"]
    assert output_paths[0].exists()

    image = cv2.imread(str(output_paths[0]))
    assert image.shape == (120, 160, 3)

