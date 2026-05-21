from pathlib import Path

from interfaces.streamlit_app.app import (
    center_zone_rect,
    find_frame_by_name,
    format_zone_rect,
    full_frame_zone_rect,
    zone_rect_values,
)
from multimodalcv.video.reader import VideoMetadata


def make_metadata(*, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        path=Path("sample.mp4"),
        frame_count=100,
        fps=25.0,
        width=width,
        height=height,
        duration_sec=4.0,
    )


def test_full_frame_zone_rect_uses_video_size() -> None:
    assert full_frame_zone_rect(make_metadata(width=1920, height=1080)) == "0,0,1920,1080"


def test_center_zone_rect_uses_center_half_of_video() -> None:
    assert center_zone_rect(make_metadata(width=1920, height=1080)) == "480,270,1440,810"


def test_format_zone_rect() -> None:
    assert format_zone_rect(1, 2, 3, 4) == "1,2,3,4"


def test_zone_rect_values() -> None:
    assert zone_rect_values("1,2,3,4") == (1, 2, 3, 4)


def test_find_frame_by_name() -> None:
    paths = [
        Path("frames/annotated_000001.jpg"),
        Path("frames/annotated_000002.jpg"),
    ]

    assert find_frame_by_name(paths, "annotated_000002.jpg") == paths[1]
