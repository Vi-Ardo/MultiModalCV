import json

from multimodalcv.reporting.video_report import video_metadata_to_dict, write_video_metadata_json
from multimodalcv.video.reader import VideoMetadata


def make_metadata(tmp_path) -> VideoMetadata:
    return VideoMetadata(
        path=tmp_path / "sample.mp4",
        frame_count=10,
        fps=5.0,
        width=640,
        height=480,
        duration_sec=2.0,
    )


def test_video_metadata_to_dict(tmp_path) -> None:
    metadata = make_metadata(tmp_path)

    data = video_metadata_to_dict(metadata)

    assert data["path"] == str(tmp_path / "sample.mp4")
    assert data["frame_count"] == 10
    assert data["fps"] == 5.0
    assert data["width"] == 640
    assert data["height"] == 480
    assert data["duration_sec"] == 2.0


def test_write_video_metadata_json(tmp_path) -> None:
    output_path = tmp_path / "reports" / "metadata.json"

    written_path = write_video_metadata_json(make_metadata(tmp_path), output_path)

    assert written_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["frame_count"] == 10
    assert data["duration_sec"] == 2.0

