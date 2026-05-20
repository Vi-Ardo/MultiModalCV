"""JSON video metadata report helpers."""

import json
from pathlib import Path
from typing import Any

from multimodalcv.video.reader import VideoMetadata


def video_metadata_to_dict(metadata: VideoMetadata) -> dict[str, Any]:
    """Convert video metadata to a JSON-serializable dictionary."""
    return {
        "path": str(metadata.path),
        "frame_count": metadata.frame_count,
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "duration_sec": metadata.duration_sec,
    }


def write_video_metadata_json(metadata: VideoMetadata, output_path: str | Path) -> Path:
    """Write video metadata to a JSON file and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = video_metadata_to_dict(metadata)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path

