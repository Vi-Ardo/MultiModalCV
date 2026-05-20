"""CLI for basic archived video inspection and frame extraction."""

import argparse
from pathlib import Path

from multimodalcv.reporting.frame_writer import write_frame_images
from multimodalcv.reporting.video_report import write_video_metadata_json
from multimodalcv.video.reader import VideoOpenError, iter_video_frames, read_video_metadata


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = inspect_video(
            video_path=args.video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            every_n_frames=args.every_n_frames,
        )
    except (VideoOpenError, ValueError) as error:
        print(error)
        return 2

    print(f"Video metadata written to {result.metadata_path}")
    print(f"Saved {len(result.frame_paths)} frame image(s) to {result.frames_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an archived video file and extract sample frames.",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to an archived video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/video_inspect"),
        help="Directory for metadata and extracted frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of frames to save.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="Save every Nth frame.",
    )
    return parser


class VideoInspectResult:
    """Output paths produced by the video inspection CLI."""

    def __init__(self, *, metadata_path: Path, frames_dir: Path, frame_paths: list[Path]) -> None:
        self.metadata_path = metadata_path
        self.frames_dir = frames_dir
        self.frame_paths = frame_paths


def inspect_video(
    *,
    video_path: Path,
    output_dir: Path,
    max_frames: int = 5,
    every_n_frames: int = 1,
) -> VideoInspectResult:
    """Read video metadata and save a small sample of decoded frames."""
    if max_frames < 0:
        raise ValueError("max_frames must be >= 0")

    metadata = read_video_metadata(video_path)
    metadata_path = write_video_metadata_json(metadata, output_dir / "metadata.json")

    frames_dir = output_dir / "frames"
    frames = list(
        iter_video_frames(
            video_path,
            every_n_frames=every_n_frames,
            max_frames=max_frames,
        )
    )
    frame_paths = write_frame_images(frames, frames_dir)

    return VideoInspectResult(
        metadata_path=metadata_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
    )


if __name__ == "__main__":
    raise SystemExit(main())

