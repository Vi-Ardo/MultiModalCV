"""Frame image output helpers."""

from pathlib import Path

import cv2

from multimodalcv.video.reader import VideoFrame


class FrameWriteError(ValueError):
    """Raised when a frame image cannot be written."""


def frame_filename(frame: VideoFrame, *, prefix: str = "frame", extension: str = "jpg") -> str:
    """Build a stable filename for a video frame."""
    normalized_extension = extension.lstrip(".")
    return f"{prefix}_{frame.frame_index:06d}.{normalized_extension}"


def write_frame_image(
    frame: VideoFrame,
    output_dir: str | Path,
    *,
    prefix: str = "frame",
    extension: str = "jpg",
) -> Path:
    """Write a decoded video frame as an image file."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / frame_filename(frame, prefix=prefix, extension=extension)
    try:
        success = cv2.imwrite(str(output_path), frame.image)
    except cv2.error as error:
        raise FrameWriteError(f"Cannot write frame image: {output_path}") from error

    if not success:
        raise FrameWriteError(f"Cannot write frame image: {output_path}")

    return output_path


def write_frame_images(
    frames: list[VideoFrame],
    output_dir: str | Path,
    *,
    prefix: str = "frame",
    extension: str = "jpg",
) -> list[Path]:
    """Write multiple frames as image files."""
    return [
        write_frame_image(
            frame,
            output_dir,
            prefix=prefix,
            extension=extension,
        )
        for frame in frames
    ]
