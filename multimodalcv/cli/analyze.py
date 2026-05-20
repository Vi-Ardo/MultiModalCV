"""CLI for command-driven video analysis."""

import argparse
from pathlib import Path

from multimodalcv.commands.parser import UnsupportedCommandError, parse_command, supported_commands
from multimodalcv.core.models import BoundingBox, Detection, Event, EventType, ObjectClass, Track, Zone
from multimodalcv.detection.base import ObjectDetector
from multimodalcv.detection.fake import FakeDetector
from multimodalcv.detection.yolo import YOLODependencyError, YOLODetector
from multimodalcv.pipeline.analysis import analyze_frames
from multimodalcv.reporting.annotate import write_annotated_frames
from multimodalcv.reporting.json_report import write_events_json
from multimodalcv.tracking.base import ObjectTracker
from multimodalcv.tracking.fake import FakeTracker
from multimodalcv.tracking.iou import IoUTracker
from multimodalcv.video.reader import VideoOpenError, VideoFrame, iter_video_frames


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        events = analyze_video(
            video_path=args.video,
            command=args.command,
            output_path=args.output,
            max_frames=args.max_frames,
            detector_name=args.detector,
            model_path=args.model,
            confidence_threshold=args.confidence,
            save_frames=args.save_frames,
            frames_dir=args.frames_dir,
        )
    except UnsupportedCommandError as error:
        print(error)
        print("Supported commands:")
        for command in supported_commands():
            print(f"- {command}")
        return 2
    except (VideoOpenError, ValueError, YOLODependencyError) as error:
        print(error)
        return 2

    print(f"Wrote {len(events)} event(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze an archived video with a restricted text command.",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to an archived video file.",
    )
    parser.add_argument(
        "command",
        help="Supported Russian command.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analyze/events.json"),
        help="Path to the JSON event report.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2,
        help="Maximum number of frames to analyze.",
    )
    parser.add_argument(
        "--detector",
        choices=("fake", "yolo"),
        default="fake",
        help="Detector backend to use.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n.pt"),
        help="YOLO model path or model name when --detector yolo is used.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold when --detector yolo is used.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frame images for visual diagnostics.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("outputs/analyze/frames"),
        help="Directory for annotated frame images.",
    )
    return parser


def analyze_video(
    *,
    video_path: Path,
    command: str,
    output_path: Path,
    max_frames: int = 2,
    detector_name: str = "fake",
    model_path: Path = Path("yolov8n.pt"),
    confidence_threshold: float = 0.25,
    save_frames: bool = False,
    frames_dir: Path = Path("outputs/analyze/frames"),
) -> list[Event]:
    """Analyze a video using the selected detector/tracker scenario."""
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    rule = parse_command(command)
    frames = list(iter_video_frames(video_path, max_frames=max_frames))
    detector, tracker = make_components(
        frames=frames,
        event_type=rule.event_type,
        object_class=rule.object_class,
        detector_name=detector_name,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    zone = make_default_zone(rule.zone_name)
    result = analyze_frames(
        frames=frames,
        detector=detector,
        tracker=tracker,
        rule=rule,
        zone=zone,
    )
    write_events_json(result.events, output_path)

    if save_frames:
        write_annotated_frames(result.frames, frames_dir, zone=zone)

    return result.events


def make_default_zone(name: str = "main") -> Zone:
    return Zone(name=name, points=((0, 0), (100, 0), (100, 100), (0, 100)))


def make_components(
    *,
    frames: list[VideoFrame],
    event_type: EventType,
    object_class: ObjectClass,
    detector_name: str,
    model_path: Path,
    confidence_threshold: float,
) -> tuple[ObjectDetector, ObjectTracker]:
    if detector_name == "fake":
        return make_fake_components(frames, event_type, object_class)

    if detector_name == "yolo":
        return (
            YOLODetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                supported_classes={ObjectClass.PERSON, ObjectClass.CAR},
            ),
            IoUTracker(),
        )

    raise ValueError(f"Unsupported detector backend: {detector_name}")


def make_fake_components(
    frames: list[VideoFrame],
    event_type: EventType,
    object_class: ObjectClass,
) -> tuple[ObjectDetector, ObjectTracker]:
    if len(frames) < 2:
        return FakeDetector({}), FakeTracker({})

    first_frame = frames[0]
    second_frame = frames[1]

    first_center, second_center = _fake_motion_centers(event_type)

    first_detection = make_detection(
        frame=first_frame,
        object_class=object_class,
        center_x=first_center[0],
        center_y=first_center[1],
    )
    second_detection = make_detection(
        frame=second_frame,
        object_class=object_class,
        center_x=second_center[0],
        center_y=second_center[1],
    )
    first_track = make_track(first_detection, track_id=1)
    second_track = make_track(second_detection, track_id=1)

    return (
        FakeDetector(
            {
                first_frame.frame_index: [first_detection],
                second_frame.frame_index: [second_detection],
            }
        ),
        FakeTracker(
            {
                first_frame.frame_index: [first_track],
                second_frame.frame_index: [second_track],
            }
        ),
    )


def _fake_motion_centers(event_type: EventType) -> tuple[tuple[float, float], tuple[float, float]]:
    if event_type == EventType.ENTER_ZONE:
        return (150, 50), (50, 50)

    if event_type in {EventType.LEAVE_ZONE, EventType.EMPTY_ZONE}:
        return (50, 50), (150, 50)

    if event_type in {EventType.COUNT_IN_ZONE, EventType.TRACK_OBJECT}:
        return (50, 50), (50, 50)

    return (150, 50), (150, 50)


def make_detection(
    *,
    frame: VideoFrame,
    object_class: ObjectClass,
    center_x: float,
    center_y: float,
) -> Detection:
    return Detection(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        object_class=object_class,
        confidence=1.0,
        bbox=BoundingBox(
            x1=center_x - 5,
            y1=center_y - 5,
            x2=center_x + 5,
            y2=center_y + 5,
        ),
    )


def make_track(detection: Detection, *, track_id: int) -> Track:
    return Track(
        track_id=track_id,
        frame_index=detection.frame_index,
        timestamp_sec=detection.timestamp_sec,
        object_class=detection.object_class,
        bbox=detection.bbox,
    )


if __name__ == "__main__":
    raise SystemExit(main())
