"""Synthetic CLI demo for command-driven rule evaluation."""

import argparse
from pathlib import Path

from multimodalcv.commands.parser import UnsupportedCommandError, parse_command, supported_commands
from multimodalcv.core.models import BoundingBox, Event, EventType, ObjectClass, Track, Zone
from multimodalcv.reporting.json_report import write_events_json
from multimodalcv.rules.engine import evaluate_rule


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        events = run_demo(command=args.command, output_path=args.output)
    except UnsupportedCommandError as error:
        print(error)
        print("Supported commands:")
        for command in supported_commands():
            print(f"- {command}")
        return 2

    print(f"Wrote {len(events)} event(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synthetic MultiModalCV command demo.",
    )
    parser.add_argument(
        "command",
        help="Supported Russian command, for example: 'Сообщи, когда человек войдет в зону'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/events.json"),
        help="Path to the JSON event report.",
    )
    return parser


def run_demo(*, command: str, output_path: Path) -> list[Event]:
    rule = parse_command(command)
    zone = make_demo_zone(rule.zone_name)
    previous_tracks, current_tracks = make_demo_scene(rule.event_type, rule.object_class)

    events = evaluate_rule(
        rule=rule,
        zone=zone,
        previous_tracks=previous_tracks,
        current_tracks=current_tracks,
    )
    write_events_json(events, output_path)
    return events


def make_demo_zone(name: str = "main") -> Zone:
    return Zone(name=name, points=((0, 0), (100, 0), (100, 100), (0, 100)))


def make_demo_scene(
    event_type: EventType,
    object_class: ObjectClass,
) -> tuple[list[Track], list[Track]]:
    if event_type == EventType.ENTER_ZONE:
        return (
            [make_track(object_class=object_class, center_x=150, center_y=50, frame_index=1)],
            [make_track(object_class=object_class, center_x=50, center_y=50, frame_index=2)],
        )

    if event_type in {EventType.LEAVE_ZONE, EventType.EMPTY_ZONE}:
        return (
            [make_track(object_class=object_class, center_x=50, center_y=50, frame_index=1)],
            [make_track(object_class=object_class, center_x=150, center_y=50, frame_index=2)],
        )

    if event_type == EventType.COUNT_IN_ZONE:
        return (
            [],
            [
                make_track(track_id=1, object_class=object_class, center_x=50, center_y=50, frame_index=2),
                make_track(track_id=2, object_class=object_class, center_x=150, center_y=50, frame_index=2),
            ],
        )

    if event_type == EventType.TRACK_OBJECT:
        return (
            [],
            [make_track(object_class=object_class, center_x=150, center_y=50, frame_index=2)],
        )

    return [], []


def make_track(
    *,
    track_id: int = 1,
    object_class: ObjectClass,
    center_x: float,
    center_y: float,
    frame_index: int,
) -> Track:
    return Track(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10,
        object_class=object_class,
        bbox=BoundingBox(
            x1=center_x - 5,
            y1=center_y - 5,
            x2=center_x + 5,
            y2=center_y + 5,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

