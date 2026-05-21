"""Analysis summary report helpers."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from multimodalcv.core.models import Event


def build_analysis_summary(
    *,
    events: list[Event],
    processed_frames: int,
    detector_name: str,
    annotated_frame_count: int = 0,
) -> dict[str, Any]:
    """Build a compact JSON-serializable analysis summary."""
    summary: dict[str, Any] = {
        "processed_frames": processed_frames,
        "detector": detector_name,
        "event_count": len(events),
        "annotated_frame_count": annotated_frame_count,
        "events_by_type": _events_by_type(events),
    }

    count_summary = _count_event_summary(events)
    if count_summary:
        summary["counts"] = count_summary

    return summary


def write_analysis_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write an analysis summary to JSON and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def default_summary_path(events_output_path: str | Path) -> Path:
    """Return the default summary path next to an events JSON file."""
    path = Path(events_output_path)
    return path.with_name("summary.json")


def _events_by_type(events: list[Event]) -> dict[str, int]:
    counter = Counter(event.event_type.value for event in events)
    return dict(sorted(counter.items()))


def _count_event_summary(events: list[Event]) -> dict[str, Any]:
    count_values = [
        int(event.metadata["count"])
        for event in events
        if "count" in event.metadata
    ]

    if not count_values:
        return {}

    distribution = Counter(count_values)
    most_common_count, most_common_frequency = distribution.most_common(1)[0]

    return {
        "min_count": min(count_values),
        "max_count": max(count_values),
        "most_common_count": most_common_count,
        "most_common_frequency": most_common_frequency,
        "count_distribution": {
            str(count): frequency
            for count, frequency in sorted(distribution.items())
        },
        "frames_with_count_gt_zero": sum(count > 0 for count in count_values),
        "frames_with_count_gte_two": sum(count >= 2 for count in count_values),
    }

