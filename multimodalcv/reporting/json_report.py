"""JSON event report helpers."""

import json
from pathlib import Path
from typing import Any

from multimodalcv.core.models import Event


def event_to_dict(event: Event) -> dict[str, Any]:
    """Convert an event to a JSON-serializable dictionary."""
    return {
        "event_type": event.event_type.value,
        "frame_index": event.frame_index,
        "timestamp_sec": event.timestamp_sec,
        "zone_name": event.zone_name,
        "track_id": event.track_id,
        "object_class": None if event.object_class is None else event.object_class.value,
        "metadata": event.metadata,
    }


def events_to_dicts(events: list[Event]) -> list[dict[str, Any]]:
    """Convert events to JSON-serializable dictionaries."""
    return [event_to_dict(event) for event in events]


def write_events_json(events: list[Event], output_path: str | Path) -> Path:
    """Write events to a JSON file and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = events_to_dicts(events)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path

