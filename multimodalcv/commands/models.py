"""Command rule models."""

from dataclasses import dataclass

from multimodalcv.core.models import EventType, ObjectClass


@dataclass(frozen=True)
class CommandRule:
    """Structured representation of a supported text command."""

    event_type: EventType
    object_class: ObjectClass
    raw_text: str
    zone_name: str = "main"

