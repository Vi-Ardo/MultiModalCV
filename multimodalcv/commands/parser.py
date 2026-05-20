"""Restricted natural-language command parser."""

import re

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import EventType, ObjectClass


class UnsupportedCommandError(ValueError):
    """Raised when a command is outside the supported command set."""


_PUNCTUATION_RE = re.compile(r"[,.!?;:]")
_WHITESPACE_RE = re.compile(r"\s+")

_COMMAND_PATTERNS: tuple[tuple[tuple[str, ...], EventType, ObjectClass], ...] = (
    (
        (
            "сообщи когда человек войдет в зону",
            "сообщи когда человек зайдет в зону",
            "сообщи когда человек входит в зону",
        ),
        EventType.ENTER_ZONE,
        ObjectClass.PERSON,
    ),
    (
        (
            "сообщи когда человек выйдет из зоны",
            "сообщи когда человек покинет зону",
            "сообщи когда человек выходит из зоны",
        ),
        EventType.LEAVE_ZONE,
        ObjectClass.PERSON,
    ),
    (
        (
            "сообщи когда все люди покинут зону",
            "сообщи когда все люди выйдут из зоны",
            "сообщи когда в зоне не останется людей",
        ),
        EventType.EMPTY_ZONE,
        ObjectClass.PERSON,
    ),
    (
        (
            "посчитай людей в зоне",
            "посчитать людей в зоне",
            "сколько людей в зоне",
        ),
        EventType.COUNT_IN_ZONE,
        ObjectClass.PERSON,
    ),
    (
        (
            "следи за человеком",
            "отслеживай человека",
        ),
        EventType.TRACK_OBJECT,
        ObjectClass.PERSON,
    ),
    (
        (
            "следи за машиной",
            "отслеживай машину",
        ),
        EventType.TRACK_OBJECT,
        ObjectClass.CAR,
    ),
)


def parse_command(command: str, *, zone_name: str = "main") -> CommandRule:
    """Parse a supported text command into a structured rule."""
    normalized = normalize_command(command)

    for phrases, event_type, object_class in _COMMAND_PATTERNS:
        if normalized in phrases:
            return CommandRule(
                event_type=event_type,
                object_class=object_class,
                zone_name=zone_name,
                raw_text=command,
            )

    raise UnsupportedCommandError(f"Unsupported command: {command}")


def normalize_command(command: str) -> str:
    """Normalize text for exact matching against supported command templates."""
    normalized = command.casefold().replace("ё", "е")
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def supported_commands() -> tuple[str, ...]:
    """Return canonical examples of supported commands."""
    return tuple(phrases[0] for phrases, _, _ in _COMMAND_PATTERNS)

