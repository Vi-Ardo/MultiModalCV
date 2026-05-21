"""Restricted natural-language command parser."""

from __future__ import annotations

import re
from dataclasses import dataclass

from multimodalcv.commands.models import CommandRule
from multimodalcv.core.models import EventType, ObjectClass


class UnsupportedCommandError(ValueError):
    """Raised when a command is outside the supported command set."""


@dataclass(frozen=True)
class _CommandPattern:
    examples: tuple[str, ...]
    event_type: EventType
    object_class: ObjectClass
    required: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...] = ()


_PUNCTUATION_RE = re.compile(r"[,.!?;:]")
_WHITESPACE_RE = re.compile(r"\s+")

_PERSON_WORDS = (
    "человек",
    "человека",
    "человеку",
    "человеком",
    "люди",
    "людей",
    "персона",
    "персону",
    "персоны",
)
_CAR_WORDS = (
    "машина",
    "машину",
    "машиной",
    "автомобиль",
    "автомобилем",
    "авто",
)
_ZONE_WORDS = (
    "зона",
    "зону",
    "зоне",
    "зоны",
    "область",
    "области",
    "комната",
    "комнату",
    "комнате",
    "помещение",
    "помещении",
)
_FRAME_WORDS = (
    "кадр",
    "кадре",
    "видео",
    "изображение",
    "изображении",
)

_COMMAND_PATTERNS: tuple[_CommandPattern, ...] = (
    _CommandPattern(
        examples=(
            "Сообщи, когда человек войдет в зону",
            "Уведоми, если человек зайдет в комнату",
            "Предупреди, когда человек появится в кадре",
        ),
        event_type=EventType.ENTER_ZONE,
        object_class=ObjectClass.PERSON,
        required=(
            ("сообщи", "уведоми", "скажи", "предупреди"),
            _PERSON_WORDS,
            ("войдет", "войдёт", "зайдет", "зайдёт", "входит", "появится", "появился"),
            _ZONE_WORDS + _FRAME_WORDS,
        ),
    ),
    _CommandPattern(
        examples=(
            "Сообщи, когда человек выйдет из зоны",
            "Уведоми, если человек покинет комнату",
        ),
        event_type=EventType.LEAVE_ZONE,
        object_class=ObjectClass.PERSON,
        required=(
            ("сообщи", "уведоми", "скажи", "предупреди"),
            _PERSON_WORDS,
            ("выйдет", "выходит", "покинет", "покидает", "исчезнет", "исчез"),
            _ZONE_WORDS + _FRAME_WORDS,
        ),
    ),
    _CommandPattern(
        examples=(
            "Сообщи, когда все люди покинут зону",
            "Уведоми, когда в комнате не останется людей",
        ),
        event_type=EventType.EMPTY_ZONE,
        object_class=ObjectClass.PERSON,
        required=(
            ("сообщи", "уведоми", "скажи", "предупреди"),
            ("все", "никого", "останется", "осталось"),
            _PERSON_WORDS,
            _ZONE_WORDS + _FRAME_WORDS,
        ),
    ),
    _CommandPattern(
        examples=(
            "Посчитай людей в зоне",
            "Сколько человек в комнате",
        ),
        event_type=EventType.COUNT_IN_ZONE,
        object_class=ObjectClass.PERSON,
        required=(
            ("посчитай", "посчитать", "сколько", "количество"),
            _PERSON_WORDS,
            _ZONE_WORDS,
        ),
    ),
    _CommandPattern(
        examples=(
            "Посчитай людей в кадре",
            "Сколько человек на видео",
            "Посчитай людей",
        ),
        event_type=EventType.COUNT_IN_FRAME,
        object_class=ObjectClass.PERSON,
        required=(
            ("посчитай", "посчитать", "сколько", "количество"),
            _PERSON_WORDS,
        ),
        forbidden=_ZONE_WORDS,
    ),
    _CommandPattern(
        examples=(
            "Следи за человеком",
            "Отслеживай человека",
        ),
        event_type=EventType.TRACK_OBJECT,
        object_class=ObjectClass.PERSON,
        required=(
            ("следи", "отслеживай", "отследи", "наблюдай"),
            _PERSON_WORDS,
        ),
    ),
    _CommandPattern(
        examples=(
            "Следи за машиной",
            "Отслеживай автомобиль",
            "Наблюдай за авто",
        ),
        event_type=EventType.TRACK_OBJECT,
        object_class=ObjectClass.CAR,
        required=(
            ("следи", "отслеживай", "отследи", "наблюдай"),
            _CAR_WORDS,
        ),
    ),
)


def parse_command(command: str, *, zone_name: str = "main") -> CommandRule:
    """Parse a supported text command into a structured rule."""
    normalized = normalize_command(command)

    for pattern in _COMMAND_PATTERNS:
        if _matches_pattern(normalized, pattern):
            return CommandRule(
                event_type=pattern.event_type,
                object_class=pattern.object_class,
                zone_name=zone_name,
                raw_text=command,
            )

    raise UnsupportedCommandError(f"Unsupported command: {command}")


def normalize_command(command: str) -> str:
    """Normalize text for deterministic matching against supported words."""
    normalized = command.casefold().replace("ё", "е")
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def supported_commands() -> tuple[str, ...]:
    """Return canonical examples of supported commands."""
    return tuple(pattern.examples[0] for pattern in _COMMAND_PATTERNS)


def supported_command_examples() -> tuple[str, ...]:
    """Return examples that show accepted command variations."""
    return tuple(example for pattern in _COMMAND_PATTERNS for example in pattern.examples)


def _matches_pattern(command: str, pattern: _CommandPattern) -> bool:
    if not command:
        return False

    if any(_contains_word(command, word) for word in pattern.forbidden):
        return False

    return all(_contains_any_word(command, words) for words in pattern.required)


def _contains_any_word(command: str, words: tuple[str, ...]) -> bool:
    return any(_contains_word(command, word) for word in words)


def _contains_word(command: str, word: str) -> bool:
    normalized_word = normalize_command(word)
    return bool(re.search(rf"(^|\s){re.escape(normalized_word)}(\s|$)", command))
