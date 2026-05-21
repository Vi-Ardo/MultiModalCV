import pytest

from multimodalcv.commands.parser import (
    UnsupportedCommandError,
    interpret_command,
    normalize_command,
    parse_command,
    supported_command_examples,
    supported_commands,
)
from multimodalcv.core.models import EventType, ObjectClass


@pytest.mark.parametrize(
    ("command", "event_type", "object_class"),
    [
        ("Сообщи, когда человек войдет в зону", EventType.ENTER_ZONE, ObjectClass.PERSON),
        ("Уведоми, если человек зайдет в комнату", EventType.ENTER_ZONE, ObjectClass.PERSON),
        ("Предупреди, когда человек появится в кадре", EventType.ENTER_ZONE, ObjectClass.PERSON),
        ("Сообщи, когда человек выйдет из зоны", EventType.LEAVE_ZONE, ObjectClass.PERSON),
        ("Уведоми, если человек покинет помещение", EventType.LEAVE_ZONE, ObjectClass.PERSON),
        ("Сообщи, когда все люди покинут зону", EventType.EMPTY_ZONE, ObjectClass.PERSON),
        ("Скажи, когда в комнате не останется людей", EventType.EMPTY_ZONE, ObjectClass.PERSON),
        ("Посчитай людей в зоне", EventType.COUNT_IN_ZONE, ObjectClass.PERSON),
        ("Сколько человек в комнате", EventType.COUNT_IN_ZONE, ObjectClass.PERSON),
        ("Посчитай людей в кадре", EventType.COUNT_IN_FRAME, ObjectClass.PERSON),
        ("Сколько человек на видео", EventType.COUNT_IN_FRAME, ObjectClass.PERSON),
        ("Посчитай людей", EventType.COUNT_IN_FRAME, ObjectClass.PERSON),
        ("Следи за человеком", EventType.TRACK_OBJECT, ObjectClass.PERSON),
        ("Отслеживай человека", EventType.TRACK_OBJECT, ObjectClass.PERSON),
        ("Следи за машиной", EventType.TRACK_OBJECT, ObjectClass.CAR),
        ("Наблюдай за авто", EventType.TRACK_OBJECT, ObjectClass.CAR),
    ],
)
def test_parses_supported_command_variations(
    command: str,
    event_type: EventType,
    object_class: ObjectClass,
) -> None:
    rule = parse_command(command)

    assert rule.event_type == event_type
    assert rule.object_class == object_class
    assert rule.zone_name == "main"


def test_parser_accepts_custom_zone_name() -> None:
    rule = parse_command("Посчитай людей в зоне", zone_name="entrance")

    assert rule.zone_name == "entrance"


def test_interpret_command_returns_user_facing_intent() -> None:
    intent = interpret_command("Сколько человек на видео")

    assert intent.name == "count_people_in_frame"
    assert intent.summary == "Буду считать людей во всем кадре."
    assert intent.confidence == "deterministic"
    assert intent.rule.event_type == EventType.COUNT_IN_FRAME


def test_interpret_command_preserves_zone_name() -> None:
    intent = interpret_command("Уведоми, если человек зайдет в комнату", zone_name="entrance")

    assert intent.name == "person_enters_area"
    assert intent.rule.zone_name == "entrance"


def test_normalization_handles_case_punctuation_spaces_and_yo() -> None:
    normalized = normalize_command("  СООБЩИ,   когда человек ВОЙДЁТ в зону!!! ")

    assert normalized == "сообщи когда человек войдет в зону"


def test_parser_rejects_unsupported_command() -> None:
    with pytest.raises(UnsupportedCommandError, match="Команда пока не поддерживается"):
        parse_command("Определи подозрительное поведение")


def test_count_in_zone_takes_priority_over_count_in_frame() -> None:
    rule = parse_command("Сколько людей в зоне")

    assert rule.event_type == EventType.COUNT_IN_ZONE


def test_supported_commands_returns_canonical_examples() -> None:
    commands = supported_commands()

    assert "Сообщи, когда человек войдет в зону" in commands
    assert "Следи за машиной" in commands


def test_supported_command_examples_returns_variations() -> None:
    examples = supported_command_examples()

    assert "Уведоми, если человек зайдет в комнату" in examples
    assert "Наблюдай за авто" in examples
