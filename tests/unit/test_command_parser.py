import pytest

from multimodalcv.commands.parser import (
    UnsupportedCommandError,
    normalize_command,
    parse_command,
    supported_commands,
)
from multimodalcv.core.models import EventType, ObjectClass


def test_parses_person_enter_zone_command() -> None:
    rule = parse_command("Сообщи, когда человек войдет в зону")

    assert rule.event_type == EventType.ENTER_ZONE
    assert rule.object_class == ObjectClass.PERSON
    assert rule.zone_name == "main"


def test_parses_person_leave_zone_command() -> None:
    rule = parse_command("Сообщи, когда человек выйдет из зоны")

    assert rule.event_type == EventType.LEAVE_ZONE
    assert rule.object_class == ObjectClass.PERSON


def test_parses_empty_zone_command() -> None:
    rule = parse_command("Сообщи, когда все люди покинут зону")

    assert rule.event_type == EventType.EMPTY_ZONE
    assert rule.object_class == ObjectClass.PERSON


def test_parses_count_people_command() -> None:
    rule = parse_command("Посчитай людей в зоне")

    assert rule.event_type == EventType.COUNT_IN_ZONE
    assert rule.object_class == ObjectClass.PERSON


def test_parses_count_people_in_frame_command() -> None:
    rule = parse_command("Посчитай людей в кадре")

    assert rule.event_type == EventType.COUNT_IN_FRAME
    assert rule.object_class == ObjectClass.PERSON


def test_parses_track_person_command() -> None:
    rule = parse_command("Следи за человеком")

    assert rule.event_type == EventType.TRACK_OBJECT
    assert rule.object_class == ObjectClass.PERSON


def test_parses_track_car_command() -> None:
    rule = parse_command("Следи за машиной")

    assert rule.event_type == EventType.TRACK_OBJECT
    assert rule.object_class == ObjectClass.CAR


def test_parser_accepts_custom_zone_name() -> None:
    rule = parse_command("Посчитай людей в зоне", zone_name="entrance")

    assert rule.zone_name == "entrance"


def test_normalization_handles_case_punctuation_spaces_and_yo() -> None:
    normalized = normalize_command("  СООБЩИ,   когда человек ВОЙДЁТ в зону!!! ")

    assert normalized == "сообщи когда человек войдет в зону"


def test_parser_rejects_unsupported_command() -> None:
    with pytest.raises(UnsupportedCommandError):
        parse_command("Определи подозрительное поведение")


def test_supported_commands_returns_examples() -> None:
    commands = supported_commands()

    assert "сообщи когда человек войдет в зону" in commands
    assert "следи за машиной" in commands
