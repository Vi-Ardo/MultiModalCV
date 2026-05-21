import pytest

from multimodalcv.commands.interpreter import (
    DeterministicCommandInterpreter,
    FallbackCommandInterpreter,
    default_command_interpreter,
)
from multimodalcv.commands.models import CommandRule
from multimodalcv.commands.parser import CommandIntent, UnsupportedCommandError
from multimodalcv.core.models import EventType, ObjectClass


class RejectingInterpreter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        self.calls.append(command)
        raise UnsupportedCommandError(f"Команда пока не поддерживается: {command}")


class StaticInterpreter:
    def __init__(self, intent: CommandIntent) -> None:
        self.intent = intent
        self.calls: list[tuple[str, str]] = []

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        self.calls.append((command, zone_name))
        return self.intent


def test_deterministic_interpreter_returns_command_intent() -> None:
    intent = DeterministicCommandInterpreter().interpret("Посчитай людей в кадре")

    assert intent.name == "count_people_in_frame"
    assert intent.rule.event_type == EventType.COUNT_IN_FRAME


def test_default_command_interpreter_is_deterministic() -> None:
    intent = default_command_interpreter().interpret("Следи за машиной")

    assert intent.name == "track_car"


def test_fallback_interpreter_uses_first_successful_interpreter() -> None:
    first = RejectingInterpreter()
    second_intent = make_intent("llm_track_person")
    second = StaticInterpreter(second_intent)
    interpreter = FallbackCommandInterpreter((first, second))

    intent = interpreter.interpret("Пожалуйста, проследи за человеком", zone_name="entrance")

    assert intent == second_intent
    assert first.calls == ["Пожалуйста, проследи за человеком"]
    assert second.calls == [("Пожалуйста, проследи за человеком", "entrance")]


def test_fallback_interpreter_does_not_call_later_interpreters_after_success() -> None:
    first_intent = make_intent("deterministic_track_person")
    first = StaticInterpreter(first_intent)
    second = StaticInterpreter(make_intent("llm_track_person"))
    interpreter = FallbackCommandInterpreter((first, second))

    assert interpreter.interpret("Следи за человеком") == first_intent
    assert second.calls == []


def test_fallback_interpreter_raises_last_unsupported_error() -> None:
    interpreter = FallbackCommandInterpreter((RejectingInterpreter(), RejectingInterpreter()))

    with pytest.raises(UnsupportedCommandError, match="Найди драку"):
        interpreter.interpret("Найди драку")


def test_fallback_interpreter_requires_at_least_one_interpreter() -> None:
    with pytest.raises(ValueError, match="at least one command interpreter"):
        FallbackCommandInterpreter(())


def make_intent(name: str) -> CommandIntent:
    return CommandIntent(
        name=name,
        summary="Тестовое намерение.",
        confidence="test",
        rule=CommandRule(
            event_type=EventType.TRACK_OBJECT,
            object_class=ObjectClass.PERSON,
            raw_text="test",
        ),
    )
