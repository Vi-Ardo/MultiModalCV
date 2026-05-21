import pytest

from multimodalcv.commands.interpreter import FallbackCommandInterpreter
from multimodalcv.commands.llm_interpreter import (
    ALLOWED_INTENTS,
    JSONLLMCommandInterpreter,
    parse_llm_intent_response,
)
from multimodalcv.commands.parser import CommandIntent, UnsupportedCommandError
from multimodalcv.core.models import EventType, ObjectClass


class StaticResponseProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, command: str) -> str:
        self.calls.append(command)
        return self.response


class RejectingInterpreter:
    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        raise UnsupportedCommandError(f"Команда пока не поддерживается: {command}")


def test_allowed_intents_cover_initial_command_set() -> None:
    assert set(ALLOWED_INTENTS) == {
        "person_enters_area",
        "person_leaves_area",
        "area_becomes_empty",
        "count_people_in_area",
        "count_people_in_frame",
        "track_person",
        "track_car",
    }


def test_parse_llm_intent_response_returns_command_intent() -> None:
    intent = parse_llm_intent_response(
        '{"intent": "count_people_in_frame", "object": "person", "zone": "main"}',
        raw_text="Сколько людей видно?",
    )

    assert intent.name == "count_people_in_frame"
    assert intent.summary == "Буду считать людей во всем кадре."
    assert intent.confidence == "llm_validated"
    assert intent.rule.event_type == EventType.COUNT_IN_FRAME
    assert intent.rule.object_class == ObjectClass.PERSON
    assert intent.rule.zone_name == "main"
    assert intent.rule.raw_text == "Сколько людей видно?"


def test_parse_llm_intent_response_uses_default_zone_when_missing() -> None:
    intent = parse_llm_intent_response(
        '{"intent": "track_car", "object": "car"}',
        raw_text="Следи за красной машиной",
        default_zone_name="parking",
    )

    assert intent.rule.zone_name == "parking"


def test_parse_llm_intent_response_rejects_unknown_intent() -> None:
    with pytest.raises(UnsupportedCommandError, match="неподдерживаемый intent"):
        parse_llm_intent_response(
            '{"intent": "detect_fight", "object": "person"}',
            raw_text="Найди драку",
        )


def test_parse_llm_intent_response_rejects_unknown_object() -> None:
    with pytest.raises(UnsupportedCommandError, match="неподдерживаемый object"):
        parse_llm_intent_response(
            '{"intent": "track_person", "object": "dog"}',
            raw_text="Следи за собакой",
        )


def test_parse_llm_intent_response_rejects_mismatched_object() -> None:
    with pytest.raises(UnsupportedCommandError, match="не подходит"):
        parse_llm_intent_response(
            '{"intent": "track_car", "object": "person"}',
            raw_text="Следи за машиной",
        )


def test_parse_llm_intent_response_rejects_invalid_json() -> None:
    with pytest.raises(UnsupportedCommandError, match="некорректный JSON"):
        parse_llm_intent_response("not json", raw_text="Следи за человеком")


def test_parse_llm_intent_response_rejects_non_object_json() -> None:
    with pytest.raises(UnsupportedCommandError, match="JSON-объект"):
        parse_llm_intent_response("[]", raw_text="Следи за человеком")


def test_parse_llm_intent_response_requires_string_fields() -> None:
    with pytest.raises(UnsupportedCommandError, match="`intent`"):
        parse_llm_intent_response('{"intent": "", "object": "person"}', raw_text="Команда")

    with pytest.raises(UnsupportedCommandError, match="`object`"):
        parse_llm_intent_response('{"intent": "track_person"}', raw_text="Команда")


def test_json_llm_interpreter_uses_response_provider() -> None:
    provider = StaticResponseProvider('{"intent": "track_car", "object": "car", "zone": "parking"}')
    interpreter = JSONLLMCommandInterpreter(provider)

    intent = interpreter.interpret("Проследи за машиной", zone_name="main")

    assert intent.name == "track_car"
    assert intent.rule.zone_name == "parking"
    assert provider.calls == ["Проследи за машиной"]


def test_json_llm_interpreter_can_be_used_as_fallback() -> None:
    llm = JSONLLMCommandInterpreter(
        StaticResponseProvider('{"intent": "track_person", "object": "person"}')
    )
    interpreter = FallbackCommandInterpreter((RejectingInterpreter(), llm))

    intent = interpreter.interpret("Понаблюдай за этим человеком")

    assert intent.name == "track_person"
    assert intent.confidence == "llm_validated"
