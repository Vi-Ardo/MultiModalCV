import pytest

from multimodalcv.commands.interpreter import FallbackCommandInterpreter
from multimodalcv.commands.llm_interpreter import (
    ALLOWED_INTENTS,
    JSONLLMCommandInterpreter,
    MockLLMResponseProvider,
    OllamaResponseProvider,
    build_ollama_prompt,
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


class CapturingOllamaTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, payload: dict, timeout_sec: float) -> dict:
        self.calls.append((url, payload, timeout_sec))
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


def test_mock_llm_response_provider_returns_valid_tracking_intent() -> None:
    interpreter = JSONLLMCommandInterpreter(MockLLMResponseProvider())

    intent = interpreter.interpret("Понаблюдай за этим человеком")

    assert intent.name == "track_person"
    assert intent.rule.object_class == ObjectClass.PERSON
    assert intent.confidence == "llm_validated"


def test_mock_llm_response_provider_returns_valid_count_intent() -> None:
    interpreter = JSONLLMCommandInterpreter(MockLLMResponseProvider())

    intent = interpreter.interpret("Сколько людей видно сейчас?")

    assert intent.name == "count_people_in_frame"
    assert intent.rule.event_type == EventType.COUNT_IN_FRAME


def test_mock_llm_response_provider_rejects_unknown_command() -> None:
    interpreter = JSONLLMCommandInterpreter(MockLLMResponseProvider())

    with pytest.raises(UnsupportedCommandError, match="неподдерживаемый intent"):
        interpreter.interpret("Определи странное поведение")


def test_build_ollama_prompt_contains_command_and_allowed_intents() -> None:
    prompt = build_ollama_prompt("Понаблюдай за человеком")

    assert "Понаблюдай за человеком" in prompt
    assert "track_person" in prompt
    assert "count_people_in_frame" in prompt


def test_ollama_response_provider_posts_structured_request() -> None:
    transport = CapturingOllamaTransport(
        {"response": '{"intent": "track_person", "object": "person"}'}
    )
    provider = OllamaResponseProvider(
        model="qwen2.5:3b",
        base_url="http://localhost:11434/",
        timeout_sec=7,
        transport=transport,
    )

    response_text = provider.generate("Понаблюдай за человеком")

    assert response_text == '{"intent": "track_person", "object": "person"}'
    assert len(transport.calls) == 1
    url, payload, timeout_sec = transport.calls[0]
    assert url == "http://localhost:11434/api/generate"
    assert payload["model"] == "qwen2.5:3b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["format"]["properties"]["intent"]["enum"]
    assert timeout_sec == 7


def test_ollama_response_provider_rejects_empty_response() -> None:
    provider = OllamaResponseProvider(transport=CapturingOllamaTransport({"response": ""}))

    with pytest.raises(UnsupportedCommandError, match="Ollama вернула пустой"):
        provider.generate("Понаблюдай за человеком")


def test_ollama_response_provider_can_drive_json_interpreter() -> None:
    provider = OllamaResponseProvider(
        transport=CapturingOllamaTransport(
            {"response": '{"intent": "count_people_in_frame", "object": "person"}'}
        )
    )
    interpreter = JSONLLMCommandInterpreter(provider)

    intent = interpreter.interpret("Сколько людей видно?")

    assert intent.name == "count_people_in_frame"
    assert intent.confidence == "llm_validated"
