"""Strict JSON contract for future model-backed command interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from multimodalcv.commands.models import CommandRule
from multimodalcv.commands.parser import CommandIntent, UnsupportedCommandError, normalize_command
from multimodalcv.core.models import EventType, ObjectClass


class LLMResponseProvider(Protocol):
    """Callable backend that converts user text into a JSON intent response."""

    def generate(self, command: str) -> str:
        """Return a JSON object encoded as text."""


class OllamaTransport(Protocol):
    """HTTP transport used by the Ollama response provider."""

    def __call__(self, url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
        """Post JSON to Ollama and return a decoded JSON response."""


@dataclass(frozen=True)
class AllowedIntent:
    """Intent that a model is allowed to select."""

    name: str
    summary: str
    event_type: EventType
    object_class: ObjectClass


ALLOWED_INTENTS: dict[str, AllowedIntent] = {
    "person_enters_area": AllowedIntent(
        name="person_enters_area",
        summary="Буду ждать появления человека в выбранной зоне.",
        event_type=EventType.ENTER_ZONE,
        object_class=ObjectClass.PERSON,
    ),
    "person_leaves_area": AllowedIntent(
        name="person_leaves_area",
        summary="Буду ждать выхода человека из выбранной зоны.",
        event_type=EventType.LEAVE_ZONE,
        object_class=ObjectClass.PERSON,
    ),
    "area_becomes_empty": AllowedIntent(
        name="area_becomes_empty",
        summary="Буду ждать момента, когда в выбранной зоне не останется людей.",
        event_type=EventType.EMPTY_ZONE,
        object_class=ObjectClass.PERSON,
    ),
    "count_people_in_area": AllowedIntent(
        name="count_people_in_area",
        summary="Буду считать людей внутри выбранной зоны.",
        event_type=EventType.COUNT_IN_ZONE,
        object_class=ObjectClass.PERSON,
    ),
    "count_people_in_frame": AllowedIntent(
        name="count_people_in_frame",
        summary="Буду считать людей во всем кадре.",
        event_type=EventType.COUNT_IN_FRAME,
        object_class=ObjectClass.PERSON,
    ),
    "track_person": AllowedIntent(
        name="track_person",
        summary="Буду отслеживать человека на видео.",
        event_type=EventType.TRACK_OBJECT,
        object_class=ObjectClass.PERSON,
    ),
    "track_car": AllowedIntent(
        name="track_car",
        summary="Буду отслеживать машину на видео.",
        event_type=EventType.TRACK_OBJECT,
        object_class=ObjectClass.CAR,
    ),
}

_OLLAMA_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": sorted(ALLOWED_INTENTS),
        },
        "object": {
            "type": "string",
            "enum": [ObjectClass.PERSON.value, ObjectClass.CAR.value],
        },
        "zone": {
            "type": "string",
        },
    },
    "required": ["intent", "object"],
    "additionalProperties": False,
}


class JSONLLMCommandInterpreter:
    """Interpreter that validates a JSON intent response from a model backend."""

    def __init__(self, response_provider: LLMResponseProvider) -> None:
        self._response_provider = response_provider

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        response_text = self._response_provider.generate(command)
        return parse_llm_intent_response(response_text, raw_text=command, default_zone_name=zone_name)


class MockLLMResponseProvider:
    """Local stand-in for a future LLM API provider."""

    def generate(self, command: str) -> str:
        normalized = normalize_command(command)
        payload = _mock_payload_for_command(normalized)
        return json.dumps(payload)


class OllamaResponseProvider:
    """Response provider backed by a local Ollama server."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout_sec: float = 30.0,
        transport: OllamaTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._transport = transport or _post_json

    def generate(self, command: str) -> str:
        payload = {
            "model": self.model,
            "prompt": build_ollama_prompt(command),
            "stream": False,
            "format": _OLLAMA_INTENT_SCHEMA,
            "options": {
                "temperature": 0,
            },
        }
        response = self._transport(f"{self.base_url}/api/generate", payload, self.timeout_sec)
        response_text = response.get("response")
        if not isinstance(response_text, str) or not response_text.strip():
            raise UnsupportedCommandError("Ollama вернула пустой или некорректный ответ")

        return response_text.strip()


def build_ollama_prompt(command: str) -> str:
    """Build a constrained prompt for local command interpretation."""
    return (
        "Ты интерпретируешь русскую команду для системы видеоаналитики. "
        "Верни только JSON без пояснений. "
        "Разрешенные intent: "
        f"{', '.join(sorted(ALLOWED_INTENTS))}. "
        "Разрешенные object: person, car. "
        "Если команда про людей в кадре, используй object=person. "
        "Если команда про машину или автомобиль, используй object=car. "
        "Если команда не соответствует разрешенным сценариям, верни "
        '{"intent":"unsupported","object":"person"}. '
        f"Команда: {command}"
    )


def parse_llm_intent_response(
    response_text: str,
    *,
    raw_text: str,
    default_zone_name: str = "main",
) -> CommandIntent:
    """Validate a JSON intent response and convert it to a command intent."""
    payload = _load_json_object(response_text)
    intent_name = _require_string(payload, "intent")
    object_name = _require_string(payload, "object")
    zone_name = _optional_string(payload, "zone", default_zone_name)

    if intent_name not in ALLOWED_INTENTS:
        raise UnsupportedCommandError(f"Модель вернула неподдерживаемый intent: {intent_name}")

    intent = ALLOWED_INTENTS[intent_name]
    object_class = _parse_object_class(object_name)
    if object_class != intent.object_class:
        raise UnsupportedCommandError(
            f"Object `{object_name}` не подходит для intent `{intent_name}`"
        )

    return CommandIntent(
        name=intent.name,
        summary=intent.summary,
        confidence="llm_validated",
        rule=CommandRule(
            event_type=intent.event_type,
            object_class=object_class,
            zone_name=zone_name,
            raw_text=raw_text,
        ),
    )


def _load_json_object(response_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise UnsupportedCommandError("Модель вернула некорректный JSON") from error

    if not isinstance(payload, dict):
        raise UnsupportedCommandError("Модель должна вернуть JSON-объект")

    return payload


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedCommandError(f"Модель не вернула строковое поле `{field_name}`")

    return value.strip()


def _optional_string(payload: dict[str, Any], field_name: str, default_value: str) -> str:
    value = payload.get(field_name, default_value)
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedCommandError(f"Поле `{field_name}` должно быть строкой")

    return value.strip()


def _parse_object_class(object_name: str) -> ObjectClass:
    try:
        return ObjectClass(object_name)
    except ValueError as error:
        raise UnsupportedCommandError(f"Модель вернула неподдерживаемый object: {object_name}") from error


def _post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except (OSError, TimeoutError, URLError) as error:
        raise UnsupportedCommandError(
            "Не удалось обратиться к Ollama. Проверь, что приложение Ollama запущено "
            "и модель скачана."
        ) from error

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise UnsupportedCommandError("Ollama вернула некорректный JSON-ответ API") from error

    if not isinstance(decoded, dict):
        raise UnsupportedCommandError("Ollama API вернула ответ не в формате JSON-объекта")

    return decoded


def _mock_payload_for_command(command: str) -> dict[str, str]:
    if _contains_any(command, ("машин", "авто", "автомобил")) and _contains_any(
        command,
        ("след", "отслед", "наблюд", "смотри"),
    ):
        return {"intent": "track_car", "object": "car"}

    if _contains_any(command, ("человек", "люд", "персон")):
        if _contains_any(command, ("след", "отслед", "наблюд", "смотри")):
            return {"intent": "track_person", "object": "person"}

        if _contains_any(command, ("сколько", "посч", "количество")):
            if _contains_any(command, ("зон", "комнат", "помещен", "област")):
                return {"intent": "count_people_in_area", "object": "person"}
            return {"intent": "count_people_in_frame", "object": "person"}

        if _contains_any(command, ("вош", "вход", "зайд", "появ")):
            return {"intent": "person_enters_area", "object": "person"}

        if _contains_any(command, ("выш", "выход", "покин", "исчез")):
            return {"intent": "person_leaves_area", "object": "person"}

        if _contains_any(command, ("никого", "пуст", "остан")):
            return {"intent": "area_becomes_empty", "object": "person"}

    return {"intent": "unsupported", "object": "person"}


def _contains_any(command: str, stems: tuple[str, ...]) -> bool:
    return any(stem in command for stem in stems)
