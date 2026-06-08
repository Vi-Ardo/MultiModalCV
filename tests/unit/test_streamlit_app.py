from pathlib import Path

from interfaces.streamlit_app.app import (
    analysis_runs_dataframe,
    audit_dataframe,
    available_pages,
    build_command_interpreter,
    center_zone_rect,
    event_dicts_to_dataframe,
    filter_analysis_runs,
    find_frame_by_name,
    format_zone_rect,
    full_frame_zone_rect,
    render_command_preview,
    role_capabilities,
    users_dataframe,
    zone_rect_values,
)
from multimodalcv.commands.parser import CommandIntent
from multimodalcv.video.reader import VideoMetadata


def make_metadata(*, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        path=Path("sample.mp4"),
        frame_count=100,
        fps=25.0,
        width=width,
        height=height,
        duration_sec=4.0,
    )


def test_full_frame_zone_rect_uses_video_size() -> None:
    assert full_frame_zone_rect(make_metadata(width=1920, height=1080)) == "0,0,1920,1080"


def test_center_zone_rect_uses_center_half_of_video() -> None:
    assert center_zone_rect(make_metadata(width=1920, height=1080)) == "480,270,1440,810"


def test_format_zone_rect() -> None:
    assert format_zone_rect(1, 2, 3, 4) == "1,2,3,4"


def test_zone_rect_values() -> None:
    assert zone_rect_values("1,2,3,4") == (1, 2, 3, 4)


def test_find_frame_by_name() -> None:
    paths = [
        Path("frames/annotated_000001.jpg"),
        Path("frames/annotated_000002.jpg"),
    ]

    assert find_frame_by_name(paths, "annotated_000002.jpg") == paths[1]


def test_render_command_preview_returns_intent() -> None:
    intent = render_command_preview("Посчитай людей в кадре")

    assert isinstance(intent, CommandIntent)
    assert intent.name == "count_people_in_frame"


def test_render_command_preview_returns_none_for_unsupported_command() -> None:
    assert render_command_preview("Найди подозрительное поведение") is None


def test_build_command_interpreter_returns_deterministic_mode() -> None:
    interpreter = build_command_interpreter("Deterministic")

    intent = interpreter.interpret("Посчитай людей в кадре")

    assert intent.name == "count_people_in_frame"
    assert intent.confidence == "deterministic"


def test_build_command_interpreter_returns_mock_llm_fallback_mode() -> None:
    interpreter = build_command_interpreter("Deterministic + mock LLM fallback")

    intent = interpreter.interpret("Понаблюдай за этим человеком")

    assert intent.name == "track_person"
    assert intent.confidence == "llm_validated"


def test_build_command_interpreter_returns_ollama_fallback_mode() -> None:
    interpreter = build_command_interpreter("Deterministic + Ollama fallback", ollama_model="qwen2.5:3b")

    intent = interpreter.interpret("Следи за человеком")

    assert intent.name == "track_person"
    assert intent.confidence == "deterministic"


def test_build_command_interpreter_rejects_unknown_mode() -> None:
    try:
        build_command_interpreter("Unknown")
    except ValueError as error:
        assert "Unsupported command interpreter mode" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_render_command_preview_uses_supplied_interpreter() -> None:
    interpreter = build_command_interpreter("Deterministic + mock LLM fallback")

    intent = render_command_preview("Понаблюдай за этим человеком", interpreter)

    assert isinstance(intent, CommandIntent)
    assert intent.name == "track_person"


def test_available_pages_depend_on_role() -> None:
    assert available_pages("admin") == (
        "Обзор",
        "Анализ видео",
        "История анализов",
        "Пользователи",
        "Журнал действий",
    )
    assert available_pages("operator") == ("Обзор", "Анализ видео", "История анализов")
    assert available_pages("viewer") == ("Обзор", "История анализов")


def test_role_capabilities_depend_on_role() -> None:
    assert len(role_capabilities("admin")) == 3
    assert len(role_capabilities("operator")) == 2
    assert role_capabilities("viewer")[0][0] == "История"


def test_filter_analysis_runs_combines_filters() -> None:
    runs = [
        {
            "video_name": "entrance.mp4",
            "command": "Посчитай людей",
            "username": "operator",
            "status": "completed",
        },
        {
            "video_name": "parking.mp4",
            "command": "Следи за машиной",
            "username": "admin",
            "status": "failed",
        },
    ]

    assert filter_analysis_runs(
        runs,
        query="людей",
        author="operator",
        status="completed",
    ) == [runs[0]]


def test_users_dataframe_translates_role_and_status() -> None:
    dataframe = users_dataframe(
        [
            {
                "id": 3,
                "username": "viewer",
                "role": "viewer",
                "is_active": False,
            }
        ]
    )

    assert dataframe.iloc[0].to_dict() == {
        "ID": 3,
        "Пользователь": "viewer",
        "Роль": "Наблюдатель",
        "Статус": "Заблокирован",
    }


def test_audit_dataframe_formats_optional_values() -> None:
    dataframe = audit_dataframe(
        [
            {
                "created_at": "2026-06-08T12:00:00+00:00",
                "username": None,
                "action": "login_failed",
                "details": None,
            }
        ]
    )

    assert dataframe.iloc[0]["Пользователь"] == "-"
    assert dataframe.iloc[0]["Подробности"] == "-"


def test_analysis_runs_dataframe_contains_demo_columns() -> None:
    dataframe = analysis_runs_dataframe(
        [
            {
                "id": 4,
                "created_at": "2026-06-08T12:00:00+00:00",
                "video_name": "sample.mp4",
                "command": "Посчитай людей",
                "username": "operator",
                "status": "completed",
                "processed_frames": 60,
                "event_count": 3,
            }
        ]
    )

    assert dataframe.iloc[0]["Видео"] == "sample.mp4"
    assert dataframe.iloc[0]["Автор"] == "operator"
    assert dataframe.iloc[0]["События"] == 3


def test_event_dicts_to_dataframe_flattens_metadata() -> None:
    dataframe = event_dicts_to_dataframe(
        [
            {
                "event_type": "count_in_frame",
                "frame_index": 10,
                "metadata": {"count": 2},
            }
        ]
    )

    assert dataframe.iloc[0]["metadata.count"] == 2
