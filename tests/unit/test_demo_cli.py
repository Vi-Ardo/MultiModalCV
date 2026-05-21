import json

from multimodalcv.cli.demo import main, run_demo
from multimodalcv.core.models import EventType


def test_run_demo_writes_enter_event_json(tmp_path) -> None:
    output_path = tmp_path / "events.json"

    events = run_demo(
        command="Сообщи, когда человек войдет в зону",
        output_path=output_path,
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ENTER_ZONE

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["event_type"] == "enter_zone"
    assert data[0]["object_class"] == "person"


def test_run_demo_writes_count_event_json(tmp_path) -> None:
    output_path = tmp_path / "events.json"

    events = run_demo(command="Посчитай людей в зоне", output_path=output_path)

    assert len(events) == 1
    assert events[0].event_type == EventType.COUNT_IN_ZONE
    assert events[0].metadata["count"] == 1


def test_main_returns_zero_for_supported_command(tmp_path, capsys) -> None:
    output_path = tmp_path / "events.json"

    exit_code = main(["Следи за машиной", "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote 1 event(s)" in captured.out
    assert output_path.exists()


def test_main_returns_error_for_unsupported_command(capsys) -> None:
    exit_code = main(["Определи подозрительное поведение"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Команда пока не поддерживается" in captured.out
    assert "Supported commands" in captured.out
