from pathlib import Path

from multimodalcv.cli.serve import build_commands


def test_build_commands_runs_api_and_streamlit_from_same_python() -> None:
    api_command, ui_command = build_commands(
        python_executable="python",
        project_root=Path("C:/project"),
        host="127.0.0.1",
        api_port=8001,
        ui_port=8502,
    )

    assert api_command == [
        "python",
        "-m",
        "uvicorn",
        "multimodalcv.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ]
    assert ui_command[:4] == ["python", "-m", "streamlit", "run"]
    assert ui_command[-6:] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8502",
        "--server.headless",
        "true",
    ]
