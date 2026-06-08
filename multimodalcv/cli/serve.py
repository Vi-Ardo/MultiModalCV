"""Run the MultiModalCV API and Streamlit interface together."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_UI_PORT = 8501


def build_commands(
    *,
    python_executable: str,
    project_root: Path,
    host: str,
    api_port: int,
    ui_port: int,
) -> tuple[list[str], list[str]]:
    api_command = [
        python_executable,
        "-m",
        "uvicorn",
        "multimodalcv.api.app:app",
        "--host",
        host,
        "--port",
        str(api_port),
    ]
    ui_command = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(project_root / "interfaces" / "streamlit_app" / "app.py"),
        "--server.address",
        host,
        "--server.port",
        str(ui_port),
        "--server.headless",
        "true",
    ]
    return api_command, ui_command


def wait_for_port(host: str, port: int, *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MultiModalCV web application.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    api_command, ui_command = build_commands(
        python_executable=sys.executable,
        project_root=project_root,
        host=args.host,
        api_port=args.api_port,
        ui_port=args.ui_port,
    )
    environment = None
    if args.api_port != DEFAULT_API_PORT:
        import os

        environment = os.environ.copy()
        environment["MULTIMODALCV_API_URL"] = f"http://{args.host}:{args.api_port}"

    print("Starting MultiModalCV...")
    processes = [
        subprocess.Popen(api_command, cwd=project_root, env=environment),
        subprocess.Popen(ui_command, cwd=project_root, env=environment),
    ]

    try:
        api_ready = wait_for_port(args.host, args.api_port)
        ui_ready = wait_for_port(args.host, args.ui_port)
        if not api_ready or not ui_ready:
            print("The services did not start in time.")
            return 1

        ui_url = f"http://{args.host}:{args.ui_port}"
        print(f"API: {args.host}:{args.api_port}")
        print(f"Interface: {ui_url}")
        print("Press Ctrl+C to stop both services.")
        if not args.no_browser:
            webbrowser.open(ui_url)

        while all(process.poll() is None for process in processes):
            time.sleep(0.5)

        failed_process = next(process for process in processes if process.poll() is not None)
        return failed_process.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping MultiModalCV...")
        return 0
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
