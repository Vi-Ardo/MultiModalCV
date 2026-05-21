"""Streamlit interface for MultiModalCV."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from multimodalcv.cli.analyze import AnalyzeResult, analyze_video
from multimodalcv.commands.interpreter import (
    CommandInterpreter,
    DeterministicCommandInterpreter,
    FallbackCommandInterpreter,
)
from multimodalcv.commands.llm_interpreter import JSONLLMCommandInterpreter, MockLLMResponseProvider
from multimodalcv.commands.llm_interpreter import OllamaResponseProvider
from multimodalcv.commands.parser import (
    CommandIntent,
    UnsupportedCommandError,
    supported_command_examples,
)
from multimodalcv.config.zones import ZoneConfigError
from multimodalcv.core.models import Event
from multimodalcv.detection.yolo import YOLODependencyError
from multimodalcv.reporting.json_report import event_to_dict
from multimodalcv.video.reader import VideoMetadata, VideoOpenError, read_video_metadata


RUNS_DIR = Path("outputs/streamlit")
DEFAULT_COMMAND = "Посчитай людей в кадре"
INTERPRETER_MODES = (
    "Deterministic",
    "Deterministic + mock LLM fallback",
    "Deterministic + Ollama fallback",
)
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


def main() -> None:
    st.set_page_config(page_title="MultiModalCV", layout="wide")
    inject_styles()

    st.title("MultiModalCV")

    with st.sidebar:
        uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        metadata = None
        if uploaded_file:
            preview_video_path = save_uploaded_video(uploaded_file, RUNS_DIR / "_preview")
            try:
                metadata = read_video_metadata(preview_video_path)
                render_video_metadata(metadata)
            except VideoOpenError as error:
                st.error(str(error))

        interpreter_mode = st.selectbox("Command interpreter", INTERPRETER_MODES, index=0)
        ollama_model = DEFAULT_OLLAMA_MODEL
        if interpreter_mode == "Deterministic + Ollama fallback":
            ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
        command_interpreter = build_command_interpreter(interpreter_mode, ollama_model=ollama_model)
        command = st.text_input("Command", value=DEFAULT_COMMAND)
        render_command_preview(command, command_interpreter)
        detector = st.selectbox("Detector", ["yolo", "fake"], index=0)
        model_path = st.text_input("YOLO model", value="yolov8n.pt")
        max_frame_limit = metadata.frame_count if metadata and metadata.frame_count > 0 else 5000
        max_frames = st.number_input(
            "Max frames",
            min_value=1,
            max_value=max_frame_limit,
            value=min(100, max_frame_limit),
            step=10,
        )
        zone_rect = render_zone_controls(metadata)
        confidence = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        count_window_size = st.number_input("Count smoothing window", min_value=1, max_value=31, value=5, step=2)
        event_cooldown_sec = st.number_input("Event cooldown, sec", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        frame_mode = st.selectbox("Annotated frames", ["events", "all"], index=0)
        run_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if not uploaded_file:
        render_empty_state()
        return

    if run_clicked:
        run_dir = RUNS_DIR / uuid4().hex[:12]
        video_path = save_uploaded_video(uploaded_file, run_dir)
        with st.spinner("Analyzing video..."):
            try:
                result = analyze_video(
                    video_path=video_path,
                    command=command,
                    output_path=run_dir / "events.json",
                    summary_output_path=run_dir / "summary.json",
                    max_frames=int(max_frames),
                    detector_name=detector,
                    model_path=Path(model_path),
                    confidence_threshold=float(confidence),
                    save_frames=True,
                    frames_dir=run_dir / "frames",
                    frame_mode=frame_mode,
                    zone_rect=zone_rect,
                    count_window_size=int(count_window_size),
                    event_cooldown_sec=float(event_cooldown_sec),
                    command_interpreter=command_interpreter,
                )
            except (UnsupportedCommandError, VideoOpenError, ValueError, YOLODependencyError, ZoneConfigError) as error:
                st.error(str(error))
                return

        st.session_state["last_run"] = {
            "run_dir": run_dir,
            "result": result,
        }

    last_run = st.session_state.get("last_run")
    if last_run:
        render_results(last_run["result"], last_run["run_dir"])
    else:
        render_ready_state(uploaded_file.name)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.info("Upload an archived video file to start analysis.")


def render_ready_state(filename: str) -> None:
    st.success(f"Ready to analyze: {filename}")


def build_command_interpreter(
    mode: str,
    *,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> CommandInterpreter:
    if mode == "Deterministic":
        return DeterministicCommandInterpreter()

    if mode == "Deterministic + mock LLM fallback":
        return FallbackCommandInterpreter(
            (
                DeterministicCommandInterpreter(),
                JSONLLMCommandInterpreter(MockLLMResponseProvider()),
            )
        )

    if mode == "Deterministic + Ollama fallback":
        return FallbackCommandInterpreter(
            (
                DeterministicCommandInterpreter(),
                JSONLLMCommandInterpreter(OllamaResponseProvider(model=ollama_model)),
            )
        )

    raise ValueError(f"Unsupported command interpreter mode: {mode}")


def render_command_preview(
    command: str,
    command_interpreter: CommandInterpreter | None = None,
) -> CommandIntent | None:
    interpreter = command_interpreter or build_command_interpreter("Deterministic")
    try:
        intent = interpreter.interpret(command)
    except UnsupportedCommandError:
        st.warning("Команда пока не поддерживается.")
        with st.expander("Примеры поддерживаемых команд"):
            for example in supported_command_examples():
                st.write(f"- {example}")
        return None

    st.success(intent.summary)
    st.caption(
        "Распознано как: "
        f"`{intent.name}` / `{intent.rule.event_type.value}` / "
        f"`{intent.rule.object_class.value}` / зона `{intent.rule.zone_name}` / "
        f"`{intent.confidence}`"
    )
    return intent


def render_video_metadata(metadata: VideoMetadata) -> None:
    st.caption(
        f"Video: {metadata.width}x{metadata.height}, "
        f"{metadata.frame_count} frames, {metadata.fps:.2f} FPS, "
        f"{metadata.duration_sec:.2f} sec"
    )


def render_zone_controls(metadata: VideoMetadata | None) -> str:
    st.markdown("**Zone**")

    if metadata is None:
        st.caption("Upload a video to configure the zone by frame size.")
        return st.text_input("Zone rectangle", value="0,0,320,240")

    zone_mode = st.selectbox("Zone mode", ["Full frame", "Center 50%", "Custom"], index=0)

    if zone_mode == "Full frame":
        zone_rect = full_frame_zone_rect(metadata)
        st.caption(f"Using full frame: `{zone_rect}`")
        return zone_rect

    if zone_mode == "Center 50%":
        zone_rect = center_zone_rect(metadata)
        st.caption(f"Using center region: `{zone_rect}`")
        return zone_rect

    x1_default, y1_default, x2_default, y2_default = zone_rect_values(center_zone_rect(metadata))
    left, right = st.columns(2)
    with left:
        x1 = st.number_input("Zone x1", min_value=0, max_value=metadata.width - 1, value=x1_default)
        y1 = st.number_input("Zone y1", min_value=0, max_value=metadata.height - 1, value=y1_default)
    with right:
        x2 = st.number_input("Zone x2", min_value=1, max_value=metadata.width, value=x2_default)
        y2 = st.number_input("Zone y2", min_value=1, max_value=metadata.height, value=y2_default)

    zone_rect = format_zone_rect(x1, y1, x2, y2)
    st.caption(f"Custom zone: `{zone_rect}`")
    return zone_rect


def render_results(result: AnalyzeResult, run_dir: Path) -> None:
    summary = load_summary(result.summary_path)
    render_metrics(summary)

    st.subheader("Events")
    events_df = events_to_dataframe(result.events)
    if events_df.empty:
        st.info("No events produced for this run.")
    else:
        st.dataframe(events_df, use_container_width=True, hide_index=True)

    st.subheader("Annotated Frames")
    frame_paths = sorted((result.frames_dir or (run_dir / "frames")).glob("*.jpg"))
    if not frame_paths:
        st.info("No annotated frames were saved for this run.")
    else:
        render_frame_viewer(frame_paths)
        render_frame_gallery(frame_paths)

    with st.expander("Output files"):
        st.write(f"Run directory: `{run_dir}`")
        st.write(f"Events: `{result.output_path}`")
        st.write(f"Summary: `{result.summary_path}`")


def render_metrics(summary: dict) -> None:
    counts = summary.get("counts", {})
    metric_columns = st.columns(5)
    metric_columns[0].metric("Frames", summary.get("processed_frames", 0))
    metric_columns[1].metric("Events", summary.get("event_count", 0))
    metric_columns[2].metric("Max people", counts.get("max_count", "-"))
    metric_columns[3].metric("Common count", counts.get("most_common_count", "-"))
    metric_columns[4].metric("Annotated", summary.get("annotated_frame_count", 0))

    if counts:
        st.caption(f"Count distribution: {counts.get('count_distribution', {})}")


def render_frame_viewer(frame_paths: list[Path]) -> None:
    selected_name = st.selectbox(
        "Annotated frame",
        [frame_path.name for frame_path in frame_paths],
        index=0,
    )
    selected_path = find_frame_by_name(frame_paths, selected_name)
    st.image(str(selected_path), caption=selected_path.name, use_container_width=True)


def find_frame_by_name(frame_paths: list[Path], frame_name: str) -> Path:
    return next(frame_path for frame_path in frame_paths if frame_path.name == frame_name)


def render_frame_gallery(frame_paths: list[Path]) -> None:
    st.markdown("**Frame overview**")
    preview_count = min(12, len(frame_paths))
    columns = st.columns(3)
    for index, frame_path in enumerate(frame_paths[:preview_count]):
        with columns[index % 3]:
            st.image(str(frame_path), caption=frame_path.name, use_container_width=True)

    if len(frame_paths) > preview_count:
        st.caption(f"Showing {preview_count} of {len(frame_paths)} annotated frames.")


def save_uploaded_video(uploaded_file, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    video_path = run_dir / f"input{suffix}"
    video_path.write_bytes(uploaded_file.getbuffer())
    return video_path


def full_frame_zone_rect(metadata: VideoMetadata) -> str:
    return format_zone_rect(0, 0, metadata.width, metadata.height)


def center_zone_rect(metadata: VideoMetadata) -> str:
    x_margin = metadata.width // 4
    y_margin = metadata.height // 4
    return format_zone_rect(
        x_margin,
        y_margin,
        metadata.width - x_margin,
        metadata.height - y_margin,
    )


def format_zone_rect(x1: int, y1: int, x2: int, y2: int) -> str:
    return f"{x1},{y1},{x2},{y2}"


def zone_rect_values(zone_rect: str) -> tuple[int, int, int, int]:
    values = tuple(int(float(part)) for part in zone_rect.split(","))
    if len(values) != 4:
        raise ValueError("zone rectangle must contain four values")
    return values


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def events_to_dataframe(events: list[Event]) -> pd.DataFrame:
    rows = [event_to_dict(event) for event in events]
    if not rows:
        return pd.DataFrame()

    for row in rows:
        metadata = row.pop("metadata", {}) or {}
        for key, value in metadata.items():
            row[f"metadata.{key}"] = value

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
