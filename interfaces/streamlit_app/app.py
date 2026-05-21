"""Streamlit interface for MultiModalCV."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from multimodalcv.cli.analyze import AnalyzeResult, analyze_video
from multimodalcv.commands.parser import UnsupportedCommandError, supported_commands
from multimodalcv.config.zones import ZoneConfigError
from multimodalcv.core.models import Event
from multimodalcv.detection.yolo import YOLODependencyError
from multimodalcv.reporting.json_report import event_to_dict
from multimodalcv.video.reader import VideoOpenError


RUNS_DIR = Path("outputs/streamlit")
DEFAULT_COMMAND = "Посчитай людей в кадре"


def main() -> None:
    st.set_page_config(page_title="MultiModalCV", layout="wide")
    inject_styles()

    st.title("MultiModalCV")

    with st.sidebar:
        uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        command = st.selectbox("Command", supported_commands(), index=_default_command_index())
        detector = st.selectbox("Detector", ["yolo", "fake"], index=0)
        model_path = st.text_input("YOLO model", value="yolov8n.pt")
        max_frames = st.number_input("Max frames", min_value=1, max_value=5000, value=100, step=10)
        zone_rect = st.text_input("Zone rectangle", value="0,0,320,240")
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


def render_frame_gallery(frame_paths: list[Path]) -> None:
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


def _default_command_index() -> int:
    commands = supported_commands()
    if DEFAULT_COMMAND in commands:
        return commands.index(DEFAULT_COMMAND)
    return 0


if __name__ == "__main__":
    main()
