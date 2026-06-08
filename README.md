# MultiModalCV

MultiModalCV is an MVP project for a text-controlled computer vision system.

The project is currently focused on archived video analysis. Live camera support is intentionally out of scope for early versions.

## Current Status

The repository already contains the first internal MVP layers:

- core domain models for detections, tracks, zones, and events;
- restricted Russian text command parser;
- zone transition rules;
- zone occupancy rules;
- rule evaluation engine;
- JSON event reporting;
- synthetic CLI demo;
- unit tests.

No real video processing or neural network inference is implemented yet. The current CLI demo uses synthetic tracks to verify the command-to-event pipeline.

## Supported Demo Commands

Examples:

```text
Сообщи, когда человек войдет в зону
Сообщи, когда человек выйдет из зоны
Сообщи, когда все люди покинут зону
Посчитай людей в зоне
Посчитай людей в кадре
Следи за человеком
Следи за машиной
```

Unsupported commands are rejected explicitly.

## Setup

Create a local virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```powershell
python -m pip install -e .[dev]
```

Install optional YOLO dependencies for real object detection:

```powershell
python -m pip install -e .[yolo]
```

Install optional UI dependencies:

```powershell
python -m pip install -e .[ui]
```

If `python` is not available in PATH, use the Python executable installed on your machine.

## Run Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Current expected result:

```text
44 passed
```

## Run Synthetic CLI Demo

```powershell
.venv\Scripts\python.exe -m multimodalcv.cli.demo "Сообщи, когда человек войдет в зону" --output outputs\demo_events.json
```

Expected output:

```text
Wrote 1 event(s) to outputs\demo_events.json
```

The generated JSON report contains structured events such as:

```json
[
  {
    "event_type": "enter_zone",
    "frame_index": 2,
    "timestamp_sec": 0.2,
    "zone_name": "main",
    "track_id": 1,
    "object_class": "person",
    "metadata": {}
  }
]
```

## Run Video Inspection

```powershell
.venv\Scripts\multimodalcv-video.exe path\to\video.mp4 --output-dir outputs\video_inspect --max-frames 5
```

This writes video metadata and extracted sample frames.

## Run Video Analysis

Fake detector mode:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Сообщи, когда человек войдет в зону" --output outputs\analyze\events.json
```

Count people in the whole frame without zone filtering:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Посчитай людей в кадре" --detector yolo --model yolov8n.pt --max-frames 100 --output outputs\frame_count\events.json --save-frames --frames-dir outputs\frame_count\frames
```

YOLO detector mode:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Посчитай людей в зоне" --detector yolo --model yolov8n.pt --max-frames 100 --output outputs\yolo\events.json
```

Set a custom rectangular zone:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Посчитай людей в зоне" --detector yolo --model yolov8n.pt --zone-rect 100,50,500,400 --max-frames 100 --output outputs\yolo\events.json
```

Save annotated diagnostic frames:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Посчитай людей в зоне" --detector yolo --model yolov8n.pt --zone-rect 100,50,500,400 --max-frames 100 --output outputs\yolo\events.json --save-frames --frames-dir outputs\yolo\frames
```

Annotated frames include the current zone, detections, tracks, object classes, confidence values, and track IDs.

Save only frames that produced events:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Сообщи, когда человек войдет в зону" --detector yolo --model yolov8n.pt --zone-rect 100,50,500,400 --max-frames 100 --output outputs\events_only\events.json --save-frames --frames-dir outputs\events_only\frames --frame-mode events
```

Smooth short count spikes:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Посчитай людей в кадре" --detector yolo --model yolov8n.pt --max-frames 100 --count-window-size 5 --output outputs\smoothed_count\events.json
```

Suppress repeated non-count events:

```powershell
.venv\Scripts\multimodalcv-analyze.exe path\to\video.mp4 "Сообщи, когда человек войдет в зону" --detector yolo --model yolov8n.pt --zone-rect 100,50,500,400 --max-frames 100 --event-cooldown-sec 2.0 --output outputs\cooldown\events.json
```

The analysis CLI prints a short summary:

```text
Processed frame(s): 100
Detector: yolo
Wrote 100 event(s) to outputs\frame_count\events.json
Wrote summary to outputs\frame_count\summary.json
Saved 100 annotated frame(s) to outputs\frame_count\frames
Events:
- 00:00.00 count_in_frame person count=2
- 00:00.10 count_in_frame person count=2
```

Each analysis run also writes `summary.json` next to `events.json` by default. Use `--summary-output` to choose another path.

## Run Streamlit Interface

Start the FastAPI backend first:

```powershell
.venv\Scripts\uvicorn.exe multimodalcv.api.app:app --host 127.0.0.1 --port 8000
```

Then start the interface in another terminal:

```powershell
.venv\Scripts\streamlit.exe run interfaces\streamlit_app\app.py
```

Then open:

```text
http://localhost:8501
```

The interface starts with a login page and builds its navigation from the current
role. Administrators can manage users and view the audit log. Operators can run
video analysis. Viewers have read-only access to the overview and saved analysis
history.

The analysis page supports video upload, text commands, YOLO/fake detector mode,
zone rectangle settings, count smoothing, event cooldown, summary metrics, event
tables, and annotated frame previews.

After upload, the interface reads video metadata and configures zones using the actual frame size. Zone modes:

- `Full frame`;
- `Center 50%`;
- `Custom` with bounded coordinate inputs.

The first YOLO run may download model weights such as `yolov8n.pt`. Model weights are intentionally ignored by git.

## Run Authentication API

Install the web dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e .[web]
```

Create the first administrator:

```powershell
.venv\Scripts\multimodalcv-create-admin.exe admin
```

Start the local FastAPI server:

```powershell
.venv\Scripts\uvicorn.exe multimodalcv.api.app:app --reload
```

The API is then available at:

```text
http://localhost:8000
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

The authentication API currently provides:

- `POST /auth/login` - create a user session;
- `GET /auth/me` - return the current user and role;
- `POST /auth/logout` - revoke the current session;
- `GET /users` - list users, administrator only;
- `POST /users` - create users, administrator only;
- `PATCH /users/{id}` - change a role or active state, administrator only;
- `POST /users/{id}/reset-password` - reset a password, administrator only;
- `GET /audit` - view the audit log, administrator only.
- `POST /analysis-runs` - save a completed analysis, administrator or operator;
- `GET /analysis-runs` - view analysis history, all authenticated roles;
- `GET /analysis-runs/{id}` - view a saved result, all authenticated roles.

Supported roles are `admin`, `operator`, and `viewer`. Role checks are enforced by
the backend, including when the API is called directly.

## Project Layout

```text
multimodalcv/
  cli/          synthetic command-line demo
  commands/     restricted text command parser
  core/         shared domain models
  detection/    planned object detection modules
  reporting/    JSON event reporting
  rules/        event and occupancy rules
  segmentation/ planned SAM integration
  tracking/     planned object tracking modules
  video/        planned video file processing modules

tests/
  unit/         fast unit tests
  integration/ planned integration tests
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).
