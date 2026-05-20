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
