# Demo Guide

This guide describes a compact demo flow for the current MVP.

The goal is to show that the system can:

- accept text commands;
- use deterministic parsing and local LLM fallback;
- run YOLO on archived video files;
- produce events, summary metrics, and annotated frames.

## Before the Demo

Start the Streamlit interface:

```powershell
.venv\Scripts\streamlit.exe run interfaces\streamlit_app\app.py
```

Open:

```text
http://localhost:8501
```

For the local LLM mode, keep the Ollama app running and make sure the model is installed:

```text
qwen2.5:3b
```

The app uses Ollama through:

```text
http://localhost:11434
```

## Recommended Demo Videos

Use the current local sample videos:

```text
data/samples/person_walks.mp4
data/samples/two_people.mp4
```

They are intentionally simple and short, which is useful for a live demonstration.

## Scenario 1: Count People

Video:

```text
data/samples/two_people.mp4
```

Settings:

```text
Command interpreter: Deterministic
Command: Сколько человек на видео
Detector: yolo
Confidence: 0.45
Zone mode: Full frame
Max frames: 60
Count smoothing window: 5
Annotated frames: events
```

Expected behavior:

```text
The app produces count_in_frame events and annotated frames.
The summary should show max people around 2.
Annotated frames show only count changes, not every processed frame.
```

## Scenario 2: Person Enters a Zone

Video:

```text
data/samples/person_walks.mp4
```

Settings:

```text
Command interpreter: Deterministic
Command: Уведоми, если человек зайдет в комнату
Detector: yolo
Confidence: 0.45
Zone mode: Center 50%
Max frames: 90
Event cooldown, sec: 2.0
Annotated frames: events
```

Expected behavior:

```text
The app produces one enter_zone event when the person crosses into the central zone.
```

Validated CLI result:

```text
Processed frame(s): 90
Wrote 1 event(s)
00:02.70 enter_zone person
```

## Scenario 3: Local LLM Fallback

Video:

```text
data/samples/person_walks.mp4
```

Settings:

```text
Command interpreter: Deterministic + Ollama fallback
Ollama model: qwen2.5:3b
Command: Понаблюдай за этим человеком
Detector: yolo
Confidence: 0.45
Zone mode: Full frame
Max frames: 30
Annotated frames: events
```

Expected behavior:

```text
The deterministic parser does not handle this wording.
Ollama maps the command to track_person.
The app shows llm_validated in the command preview.
The analysis produces a track_object event.
```

## Optional Scenario: Deterministic Tracking

Video:

```text
data/samples/person_walks.mp4
```

Settings:

```text
Command interpreter: Deterministic
Command: Следи за человеком
Detector: yolo
Confidence: 0.45
Zone mode: Full frame
Max frames: 30
Annotated frames: events
```

Validated CLI result:

```text
Processed frame(s): 30
Wrote 1 event(s)
00:00.27 track_object person
```

## Picking Additional Videos

For the next demo videos, prefer:

- short clips, 5-20 seconds;
- stable camera;
- people or cars clearly visible;
- minimal motion blur;
- objects not too small in the frame;
- a visible region crossing if demonstrating enter/leave-zone commands.

Avoid for the first demonstration:

- crowded scenes;
- very dark footage;
- heavy occlusion;
- fast camera movement;
- tiny distant people or cars.
