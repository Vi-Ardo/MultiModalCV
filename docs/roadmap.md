# MultiModalCV Roadmap

## Project Goal

MultiModalCV is a computer vision application for analyzing archived video files using restricted text commands.

The long-term goal is to build a modular system that can detect, track, segment, and report events in video. The first versions focus on archived files only. Live camera support is intentionally out of scope for now.

## MVP Scope

The first MVP should process one video file and one supported text command at a time.

Input:

- archived video file;
- predefined text command;
- predefined area of interest;
- basic runtime parameters.

Processing:

- read video frames;
- detect supported object classes;
- track detected objects across frames;
- evaluate event rules;
- save event evidence.

Output:

- event log in JSON or CSV format;
- annotated event frames;
- optionally annotated output video;
- short textual summary of detected events.

## Supported MVP Objects

Initial object classes:

- person;
- car.

The first implementation may focus only on `person` if that makes the MVP more stable.

## Supported MVP Commands

The MVP should support a limited set of natural-language-like commands.

Initial command set:

- report when a person enters a zone;
- report when a person leaves a zone;
- report when all people leave a zone;
- count people inside a zone;
- track a person;
- track a car.

Unsupported commands should produce a clear message instead of trying to guess the user's intent.

## Area Of Interest

For the first version, an area of interest can be defined manually through configuration.

Initial zone formats:

- rectangle;
- polygon, if the implementation cost is reasonable.

Interactive zone drawing is deferred until a later interface stage.

## Initial Architecture

High-level pipeline:

```text
Video file
  -> frame reader
  -> object detector
  -> object tracker
  -> scene state
  -> rule engine
  -> event report
```

Text command pipeline:

```text
Text command
  -> restricted command parser
  -> structured rule
  -> rule engine
```

SAM integration is planned as a later module:

```text
Detector bbox
  -> SAM segmentation
  -> object mask
  -> refined zone/event checks
```

## Development Stages

### Stage 1: Project Skeleton

- define repository structure;
- add core configuration files;
- add documentation for MVP scope;
- prepare basic CLI entry point.

### Stage 2: Video Processing Core

- read archived video files;
- sample or iterate frames;
- save selected frames;
- produce basic processing metadata.

### Stage 3: Object Detection

- integrate an object detector;
- support at least person detection;
- save annotated frames for verification.

### Stage 4: Tracking

- assign stable IDs to detected objects;
- keep object state across frames;
- expose track history to the rule engine.

### Stage 5: Rules And Events

- implement zone membership checks;
- detect entry and exit events;
- detect empty-zone state;
- save events to JSON or CSV.

### Stage 6: Restricted Text Commands

- parse supported command templates;
- map commands to structured rules;
- reject unsupported commands clearly.

### Stage 7: SAM Integration

- run SAM from detector boxes;
- produce object masks;
- refine zone intersection checks;
- improve visual event evidence.

### Stage 8: Interface

- start with CLI for reliability;
- add Streamlit or desktop interface after the core is stable;
- keep UI separate from core processing logic.

## Explicit Non-Goals For Early Versions

The following features are intentionally deferred:

- live camera or RTSP support;
- fully free-form natural language;
- multi-camera processing;
- interactive zone drawing;
- user authentication;
- production deployment;
- mobile application;
- real-time alerting.

## Technical Principles

- Keep the core independent from any interface.
- Prefer simple, inspectable outputs before complex UI.
- Add SAM after detection, tracking, and events are working.
- Make command handling predictable rather than overly flexible.
- Keep each milestone demonstrable with a short video example.

