"""Domain models for users, sessions, roles, and audit records."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Roles supported by the multi-user application."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: Role
    is_active: bool
    created_at: datetime


@dataclass(frozen=True)
class Session:
    token: str
    user_id: int
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class AuditEntry:
    id: int
    action: str
    created_at: datetime
    user_id: int | None = None
    username: str | None = None
    details: str | None = None


@dataclass(frozen=True)
class AnalysisRun:
    id: int
    user_id: int
    username: str
    video_name: str
    command: str
    detector: str
    status: str
    processed_frames: int
    event_count: int
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    frame_paths: list[str]
    created_at: datetime
