"""Domain models for users, sessions, roles, and audit records."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


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
