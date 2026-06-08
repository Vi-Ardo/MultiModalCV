"""SQLite persistence for authentication and audit data."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from multimodalcv.auth.models import AuditEntry, Role, Session, User
from multimodalcv.auth.security import generate_session_token, hash_password, verify_password


DEFAULT_DATABASE_PATH = Path("outputs/server/multimodalcv.db")


class AuthenticationError(ValueError):
    """Raised when supplied credentials are invalid."""


class DuplicateUsernameError(ValueError):
    """Raised when attempting to create an existing username."""


class AuthStore:
    """Small SQLite repository used by the local demonstration server."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    username TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at);
                """
            )
            connection.commit()

    def create_user(self, username: str, password: str, role: Role) -> User:
        normalized_username = normalize_username(username)
        created_at = utc_now()
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, password_hash, role, is_active, created_at)
                    VALUES (?, ?, ?, 1, ?)
                    """,
                    (normalized_username, hash_password(password), role.value, created_at.isoformat()),
                )
                user_id = int(cursor.lastrowid)
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise DuplicateUsernameError(f"username already exists: {normalized_username}") from error

        user = User(user_id, normalized_username, role, True, created_at)
        self.record_audit("user_created", user=user, details=f"role={role.value}")
        return user

    def list_users(self) -> list[User]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, username, role, is_active, created_at FROM users ORDER BY username"
            ).fetchall()
        return [_user_from_row(row) for row in rows]

    def authenticate(self, username: str, password: str) -> User:
        normalized_username = normalize_username(username)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, username, password_hash, role, is_active, created_at
                FROM users
                WHERE username = ? COLLATE NOCASE
                """,
                (normalized_username,),
            ).fetchone()

        if row is None or not bool(row["is_active"]) or not verify_password(password, row["password_hash"]):
            self.record_audit("login_failed", username=normalized_username)
            raise AuthenticationError("invalid username or password")

        user = _user_from_row(row)
        self.record_audit("login_succeeded", user=user)
        return user

    def create_session(self, user: User, *, lifetime: timedelta = timedelta(hours=8)) -> Session:
        created_at = utc_now()
        session = Session(
            token=generate_session_token(),
            user_id=user.id,
            created_at=created_at,
            expires_at=created_at + lifetime,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session.token,
                    session.user_id,
                    session.created_at.isoformat(),
                    session.expires_at.isoformat(),
                ),
            )
            connection.commit()
        return session

    def get_user_by_session(self, token: str) -> User | None:
        now = utc_now()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT u.id, u.username, u.role, u.is_active, u.created_at, s.expires_at
                FROM sessions AS s
                JOIN users AS u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()

            if row is None:
                return None
            if not bool(row["is_active"]) or datetime.fromisoformat(row["expires_at"]) <= now:
                connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
                connection.commit()
                return None

        return _user_from_row(row)

    def revoke_session(self, token: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()

    def record_audit(
        self,
        action: str,
        *,
        user: User | None = None,
        username: str | None = None,
        details: str | None = None,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO audit_log (user_id, username, action, details, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user.id if user else None,
                    user.username if user else username,
                    action,
                    details,
                    utc_now().isoformat(),
                ),
            )
            connection.commit()

    def list_audit_entries(self, *, limit: int = 100) -> list[AuditEntry]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, username, action, details, created_at
                FROM audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            AuditEntry(
                id=int(row["id"]),
                user_id=row["user_id"],
                username=row["username"],
                action=row["action"],
                details=row["details"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def normalize_username(username: str) -> str:
    normalized = username.strip().casefold()
    if len(normalized) < 3:
        raise ValueError("username must contain at least 3 characters")
    return normalized


def utc_now() -> datetime:
    return datetime.now(UTC)


def _user_from_row(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        username=row["username"],
        role=Role(row["role"]),
        is_active=bool(row["is_active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
