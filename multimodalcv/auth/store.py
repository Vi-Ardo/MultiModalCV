"""SQLite persistence for authentication and audit data."""

from __future__ import annotations

import sqlite3
import json
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from multimodalcv.auth.models import AnalysisRun, AuditEntry, Role, Session, User
from multimodalcv.auth.security import generate_session_token, hash_password, verify_password


DEFAULT_DATABASE_PATH = Path("outputs/server/multimodalcv.db")


class AuthenticationError(ValueError):
    """Raised when supplied credentials are invalid."""


class DuplicateUsernameError(ValueError):
    """Raised when attempting to create an existing username."""


class UserNotFoundError(ValueError):
    """Raised when a requested user does not exist."""


class AnalysisRunNotFoundError(ValueError):
    """Raised when a requested analysis run does not exist."""


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

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
                    username TEXT NOT NULL,
                    video_name TEXT NOT NULL,
                    command TEXT NOT NULL,
                    detector TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_frames INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    events_json TEXT NOT NULL,
                    frame_paths_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
                    ON analysis_runs(created_at);
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

    def get_user(self, user_id: int) -> User:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, username, role, is_active, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            raise UserNotFoundError(f"user not found: {user_id}")
        return _user_from_row(row)

    def update_user(
        self,
        user_id: int,
        *,
        role: Role | None = None,
        is_active: bool | None = None,
    ) -> User:
        current_user = self.get_user(user_id)
        updated_role = role or current_user.role
        updated_active = current_user.is_active if is_active is None else is_active

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE users
                SET role = ?, is_active = ?
                WHERE id = ?
                """,
                (updated_role.value, int(updated_active), user_id),
            )
            if not updated_active:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            connection.commit()

        return User(
            id=current_user.id,
            username=current_user.username,
            role=updated_role,
            is_active=updated_active,
            created_at=current_user.created_at,
        )

    def reset_password(self, user_id: int, password: str) -> None:
        self.get_user(user_id)
        password_hash = hash_password(password)
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            connection.commit()

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

    def create_analysis_run(
        self,
        *,
        user: User,
        video_name: str,
        command: str,
        detector: str,
        status: str,
        processed_frames: int,
        event_count: int,
        summary: dict,
        events: list[dict],
        frame_paths: list[str],
    ) -> AnalysisRun:
        created_at = utc_now()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_runs (
                    user_id, username, video_name, command, detector, status,
                    processed_frames, event_count, summary_json, events_json,
                    frame_paths_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.username,
                    video_name,
                    command,
                    detector,
                    status,
                    processed_frames,
                    event_count,
                    json.dumps(summary, ensure_ascii=False),
                    json.dumps(events, ensure_ascii=False),
                    json.dumps(frame_paths, ensure_ascii=False),
                    created_at.isoformat(),
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.commit()

        return AnalysisRun(
            id=run_id,
            user_id=user.id,
            username=user.username,
            video_name=video_name,
            command=command,
            detector=detector,
            status=status,
            processed_frames=processed_frames,
            event_count=event_count,
            summary=summary,
            events=events,
            frame_paths=frame_paths,
            created_at=created_at,
        )

    def list_analysis_runs(self, *, limit: int = 100) -> list[AnalysisRun]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM analysis_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_analysis_run_from_row(row) for row in rows]

    def get_analysis_run(self, run_id: int) -> AnalysisRun:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise AnalysisRunNotFoundError(f"analysis run not found: {run_id}")
        return _analysis_run_from_row(row)

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


def _analysis_run_from_row(row: sqlite3.Row) -> AnalysisRun:
    return AnalysisRun(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        username=row["username"],
        video_name=row["video_name"],
        command=row["command"],
        detector=row["detector"],
        status=row["status"],
        processed_frames=int(row["processed_frames"]),
        event_count=int(row["event_count"]),
        summary=json.loads(row["summary_json"]),
        events=json.loads(row["events_json"]),
        frame_paths=json.loads(row["frame_paths_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
