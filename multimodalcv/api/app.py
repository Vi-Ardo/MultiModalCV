"""FastAPI backend for the MultiModalCV web application."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

from multimodalcv.auth.models import AnalysisRun, AuditEntry, Role, User
from multimodalcv.auth.store import (
    AnalysisRunNotFoundError,
    DEFAULT_DATABASE_PATH,
    AuthStore,
    AuthenticationError,
    DuplicateUsernameError,
    UserNotFoundError,
)


class LoginRequest(BaseModel):
    username: str
    password: str


class SessionResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: Role
    is_active: bool


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Role


class UpdateUserRequest(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class ResetPasswordRequest(BaseModel):
    password: str


class AuditEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None
    username: str | None
    action: str
    details: str | None
    created_at: datetime


class CreateAnalysisRunRequest(BaseModel):
    video_name: str
    command: str
    detector: str
    status: str = "completed"
    processed_frames: int
    event_count: int
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    frame_paths: list[str]


class AnalysisRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


def create_app(database_path: Path | None = None) -> FastAPI:
    store = AuthStore(database_path or database_path_from_environment())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.initialize()
        yield

    app = FastAPI(
        title="MultiModalCV API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.auth_store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/auth/login", response_model=SessionResponse)
    def login(payload: LoginRequest) -> SessionResponse:
        try:
            user = store.authenticate(payload.username, payload.password)
        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            ) from error
        session = store.create_session(user)
        return SessionResponse(access_token=session.token)

    def current_token(authorization: Annotated[str | None, Header()] = None) -> str:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
            )
        return authorization.removeprefix("Bearer ").strip()

    def current_user(token: Annotated[str, Depends(current_token)]) -> User:
        user = store.get_user_by_session(token)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or expired session",
            )
        return user

    def require_role(*roles: Role) -> Callable[[User], User]:
        def role_dependency(user: Annotated[User, Depends(current_user)]) -> User:
            if user.role not in roles:
                store.record_audit(
                    "access_denied",
                    user=user,
                    details=f"required_roles={','.join(role.value for role in roles)}",
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="insufficient permissions",
                )
            return user

        return role_dependency

    require_admin = require_role(Role.ADMIN)
    require_analysis_operator = require_role(Role.ADMIN, Role.OPERATOR)

    @app.get("/auth/me", response_model=UserResponse)
    def me(user: Annotated[User, Depends(current_user)]) -> User:
        return user

    @app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(token: Annotated[str, Depends(current_token)]) -> None:
        store.revoke_session(token)

    @app.get("/users", response_model=list[UserResponse])
    def list_users(_: Annotated[User, Depends(require_admin)]) -> list[User]:
        return store.list_users()

    @app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    def create_user(
        payload: CreateUserRequest,
        admin: Annotated[User, Depends(require_admin)],
    ) -> User:
        try:
            user = store.create_user(payload.username, payload.password, payload.role)
        except DuplicateUsernameError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

        store.record_audit(
            "admin_created_user",
            user=admin,
            details=f"target_user={user.username};role={user.role.value}",
        )
        return user

    @app.patch("/users/{user_id}", response_model=UserResponse)
    def update_user(
        user_id: int,
        payload: UpdateUserRequest,
        admin: Annotated[User, Depends(require_admin)],
    ) -> User:
        if payload.role is None and payload.is_active is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="at least one user field must be provided",
            )
        if user_id == admin.id and (
            payload.role not in {None, Role.ADMIN} or payload.is_active is False
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="administrator cannot remove own access",
            )

        try:
            user = store.update_user(
                user_id,
                role=payload.role,
                is_active=payload.is_active,
            )
        except UserNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

        store.record_audit(
            "admin_updated_user",
            user=admin,
            details=(
                f"target_user={user.username};role={user.role.value};"
                f"is_active={str(user.is_active).lower()}"
            ),
        )
        return user

    @app.post("/users/{user_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
    def reset_password(
        user_id: int,
        payload: ResetPasswordRequest,
        admin: Annotated[User, Depends(require_admin)],
    ) -> None:
        try:
            store.reset_password(user_id, payload.password)
            target_user = store.get_user(user_id)
        except UserNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

        store.record_audit(
            "admin_reset_password",
            user=admin,
            details=f"target_user={target_user.username}",
        )

    @app.get("/audit", response_model=list[AuditEntryResponse])
    def list_audit(
        _: Annotated[User, Depends(require_admin)],
        limit: int = 100,
    ) -> list[AuditEntry]:
        if limit < 1 or limit > 500:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="limit must be between 1 and 500",
            )
        return store.list_audit_entries(limit=limit)

    @app.post(
        "/analysis-runs",
        response_model=AnalysisRunResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_analysis_run(
        payload: CreateAnalysisRunRequest,
        user: Annotated[User, Depends(require_analysis_operator)],
    ) -> AnalysisRun:
        run = store.create_analysis_run(
            user=user,
            video_name=payload.video_name,
            command=payload.command,
            detector=payload.detector,
            status=payload.status,
            processed_frames=payload.processed_frames,
            event_count=payload.event_count,
            summary=payload.summary,
            events=payload.events,
            frame_paths=payload.frame_paths,
        )
        store.record_audit(
            "analysis_created",
            user=user,
            details=f"run_id={run.id};video={run.video_name}",
        )
        return run

    @app.get("/analysis-runs", response_model=list[AnalysisRunResponse])
    def list_analysis_runs(
        user: Annotated[User, Depends(current_user)],
        limit: int = 100,
    ) -> list[AnalysisRun]:
        if limit < 1 or limit > 500:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="limit must be between 1 and 500",
            )
        runs = store.list_analysis_runs(limit=limit)
        store.record_audit(
            "analysis_history_viewed",
            user=user,
            details=f"count={len(runs)}",
        )
        return runs

    @app.get("/analysis-runs/{run_id}", response_model=AnalysisRunResponse)
    def get_analysis_run(
        run_id: int,
        user: Annotated[User, Depends(current_user)],
    ) -> AnalysisRun:
        try:
            run = store.get_analysis_run(run_id)
        except AnalysisRunNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        store.record_audit(
            "analysis_result_viewed",
            user=user,
            details=f"run_id={run.id}",
        )
        return run

    return app


def database_path_from_environment() -> Path:
    return Path(os.environ.get("MULTIMODALCV_DATABASE_PATH", DEFAULT_DATABASE_PATH))


app = create_app()
