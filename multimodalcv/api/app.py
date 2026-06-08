"""FastAPI backend for the MultiModalCV web application."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

from multimodalcv.auth.models import Role, User
from multimodalcv.auth.store import DEFAULT_DATABASE_PATH, AuthStore, AuthenticationError


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

    @app.get("/auth/me", response_model=UserResponse)
    def me(user: Annotated[User, Depends(current_user)]) -> User:
        return user

    @app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(token: Annotated[str, Depends(current_token)]) -> None:
        store.revoke_session(token)

    return app


def database_path_from_environment() -> Path:
    return Path(os.environ.get("MULTIMODALCV_DATABASE_PATH", DEFAULT_DATABASE_PATH))


app = create_app()
