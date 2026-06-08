"""HTTP client used by the Streamlit interface."""

from __future__ import annotations

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when the backend rejects a request or cannot be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MultiModalCVApiClient:
    """Small typed wrapper around the local FastAPI backend."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self.timeout = timeout

    def login(self, username: str, password: str) -> str:
        payload = self._request(
            "POST",
            "/auth/login",
            json={"username": username, "password": password},
        )
        return str(payload["access_token"])

    def me(self, token: str) -> dict[str, Any]:
        return self._request("GET", "/auth/me", token=token)

    def logout(self, token: str) -> None:
        self._request("POST", "/auth/logout", token=token, expect_json=False)

    def list_users(self, token: str) -> list[dict[str, Any]]:
        return self._request("GET", "/users", token=token)

    def create_user(
        self,
        token: str,
        *,
        username: str,
        password: str,
        role: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/users",
            token=token,
            json={"username": username, "password": password, "role": role},
        )

    def update_user(
        self,
        token: str,
        user_id: int,
        *,
        role: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if role is not None:
            payload["role"] = role
        if is_active is not None:
            payload["is_active"] = is_active
        return self._request("PATCH", f"/users/{user_id}", token=token, json=payload)

    def reset_password(self, token: str, user_id: int, password: str) -> None:
        self._request(
            "POST",
            f"/users/{user_id}/reset-password",
            token=token,
            json={"password": password},
            expect_json=False,
        )

    def list_audit(self, token: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", f"/audit?limit={limit}", token=token)

    def create_analysis_run(
        self,
        token: str,
        *,
        video_name: str,
        command: str,
        detector: str,
        processed_frames: int,
        event_count: int,
        summary: dict[str, Any],
        events: list[dict[str, Any]],
        frame_paths: list[str],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/analysis-runs",
            token=token,
            json={
                "video_name": video_name,
                "command": command,
                "detector": detector,
                "status": "completed",
                "processed_frames": processed_frames,
                "event_count": event_count,
                "summary": summary,
                "events": events,
                "frame_paths": frame_paths,
            },
        )

    def list_analysis_runs(self, token: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", f"/analysis-runs?limit={limit}", token=token)

    def get_analysis_run(self, token: str, run_id: int) -> dict[str, Any]:
        return self._request("GET", f"/analysis-runs/{run_id}", token=token)

    def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        json: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            with httpx.Client(
                base_url=self.base_url,
                transport=self._transport,
                timeout=self.timeout,
            ) as client:
                response = client.request(method, path, headers=headers, json=json)
        except httpx.HTTPError as error:
            raise ApiClientError(
                "Не удалось подключиться к серверу MultiModalCV. Проверьте, что FastAPI запущен."
            ) from error

        if response.is_error:
            message = _error_message(response)
            raise ApiClientError(message, status_code=response.status_code)

        if not expect_json or response.status_code == 204:
            return None
        return response.json()


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Ошибка API: HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    return f"Ошибка API: HTTP {response.status_code}"
