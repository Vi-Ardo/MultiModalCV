import httpx
import pytest

from interfaces.streamlit_app.api_client import ApiClientError, MultiModalCVApiClient


def make_client(handler) -> MultiModalCVApiClient:
    return MultiModalCVApiClient(
        "http://testserver",
        transport=httpx.MockTransport(handler),
    )


def test_login_returns_access_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/auth/login"
        assert request.method == "POST"
        return httpx.Response(200, json={"access_token": "token-123", "token_type": "bearer"})

    assert make_client(handler).login("admin", "password") == "token-123"


def test_me_sends_bearer_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer token-123"
        return httpx.Response(
            200,
            json={"id": 1, "username": "admin", "role": "admin", "is_active": True},
        )

    user = make_client(handler).me("token-123")

    assert user["role"] == "admin"


def test_create_and_update_user_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200 if request.method == "PATCH" else 201,
            json={"id": 2, "username": "viewer", "role": "viewer", "is_active": True},
        )

    client = make_client(handler)
    client.create_user("token", username="viewer", password="viewer-password", role="viewer")
    client.update_user("token", 2, role="operator", is_active=False)

    assert [request.url.path for request in requests] == ["/users", "/users/2"]
    assert requests[0].headers["Authorization"] == "Bearer token"


def test_client_raises_api_error_with_backend_detail() -> None:
    client = make_client(
        lambda _: httpx.Response(403, json={"detail": "insufficient permissions"})
    )

    with pytest.raises(ApiClientError, match="insufficient permissions") as error:
        client.list_users("token")

    assert error.value.status_code == 403


def test_client_reports_connection_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with pytest.raises(ApiClientError, match="Не удалось подключиться"):
        make_client(handler).login("admin", "password")


def test_analysis_run_client_methods() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(201, json={"id": 7})
        if request.url.path.endswith("/7"):
            return httpx.Response(200, json={"id": 7})
        return httpx.Response(200, json=[{"id": 7}])

    client = make_client(handler)
    created = client.create_analysis_run(
        "token",
        video_name="sample.mp4",
        command="Посчитай людей",
        detector="yolo",
        processed_frames=10,
        event_count=1,
        summary={"event_count": 1},
        events=[],
        frame_paths=[],
    )
    history = client.list_analysis_runs("token")
    detail = client.get_analysis_run("token", 7)

    assert created["id"] == 7
    assert history == [{"id": 7}]
    assert detail == {"id": 7}
    assert [request.url.path for request in requests] == [
        "/analysis-runs",
        "/analysis-runs",
        "/analysis-runs/7",
    ]
