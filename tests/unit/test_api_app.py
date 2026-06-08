from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from multimodalcv.api.app import create_app
from multimodalcv.auth.models import Role


@contextmanager
def make_client(tmp_path) -> Iterator[TestClient]:
    app = create_app(tmp_path / "api.db")
    with TestClient(app) as client:
        store = app.state.auth_store
        store.create_user("admin", "admin-password", Role.ADMIN)
        store.create_user("operator", "operator-password", Role.OPERATOR)
        store.create_user("viewer", "viewer-password", Role.VIEWER)
        yield client


def login_headers(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_health_endpoint(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_login_me_and_logout_flow(tmp_path) -> None:
    with make_client(tmp_path) as client:
        headers = login_headers(client, "operator", "operator-password")

        me_response = client.get("/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "operator"
        assert me_response.json()["role"] == "operator"

        logout_response = client.post("/auth/logout", headers=headers)
        assert logout_response.status_code == 204

        expired_response = client.get("/auth/me", headers=headers)
        assert expired_response.status_code == 401


def test_login_rejects_invalid_password(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/auth/login",
            json={"username": "operator", "password": "wrong-password"},
        )

        assert response.status_code == 401


def test_me_requires_bearer_token(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/auth/me")

        assert response.status_code == 401


def test_admin_can_list_and_create_users(tmp_path) -> None:
    with make_client(tmp_path) as client:
        headers = login_headers(client, "admin", "admin-password")

        list_response = client.get("/users", headers=headers)
        assert list_response.status_code == 200
        assert {user["role"] for user in list_response.json()} == {
            "admin",
            "operator",
            "viewer",
        }

        create_response = client.post(
            "/users",
            headers=headers,
            json={
                "username": "second-operator",
                "password": "second-password",
                "role": "operator",
            },
        )
        assert create_response.status_code == 201
        assert create_response.json()["role"] == "operator"


def test_operator_and_viewer_cannot_manage_users(tmp_path) -> None:
    with make_client(tmp_path) as client:
        for username, password in (
            ("operator", "operator-password"),
            ("viewer", "viewer-password"),
        ):
            headers = login_headers(client, username, password)

            assert client.get("/users", headers=headers).status_code == 403
            assert (
                client.post(
                    "/users",
                    headers=headers,
                    json={
                        "username": f"{username}-created",
                        "password": "created-password",
                        "role": "viewer",
                    },
                ).status_code
                == 403
            )


def test_admin_can_change_role_and_block_user(tmp_path) -> None:
    with make_client(tmp_path) as client:
        admin_headers = login_headers(client, "admin", "admin-password")
        operator_headers = login_headers(client, "operator", "operator-password")
        users = client.get("/users", headers=admin_headers).json()
        operator = next(user for user in users if user["username"] == "operator")

        response = client.patch(
            f"/users/{operator['id']}",
            headers=admin_headers,
            json={"role": "viewer", "is_active": False},
        )

        assert response.status_code == 200
        assert response.json()["role"] == "viewer"
        assert response.json()["is_active"] is False
        assert client.get("/auth/me", headers=operator_headers).status_code == 401


def test_admin_cannot_remove_own_access(tmp_path) -> None:
    with make_client(tmp_path) as client:
        headers = login_headers(client, "admin", "admin-password")
        admin = client.get("/auth/me", headers=headers).json()

        demote_response = client.patch(
            f"/users/{admin['id']}",
            headers=headers,
            json={"role": "viewer"},
        )
        block_response = client.patch(
            f"/users/{admin['id']}",
            headers=headers,
            json={"is_active": False},
        )

        assert demote_response.status_code == 400
        assert block_response.status_code == 400


def test_admin_can_reset_password_and_old_session(tmp_path) -> None:
    with make_client(tmp_path) as client:
        admin_headers = login_headers(client, "admin", "admin-password")
        operator_headers = login_headers(client, "operator", "operator-password")
        users = client.get("/users", headers=admin_headers).json()
        operator = next(user for user in users if user["username"] == "operator")

        response = client.post(
            f"/users/{operator['id']}/reset-password",
            headers=admin_headers,
            json={"password": "updated-password"},
        )

        assert response.status_code == 204
        assert client.get("/auth/me", headers=operator_headers).status_code == 401
        assert (
            client.post(
                "/auth/login",
                json={"username": "operator", "password": "operator-password"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/auth/login",
                json={"username": "operator", "password": "updated-password"},
            ).status_code
            == 200
        )


def test_only_admin_can_view_audit_log(tmp_path) -> None:
    with make_client(tmp_path) as client:
        admin_headers = login_headers(client, "admin", "admin-password")
        operator_headers = login_headers(client, "operator", "operator-password")

        admin_response = client.get("/audit?limit=10", headers=admin_headers)
        operator_response = client.get("/audit", headers=operator_headers)

        assert admin_response.status_code == 200
        assert admin_response.json()
        assert operator_response.status_code == 403


def test_admin_and_operator_can_create_analysis_runs(tmp_path) -> None:
    with make_client(tmp_path) as client:
        for username, password in (
            ("admin", "admin-password"),
            ("operator", "operator-password"),
        ):
            headers = login_headers(client, username, password)
            response = client.post(
                "/analysis-runs",
                headers=headers,
                json=analysis_run_payload(video_name=f"{username}.mp4"),
            )

            assert response.status_code == 201
            assert response.json()["username"] == username


def test_viewer_cannot_create_but_can_read_analysis_runs(tmp_path) -> None:
    with make_client(tmp_path) as client:
        operator_headers = login_headers(client, "operator", "operator-password")
        viewer_headers = login_headers(client, "viewer", "viewer-password")
        created = client.post(
            "/analysis-runs",
            headers=operator_headers,
            json=analysis_run_payload(),
        ).json()

        forbidden = client.post(
            "/analysis-runs",
            headers=viewer_headers,
            json=analysis_run_payload(video_name="forbidden.mp4"),
        )
        history = client.get("/analysis-runs", headers=viewer_headers)
        detail = client.get(f"/analysis-runs/{created['id']}", headers=viewer_headers)

        assert forbidden.status_code == 403
        assert history.status_code == 200
        assert history.json()[0]["video_name"] == "sample.mp4"
        assert detail.status_code == 200
        assert detail.json()["events"][0]["metadata"]["count"] == 2


def test_get_analysis_run_returns_not_found(tmp_path) -> None:
    with make_client(tmp_path) as client:
        headers = login_headers(client, "viewer", "viewer-password")

        assert client.get("/analysis-runs/999", headers=headers).status_code == 404


def analysis_run_payload(*, video_name: str = "sample.mp4") -> dict:
    return {
        "video_name": video_name,
        "command": "Посчитай людей",
        "detector": "yolo",
        "status": "completed",
        "processed_frames": 60,
        "event_count": 1,
        "summary": {"processed_frames": 60, "event_count": 1},
        "events": [
            {
                "event_type": "count_in_frame",
                "frame_index": 12,
                "timestamp_sec": 0.4,
                "metadata": {"count": 2},
            }
        ],
        "frame_paths": ["C:/frames/annotated_000012.jpg"],
    }
