from fastapi.testclient import TestClient

from multimodalcv.api.app import create_app
from multimodalcv.auth.models import Role


def make_client(tmp_path) -> TestClient:
    app = create_app(tmp_path / "api.db")
    with TestClient(app) as client:
        app.state.auth_store.create_user("operator", "operator-password", Role.OPERATOR)
        yield client


def test_health_endpoint(tmp_path) -> None:
    for client in make_client(tmp_path):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_login_me_and_logout_flow(tmp_path) -> None:
    for client in make_client(tmp_path):
        login_response = client.post(
            "/auth/login",
            json={"username": "operator", "password": "operator-password"},
        )

        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_response = client.get("/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "operator"
        assert me_response.json()["role"] == "operator"

        logout_response = client.post("/auth/logout", headers=headers)
        assert logout_response.status_code == 204

        expired_response = client.get("/auth/me", headers=headers)
        assert expired_response.status_code == 401


def test_login_rejects_invalid_password(tmp_path) -> None:
    for client in make_client(tmp_path):
        response = client.post(
            "/auth/login",
            json={"username": "operator", "password": "wrong-password"},
        )

        assert response.status_code == 401


def test_me_requires_bearer_token(tmp_path) -> None:
    for client in make_client(tmp_path):
        response = client.get("/auth/me")

        assert response.status_code == 401
