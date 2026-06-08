from datetime import timedelta

import pytest

from multimodalcv.auth.models import Role
from multimodalcv.auth.store import (
    AuthStore,
    AuthenticationError,
    DuplicateUsernameError,
)


def make_store(tmp_path) -> AuthStore:
    store = AuthStore(tmp_path / "auth.db")
    store.initialize()
    return store


def test_initialize_creates_database(tmp_path) -> None:
    store = make_store(tmp_path)

    assert store.database_path.exists()


def test_create_and_list_users_with_three_roles(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_user("Admin", "admin-password", Role.ADMIN)
    store.create_user("Operator", "operator-password", Role.OPERATOR)
    store.create_user("Viewer", "viewer-password", Role.VIEWER)

    users = store.list_users()

    assert [(user.username, user.role) for user in users] == [
        ("admin", Role.ADMIN),
        ("operator", Role.OPERATOR),
        ("viewer", Role.VIEWER),
    ]


def test_create_user_rejects_duplicate_username_case_insensitively(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_user("Admin", "admin-password", Role.ADMIN)

    with pytest.raises(DuplicateUsernameError):
        store.create_user("ADMIN", "another-password", Role.ADMIN)


def test_authenticate_accepts_valid_credentials_and_audits_login(tmp_path) -> None:
    store = make_store(tmp_path)
    created_user = store.create_user("operator", "operator-password", Role.OPERATOR)

    authenticated_user = store.authenticate("OPERATOR", "operator-password")

    assert authenticated_user == created_user
    actions = [entry.action for entry in store.list_audit_entries()]
    assert "login_succeeded" in actions
    assert "user_created" in actions


def test_authenticate_rejects_invalid_credentials_and_audits_failure(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_user("viewer", "viewer-password", Role.VIEWER)

    with pytest.raises(AuthenticationError):
        store.authenticate("viewer", "wrong-password")

    latest_entry = store.list_audit_entries(limit=1)[0]
    assert latest_entry.action == "login_failed"
    assert latest_entry.username == "viewer"


def test_session_resolves_user_and_can_be_revoked(tmp_path) -> None:
    store = make_store(tmp_path)
    user = store.create_user("viewer", "viewer-password", Role.VIEWER)
    session = store.create_session(user)

    assert store.get_user_by_session(session.token) == user

    store.revoke_session(session.token)

    assert store.get_user_by_session(session.token) is None


def test_expired_session_is_rejected(tmp_path) -> None:
    store = make_store(tmp_path)
    user = store.create_user("viewer", "viewer-password", Role.VIEWER)
    session = store.create_session(user, lifetime=timedelta(seconds=-1))

    assert store.get_user_by_session(session.token) is None
