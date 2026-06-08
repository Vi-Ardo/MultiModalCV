from datetime import timedelta

import pytest

from multimodalcv.auth.models import Role
from multimodalcv.auth.store import (
    AnalysisRunNotFoundError,
    AuthStore,
    AuthenticationError,
    DuplicateUsernameError,
    UserNotFoundError,
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


def test_get_user_rejects_unknown_id(tmp_path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(UserNotFoundError):
        store.get_user(999)


def test_update_user_changes_role_and_active_state(tmp_path) -> None:
    store = make_store(tmp_path)
    user = store.create_user("viewer", "viewer-password", Role.VIEWER)

    updated_user = store.update_user(user.id, role=Role.OPERATOR, is_active=False)

    assert updated_user.role == Role.OPERATOR
    assert not updated_user.is_active
    assert store.get_user(user.id) == updated_user


def test_deactivating_user_revokes_sessions(tmp_path) -> None:
    store = make_store(tmp_path)
    user = store.create_user("viewer", "viewer-password", Role.VIEWER)
    session = store.create_session(user)

    store.update_user(user.id, is_active=False)

    assert store.get_user_by_session(session.token) is None
    with pytest.raises(AuthenticationError):
        store.authenticate("viewer", "viewer-password")


def test_reset_password_revokes_sessions_and_changes_credentials(tmp_path) -> None:
    store = make_store(tmp_path)
    user = store.create_user("operator", "old-password", Role.OPERATOR)
    session = store.create_session(user)

    store.reset_password(user.id, "new-password")

    assert store.get_user_by_session(session.token) is None
    with pytest.raises(AuthenticationError):
        store.authenticate("operator", "old-password")
    assert store.authenticate("operator", "new-password") == user


def test_create_list_and_get_analysis_run(tmp_path) -> None:
    store = make_store(tmp_path)
    operator = store.create_user("operator", "operator-password", Role.OPERATOR)

    created_run = store.create_analysis_run(
        user=operator,
        video_name="sample.mp4",
        command="Посчитай людей",
        detector="yolo",
        status="completed",
        processed_frames=60,
        event_count=3,
        summary={"event_count": 3},
        events=[{"event_type": "count_in_frame", "metadata": {"count": 2}}],
        frame_paths=["C:/frames/annotated_000010.jpg"],
    )

    assert store.list_analysis_runs() == [created_run]
    assert store.get_analysis_run(created_run.id) == created_run


def test_get_analysis_run_rejects_unknown_id(tmp_path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(AnalysisRunNotFoundError):
        store.get_analysis_run(999)
