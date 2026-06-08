from multimodalcv.auth.models import Role
from multimodalcv.auth.store import AuthStore
from multimodalcv.cli.create_admin import main


def test_create_admin_cli_creates_administrator(tmp_path, capsys) -> None:
    database_path = tmp_path / "auth.db"

    exit_code = main(
        [
            "demo-admin",
            "--password",
            "admin-password",
            "--database",
            str(database_path),
        ]
    )

    assert exit_code == 0
    user = AuthStore(database_path).list_users()[0]
    assert user.username == "demo-admin"
    assert user.role == Role.ADMIN
    assert "Created administrator" in capsys.readouterr().out


def test_create_admin_cli_rejects_duplicate_username(tmp_path) -> None:
    database_path = tmp_path / "auth.db"
    arguments = [
        "demo-admin",
        "--password",
        "admin-password",
        "--database",
        str(database_path),
    ]

    assert main(arguments) == 0
    assert main(arguments) == 2
