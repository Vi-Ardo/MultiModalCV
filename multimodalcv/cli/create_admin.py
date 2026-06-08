"""Create the first local administrator account."""

from __future__ import annotations

import argparse
from getpass import getpass
from pathlib import Path

from multimodalcv.auth.models import Role
from multimodalcv.auth.store import DEFAULT_DATABASE_PATH, AuthStore, DuplicateUsernameError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a MultiModalCV administrator.")
    parser.add_argument("username")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--password")
    args = parser.parse_args(argv)

    password = args.password or getpass("Password: ")
    store = AuthStore(args.database)
    store.initialize()
    try:
        user = store.create_user(args.username, password, Role.ADMIN)
    except (DuplicateUsernameError, ValueError) as error:
        print(error)
        return 2

    print(f"Created administrator: {user.username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
