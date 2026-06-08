from multimodalcv.auth.security import generate_session_token, hash_password, verify_password


def test_hash_password_uses_random_salt() -> None:
    first_hash = hash_password("secure-password")
    second_hash = hash_password("secure-password")

    assert first_hash != second_hash
    assert verify_password("secure-password", first_hash)
    assert verify_password("secure-password", second_hash)


def test_verify_password_rejects_invalid_password_and_hash() -> None:
    password_hash = hash_password("secure-password")

    assert not verify_password("wrong-password", password_hash)
    assert not verify_password("secure-password", "invalid")


def test_hash_password_rejects_short_password() -> None:
    try:
        hash_password("short")
    except ValueError as error:
        assert "at least 8 characters" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_generate_session_token_is_unique() -> None:
    assert generate_session_token() != generate_session_token()
