"""Password hashing and session token helpers."""

from __future__ import annotations

import hashlib
import hmac
import secrets


_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Hash a password using scrypt and a random salt."""
    validate_password(password)
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _derive_password_hash(password, salt)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a password against an encoded scrypt hash."""
    try:
        algorithm, n, r, p, salt_hex, digest_hex = encoded_hash.split("$")
        if algorithm != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected_digest = bytes.fromhex(digest_hex)
        actual_digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
        )
    except (TypeError, ValueError):
        return False

    return hmac.compare_digest(actual_digest, expected_digest)


def generate_session_token() -> str:
    """Generate an opaque session token."""
    return secrets.token_urlsafe(32)


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("password must contain at least 8 characters")


def _derive_password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
