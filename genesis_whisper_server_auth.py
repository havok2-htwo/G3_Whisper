import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from fastapi import HTTPException, Request, Response

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ADMIN_SESSION_COOKIE = "genesis_admin_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 12


def _get_env(name: str) -> str:
    return str(os.getenv(name, "")).strip()


def admin_is_configured() -> bool:
    return bool(_get_env("GENESIS_ADMIN_USERNAME") and _get_env("GENESIS_ADMIN_PASSWORD_HASH") and _get_env("GENESIS_SESSION_SECRET"))


def _get_admin_username() -> str:
    return _get_env("GENESIS_ADMIN_USERNAME")


def _get_admin_password_hash() -> str:
    return _get_env("GENESIS_ADMIN_PASSWORD_HASH")


def _get_session_secret() -> str:
    return _get_env("GENESIS_SESSION_SECRET")


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False

    if stored_hash.startswith("pbkdf2_sha256$"):
        _, iterations_str, salt, expected = stored_hash.split("$", 3)
        iterations = int(iterations_str)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
        return hmac.compare_digest(actual, expected)

    if stored_hash.startswith("sha256$"):
        expected = stored_hash.split("$", 1)[1]
        actual = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual, expected)

    if stored_hash.startswith("plain$"):
        expected = stored_hash.split("$", 1)[1]
        return hmac.compare_digest(password, expected)

    return hmac.compare_digest(hashlib.sha256(password.encode("utf-8")).hexdigest(), stored_hash)


def _encode_payload(payload: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")


def _decode_payload(payload_b64: str) -> dict:
    padding = "=" * (-len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode((payload_b64 + padding).encode("ascii")).decode("utf-8"))


def _sign(payload_b64: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()


def create_session_token(username: str) -> str:
    secret = _get_session_secret()
    expires_at = int(time.time()) + SESSION_MAX_AGE_SECONDS
    payload_b64 = _encode_payload({"sub": username, "exp": expires_at})
    signature = _sign(payload_b64, secret)
    return f"{payload_b64}.{signature}"


def get_authenticated_username_from_token(token: Optional[str]) -> Optional[str]:
    if not token or "." not in token or not admin_is_configured():
        return None

    payload_b64, signature = token.rsplit(".", 1)
    expected_signature = _sign(payload_b64, _get_session_secret())
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        payload = _decode_payload(payload_b64)
    except Exception:
        return None

    if int(payload.get("exp", 0)) < int(time.time()):
        return None

    username = str(payload.get("sub", "")).strip()
    if username != _get_admin_username():
        return None
    return username


def set_admin_session(response: Response, username: str):
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=create_session_token(username),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )


def clear_admin_session(response: Response):
    response.delete_cookie(key=ADMIN_SESSION_COOKIE, path="/")


def authenticate_admin_credentials(username: str, password: str) -> bool:
    if not admin_is_configured():
        return False
    return username == _get_admin_username() and verify_password(password, _get_admin_password_hash())


def require_admin(request: Request) -> str:
    if not admin_is_configured():
        raise HTTPException(status_code=503, detail="Admin-Login ist nicht konfiguriert. Bitte GENESIS_ADMIN_USERNAME, GENESIS_ADMIN_PASSWORD_HASH und GENESIS_SESSION_SECRET setzen.")

    username = get_authenticated_username_from_token(request.cookies.get(ADMIN_SESSION_COOKIE))
    if not username:
        raise HTTPException(status_code=401, detail="Nicht eingeloggt.")
    return username
