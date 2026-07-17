from __future__ import annotations

import hashlib
import json
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request, Response

from .genesis_whisper_password_hash import build_pbkdf2_hash, verify_pbkdf2_hash

# --- configuration ---
SESSION_COOKIE_NAME = "g3_whisper_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
API_KEY_PREFIX = "whisper"
SECRETS_VERSION = 2


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _new_api_key_id() -> str:
    return f"key_{secrets.token_urlsafe(9)}"


def _generate_api_key() -> str:
    return f"{API_KEY_PREFIX}_{secrets.token_urlsafe(24)}"


class AuthStore:
    """Admin users (username/password login), browser sessions (cookie) and client API
    keys, all persisted to one JSON file (logs/genesis_whisper_secrets.json)."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.secrets_path = self.logs_dir / "genesis_whisper_secrets.json"
        self._lock = threading.RLock()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    # ---- persistence ----
    def _load(self) -> dict[str, Any]:
        return _json_load(self.secrets_path)

    def _save(self, payload: dict[str, Any]) -> None:
        _json_dump(self.secrets_path, payload)

    def _mutate(self, mutator: Callable[[dict[str, Any]], Any]) -> Any:
        with self._lock:
            payload = self._load()
            self._normalize_shape(payload)
            result = mutator(payload)
            self._save(payload)
            return result

    @staticmethod
    def _normalize_shape(payload: dict[str, Any]) -> None:
        payload.setdefault("version", SECRETS_VERSION)
        payload.setdefault("users", {})
        payload.setdefault("sessions", {})
        payload.setdefault("api_keys", {})

    def _bootstrap(self) -> None:
        with self._lock:
            payload = self._load()
            # Legacy file only had {"admin_key": {...}} — that token is obsolete now.
            if "version" not in payload:
                payload.pop("admin_key", None)
            self._normalize_shape(payload)
            payload["version"] = SECRETS_VERSION
            if not payload["users"]:
                payload["users"][DEFAULT_ADMIN_USERNAME] = self._make_user(
                    DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, must_change=True
                )
            self._prune_sessions_in(payload)
            self._save(payload)

    @staticmethod
    def _make_user(username: str, password: str, *, must_change: bool) -> dict[str, Any]:
        stamp = now_iso()
        return {
            "username": username,
            "password_hash": build_pbkdf2_hash(password),
            "must_change_password": must_change,
            "created_at": stamp,
            "updated_at": stamp,
            "last_login_at": None,
        }

    # ---- users ----
    def verify_user(self, username: str, password: str) -> Optional[dict[str, Any]]:
        username = (username or "").strip().lower()
        if not username or not password:
            return None
        payload = self._load()
        self._normalize_shape(payload)
        user = payload["users"].get(username)
        if not user or not verify_pbkdf2_hash(password, user.get("password_hash", "")):
            return None
        return user

    def touch_login(self, username: str) -> None:
        def apply(payload: dict[str, Any]) -> None:
            user = payload["users"].get(username)
            if user:
                user["last_login_at"] = now_iso()

        self._mutate(apply)

    def set_password(self, username: str, new_password: str) -> bool:
        username = (username or "").strip().lower()
        if len(new_password or "") < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")

        def apply(payload: dict[str, Any]) -> bool:
            user = payload["users"].get(username)
            if not user:
                return False
            user["password_hash"] = build_pbkdf2_hash(new_password)
            user["must_change_password"] = False
            user["updated_at"] = now_iso()
            # A password change invalidates all existing sessions of that user.
            stale = [th for th, s in payload["sessions"].items() if s.get("username") == username]
            for th in stale:
                payload["sessions"].pop(th, None)
            return True

        return self._mutate(apply)

    # ---- sessions ----
    def create_session(self, username: str) -> str:
        raw = secrets.token_urlsafe(32)
        token_hash = hash_token(raw)
        stamp = now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat()

        def apply(payload: dict[str, Any]) -> None:
            payload["sessions"][token_hash] = {
                "username": username,
                "created_at": stamp,
                "expires_at": expires,
                "last_seen_at": stamp,
            }

        self._mutate(apply)
        return raw

    def get_session(self, raw_token: Optional[str]) -> Optional[dict[str, Any]]:
        if not raw_token:
            return None
        token_hash = hash_token(raw_token)
        payload = self._load()
        self._normalize_shape(payload)
        session = payload["sessions"].get(token_hash)
        if not session:
            return None
        expires = _parse_iso(session.get("expires_at"))
        if expires is not None and datetime.now(timezone.utc) > expires:
            self.delete_session(raw_token)
            return None
        user = payload["users"].get(session.get("username"))
        if not user:
            return None
        return {
            "username": user["username"],
            "must_change_password": bool(user.get("must_change_password")),
        }

    def delete_session(self, raw_token: Optional[str]) -> None:
        if not raw_token:
            return
        token_hash = hash_token(raw_token)

        def apply(payload: dict[str, Any]) -> None:
            payload["sessions"].pop(token_hash, None)

        self._mutate(apply)

    @staticmethod
    def _prune_sessions_in(payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            th
            for th, s in payload.get("sessions", {}).items()
            if (_parse_iso(s.get("expires_at")) is not None and now > _parse_iso(s.get("expires_at")))
        ]
        for th in expired:
            payload["sessions"].pop(th, None)

    # ---- client API keys ----
    def list_api_keys(self) -> list[dict[str, Any]]:
        payload = self._load()
        self._normalize_shape(payload)
        out: list[dict[str, Any]] = []
        for rec in payload["api_keys"].values():
            usage = rec.get("usage") or {}
            out.append(
                {
                    "id": rec.get("id"),
                    "alias": rec.get("alias", ""),
                    "created_at": rec.get("created_at"),
                    "usage": {
                        "total_seconds_processed": round(float(usage.get("total_seconds_processed") or 0.0), 3),
                        "request_count": int(usage.get("request_count") or 0),
                        "last_used_at": usage.get("last_used_at"),
                    },
                }
            )
        out.sort(key=lambda r: r.get("created_at") or "")
        return out

    def has_api_keys(self) -> bool:
        payload = self._load()
        self._normalize_shape(payload)
        return len(payload["api_keys"]) > 0

    def create_api_key(self, alias: str) -> dict[str, Any]:
        alias = (alias or "").strip() or "Unnamed key"
        raw = _generate_api_key()
        key_id = _new_api_key_id()
        stamp = now_iso()

        def apply(payload: dict[str, Any]) -> None:
            payload["api_keys"][key_id] = {
                "id": key_id,
                "key_hash": hash_token(raw),
                "alias": alias,
                "created_at": stamp,
                "usage": {"total_seconds_processed": 0.0, "request_count": 0, "last_used_at": None},
            }

        self._mutate(apply)
        return {"id": key_id, "alias": alias, "created_at": stamp, "token": raw}

    def delete_api_key(self, key_id: str) -> bool:
        def apply(payload: dict[str, Any]) -> bool:
            return payload["api_keys"].pop(key_id, None) is not None

        return self._mutate(apply)

    def match_api_key(self, raw_key: Optional[str]) -> Optional[str]:
        if not raw_key:
            return None
        provided = hash_token(raw_key.strip())
        payload = self._load()
        self._normalize_shape(payload)
        for rec in payload["api_keys"].values():
            if secrets.compare_digest(rec.get("key_hash") or "", provided):
                return rec.get("id")
        return None

    def record_api_key_usage(self, key_id: Optional[str], audio_seconds: float) -> None:
        if not key_id:
            return

        def apply(payload: dict[str, Any]) -> None:
            rec = payload["api_keys"].get(key_id)
            if not rec:
                return
            usage = rec.setdefault(
                "usage", {"total_seconds_processed": 0.0, "request_count": 0, "last_used_at": None}
            )
            usage["total_seconds_processed"] = round(
                float(usage.get("total_seconds_processed") or 0.0) + float(audio_seconds or 0.0), 3
            )
            usage["request_count"] = int(usage.get("request_count") or 0) + 1
            usage["last_used_at"] = now_iso()

        self._mutate(apply)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_auth_store = AuthStore(PROJECT_ROOT)


def get_auth_store() -> AuthStore:
    return _auth_store


# ---- cookie helpers ----
def _cookie_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def set_session_cookie(response: Response, request: Request, raw_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


# ---- FastAPI dependencies ----
def require_session(request: Request) -> dict[str, Any]:
    ctx = get_auth_store().get_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not ctx:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return ctx


def require_admin(request: Request) -> dict[str, Any]:
    ctx = require_session(request)
    if ctx.get("must_change_password"):
        raise HTTPException(status_code=403, detail="password_change_required")
    return ctx


def authorize_api_key(request: Request) -> Optional[str]:
    """Public-endpoint gate. No client keys configured -> open (returns None). Otherwise a
    valid X-API-Key header is required; returns the matched key id (for usage accounting)."""
    store = get_auth_store()
    if not store.has_api_keys():
        return None
    key_id = store.match_api_key(request.headers.get("x-api-key"))
    if not key_id:
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
    return key_id
