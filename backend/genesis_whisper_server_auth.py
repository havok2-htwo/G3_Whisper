from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Header, HTTPException

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _clamp_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


class WhisperAdminKeyStore:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.secrets_path = self.logs_dir / "genesis_whisper_secrets.json"
        self._lock = threading.RLock()
        self._startup_admin_key = str(os.getenv("GENESIS_STARTUP_ADMIN_KEY") or "").strip()
        self._startup_admin_key_expires_at: datetime | None = None
        if self._startup_admin_key:
            ttl_seconds = _clamp_int(
                os.getenv("GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS"),
                minimum=1,
                maximum=3600,
                default=300,
            )
            self._startup_admin_key_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._seed_secrets()

    def _seed_secrets(self) -> None:
        with self._lock:
            payload = _json_load(self.secrets_path)
            if not payload.get("admin_key"):
                bootstrap_token = str(os.getenv("GENESIS_ADMIN_KEY") or "").strip() or self._generate_raw_key()
                payload["admin_key"] = {
                    "id": "admin",
                    "label": "Master Admin Key",
                    "hash": hash_token(bootstrap_token),
                    "created_at": now_iso(),
                    "last_used_at": None,
                }
                print(
                    "GENESIS admin key initialized. Save this key now because it is only shown once:",
                    bootstrap_token,
                    flush=True,
                )
            _json_dump(self.secrets_path, payload)

    def _load_secrets(self) -> dict[str, Any]:
        with self._lock:
            return _json_load(self.secrets_path)

    def _save_secrets(self, payload: dict[str, Any]) -> None:
        with self._lock:
            _json_dump(self.secrets_path, payload)

    def _mutate_secrets(self, mutator: Any) -> Any:
        with self._lock:
            payload = self._load_secrets()
            result = mutator(payload)
            self._save_secrets(payload)
            return result

    def _generate_raw_key(self) -> str:
        return f"genesis_admin_{secrets.token_urlsafe(24)}"

    def list_keys(self) -> dict[str, Any]:
        payload = self._load_secrets()
        admin_key = payload.get("admin_key") or {}
        return {
            "admin_key": {
                "id": admin_key.get("id", "admin"),
                "label": admin_key.get("label") or "Master Admin Key",
                "created_at": admin_key.get("created_at"),
                "last_used_at": admin_key.get("last_used_at"),
            }
        }

    def rotate_admin_key(self, *, label: str = "Master Admin Key") -> dict[str, Any]:
        plaintext = self._generate_raw_key()
        clean_label = label.strip() or "Master Admin Key"

        def apply_rotate(payload: dict[str, Any]) -> dict[str, Any]:
            payload["admin_key"] = {
                "id": "admin",
                "label": clean_label,
                "hash": hash_token(plaintext),
                "created_at": now_iso(),
                "last_used_at": None,
            }
            return {
                "id": "admin",
                "label": clean_label,
                "token": plaintext,
                "created_at": payload["admin_key"]["created_at"],
            }

        return self._mutate_secrets(apply_rotate)

    def verify_admin_key(self, raw_key: str) -> bool:
        if not raw_key:
            return False

        def apply_verify(payload: dict[str, Any]) -> bool:
            record = payload.get("admin_key") or {}
            is_valid = secrets.compare_digest(record.get("hash") or "", hash_token(raw_key))
            if not is_valid and self._startup_admin_key:
                not_expired = (
                    self._startup_admin_key_expires_at is None
                    or datetime.now(timezone.utc) <= self._startup_admin_key_expires_at
                )
                is_valid = not_expired and secrets.compare_digest(self._startup_admin_key, raw_key)
            if is_valid and record:
                record["last_used_at"] = now_iso()
                payload["admin_key"] = record
            return is_valid

        return self._mutate_secrets(apply_verify)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_admin_key_store = WhisperAdminKeyStore(PROJECT_ROOT)


def get_admin_key_store() -> WhisperAdminKeyStore:
    return _admin_key_store


def require_admin(x_admin_key: Optional[str] = Header(None)) -> dict[str, str]:
    if not get_admin_key_store().verify_admin_key((x_admin_key or "").strip()):
        raise HTTPException(status_code=401, detail="A valid X-Admin-Key header is required.")
    return {"role": "admin"}
