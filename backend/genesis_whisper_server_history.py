"""Thread-safe in-memory history and short-lived debug-audio retention.

The module deliberately keeps filesystem paths and blob identifiers out of history
entries.  Routes may acquire a :class:`HistoryAudioLease` for the short period in
which they need to stream or decode a retained upload.  A lease pins the file so a
concurrent purge can revoke new access immediately without breaking an operation
that is already in progress.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .genesis_whisper_server_globals import LOGS_DIR, history_lock, transcription_history


VISIBLE_HISTORY_LIMIT = 25
COPY_CHUNK_BYTES = 1024 * 1024
MAX_DEBUG_AUDIO_FILE_BYTES = 256 * 1024 * 1024
MAX_DEBUG_AUDIO_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

_SAFE_CONTENT_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+\-]+/[A-Za-z0-9!#$&^_.+\-]+$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_filename(value: str | None) -> str:
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL_CHARACTERS.sub("", name).strip().strip(".")
    if not name:
        return "audio.bin"
    # Keep enough room for Content-Disposition and common filesystem limits.
    encoded = name.encode("utf-8", errors="ignore")
    if len(encoded) <= 240:
        return name
    encoded = encoded[:240]
    while encoded:
        try:
            shortened = encoded.decode("utf-8")
            return shortened or "audio.bin"
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return "audio.bin"


def _safe_content_type(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) <= 127 and _SAFE_CONTENT_TYPE.fullmatch(normalized):
        return normalized
    return "application/octet-stream"


@dataclass
class _Blob:
    blob_id: str
    path: Path
    filename: str
    content_type: str
    size_bytes: int
    capture_duration_ms: int
    generation: int
    references: int = 0
    pins: int = 0
    pending_delete: bool = False


@dataclass
class _PendingCapture:
    token: str
    history_id: str
    expected_size: int
    generation: int
    final_path: Path
    part_path: Path


class HistoryAudioLease:
    """Pinned access to one retained upload.

    ``path`` is intentionally available only through this backend-only object; it
    is never placed in a history entry or serialized by ``get_history_snapshot``.
    Callers must release the lease after a download or retry finishes.
    """

    def __init__(
        self,
        manager: "HistoryDebugAudioStore",
        token: str,
        blob_id: str,
        path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> None:
        self._manager = manager
        self._token = token
        self.blob_id = blob_id
        self.path = path
        self.filename = filename
        self.content_type = content_type
        self.size_bytes = size_bytes
        self._released = False
        self._release_lock = threading.Lock()

    def open(self) -> BinaryIO:
        if self._released:
            raise RuntimeError("History audio lease has already been released.")
        return self.path.open("rb")

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._manager._release_lease(self._token)

    def __enter__(self) -> "HistoryAudioLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class HistoryDebugAudioStore:
    """Owns history IDs and optional, ref-counted debug-audio blobs."""

    def __init__(
        self,
        history,
        history_mutex: threading.Lock,
        directory: str | os.PathLike[str],
        *,
        visible_limit: int = VISIBLE_HISTORY_LIMIT,
        max_file_bytes: int = MAX_DEBUG_AUDIO_FILE_BYTES,
        max_total_bytes: int = MAX_DEBUG_AUDIO_TOTAL_BYTES,
    ) -> None:
        self._history = history
        self._history_lock = history_mutex
        self._directory = Path(directory)
        self._visible_limit = max(1, int(visible_limit))
        self._max_file_bytes = max(0, int(max_file_bytes))
        self._max_total_bytes = max(0, int(max_total_bytes))
        self._lock = threading.RLock()
        self._enabled = False
        self._generation = 0
        self._records: dict[str, dict[str, Any]] = {}
        self._entry_blobs: dict[str, str] = {}
        self._blobs: dict[str, _Blob] = {}
        self._leases: dict[str, str] = {}
        self._pending_captures: dict[str, _PendingCapture] = {}
        self._reserved_bytes = 0
        self._stored_bytes = 0
        self._active_retries: set[str] = set()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        with self._history_lock:
            with self._lock:
                if enabled:
                    self._enabled = True
                    for history_id in self._visible_ids_locked():
                        record = self._records.setdefault(history_id, {})
                        if record.get("status") == "disabled":
                            record.clear()
                            record.update(status="not_captured")
                    return
                self._enabled = False
                self._purge_locked(reason="disabled")

    def append(self, entry: MutableMapping[str, Any] | Mapping[str, Any], *, existing_blob_id: str | None = None) -> str:
        stored_entry = entry if isinstance(entry, MutableMapping) else dict(entry)
        # Debug-audio storage details live exclusively in this store.  Scrubbing
        # known legacy/internal keys here also keeps callers from accidentally
        # forwarding a local path to the JSONL logger after append returns.
        for unsafe_key in ("debug_audio", "debug_audio_path", "audio_path", "blob_id"):
            stored_entry.pop(unsafe_key, None)
        with self._history_lock:
            with self._lock:
                history_id = str(stored_entry.get("history_id") or "").strip()
                existing_ids = {
                    str(item.get("history_id"))
                    for item in self._history
                    if isinstance(item, Mapping) and item.get("history_id")
                }
                if history_id and history_id in existing_ids:
                    raise ValueError(f"Duplicate history_id: {history_id}")
                if not history_id:
                    history_id = self._new_history_id_locked(existing_ids)
                    stored_entry["history_id"] = history_id

                self._history.appendleft(stored_entry)
                self._records[history_id] = {
                    "status": "not_captured" if self._enabled else "disabled"
                }
                if existing_blob_id is not None:
                    self._attach_blob_locked(history_id, existing_blob_id)
                self._sync_visible_locked()
                return history_id

    def snapshot(self, limit: int = VISIBLE_HISTORY_LIMIT) -> list[dict[str, Any]]:
        requested_limit = max(0, min(int(limit), self._visible_limit))
        with self._history_lock:
            with self._lock:
                self._ensure_visible_ids_locked()
                self._sync_visible_locked()
                entries = list(self._history)[:requested_limit]
                return [self._sanitize_entry_locked(entry) for entry in entries]

    def get_entry(self, history_id: str) -> dict[str, Any] | None:
        normalized = str(history_id or "").strip()
        with self._history_lock:
            with self._lock:
                self._ensure_visible_ids_locked()
                self._sync_visible_locked()
                for entry in list(self._history)[: self._visible_limit]:
                    if isinstance(entry, Mapping) and entry.get("history_id") == normalized:
                        return self._sanitize_entry_locked(entry)
        return None

    def capture(
        self,
        history_id: str,
        source: BinaryIO | bytes | bytearray | memoryview | str | os.PathLike[str],
        filename: str | None,
        content_type: str | None,
    ) -> dict[str, Any]:
        normalized_id = str(history_id or "").strip()
        safe_name = _safe_filename(filename)
        safe_type = _safe_content_type(content_type)
        # Avoid even touching the upload stream when retention is disabled or
        # the row has already expired.  Re-check after measuring because purge
        # or retention pruning may race with that operation.
        with self._lock:
            record = self._records.get(normalized_id)
            if not self._enabled or record is None:
                return self._public_metadata_locked(normalized_id)
            if record.get("status") == "capturing" or normalized_id in self._entry_blobs:
                return self._public_metadata_locked(normalized_id)
        try:
            source_size = self._measure_source(source)
        except Exception:
            return self._set_capture_failure(normalized_id, "capture_failed")

        with self._lock:
            self._collect_garbage_locked()
            record = self._records.get(normalized_id)
            if not self._enabled or record is None:
                return self._public_metadata_locked(normalized_id)
            if record.get("status") == "capturing" or normalized_id in self._entry_blobs:
                return self._public_metadata_locked(normalized_id)
            if source_size > self._max_file_bytes:
                return self._set_failure_locked(normalized_id, "file_too_large")
            if self._stored_bytes + self._reserved_bytes + source_size > self._max_total_bytes:
                return self._set_failure_locked(normalized_id, "storage_quota")

            token = uuid.uuid4().hex
            final_path = self._directory / f"{uuid.uuid4().hex}.bin"
            pending = _PendingCapture(
                token=token,
                history_id=normalized_id,
                expected_size=source_size,
                generation=self._generation,
                final_path=final_path,
                part_path=final_path.with_suffix(".bin.part"),
            )
            self._pending_captures[token] = pending
            self._reserved_bytes += source_size
            record.clear()
            record.update(
                status="capturing",
                token=token,
                filename=safe_name,
                content_type=safe_type,
                size_bytes=source_size,
            )

        started = time.monotonic()
        try:
            self._prepare_directory()
            copied = self._copy_source(source, pending.part_path)
            if copied != source_size:
                raise OSError("Upload changed while debug audio was being copied.")
            with contextlib.suppress(OSError):
                os.chmod(pending.part_path, 0o600)
            os.replace(pending.part_path, pending.final_path)
            with contextlib.suppress(OSError):
                os.chmod(pending.final_path, 0o600)
        except Exception:
            self._unlink_best_effort(pending.part_path)
            self._unlink_best_effort(pending.final_path)
            return self._finish_capture_failure(token)

        duration_ms = round((time.monotonic() - started) * 1000)
        with self._lock:
            active = self._pending_captures.pop(token, None)
            if active is not None:
                self._reserved_bytes = max(0, self._reserved_bytes - active.expected_size)
            record = self._records.get(normalized_id)
            valid = (
                active is not None
                and self._enabled
                and active.generation == self._generation
                and record is not None
                and record.get("token") == token
            )
            if not valid:
                self._unlink_best_effort(pending.final_path)
                return self._public_metadata_locked(normalized_id)

            blob_id = uuid.uuid4().hex
            blob = _Blob(
                blob_id=blob_id,
                path=pending.final_path,
                filename=safe_name,
                content_type=safe_type,
                size_bytes=source_size,
                capture_duration_ms=duration_ms,
                generation=self._generation,
                references=1,
            )
            self._blobs[blob_id] = blob
            self._stored_bytes += source_size
            self._entry_blobs[normalized_id] = blob_id
            record.clear()
            record.update(status="available", blob_id=blob_id)
            return self._public_metadata_locked(normalized_id)

    def acquire(self, history_id: str) -> HistoryAudioLease | None:
        normalized_id = str(history_id or "").strip()
        with self._lock:
            if not self._enabled:
                return None
            blob_id = self._entry_blobs.get(normalized_id)
            blob = self._blobs.get(blob_id or "")
            if blob is None or blob.generation != self._generation or not blob.path.is_file():
                if blob_id:
                    self._invalidate_missing_blob_locked(blob_id)
                return None
            token = uuid.uuid4().hex
            blob.pins += 1
            self._leases[token] = blob.blob_id
            return HistoryAudioLease(
                self,
                token,
                blob.blob_id,
                blob.path,
                blob.filename,
                blob.content_type,
                blob.size_bytes,
            )

    def attach(self, history_id: str, blob_id: str) -> bool:
        normalized_id = str(history_id or "").strip()
        with self._lock:
            return self._attach_blob_locked(normalized_id, str(blob_id or ""))

    def begin_retry(self, history_id: str) -> bool:
        normalized_id = str(history_id or "").strip()
        with self._lock:
            if normalized_id in self._active_retries:
                return False
            self._active_retries.add(normalized_id)
            return True

    def end_retry(self, history_id: str) -> None:
        with self._lock:
            self._active_retries.discard(str(history_id or "").strip())

    def purge(self) -> None:
        with self._history_lock:
            with self._lock:
                self._purge_locked(reason="not_captured")

    def reset(self, *, enabled: bool = False) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        with self._history_lock:
            with self._lock:
                self._enabled = False
                self._purge_locked(reason="disabled")
                self._records.clear()
                self._entry_blobs.clear()
                self._active_retries.clear()
                self._pending_captures.clear()
                self._reserved_bytes = 0
                self._stored_bytes = sum(blob.size_bytes for blob in self._blobs.values())
                self._enabled = enabled
                self._ensure_visible_ids_locked()
                for history_id in self._visible_ids_locked():
                    self._records[history_id] = {
                        "status": "not_captured" if enabled else "disabled"
                    }
        self._remove_orphan_files()

    def shutdown(self) -> None:
        self.set_enabled(False)
        self._remove_orphan_files()

    def _new_history_id_locked(self, existing_ids: set[str]) -> str:
        while True:
            candidate = uuid.uuid4().hex
            if candidate not in existing_ids and candidate not in self._records:
                return candidate

    def _visible_ids_locked(self) -> list[str]:
        return [
            str(entry["history_id"])
            for entry in list(self._history)[: self._visible_limit]
            if isinstance(entry, Mapping) and entry.get("history_id")
        ]

    def _ensure_visible_ids_locked(self) -> None:
        existing_ids = {
            str(item.get("history_id"))
            for item in self._history
            if isinstance(item, Mapping) and item.get("history_id")
        }
        for entry in list(self._history)[: self._visible_limit]:
            if not isinstance(entry, MutableMapping):
                continue
            history_id = str(entry.get("history_id") or "").strip()
            if not history_id:
                history_id = self._new_history_id_locked(existing_ids)
                entry["history_id"] = history_id
                existing_ids.add(history_id)
            self._records.setdefault(
                history_id,
                {"status": "not_captured" if self._enabled else "disabled"},
            )

    def _sync_visible_locked(self) -> None:
        visible = set(self._visible_ids_locked())
        for history_id in list(self._records):
            if history_id not in visible:
                self._remove_entry_record_locked(history_id)

    def _remove_entry_record_locked(self, history_id: str) -> None:
        self._records.pop(history_id, None)
        blob_id = self._entry_blobs.pop(history_id, None)
        if blob_id is None:
            return
        blob = self._blobs.get(blob_id)
        if blob is None:
            return
        blob.references = max(0, blob.references - 1)
        if blob.references == 0:
            blob.pending_delete = True
            self._try_delete_blob_locked(blob)

    def _attach_blob_locked(self, history_id: str, blob_id: str) -> bool:
        if not self._enabled or history_id not in self._records:
            return False
        blob = self._blobs.get(blob_id)
        if (
            blob is None
            or blob.generation != self._generation
            or (blob.references == 0 and blob.pins == 0)
            or not blob.path.is_file()
        ):
            return False
        current_blob_id = self._entry_blobs.get(history_id)
        if current_blob_id == blob_id:
            return True
        if current_blob_id is not None:
            self._remove_blob_reference_locked(history_id, current_blob_id)
        blob.references += 1
        blob.pending_delete = False
        self._entry_blobs[history_id] = blob_id
        record = self._records[history_id]
        record.clear()
        record.update(status="available", blob_id=blob_id)
        return True

    def _remove_blob_reference_locked(self, history_id: str, blob_id: str) -> None:
        self._entry_blobs.pop(history_id, None)
        blob = self._blobs.get(blob_id)
        if blob is not None:
            blob.references = max(0, blob.references - 1)
            if blob.references == 0:
                blob.pending_delete = True
                self._try_delete_blob_locked(blob)

    def _release_lease(self, token: str) -> None:
        with self._lock:
            blob_id = self._leases.pop(token, None)
            blob = self._blobs.get(blob_id or "")
            if blob is None:
                return
            blob.pins = max(0, blob.pins - 1)
            if blob.references == 0:
                blob.pending_delete = True
            self._try_delete_blob_locked(blob)

    def _try_delete_blob_locked(self, blob: _Blob) -> None:
        if not blob.pending_delete or blob.references or blob.pins:
            return
        if not self._unlink_best_effort(blob.path):
            return
        self._stored_bytes = max(0, self._stored_bytes - blob.size_bytes)
        self._blobs.pop(blob.blob_id, None)

    def _collect_garbage_locked(self) -> None:
        for blob in list(self._blobs.values()):
            self._try_delete_blob_locked(blob)

    def _invalidate_missing_blob_locked(self, blob_id: str) -> None:
        blob = self._blobs.get(blob_id)
        if blob is None:
            return
        for history_id, candidate in list(self._entry_blobs.items()):
            if candidate == blob_id:
                self._entry_blobs.pop(history_id, None)
                record = self._records.get(history_id)
                if record is not None:
                    record.clear()
                    record.update(status="failed", reason="capture_failed")
        self._stored_bytes = max(0, self._stored_bytes - blob.size_bytes)
        self._blobs.pop(blob_id, None)

    def _purge_locked(self, *, reason: str) -> None:
        self._generation += 1
        for history_id in list(self._entry_blobs):
            self._remove_entry_record_blob_only_locked(history_id)
        for record in self._records.values():
            record.clear()
            record.update(status="disabled" if reason == "disabled" else "not_captured")
        for blob in list(self._blobs.values()):
            if blob.references == 0:
                blob.pending_delete = True
            self._try_delete_blob_locked(blob)

    def _remove_entry_record_blob_only_locked(self, history_id: str) -> None:
        blob_id = self._entry_blobs.pop(history_id, None)
        blob = self._blobs.get(blob_id or "")
        if blob is None:
            return
        blob.references = max(0, blob.references - 1)
        if blob.references == 0:
            blob.pending_delete = True

    def _set_capture_failure(self, history_id: str, reason: str) -> dict[str, Any]:
        with self._lock:
            if not self._enabled or history_id not in self._records:
                return self._public_metadata_locked(history_id)
            return self._set_failure_locked(history_id, reason)

    def _set_failure_locked(self, history_id: str, reason: str) -> dict[str, Any]:
        record = self._records.get(history_id)
        if record is not None:
            record.clear()
            record.update(status="failed", reason=reason)
        return self._public_metadata_locked(history_id)

    def _finish_capture_failure(self, token: str) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_captures.pop(token, None)
            if pending is None:
                return {"status": "not_retained", "reason": "capture_failed"}
            self._reserved_bytes = max(0, self._reserved_bytes - pending.expected_size)
            record = self._records.get(pending.history_id)
            if (
                record is not None
                and record.get("token") == token
                and pending.generation == self._generation
                and self._enabled
            ):
                record.clear()
                record.update(status="failed", reason="capture_failed")
            return self._public_metadata_locked(pending.history_id)

    def _public_metadata_locked(self, history_id: str) -> dict[str, Any]:
        record = self._records.get(history_id)
        if record is None:
            return {"status": "not_retained", "reason": "not_found"}
        status = record.get("status")
        if status == "available":
            blob = self._blobs.get(str(record.get("blob_id") or ""))
            if blob is not None:
                return {
                    "status": "available",
                    "filename": blob.filename,
                    "content_type": blob.content_type,
                    "size_bytes": blob.size_bytes,
                    "capture_duration_ms": blob.capture_duration_ms,
                }
            return {"status": "not_retained", "reason": "capture_failed"}
        reason_by_status = {
            "disabled": "disabled",
            "not_captured": "not_captured",
            "capturing": "capture_in_progress",
            "failed": str(record.get("reason") or "capture_failed"),
        }
        return {"status": "not_retained", "reason": reason_by_status.get(status, "not_captured")}

    def _sanitize_entry_locked(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        sanitized = copy.deepcopy(dict(entry))
        sanitized.pop("debug_audio_path", None)
        sanitized.pop("audio_path", None)
        sanitized.pop("blob_id", None)
        sanitized.pop("debug_audio", None)
        history_id = str(sanitized.get("history_id") or "")
        sanitized["debug_audio"] = self._public_metadata_locked(history_id)
        return sanitized

    def _prepare_directory(self) -> None:
        if self._directory_is_unsafe():
            raise OSError("Debug-audio directory must not be a symlink, junction, or reparse point.")
        self._directory.mkdir(parents=True, exist_ok=True)
        if self._directory_is_unsafe():
            raise OSError("Debug-audio directory changed to an unsafe reparse point.")
        with contextlib.suppress(OSError):
            os.chmod(self._directory, 0o700)

    @staticmethod
    def _measure_source(source) -> int:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return len(source)
        if isinstance(source, (str, os.PathLike)):
            return Path(source).stat().st_size
        original_position = source.tell()
        try:
            source.seek(0, os.SEEK_END)
            return int(source.tell())
        finally:
            source.seek(original_position, os.SEEK_SET)

    @staticmethod
    def _copy_source(source, destination: Path) -> int:
        close_source = False
        original_position: int | None = None
        if isinstance(source, (bytes, bytearray, memoryview)):
            input_stream = io.BytesIO(bytes(source))
            close_source = True
        elif isinstance(source, (str, os.PathLike)):
            input_stream = Path(source).open("rb")
            close_source = True
        else:
            input_stream = source
            original_position = input_stream.tell()
            input_stream.seek(0, os.SEEK_SET)

        copied = 0
        try:
            with destination.open("xb") as output:
                while True:
                    chunk = input_stream.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                    copied += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if original_position is not None:
                with contextlib.suppress(Exception):
                    input_stream.seek(original_position, os.SEEK_SET)
            if close_source:
                input_stream.close()
        return copied

    def _directory_is_unsafe(self) -> bool:
        try:
            if self._directory.is_symlink():
                return True
            is_junction = getattr(self._directory, "is_junction", None)
            if is_junction is not None and is_junction():
                return True
            try:
                attributes = int(getattr(os.lstat(self._directory), "st_file_attributes", 0))
            except FileNotFoundError:
                return False
            return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))
        except OSError:
            return True

    def _unlink_best_effort(self, path: Path) -> bool:
        if path.parent != self._directory or self._directory_is_unsafe():
            return False
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _remove_orphan_files(self) -> None:
        with self._lock:
            protected = {blob.path for blob in self._blobs.values() if blob.pins > 0}
            protected.update(pending.part_path for pending in self._pending_captures.values())
            protected.update(pending.final_path for pending in self._pending_captures.values())
        if self._directory_is_unsafe() or not self._directory.is_dir():
            return
        for candidate in self._directory.iterdir():
            if candidate in protected or not (candidate.is_file() or candidate.is_symlink()):
                continue
            self._unlink_best_effort(candidate)


_history_audio_store = HistoryDebugAudioStore(
    transcription_history,
    history_lock,
    Path(LOGS_DIR) / "debug-audio",
)


def set_history_audio_enabled(enabled: bool) -> None:
    _history_audio_store.set_enabled(enabled)


def append_history_entry(entry: MutableMapping[str, Any] | Mapping[str, Any], *, existing_blob_id: str | None = None) -> str:
    return _history_audio_store.append(entry, existing_blob_id=existing_blob_id)


def get_history_snapshot(limit: int = VISIBLE_HISTORY_LIMIT) -> list[dict[str, Any]]:
    return _history_audio_store.snapshot(limit)


def get_history_entry(history_id: str) -> dict[str, Any] | None:
    return _history_audio_store.get_entry(history_id)


def capture_history_audio(
    history_id: str,
    file_obj: BinaryIO | bytes | bytearray | memoryview | str | os.PathLike[str],
    filename: str | None,
    content_type: str | None,
) -> dict[str, Any]:
    return _history_audio_store.capture(history_id, file_obj, filename, content_type)


def acquire_history_audio(history_id: str) -> HistoryAudioLease | None:
    return _history_audio_store.acquire(history_id)


def attach_history_audio(history_id: str, blob_id: str) -> bool:
    return _history_audio_store.attach(history_id, blob_id)


def begin_history_retry(history_id: str) -> bool:
    return _history_audio_store.begin_retry(history_id)


def end_history_retry(history_id: str) -> None:
    _history_audio_store.end_retry(history_id)


def purge_history_audio() -> None:
    _history_audio_store.purge()


def reset_history_audio_store(*, enabled: bool = False) -> None:
    _history_audio_store.reset(enabled=enabled)


def shutdown_history_audio_store() -> None:
    _history_audio_store.shutdown()


__all__ = [
    "HistoryAudioLease",
    "HistoryDebugAudioStore",
    "MAX_DEBUG_AUDIO_FILE_BYTES",
    "MAX_DEBUG_AUDIO_TOTAL_BYTES",
    "VISIBLE_HISTORY_LIMIT",
    "acquire_history_audio",
    "append_history_entry",
    "attach_history_audio",
    "begin_history_retry",
    "capture_history_audio",
    "end_history_retry",
    "get_history_entry",
    "get_history_snapshot",
    "purge_history_audio",
    "reset_history_audio_store",
    "set_history_audio_enabled",
    "shutdown_history_audio_store",
]
