import io
import json
import os
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path

from backend.genesis_whisper_server_history import HistoryDebugAudioStore


class _BlockingBytesIO(io.BytesIO):
    def __init__(self, value: bytes, started: threading.Event, proceed: threading.Event):
        super().__init__(value)
        self._started = started
        self._proceed = proceed
        self._blocked_once = False

    def read(self, size=-1):
        if not self._blocked_once:
            self._blocked_once = True
            self._started.set()
            if not self._proceed.wait(timeout=5):
                raise TimeoutError("test capture was not released")
        return super().read(size)


class _BrokenBytesIO(io.BytesIO):
    def read(self, size=-1):
        raise OSError("synthetic read failure")


class HistoryDebugAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-history-audio-")
        self.audio_dir = Path(self.temp_dir.name) / "debug-audio"
        self.history = deque(maxlen=100)
        self.history_lock = threading.Lock()
        self.store = HistoryDebugAudioStore(
            self.history,
            self.history_lock,
            self.audio_dir,
            visible_limit=3,
            max_file_bytes=32,
            max_total_bytes=64,
        )

    def tearDown(self) -> None:
        self.store.shutdown()
        self.temp_dir.cleanup()

    def _append(self, **values):
        entry = {"timestamp": "2026-07-23 01:00:00", **values}
        history_id = self.store.append(entry)
        return history_id, entry

    def test_append_assigns_stable_id_and_disabled_metadata(self) -> None:
        history_id, entry = self._append(transcript="Hallo")

        self.assertEqual(entry["history_id"], history_id)
        self.assertEqual(len(history_id), 32)
        first = self.store.snapshot()[0]
        second = self.store.snapshot()[0]
        self.assertEqual(first["history_id"], second["history_id"])
        self.assertEqual(first["debug_audio"], {"status": "not_retained", "reason": "disabled"})

    def test_capture_is_byte_identical_preserves_position_and_sanitizes_metadata(self) -> None:
        self.store.set_enabled(True)
        history_id, entry = self._append(
            debug_audio={"path": "C:/secret/source.m4a"},
            debug_audio_path="C:/secret/source.m4a",
            audio_path="C:/secret/source.m4a",
            blob_id="must-not-leak",
        )
        source = io.BytesIO(b"0123456789")
        source.seek(4)

        metadata = self.store.capture(
            history_id,
            source,
            "../../unsafe\r\nname.m4a",
            "audio/mp4\r\nX-Evil: yes",
        )

        self.assertEqual(source.tell(), 4)
        self.assertEqual(metadata["status"], "available")
        self.assertEqual(metadata["filename"], "unsafename.m4a")
        self.assertEqual(metadata["content_type"], "application/octet-stream")
        self.assertEqual(metadata["size_bytes"], 10)
        self.assertGreaterEqual(metadata["capture_duration_ms"], 0)

        with self.store.acquire(history_id) as lease:
            self.assertEqual(lease.path.suffix, ".bin")
            with lease.open() as retained:
                self.assertEqual(retained.read(), b"0123456789")

        serialized = json.dumps(self.store.snapshot())
        self.assertNotIn(str(self.audio_dir), serialized)
        self.assertNotIn("must-not-leak", serialized)
        self.assertNotIn("debug_audio_path", serialized)
        self.assertNotIn("audio_path", serialized)
        self.assertNotIn("blob_id", serialized)
        self.assertNotIn(".part", serialized)
        self.assertNotIn("path", self.store.snapshot()[0]["debug_audio"])
        self.assertNotIn("debug_audio", entry)

    def test_safe_filename_handles_windows_paths_and_control_characters(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()

        metadata = self.store.capture(
            history_id,
            b"audio",
            "C:\\Users\\someone\\voice\x00.m4a",
            "Audio/MP4",
        )

        self.assertEqual(metadata["filename"], "voice.m4a")
        self.assertEqual(metadata["content_type"], "audio/mp4")

    def test_file_and_unique_blob_quota_failures_are_non_fatal(self) -> None:
        store = HistoryDebugAudioStore(
            self.history,
            self.history_lock,
            self.audio_dir,
            visible_limit=3,
            max_file_bytes=5,
            max_total_bytes=5,
        )
        store.set_enabled(True)
        too_large = store.append({})
        self.assertEqual(
            store.capture(too_large, b"123456", "large.wav", "audio/wav"),
            {"status": "not_retained", "reason": "file_too_large"},
        )

        source_id = store.append({})
        self.assertEqual(store.capture(source_id, b"12345", "ok.wav", "audio/wav")["status"], "available")
        with store.acquire(source_id) as lease:
            retry_id = store.append({"retry_of": source_id}, existing_blob_id=lease.blob_id)
        self.assertEqual(store.snapshot()[0]["debug_audio"]["status"], "available")

        quota_id = store.append({})
        self.assertEqual(
            store.capture(quota_id, b"x", "other.wav", "audio/wav"),
            {"status": "not_retained", "reason": "storage_quota"},
        )
        self.assertEqual(len(list(self.audio_dir.glob("*.bin"))), 1)
        lease = store.acquire(retry_id)
        self.assertIsNotNone(lease)
        if lease is not None:
            lease.release()
        store.shutdown()

    def test_copy_failure_removes_partial_file_and_reports_capture_failed(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()

        metadata = self.store.capture(history_id, _BrokenBytesIO(b"broken"), "bad.wav", "audio/wav")

        self.assertEqual(metadata, {"status": "not_retained", "reason": "capture_failed"})
        self.assertEqual(list(self.audio_dir.glob("*")), [])

    def test_repeated_capture_keeps_existing_blob_even_when_new_source_exceeds_limits(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()
        first = self.store.capture(history_id, b"kept", "voice.wav", "audio/wav")

        repeated = self.store.capture(history_id, b"x" * 128, "replacement.wav", "audio/wav")

        self.assertEqual(repeated, first)
        with self.store.acquire(history_id) as lease:
            with lease.open() as retained:
                self.assertEqual(retained.read(), b"kept")

    def test_pruning_from_visible_history_revokes_access_and_defers_delete_for_lease(self) -> None:
        store = HistoryDebugAudioStore(
            self.history,
            self.history_lock,
            self.audio_dir,
            visible_limit=2,
            max_file_bytes=32,
            max_total_bytes=64,
        )
        store.set_enabled(True)
        source_id = store.append({"name": "source"})
        store.capture(source_id, b"retained", "source.wav", "audio/wav")
        lease = store.acquire(source_id)
        self.assertIsNotNone(lease)
        retained_path = lease.path

        store.append({"name": "new-1"})
        store.append({"name": "new-2"})

        self.assertIsNone(store.get_entry(source_id))
        self.assertIsNone(store.acquire(source_id))
        self.assertTrue(retained_path.is_file())
        lease.release()
        self.assertFalse(retained_path.exists())
        store.shutdown()

    def test_existing_blob_is_atomically_attached_to_retry_without_recopy(self) -> None:
        store = HistoryDebugAudioStore(
            self.history,
            self.history_lock,
            self.audio_dir,
            visible_limit=2,
            max_file_bytes=32,
            max_total_bytes=64,
        )
        store.set_enabled(True)
        source_id = store.append({"name": "source"})
        store.capture(source_id, b"same-bytes", "voice.m4a", "audio/mp4")
        store.append({"name": "middle"})
        source_lease = store.acquire(source_id)
        self.assertIsNotNone(source_lease)

        retry_id = store.append(
            {"retry_of": source_id, "retry_mode": "transcript_embedding"},
            existing_blob_id=source_lease.blob_id,
        )
        source_lease.release()

        self.assertIsNone(store.acquire(source_id))
        retry_lease = store.acquire(retry_id)
        self.assertIsNotNone(retry_lease)
        with retry_lease:
            with retry_lease.open() as retained:
                self.assertEqual(retained.read(), b"same-bytes")
        self.assertEqual(len(list(self.audio_dir.glob("*.bin"))), 1)
        store.shutdown()

    def test_twenty_sixth_row_evicts_audio_from_visible_retention(self) -> None:
        history = deque(maxlen=100)
        store = HistoryDebugAudioStore(
            history,
            threading.Lock(),
            self.audio_dir,
            visible_limit=25,
            max_file_bytes=32,
            max_total_bytes=64,
        )
        store.set_enabled(True)
        source_id = store.append({"index": 0})
        store.capture(source_id, b"oldest", "old.wav", "audio/wav")
        initial_lease = store.acquire(source_id)
        self.assertIsNotNone(initial_lease)
        retained_path = initial_lease.path
        initial_lease.release()

        for index in range(1, 26):
            store.append({"index": index})

        self.assertEqual(len(store.snapshot()), 25)
        self.assertIsNone(store.get_entry(source_id))
        self.assertIsNone(store.acquire(source_id))
        self.assertFalse(retained_path.exists())
        store.shutdown()

    def test_disable_revokes_new_access_but_active_lease_finishes_before_delete(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()
        self.store.capture(history_id, b"pinned", "voice.wav", "audio/wav")
        lease = self.store.acquire(history_id)
        self.assertIsNotNone(lease)
        retained_path = lease.path

        self.store.set_enabled(False)

        self.assertIsNone(self.store.acquire(history_id))
        self.assertTrue(retained_path.is_file())
        with lease.open() as retained:
            self.assertEqual(retained.read(), b"pinned")
        self.assertEqual(
            self.store.snapshot()[0]["debug_audio"],
            {"status": "not_retained", "reason": "disabled"},
        )
        lease.release()
        self.assertFalse(retained_path.exists())

    def test_concurrent_capture_reservation_enforces_total_quota(self) -> None:
        store = HistoryDebugAudioStore(
            self.history,
            self.history_lock,
            self.audio_dir,
            visible_limit=3,
            max_file_bytes=10,
            max_total_bytes=10,
        )
        store.set_enabled(True)
        first_id = store.append({})
        second_id = store.append({})
        started = threading.Event()
        proceed = threading.Event()
        source = _BlockingBytesIO(b"123456", started, proceed)
        result: dict[str, object] = {}

        worker = threading.Thread(
            target=lambda: result.update(first=store.capture(first_id, source, "one.wav", "audio/wav")),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(timeout=3))

        second = store.capture(second_id, b"abcdef", "two.wav", "audio/wav")
        self.assertEqual(second, {"status": "not_retained", "reason": "storage_quota"})
        proceed.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["first"]["status"], "available")
        store.shutdown()

    def test_purge_during_capture_invalidates_generation_and_removes_completed_file(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()
        started = threading.Event()
        proceed = threading.Event()
        source = _BlockingBytesIO(b"in-flight", started, proceed)
        result: dict[str, object] = {}
        worker = threading.Thread(
            target=lambda: result.update(
                capture=self.store.capture(history_id, source, "flight.wav", "audio/wav")
            ),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(timeout=3))

        self.store.set_enabled(False)
        self.assertIsNone(self.store.acquire(history_id))
        proceed.set()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result["capture"], {"status": "not_retained", "reason": "disabled"})
        self.assertEqual(list(self.audio_dir.glob("*")), [])

    def test_reset_removes_orphans_and_reinitializes_visible_metadata(self) -> None:
        self.audio_dir.mkdir(parents=True)
        orphan = self.audio_dir / "orphan.bin.part"
        orphan.write_bytes(b"orphan")
        history_id, _ = self._append()

        self.store.reset(enabled=True)

        self.assertFalse(orphan.exists())
        self.assertEqual(
            self.store.get_entry(history_id)["debug_audio"],
            {"status": "not_retained", "reason": "not_captured"},
        )

    def test_symlinked_debug_directory_is_rejected_without_touching_target(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        sentinel = outside / "must-survive.txt"
        sentinel.write_bytes(b"do not delete")
        try:
            os.symlink(outside, self.audio_dir, target_is_directory=True)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")

        self.store.set_enabled(True)
        history_id, _ = self._append()
        metadata = self.store.capture(history_id, b"audio", "voice.wav", "audio/wav")
        self.store.reset(enabled=True)

        self.assertEqual(metadata, {"status": "not_retained", "reason": "capture_failed"})
        self.assertEqual(sentinel.read_bytes(), b"do not delete")
        self.assertEqual(list(outside.iterdir()), [sentinel])

    def test_retry_guard_is_thread_safe_and_reusable(self) -> None:
        history_id, _ = self._append()

        self.assertTrue(self.store.begin_retry(history_id))
        self.assertFalse(self.store.begin_retry(history_id))
        self.store.end_retry(history_id)
        self.assertTrue(self.store.begin_retry(history_id))
        self.store.end_retry(history_id)

    @unittest.skipIf(os.name == "nt", "POSIX mode bits are not meaningful on Windows")
    def test_directory_and_file_permissions_are_restrictive(self) -> None:
        self.store.set_enabled(True)
        history_id, _ = self._append()
        self.store.capture(history_id, b"secure", "voice.wav", "audio/wav")
        lease = self.store.acquire(history_id)
        self.assertIsNotNone(lease)
        try:
            self.assertEqual(self.audio_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(lease.path.stat().st_mode & 0o777, 0o600)
        finally:
            lease.release()


if __name__ == "__main__":
    unittest.main()
