import asyncio
import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as legacy_api
from backend import genesis_whisper_server_history as history
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_globals import (
    current_settings,
    history_lock,
    settings_lock,
    transcription_history,
)


class AdminHistoryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-history-admin-routes-")
        self.audio_dir = Path(self.temp_dir.name) / "debug-audio"
        self.original_settings = current_settings.copy()

        history.reset_history_audio_store(enabled=False)
        with history_lock:
            transcription_history.clear()

        self.store = history.HistoryDebugAudioStore(
            transcription_history,
            history_lock,
            self.audio_dir,
            visible_limit=25,
        )
        self.store.set_enabled(True)
        self.store_patch = mock.patch.object(history, "_history_audio_store", self.store)
        self.store_patch.start()

        with settings_lock:
            current_settings.clear()
            current_settings.update(
                storage.normalize_settings(
                    {
                        **storage.DEFAULT_SETTINGS,
                        "debug_retain_history_audio": True,
                    }
                )
            )

        self.app = FastAPI()
        legacy_api.create_api(self.app)
        v2.create_v2_api(self.app)
        admin.create_admin_api(self.app)
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.store.shutdown()
        self.store_patch.stop()
        history.reset_history_audio_store(enabled=False)
        with history_lock:
            transcription_history.clear()
        with settings_lock:
            current_settings.clear()
            current_settings.update(self.original_settings)
        self.temp_dir.cleanup()

    def _append_with_audio(
        self,
        *,
        mode: str = "transcript",
        retry_mode: str | None = "transcript",
        audio: bytes = b"byte-identical retained audio",
        filename: str = "voice.m4a",
        content_type: str = "audio/mp4",
        **extra,
    ) -> str:
        entry = {
            "timestamp": "2026-07-23 02:00:00",
            "source_ip": "127.0.0.1",
            "engine": "v2",
            "mode": mode,
            "model_id": "mock-asr",
            "total_duration_ms": 40,
            "transcription_duration_ms": 12,
            "voice_vector_duration_ms": 7 if mode == "transcript_embedding" else None,
            "transcript": "Original transcript",
            "retry_of": None,
            "retry_mode": retry_mode,
            **extra,
        }
        history_id = history.append_history_entry(entry)
        metadata = history.capture_history_audio(
            history_id,
            audio,
            filename,
            content_type,
        )
        self.assertEqual(metadata["status"], "available")
        return history_id

    def test_download_and_retry_routes_require_admin_authentication(self) -> None:
        history_id = self._append_with_audio()
        self.app.dependency_overrides.pop(admin.require_admin)
        try:
            with TestClient(self.app) as client:
                download = client.get(f"/api/admin/history/{history_id}/audio")
                retry = client.post(f"/api/admin/history/{history_id}/retry")
        finally:
            self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}

        self.assertEqual(download.status_code, 401, download.text)
        self.assertEqual(retry.status_code, 401, retry.text)

    def test_download_is_exact_sanitized_and_never_cacheable(self) -> None:
        original = b"\x00\x01exact-m4a-upload\xff"
        history_id = self._append_with_audio(
            audio=original,
            filename="C:\\private\\recordings\\voice.m4a",
            content_type="audio/mp4",
        )

        with TestClient(self.app) as client:
            response = client.get(f"/api/admin/history/{history_id}/audio")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, original)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertTrue(response.headers["content-type"].startswith("audio/mp4"))
        disposition = response.headers["content-disposition"]
        self.assertIn("voice.m4a", disposition)
        self.assertNotIn("private", disposition)
        self.assertNotIn("recordings", disposition)
        self.assertNotIn("..", disposition)

    def test_invalid_range_releases_download_lease_before_purge(self) -> None:
        history_id = self._append_with_audio(audio=b"range-test-audio")
        retained = history.acquire_history_audio(history_id)
        self.assertIsNotNone(retained)
        retained_path = retained.path
        retained.release()

        with TestClient(self.app) as client:
            response = client.get(
                f"/api/admin/history/{history_id}/audio",
                headers={"Range": "definitely-not-a-byte-range"},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.store.set_enabled(False)
        self.assertFalse(retained_path.exists())

    def test_interrupted_file_send_releases_download_lease_before_purge(self) -> None:
        history_id = self._append_with_audio(audio=b"interrupted-download")
        lease = history.acquire_history_audio(history_id)
        self.assertIsNotNone(lease)
        retained_path = lease.path
        response = admin._HistoryAudioFileResponse(
            lease.path,
            lease=lease,
            media_type=lease.content_type,
            filename=lease.filename,
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise ConnectionError("synthetic client disconnect")

        with self.assertRaises(ConnectionError):
            asyncio.run(
                response(
                    {
                        "type": "http",
                        "http_version": "1.1",
                        "method": "GET",
                        "scheme": "http",
                        "path": "/audio",
                        "raw_path": b"/audio",
                        "query_string": b"",
                        "headers": [],
                        "client": ("127.0.0.1", 1),
                        "server": ("testserver", 80),
                        "extensions": {},
                    },
                    receive,
                    send,
                )
            )

        self.store.set_enabled(False)
        self.assertFalse(retained_path.exists())

    def test_unavailable_audio_returns_404_for_download_and_retry(self) -> None:
        history_id = history.append_history_entry(
            {
                "mode": "transcript",
                "retry_mode": "transcript",
                "retry_of": None,
                "transcription_duration_ms": 1,
                "voice_vector_duration_ms": None,
            }
        )

        with TestClient(self.app) as client:
            download = client.get(f"/api/admin/history/{history_id}/audio")
            retry = client.post(f"/api/admin/history/{history_id}/retry")
            missing = client.post("/api/admin/history/missing-history-id/retry")

        self.assertEqual(download.status_code, 404, download.text)
        self.assertEqual(retry.status_code, 404, retry.text)
        self.assertEqual(missing.status_code, 404, missing.text)

    def test_duplicate_retry_returns_409_without_starting_asr(self) -> None:
        history_id = self._append_with_audio()
        self.assertTrue(history.begin_history_retry(history_id))
        try:
            with (
                mock.patch.object(v2, "_transcribe_audio", new=mock.AsyncMock()) as transcribe,
                TestClient(self.app) as client,
            ):
                response = client.post(f"/api/admin/history/{history_id}/retry")
        finally:
            history.end_history_retry(history_id)

        self.assertEqual(response.status_code, 409, response.text)
        transcribe.assert_not_awaited()

    def test_unsupported_history_mode_returns_422(self) -> None:
        history_id = self._append_with_audio(mode="diarization", retry_mode=None)

        with TestClient(self.app) as client:
            response = client.post(f"/api/admin/history/{history_id}/retry")

        self.assertEqual(response.status_code, 422, response.text)

    def test_transcript_and_embedding_retries_create_linked_timed_rows_and_reuse_blob(self) -> None:
        original = b"one retained blob for source and retry"
        audio_array = np.full(16_000, 0.1, dtype=np.float32)

        for mode in ("transcript", "transcript_embedding"):
            with self.subTest(mode=mode):
                source_id = self._append_with_audio(
                    mode=mode,
                    retry_mode=mode,
                    audio=original,
                    filename=f"{mode}.m4a",
                )
                embedding = mock.AsyncMock(return_value=([0.0] * 192, 7))
                with (
                    mock.patch.object(admin, "load_audio_file", return_value=audio_array),
                    mock.patch.object(
                        v2,
                        "_transcribe_audio",
                        new=mock.AsyncMock(return_value=(f"Retried {mode}", 12, 1, "mock-asr")),
                    ),
                    mock.patch.object(v2, "_generate_embedding", new=embedding),
                    mock.patch.object(v2, "log_transcription"),
                    TestClient(self.app) as client,
                ):
                    response = client.post(f"/api/admin/history/{source_id}/retry")
                    self.assertEqual(response.status_code, 200, response.text)
                    payload = response.json()
                    stats = client.get("/api/admin/stats")
                    self.assertEqual(stats.status_code, 200, stats.text)
                    retry_download = client.get(
                        f"/api/admin/history/{payload['history_id']}/audio"
                    )

                self.assertTrue(payload["ok"])
                self.assertNotEqual(payload["history_id"], source_id)
                self.assertEqual(payload["retry_of"], source_id)
                self.assertEqual(payload["mode"], mode)
                self.assertEqual(payload["timings_ms"]["transcription"], 12)
                self.assertIn("decode", payload["timings_ms"])
                self.assertIn("total", payload["timings_ms"])
                if mode == "transcript_embedding":
                    self.assertIn("embedding", payload["timings_ms"])
                    embedding.assert_awaited_once()
                else:
                    self.assertNotIn("embedding", payload["timings_ms"])
                    embedding.assert_not_awaited()

                retry_entry = next(
                    item
                    for item in stats.json()["history"]
                    if item["history_id"] == payload["history_id"]
                )
                self.assertEqual(retry_entry["retry_of"], source_id)
                self.assertEqual(retry_entry["retry_mode"], mode)
                self.assertEqual(retry_entry["transcription_duration_ms"], 12)
                if mode == "transcript_embedding":
                    self.assertIsNotNone(retry_entry["voice_vector_duration_ms"])
                else:
                    self.assertIsNone(retry_entry["voice_vector_duration_ms"])
                self.assertEqual(retry_entry["debug_audio"]["status"], "available")
                self.assertEqual(retry_download.status_code, 200, retry_download.text)
                self.assertEqual(retry_download.content, original)
                self.assertEqual(len(list(self.audio_dir.glob("*.bin"))), 1)

                self.store.reset(enabled=True)
                with history_lock:
                    transcription_history.clear()

    def test_stats_never_exposes_audio_paths_blob_ids_or_base64(self) -> None:
        secret_audio = b"RAW-AUDIO-MUST-NOT-APPEAR-IN-STATS"
        history_id = self._append_with_audio(
            audio=secret_audio,
            filename="C:\\sensitive\\speaker.m4a",
            debug_audio_path=str(self.audio_dir / "secret.bin"),
            audio_path=str(self.audio_dir / "secret.bin"),
            blob_id="private-blob-id",
            debug_audio={"path": str(self.audio_dir / "secret.bin")},
        )

        with TestClient(self.app) as client:
            response = client.get("/api/admin/stats")

        self.assertEqual(response.status_code, 200, response.text)
        entry = next(item for item in response.json()["history"] if item["history_id"] == history_id)
        self.assertEqual(entry["debug_audio"]["status"], "available")
        self.assertEqual(entry["debug_audio"]["filename"], "speaker.m4a")
        serialized = response.text
        self.assertNotIn(str(self.audio_dir), serialized)
        self.assertNotIn("private-blob-id", serialized)
        self.assertNotIn("blob_id", serialized)
        self.assertNotIn("audio_path", serialized)
        self.assertNotIn("debug_audio_path", serialized)
        self.assertNotIn("base64", serialized.lower())
        self.assertNotIn(secret_audio.decode("ascii"), serialized)

    def test_successful_public_v2_request_captures_exact_original_upload(self) -> None:
        original = b"original-container-bytes-not-decoded-pcm"
        decoded = np.full(16_000, 0.1, dtype=np.float32)
        request_json = json.dumps({"schema_version": "2.0", "mode": "transcript"})

        with (
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", return_value=decoded),
            mock.patch.object(
                v2,
                "_transcribe_audio",
                new=mock.AsyncMock(return_value=("Public transcript", 9, 1, "mock-asr")),
            ),
            mock.patch.object(v2, "log_transcription"),
            TestClient(self.app) as client,
        ):
            response = client.post(
                "/v2/audio/process",
                files={
                    "file": ("original.m4a", original, "audio/mp4"),
                    "request": (None, request_json, "application/json"),
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            stats = client.get("/api/admin/stats")
            download = client.get(
                f"/api/admin/history/{payload['request_id']}/audio"
            )

        self.assertEqual(download.status_code, 200, download.text)
        self.assertEqual(download.content, original)
        entry = next(
            item
            for item in stats.json()["history"]
            if item["history_id"] == payload["request_id"]
        )
        self.assertEqual(entry["mode"], "transcript")
        self.assertEqual(entry["transcription_duration_ms"], 9)
        self.assertIsNone(entry["voice_vector_duration_ms"])
        self.assertEqual(entry["debug_audio"]["status"], "available")
        self.assertEqual(entry["debug_audio"]["size_bytes"], len(original))

    def test_legacy_and_v1_successes_capture_the_exact_original_upload(self) -> None:
        original = b"legacy-container-bytes-not-decoded-pcm"
        decoded = np.full(16_000, 0.1, dtype=np.float32)

        for endpoint, form in (
            ("/transcribe/", {"engine": "local", "voice_ident": "false"}),
            ("/v1/audio/transcriptions", {"model": "whisper-1"}),
        ):
            with self.subTest(endpoint=endpoint):
                batch_manager = mock.Mock()
                batch_manager.enqueue = mock.AsyncMock(
                    return_value=SimpleNamespace(text="Legacy transcript", batch_id="batch-1")
                )
                self.app.state.whisper_batch_manager = batch_manager
                with (
                    mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
                    mock.patch.object(legacy_api, "load_audio_file", return_value=decoded),
                    mock.patch.object(legacy_api, "uses_cohere_backend", return_value=True),
                    mock.patch.object(legacy_api, "log_transcription"),
                    TestClient(self.app) as client,
                ):
                    response = client.post(
                        endpoint,
                        data=form,
                        files={"file": ("original.m4a", original, "audio/mp4")},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    stats = client.get("/api/admin/stats")
                    self.assertEqual(stats.status_code, 200, stats.text)
                    entry = stats.json()["history"][0]
                    download = client.get(
                        f"/api/admin/history/{entry['history_id']}/audio"
                    )

                self.assertEqual(download.status_code, 200, download.text)
                self.assertEqual(download.content, original)
                self.assertEqual(entry["mode"], "transcript")
                self.assertEqual(entry["retry_mode"], "transcript")
                self.assertEqual(entry["debug_audio"]["status"], "available")

                self.store.reset(enabled=True)
                with history_lock:
                    transcription_history.clear()

    def test_every_v2_mode_retains_audio_but_only_transcript_modes_are_retryable(self) -> None:
        original = b"one-original-for-every-v2-mode"
        decoded = np.full(16_000, 0.1, dtype=np.float32)

        for mode in ("embedding", "transcript", "transcript_embedding", "diarization"):
            with self.subTest(mode=mode):
                request_json = json.dumps({"schema_version": "2.0", "mode": mode})
                with (
                    mock.patch.object(v2, "authorize_api_key", return_value=None),
                    mock.patch.object(v2, "load_audio_file", return_value=decoded),
                    mock.patch.object(
                        v2,
                        "_transcribe_audio",
                        new=mock.AsyncMock(return_value=("Transcript", 9, 1, "mock-asr")),
                    ),
                    mock.patch.object(
                        v2,
                        "_generate_embedding",
                        new=mock.AsyncMock(return_value=([0.0] * 192, 7)),
                    ),
                    mock.patch.object(
                        v2,
                        "_process_diarization",
                        new=mock.AsyncMock(
                            return_value=(
                                {"segments": [], "unknown_speakers": []},
                                {"dia": 3, "transcription": 9, "embedding": 7},
                                {"asr": {"id": "mock-asr"}},
                                [],
                                "Transcript",
                            )
                        ),
                    ),
                    mock.patch.object(v2, "log_transcription"),
                    TestClient(self.app) as client,
                ):
                    response = client.post(
                        "/v2/audio/process",
                        files={
                            "file": ("original.m4a", original, "audio/mp4"),
                            "request": (None, request_json, "application/json"),
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    request_id = response.json()["request_id"]
                    stats = client.get("/api/admin/stats")
                    entry = next(
                        item for item in stats.json()["history"]
                        if item["history_id"] == request_id
                    )
                    download = client.get(f"/api/admin/history/{request_id}/audio")

                self.assertEqual(download.status_code, 200, download.text)
                self.assertEqual(download.content, original)
                self.assertEqual(entry["mode"], mode)
                self.assertEqual(entry["debug_audio"]["status"], "available")
                expected_retry_mode = mode if mode in {"transcript", "transcript_embedding"} else None
                self.assertEqual(entry["retry_mode"], expected_retry_mode)

                self.store.reset(enabled=True)
                with history_lock:
                    transcription_history.clear()


if __name__ == "__main__":
    unittest.main()
