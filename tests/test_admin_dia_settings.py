import asyncio
import json
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_storage as storage


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.closed = False

    def json(self):
        return self._payload

    def close(self):
        self.closed = True


class AdminDiaSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-admin-tests-")
        self.settings_file = Path(self.temp_dir.name) / "settings.json"
        self.original_settings = admin.current_settings.copy()
        self.patches = [
            mock.patch.object(storage, "LOGS_DIR", self.temp_dir.name),
            mock.patch.object(storage, "SETTINGS_FILE", str(self.settings_file)),
            mock.patch.object(admin, "list_model_statuses", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()

        admin.current_settings.clear()
        admin.current_settings.update(
            storage.normalize_settings(
                {
                    **storage.DEFAULT_SETTINGS,
                    "dia_server_base_url": "http://saved-dia:7864",
                    "dia_api_key": "saved-secret",
                }
            )
        )
        self.app = admin.create_admin_api(FastAPI())

    def tearDown(self) -> None:
        admin.current_settings.clear()
        admin.current_settings.update(self.original_settings)
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def _endpoint(self, path: str, method: str):
        for route in self.app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise AssertionError(f"Route not found: {method} {path}")

    def test_settings_response_redacts_dia_key(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "GET")

        response = asyncio.run(endpoint({"username": "admin"}))

        settings = response["settings"]
        self.assertEqual(settings["dia_api_key"], "")
        self.assertTrue(settings["dia_api_key_configured"])
        self.assertEqual(settings["dia_api_key_source"], "settings")
        self.assertNotIn("saved-secret", json.dumps(response))

    def test_partial_legacy_update_preserves_dia_fields_and_blank_key(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")

        response = asyncio.run(
            endpoint(admin.AdminSettingsPayload(batch_wait_time_ms=123, dia_api_key=""), {"username": "admin"})
        )

        self.assertEqual(admin.current_settings["batch_wait_time_ms"], 123)
        self.assertEqual(admin.current_settings["dia_server_base_url"], "http://saved-dia:7864")
        self.assertEqual(admin.current_settings["dia_api_key"], "saved-secret")
        self.assertEqual(response["settings"]["dia_api_key"], "")
        self.assertNotIn("saved-secret", json.dumps(response))
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["dia_api_key"], "saved-secret")

    def test_cuda_trim_setting_defaults_off_and_survives_partial_updates(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertFalse(storage.DEFAULT_SETTINGS["cuda_memory_trim_after_batch"])
        self.assertFalse(admin.current_settings["cuda_memory_trim_after_batch"])

        enabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cuda_memory_trim_after_batch=True),
                {"username": "admin"},
            )
        )
        self.assertTrue(enabled["settings"]["cuda_memory_trim_after_batch"])

        partial = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(batch_wait_time_ms=7),
                {"username": "admin"},
            )
        )
        self.assertTrue(partial["settings"]["cuda_memory_trim_after_batch"])

        disabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cuda_memory_trim_after_batch=False),
                {"username": "admin"},
            )
        )
        self.assertFalse(disabled["settings"]["cuda_memory_trim_after_batch"])
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertFalse(persisted["cuda_memory_trim_after_batch"])

    def test_debug_history_audio_setting_defaults_off_is_strict_and_survives_partial_updates(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertFalse(storage.DEFAULT_SETTINGS["debug_retain_history_audio"])
        self.assertFalse(admin.current_settings["debug_retain_history_audio"])
        self.assertFalse(
            storage.normalize_settings(
                {**storage.DEFAULT_SETTINGS, "debug_retain_history_audio": "true"}
            )["debug_retain_history_audio"]
        )
        self.assertTrue(
            storage.normalize_settings(
                {**storage.DEFAULT_SETTINGS, "debug_retain_history_audio": True}
            )["debug_retain_history_audio"]
        )

        enabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(debug_retain_history_audio=True),
                {"username": "admin"},
            )
        )
        self.assertTrue(enabled["settings"]["debug_retain_history_audio"])

        partial = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(batch_wait_time_ms=7),
                {"username": "admin"},
            )
        )
        self.assertTrue(partial["settings"]["debug_retain_history_audio"])

        disabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(debug_retain_history_audio=False),
                {"username": "admin"},
            )
        )
        self.assertFalse(disabled["settings"]["debug_retain_history_audio"])
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertFalse(persisted["debug_retain_history_audio"])

    def test_debug_history_audio_setting_rejects_non_boolean_json_values(self) -> None:
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        try:
            with TestClient(self.app) as client:
                for invalid_value in ("true", "false", 1, 0):
                    with self.subTest(invalid_value=invalid_value):
                        response = client.put(
                            "/api/admin/settings",
                            json={"debug_retain_history_audio": invalid_value},
                        )
                        self.assertEqual(response.status_code, 422, response.text)
        finally:
            self.app.dependency_overrides.clear()

    def test_disabling_debug_history_audio_immediately_updates_the_retention_store(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        admin.current_settings["debug_retain_history_audio"] = True

        with mock.patch.object(admin, "set_history_audio_enabled") as set_enabled:
            response = asyncio.run(
                endpoint(
                    admin.AdminSettingsPayload(debug_retain_history_audio=False),
                    {"username": "admin"},
                )
            )

        self.assertFalse(response["settings"]["debug_retain_history_audio"])
        set_enabled.assert_called_once_with(False)

    def test_replacing_write_only_key_never_reflects_it_in_response(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")

        response = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(dia_api_key="replacement-secret"),
                types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace())),
                {"username": "admin"},
            )
        )

        self.assertEqual(admin.current_settings["dia_api_key"], "replacement-secret")
        self.assertEqual(response["settings"]["dia_api_key"], "")
        self.assertTrue(response["settings"]["dia_api_key_configured"])
        self.assertNotIn("replacement-secret", json.dumps(response))
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["dia_api_key"], "replacement-secret")

    def test_explicit_delete_removes_saved_key_but_reports_env_fallback(self) -> None:
        endpoint = self._endpoint("/api/admin/settings/dia-api-key", "DELETE")

        with mock.patch.dict(os.environ, {"DIA_SERVER_API_KEY": "environment-secret"}):
            response = asyncio.run(endpoint({"username": "admin"}))

        self.assertTrue(response["removed"])
        self.assertTrue(response["environment_fallback_active"])
        self.assertEqual(admin.current_settings["dia_api_key"], "")
        self.assertEqual(response["settings"]["dia_api_key"], "")
        self.assertNotIn("environment-secret", json.dumps(response))

    def test_connection_test_uses_candidate_key_without_returning_it(self) -> None:
        endpoint = self._endpoint("/api/admin/dia/test", "POST")
        fake_response = _FakeResponse(payload={"api_version": "2.0"})

        with mock.patch.object(admin.requests, "get", return_value=fake_response) as request_get:
            response = asyncio.run(
                endpoint(
                    admin.DiaConnectionTestPayload(
                        dia_server_base_url="https://dia.example.test/root/",
                        dia_api_key="candidate-secret",
                    ),
                    {"username": "admin"},
                )
            )

        request_get.assert_called_once_with(
            "https://dia.example.test/root/v2/capabilities",
            headers={"Accept": "application/json", "X-API-Key": "candidate-secret"},
            timeout=(3.05, 10.0),
            allow_redirects=False,
        )
        self.assertTrue(fake_response.closed)
        self.assertTrue(response["ok"])
        self.assertNotIn("candidate-secret", json.dumps(response))

    def test_connection_test_redacts_key_on_auth_failure(self) -> None:
        endpoint = self._endpoint("/api/admin/dia/test", "POST")
        fake_response = _FakeResponse(status_code=401, payload={"detail": "candidate-secret"})

        with mock.patch.object(admin.requests, "get", return_value=fake_response):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    endpoint(
                        admin.DiaConnectionTestPayload(dia_api_key="candidate-secret"),
                        {"username": "admin"},
                    )
                )

        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("candidate-secret", raised.exception.detail)
        self.assertTrue(fake_response.closed)

    def test_environment_fallback_does_not_enter_normalized_settings(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DIA_SERVER_BASE_URL": "http://environment-dia:7864/",
                "DIA_SERVER_API_KEY": "environment-secret",
            },
        ):
            normalized = storage.normalize_settings(storage.DEFAULT_SETTINGS)
            effective = storage.resolve_dia_server_config(normalized)

        self.assertEqual(normalized["dia_server_base_url"], "")
        self.assertEqual(normalized["dia_api_key"], "")
        self.assertEqual(effective["base_url"], "http://environment-dia:7864")
        self.assertEqual(effective["api_key"], "environment-secret")
        self.assertEqual(effective["base_url_source"], "environment")
        self.assertEqual(effective["api_key_source"], "environment")

    def test_valid_environment_fallback_is_visible_without_exposing_key(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "GET")
        admin.current_settings["dia_server_base_url"] = ""
        admin.current_settings["dia_api_key"] = ""

        with mock.patch.dict(
            os.environ,
            {
                "DIA_SERVER_BASE_URL": "http://environment-dia:7864/",
                "DIA_SERVER_API_KEY": "environment-secret",
            },
        ):
            response = asyncio.run(endpoint({"username": "admin"}))

        settings = response["settings"]
        self.assertEqual(settings["dia_server_base_url"], "")
        self.assertEqual(settings["dia_server_base_url_effective"], "http://environment-dia:7864")
        self.assertEqual(settings["dia_server_base_url_source"], "environment")
        self.assertEqual(settings["dia_api_key"], "")
        self.assertEqual(settings["dia_api_key_source"], "environment")
        self.assertTrue(settings["dia_api_key_configured"])
        self.assertNotIn("environment-secret", json.dumps(response))

    def test_malformed_environment_url_is_not_reflected_to_admin(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "GET")
        admin.current_settings["dia_server_base_url"] = ""

        with mock.patch.dict(
            os.environ,
            {"DIA_SERVER_BASE_URL": "https://user:password@dia.example.test?token=secret"},
        ):
            response = asyncio.run(endpoint({"username": "admin"}))

        self.assertEqual(response["settings"]["dia_server_base_url_effective"], "")
        self.assertEqual(response["settings"]["dia_server_base_url_source"], "environment")
        self.assertNotIn("password", json.dumps(response))
        self.assertNotIn("token=secret", json.dumps(response))

    def test_invalid_or_credentialed_dia_url_is_rejected(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")

        for invalid_url in ("dia:7864", "http://user:password@dia:7864", "https://dia:7864?key=secret"):
            with self.subTest(invalid_url=invalid_url):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(
                        endpoint(
                            admin.AdminSettingsPayload(dia_server_base_url=invalid_url),
                            {"username": "admin"},
                        )
                    )
                self.assertEqual(raised.exception.status_code, 422)

    def test_cancelled_model_reload_holds_local_gpu_lock_until_worker_stops(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        worker_started = threading.Event()
        release_worker = threading.Event()

        def blocking_model_load(_settings):
            worker_started.set()
            self.assertTrue(release_worker.wait(timeout=2.0))
            return True

        async def scenario() -> None:
            local_gpu_lock = asyncio.Lock()
            request = types.SimpleNamespace(
                app=types.SimpleNamespace(state=types.SimpleNamespace(local_gpu_lock=local_gpu_lock))
            )
            with mock.patch.object(admin, "_load_model_for_settings", side_effect=blocking_model_load):
                task = asyncio.create_task(
                    endpoint(
                        admin.AdminSettingsPayload(local_gpu_device="cpu"),
                        request,
                        {"username": "admin"},
                    )
                )
                self.assertTrue(await asyncio.to_thread(worker_started.wait, 1.0))
                task.cancel()
                await asyncio.sleep(0.02)

                self.assertFalse(task.done())
                self.assertTrue(local_gpu_lock.locked())

                release_worker.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(local_gpu_lock.locked())

        asyncio.run(scenario())

    def test_concurrent_model_reloads_are_serialized_by_local_gpu_lock(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        first_worker_started = threading.Event()
        release_first_worker = threading.Event()
        active_workers = 0
        max_active_workers = 0
        worker_calls = 0
        state_lock = threading.Lock()

        def blocking_model_load(_settings):
            nonlocal active_workers, max_active_workers, worker_calls
            with state_lock:
                worker_calls += 1
                call_number = worker_calls
                active_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
            try:
                if call_number == 1:
                    first_worker_started.set()
                    self.assertTrue(release_first_worker.wait(timeout=2.0))
                return True
            finally:
                with state_lock:
                    active_workers -= 1

        async def scenario() -> None:
            request = types.SimpleNamespace(
                app=types.SimpleNamespace(state=types.SimpleNamespace(local_gpu_lock=asyncio.Lock()))
            )
            with mock.patch.object(admin, "_load_model_for_settings", side_effect=blocking_model_load):
                first = asyncio.create_task(
                    endpoint(
                        admin.AdminSettingsPayload(local_gpu_device="cpu"),
                        request,
                        {"username": "admin"},
                    )
                )
                self.assertTrue(await asyncio.to_thread(first_worker_started.wait, 1.0))
                second = asyncio.create_task(
                    endpoint(
                        admin.AdminSettingsPayload(local_gpu_device="cuda"),
                        request,
                        {"username": "admin"},
                    )
                )
                await asyncio.sleep(0.02)
                self.assertEqual(worker_calls, 1)

                release_first_worker.set()
                first_response, second_response = await asyncio.gather(first, second)
                self.assertTrue(first_response["model_loaded"])
                self.assertTrue(second_response["model_loaded"])

            self.assertEqual(worker_calls, 2)
            self.assertEqual(max_active_workers, 1)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
