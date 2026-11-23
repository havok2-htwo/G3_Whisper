import asyncio
import json
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_storage as storage


LEGACY_DIA_KEYS = ("diarization_engine", "dia_server_base_url", "dia_api_key")


class AdminSettingsTests(unittest.TestCase):
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
        admin.current_settings.update(storage.normalize_settings(dict(storage.DEFAULT_SETTINGS)))
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

    def test_cuda_trim_setting_defaults_on_and_survives_partial_updates(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertTrue(storage.DEFAULT_SETTINGS["cuda_memory_trim_after_batch"])
        self.assertTrue(admin.current_settings["cuda_memory_trim_after_batch"])

        disabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cuda_memory_trim_after_batch=False),
                {"username": "admin"},
            )
        )
        self.assertFalse(disabled["settings"]["cuda_memory_trim_after_batch"])

        partial = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(batch_wait_time_ms=7),
                {"username": "admin"},
            )
        )
        self.assertFalse(partial["settings"]["cuda_memory_trim_after_batch"])

        enabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cuda_memory_trim_after_batch=True),
                {"username": "admin"},
            )
        )
        self.assertTrue(enabled["settings"]["cuda_memory_trim_after_batch"])
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertTrue(persisted["cuda_memory_trim_after_batch"])

    def test_cohere_cuda_graphs_setting_round_trips_without_model_reload(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertTrue(storage.DEFAULT_SETTINGS["cohere_cuda_graphs"])
        self.assertTrue(admin.current_settings["cohere_cuda_graphs"])

        disabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cohere_cuda_graphs=False),
                {"username": "admin"},
            )
        )
        self.assertFalse(disabled["settings"]["cohere_cuda_graphs"])
        # Takes effect on the next model (re)load; toggling alone never reloads the model.
        self.assertFalse(disabled["model_reloaded"])
        self.assertIsNone(disabled["model_loaded"])

        partial = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(batch_wait_time_ms=7),
                {"username": "admin"},
            )
        )
        self.assertFalse(partial["settings"]["cohere_cuda_graphs"])

        enabled = asyncio.run(
            endpoint(
                admin.AdminSettingsPayload(cohere_cuda_graphs=True),
                {"username": "admin"},
            )
        )
        self.assertTrue(enabled["settings"]["cohere_cuda_graphs"])
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertTrue(persisted["cohere_cuda_graphs"])

    def _request(self):
        return types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(local_gpu_lock=asyncio.Lock(), whisper_batch_manager=mock.Mock())
            )
        )

    def test_redimnet_max_batch_is_saved_normalised_and_applied_without_restart(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertEqual(admin.current_settings["redimnet_max_batch_size"], 8)

        with mock.patch.object(admin, "_apply_voice_model_settings", return_value={}) as apply:
            response = asyncio.run(
                endpoint(admin.AdminSettingsPayload(redimnet_max_batch_size=6), self._request(), {"username": "admin"})
            )
            self.assertEqual(response["settings"]["redimnet_max_batch_size"], 4)
            self.assertFalse(response["model_reloaded"])
            apply.assert_called_once_with()

            asyncio.run(endpoint(admin.AdminSettingsPayload(batch_wait_time_ms=7), self._request(), {"username": "admin"}))
            apply.assert_called_once_with()

        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["redimnet_max_batch_size"], 4)

    def test_language_detection_settings_apply_without_reloading_the_asr_model(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        self.assertTrue(admin.current_settings["cohere_language_detection"])
        self.assertEqual(admin.current_settings["language_detection_model"], "openai/whisper-base")

        with (
            mock.patch.object(admin, "_reload_model", new=mock.AsyncMock(return_value=True)) as reload,
            mock.patch.object(admin, "apply_lid_device_setting", return_value=True) as lid_apply,
            mock.patch.object(admin, "apply_vid_settings", return_value=False),
            mock.patch.object(admin, "apply_mms_device_setting", return_value=False),
        ):
            for payload, expected in (
                (admin.AdminSettingsPayload(language_detection_model="openai/whisper-small"), "openai/whisper-small"),
                # An unknown id is not an error, it falls back to the default detector.
                (admin.AdminSettingsPayload(language_detection_model="facebook/mms-lid-126"), "openai/whisper-base"),
            ):
                response = asyncio.run(endpoint(payload, self._request(), {"username": "admin"}))
                self.assertEqual(response["settings"]["language_detection_model"], expected)
                self.assertFalse(response["model_reloaded"])
            response = asyncio.run(
                endpoint(admin.AdminSettingsPayload(cohere_language_detection=False), self._request(), {"username": "admin"})
            )
        self.assertFalse(response["settings"]["cohere_language_detection"])
        reload.assert_not_awaited()
        self.assertEqual(lid_apply.call_count, 3)
        self.assertEqual(
            [option["value"] for option in response["options"]["language_detection_models"]],
            ["openai/whisper-base", "openai/whisper-small", "openai/whisper-large-v3-turbo"],
        )
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertFalse(persisted["cohere_language_detection"])

    def test_language_detection_switch_is_a_strict_boolean(self) -> None:
        self.assertTrue(storage.normalize_settings({"cohere_language_detection": "false"})["cohere_language_detection"])
        self.assertFalse(storage.normalize_settings({"cohere_language_detection": False})["cohere_language_detection"])
        with self.assertRaises(Exception):
            admin.AdminSettingsPayload(cohere_language_detection="false")

    def test_gpu_device_change_also_moves_the_voice_models(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        with (
            mock.patch.object(admin, "_load_model_for_settings", return_value=True),
            mock.patch.object(admin, "apply_vid_settings", return_value=True) as vid_apply,
            mock.patch.object(admin, "apply_mms_device_setting", return_value=False) as mms_apply,
        ):
            response = asyncio.run(
                endpoint(admin.AdminSettingsPayload(local_gpu_device="cuda:1"), self._request(), {"username": "admin"})
            )
        self.assertTrue(response["model_reloaded"])
        vid_apply.assert_called_once_with()
        mms_apply.assert_called_once_with()

    def test_voice_model_reprepare_failure_does_not_fail_the_save(self) -> None:
        endpoint = self._endpoint("/api/admin/settings", "PUT")
        with mock.patch.object(admin, "_apply_voice_model_settings", side_effect=RuntimeError("CUDA out of memory")):
            response = asyncio.run(
                endpoint(admin.AdminSettingsPayload(redimnet_max_batch_size=2), self._request(), {"username": "admin"})
            )
        self.assertTrue(response["ok"])
        self.assertEqual(admin.current_settings["redimnet_max_batch_size"], 2)

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

    # --- The removed G3_DIA integration: old clients and old settings files keep working -----------------------

    def test_legacy_dia_fields_in_a_put_are_accepted_and_never_saved(self) -> None:
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        try:
            with TestClient(self.app) as client:
                response = client.put(
                    "/api/admin/settings",
                    json={
                        "batch_wait_time_ms": 321,
                        "diarization_engine": "dia",
                        "dia_server_base_url": "http://saved-dia:7864",
                        "dia_api_key": "legacy-secret",
                    },
                )
        finally:
            self.app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200, response.text)
        # Counter-check: the rest of the same PUT is applied, so the request was not dropped as a whole.
        self.assertEqual(admin.current_settings["batch_wait_time_ms"], 321)
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["batch_wait_time_ms"], 321)
        for key in LEGACY_DIA_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, admin.current_settings)
                self.assertNotIn(key, persisted)
                self.assertNotIn(key, response.json()["settings"])
        self.assertNotIn("legacy-secret", response.text)
        self.assertNotIn("legacy-secret", self.settings_file.read_text(encoding="utf-8"))

    def test_settings_response_offers_no_dia_fields_or_engine_choice(self) -> None:
        response = asyncio.run(self._endpoint("/api/admin/settings", "GET")({"username": "admin"}))

        dia_fields = [key for key in response["settings"] if key.startswith("dia_") or key == "diarization_engine"]
        self.assertEqual(dia_fields, [])
        self.assertNotIn("diarization_engines", response["options"])
        # Counter-check: the diarization settings that stay are still served.
        self.assertIn("segmentation_model_path", response["settings"])
        self.assertIn("speaker_turns_enabled", response["settings"])

    def test_removed_dia_routes_answer_not_found(self) -> None:
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        try:
            with TestClient(self.app) as client:
                removed = [
                    client.post("/api/admin/dia/test", json={}),
                    client.delete("/api/admin/settings/dia-api-key"),
                ]
                kept = client.get("/api/admin/settings")
        finally:
            self.app.dependency_overrides.clear()

        for response in removed:
            self.assertIn(response.status_code, (404, 405), response.text)
        # Counter-check: the same client reaches a route that still exists.
        self.assertEqual(kept.status_code, 200, kept.text)

    def test_old_settings_file_with_dia_keys_loads_and_the_next_save_drops_them(self) -> None:
        legacy = {
            **{key: value for key, value in storage.DEFAULT_SETTINGS.items() if key != "model_profiles"},
            "batch_wait_time_ms": 432,
            "speaker_turns_enabled": False,
            "diarization_engine": "dia",
            "dia_server_base_url": "http://x:7864",
            "dia_api_key": "s",
        }
        self.settings_file.write_text(json.dumps(legacy), encoding="utf-8")

        loaded = storage.load_settings()

        # Counter-check: the file was really read (non-default values survive), not replaced by the defaults.
        self.assertEqual(loaded["batch_wait_time_ms"], 432)
        self.assertIs(loaded["speaker_turns_enabled"], False)
        for key in LEGACY_DIA_KEYS:
            self.assertNotIn(key, loaded)
        storage.save_settings(loaded)
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["batch_wait_time_ms"], 432)
        for key in LEGACY_DIA_KEYS:
            self.assertNotIn(key, persisted)

    def test_seed_file_ignores_dia_keys(self) -> None:
        seed_file = Path(self.temp_dir.name) / "settings.seed.json"
        seed_file.write_text(
            json.dumps({"batch_wait_time_ms": 77, "diarization_engine": "dia", "dia_server_base_url": "http://dia:7864"}),
            encoding="utf-8",
        )
        with mock.patch.object(storage, "SETTINGS_SEED_FILE", str(seed_file)):
            seed = storage.load_settings_seed()

        self.assertEqual(seed, {"batch_wait_time_ms": 77})

    def test_dia_environment_variables_are_ignored(self) -> None:
        baseline = storage.normalize_settings(dict(storage.DEFAULT_SETTINGS))
        with mock.patch.dict(
            os.environ,
            {"DIA_SERVER_BASE_URL": "http://dia:7864", "DIA_SERVER_API_KEY": "env-secret"},
        ):
            with_env = storage.normalize_settings(dict(storage.DEFAULT_SETTINGS))
            response = asyncio.run(self._endpoint("/api/admin/settings", "GET")({"username": "admin"}))

        self.assertEqual(with_env, baseline)
        self.assertNotIn("env-secret", json.dumps(response))
        self.assertNotIn("dia:7864", json.dumps(response))

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
                app=types.SimpleNamespace(
                    state=types.SimpleNamespace(local_gpu_lock=local_gpu_lock, whisper_batch_manager=mock.Mock())
                )
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
                app=types.SimpleNamespace(
                    state=types.SimpleNamespace(local_gpu_lock=asyncio.Lock(), whisper_batch_manager=mock.Mock())
                )
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
