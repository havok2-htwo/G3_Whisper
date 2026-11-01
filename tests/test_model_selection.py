"""Model selection: fields per backend, profiles per model, POST /api/admin/models/select, presets per model, the
active-model delete guard and GET /v2/capabilities.

Runs without a GPU: model loads are faked, settings and presets live in a temporary directory.
"""
import asyncio
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server as server
from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as api
from backend import genesis_whisper_server_presets as presets
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_batching import ModelFailoverError, WhisperBatchManager
from backend.genesis_whisper_server_globals import COHERE_MODEL_ID, LOCAL_ASR_MODEL_MAP, PARAKEET_MODEL_ID

WHISPER_TURBO = "openai/whisper-large-v3-turbo"
WHISPER_BASE = "openai/whisper-base"


def status_rows(ready=(COHERE_MODEL_ID, PARAKEET_MODEL_ID, WHISPER_TURBO), others="missing"):
    return [
        {"id": model_id, "backend": "x", "status": "ready" if model_id in ready else others}
        for model_id in LOCAL_ASR_MODEL_MAP.values()
    ]


class ModelFieldTests(unittest.TestCase):
    def test_every_field_is_a_real_setting(self) -> None:
        for backend, fields in storage.MODEL_SETTINGS_FIELDS.items():
            with self.subTest(backend=backend):
                self.assertLessEqual(set(fields), set(storage.DEFAULT_SETTINGS))
        self.assertLessEqual(set(storage.SERVICE_SETTINGS_FIELDS), set(storage.DEFAULT_SETTINGS))

    def test_a_setting_is_either_per_model_or_service_wide(self) -> None:
        per_model = {field for fields in storage.MODEL_SETTINGS_FIELDS.values() for field in fields}
        self.assertFalse(per_model & set(storage.SERVICE_SETTINGS_FIELDS))
        self.assertEqual(
            per_model | set(storage.SERVICE_SETTINGS_FIELDS) | {"local_model", "model_profiles"},
            set(storage.DEFAULT_SETTINGS),
        )

    def test_fields_match_what_each_backend_reads(self) -> None:
        fields = storage.MODEL_SETTINGS_FIELDS
        self.assertEqual(set(fields), {"whisper", "cohere_transcribe", "parakeet"})
        self.assertIn("cohere_cuda_graphs", fields["cohere_transcribe"])
        self.assertNotIn("cohere_cuda_graphs", fields["whisper"])
        for field in ("local_model_precision", "transcription_language", "cohere_cuda_graphs"):
            self.assertNotIn(field, fields["parakeet"])
        self.assertEqual(set(fields["whisper"]) - set(fields["parakeet"]), {"local_model_precision", "transcription_language"})

    def test_tuned_keys_are_the_preset_keys_without_model_and_device_plus_language(self) -> None:
        self.assertEqual(
            set(storage.MODEL_TUNED_KEYS),
            (set(presets.PRESET_KEYS) - {"local_model", "local_gpu_device"}) | {"transcription_language"},
        )


class ModelProfileStorageTests(unittest.TestCase):
    def test_settings_without_the_key_load_with_empty_profiles(self) -> None:
        self.assertEqual(storage.normalize_settings({"batch_max_segments": 4})["model_profiles"], {})

    def test_unknown_ids_and_foreign_keys_are_dropped_and_values_checked(self) -> None:
        normalized = storage.normalize_settings(
            {
                "model_profiles": {
                    "retired/model": {"batch_max_segments": 4},
                    WHISPER_TURBO: {
                        "batch_max_segments": 0,
                        "redimnet_max_batch_size": 6,
                        "transcription_language": "DE",
                        "local_model_precision": "nonsense",
                        "local_gpu_device": "cuda:1",
                        "huggingface_token": "hf_secret",
                    },
                    COHERE_MODEL_ID: "not a profile",
                    PARAKEET_MODEL_ID: {"unknown": 1},
                }
            }
        )
        self.assertEqual(
            normalized["model_profiles"],
            {
                WHISPER_TURBO: {
                    "batch_max_segments": 1,
                    "redimnet_max_batch_size": 4,
                    "transcription_language": "de",
                    "local_model_precision": "fp16",
                }
            },
        )

    def test_non_dict_profiles_are_ignored(self) -> None:
        for value in (None, "x", 3, ["a"]):
            with self.subTest(value=value):
                self.assertEqual(storage.normalize_settings({"model_profiles": value})["model_profiles"], {})

    def test_profiles_round_trip_and_the_other_keys_stay_top_level(self) -> None:
        with tempfile.TemporaryDirectory(prefix="genesis-whisper-profile-tests-") as temp_dir:
            settings_file = os.path.join(temp_dir, "settings.json")
            with mock.patch.object(storage, "LOGS_DIR", temp_dir), mock.patch.object(storage, "SETTINGS_FILE", settings_file):
                storage.save_settings(
                    {**storage.DEFAULT_SETTINGS, "model_profiles": {WHISPER_TURBO: {"batch_max_segments": 4}}}
                )
                self.assertEqual(storage.load_settings()["model_profiles"], {WHISPER_TURBO: {"batch_max_segments": 4}})
            with open(settings_file, "r", encoding="utf-8") as file_obj:
                raw = json.load(file_obj)
        for key in ("local_model", "local_model_cache_path", "local_gpu_device", "dia_server_base_url"):
            self.assertIn(key, raw)

    def test_with_model_profiles_later_sources_win_and_input_stays_untouched(self) -> None:
        settings = {"local_model": WHISPER_TURBO, "batch_max_segments": 2, "model_profiles": {}}
        stored = storage.with_model_profiles(settings, {"local_model": WHISPER_TURBO, "batch_max_segments": 1}, settings)
        self.assertEqual(stored["model_profiles"], {WHISPER_TURBO: {"batch_max_segments": 2}})
        self.assertEqual(settings["model_profiles"], {})


class _AdminCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-selection-tests-")
        self.addCleanup(self.temp_dir.cleanup)
        self.settings_file = os.path.join(self.temp_dir.name, "settings.json")
        self.original_settings = admin.current_settings.copy()
        self.statuses = status_rows()
        self.loaded = []
        self.failing = set()
        for patcher in (
            mock.patch.object(storage, "LOGS_DIR", self.temp_dir.name),
            mock.patch.object(storage, "SETTINGS_FILE", self.settings_file),
            mock.patch.object(presets, "PRESETS_FILE", os.path.join(self.temp_dir.name, "presets.json")),
            mock.patch.object(presets, "_cuda_cards_mib", return_value=[]),
            mock.patch.object(admin, "list_model_statuses", side_effect=lambda _path: self.statuses),
            mock.patch.object(admin, "_load_model_for_settings", side_effect=self.fake_load),
            mock.patch.object(admin, "get_last_local_asr_load_error", side_effect=lambda: self.load_error),
            mock.patch.object(admin, "_apply_voice_model_settings", return_value={}),
            mock.patch.dict(admin.local_model_components, {"model": None, "model_identifier": None}, clear=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.load_error = None

        self.use_settings(batch_max_segments=7, batch_max_audio_seconds=222.0, redimnet_max_batch_size=8)
        self.mark_loaded(COHERE_MODEL_ID)

        self.app = admin.create_admin_api(FastAPI())
        self.app.state.local_gpu_lock = asyncio.Lock()
        self.app.state.whisper_batch_manager = WhisperBatchManager(mock.Mock(), self.app.state.local_gpu_lock)
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        self.client = TestClient(self.app, raise_server_exceptions=True)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        admin.current_settings.clear()
        admin.current_settings.update(self.original_settings)

    def use_settings(self, **values) -> None:
        admin.current_settings.clear()
        admin.current_settings.update(
            storage.normalize_settings({**storage.DEFAULT_SETTINGS, "local_model_cache_path": "cache", **values})
        )

    def mark_loaded(self, model_id) -> None:
        admin.local_model_components.update({"model": object(), "model_identifier": (model_id, "auto", "cache", "fp16")})

    def fake_load(self, settings) -> bool:
        model_id = settings["local_model"]
        self.loaded.append(model_id)
        if model_id in self.failing:
            admin.local_model_components.update({"model": None, "model_identifier": None})
            self.load_error = f"OSError: cannot load {model_id}"
            return False
        self.load_error = None
        self.mark_loaded(model_id)
        return True

    def select(self, model_id):
        return self.client.post("/api/admin/models/select", json={"model_id": model_id})

    def saved(self) -> dict:
        with open(self.settings_file, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)


class SettingsResponseTests(_AdminCase):
    def assert_model_block(self, body: dict) -> None:
        self.assertEqual(
            body["model_fields"], {backend: list(fields) for backend, fields in storage.MODEL_SETTINGS_FIELDS.items()}
        )
        self.assertEqual(body["service_fields"], list(storage.SERVICE_SETTINGS_FIELDS))
        options = {item["value"]: item for item in body["options"]["models"]}
        self.assertEqual(options[COHERE_MODEL_ID]["backend"], "cohere_transcribe")
        self.assertEqual(options[PARAKEET_MODEL_ID]["backend"], "parakeet")
        self.assertEqual(options[WHISPER_TURBO]["backend"], "whisper")
        self.assertEqual([model for model, item in options.items() if item["active"]], [COHERE_MODEL_ID])
        self.assertEqual([model for model, item in options.items() if item["loaded"]], [COHERE_MODEL_ID])
        self.assertIn("label", options[COHERE_MODEL_ID])
        rows = {row["id"]: row for row in body["models"]}
        self.assertTrue(rows[COHERE_MODEL_ID]["active"] and rows[COHERE_MODEL_ID]["loaded"])
        self.assertFalse(rows[WHISPER_TURBO]["active"] or rows[WHISPER_TURBO]["loaded"])
        self.assertIn("backend", rows[WHISPER_TURBO])

    def test_get_settings_carries_the_model_block(self) -> None:
        response = self.client.get("/api/admin/settings")
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_model_block(response.json())

    def test_put_settings_carries_the_model_block(self) -> None:
        response = self.client.put("/api/admin/settings", json={"batch_wait_time_ms": 7})
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_model_block(response.json())

    def test_models_route_rows_carry_active_and_loaded(self) -> None:
        rows = {row["id"]: row for row in self.client.get("/api/admin/models").json()["models"]}
        self.assertTrue(rows[COHERE_MODEL_ID]["active"])
        self.assertFalse(rows[PARAKEET_MODEL_ID]["active"])

    def test_loaded_is_false_while_another_model_or_nothing_is_resident(self) -> None:
        self.mark_loaded(WHISPER_TURBO)
        options = self.client.get("/api/admin/settings").json()["options"]["models"]
        self.assertEqual({item["value"]: item["loaded"] for item in options}[COHERE_MODEL_ID], False)
        admin.local_model_components.update({"model": None, "model_identifier": None})
        options = self.client.get("/api/admin/settings").json()["options"]["models"]
        self.assertFalse(any(item["loaded"] for item in options))


class ProfileOnPutTests(_AdminCase):
    def test_put_stores_tuned_values_under_the_current_model(self) -> None:
        response = self.client.put("/api/admin/settings", json={"batch_max_segments": 9, "transcription_language": "fr"})
        self.assertEqual(response.status_code, 200, response.text)
        profile = self.saved()["model_profiles"][COHERE_MODEL_ID]
        self.assertEqual(set(profile), set(storage.MODEL_TUNED_KEYS))
        self.assertEqual((profile["batch_max_segments"], profile["transcription_language"]), (9, "fr"))
        self.assertEqual(self.saved()["batch_max_segments"], 9)
        self.assertEqual(admin.current_settings["model_profiles"], self.saved()["model_profiles"])

    def test_put_stores_normalised_values(self) -> None:
        self.client.put("/api/admin/settings", json={"redimnet_max_batch_size": 6, "batch_max_segments": 0})
        profile = self.saved()["model_profiles"][COHERE_MODEL_ID]
        self.assertEqual((profile["redimnet_max_batch_size"], profile["batch_max_segments"]), (4, 1))

    def test_a_rejected_put_stores_nothing(self) -> None:
        response = self.client.put("/api/admin/settings", json={"batch_max_segments": 9, "dia_server_base_url": "dia:7864"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(os.path.exists(self.settings_file))

    def test_clients_cannot_write_profiles_through_put(self) -> None:
        self.client.put(
            "/api/admin/settings",
            json={"batch_max_segments": 9, "model_profiles": {WHISPER_TURBO: {"batch_max_segments": 1}}},
        )
        self.assertNotIn(WHISPER_TURBO, self.saved()["model_profiles"])

    def test_put_with_a_changed_model_keeps_working_and_applies_no_profile(self) -> None:
        admin.current_settings["model_profiles"] = {WHISPER_TURBO: {"batch_max_segments": 3}}
        response = self.client.put("/api/admin/settings", json={"local_model": WHISPER_TURBO})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["model_reloaded"] and body["model_loaded"])
        self.assertIsNone(body["load_error"])
        self.assertEqual(self.loaded, [WHISPER_TURBO])
        saved = self.saved()
        self.assertEqual((saved["local_model"], saved["batch_max_segments"]), (WHISPER_TURBO, 7))
        self.assertEqual(saved["model_profiles"][WHISPER_TURBO]["batch_max_segments"], 7)
        self.assertEqual(saved["model_profiles"][COHERE_MODEL_ID]["batch_max_segments"], 7)

    def test_put_reports_the_load_error_and_keeps_its_saved_selection(self) -> None:
        self.failing.add(WHISPER_TURBO)
        response = self.client.put("/api/admin/settings", json={"local_model": WHISPER_TURBO})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["model_loaded"])
        self.assertEqual(body["load_error"], f"OSError: cannot load {WHISPER_TURBO}")
        self.assertEqual(self.saved()["local_model"], WHISPER_TURBO)

    def test_load_error_is_null_when_nothing_was_loaded(self) -> None:
        self.load_error = "an old failure"
        body = self.client.put("/api/admin/settings", json={"batch_wait_time_ms": 7}).json()
        self.assertIsNone(body["model_loaded"])
        self.assertIsNone(body["load_error"])


class SelectModelTests(_AdminCase):
    def test_unknown_id_is_400(self) -> None:
        self.assertEqual(self.select("nobody/nothing").status_code, 400)
        self.assertEqual(self.select("whisper-1").status_code, 400)
        self.assertEqual(self.loaded, [])

    def test_a_model_that_is_not_ready_is_409_and_never_loaded(self) -> None:
        for status in ("missing", "partial", "downloading", "error"):
            with self.subTest(status=status):
                self.statuses = status_rows(ready=(COHERE_MODEL_ID,), others=status)
                response = self.select(WHISPER_BASE)
                self.assertEqual(response.status_code, 409)
                self.assertIn(status, response.json()["detail"])
        self.assertEqual(self.loaded, [])
        self.assertEqual(admin.current_settings["local_model"], COHERE_MODEL_ID)
        self.assertFalse(os.path.exists(self.settings_file))

    def test_the_same_model_is_a_no_op(self) -> None:
        self.statuses = status_rows(ready=())
        response = self.select(COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["ok"] and body["model_loaded"])
        self.assertFalse(body["model_reloaded"])
        self.assertEqual(self.loaded, [])
        self.assertFalse(os.path.exists(self.settings_file))

    def test_the_active_model_loads_when_it_is_not_loaded(self) -> None:
        admin.local_model_components.update({"model": None, "model_identifier": None})
        self.load_error = "OSError: disk"
        response = self.select(COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["ok"] and body["model_reloaded"] and body["model_loaded"])
        self.assertIsNone(body["load_error"])
        self.assertEqual(self.loaded, [COHERE_MODEL_ID])
        self.assertEqual(admin.local_model_components["model_identifier"][0], COHERE_MODEL_ID)
        self.assertFalse(os.path.exists(self.settings_file))

    def test_a_retry_that_fails_reports_the_real_error_and_keeps_the_selection(self) -> None:
        admin.local_model_components.update({"model": None, "model_identifier": None})
        before = admin.current_settings.copy()
        self.failing.add(COHERE_MODEL_ID)
        response = self.select(COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 500)
        self.assertIn(f"OSError: cannot load {COHERE_MODEL_ID}", response.json()["detail"])
        self.assertEqual(self.loaded, [COHERE_MODEL_ID])
        self.assertEqual(admin.current_settings, before)
        self.assertFalse(os.path.exists(self.settings_file))

        self.failing.clear()
        self.assertTrue(self.select(COHERE_MODEL_ID).json()["model_loaded"])

    def test_a_retry_of_a_model_that_is_gone_is_409_and_never_loaded(self) -> None:
        admin.local_model_components.update({"model": None, "model_identifier": None})
        self.statuses = status_rows(ready=())
        self.assertEqual(self.select(COHERE_MODEL_ID).status_code, 409)
        self.assertEqual(self.loaded, [])

    def test_success_returns_the_put_shape(self) -> None:
        response = self.select(WHISPER_TURBO)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        put_shape = self.client.put("/api/admin/settings", json={"batch_wait_time_ms": 7}).json()
        self.assertEqual(set(body), set(put_shape))
        self.assertTrue(body["ok"] and body["model_reloaded"] and body["model_loaded"])
        self.assertIsNone(body["load_error"])
        self.assertEqual(body["settings"]["local_model"], WHISPER_TURBO)
        self.assertEqual(self.loaded, [WHISPER_TURBO])

    def test_old_values_go_to_its_profile_and_the_targets_profile_applies(self) -> None:
        self.use_settings(
            batch_max_segments=7,
            model_profiles={WHISPER_TURBO: {"batch_max_segments": 3, "transcription_language": "fr"}},
        )
        self.select(WHISPER_TURBO)
        saved = self.saved()
        self.assertEqual(saved["local_model"], WHISPER_TURBO)
        self.assertEqual((saved["batch_max_segments"], saved["transcription_language"]), (3, "fr"))
        self.assertEqual(saved["model_profiles"][COHERE_MODEL_ID]["batch_max_segments"], 7)

        self.select(COHERE_MODEL_ID)
        saved = self.saved()
        self.assertEqual((saved["batch_max_segments"], saved["transcription_language"]), (7, "auto"))

    def test_no_profile_on_cohere_uses_the_24gb_tier(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO, batch_max_segments=2, redimnet_max_batch_size=8)
        self.select(COHERE_MODEL_ID)
        saved = self.saved()
        tier = presets.BUILTIN_PRESETS["24GB VRAM"]["values"]
        for key, value in tier.items():
            self.assertEqual(saved[key], value, key)

    def test_no_profile_on_parakeet_uses_its_tier_and_leaves_the_device(self) -> None:
        self.select(PARAKEET_MODEL_ID)
        saved = self.saved()
        self.assertEqual(saved["local_model"], PARAKEET_MODEL_ID)
        self.assertEqual(
            (saved["batch_max_segments"], saved["batch_max_audio_seconds"], saved["redimnet_max_batch_size"]),
            (4, 100.0, 2),
        )
        self.assertEqual(saved["local_gpu_device"], "auto")

    def test_no_profile_on_a_whisper_model_keeps_current_values(self) -> None:
        self.select(WHISPER_TURBO)
        saved = self.saved()
        self.assertEqual((saved["batch_max_segments"], saved["batch_max_audio_seconds"]), (7, 222.0))

    def test_cohere_auto_becomes_german_on_the_next_model(self) -> None:
        self.assertEqual(admin.current_settings["transcription_language"], "auto")
        self.select(WHISPER_TURBO)
        saved = self.saved()
        self.assertEqual(saved["transcription_language"], "de")
        self.assertEqual(saved["model_profiles"][COHERE_MODEL_ID]["transcription_language"], "auto")

    def test_an_explicit_language_is_inherited(self) -> None:
        self.use_settings(transcription_language="fr")
        self.select(PARAKEET_MODEL_ID)
        self.assertEqual(self.saved()["transcription_language"], "fr")

    def test_german_survives_a_chain_through_parakeet(self) -> None:
        self.select(PARAKEET_MODEL_ID)
        self.select(WHISPER_TURBO)
        self.assertEqual(self.saved()["transcription_language"], "de")

    def test_whisper_auto_stays_auto_on_a_model_without_profile(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO)
        self.select(PARAKEET_MODEL_ID)
        self.assertEqual(self.saved()["transcription_language"], "auto")

    def test_a_profile_without_language_inherits_it(self) -> None:
        self.use_settings(transcription_language="fr", model_profiles={WHISPER_TURBO: {"batch_max_segments": 3}})
        self.select(WHISPER_TURBO)
        self.assertEqual(self.saved()["transcription_language"], "fr")

    def test_service_wide_values_and_the_device_never_move(self) -> None:
        self.use_settings(local_gpu_device="cuda:1", huggingface_token="hf_x", dia_server_base_url="http://dia:7864")
        self.select(PARAKEET_MODEL_ID)
        saved = self.saved()
        self.assertEqual(
            (saved["local_gpu_device"], saved["huggingface_token"], saved["dia_server_base_url"], saved["local_model_cache_path"]),
            ("cuda:1", "hf_x", "http://dia:7864", "cache"),
        )

    def test_voice_models_are_prepared_when_the_redimnet_batch_changes(self) -> None:
        self.select(WHISPER_TURBO)
        admin._apply_voice_model_settings.assert_not_called()
        self.select(PARAKEET_MODEL_ID)
        admin._apply_voice_model_settings.assert_called_once_with()

    def test_failed_load_restores_selection_and_old_model_and_reports_the_real_error(self) -> None:
        self.use_settings(model_profiles={WHISPER_TURBO: {"batch_max_segments": 3}}, batch_max_segments=7)
        before = admin.current_settings.copy()
        self.failing.add(WHISPER_TURBO)
        response = self.select(WHISPER_TURBO)
        self.assertEqual(response.status_code, 500)
        self.assertIn(f"OSError: cannot load {WHISPER_TURBO}", response.json()["detail"])
        self.assertEqual(self.loaded, [WHISPER_TURBO, COHERE_MODEL_ID])
        self.assertEqual(self.saved()["local_model"], COHERE_MODEL_ID)
        self.assertEqual(admin.current_settings, before)
        self.assertEqual(self.saved()["batch_max_segments"], 7)
        self.assertEqual(admin.local_model_components["model_identifier"][0], COHERE_MODEL_ID)
        admin._apply_voice_model_settings.assert_not_called()

    def test_a_load_that_raises_is_handled_like_a_failed_one(self) -> None:
        def raising_load(settings):
            self.loaded.append(settings["local_model"])
            if settings["local_model"] == WHISPER_TURBO:
                raise TimeoutError("GPU lease")
            return True

        admin._load_model_for_settings.side_effect = raising_load
        response = self.select(WHISPER_TURBO)
        self.assertEqual(response.status_code, 500)
        self.assertIn("TimeoutError: GPU lease", response.json()["detail"])
        self.assertEqual(self.saved()["local_model"], COHERE_MODEL_ID)

    def test_when_the_old_model_fails_too_both_errors_are_reported(self) -> None:
        self.failing.update({WHISPER_TURBO, COHERE_MODEL_ID})
        response = self.select(WHISPER_TURBO)
        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        self.assertIn(f"cannot load {WHISPER_TURBO}", detail)
        self.assertIn(f"cannot load {COHERE_MODEL_ID}", detail)
        self.assertEqual(self.saved()["local_model"], COHERE_MODEL_ID)

    def test_rollback_keeps_a_token_saved_while_the_load_ran(self) -> None:
        def load(settings):
            if settings["local_model"] == WHISPER_TURBO:
                admin.current_settings["huggingface_token"] = "hf_new"
            return self.fake_load(settings)

        admin._load_model_for_settings.side_effect = load
        self.failing.add(WHISPER_TURBO)
        self.select(WHISPER_TURBO)
        self.assertEqual(self.saved()["huggingface_token"], "hf_new")

    def _during_load(self, scenario) -> None:
        """Run scenario(task) on one event loop while the selection of Whisper Turbo sits inside its model load."""
        started = threading.Event()
        release = threading.Event()

        def blocking_load(settings):
            started.set()
            self.assertTrue(release.wait(timeout=5.0))
            return self.fake_load(settings)

        admin._load_model_for_settings.side_effect = blocking_load
        endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/select"
        )

        async def run() -> None:
            self.app.state.local_gpu_lock = asyncio.Lock()
            request = mock.Mock(app=self.app)
            task = asyncio.create_task(endpoint(admin.AdminModelSelectPayload(model_id=WHISPER_TURBO), request, {}))
            self.assertTrue(await asyncio.to_thread(started.wait, 2.0))
            try:
                await scenario(task, release)
            finally:
                release.set()

        asyncio.run(run())

    def test_the_load_runs_under_the_gpu_lock(self) -> None:
        async def scenario(task, release) -> None:
            self.assertTrue(self.app.state.local_gpu_lock.locked())
            release.set()
            self.assertTrue((await task)["model_loaded"])
            self.assertFalse(self.app.state.local_gpu_lock.locked())

        self._during_load(scenario)

    def test_second_selection_and_settings_put_are_refused_while_one_runs(self) -> None:
        select_endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/select"
        )
        put_endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/settings" and "PUT" in route.methods
        )

        async def scenario(task, release) -> None:
            with self.assertRaises(admin.HTTPException) as selected:
                await select_endpoint(admin.AdminModelSelectPayload(model_id=PARAKEET_MODEL_ID), mock.Mock(), {})
            self.assertEqual(selected.exception.status_code, 409)
            with self.assertRaises(admin.HTTPException) as put:
                await put_endpoint(admin.AdminSettingsPayload(batch_wait_time_ms=1), mock.Mock(), {})
            self.assertEqual(put.exception.status_code, 409)
            release.set()
            await task

        self._during_load(scenario)
        self.assertEqual(self.saved()["local_model"], WHISPER_TURBO)

    def test_a_running_selection_keeps_its_own_409_for_a_form_of_the_old_model(self) -> None:
        # The UI reloads the form only on active_model_changed. Mid-switch the outcome is open, a failed load rolls back.
        put_endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/settings" and "PUT" in route.methods
        )

        async def scenario(task, release) -> None:
            with self.assertRaises(admin.HTTPException) as put:
                await put_endpoint(
                    admin.AdminSettingsPayload(batch_wait_time_ms=1, expected_local_model=COHERE_MODEL_ID),
                    mock.Mock(),
                    {},
                )
            self.assertEqual(put.exception.status_code, 409)
            self.assertNotEqual(put.exception.detail, admin.ACTIVE_MODEL_CHANGED)
            release.set()
            await task

        self._during_load(scenario)

    def test_a_cancelled_failing_selection_does_not_stay_saved(self) -> None:
        self.failing.add(WHISPER_TURBO)

        async def scenario(task, release) -> None:
            task.cancel()
            await asyncio.sleep(0.02)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self._during_load(scenario)
        self.assertEqual(self.saved()["local_model"], COHERE_MODEL_ID)

    def test_a_cancelled_selection_that_loaded_stays_selected(self) -> None:
        async def scenario(task, release) -> None:
            task.cancel()
            await asyncio.sleep(0.02)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self._during_load(scenario)
        self.assertEqual(self.saved()["local_model"], WHISPER_TURBO)

    def test_route_requires_admin(self) -> None:
        self.app.dependency_overrides.clear()
        self.assertIn(self.select(WHISPER_TURBO).status_code, (401, 403))
        self.assertEqual(self.loaded, [])

    def test_delete_is_refused_while_a_selection_runs(self) -> None:
        delete_endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/delete"
        )

        async def scenario(task, release) -> None:
            with mock.patch.object(admin, "delete_model_cache") as delete_cache:
                for model_id in (WHISPER_TURBO, PARAKEET_MODEL_ID):
                    with self.assertRaises(admin.HTTPException) as deleted:
                        await delete_endpoint(admin.AdminModelActionPayload(model_id=model_id), {})
                    self.assertEqual(deleted.exception.status_code, 409)
            delete_cache.assert_not_called()
            release.set()
            await task

        self._during_load(scenario)


class FailedSelectionQueueTests(_AdminCase):
    """Jobs accepted while a selection fails: a real scheduler on one loop with the select route."""

    def setUp(self) -> None:
        super().setUp()
        self.failing.add(WHISPER_TURBO)
        self.use_settings(batch_wait_time_ms=0, transcription_language="fr", batch_max_segments=7)
        self.served = []

    def process(self, audio_batch, processing_key) -> list:
        self.served.append(processing_key)
        if not self.fake_load({"local_model": processing_key[0]}):
            raise RuntimeError(self.load_error)
        return ["text"] * len(audio_batch)

    def run_failing_selection(self, scenario, target=WHISPER_TURBO, seconds=1.0) -> None:
        """scenario(manager, audio, key) runs during the load of target, the function it returns after the 500."""
        started = threading.Event()
        release = threading.Event()
        self.failing.add(target)

        def blocking_load(settings):
            started.set()
            self.assertTrue(release.wait(timeout=5.0))
            return self.fake_load(settings)

        admin._load_model_for_settings.side_effect = blocking_load
        endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/select"
        )

        async def run() -> None:
            self.app.state.local_gpu_lock = asyncio.Lock()
            manager = WhisperBatchManager(self.process, self.app.state.local_gpu_lock)
            self.app.state.whisper_batch_manager = manager
            await manager.start()
            try:
                request = mock.Mock(app=self.app)
                select = asyncio.create_task(endpoint(admin.AdminModelSelectPayload(model_id=target), request, {}))
                self.assertTrue(await asyncio.to_thread(started.wait, 2.0))
                audio = [np.zeros(int(16000 * seconds), dtype=np.float32)]
                after_failure = await scenario(manager, audio, admin._get_local_processing_key())
                release.set()
                with self.assertRaises(admin.HTTPException) as failed:
                    await select
                self.assertEqual(failed.exception.status_code, 500)
                await after_failure()
            finally:
                release.set()
                await manager.stop()

        asyncio.run(run())

    def test_a_queued_job_runs_on_the_restored_model_and_never_repeats_the_failed_load(self) -> None:
        async def scenario(manager, audio, key):
            self.assertEqual(key[0], WHISPER_TURBO)
            job = asyncio.create_task(manager.submit_job(audio, "queued", key))
            # The worker reserves the batch and waits behind the load for the GPU lock.
            await asyncio.sleep(0.05)
            self.assertEqual(len(manager._jobs), 1)
            self.assertEqual(next(iter(manager._jobs.values())).batch_count, 1)

            async def after_failure() -> None:
                results = await job
                self.assertEqual([result.text for result in results], ["text"])
                self.assertEqual(results[0].model_id, COHERE_MODEL_ID)

            return after_failure

        self.run_failing_selection(scenario)
        self.assertEqual(self.loaded, [WHISPER_TURBO, COHERE_MODEL_ID, COHERE_MODEL_ID])
        self.assertEqual(self.served, [(COHERE_MODEL_ID, "auto", "cache", "fr", "fp16")])
        self.assertEqual(admin.local_model_components["model_identifier"][0], COHERE_MODEL_ID)

    def test_a_job_that_prepared_its_chunks_past_the_rollback_moves_too(self) -> None:
        async def scenario(manager, audio, key):
            async def after_failure() -> None:
                await manager.submit_job(audio, "late", key)

            return after_failure

        self.run_failing_selection(scenario)
        self.assertEqual([key[0] for key in self.served], [COHERE_MODEL_ID])

    def test_a_cancelled_failing_selection_moves_the_jobs_too(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_load(settings):
            started.set()
            self.assertTrue(release.wait(timeout=5.0))
            return self.fake_load(settings)

        admin._load_model_for_settings.side_effect = blocking_load
        endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/select"
        )

        async def run() -> None:
            self.app.state.local_gpu_lock = asyncio.Lock()
            manager = WhisperBatchManager(self.process, self.app.state.local_gpu_lock)
            self.app.state.whisper_batch_manager = manager
            await manager.start()
            try:
                request = mock.Mock(app=self.app)
                select = asyncio.create_task(endpoint(admin.AdminModelSelectPayload(model_id=WHISPER_TURBO), request, {}))
                self.assertTrue(await asyncio.to_thread(started.wait, 2.0))
                job = asyncio.create_task(
                    manager.submit_job([np.zeros(16000, dtype=np.float32)], "queued", admin._get_local_processing_key())
                )
                await asyncio.sleep(0)
                select.cancel()
                await asyncio.sleep(0.02)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await select
                await job
            finally:
                release.set()
                await manager.stop()

        asyncio.run(run())
        self.assertEqual(admin.current_settings["local_model"], COHERE_MODEL_ID)
        self.assertEqual([key[0] for key in self.served], [COHERE_MODEL_ID])

    def test_a_per_request_language_survives_and_the_targets_language_does_not(self) -> None:
        self.use_settings(
            batch_wait_time_ms=0,
            transcription_language="fr",
            model_profiles={WHISPER_TURBO: {"transcription_language": "it"}},
        )

        async def scenario(manager, audio, key):
            self.assertEqual(key[3], "it")
            default = asyncio.create_task(manager.submit_job(audio, "default", key))
            override = asyncio.create_task(manager.submit_job(audio, "override", (*key[:3], "en", key[4])))

            async def after_failure() -> None:
                await asyncio.gather(default, override)

            return after_failure

        self.run_failing_selection(scenario)
        self.assertEqual({key[3] for key in self.served}, {"fr", "en"})

    def test_a_chunk_whisper_would_cut_fails_retryably_instead_of_losing_its_tail(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO, batch_wait_time_ms=0)
        self.mark_loaded(WHISPER_TURBO)
        self.failing.discard(WHISPER_TURBO)

        async def scenario(manager, audio, key):
            self.assertEqual(key[0], COHERE_MODEL_ID)
            queued = asyncio.create_task(manager.submit_job(audio, "queued", key))
            await asyncio.sleep(0.05)

            async def after_failure() -> None:
                with self.assertRaises(ModelFailoverError):
                    await queued
                with self.assertRaises(ModelFailoverError):
                    await manager.submit_job(audio, "late", key)

            return after_failure

        self.run_failing_selection(scenario, target=COHERE_MODEL_ID, seconds=31.0)
        self.assertEqual(self.served, [])
        self.assertEqual(self.loaded, [COHERE_MODEL_ID, WHISPER_TURBO])

    def test_a_chunk_inside_the_whisper_window_moves_to_whisper(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO, batch_wait_time_ms=0)
        self.mark_loaded(WHISPER_TURBO)
        self.failing.discard(WHISPER_TURBO)

        async def scenario(manager, audio, key):
            job = asyncio.create_task(manager.submit_job(audio, "queued", key))

            async def after_failure() -> None:
                await job

            return after_failure

        self.run_failing_selection(scenario, target=COHERE_MODEL_ID, seconds=30.0)
        self.assertEqual([key[0] for key in self.served], [WHISPER_TURBO])

    def test_jobs_of_other_models_keep_their_key(self) -> None:
        other_key = (PARAKEET_MODEL_ID, "auto", "cache", "de", "fp16")

        async def scenario(manager, audio, key):
            job = asyncio.create_task(manager.submit_job(audio, "other", other_key))

            async def after_failure() -> None:
                await job

            return after_failure

        self.run_failing_selection(scenario)
        self.assertEqual(self.served, [other_key])

    def test_the_failed_model_works_again_once_it_is_selected(self) -> None:
        async def scenario(manager, audio, key):
            async def after_failure() -> None:
                self.failing.clear()
                admin._load_model_for_settings.side_effect = self.fake_load
                self.use_settings(batch_wait_time_ms=0, local_model=WHISPER_TURBO)
                await manager.submit_job(audio, "next", admin._get_local_processing_key())

            return after_failure

        self.run_failing_selection(scenario)
        self.assertEqual([key[0] for key in self.served], [WHISPER_TURBO])

    def whisper_active(self, **values) -> None:
        self.failing.clear()
        self.use_settings(local_model=WHISPER_TURBO, batch_wait_time_ms=0, **values)
        self.mark_loaded(WHISPER_TURBO)

    def run_on_scheduler(self, scenario, process=None) -> None:
        """scenario(manager, select) with a real scheduler, select(model_id) awaits the real route."""
        endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/api/admin/models/select"
        )

        async def run() -> None:
            self.app.state.local_gpu_lock = asyncio.Lock()
            manager = WhisperBatchManager(process or self.process, self.app.state.local_gpu_lock)
            self.app.state.whisper_batch_manager = manager
            await manager.start()
            request = mock.Mock(app=self.app)

            async def select(model_id):
                return await endpoint(admin.AdminModelSelectPayload(model_id=model_id), request, {})

            try:
                await scenario(manager, select)
            finally:
                await manager.stop()

        asyncio.run(run())

    async def fail_then_select(self, select, model_id) -> None:
        self.failing.add(model_id)
        with self.assertRaises(admin.HTTPException) as failed:
            await select(model_id)
        self.assertEqual(failed.exception.status_code, 500)
        self.failing.clear()
        await select(model_id)

    def test_a_model_that_loaded_again_keeps_late_jobs_after_the_next_switch(self) -> None:
        self.whisper_active()

        async def scenario(manager, select) -> None:
            await self.fail_then_select(select, COHERE_MODEL_ID)
            # This request read the Cohere key and still prepares a chunk Whisper would cut.
            key = admin._get_local_processing_key()
            await select(WHISPER_TURBO)
            results = await manager.submit_job([np.zeros(16000 * 31, dtype=np.float32)], "late", key)
            self.assertEqual(results[0].model_id, COHERE_MODEL_ID)

        self.run_on_scheduler(scenario)
        self.assertEqual([key[0] for key in self.served], [COHERE_MODEL_ID])

    def test_a_failed_selection_leaves_a_queue_of_another_model_alone(self) -> None:
        profile = {"batch_max_segments": 1, "batch_wait_time_ms": 0}
        self.whisper_active(
            batch_max_segments=1,
            model_profiles={model: profile for model in (COHERE_MODEL_ID, PARAKEET_MODEL_ID, WHISPER_TURBO)},
        )
        gates = [threading.Event(), threading.Event()]

        def process(audio_batch, processing_key) -> list:
            number = len(self.served)
            results = self.process(audio_batch, processing_key)
            if number < len(gates):
                self.assertTrue(gates[number].wait(timeout=5.0))
            return results

        async def wait_for_served(count) -> None:
            while len(self.served) < count:
                await asyncio.sleep(0.01)

        async def scenario(manager, select) -> None:
            await self.fail_then_select(select, COHERE_MODEL_ID)
            chunks = [np.zeros(16000, dtype=np.float32) for _ in range(4)]
            job = asyncio.create_task(manager.submit_job(chunks, "old-queue", admin._get_local_processing_key()))
            await wait_for_served(1)
            switch = asyncio.create_task(select(WHISPER_TURBO))
            await asyncio.sleep(0.05)
            gates[0].set()
            await switch
            await wait_for_served(2)

            self.failing.add(PARAKEET_MODEL_ID)
            failing = asyncio.create_task(select(PARAKEET_MODEL_ID))
            await asyncio.sleep(0.05)
            gates[1].set()
            with self.assertRaises(admin.HTTPException):
                await failing
            results = await job
            self.assertEqual([result.model_id for result in results], [COHERE_MODEL_ID] * 4)

        try:
            self.run_on_scheduler(scenario, process)
        finally:
            for gate in gates:
                gate.set()
        self.assertEqual([key[0] for key in self.served], [COHERE_MODEL_ID] * 4)

    def test_a_failover_only_runs_the_retarget_of_the_model_that_failed(self) -> None:
        async def scenario(manager, select) -> None:
            still_selected = True

            def older_retarget(key, segments):
                return key if still_selected else ("moved", *key[1:])

            def unrelated_retarget(key, segments):
                self.fail(f"retarget of another model ran for {key}")

            key = admin._get_local_processing_key()
            manager.fail_over_model(key[0], older_retarget)
            async with self.app.state.local_gpu_lock:
                job = asyncio.create_task(manager.submit_job([np.zeros(16000, dtype=np.float32)], "queued", key))
                await asyncio.sleep(0.01)
                still_selected = False
                manager.fail_over_model("another/model", unrelated_retarget)
                self.assertEqual([queued.processing_key for queued in manager._jobs.values()], [key])
            await job

        self.whisper_active()
        self.run_on_scheduler(scenario)
        self.assertEqual([key[0] for key in self.served], [WHISPER_TURBO])

    def test_a_retarget_goes_when_the_model_loads_again(self) -> None:
        manager = self.app.state.whisper_batch_manager

        self.assertEqual(self.select(WHISPER_TURBO).status_code, 500)
        self.assertIn(WHISPER_TURBO, manager._failovers)
        self.failing.clear()
        self.assertEqual(self.select(WHISPER_TURBO).status_code, 200)
        self.assertEqual(manager._failovers, {})

        self.assertEqual(self.select(COHERE_MODEL_ID).status_code, 200)
        self.failing.add(WHISPER_TURBO)
        self.assertEqual(self.select(WHISPER_TURBO).status_code, 500)
        self.assertIn(WHISPER_TURBO, manager._failovers)
        self.failing.clear()
        self.assertEqual(self.client.put("/api/admin/settings", json={"local_model": WHISPER_TURBO}).status_code, 200)
        self.assertEqual(manager._failovers, {})

    def test_a_save_whose_load_failed_keeps_the_retarget(self) -> None:
        manager = self.app.state.whisper_batch_manager

        self.assertEqual(self.select(WHISPER_TURBO).status_code, 500)
        body = self.client.put("/api/admin/settings", json={"local_model": WHISPER_TURBO}).json()
        self.assertIs(body["model_loaded"], False)
        self.assertIn(WHISPER_TURBO, manager._failovers)

    def test_a_retry_of_the_active_model_clears_its_retarget(self) -> None:
        manager = self.app.state.whisper_batch_manager

        self.assertEqual(self.select(WHISPER_TURBO).status_code, 500)
        self.client.put("/api/admin/settings", json={"local_model": WHISPER_TURBO})
        self.failing.clear()
        self.assertEqual(self.select(WHISPER_TURBO).status_code, 200)
        self.assertEqual(manager._failovers, {})

    def test_a_batch_on_the_failed_model_clears_its_retarget(self) -> None:
        self.whisper_active()

        async def scenario(manager, audio, key):
            async def after_failure() -> None:
                # A PUT left this model selected unloaded, the worker's own load brought it back.
                self.failing.clear()
                admin._load_model_for_settings.side_effect = self.fake_load
                self.use_settings(batch_wait_time_ms=0, local_model=COHERE_MODEL_ID)
                await manager.submit_job(audio, "lazy", key)
                self.use_settings(batch_wait_time_ms=0, local_model=WHISPER_TURBO)
                results = await manager.submit_job(audio, "late", key)
                self.assertEqual(results[0].model_id, COHERE_MODEL_ID)

            return after_failure

        self.run_failing_selection(scenario, target=COHERE_MODEL_ID, seconds=31.0)
        self.assertEqual([key[0] for key in self.served], [COHERE_MODEL_ID, COHERE_MODEL_ID])


class StaleFormTests(_AdminCase):
    """A form opened before a switch holds the old model's values, expected_local_model keeps them off the new one."""

    def setUp(self) -> None:
        super().setUp()
        self.use_settings(transcription_language="auto")

    def open_form(self) -> dict:
        settings = self.client.get("/api/admin/settings").json()["settings"]
        return {key: settings[key] for key in ("transcription_language", "batch_max_segments", "batch_wait_time_ms")}

    def put(self, form: dict, **fields):
        return self.client.put("/api/admin/settings", json={**form, **fields})

    def test_a_form_of_the_old_model_cannot_move_clients_off_their_language(self) -> None:
        form = self.open_form()
        self.assertEqual(self.select(WHISPER_TURBO).status_code, 200)
        before = self.saved()
        self.assertEqual(before["transcription_language"], "de")
        self.loaded.clear()

        response = self.put(form, expected_local_model=COHERE_MODEL_ID)

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], admin.ACTIVE_MODEL_CHANGED)
        self.assertEqual(self.saved(), before)
        self.assertEqual(admin.current_settings["transcription_language"], "de")
        self.assertEqual(self.loaded, [])

    def test_a_form_of_the_active_model_saves_and_the_field_is_not_stored(self) -> None:
        response = self.put(self.open_form(), batch_max_segments=9, expected_local_model=COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["settings"]["batch_max_segments"], 9)
        self.assertNotIn("expected_local_model", response.json()["settings"])
        self.assertNotIn("expected_local_model", self.saved())
        self.assertEqual(self.saved()["model_profiles"][COHERE_MODEL_ID]["batch_max_segments"], 9)

    def test_callers_that_omit_the_field_keep_todays_behaviour(self) -> None:
        form = self.open_form()
        self.select(WHISPER_TURBO)
        for fields in ({}, {"expected_local_model": None}):
            with self.subTest(fields=fields):
                self.assertEqual(self.put(form, **fields).status_code, 200)
                self.assertEqual(self.saved()["transcription_language"], "auto")

    def test_a_form_stays_valid_after_a_failed_selection(self) -> None:
        form = self.open_form()
        self.failing.add(WHISPER_TURBO)
        self.assertEqual(self.select(WHISPER_TURBO).status_code, 500)
        self.assertEqual(self.put(form, expected_local_model=COHERE_MODEL_ID).status_code, 200)

    def test_put_that_changes_the_model_is_checked_against_the_model_before_it(self) -> None:
        response = self.put({}, local_model=WHISPER_TURBO, expected_local_model=COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.saved()["local_model"], WHISPER_TURBO)

        self.loaded.clear()
        response = self.put({}, local_model=PARAKEET_MODEL_ID, expected_local_model=COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.saved()["local_model"], WHISPER_TURBO)
        self.assertEqual(self.loaded, [])


class ModelPresetTests(_AdminCase):
    def listed(self, query: str = "") -> dict:
        return {preset["name"]: preset for preset in self.client.get(f"/api/admin/settings/presets{query}").json()["presets"]}

    def test_builtins_belong_to_their_model_and_their_values_stay_clean(self) -> None:
        listed = self.listed()
        for name in ("32GB VRAM", "24GB VRAM", "24+8 VRAM"):
            self.assertEqual(listed[name]["model_id"], COHERE_MODEL_ID)
            self.assertNotIn("model_id", listed[name]["values"])
            self.assertNotIn("local_model", listed[name]["values"])
        self.assertEqual(listed["Parakeet 8GB VRAM"]["model_id"], PARAKEET_MODEL_ID)

    def test_user_presets_derive_it_from_their_values(self) -> None:
        self.client.put("/api/admin/settings/presets/mine", json={"values": {"local_model": WHISPER_TURBO, "batch_max_segments": 4}})
        self.client.put("/api/admin/settings/presets/any", json={"values": {"batch_max_segments": 4}})
        listed = self.listed()
        self.assertEqual(listed["mine"]["model_id"], WHISPER_TURBO)
        self.assertIsNone(listed["any"]["model_id"])

    def test_saved_response_and_stored_format_are_unchanged_apart_from_the_derived_field(self) -> None:
        response = self.client.put("/api/admin/settings/presets/mine", json={"values": {"local_model": WHISPER_TURBO}})
        self.assertEqual(response.json()["preset"]["model_id"], WHISPER_TURBO)
        with open(presets.PRESETS_FILE, "r", encoding="utf-8") as file_obj:
            stored = json.load(file_obj)
        self.assertEqual(set(stored["mine"]), {"name", "values", "created_at"})

    def test_filter_returns_the_model_plus_the_ones_for_every_model(self) -> None:
        self.client.put("/api/admin/settings/presets/mine", json={"values": {"local_model": WHISPER_TURBO, "batch_max_segments": 4}})
        self.client.put("/api/admin/settings/presets/any", json={"values": {"batch_max_segments": 4}})
        self.assertEqual(list(self.listed(f"?model_id={WHISPER_TURBO}")), ["any", "mine"])
        self.assertEqual(list(self.listed(f"?model_id={PARAKEET_MODEL_ID}")), ["Parakeet 8GB VRAM", "any"])
        self.assertEqual(list(self.listed(f"?model_id={COHERE_MODEL_ID}")), ["32GB VRAM", "24GB VRAM", "24+8 VRAM", "any"])

    def test_no_or_empty_filter_lists_everything(self) -> None:
        self.assertEqual(list(self.listed()), list(presets.BUILTIN_PRESETS))
        self.assertEqual(list(self.listed("?model_id=")), list(presets.BUILTIN_PRESETS))

    def test_every_builtin_has_a_model_and_start_tiers_exist(self) -> None:
        self.assertEqual(set(presets.BUILTIN_PRESET_MODELS), set(presets.BUILTIN_PRESETS))
        for model_id, tier in presets.MODEL_START_PRESETS.items():
            self.assertEqual(presets.BUILTIN_PRESET_MODELS[tier], model_id)


class DeleteGuardTests(_AdminCase):
    def delete(self, model_id):
        return self.client.post("/api/admin/models/delete", json={"model_id": model_id})

    def test_active_model_is_refused(self) -> None:
        with mock.patch.object(admin, "delete_model_cache") as delete_cache:
            response = self.delete(COHERE_MODEL_ID)
        self.assertEqual(response.status_code, 409)
        delete_cache.assert_not_called()

    def test_another_model_can_be_deleted(self) -> None:
        with mock.patch.object(admin, "delete_model_cache", return_value={"ok": True}) as delete_cache:
            response = self.delete(WHISPER_TURBO)
        self.assertEqual(response.status_code, 200, response.text)
        delete_cache.assert_called_once()
        self.assertIn("active", response.json()["models"][0])


class ClientDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_settings = admin.current_settings.copy()
        patcher = mock.patch.dict(admin.local_model_components, {"model": None, "model_identifier": None}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.use_settings()
        app = FastAPI()
        api.create_api(app)
        v2.create_v2_capabilities_api(app)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        admin.current_settings.clear()
        admin.current_settings.update(self.original_settings)

    def use_settings(self, **values) -> None:
        admin.current_settings.clear()
        admin.current_settings.update(storage.normalize_settings({**storage.DEFAULT_SETTINGS, **values}))

    def capabilities(self) -> dict:
        response = self.client.get("/v2/capabilities")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_shape_on_cohere(self) -> None:
        admin.local_model_components.update({"model": object(), "model_identifier": (COHERE_MODEL_ID, "auto", "", "fp16")})
        body = self.capabilities()
        self.assertEqual(set(body), {"schema_version", "service", "model", "language", "modes", "embedding", "diarization"})
        self.assertEqual((body["schema_version"], body["service"]), ("2.0", "asr"))
        self.assertEqual(
            body["model"],
            {"id": COHERE_MODEL_ID, "label": "Cohere Transcribe 03/2026", "backend": "cohere_transcribe", "loaded": True},
        )
        language = body["language"]
        self.assertEqual((language["configured"], language["effective"], language["applied"]), ("auto", "de", True))
        self.assertEqual(language["options"][:2], ["auto", "de"])
        self.assertEqual(body["modes"], ["embedding", "transcript", "transcript_embedding", "diarization"])
        self.assertEqual(body["embedding"], {"id": "ReDimNet2-B6"})
        self.assertTrue(body["diarization"]["configured"])

    def test_language_is_not_applied_on_parakeet_but_options_stay_global(self) -> None:
        cohere_options = self.capabilities()["language"]["options"]
        self.use_settings(local_model=PARAKEET_MODEL_ID)
        language = self.capabilities()["language"]
        self.assertFalse(language["applied"])
        self.assertEqual(language["options"], cohere_options)
        self.assertEqual(set(language["options"]), v2.SUPPORTED_LANGUAGE_VALUES)

    def test_whisper_auto_stays_auto_and_loaded_is_truthful(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO)
        body = self.capabilities()
        self.assertEqual(body["language"]["effective"], "auto")
        self.assertTrue(body["language"]["applied"])
        self.assertFalse(body["model"]["loaded"])

    def test_diarization_reports_a_broken_url_as_not_configured(self) -> None:
        self.use_settings(dia_server_base_url="not a url")
        self.assertFalse(self.capabilities()["diarization"]["configured"])

    def test_needs_no_key_and_leaks_no_secret(self) -> None:
        self.use_settings(huggingface_token="hf_secret", dia_api_key="dia_secret", dia_server_base_url="http://dia.internal:7864")
        with mock.patch.object(api, "authorize_api_key", side_effect=AssertionError("must stay open")):
            with mock.patch.object(v2, "authorize_api_key", side_effect=AssertionError("must stay open")):
                text = self.client.get("/v2/capabilities").text
        for secret in ("hf_secret", "dia_secret", "dia.internal"):
            self.assertNotIn(secret, text)

    def test_is_registered_before_the_catch_all(self) -> None:
        paths = [getattr(route, "path", None) for route in server.app.routes]
        self.assertLess(paths.index("/v2/capabilities"), paths.index("/{full_path:path}"))

    def test_v1_models_keeps_active_first_then_the_alias_without_auth(self) -> None:
        self.use_settings(local_model=WHISPER_TURBO)
        with mock.patch.object(api, "authorize_api_key", side_effect=AssertionError("must stay open")):
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"object", "data"})
        self.assertEqual([entry["id"] for entry in body["data"]], [WHISPER_TURBO, "whisper-1"])
        self.assertEqual(set(body["data"][0]), {"id", "object", "created", "owned_by"})


if __name__ == "__main__":
    unittest.main()
