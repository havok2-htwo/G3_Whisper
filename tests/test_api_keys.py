"""Optional API keys, admin login hardening, HF token masking, the static catch-all and the performance metrics.

Shared G3 admin standard (schema 1): no key -> the public API is open; once a key exists, Bearer or X-API-Key (or the
admin's own session, without an explicitly sent key) is required; the key store fails closed.
"""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend import g3_gpu
from backend import genesis_whisper_server as server
from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as api
from backend import genesis_whisper_server_auth as auth
from backend import genesis_whisper_server_perf as perf
from backend import genesis_whisper_server_storage as storage
from backend.genesis_whisper_server_globals import settings_lock


class _StoreCase(unittest.TestCase):
    """A fresh AuthStore in a temp dir, installed as the process-wide store."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-whisper-auth-")
        self.root = Path(self.temp_dir.name)
        self.store = auth.AuthStore(self.root)
        self.store_patch = mock.patch.object(auth, "_auth_store", self.store)
        self.store_patch.start()
        auth.login_rate_limiter.reset()

    def tearDown(self) -> None:
        self.store.flush()
        self.store_patch.stop()
        auth.login_rate_limiter.reset()
        self.temp_dir.cleanup()

    def _session(self, *, must_change: bool = False) -> str:
        user = self.store._payload["users"]["admin"]
        user["must_change_password"] = must_change
        return self.store.create_session("admin")

    def _gate_app(self) -> TestClient:
        app = FastAPI()

        @app.get("/gated")
        async def gated(request: Request):
            return {"key_id": auth.authorize_api_key(request)}

        return TestClient(app)


class ApiKeyGateTests(_StoreCase):
    def test_open_without_keys(self) -> None:
        client = self._gate_app()
        response = client.get("/gated")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["key_id"])

    def test_bearer_and_x_api_key_are_accepted_and_no_key_is_refused(self) -> None:
        created = self.store.create_api_key("client")
        client = self._gate_app()
        bearer = client.get("/gated", headers={"Authorization": f"Bearer {created['token']}"})
        header = client.get("/gated", headers={"X-API-Key": created["token"]})
        missing = client.get("/gated")
        self.assertEqual((bearer.status_code, bearer.json()["key_id"]), (200, created["id"]))
        self.assertEqual((header.status_code, header.json()["key_id"]), (200, created["id"]))
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers.get("www-authenticate"), "Bearer")
        self.assertIn("Authorization: Bearer", missing.json()["detail"])

    def test_admin_session_works_without_a_key_but_an_explicit_wrong_key_wins(self) -> None:
        self.store.create_api_key("client")
        client = self._gate_app()
        client.cookies.set(auth.SESSION_COOKIE_NAME, self._session())
        with_session = client.get("/gated")
        wrong_key = client.get("/gated", headers={"X-API-Key": "whisper_wrong"})
        wrong_bearer = client.get("/gated", headers={"Authorization": "Bearer whisper_wrong"})
        self.assertEqual(with_session.status_code, 200)
        self.assertIsNone(with_session.json()["key_id"])  # no usage counted for the admin
        self.assertEqual(wrong_key.status_code, 401)
        self.assertEqual(wrong_bearer.status_code, 401)

    def test_session_that_must_change_its_password_is_not_enough(self) -> None:
        self.store.create_api_key("client")
        client = self._gate_app()
        client.cookies.set(auth.SESSION_COOKIE_NAME, self._session(must_change=True))
        self.assertEqual(client.get("/gated").status_code, 401)

    def test_revoking_the_last_key_opens_the_api_again(self) -> None:
        created = self.store.create_api_key("client")
        client = self._gate_app()
        self.assertEqual(client.get("/gated").status_code, 401)
        self.assertTrue(self.store.delete_api_key(created["id"]))
        self.assertEqual(client.get("/gated").status_code, 200)
        # Counter-check: the revoked key itself is no longer matched.
        self.assertIsNone(self.store.match_api_key(created["token"]))

    def test_hint_is_stored_and_listed_but_never_the_hash_or_token(self) -> None:
        created = self.store.create_api_key("  client  ")
        token = created["token"]
        body = token.split("_", 1)[1]
        self.assertEqual(created["hint"], f"whisper_{body[:4]}…{body[-4:]}")
        self.assertEqual(created["alias"], "client")
        listed = self.store.list_api_keys()
        self.assertEqual(listed[0]["hint"], created["hint"])
        text = json.dumps(listed)
        self.assertNotIn(token, text)
        self.assertNotIn("key_hash", text)
        self.assertNotIn(auth.hash_token(token), text)

    def test_legacy_key_without_hint_lists_none(self) -> None:
        created = self.store.create_api_key("old")
        self.store._payload["api_keys"][created["id"]].pop("hint")
        self.assertIsNone(self.store.list_api_keys()[0]["hint"])

    def test_alias_is_capped_at_64_characters(self) -> None:
        self.assertEqual(len(self.store.create_api_key("x" * 200)["alias"]), auth.API_KEY_ALIAS_MAX)

    def test_usage_is_counted_in_memory_and_flushed_debounced(self) -> None:
        created = self.store.create_api_key("client")
        before = self.store.secrets_path.read_text(encoding="utf-8")
        for _ in range(3):
            self.store.record_api_key_usage(created["id"], 2.5)
        # No write per request ...
        self.assertEqual(self.store.secrets_path.read_text(encoding="utf-8"), before)
        usage = self.store.list_api_keys()[0]["usage"]
        self.assertEqual((usage["request_count"], usage["total_seconds_processed"]), (3, 7.5))
        # ... one write on flush (timer or shutdown).
        self.store.flush()
        saved = json.loads(self.store.secrets_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["api_keys"][created["id"]]["usage"]["request_count"], 3)

    def test_keys_survive_a_reload_of_the_file(self) -> None:
        created = self.store.create_api_key("client")
        reloaded = auth.AuthStore(self.root)
        self.assertEqual(reloaded.match_api_key(created["token"]), created["id"])

    def test_unreadable_secrets_file_fails_closed_and_is_left_untouched(self) -> None:
        self.store.create_api_key("client")
        broken = "{ this is not json"
        self.store.secrets_path.write_text(broken, encoding="utf-8")
        with mock.patch("sys.stderr"):
            store = auth.AuthStore(self.root)
        self.assertIsNotNone(store.load_error)
        with mock.patch.object(auth, "_auth_store", store):
            response = self._gate_app().get("/gated")
            self.assertEqual(response.status_code, 503)
            with self.assertRaises(auth.HTTPException) as ctx:
                store.create_api_key("new")
            self.assertEqual(ctx.exception.status_code, 503)
        self.assertIsNone(store.verify_user("admin", "admin"))
        self.assertEqual(self.store.secrets_path.read_text(encoding="utf-8"), broken)

    def test_secrets_file_is_written_atomically_without_temp_leftovers(self) -> None:
        self.store.create_api_key("client")
        leftovers = [p.name for p in self.store.logs_dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class PublicRouteTests(_StoreCase):
    def test_v1_models_is_open_without_keys_and_gated_with_a_key(self) -> None:
        client = TestClient(server.app)
        self.assertEqual(client.get("/v1/models").status_code, 200)
        created = self.store.create_api_key("client")
        self.assertEqual(client.get("/v1/models").status_code, 401)
        ok = client.get("/v1/models", headers={"Authorization": f"Bearer {created['token']}"})
        self.assertEqual(ok.status_code, 200)

    def test_capabilities_health_and_landing_stay_public(self) -> None:
        self.store.create_api_key("client")
        client = TestClient(server.app)
        self.assertEqual(client.get("/v2/capabilities").status_code, 200)
        landing = client.get("/")
        self.assertEqual(landing.status_code, 200)
        self.assertIn('lang="en"', landing.text)
        self.assertIn("G3 Whisper", landing.text)


class StaticCatchAllTests(unittest.TestCase):
    """Regression: GET /..%2f..%2f<file> used to serve any file of the project (path traversal)."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-whisper-dist-")
        root = Path(self.temp_dir.name)
        self.dist = root / "frontend" / "dist"
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text("<!doctype html><title>G3 Whisper · Admin</title>", encoding="utf-8")
        (self.dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
        (root / "README.md").write_text("SECRET-README", encoding="utf-8")
        (root / "frontend" / "secret.env").write_text("SECRET-ENV", encoding="utf-8")
        self.patch = mock.patch.object(server, "FRONTEND_DIST_DIR", self.dist)
        self.patch.start()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        self.patch.stop()
        self.temp_dir.cleanup()

    def test_assets_and_index_are_served(self) -> None:
        self.assertEqual(self.client.get("/assets/app.js").text, "console.log('ok')")
        self.assertEqual(self.client.get("/admin/assets/app.js").text, "console.log('ok')")
        self.assertIn("G3 Whisper", self.client.get("/admin").text)

    def test_traversal_never_serves_files_outside_dist(self) -> None:
        for path in (
            "/..%2f..%2fREADME.md",
            "/..%2F..%2FREADME.md",
            "/assets/..%2f..%2f..%2fREADME.md",
            "/%2e%2e/%2e%2e/README.md",
            "/admin/..%2f..%2f..%2fREADME.md",
            "/..%2fsecret.env",
            "/..%5c..%5cREADME.md",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertNotIn("SECRET", response.text)
                self.assertNotEqual(response.headers.get("content-type", ""), "text/markdown; charset=utf-8")

    def test_path_as_is_dot_dot_is_refused(self) -> None:
        # httpx normalises "/../" in URLs, so call the resolver like Starlette would after decoding.
        self.assertIsNone(server._resolve_frontend_asset("../../README.md"))
        self.assertIsNone(server._resolve_frontend_asset("assets/../../README.md"))
        self.assertIsNone(server._resolve_frontend_asset("..\\..\\README.md"))
        self.assertIsNone(server._resolve_frontend_asset("C:/Windows/win.ini"))
        self.assertIsNotNone(server._resolve_frontend_asset("assets/app.js"))

    def test_unknown_api_paths_answer_json_404(self) -> None:
        for path in ("/api/nope", "/v1/nope", "/v2/nope"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], "Not Found")


class AdminAuthHardeningTests(_StoreCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(admin.create_admin_api(FastAPI()))

    def test_setup_flag_follows_the_default_password(self) -> None:
        self.assertTrue(self.client.get("/api/admin/auth/setup").json()["setup_required"])
        self.store.set_password("admin", "a-long-password")
        self.assertFalse(self.client.get("/api/admin/auth/setup").json()["setup_required"])

    def test_login_sets_a_strict_http_only_cookie(self) -> None:
        response = self.client.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
        self.assertEqual(response.status_code, 200, response.text)
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("samesite=strict", cookie)
        self.assertIn("httponly", cookie)

    def test_login_is_rate_limited_after_five_failures(self) -> None:
        codes = [
            self.client.post("/api/admin/auth/login", json={"username": "admin", "password": "wrong"}).status_code
            for _ in range(6)
        ]
        self.assertEqual(codes, [401] * 5 + [429])
        # Counter-check: another client is not affected.
        auth.login_rate_limiter.succeeded("testclient")
        ok = self.client.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
        self.assertEqual(ok.status_code, 200)

    def test_cross_origin_admin_writes_are_refused(self) -> None:
        cross = self.client.post(
            "/api/admin/auth/login",
            json={"username": "admin", "password": "admin"},
            headers={"Origin": "http://evil.example"},
        )
        same = self.client.post(
            "/api/admin/auth/login",
            json={"username": "admin", "password": "admin"},
            headers={"Origin": "http://testserver"},
        )
        self.assertEqual(cross.status_code, 403)
        self.assertEqual(same.status_code, 200)

    def test_new_password_needs_eight_characters(self) -> None:
        self.client.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
        short = self.client.post(
            "/api/admin/auth/change-password", json={"current_password": "admin", "new_password": "short"}
        )
        self.assertEqual(short.status_code, 400)
        self.assertIn("8", short.json()["detail"])
        ok = self.client.post(
            "/api/admin/auth/change-password", json={"current_password": "admin", "new_password": "long-enough"}
        )
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_api_key_routes_create_list_and_revoke(self) -> None:
        self.store.set_password("admin", "long-enough")
        self.client.cookies.set(auth.SESSION_COOKIE_NAME, self.store.create_session("admin"))
        created = self.client.post("/api/admin/api-keys", json={"alias": "client"}).json()
        self.assertTrue(created["token"].startswith("whisper_"))
        listed = self.client.get("/api/admin/api-keys").json()["keys"]
        self.assertEqual([(k["alias"], k["hint"]) for k in listed], [("client", created["hint"])])
        self.assertNotIn(created["token"], json.dumps(listed))
        self.assertEqual(self.client.delete(f"/api/admin/api-keys/{created['id']}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/admin/api-keys/{created['id']}").status_code, 404)


class HuggingFaceTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-whisper-token-")
        self.settings_file = Path(self.temp_dir.name) / "settings.json"
        self.original = admin.current_settings.copy()
        self.patches = [
            mock.patch.object(storage, "LOGS_DIR", self.temp_dir.name),
            mock.patch.object(storage, "SETTINGS_FILE", str(self.settings_file)),
            mock.patch.object(admin, "list_model_statuses", return_value=[]),
        ]
        for patcher in self.patches:
            patcher.start()
        with settings_lock:
            admin.current_settings.clear()
            admin.current_settings.update(
                storage.normalize_settings({**storage.DEFAULT_SETTINGS, "huggingface_token": "hf_SECRETVALUEabcd"})
            )
        app = admin.create_admin_api(FastAPI())
        app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        self.client = TestClient(app)

    def tearDown(self) -> None:
        with settings_lock:
            admin.current_settings.clear()
            admin.current_settings.update(self.original)
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def test_get_returns_the_token_masked(self) -> None:
        response = self.client.get("/api/admin/settings")
        self.assertNotIn("SECRETVALUE", response.text)
        settings = response.json()["settings"]
        self.assertEqual(settings["huggingface_token"], "hf_…abcd")
        self.assertTrue(settings["huggingface_token_set"])

    def test_empty_or_masked_put_keeps_the_token(self) -> None:
        for sent in ({}, {"huggingface_token": ""}, {"huggingface_token": "hf_…abcd"}):
            with self.subTest(sent=sent):
                response = self.client.put("/api/admin/settings", json={"batch_wait_time_ms": 50, **sent})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertNotIn("SECRETVALUE", response.text)
                self.assertEqual(admin.current_settings["huggingface_token"], "hf_SECRETVALUEabcd")

    def test_new_token_replaces_and_clear_removes(self) -> None:
        self.client.put("/api/admin/settings", json={"huggingface_token": "hf_NEWTOKEN12345678"})
        self.assertEqual(admin.current_settings["huggingface_token"], "hf_NEWTOKEN12345678")
        cleared = self.client.put("/api/admin/settings", json={"huggingface_token_clear": True})
        self.assertEqual(admin.current_settings["huggingface_token"], "")
        self.assertFalse(cleared.json()["settings"]["huggingface_token_set"])
        self.assertEqual(cleared.json()["settings"]["huggingface_token"], "")
        persisted = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted["huggingface_token"], "")
        self.assertNotIn("huggingface_token_set", persisted)


class PerfMetricTests(unittest.TestCase):
    def test_realtime_factor_is_audio_seconds_per_wall_second_over_the_window(self) -> None:
        window = g3_gpu.RateWindow(30)
        window.add(45.0)
        window.add(15.0)
        metrics = perf.service_metrics(window, {"queue_size": 3, "pending_buffer_size": 4})
        throughput, queue = metrics
        self.assertEqual(throughput["key"], "throughput")
        self.assertEqual((throughput["label"], throughput["unit"], throughput["window_s"]), ("Realtime factor", "x", 30))
        self.assertAlmostEqual(throughput["value"], 2.0)  # 60 audio seconds in 30 s
        self.assertEqual((queue["key"], queue["label"], queue["unit"], queue["integer"]), ("queue", "Queue (chunks)", "", True))
        self.assertEqual(queue["value"], 7)

    def test_rate_falls_to_zero_once_the_window_has_passed(self) -> None:
        window = g3_gpu.RateWindow(30)
        with mock.patch.object(g3_gpu.time, "monotonic", return_value=1000.0):
            window.add(90.0)
            self.assertAlmostEqual(window.rate(), 3.0)
        with mock.patch.object(g3_gpu.time, "monotonic", return_value=1031.0):
            self.assertEqual(window.rate(), 0.0)

    def test_queue_never_negative_and_missing_values_count_zero(self) -> None:
        self.assertEqual(perf.queued_chunks({"queue_size": None, "pending_buffer_size": -2}), 0)

    def test_perf_endpoint_returns_schema_1(self) -> None:
        gpus = [
            {"nvml_index": 1, "cuda_index": 0, "uuid": "GPU-x", "pci_bus_id": None, "name": "NVIDIA GeForce RTX 3090",
             "used_by_service": True, "util_pct": 37, "mem_total_mib": 24576, "mem_used_mib": 17978, "process_mib": 7254,
             "temp_c": 54, "power_w": 112.4, "power_limit_w": 350},
        ]
        app = admin.create_admin_api(FastAPI())
        app.state.whisper_batch_manager = object()
        app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        with mock.patch.object(g3_gpu, "gpu_snapshot", return_value=gpus):
            body = TestClient(app).get("/api/admin/perf").json()
        self.assertEqual((body["schema"], body["service"], body["error"]), (1, "G3 Whisper", None))
        self.assertEqual(body["gpus"][0]["mem_total_mib"], 24576)  # MiB on the wire, the UI converts to GB
        self.assertEqual([m["key"] for m in body["metrics"]], ["throughput", "queue"])
        self.assertLessEqual(abs(body["sampled_at_ms"] - time.time() * 1000), 5000)

    def test_gpu_option_is_named_like_nvidia_smi_not_by_cuda_index(self) -> None:
        # CUDA_VISIBLE_DEVICES=<3090 UUID>: the 3090 is cuda:0 in this process but GPU 1 in nvidia-smi.
        card = {"nvml_index": 1, "cuda_index": 0, "name": "NVIDIA GeForce RTX 3090", "mem_total_mib": 24576}
        with mock.patch.dict(admin._device_label_cache, clear=True), mock.patch.object(
            admin, "gpu_for_cuda_index", return_value=card
        ) as lookup:
            label = admin._device_label("GPU 0 (NVIDIA GeForce RTX 3090, 24 GB)", "cuda:0")
            again = admin._device_label("GPU 0 (NVIDIA GeForce RTX 3090, 24 GB)", "cuda:0")
        self.assertEqual(label, "GPU 1 · GeForce RTX 3090 · 24 GB")
        self.assertEqual(again, label)
        lookup.assert_called_once_with(0)  # identity is cached, nvidia-smi runs once
        self.assertEqual(admin._device_label("Auto (Empfohlen)", "auto"), "Auto (recommended)")

    def test_finished_batches_feed_the_window(self) -> None:
        from backend import genesis_whisper_server_batching as batching

        self.assertIs(batching.asr_audio_rate, perf.asr_audio_rate)


if __name__ == "__main__":
    unittest.main()
