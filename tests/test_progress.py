"""Progress + ETA of a running /v2/audio/process request (Genesis3 docs/254). CPU only, no models."""

from __future__ import annotations

import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from starlette.datastructures import FormData

from backend import genesis_whisper_server_auth as auth
from backend import genesis_whisper_server_gpu as gpu
from backend import genesis_whisper_server_progress as progress
from backend import genesis_whisper_server_v2 as v2
from tests.test_v2_api import _CapturingApp, _payload, _request, _upload

PID = "0123456789abcdef0123456789abcdef"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _registry() -> tuple[progress.ProgressRegistry, _Clock]:
    clock = _Clock()
    return progress.ProgressRegistry(clock=clock, max_entries=4, ttl_s=60.0), clock


class RegistryTests(unittest.TestCase):
    def test_only_well_formed_free_ids_are_tracked(self) -> None:
        registry, _ = _registry()
        self.assertIsNone(registry.start(None, "k1"))
        self.assertIsNone(registry.start("short", "k1"))
        self.assertIsNone(registry.start("x" * 65, "k1"))
        self.assertIsNone(registry.start("has space in it 123", "k1"))
        entry = registry.start(PID, "k1")
        self.assertIsNotNone(entry)
        # Still running: a second request with the same id is not tracked (the first keeps its entry).
        self.assertIsNone(registry.start(PID, "k1"))
        registry.finish(entry, "completed")
        # Finished: another key cannot take the id over, the same key can reuse it.
        self.assertIsNone(registry.start(PID, "k2"))
        self.assertIsNotNone(registry.start(PID, "k1"))

    def test_lookup_is_per_key(self) -> None:
        registry, _ = _registry()
        registry.start(PID, "k1")
        self.assertIsNotNone(registry.get(PID, "k1"))
        self.assertIsNone(registry.get(PID, "k2"))
        self.assertIsNone(registry.get(PID, None))

    def test_finished_entries_expire_and_the_table_is_bounded(self) -> None:
        registry, clock = _registry()
        first = registry.start(PID, None)
        registry.finish(first, "completed")
        clock.now += 61
        self.assertIsNone(registry.get(PID, None))
        ids = [f"{index:032d}" for index in range(6)]
        for value in ids:
            registry.start(value, None)
        self.assertLessEqual(len(registry._entries), 4)
        self.assertIsNotNone(registry.get(ids[-1], None))

    def test_bounded_table_drops_finished_entries_before_running_ones(self) -> None:
        registry, _ = _registry()
        running = [registry.start(f"{index:032d}", None) for index in range(3)]
        done = registry.start("f" * 32, None)
        registry.finish(done, "completed")
        registry.start("e" * 32, None)
        self.assertIsNone(registry.get("f" * 32, None))
        for entry in running:
            self.assertIsNotNone(registry.get(entry.progress_id, None))


class SnapshotTests(unittest.TestCase):
    def _diarization(self, audio_s: float = 600.0):
        registry, clock = _registry()
        entry = registry.start(PID, None)
        registry.set_mode(entry, "diarization")
        registry.enter_phase(entry, "decode")
        clock.now += 0.4
        registry.set_audio_seconds(entry, audio_s)
        return registry, clock, entry

    def test_no_eta_before_the_audio_is_decoded(self) -> None:
        registry, _ = _registry()
        entry = registry.start(PID, None)
        snapshot = registry.snapshot(entry)
        self.assertEqual(snapshot["state"], "running")
        self.assertEqual(snapshot["phase"], "upload")
        self.assertIsNone(snapshot["eta_ms"])
        self.assertIsNone(snapshot["audio_duration_ms"])

    def test_declared_units_give_fraction_and_rate_based_eta(self) -> None:
        registry, clock, entry = self._diarization()
        registry.enter_phase(entry, "speaker_windows", total=100)
        clock.now += 10.0
        registry.add_units(entry, done=25)
        snapshot = registry.snapshot(entry)
        self.assertEqual((snapshot["phase"], snapshot["phase_done"], snapshot["phase_total"]),
                         ("speaker_windows", 25, 100))
        # 10 s for 25 % -> 30 s left in the phase, plus the later phases at their expected rates.
        later = sum(registry.rates[name] for name in entry.phases[entry.phases.index("speaker_windows") + 1:])
        self.assertAlmostEqual(snapshot["eta_ms"] / 1000.0, 30.0 + later * 600.0 / 1000.0, delta=8.0)
        self.assertGreater(snapshot["progress"], 0.0)
        self.assertLess(snapshot["progress"], 1.0)

    def test_progress_never_goes_backwards_and_skipped_phases_count_as_done(self) -> None:
        registry, clock, entry = self._diarization()
        registry.enter_phase(entry, "segmentation")
        registry.add_units(entry, total=10)
        registry.add_units(entry, done=10)
        high = registry.snapshot(entry)["progress"]
        # A second sliced call in the same phase adds a new total: the fraction drops, the reported progress not.
        registry.add_units(entry, total=50)
        self.assertGreaterEqual(registry.snapshot(entry)["progress"], high)
        registry.enter_phase(entry, "transcription")  # embedding + language detection skipped
        snapshot = registry.snapshot(entry)
        self.assertEqual(snapshot["phase"], "transcription")
        self.assertEqual(snapshot["phase_index"], entry.phases.index("transcription"))
        # Back to an earlier phase is ignored (the legacy fallback path after WXC).
        registry.enter_phase(entry, "embedding")
        self.assertEqual(registry.snapshot(entry)["phase"], "transcription")

    def test_waiting_for_the_gpu_is_a_state_without_eta_and_does_not_count_as_work(self) -> None:
        registry, clock, entry = self._diarization()
        registry.enter_phase(entry, "speaker_windows", total=100)
        registry.add_units(entry, done=10)
        registry.begin_wait(entry)
        clock.now += 30.0
        snapshot = registry.snapshot(entry)
        self.assertEqual(snapshot["state"], "waiting")
        self.assertIsNone(snapshot["eta_ms"])
        registry.end_wait(entry)
        clock.now += 2.0
        registry.add_units(entry, done=10)
        snapshot = registry.snapshot(entry)
        self.assertEqual(snapshot["state"], "running")
        # 2 s of work for 20 % (the 30 s wait excluded) -> 8 s left in this phase.
        later = sum(registry.rates[name] for name in entry.phases[entry.phases.index("speaker_windows") + 1:])
        self.assertAlmostEqual(snapshot["eta_ms"] / 1000.0, 8.0 + later * 0.6, delta=6.0)

    def test_a_phase_without_units_past_its_expectation_has_no_eta(self) -> None:
        registry, clock, entry = self._diarization(audio_s=100.0)
        registry.enter_phase(entry, "clustering")  # expected ~0.6 s for 100 s audio
        self.assertIsNotNone(registry.snapshot(entry)["eta_ms"])
        clock.now += 30.0
        self.assertIsNone(registry.snapshot(entry)["eta_ms"])

    def test_probe_units_and_queue_wait(self) -> None:
        registry, clock, entry = self._diarization()
        state = {"value": (0, 40, True)}
        registry.enter_phase(entry, "transcription", probe=lambda: state["value"])
        self.assertEqual(registry.snapshot(entry)["state"], "waiting")
        state["value"] = (10, 40, False)
        clock.now += 5.0
        snapshot = registry.snapshot(entry)
        self.assertEqual((snapshot["state"], snapshot["phase_done"], snapshot["phase_total"]), ("running", 10, 40))
        self.assertIsNotNone(snapshot["eta_ms"])
        registry.add_units(entry, done=5)  # sliced calls never override a probe
        self.assertEqual(registry.snapshot(entry)["phase_done"], 10)

    def test_a_failing_probe_only_loses_the_units(self) -> None:
        registry, _, entry = self._diarization()

        def broken():
            raise RuntimeError("boom")

        registry.enter_phase(entry, "transcription", probe=broken)
        snapshot = registry.snapshot(entry)
        self.assertIsNone(snapshot["phase_done"])
        self.assertEqual(snapshot["phase"], "transcription")

    def test_finish_reports_completion_and_teaches_the_rates(self) -> None:
        registry, clock, entry = self._diarization(audio_s=600.0)
        registry.enter_phase(entry, "transcription")
        clock.now += 6.0  # 10 ms per audio second
        registry.finish(entry, "completed")
        snapshot = registry.snapshot(entry)
        self.assertEqual((snapshot["state"], snapshot["progress"], snapshot["eta_ms"]), ("completed", 1.0, 0))
        self.assertAlmostEqual(registry.rates["transcription"], 10.0, places=3)
        self.assertNotIn("upload", registry.learned)

    def test_failed_requests_keep_their_progress_and_teach_nothing(self) -> None:
        registry, clock, entry = self._diarization()
        registry.enter_phase(entry, "transcription")
        before = dict(registry.rates)
        clock.now += 50.0
        registry.finish(entry, "failed")
        snapshot = registry.snapshot(entry)
        self.assertEqual(snapshot["state"], "failed")
        self.assertLess(snapshot["progress"], 1.0)
        self.assertEqual(registry.rates, before)

    def test_short_audio_does_not_teach(self) -> None:
        registry, clock, entry = self._diarization(audio_s=10.0)
        registry.enter_phase(entry, "transcription")
        clock.now += 5.0
        before = dict(registry.rates)
        registry.finish(entry, "completed")
        self.assertEqual(registry.rates, before)

    def test_other_modes_use_the_short_phase_list(self) -> None:
        registry, _ = _registry()
        entry = registry.start(PID, None)
        registry.set_mode(entry, "transcript")
        registry.enter_phase(entry, "speaker_windows")  # not a phase of this mode: ignored
        registry.enter_phase(entry, "processing")
        self.assertEqual(registry.snapshot(entry)["phase"], "processing")


class SlicedPhaseCountingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.registry, self.clock = _registry()
        self.patch = mock.patch.object(progress, "registry", self.registry)
        self.patch.start()

    async def asyncTearDown(self) -> None:
        self.patch.stop()

    async def test_slices_count_items_and_a_foreign_lock_holder_is_waiting(self) -> None:
        entry = self.registry.start(PID, None)
        self.registry.set_mode(entry, "diarization")
        lock = asyncio.Lock()
        seen: list[dict] = []
        release = asyncio.Event()

        async def fake_phase(function, items, *args):
            seen.append(self.registry.snapshot(entry))
            return function(items, *args)

        async def foreign_holder() -> None:
            async with lock:
                await release.wait()

        with mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase):
            holder = asyncio.create_task(foreign_holder())
            await asyncio.sleep(0)
            token = progress.bind(entry)
            try:
                progress.enter_phase("segmentation")
                phase = asyncio.create_task(gpu.run_sliced_gpu_phase(lock, list(range(10)), len, 4))
            finally:
                progress.unbind(token)
            await asyncio.sleep(0)
            self.assertEqual(self.registry.snapshot(entry)["state"], "waiting")
            release.set()
            results = await phase
            await holder

        self.assertEqual(results, [4, 4, 2])
        self.assertEqual(seen[0]["state"], "running")
        self.assertEqual([item["phase_done"] for item in seen], [0, 4, 8])
        final = self.registry.snapshot(entry)
        self.assertEqual((final["phase_done"], final["phase_total"]), (10, 10))

    async def test_untracked_calls_are_unchanged(self) -> None:
        async def fake_phase(function, items, *args):
            return function(items, *args)

        with mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase):
            self.assertEqual(await gpu.run_sliced_gpu_phase(asyncio.Lock(), [1, 2, 3], sum, 2), [3, 3])


class _FakeJob:
    def __init__(self, request_id: str, pending: int, inflight: int, started: bool) -> None:
        self.request_id = request_id
        self.pending = list(range(pending))
        self.inflight = set(range(inflight))
        self.started_at = 1.0 if started else None


class AsrProbeTests(unittest.TestCase):
    def test_job_progress_sums_the_language_jobs_of_one_request(self) -> None:
        from backend.genesis_whisper_server_batching import WhisperBatchManager

        manager = WhisperBatchManager.__new__(WhisperBatchManager)
        manager._jobs = {
            "a": _FakeJob(PID, 3, 1, True),
            "b": _FakeJob(f"{PID}#1", 2, 0, False),
            "c": _FakeJob("someone-else", 9, 9, True),
        }
        self.assertEqual(manager.job_progress(PID), (6, 2, True))
        self.assertEqual(manager.job_progress("nobody"), (0, 0, False))

    def test_probe_before_registration_while_running_and_after_the_jobs_are_gone(self) -> None:
        registry, _ = _registry()
        stats = {"value": (0, 0, False)}
        batch_manager = types.SimpleNamespace(job_progress=lambda _pid: stats["value"])
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(
            whisper_batch_manager=batch_manager)))
        entry = registry.start(PID, None)
        registry.set_mode(entry, "diarization")
        registry.set_audio_seconds(entry, 100.0)
        with mock.patch.object(progress, "registry", registry), progress.tracking(entry):
            self.assertEqual(v2._enter_asr_phase(request, 20), PID)
        before = registry.snapshot(entry)
        self.assertEqual((before["phase_done"], before["state"]), (0, "waiting"))
        stats["value"] = (15, 1, True)
        self.assertEqual(registry.snapshot(entry)["phase_done"], 5)
        stats["value"] = (0, 0, False)
        self.assertEqual(registry.snapshot(entry)["phase_done"], 20)

    def test_no_tracked_request_keeps_the_fresh_job_tokens(self) -> None:
        self.assertIsNone(v2._enter_asr_phase(types.SimpleNamespace(), 5))


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _CapturingApp()
        v2.create_v2_api(self.app)
        self.post = self.app.routes[("POST", "/v2/audio/process")]
        self.get = self.app.routes[("GET", "/v2/progress/{progress_id}")]
        self.registry, self.clock = _registry()
        self.registry_patch = mock.patch.object(progress, "registry", self.registry)
        self.registry_patch.start()
        self.audio = np.full(16000 * 60, 0.1, dtype=np.float32)

    def tearDown(self) -> None:
        self.registry_patch.stop()

    def _poll(self, progress_id: str = PID):
        return asyncio.run(self.get(progress_id, _request(self.app)))

    def _post(self, headers: dict[str, str], diarization):
        upload = _upload()
        raw_request = json.dumps({"schema_version": "2.0", "mode": "diarization", "diarization": {}})
        form = FormData([("file", upload), ("request", raw_request)])
        with (
            mock.patch.object(
                v2, "_parse_multipart_parts", new=mock.AsyncMock(return_value=(form, upload, raw_request))
            ),
            mock.patch.object(v2, "authorize_api_key", return_value="key-1"),
            mock.patch.object(v2, "load_audio_file", return_value=self.audio),
            mock.patch.object(v2, "_record_log"),
            mock.patch.object(v2.get_auth_store(), "record_api_key_usage"),
            mock.patch.object(v2, "_process_diarization", new=diarization),
        ):
            return asyncio.run(self.post(_request(self.app, headers)))

    def _diarization_result(self):
        return (
            {"transcript": {"text": "", "segments": []}, "speaker_counts": {}, "speaker_assignments": [],
             "unknown_speakers": [], "unresolved_known_speakers": []},
            {"diarization": 1},
            {"asr": {"id": "asr"}},
            [],
            "",
        )

    def test_a_tracked_request_is_pollable_while_it_runs_and_reports_completion(self) -> None:
        observed: dict = {}

        async def diarization(*_args, **_kwargs):
            progress.enter_phase("speaker_windows", total=8)
            progress.add_units(done=2)
            with mock.patch.object(v2, "authorize_api_key", return_value="key-1"):
                observed["own"] = await self.get(PID, _request(self.app))
            with mock.patch.object(v2, "authorize_api_key", return_value="key-2"):
                observed["foreign"] = await self.get(PID, _request(self.app))
            return self._diarization_result()

        response = self._post({"X-Progress-ID": PID}, diarization)
        self.assertEqual(response.status_code, 200)
        own = _payload(observed["own"])
        self.assertEqual(observed["own"].status_code, 200)
        self.assertEqual((own["state"], own["mode"], own["phase"]), ("running", "diarization", "speaker_windows"))
        self.assertEqual((own["phase_done"], own["phase_total"], own["audio_duration_ms"]), (2, 8, 60000))
        self.assertEqual(set(own), {"schema_version", "progress_id", "state", "mode", "phase", "phase_index",
                                    "phase_count", "phase_done", "phase_total", "progress", "elapsed_ms",
                                    "eta_ms", "audio_duration_ms"})
        self.assertEqual(observed["foreign"].status_code, 404)
        with mock.patch.object(v2, "authorize_api_key", return_value="key-1"):
            done = _payload(self._poll())
        self.assertEqual((done["state"], done["progress"], done["eta_ms"]), ("completed", 1.0, 0))

    def test_a_failed_request_is_reported_failed(self) -> None:
        response = self._post({"X-Progress-ID": PID}, mock.AsyncMock(side_effect=RuntimeError("down")))
        self.assertEqual(response.status_code, 503)
        with mock.patch.object(v2, "authorize_api_key", return_value="key-1"):
            self.assertEqual(_payload(self._poll())["state"], "failed")

    def test_without_or_with_a_malformed_id_nothing_is_tracked_and_the_request_works(self) -> None:
        for headers in ({}, {"X-Progress-ID": "bad id"}):
            response = self._post(headers, mock.AsyncMock(return_value=self._diarization_result()))
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.registry._entries), 0)

    def test_poll_with_a_refused_key_is_401(self) -> None:
        from fastapi import HTTPException

        with mock.patch.object(v2, "authorize_api_key", side_effect=HTTPException(status_code=401, detail="no")):
            response = self._poll()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(_payload(response)["error"]["code"], "INVALID_API_KEY")


class AppRoutingTests(unittest.TestCase):
    """The real app: capabilities announce progress, the GET path is routed (not the static catch-all)."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="g3-whisper-progress-")
        self.store = auth.AuthStore(Path(self.temp_dir.name))
        self.store_patch = mock.patch.object(auth, "_auth_store", self.store)
        self.store_patch.start()

    def tearDown(self) -> None:
        self.store.flush()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def test_capabilities_and_route(self) -> None:
        from backend import genesis_whisper_server as server

        client = TestClient(server.app)
        capability = client.get("/v2/capabilities").json()["progress"]
        self.assertEqual(capability["supported"], True)
        self.assertEqual(capability["header"], "X-Progress-ID")
        missing = client.get(f"/v2/progress/{PID}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "PROGRESS_NOT_FOUND")
        created = self.store.create_api_key("client")
        self.assertEqual(client.get(f"/v2/progress/{PID}").status_code, 401)
        ok = client.get(f"/v2/progress/{PID}", headers={"Authorization": f"Bearer {created['token']}"})
        self.assertEqual(ok.status_code, 404)


if __name__ == "__main__":
    unittest.main()
