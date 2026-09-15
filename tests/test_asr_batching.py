from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as legacy_api
from backend import genesis_whisper_server_batching as batching
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend import genesis_whisper_server_wxc as wxc
from backend.genesis_whisper_server_batching import WhisperBatchManager
from backend.genesis_whisper_server_chunking import split_audio_for_whisper
from backend.genesis_whisper_server_globals import (
    batch_history,
    batch_runtime_state,
    batch_state_lock,
    current_settings,
    job_history,
    settings_lock,
)


KEY = ("model", "cpu", "cache", "de", "fp32")
OTHER_KEY = ("model", "cpu", "cache", "en", "fp32")

SCHEDULER_TEST_SETTINGS = {
    "batch_wait_time_ms": 0,
    "batch_max_segments": 16,
    "batch_max_audio_seconds": 1000.0,
    "scheduler_long_job_min_chunks": 5,
    "scheduler_max_parallel_long_jobs": 0,
    "scheduler_first_chunk_fast_path": True,
    "cuda_memory_trim_after_batch": False,
}


class _Scenario:
    """Owns the chunks of every simulated request and gates each GPU batch.

    ``gpu_phase`` replaces ``run_blocking_gpu_phase`` so a test can inspect
    which requests share a batch and decide exactly when that batch finishes.
    """

    def __init__(self, *, manual: bool = True, failing_owner: str | None = None) -> None:
        self.owner_by_id: dict[int, tuple[str, int]] = {}
        self.batches: list[list[str]] = []
        self.manual = manual
        self.failing_owner = failing_owner
        self._gate = asyncio.Semaphore(0)

    def chunks(self, owner: str, count: int, seconds: float = 1.0) -> list[np.ndarray]:
        out = []
        for index in range(count):
            chunk = np.full(int(16000 * seconds), 0.1, dtype=np.float32)
            self.owner_by_id[id(chunk)] = (owner, index)
            out.append(chunk)
        return out

    def process(self, audio_batch, _processing_key) -> list[str]:
        owners = [self.owner_by_id[id(chunk)] for chunk in audio_batch]
        if self.failing_owner is not None and any(owner == self.failing_owner for owner, _ in owners):
            raise RuntimeError("simulierter Modellfehler")
        return [f"{owner}:{index}" for owner, index in owners]

    async def gpu_phase(self, function, audio_batch, processing_key):
        self.batches.append([self.owner_by_id[id(chunk)][0] for chunk in audio_batch])
        if self.manual:
            await self._gate.acquire()
        return function(audio_batch, processing_key)

    def release(self) -> None:
        self._gate.release()

    async def wait_for_batches(self, count: int) -> None:
        async def _poll():
            while len(self.batches) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(_poll(), timeout=5.0)

    async def drain(self, *tasks: asyncio.Task) -> None:
        async def _poll():
            while not all(task.done() for task in tasks):
                self.release()
                await asyncio.sleep(0.001)

        await asyncio.wait_for(_poll(), timeout=5.0)


class _SchedulerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._previous_settings = current_settings.copy()
        with settings_lock:
            current_settings.clear()
            current_settings.update(SCHEDULER_TEST_SETTINGS)
        batch_history.clear()
        job_history.clear()
        self.scenario = _Scenario()
        self.manager = WhisperBatchManager(self.scenario.process, asyncio.Lock())
        self._gpu_patch = mock.patch.object(batching, "run_blocking_gpu_phase", new=self.scenario.gpu_phase)
        self._gpu_patch.start()
        await self.manager.start()

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        self._gpu_patch.stop()
        with settings_lock:
            current_settings.clear()
            current_settings.update(self._previous_settings)

    def set_settings(self, **values) -> None:
        with settings_lock:
            current_settings.update(values)

    def submit(self, owner: str, count: int, key=KEY, seconds: float = 1.0) -> asyncio.Task:
        chunks = self.scenario.chunks(owner, count, seconds)
        return asyncio.create_task(self.manager.submit_job(chunks, owner, key))

    @staticmethod
    async def settle() -> None:
        # Let freshly created submit tasks register their jobs and the worker plan.
        for _ in range(3):
            await asyncio.sleep(0)


class FairSchedulingTests(_SchedulerTestCase):
    async def test_short_job_gets_its_own_batch_ahead_of_the_long_backlog(self) -> None:
        long_task = self.submit("long", 40)
        await self.scenario.wait_for_batches(1)
        self.assertEqual(self.scenario.batches[0], ["long"] * 16)

        short_task = self.submit("short", 1)
        await self.settle()
        self.scenario.release()

        # The next batch carries only the newcomer, not the 24 remaining long chunks.
        await self.scenario.wait_for_batches(2)
        self.assertEqual(self.scenario.batches[1], ["short"])
        self.scenario.release()
        short_results = await asyncio.wait_for(short_task, timeout=5.0)
        self.assertEqual([item.text for item in short_results], ["short:0"])
        self.assertFalse(long_task.done())

        await self.scenario.drain(long_task)
        long_results = await long_task
        self.assertEqual([item.text for item in long_results], [f"long:{index}" for index in range(40)])
        self.assertTrue(all(owner == "long" for owner in self.scenario.batches[2]))
        fast_flags = [entry["fast_path"] for entry in reversed(batch_history)]
        self.assertEqual(fast_flags[:3], [False, True, False])
        finished = {entry["request_id"]: entry for entry in job_history}
        self.assertEqual(finished["short"]["batch_count"], 1)
        self.assertEqual(finished["long"]["total_chunks"], 40)

    async def test_round_robin_shares_every_batch_between_active_jobs(self) -> None:
        self.set_settings(scheduler_first_chunk_fast_path=False, batch_max_segments=4)
        self.scenario.manual = False
        a_task = self.submit("a", 10)
        b_task = self.submit("b", 10)
        await self.scenario.drain(a_task, b_task)

        self.assertEqual(self.scenario.batches[0], ["a", "b", "a", "b"])
        self.assertEqual([item.text for item in await a_task], [f"a:{index}" for index in range(10)])
        self.assertEqual([item.text for item in await b_task], [f"b:{index}" for index in range(10)])
        self.assertTrue(all({"a", "b"} == set(batch) for batch in self.scenario.batches))

    async def test_fast_path_alternates_so_the_long_job_keeps_progressing(self) -> None:
        long_task = self.submit("long", 40)
        await self.scenario.wait_for_batches(1)
        first_short = self.submit("s1", 1)
        await self.settle()
        self.scenario.release()
        await self.scenario.wait_for_batches(2)
        self.assertEqual(self.scenario.batches[1], ["s1"])

        # A second newcomer arriving right after a fast-path batch joins a regular
        # round-robin batch instead of triggering another exclusive one.
        second_short = self.submit("s2", 1)
        await self.settle()
        self.scenario.release()
        await self.scenario.wait_for_batches(3)
        third = self.scenario.batches[2]
        self.assertEqual(len(third), 16)
        self.assertEqual(third[1], "s2")
        self.assertEqual(third.count("long"), 15)

        await self.scenario.drain(long_task, first_short, second_short)
        self.assertEqual([item.text for item in await second_short], ["s2:0"])
        self.assertEqual(len(await long_task), 40)

    async def test_long_jobs_wait_for_a_free_slot_while_short_jobs_bypass(self) -> None:
        self.set_settings(scheduler_max_parallel_long_jobs=1)
        a_task = self.submit("a", 8)
        b_task = self.submit("b", 8)
        short_task = self.submit("short", 1)
        await self.scenario.wait_for_batches(1)

        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["active_jobs"], 2)
        self.assertEqual(snapshot["waiting_jobs"], 1)
        states = {job["request_id"]: job["state"] for job in snapshot["jobs"]}
        self.assertEqual(states, {"a": "active", "short": "active", "b": "waiting"})
        self.assertEqual(snapshot["pending_buffer_size"], 8)
        self.assertNotIn("b", self.scenario.batches[0])
        self.assertIn("short", self.scenario.batches[0])

        await self.scenario.drain(a_task, b_task, short_task)
        self.assertEqual([item.text for item in await b_task], [f"b:{index}" for index in range(8)])
        first_b_batch = next(index for index, batch in enumerate(self.scenario.batches) if "b" in batch)
        self.assertTrue(all("a" not in batch for batch in self.scenario.batches[first_b_batch:]))
        self.assertEqual(self.manager.snapshot()["waiting_jobs"], 0)

    async def test_jobs_with_different_processing_keys_alternate_but_never_share(self) -> None:
        self.set_settings(scheduler_first_chunk_fast_path=False, batch_max_segments=2)
        self.scenario.manual = False
        a_task = self.submit("a", 3, key=KEY)
        b_task = self.submit("b", 3, key=OTHER_KEY)
        await self.scenario.drain(a_task, b_task)

        self.assertEqual(self.scenario.batches, [["a", "a"], ["b", "b"], ["a"], ["b"]])
        self.assertEqual(len(await a_task), 3)
        self.assertEqual(len(await b_task), 3)

    async def test_batch_wait_lets_simultaneous_newcomers_share_one_batch(self) -> None:
        self.set_settings(batch_wait_time_ms=150)
        self.scenario.manual = False
        first = self.submit("s1", 1)
        await self.settle()
        await asyncio.sleep(0.02)
        second = self.submit("s2", 1)
        await self.scenario.drain(first, second)

        self.assertEqual(self.scenario.batches[0], ["s1", "s2"])

    async def test_audio_budget_limits_a_batch_but_always_admits_one_chunk(self) -> None:
        self.set_settings(batch_max_audio_seconds=2.5, scheduler_first_chunk_fast_path=False)
        self.scenario.manual = False
        task = self.submit("a", 5, seconds=1.0)
        await self.scenario.drain(task)

        self.assertEqual([len(batch) for batch in self.scenario.batches], [2, 2, 1])
        self.assertEqual(len(await task), 5)


class CancellationAndFailureTests(_SchedulerTestCase):
    async def test_cancelled_job_is_withdrawn_before_the_gpu_phase(self) -> None:
        long_task = self.submit("long", 20)
        await self.scenario.wait_for_batches(1)
        short_task = self.submit("short", 1)
        await self.settle()
        short_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await short_task

        self.scenario.release()
        await self.scenario.drain(long_task)
        self.assertTrue(all("short" not in batch for batch in self.scenario.batches))
        cancelled = next(entry for entry in job_history if entry["request_id"] == "short")
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(len(await long_task), 20)

    async def test_cancelling_a_job_with_inflight_chunks_keeps_the_worker_healthy(self) -> None:
        long_task = self.submit("long", 20)
        await self.scenario.wait_for_batches(1)
        long_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await long_task
        self.scenario.release()
        await asyncio.sleep(0.01)

        self.assertEqual(self.manager.snapshot()["jobs"], [])
        self.assertEqual(self.manager.snapshot()["queue_size"], 0)
        with batch_state_lock:
            self.assertTrue(batch_runtime_state["worker_running"])

        follow_up = self.submit("next", 1)
        await self.scenario.drain(follow_up)
        self.assertEqual([item.text for item in await follow_up], ["next:0"])

    async def test_batch_failure_fails_only_the_involved_jobs(self) -> None:
        self.scenario.failing_owner = "bad"
        self.scenario.manual = False
        good_task = self.submit("good", 2, key=KEY)
        bad_task = self.submit("bad", 1, key=OTHER_KEY)
        await self.scenario.drain(good_task, bad_task)

        self.assertEqual([item.text for item in await good_task], ["good:0", "good:1"])
        with self.assertRaisesRegex(RuntimeError, "simulierter Modellfehler"):
            await bad_task
        statuses = {entry["request_id"]: entry["status"] for entry in job_history}
        self.assertEqual(statuses, {"good": "ok", "bad": "error"})
        self.assertIn("error", {entry["status"] for entry in batch_history})
        self.assertEqual(self.manager.snapshot()["jobs"], [])

    async def test_stop_fails_outstanding_jobs_instead_of_hanging_them(self) -> None:
        task = self.submit("pending", 3)
        await self.scenario.wait_for_batches(1)
        await self.manager.stop()

        with self.assertRaisesRegex(RuntimeError, "gestoppt"):
            await task
        with self.assertRaisesRegex(RuntimeError, "laeuft nicht"):
            await self.manager.submit_job(self.scenario.chunks("late", 1), "late", KEY)


class SchedulerSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._previous_settings = current_settings.copy()

    def tearDown(self) -> None:
        with settings_lock:
            current_settings.clear()
            current_settings.update(self._previous_settings)

    async def test_idle_cuda_trim_is_enabled_by_default(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings.clear()
        with mock.patch.object(manager, "_trim_cuda_cache") as trim:
            trimmed = await manager._trim_cuda_cache_if_enabled()

        self.assertTrue(trimmed)
        trim.assert_called_once_with()

    async def test_idle_cuda_trim_can_be_disabled_explicitly(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings["cuda_memory_trim_after_batch"] = False
        with mock.patch.object(manager, "_trim_cuda_cache") as trim:
            trimmed = await manager._trim_cuda_cache_if_enabled()

        self.assertFalse(trimmed)
        trim.assert_not_called()

    async def test_limit_fallbacks_match_the_shipped_defaults(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings.clear()

        limits = manager._get_limits()
        self.assertEqual(storage.DEFAULT_SETTINGS["batch_max_segments"], 16)
        self.assertTrue(storage.DEFAULT_SETTINGS["cuda_memory_trim_after_batch"])
        self.assertEqual(limits["max_segments"], 16)
        self.assertEqual(limits["long_job_min_chunks"], storage.DEFAULT_SETTINGS["scheduler_long_job_min_chunks"])
        self.assertEqual(
            limits["max_parallel_long_jobs"],
            storage.DEFAULT_SETTINGS["scheduler_max_parallel_long_jobs"],
        )
        self.assertTrue(limits["first_chunk_fast_path"])

    def test_normalize_settings_clamps_scheduler_values(self) -> None:
        normalized = storage.normalize_settings(
            {
                "scheduler_long_job_min_chunks": 0,
                "scheduler_max_parallel_long_jobs": -3,
                "scheduler_first_chunk_fast_path": "yes",
            }
        )

        self.assertEqual(normalized["scheduler_long_job_min_chunks"], 1)
        self.assertEqual(normalized["scheduler_max_parallel_long_jobs"], 0)
        self.assertTrue(normalized["scheduler_first_chunk_fast_path"])
        self.assertFalse(
            storage.normalize_settings({"scheduler_first_chunk_fast_path": False})["scheduler_first_chunk_fast_path"]
        )


class LongAudioSuperchunkTests(unittest.TestCase):
    def test_short_audio_stays_one_item_without_a_vad_pass(self) -> None:
        audio = np.full(16000 * 30, 0.1, dtype=np.float32)

        with mock.patch.object(wxc, "silero_frame_probs", side_effect=AssertionError("VAD must not run")):
            chunks = wxc.split_audio_into_superchunks(audio)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(np.shares_memory(chunks[0], audio))

    def test_long_audio_is_cut_at_speech_pauses_into_scheduler_visible_chunks(self) -> None:
        audio = np.full(16000 * 100, 0.1, dtype=np.float32)
        regions = [(float(start), float(start + 9)) for start in range(0, 100, 10)]

        with (
            mock.patch.object(wxc, "silero_frame_probs", return_value=np.ones(10, dtype=np.float32)),
            mock.patch.object(wxc, "speech_regions_from_probs", return_value=regions),
        ):
            chunks = wxc.split_audio_into_superchunks(audio)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 16000 * (wxc.SUPERCHUNK_TARGET_S + 2 * wxc.CHUNK_PAD_S) + 1 for chunk in chunks))
        self.assertTrue(all(np.shares_memory(chunk, audio) for chunk in chunks))

    def test_long_audio_without_speech_regions_is_kept_whole(self) -> None:
        audio = np.zeros(16000 * 100, dtype=np.float32)

        with (
            mock.patch.object(wxc, "silero_frame_probs", return_value=np.zeros(10, dtype=np.float32)),
            mock.patch.object(wxc, "speech_regions_from_probs", return_value=[]),
        ):
            chunks = wxc.split_audio_into_superchunks(audio)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), len(audio))


class SharedSchedulerPathTests(unittest.TestCase):
    def _legacy_app(self, batch_manager):
        app = FastAPI()
        app.state.whisper_batch_manager = batch_manager
        app.state.local_gpu_lock = asyncio.Lock()
        legacy_api.create_api(app)
        return app

    def test_legacy_transcribe_submits_whisper_chunks_as_one_job(self) -> None:
        audio = np.full(16000 * 65, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:32000], audio[32000:]]
        results = [SimpleNamespace(text=f"Teil {index}", batch_id="batch") for index in range(len(segments))]
        batch_manager = SimpleNamespace(submit_job=mock.AsyncMock(return_value=results))
        app = self._legacy_app(batch_manager)
        settings = {
            "local_model": "openai/whisper-tiny",
            "local_gpu_device": "cpu",
            "local_model_cache_path": "",
            "transcription_language": "de",
            "local_model_precision": "fp32",
        }

        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=audio),
            mock.patch.object(legacy_api, "split_audio_for_whisper", return_value=segments),
            mock.patch.object(legacy_api, "log_transcription"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/transcribe/",
                    files={"file": ("long.wav", b"fixture", "audio/wav")},
                    data={"engine": "local"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["transcription"], "Teil 0 Teil 1 Teil 2")
        batch_manager.submit_job.assert_awaited_once()
        self.assertIs(batch_manager.submit_job.await_args.args[0], segments)

    def test_legacy_transcribe_splits_long_cohere_audio_into_superchunks(self) -> None:
        audio = np.full(16000 * 90, 0.1, dtype=np.float32)
        segments = [audio[: 16000 * 45], audio[16000 * 45 :]]
        results = [SimpleNamespace(text=f"Teil {index}", batch_id="batch") for index in range(len(segments))]
        batch_manager = SimpleNamespace(submit_job=mock.AsyncMock(return_value=results))
        app = self._legacy_app(batch_manager)
        settings = {
            "local_model": "CohereLabs/cohere-transcribe-03-2026",
            "local_gpu_device": "cpu",
            "local_model_cache_path": "",
            "transcription_language": "de",
            "local_model_precision": "fp32",
        }

        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=audio),
            mock.patch.object(legacy_api, "split_audio_into_superchunks", return_value=segments) as splitter,
            mock.patch.object(legacy_api, "log_transcription") as log_transcription,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/transcribe/",
                    files={"file": ("long.wav", b"fixture", "audio/wav")},
                    data={"engine": "local"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["transcription"], "Teil 0 Teil 1")
        self.assertEqual(log_transcription.call_args.args[0]["segment_count"], 2)
        splitter.assert_called_once()
        self.assertIs(batch_manager.submit_job.await_args.args[0], segments)


class SharedSchedulerAsyncPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_v2_transcript_submits_whisper_chunks_as_one_job(self) -> None:
        audio = np.full(16000 * 65, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:32000], audio[32000:]]
        results = [SimpleNamespace(text=f"V2 {index}", batch_id="batch") for index in range(len(segments))]
        manager = SimpleNamespace(submit_job=mock.AsyncMock(return_value=results))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(whisper_batch_manager=manager)))
        processing_key = ("openai/whisper-tiny", "cpu", "", "de", "fp32")

        with (
            mock.patch.object(v2, "_processing_key", return_value=processing_key),
            mock.patch.object(v2, "uses_cohere_backend", return_value=False),
            mock.patch.object(v2, "split_audio_for_whisper", return_value=segments),
        ):
            text, _, segment_count, model_id = await v2._transcribe_audio(request, audio)

        self.assertEqual(text, "V2 0 V2 1 V2 2")
        self.assertEqual(segment_count, 3)
        self.assertEqual(model_id, "openai/whisper-tiny")
        manager.submit_job.assert_awaited_once()
        self.assertIs(manager.submit_job.await_args.args[0], segments)

    async def test_v2_transcript_splits_long_cohere_audio_into_superchunks(self) -> None:
        audio = np.full(16000 * 90, 0.1, dtype=np.float32)
        segments = [audio[: 16000 * 45], audio[16000 * 45 :]]
        results = [SimpleNamespace(text=f"C {index}", batch_id="batch") for index in range(len(segments))]
        manager = SimpleNamespace(submit_job=mock.AsyncMock(return_value=results))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(whisper_batch_manager=manager)))
        processing_key = ("CohereLabs/cohere-transcribe-03-2026", "cpu", "", "de", "fp16")

        with (
            mock.patch.object(v2, "_processing_key", return_value=processing_key),
            mock.patch.object(v2, "split_audio_into_superchunks", return_value=segments) as splitter,
        ):
            text, _, segment_count, _ = await v2._transcribe_audio(request, audio)

        self.assertEqual(text, "C 0 C 1")
        self.assertEqual(segment_count, 2)
        splitter.assert_called_once()
        self.assertIs(manager.submit_job.await_args.args[0], segments)

    async def test_admin_benchmark_submits_every_repeat_as_its_own_job(self) -> None:
        audio = np.full(32000, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:]]

        async def submit_job(audio_segments, request_id, processing_key):
            return [SimpleNamespace(text=text, batch_id=f"batch-{request_id}") for text in ("A", "B")]

        manager = SimpleNamespace(submit_job=mock.AsyncMock(side_effect=submit_job))
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    local_gpu_lock=asyncio.Lock(),
                    whisper_batch_manager=manager,
                )
            ),
            headers={},
        )
        processing_key = ("openai/whisper-tiny", "cpu", "", "de", "fp32")

        with (
            mock.patch.object(admin, "_get_local_processing_key", return_value=processing_key),
            mock.patch.object(admin, "run_blocking_gpu_phase", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(admin, "_get_loaded_model_cuda_index", return_value=None),
            mock.patch.object(admin, "_reset_peak_vram_tracking"),
            mock.patch.object(
                admin,
                "_read_peak_vram_metrics",
                return_value={"peak_vram_reserved_mb": None, "peak_vram_allocated_mb": None},
            ),
            mock.patch.object(admin, "get_audio_duration_seconds", return_value=2.0),
            mock.patch.object(admin, "uses_cohere_backend", return_value=False),
            mock.patch.object(admin, "split_audio_for_whisper", return_value=segments),
            mock.patch.object(admin, "repetition_filter_enabled", return_value=False),
        ):
            response = await admin._run_admin_benchmark(request, audio, repeat_count=3)

        self.assertEqual(response["transcript"], "A B")
        self.assertTrue(response["transcripts_match"])
        self.assertEqual(response["total_chunks"], 6)
        self.assertEqual(response["batches_used"], 3)
        self.assertEqual(manager.submit_job.await_count, 3)
        request_ids = [call.args[1] for call in manager.submit_job.await_args_list]
        self.assertEqual(len(set(request_ids)), 3)
        self.assertTrue(all(call.args[0] is segments for call in manager.submit_job.await_args_list))


class WhisperChunkMemoryTests(unittest.TestCase):
    def test_split_returns_views_instead_of_materializing_the_recording_twice(self) -> None:
        audio = np.linspace(-0.25, 0.25, 40000, dtype=np.float32)

        with mock.patch(
            "backend.genesis_whisper_server_chunking._detect_speech_segments",
            return_value=[(0, len(audio))],
        ):
            chunks = split_audio_for_whisper(
                audio,
                max_chunk_seconds=1.0,
                overlap_ms=0,
            )

        self.assertEqual([len(chunk) for chunk in chunks], [16000, 16000, 8000])
        self.assertTrue(all(np.shares_memory(chunk, audio) for chunk in chunks))

    def test_short_recording_is_also_returned_without_a_pcm_copy(self) -> None:
        audio = np.ones(8000, dtype=np.float32)

        chunks = split_audio_for_whisper(audio)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(np.shares_memory(chunks[0], audio))


if __name__ == "__main__":
    unittest.main()
