from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as legacy_api
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_batching import (
    QueuedWhisperSegment,
    WhisperBatchManager,
    enqueue_audio_segments_bounded,
    get_asr_enqueue_in_flight_limit,
)
from backend.genesis_whisper_server_chunking import split_audio_for_whisper
from backend.genesis_whisper_server_globals import (
    batch_history,
    batch_runtime_state,
    batch_state_lock,
    current_settings,
    settings_lock,
)


class _TrackingBatchManager:
    def __init__(self, *, block: bool = False) -> None:
        self.active = 0
        self.max_active = 0
        self.started = 0
        self._block = block
        self.release = asyncio.Event()

    async def enqueue(self, **kwargs):
        self.active += 1
        self.started += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._block:
                await self.release.wait()
            else:
                # Complete out of order to verify the helper restores segment order.
                await asyncio.sleep((kwargs["segment_index"] % 5) * 0.0005)
            index = kwargs["segment_index"]
            return SimpleNamespace(text=f"chunk-{index}", batch_id=f"batch-{index // 4}")
        finally:
            self.active -= 1


def _queued_item(loop: asyncio.AbstractEventLoop, index: int) -> QueuedWhisperSegment:
    return QueuedWhisperSegment(
        request_id="request",
        segment_index=index,
        total_segments=2,
        audio_data=np.zeros(1600, dtype=np.float32),
        processing_key=("model", "cpu", "cache", "de", "fp32"),
        future=loop.create_future(),
        queued_at_monotonic=time.monotonic(),
    )


class BoundedV2AsrEnqueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with settings_lock:
            self._previous_batch_size = current_settings.get("batch_max_segments")
            current_settings["batch_max_segments"] = 3

    def tearDown(self) -> None:
        with settings_lock:
            if self._previous_batch_size is None:
                current_settings.pop("batch_max_segments", None)
            else:
                current_settings["batch_max_segments"] = self._previous_batch_size

    async def test_limits_tasks_and_preserves_result_order(self) -> None:
        manager = _TrackingBatchManager()
        chunks = [np.zeros(160, dtype=np.float32) for _ in range(80)]
        expected_limit = get_asr_enqueue_in_flight_limit()

        results = await enqueue_audio_segments_bounded(
            manager,
            chunks,
            "request",
            ("model", "cpu", "cache", "de", "fp32"),
        )

        self.assertEqual([item.text for item in results], [f"chunk-{index}" for index in range(80)])
        self.assertGreater(manager.max_active, 1)
        self.assertLessEqual(manager.max_active, expected_limit)

    async def test_cancellation_cleans_up_bounded_child_tasks(self) -> None:
        manager = _TrackingBatchManager(block=True)
        chunks = [np.zeros(160, dtype=np.float32) for _ in range(1000)]
        expected_limit = get_asr_enqueue_in_flight_limit()
        processing = asyncio.create_task(
            enqueue_audio_segments_bounded(
                manager,
                chunks,
                "request",
                ("model", "cpu", "cache", "de", "fp32"),
            )
        )

        while manager.started < expected_limit:
            await asyncio.sleep(0)
        processing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await processing

        self.assertEqual(manager.started, expected_limit)
        self.assertEqual(manager.active, 0)


class BatchManagerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_next_item_skips_cancelled_queue_entries(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        loop = asyncio.get_running_loop()
        cancelled = _queued_item(loop, 0)
        live = _queued_item(loop, 1)
        cancelled.future.cancel()
        await manager._queue.put(cancelled)
        await manager._queue.put(live)

        selected = await manager._get_next_item()

        self.assertIs(selected, live)

    async def test_process_batch_drops_items_cancelled_before_gpu_phase(self) -> None:
        processed_batch_sizes: list[int] = []

        def process_batch(audio_batch, _processing_key):
            processed_batch_sizes.append(len(audio_batch))
            return ["ok"] * len(audio_batch)

        manager = WhisperBatchManager(process_batch, asyncio.Lock())
        loop = asyncio.get_running_loop()
        cancelled = _queued_item(loop, 0)
        live = _queued_item(loop, 1)
        cancelled.future.cancel()

        await manager._process_batch([cancelled, live])

        self.assertEqual(processed_batch_sizes, [1])
        self.assertEqual(live.future.result().text, "ok")

    async def test_process_batch_refilters_cancellation_after_waiting_for_gpu_lock(self) -> None:
        processed_batch_sizes: list[int] = []

        def process_batch(audio_batch, _processing_key):
            processed_batch_sizes.append(len(audio_batch))
            return ["ok"] * len(audio_batch)

        gpu_lock = asyncio.Lock()
        await gpu_lock.acquire()
        manager = WhisperBatchManager(process_batch, gpu_lock)
        loop = asyncio.get_running_loop()
        cancelled_while_waiting = _queued_item(loop, 0)
        live = _queued_item(loop, 1)
        processing = asyncio.create_task(
            manager._process_batch([cancelled_while_waiting, live])
        )

        # Seeing an active id proves the first filter already ran and the batch
        # is now waiting for the deliberately held GPU lock.
        while True:
            with batch_state_lock:
                active_batch_id = batch_runtime_state["active_batch_id"]
            if active_batch_id:
                break
            await asyncio.sleep(0)

        cancelled_while_waiting.future.cancel()
        gpu_lock.release()
        await processing

        self.assertEqual(processed_batch_sizes, [1])
        self.assertEqual(live.future.result().text, "ok")
        self.assertEqual(batch_history[0]["batch_id"], active_batch_id)
        self.assertEqual(batch_history[0]["batch_size"], 1)
        self.assertEqual(batch_history[0]["audio_seconds"], 0.1)

    async def test_process_batch_skips_gpu_when_every_item_cancels_during_lock_wait(self) -> None:
        process_batch = mock.Mock(return_value=[])
        gpu_lock = asyncio.Lock()
        await gpu_lock.acquire()
        manager = WhisperBatchManager(process_batch, gpu_lock)
        loop = asyncio.get_running_loop()
        first = _queued_item(loop, 0)
        second = _queued_item(loop, 1)
        processing = asyncio.create_task(manager._process_batch([first, second]))

        while True:
            with batch_state_lock:
                active_batch_id = batch_runtime_state["active_batch_id"]
            if active_batch_id:
                break
            await asyncio.sleep(0)

        first.future.cancel()
        second.future.cancel()
        gpu_lock.release()
        await processing

        process_batch.assert_not_called()
        self.assertFalse(any(entry["batch_id"] == active_batch_id for entry in batch_history))
        with batch_state_lock:
            self.assertIsNone(batch_runtime_state["active_batch_id"])


class CudaCacheTrimSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._previous_settings = current_settings.copy()

    def tearDown(self) -> None:
        with settings_lock:
            current_settings.clear()
            current_settings.update(self._previous_settings)

    async def test_idle_cuda_trim_is_disabled_by_default(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings.clear()
        with mock.patch.object(manager, "_trim_cuda_cache") as trim:
            trimmed = await manager._trim_cuda_cache_if_enabled()

        self.assertFalse(trimmed)
        trim.assert_not_called()

    async def test_idle_cuda_trim_runs_only_when_explicitly_enabled(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings["cuda_memory_trim_after_batch"] = True
        with mock.patch.object(manager, "_trim_cuda_cache") as trim:
            trimmed = await manager._trim_cuda_cache_if_enabled()

        self.assertTrue(trimmed)
        trim.assert_called_once_with()

    async def test_batch_fallbacks_match_the_16_segment_default(self) -> None:
        manager = WhisperBatchManager(lambda audio, key: [""] * len(audio), asyncio.Lock())
        with settings_lock:
            current_settings.clear()

        self.assertEqual(storage.DEFAULT_SETTINGS["batch_max_segments"], 16)
        self.assertFalse(storage.DEFAULT_SETTINGS["cuda_memory_trim_after_batch"])
        self.assertEqual(manager._get_limits()["max_segments"], 16)
        self.assertEqual(get_asr_enqueue_in_flight_limit(), 32)


class SharedSchedulerPathTests(unittest.TestCase):
    def test_legacy_transcribe_uses_shared_bounded_scheduler(self) -> None:
        app = FastAPI()
        app.state.whisper_batch_manager = SimpleNamespace()
        app.state.local_gpu_lock = asyncio.Lock()
        legacy_api.create_api(app)
        audio = np.full(16000 * 65, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:32000], audio[32000:]]
        results = [
            SimpleNamespace(text=f"Teil {index}", batch_id="batch")
            for index in range(len(segments))
        ]
        settings = {
            "local_model": "openai/whisper-tiny",
            "local_gpu_device": "cpu",
            "local_model_cache_path": "",
            "transcription_language": "de",
            "local_model_precision": "fp32",
        }
        bounded = mock.AsyncMock(return_value=results)

        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=audio),
            mock.patch.object(legacy_api, "split_audio_for_whisper", return_value=segments),
            mock.patch.object(legacy_api, "enqueue_audio_segments_bounded", new=bounded),
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
        bounded.assert_awaited_once()
        self.assertIs(bounded.await_args.args[1], segments)


class SharedSchedulerAsyncPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_v2_transcript_uses_shared_bounded_scheduler(self) -> None:
        audio = np.full(16000 * 65, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:32000], audio[32000:]]
        results = [
            SimpleNamespace(text=f"V2 {index}", batch_id="batch")
            for index in range(len(segments))
        ]
        manager = SimpleNamespace()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(whisper_batch_manager=manager)))
        processing_key = ("openai/whisper-tiny", "cpu", "", "de", "fp32")
        bounded = mock.AsyncMock(return_value=results)

        with (
            mock.patch.object(v2, "_processing_key", return_value=processing_key),
            mock.patch.object(v2, "uses_cohere_backend", return_value=False),
            mock.patch.object(v2, "split_audio_for_whisper", return_value=segments),
            mock.patch.object(v2, "enqueue_audio_segments_bounded", new=bounded),
        ):
            text, _, segment_count, model_id = await v2._transcribe_audio(request, audio)

        self.assertEqual(text, "V2 0 V2 1 V2 2")
        self.assertEqual(segment_count, 3)
        self.assertEqual(model_id, "openai/whisper-tiny")
        bounded.assert_awaited_once()
        self.assertIs(bounded.await_args.args[1], segments)

    async def test_admin_benchmark_uses_shared_bounded_scheduler_across_runs(self) -> None:
        audio = np.full(32000, 0.1, dtype=np.float32)
        segments = [audio[:16000], audio[16000:]]
        results = [
            SimpleNamespace(text=text, batch_id=f"batch-{index // 2}")
            for index, text in enumerate(["A", "B", "A", "B", "A", "B"])
        ]
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    local_gpu_lock=asyncio.Lock(),
                    whisper_batch_manager=SimpleNamespace(),
                )
            ),
            headers={},
        )
        processing_key = ("openai/whisper-tiny", "cpu", "", "de", "fp32")
        bounded = mock.AsyncMock(return_value=results)

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
            mock.patch.object(admin, "enqueue_batch_specs_bounded", new=bounded),
            mock.patch.object(admin, "repetition_filter_enabled", return_value=False),
        ):
            response = await admin._run_admin_benchmark(request, audio, repeat_count=3)

        self.assertEqual(response["transcript"], "A B")
        self.assertTrue(response["transcripts_match"])
        specs = bounded.await_args.args[1]
        self.assertEqual(len(specs), 6)
        self.assertEqual([spec.segment_index for spec in specs], [0, 1, 0, 1, 0, 1])
        self.assertEqual(len({spec.request_id for spec in specs}), 3)


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
