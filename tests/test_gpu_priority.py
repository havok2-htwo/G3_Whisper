"""Interactive work before background work on the local GPU lock (PriorityGpuLock, batch manager, v2 classes)."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import numpy as np

from backend import genesis_whisper_server_batching as batching
from backend import genesis_whisper_server_gpu as gpu
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_batching import WhisperBatchManager
from backend.genesis_whisper_server_globals import batch_history, current_settings, job_history, settings_lock


KEY = ("model", "cpu", "cache", "de", "fp32")
SETTINGS = {
    "batch_wait_time_ms": 0,
    "batch_max_segments": 16,
    "batch_max_audio_seconds": 1000.0,
    "scheduler_long_job_min_chunks": 5,
    "scheduler_max_parallel_long_jobs": 2,
    "scheduler_first_chunk_fast_path": True,
    "cuda_memory_trim_after_batch": False,
}


async def settle(rounds: int = 5) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def hold(lock, log: list, name: str, gpu_class: str, *, steps: int = 1) -> None:
    """One lock hold in ``gpu_class`` that logs its start and end and yields ``steps`` times while holding."""

    with gpu.gpu_class_scope(gpu_class):
        async with lock:
            log.append(f"{name}+")
            for _ in range(steps):
                await asyncio.sleep(0)
            log.append(f"{name}-")


def starts(log: list) -> list:
    return [entry[:-1] for entry in log if entry.endswith("+")]


class PriorityGpuLockTests(unittest.IsolatedAsyncioTestCase):
    async def _queue_behind_holder(self, lock, waiters):
        """Hold the lock, queue ``waiters`` [(name, class)] in that order, release; returns the log."""

        log: list = []
        await lock.acquire()
        tasks = []
        for name, gpu_class in waiters:
            tasks.append(asyncio.create_task(hold(lock, log, name, gpu_class)))
            await settle()
        lock.release()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        return log

    async def test_interactive_waiter_overtakes_waiting_background_waiters(self) -> None:
        log = await self._queue_behind_holder(
            gpu.PriorityGpuLock(),
            [("b1", gpu.BACKGROUND), ("b2", gpu.BACKGROUND), ("i1", gpu.INTERACTIVE)],
        )
        self.assertEqual(starts(log), ["i1", "b1", "b2"])

    async def test_fifo_inside_each_class(self) -> None:
        log = await self._queue_behind_holder(
            gpu.PriorityGpuLock(),
            [("b1", gpu.BACKGROUND), ("i1", gpu.INTERACTIVE), ("b2", gpu.BACKGROUND), ("i2", gpu.INTERACTIVE)],
        )
        self.assertEqual(starts(log), ["i1", "i2", "b1", "b2"])

    async def test_plain_asyncio_lock_is_fifo_counterproof(self) -> None:
        # Same queue on the old lock: the interactive waiter waits behind both background ones.
        log = await self._queue_behind_holder(
            asyncio.Lock(),
            [("b1", gpu.BACKGROUND), ("b2", gpu.BACKGROUND), ("i1", gpu.INTERACTIVE)],
        )
        self.assertEqual(starts(log), ["b1", "b2", "i1"])

    async def test_a_running_hold_is_never_interrupted(self) -> None:
        lock = gpu.PriorityGpuLock()
        log: list = []
        background = asyncio.create_task(hold(lock, log, "b", gpu.BACKGROUND, steps=20))
        await settle(2)
        self.assertEqual(log, ["b+"])
        interactive = asyncio.create_task(hold(lock, log, "i", gpu.INTERACTIVE))
        await asyncio.wait_for(asyncio.gather(background, interactive), timeout=5.0)
        self.assertEqual(log, ["b+", "b-", "i+", "i-"])

    async def test_background_runs_through_without_interactive_load(self) -> None:
        lock = gpu.PriorityGpuLock()
        log: list = []
        tasks = [asyncio.create_task(hold(lock, log, f"b{n}", gpu.BACKGROUND, steps=2)) for n in range(5)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        self.assertEqual(starts(log), [f"b{n}" for n in range(5)])
        self.assertFalse(lock.locked())

    async def test_cancelled_waiters_are_skipped_and_a_cancelled_handoff_is_passed_on(self) -> None:
        lock = gpu.PriorityGpuLock()
        log: list = []
        await lock.acquire()
        gone = asyncio.create_task(hold(lock, log, "gone", gpu.INTERACTIVE))
        late = asyncio.create_task(hold(lock, log, "late", gpu.INTERACTIVE))
        rest = asyncio.create_task(hold(lock, log, "rest", gpu.BACKGROUND))
        await settle()
        gone.cancel()
        await settle()
        self.assertEqual(lock.waiting(gpu.INTERACTIVE), 1)
        # Hand the lock to "late", then cancel it before it ran: the lock must go on to "rest".
        lock.release()
        late.cancel()
        await asyncio.wait_for(asyncio.gather(gone, late, rest, return_exceptions=True), timeout=5.0)
        self.assertEqual(starts(log), ["rest"])
        self.assertFalse(lock.locked())

    async def test_interactive_request_goes_between_two_sliced_background_phases(self) -> None:
        lock = gpu.PriorityGpuLock()
        log: list = []

        def slice_fn(tag):
            def run(part):
                log.append(f"{tag}{part[0]}")
                return part
            return run

        async def fake_phase(function, *args, **kwargs):
            result = function(*args, **kwargs)  # logs at the start of the hold
            await settle(3)
            return result

        async def background(tag):
            with gpu.gpu_class_scope(gpu.BACKGROUND):
                await gpu.run_sliced_gpu_phase(lock, list(range(4)), slice_fn(tag), 1)

        async def interactive():
            with gpu.gpu_class_scope(gpu.INTERACTIVE):
                await gpu.run_sliced_gpu_phase(lock, [0], slice_fn("live"), 1)

        with mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase):
            first = asyncio.create_task(background("a"))
            second = asyncio.create_task(background("b"))
            while "a0" not in log:
                await asyncio.sleep(0)
            live = asyncio.create_task(interactive())
            await asyncio.wait_for(asyncio.gather(first, second, live), timeout=5.0)
        # The live request runs at the very next slice boundary, ahead of the waiting slice of the other job.
        self.assertEqual(log.index("live0"), 1, log)
        self.assertEqual(sorted(log), sorted(["live0"] + [f"{t}{n}" for t in "ab" for n in range(4)]))


class _BatchPriorityCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._previous = current_settings.copy()
        with settings_lock:
            current_settings.clear()
            current_settings.update(SETTINGS)
        batch_history.clear()
        job_history.clear()
        self.log: list = []
        self.owners: dict[int, str] = {}
        self.lock = gpu.PriorityGpuLock()
        self.manager = WhisperBatchManager(lambda audio, key: ["t"] * len(audio), self.lock)
        self._patch = mock.patch.object(batching, "run_blocking_gpu_phase", new=self._gpu_phase)
        self._patch.start()
        self._spy_on(self.lock)
        await self.manager.start()

    def _spy_on(self, lock) -> None:
        """Log ("granted", class) whenever ``lock`` is handed to a waiter, in the class it waited in."""

        original = lock.acquire

        async def acquire():
            gpu_class = gpu.current_gpu_class()
            result = await original()
            self.log.append(("granted", gpu_class))
            return result

        lock.acquire = acquire

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        self._patch.stop()
        with settings_lock:
            current_settings.clear()
            current_settings.update(self._previous)

    async def _gpu_phase(self, function, audio_batch, key):
        self.log.append(("asr", sorted({self.owners[id(chunk)] for chunk in audio_batch})))
        await asyncio.sleep(0)
        return function(audio_batch, key)

    def submit(self, owner: str, count: int, gpu_class: str = gpu.INTERACTIVE) -> asyncio.Task:
        chunks = [np.zeros(16000, dtype=np.float32) for _ in range(count)]
        for chunk in chunks:
            self.owners[id(chunk)] = owner
        with gpu.gpu_class_scope(gpu.INTERACTIVE if gpu_class == gpu.INTERACTIVE else gpu.BACKGROUND):
            # The task copies the context: the job takes the class of its request.
            return asyncio.create_task(self.manager.submit_job(chunks, owner, KEY))

    def batches(self) -> list:
        return [entry[1] for entry in self.log if entry[0] == "asr"]

    def batch_classes(self) -> list:
        """(class the lock was granted in, owners) of every ASR batch."""

        out = []
        for index, entry in enumerate(self.log):
            if entry[0] == "asr":
                granted = [e for e in self.log[:index] if isinstance(e, tuple) and e[0] == "granted"]
                out.append((granted[-1][1], entry[1]))
        return out


class BatchPriorityTests(_BatchPriorityCase):
    async def test_jobs_take_the_class_of_their_request(self) -> None:
        await self.lock.acquire()
        live = self.submit("live", 1)
        bulk = self.submit("bulk", 6, gpu.BACKGROUND)
        await settle()
        classes = {job.request_id: job.interactive for job in self.manager._jobs.values()}
        self.assertEqual(classes, {"live": True, "bulk": False})
        self.lock.release()
        await asyncio.wait_for(asyncio.gather(live, bulk), timeout=5.0)

    async def test_interactive_jobs_are_planned_alone_and_batches_lock_in_their_class(self) -> None:
        await self.lock.acquire()
        bulk = self.submit("bulk", 6, gpu.BACKGROUND)
        live = self.submit("live", 1)
        await settle()
        plan = self.manager._build_batch_plan(self.manager._get_limits())
        self.assertEqual({item.job.request_id for item in plan.items}, {"live"})
        self.assertTrue(self.manager._plan_is_interactive(plan))
        self.lock.release()
        await asyncio.wait_for(asyncio.gather(live, bulk), timeout=5.0)
        self.assertEqual(self.batch_classes(), [(gpu.INTERACTIVE, ["live"]), (gpu.BACKGROUND, ["bulk"])])

    async def test_background_job_runs_through_without_interactive_load(self) -> None:
        bulk = self.submit("bulk", 40, gpu.BACKGROUND)
        results = await asyncio.wait_for(bulk, timeout=5.0)
        self.assertEqual(len(results), 40)
        self.assertEqual(self.batch_classes(), [(gpu.BACKGROUND, ["bulk"])] * 3)  # 16 + 16 + 8 chunks

    async def test_a_job_that_arrives_while_the_worker_waits_rides_in_the_next_batch(self) -> None:
        # Reservation fix: the worker plans the batch only once it owns the lock (here a plain asyncio.Lock,
        # so this is about the planning order, not about the classes).
        lock = asyncio.Lock()
        await self.manager.stop()
        self.manager = WhisperBatchManager(lambda audio, key: ["t"] * len(audio), lock)
        await self.manager.start()
        await lock.acquire()
        first = self.submit("first", 1)
        await settle()
        second = self.submit("second", 1)
        await settle()
        lock.release()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)
        self.assertEqual(self.batches(), [["first", "second"]])

    async def test_background_wait_of_the_worker_turns_interactive_when_a_live_job_arrives(self) -> None:
        await self.lock.acquire()
        bulk = self.submit("bulk", 6, gpu.BACKGROUND)
        await settle()
        # Three background slices of diarizations queue up behind the worker's background wait ...
        slices = [asyncio.create_task(hold(self.lock, self.log, f"slice{n}", gpu.BACKGROUND)) for n in range(3)]
        await settle()
        # ... then a live request submits its job.
        live = self.submit("live", 1)
        await settle(20)
        self.lock.release()
        await asyncio.wait_for(asyncio.gather(bulk, live, *slices), timeout=5.0)
        order = [entry for entry in self.log if isinstance(entry, str) and entry.endswith("+") or entry[0] == "asr"]
        self.assertEqual(
            order,
            [("asr", ["live"]), "slice0+", "slice1+", "slice2+", ("asr", ["bulk"])],
            self.log,
        )
        self.assertEqual(self.batch_classes(), [(gpu.INTERACTIVE, ["live"]), (gpu.BACKGROUND, ["bulk"])])

    async def test_interactive_long_job_does_not_wait_for_a_background_long_slot(self) -> None:
        self.set_slots(1)
        await self.lock.acquire()
        bulk = self.submit("bulk", 6, gpu.BACKGROUND)
        await settle()
        live_long = self.submit("live-long", 6)
        late_bulk = self.submit("late-bulk", 6, gpu.BACKGROUND)
        await settle()
        active = {self.manager._jobs[key].request_id for key in self.manager._active}
        waiting = {self.manager._jobs[key].request_id for key in self.manager._waiting}
        self.assertEqual(active, {"bulk", "live-long"})
        self.assertEqual(waiting, {"late-bulk"})
        self.lock.release()
        await asyncio.wait_for(asyncio.gather(bulk, live_long, late_bulk), timeout=5.0)
        self.assertEqual(self.batches()[0], ["live-long"])

    def set_slots(self, count: int) -> None:
        with settings_lock:
            current_settings["scheduler_max_parallel_long_jobs"] = count


class RequestGpuClassTests(unittest.TestCase):
    def test_classes_follow_the_mode_and_the_existing_long_transcript_limit(self) -> None:
        limit = v2.WHOLE_AUDIO_MAX_S
        self.assertEqual(v2.request_gpu_class("diarization", 5.0), gpu.BACKGROUND)
        self.assertEqual(v2.request_gpu_class("transcript", 4.0), gpu.INTERACTIVE)
        self.assertEqual(v2.request_gpu_class("transcript", limit), gpu.INTERACTIVE)
        self.assertEqual(v2.request_gpu_class("transcript", limit + 1), gpu.BACKGROUND)
        self.assertEqual(v2.request_gpu_class("transcript_embedding", limit + 1), gpu.BACKGROUND)
        self.assertEqual(v2.request_gpu_class("transcript_embedding", 8.0), gpu.INTERACTIVE)
        self.assertEqual(v2.request_gpu_class("embedding", 600.0), gpu.INTERACTIVE)

    def test_default_class_is_interactive(self) -> None:
        self.assertEqual(gpu.current_gpu_class(), gpu.INTERACTIVE)
        with gpu.gpu_class_scope(gpu.BACKGROUND):
            self.assertEqual(gpu.current_gpu_class(), gpu.BACKGROUND)
        self.assertEqual(gpu.current_gpu_class(), gpu.INTERACTIVE)
        with self.assertRaises(ValueError):
            gpu.set_gpu_class("urgent")


if __name__ == "__main__":
    unittest.main()
