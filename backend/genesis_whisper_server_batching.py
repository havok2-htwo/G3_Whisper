from __future__ import annotations

import asyncio
import datetime
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

from .genesis_whisper_server_globals import (
    batch_history,
    batch_runtime_state,
    batch_state_lock,
    current_settings,
    job_history,
    settings_lock,
)
from .genesis_whisper_server_gpu import run_blocking_gpu_phase


SAMPLE_RATE = 16000
# How long the ASR queue must stay empty before the (optional) idle trim releases
# the reserved CUDA pool. Five seconds keeps the pool warm under steady live
# traffic (requests every few seconds never pay the re-reservation) while a
# genuinely idle server still drops to its model floor on a shared GPU.
IDLE_TRIM_GRACE_SECONDS = 5.0

DEFAULT_LONG_JOB_MIN_CHUNKS = 5
DEFAULT_MAX_PARALLEL_LONG_JOBS = 2


@dataclass
class BatchTranscriptionResult:
    text: str
    batch_id: str


class TranscriptionJob:
    """One caller's chunk list plus the scheduler bookkeeping for it.

    A job owns its chunks for its whole lifetime (they are views into the
    decoded recording), so the scheduler can pick any pending chunk at any
    time without the caller feeding items through a sliding window.
    """

    def __init__(
        self,
        request_id: str,
        processing_key: Any,
        segments: Sequence[np.ndarray],
        is_long: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.key = uuid.uuid4().hex
        self.request_id = request_id
        self.processing_key = processing_key
        self.segments: List[np.ndarray] = list(segments)
        self.is_long = is_long
        self.pending: Deque[int] = deque(range(len(self.segments)))
        self.inflight: set[int] = set()
        self.results: List[Optional[BatchTranscriptionResult]] = [None] * len(self.segments)
        self.batch_count = 0
        self.created_at = time.monotonic()
        self.started_at: Optional[float] = None
        self.cancel_requested = False
        self.finished = False
        self.done: asyncio.Future = loop.create_future()

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def completed_segments(self) -> int:
        return self.total_segments - len(self.pending) - len(self.inflight)

    @property
    def queue_wait_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        return round((self.started_at - self.created_at) * 1000)

    def has_pending(self) -> bool:
        return bool(self.pending) and not self.cancel_requested

    def is_complete(self) -> bool:
        return not self.pending and not self.inflight

    def segment_duration(self, index: int) -> float:
        return len(self.segments[index]) / SAMPLE_RATE


@dataclass
class BatchItem:
    job: TranscriptionJob
    segment_index: int

    @property
    def audio_data(self) -> np.ndarray:
        return self.job.segments[self.segment_index]

    @property
    def duration_seconds(self) -> float:
        return self.job.segment_duration(self.segment_index)

    @property
    def request_id(self) -> str:
        return self.job.request_id


@dataclass
class BatchPlan:
    items: List[BatchItem]
    fast_path: bool

    @property
    def audio_seconds(self) -> float:
        return sum(item.duration_seconds for item in self.items)


class WhisperBatchManager:
    """Fair ASR scheduler: one worker, many jobs, round-robin batches.

    Modelled on the OmniVoice QueueService so short live requests never queue
    behind the chunk backlog of a long recording:

    * every submitted job is either ``active`` (eligible for batches) or, for
      long jobs beyond ``scheduler_max_parallel_long_jobs``, ``waiting``;
      short jobs are always active
    * a batch is planned round-robin over the active jobs that share the
      anchor's processing key: one chunk per job per round until the batch
      limits are reached, so a one-chunk request is in the very next batch
    * first-chunk fast path: when never-served jobs are present next to jobs
      that already received a batch, the next batch carries only the
      newcomers' first chunks; fast-path and regular batches alternate so
      the long job cannot starve under a steady stream of live requests
    * jobs served by a batch rotate to the end of the active list, so the
      next anchor is always the job that waited longest (also across keys)
    """

    def __init__(self, process_batch_fn: Callable[[List[np.ndarray], Any], List[str]], gpu_lock: asyncio.Lock):
        self._process_batch_fn = process_batch_fn
        self._gpu_lock = gpu_lock
        self._jobs: Dict[str, TranscriptionJob] = {}
        self._waiting: Deque[str] = deque()
        self._active: List[str] = []
        self._wake = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_batch_was_fast_path = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self):
        if self._worker_task and not self._worker_task.done():
            return
        self._stop_event = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="whisper-batch-worker")
        with batch_state_lock:
            batch_runtime_state["worker_running"] = True
            batch_runtime_state["last_error"] = None

    async def stop(self):
        self._stop_event.set()
        self._wake.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        for job in list(self._jobs.values()):
            self._finish_job(job, error=RuntimeError("Batch-Worker wurde gestoppt."), status="stopped")

        with batch_state_lock:
            batch_runtime_state["worker_running"] = False
            batch_runtime_state["queue_size"] = 0
            batch_runtime_state["pending_buffer_size"] = 0
            batch_runtime_state["active_jobs"] = 0
            batch_runtime_state["waiting_jobs"] = 0
            batch_runtime_state["active_batch_id"] = None
            batch_runtime_state["active_batch_size"] = 0
            batch_runtime_state["active_batch_audio_seconds"] = 0.0

    # ------------------------------------------------------------------ submission

    async def submit_job(
        self,
        audio_segments: Sequence[np.ndarray],
        request_id: str,
        processing_key: Any,
    ) -> List[BatchTranscriptionResult]:
        """Transcribe one request's chunks; results keep the chunk order.

        Cancelling the awaiting task (client disconnect, timeout) withdraws
        every chunk that has not reached the GPU yet.
        """

        segments = list(audio_segments)
        if not segments:
            return []
        if self._worker_task is None or self._worker_task.done():
            # Callers map RuntimeError to a retryable 503 instead of hanging forever.
            raise RuntimeError("Batch-Worker laeuft nicht.")
        limits = self._get_limits()
        job = TranscriptionJob(
            request_id=request_id,
            processing_key=processing_key,
            segments=segments,
            is_long=len(segments) >= limits["long_job_min_chunks"],
            loop=asyncio.get_running_loop(),
        )
        self._jobs[job.key] = job
        if job.is_long and not self._long_slot_available(limits):
            self._waiting.append(job.key)
        else:
            self._active.append(job.key)
        self._update_queue_state()
        self._wake.set()
        try:
            return await job.done
        finally:
            if not job.finished:
                self._cancel_job(job)

    async def enqueue(
        self,
        audio_data: np.ndarray,
        request_id: str,
        segment_index: int = 0,
        total_segments: int = 1,
        processing_key: Any = None,
    ) -> BatchTranscriptionResult:
        """Single-chunk convenience wrapper around ``submit_job``."""

        _ = (segment_index, total_segments)
        results = await self.submit_job([audio_data], request_id, processing_key)
        return results[0]

    # ------------------------------------------------------------------ state

    def snapshot(self) -> Dict[str, Any]:
        with batch_state_lock:
            state = dict(batch_runtime_state)
        state["recent_batches"] = list(batch_history)
        state["recent_jobs"] = list(job_history)
        state["jobs"] = [self._describe_job(key, "active") for key in self._active] + [
            self._describe_job(key, "waiting") for key in self._waiting
        ]
        return state

    def _describe_job(self, key: str, state: str) -> Dict[str, Any]:
        job = self._jobs[key]
        waited_ms = job.queue_wait_ms
        if waited_ms is None:
            waited_ms = round((time.monotonic() - job.created_at) * 1000)
        return {
            "request_id": job.request_id,
            "state": state,
            "is_long": job.is_long,
            "total_chunks": job.total_segments,
            "completed_chunks": job.completed_segments,
            "pending_chunks": len(job.pending),
            "inflight_chunks": len(job.inflight),
            "batch_count": job.batch_count,
            "queue_wait_ms": waited_ms,
        }

    def _update_queue_state(self):
        active_pending = sum(len(self._jobs[key].pending) for key in self._active)
        waiting_pending = sum(len(self._jobs[key].pending) for key in self._waiting)
        with batch_state_lock:
            batch_runtime_state["queue_size"] = active_pending
            batch_runtime_state["pending_buffer_size"] = waiting_pending
            batch_runtime_state["active_jobs"] = len(self._active)
            batch_runtime_state["waiting_jobs"] = len(self._waiting)

    def _get_limits(self) -> Dict[str, Any]:
        with settings_lock:
            return {
                "wait_time_ms": int(current_settings.get("batch_wait_time_ms", 250)),
                "max_segments": int(current_settings.get("batch_max_segments", 16)),
                "max_audio_seconds": float(current_settings.get("batch_max_audio_seconds", 120.0)),
                "long_job_min_chunks": max(
                    1, int(current_settings.get("scheduler_long_job_min_chunks", DEFAULT_LONG_JOB_MIN_CHUNKS))
                ),
                "max_parallel_long_jobs": max(
                    0, int(current_settings.get("scheduler_max_parallel_long_jobs", DEFAULT_MAX_PARALLEL_LONG_JOBS))
                ),
                "first_chunk_fast_path": current_settings.get("scheduler_first_chunk_fast_path", True) is not False,
            }

    # ------------------------------------------------------------------ job bookkeeping

    def _remove_job(self, job: TranscriptionJob) -> None:
        self._jobs.pop(job.key, None)
        if job.key in self._active:
            self._active.remove(job.key)
        if job.key in self._waiting:
            self._waiting.remove(job.key)

    def _record_job(self, job: TranscriptionJob, status: str, error: Optional[str] = None) -> None:
        entry: Dict[str, Any] = {
            "request_id": job.request_id,
            "status": status,
            "is_long": job.is_long,
            "total_chunks": job.total_segments,
            "completed_chunks": job.completed_segments,
            "batch_count": job.batch_count,
            "queue_wait_ms": job.queue_wait_ms,
            "duration_ms": round((time.monotonic() - job.created_at) * 1000),
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if error:
            entry["error"] = error
        job_history.appendleft(entry)

    def _finish_job(
        self,
        job: TranscriptionJob,
        *,
        results: Optional[List[BatchTranscriptionResult]] = None,
        error: Optional[BaseException] = None,
        status: str = "ok",
    ) -> None:
        if job.finished:
            return
        job.finished = True
        self._remove_job(job)
        if not job.done.done():
            if error is not None:
                job.done.set_exception(error)
            else:
                job.done.set_result(results or [])
        self._record_job(job, status, str(error) if error is not None else None)
        self._update_queue_state()
        self._wake.set()

    def _cancel_job(self, job: TranscriptionJob) -> None:
        job.cancel_requested = True
        job.finished = True
        job.pending.clear()
        self._remove_job(job)
        self._record_job(job, "cancelled")
        self._update_queue_state()
        self._wake.set()

    # ------------------------------------------------------------------ scheduling rules

    def _active_jobs(self) -> List[TranscriptionJob]:
        return [self._jobs[key] for key in self._active]

    def _eligible_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self._active_jobs() if job.has_pending()]

    def _long_slot_available(self, limits: Dict[str, Any]) -> bool:
        max_parallel = int(limits["max_parallel_long_jobs"])
        if max_parallel <= 0:
            return True
        active_long = sum(1 for job in self._active_jobs() if job.is_long)
        return active_long < max_parallel

    def _has_schedulable_work(self) -> bool:
        if any(job.has_pending() for job in self._active_jobs()):
            return True
        return bool(self._waiting) and self._long_slot_available(self._get_limits())

    def _promote_waiting(self, limits: Dict[str, Any]) -> None:
        promoted = False
        while self._waiting and self._long_slot_available(limits):
            key = self._waiting.popleft()
            if key in self._jobs:
                self._active.append(key)
                promoted = True
        if promoted:
            self._update_queue_state()

    def _plan_is_full(self, plan: BatchPlan, limits: Dict[str, Any]) -> bool:
        if len(plan.items) >= max(1, int(limits["max_segments"])):
            return True
        return plan.audio_seconds >= max(1.0, float(limits["max_audio_seconds"]))

    def _should_batch_wait(self, plan: BatchPlan, limits: Dict[str, Any]) -> bool:
        """Hold a not-yet-full batch briefly only when a new job just arrived.

        Chunks of an already-served job are all present, so waiting for them
        gains nothing. A fresh job, however, is often one of several live
        requests arriving together; the short wait lets them share one batch.
        """

        if int(limits["wait_time_ms"]) <= 0:
            return False
        if not any(job.batch_count == 0 for job in self._eligible_jobs()):
            return False
        return not self._plan_is_full(plan, limits)

    def _build_batch_plan(self, limits: Dict[str, Any]) -> BatchPlan:
        eligible = self._eligible_jobs()
        if not eligible:
            return BatchPlan(items=[], fast_path=False)
        max_segments = max(1, int(limits["max_segments"]))
        max_audio_seconds = max(1.0, float(limits["max_audio_seconds"]))

        fresh = [job for job in eligible if job.batch_count == 0]
        seasoned = [job for job in eligible if job.batch_count > 0]
        use_fast_path = (
            bool(limits["first_chunk_fast_path"])
            and bool(fresh)
            and bool(seasoned)
            and not self._last_batch_was_fast_path
        )
        if use_fast_path:
            anchor = fresh[0]
            items: List[BatchItem] = []
            audio_seconds = 0.0
            for job in fresh:
                if job.processing_key != anchor.processing_key:
                    continue
                segment_index = job.pending[0]
                duration = job.segment_duration(segment_index)
                if items and (len(items) >= max_segments or audio_seconds + duration > max_audio_seconds):
                    break
                items.append(BatchItem(job=job, segment_index=segment_index))
                audio_seconds += duration
            return BatchPlan(items=items, fast_path=True)

        anchor = eligible[0]
        candidates = [job for job in eligible if job.processing_key == anchor.processing_key]
        offsets = {job.key: 0 for job in candidates}
        items = []
        audio_seconds = 0.0
        made_progress = True
        while made_progress and len(items) < max_segments:
            made_progress = False
            for job in candidates:
                offset = offsets[job.key]
                if offset >= len(job.pending):
                    continue
                segment_index = job.pending[offset]
                duration = job.segment_duration(segment_index)
                if items and audio_seconds + duration > max_audio_seconds:
                    return BatchPlan(items=items, fast_path=False)
                items.append(BatchItem(job=job, segment_index=segment_index))
                audio_seconds += duration
                offsets[job.key] = offset + 1
                made_progress = True
                if len(items) >= max_segments:
                    break
        return BatchPlan(items=items, fast_path=False)

    def _reserve_batch(self, plan: BatchPlan) -> List[BatchItem]:
        involved: List[TranscriptionJob] = []
        for item in plan.items:
            job = item.job
            actual = job.pending.popleft()
            if actual != item.segment_index:
                raise RuntimeError(
                    f"Chunk-Reihenfolge fuer Request {job.request_id} ist verrutscht: "
                    f"erwartet {item.segment_index}, erhalten {actual}."
                )
            job.inflight.add(actual)
            if job not in involved:
                involved.append(job)

        now = time.monotonic()
        for job in involved:
            job.batch_count += 1
            if job.started_at is None:
                job.started_at = now
            # Served jobs rotate to the back so the next anchor is the job that
            # waited longest, including jobs with a different processing key.
            self._active.remove(job.key)
            self._active.append(job.key)
        self._last_batch_was_fast_path = plan.fast_path
        self._update_queue_state()
        return list(plan.items)

    # ------------------------------------------------------------------ worker

    @staticmethod
    def _trim_cuda_cache() -> None:
        """Release the reserved CUDA cache pool back to the driver so idle VRAM drops to
        the model floor (leaves room for the other GPU tenants). Best-effort."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def _trim_cuda_cache_if_enabled(self) -> bool:
        """Trim only when the operator explicitly trades latency for idle VRAM.

        ``torch.cuda.empty_cache()`` is process-wide rather than model-scoped.
        Keeping this disabled therefore preserves the warm allocator state used
        by both Cohere ASR and ReDimNet.  When enabled, serialize the trim with
        every other local CUDA phase so it cannot race an embedding inference.
        """

        with settings_lock:
            enabled = current_settings.get("cuda_memory_trim_after_batch", True) is not False
        if not enabled:
            return False

        async with self._gpu_lock:
            # A request may have arrived while this worker waited for the GPU.
            # Serving it has priority over an optional housekeeping operation.
            if self._has_schedulable_work():
                return False
            await run_blocking_gpu_phase(self._trim_cuda_cache)
        return True

    async def _wait_for_wake(self, timeout: Optional[float]) -> bool:
        """Sleep until new work or a state change arrives; False on timeout."""

        if timeout is None:
            await self._wake.wait()
            return True
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def _worker_loop(self):
        dirty = False
        try:
            while not self._stop_event.is_set():
                if not self._has_schedulable_work():
                    # Clearing before waiting cannot lose a wake-up: every state
                    # change sets the event synchronously on this loop.
                    self._wake.clear()
                    if not dirty:
                        await self._wait_for_wake(None)
                    elif not await self._wait_for_wake(IDLE_TRIM_GRACE_SECONDS):
                        # Queue drained: trim once after a burst so the inference
                        # reserved-pool spike is returned to the driver instead of
                        # staying pinned while idle.
                        await self._trim_cuda_cache_if_enabled()
                        dirty = False
                    continue

                limits = self._get_limits()
                self._promote_waiting(limits)
                plan = self._build_batch_plan(limits)
                if plan.items and self._should_batch_wait(plan, limits):
                    self._wake.clear()
                    await self._wait_for_wake(int(limits["wait_time_ms"]) / 1000.0)
                    self._promote_waiting(limits)
                    plan = self._build_batch_plan(limits)
                if not plan.items:
                    await asyncio.sleep(0)
                    continue

                batch_items = self._reserve_batch(plan)
                await self._process_batch(batch_items, fast_path=plan.fast_path)
                dirty = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[BATCH-FEHLER] Whisper-Batch-Worker abgestürzt: {exc}", file=sys.stderr)
            with batch_state_lock:
                batch_runtime_state["last_error"] = str(exc)
                batch_runtime_state["worker_running"] = False
            raise

    def _apply_batch_results(self, batch_items: List[BatchItem], results: List[str], batch_id: str) -> None:
        touched: List[TranscriptionJob] = []
        for item, text in zip(batch_items, results):
            job = item.job
            if job.cancel_requested:
                continue
            job.results[item.segment_index] = BatchTranscriptionResult(text=text, batch_id=batch_id)
            job.inflight.discard(item.segment_index)
            if job not in touched:
                touched.append(job)
        for job in touched:
            if job.is_complete():
                self._finish_job(job, results=[result for result in job.results if result is not None])
        self._update_queue_state()

    def _fail_batch_jobs(self, batch_items: List[BatchItem], exc: BaseException) -> None:
        for job in {item.job.key: item.job for item in batch_items}.values():
            if job.cancel_requested:
                continue
            self._finish_job(job, error=RuntimeError(str(exc)), status="error")

    async def _process_batch(self, batch_items: List[BatchItem], *, fast_path: bool = False):
        batch_items = [item for item in batch_items if not item.job.cancel_requested]
        if not batch_items:
            self._update_queue_state()
            return

        batch_id = f"batch-{uuid.uuid4().hex[:10]}"
        batch_started_at = time.monotonic()
        batch_started_at_wall = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_audio_seconds = sum(item.duration_seconds for item in batch_items)

        with batch_state_lock:
            batch_runtime_state["active_batch_id"] = batch_id
            batch_runtime_state["active_batch_size"] = len(batch_items)
            batch_runtime_state["active_batch_audio_seconds"] = round(total_audio_seconds, 3)
            batch_runtime_state["active_batch_started_at"] = batch_started_at_wall
            batch_runtime_state["active_batch_fast_path"] = fast_path
            batch_runtime_state["last_error"] = None

        try:
            async with self._gpu_lock:
                # A request may be cancelled while this batch waits behind a
                # different GPU phase. Re-evaluate only after ownership is
                # acquired so cancelled work never reaches the model.
                batch_items = [item for item in batch_items if not item.job.cancel_requested]
                if not batch_items:
                    return

                audio_batch = [item.audio_data for item in batch_items]
                total_audio_seconds = sum(item.duration_seconds for item in batch_items)
                with batch_state_lock:
                    batch_runtime_state["active_batch_size"] = len(batch_items)
                    batch_runtime_state["active_batch_audio_seconds"] = round(total_audio_seconds, 3)

                results = await run_blocking_gpu_phase(
                    self._process_batch_fn,
                    audio_batch,
                    batch_items[0].job.processing_key,
                )

            if len(results) != len(batch_items):
                raise RuntimeError(f"Batch-Transkription lieferte {len(results)} Ergebnisse für {len(batch_items)} Segmente.")

            self._apply_batch_results(batch_items, results, batch_id)

            duration_ms = round((time.monotonic() - batch_started_at) * 1000)
            request_ids = sorted({item.request_id for item in batch_items})
            history_entry = {
                "batch_id": batch_id,
                "timestamp": batch_started_at_wall,
                "batch_size": len(batch_items),
                "audio_seconds": round(total_audio_seconds, 3),
                "duration_ms": duration_ms,
                "request_ids": request_ids,
                "unique_request_count": len(request_ids),
                "fast_path": fast_path,
                "status": "ok",
            }
            batch_history.appendleft(history_entry)
            with batch_state_lock:
                batch_runtime_state["last_batch_completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                batch_runtime_state["last_batch_duration_ms"] = duration_ms
                batch_runtime_state["total_batches_processed"] += 1
                batch_runtime_state["total_segments_processed"] += len(batch_items)
        except Exception as exc:
            print(f"[BATCH-FEHLER] Batch {batch_id} fehlgeschlagen: {exc}", file=sys.stderr)
            self._fail_batch_jobs(batch_items, exc)
            request_ids = sorted({item.request_id for item in batch_items})
            batch_history.appendleft(
                {
                    "batch_id": batch_id,
                    "timestamp": batch_started_at_wall,
                    "batch_size": len(batch_items),
                    "audio_seconds": round(total_audio_seconds, 3),
                    "duration_ms": round((time.monotonic() - batch_started_at) * 1000),
                    "request_ids": request_ids,
                    "unique_request_count": len(request_ids),
                    "fast_path": fast_path,
                    "status": "error",
                    "error": str(exc),
                }
            )
            with batch_state_lock:
                batch_runtime_state["last_error"] = str(exc)
        finally:
            self._update_queue_state()
            with batch_state_lock:
                batch_runtime_state["active_batch_id"] = None
                batch_runtime_state["active_batch_size"] = 0
                batch_runtime_state["active_batch_audio_seconds"] = 0.0
                batch_runtime_state["active_batch_started_at"] = None
                batch_runtime_state["active_batch_fast_path"] = False
