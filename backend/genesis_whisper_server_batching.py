from __future__ import annotations

import asyncio
import datetime
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Sequence, TypeVar

import numpy as np

from .genesis_whisper_server_globals import batch_history, batch_runtime_state, batch_state_lock, current_settings, settings_lock
from .genesis_whisper_server_gpu import run_blocking_gpu_phase


ASR_ENQUEUE_MIN_IN_FLIGHT = 2
ASR_ENQUEUE_MAX_IN_FLIGHT = 256
ASR_ENQUEUE_BATCH_MULTIPLIER = 2
BatchResultT = TypeVar("BatchResultT")


@dataclass
class BatchTranscriptionResult:
    text: str
    batch_id: str


@dataclass
class QueuedWhisperSegment:
    request_id: str
    segment_index: int
    total_segments: int
    audio_data: np.ndarray
    processing_key: Any
    future: asyncio.Future
    queued_at_monotonic: float

    @property
    def duration_seconds(self) -> float:
        return len(self.audio_data) / 16000.0


@dataclass(frozen=True)
class BatchEnqueueSpec:
    """Metadata for one item submitted through the shared bounded scheduler."""

    audio_data: np.ndarray
    request_id: str
    segment_index: int
    total_segments: int


def get_asr_enqueue_in_flight_limit() -> int:
    """Keep two configured batches ready without allowing unbounded task bursts."""

    with settings_lock:
        configured_batch_size = max(1, int(current_settings.get("batch_max_segments", 16)))
    return max(
        ASR_ENQUEUE_MIN_IN_FLIGHT,
        min(ASR_ENQUEUE_MAX_IN_FLIGHT, configured_batch_size * ASR_ENQUEUE_BATCH_MULTIPLIER),
    )


async def _run_bounded_enqueues(
    item_count: int,
    enqueue_at_index: Callable[[int], Awaitable[BatchResultT]],
) -> list[BatchResultT]:
    """Run lazy enqueue calls with bounded concurrency and stable result order."""

    if item_count <= 0:
        return []

    result_slots: list[BatchResultT | None] = [None] * item_count
    pending: dict[asyncio.Task[BatchResultT], int] = {}
    next_index = 0
    in_flight_limit = min(item_count, get_asr_enqueue_in_flight_limit())

    def fill_window() -> None:
        nonlocal next_index
        while next_index < item_count and len(pending) < in_flight_limit:
            index = next_index
            next_index += 1
            task = asyncio.create_task(enqueue_at_index(index))
            pending[task] = index

    try:
        fill_window()
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = pending.pop(task)
                result_slots[index] = task.result()
            fill_window()
    finally:
        if pending:
            remaining = list(pending)
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)

    if any(result is None for result in result_slots):
        raise RuntimeError("ASR-Batchverarbeitung lieferte nicht fuer jeden Chunk ein Ergebnis.")
    return [result for result in result_slots if result is not None]


async def enqueue_audio_segments_bounded(
    batch_manager: Any,
    audio_segments: Sequence[np.ndarray],
    request_id: str,
    processing_key: Any,
) -> list[BatchTranscriptionResult]:
    """Submit one request's audio segments without scheduling them all at once."""

    total_segments = len(audio_segments)
    return await _run_bounded_enqueues(
        total_segments,
        lambda index: batch_manager.enqueue(
            audio_data=audio_segments[index],
            request_id=request_id,
            segment_index=index,
            total_segments=total_segments,
            processing_key=processing_key,
        ),
    )


async def enqueue_batch_specs_bounded(
    batch_manager: Any,
    items: Sequence[BatchEnqueueSpec],
    processing_key: Any,
) -> list[BatchTranscriptionResult]:
    """Submit pre-described items, used when one workload has several request IDs."""

    return await _run_bounded_enqueues(
        len(items),
        lambda index: batch_manager.enqueue(
            audio_data=items[index].audio_data,
            request_id=items[index].request_id,
            segment_index=items[index].segment_index,
            total_segments=items[index].total_segments,
            processing_key=processing_key,
        ),
    )


class WhisperBatchManager:
    def __init__(self, process_batch_fn: Callable[[List[np.ndarray], Any], List[str]], gpu_lock: asyncio.Lock):
        self._process_batch_fn = process_batch_fn
        self._gpu_lock = gpu_lock
        self._queue: asyncio.Queue[QueuedWhisperSegment] = asyncio.Queue()
        self._pending_items: Deque[QueuedWhisperSegment] = deque()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

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
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        with batch_state_lock:
            batch_runtime_state["worker_running"] = False
            batch_runtime_state["queue_size"] = 0
            batch_runtime_state["pending_buffer_size"] = 0
            batch_runtime_state["active_batch_id"] = None
            batch_runtime_state["active_batch_size"] = 0
            batch_runtime_state["active_batch_audio_seconds"] = 0.0

    async def enqueue(
        self,
        audio_data: np.ndarray,
        request_id: str,
        segment_index: int,
        total_segments: int,
        processing_key: Any,
    ) -> BatchTranscriptionResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = QueuedWhisperSegment(
            request_id=request_id,
            segment_index=segment_index,
            total_segments=total_segments,
            audio_data=audio_data,
            processing_key=processing_key,
            future=future,
            queued_at_monotonic=time.monotonic(),
        )
        await self._queue.put(item)
        self._update_queue_state()
        return await future

    def snapshot(self) -> Dict[str, Any]:
        with batch_state_lock:
            state = dict(batch_runtime_state)
        state["recent_batches"] = list(batch_history)
        return state

    def _update_queue_state(self):
        with batch_state_lock:
            batch_runtime_state["queue_size"] = self._queue.qsize()
            batch_runtime_state["pending_buffer_size"] = len(self._pending_items)

    def _get_limits(self) -> Dict[str, float]:
        with settings_lock:
            return {
                "wait_time_ms": int(current_settings.get("batch_wait_time_ms", 250)),
                "max_segments": int(current_settings.get("batch_max_segments", 16)),
                "max_audio_seconds": float(current_settings.get("batch_max_audio_seconds", 120.0)),
            }

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
            enabled = current_settings.get("cuda_memory_trim_after_batch", False) is True
        if not enabled:
            return False

        async with self._gpu_lock:
            # A request may have arrived while this worker waited for the GPU.
            # Serving it has priority over an optional housekeeping operation.
            if not self._queue.empty() or self._pending_items:
                return False
            await run_blocking_gpu_phase(self._trim_cuda_cache)
        return True

    async def _worker_loop(self):
        dirty = False
        try:
            while not self._stop_event.is_set():
                first_item = await self._get_next_item()
                if first_item is None:
                    # Queue drained: trim once after a burst so the inference reserved-pool
                    # spike is returned to the driver instead of staying pinned while idle.
                    if dirty:
                        await self._trim_cuda_cache_if_enabled()
                        dirty = False
                    continue
                batch_items = await self._collect_batch(first_item)
                if batch_items:
                    await self._process_batch(batch_items)
                    dirty = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[BATCH-FEHLER] Whisper-Batch-Worker abgestürzt: {exc}", file=sys.stderr)
            with batch_state_lock:
                batch_runtime_state["last_error"] = str(exc)
                batch_runtime_state["worker_running"] = False
            raise

    async def _get_next_item(self) -> Optional[QueuedWhisperSegment]:
        deadline = time.monotonic() + 0.25
        while True:
            if self._pending_items:
                item = self._pending_items.popleft()
                self._update_queue_state()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    self._update_queue_state()
                except asyncio.TimeoutError:
                    return None

            # Cancelling a request also cancels the Future it was awaiting.
            # Do not spend ASR/GPU time on an item whose caller is already gone.
            if not item.future.done():
                return item

    async def _collect_batch(self, first_item: QueuedWhisperSegment) -> List[QueuedWhisperSegment]:
        limits = self._get_limits()
        max_segments = max(1, int(limits["max_segments"]))
        max_audio_seconds = max(1.0, float(limits["max_audio_seconds"]))
        wait_time_ms = max(0, int(limits["wait_time_ms"]))

        batch_items = [first_item]
        total_audio_seconds = first_item.duration_seconds
        deadline = time.monotonic() + (wait_time_ms / 1000.0)

        while len(batch_items) < max_segments:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            next_item: Optional[QueuedWhisperSegment] = None
            if self._pending_items:
                next_item = self._pending_items.popleft()
                self._update_queue_state()
            else:
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    self._update_queue_state()
                except asyncio.TimeoutError:
                    break

            if next_item is None:
                break

            if next_item.future.done():
                continue

            if next_item.processing_key != first_item.processing_key:
                self._pending_items.appendleft(next_item)
                self._update_queue_state()
                break

            proposed_audio_seconds = total_audio_seconds + next_item.duration_seconds
            if proposed_audio_seconds > max_audio_seconds and batch_items:
                self._pending_items.appendleft(next_item)
                self._update_queue_state()
                break

            batch_items.append(next_item)
            total_audio_seconds = proposed_audio_seconds

        return batch_items

    async def _process_batch(self, batch_items: List[QueuedWhisperSegment]):
        batch_items = [item for item in batch_items if not item.future.done()]
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
            batch_runtime_state["last_error"] = None

        try:
            async with self._gpu_lock:
                # A request may be cancelled while this batch waits behind a
                # different GPU phase. Re-evaluate only after ownership is
                # acquired so cancelled work never reaches the model.
                batch_items = [item for item in batch_items if not item.future.done()]
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
                    batch_items[0].processing_key,
                )

            if len(results) != len(batch_items):
                raise RuntimeError(f"Batch-Transkription lieferte {len(results)} Ergebnisse für {len(batch_items)} Segmente.")

            for item, text in zip(batch_items, results):
                if not item.future.done():
                    item.future.set_result(BatchTranscriptionResult(text=text, batch_id=batch_id))

            duration_ms = round((time.monotonic() - batch_started_at) * 1000)
            history_entry = {
                "batch_id": batch_id,
                "timestamp": batch_started_at_wall,
                "batch_size": len(batch_items),
                "audio_seconds": round(total_audio_seconds, 3),
                "duration_ms": duration_ms,
                "request_ids": sorted({item.request_id for item in batch_items}),
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
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(RuntimeError(str(exc)))
            batch_history.appendleft(
                {
                    "batch_id": batch_id,
                    "timestamp": batch_started_at_wall,
                    "batch_size": len(batch_items),
                    "audio_seconds": round(total_audio_seconds, 3),
                    "duration_ms": round((time.monotonic() - batch_started_at) * 1000),
                    "request_ids": sorted({item.request_id for item in batch_items}),
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
