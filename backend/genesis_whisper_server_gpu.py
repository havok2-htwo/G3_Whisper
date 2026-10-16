"""Optional cross-process GPU coordination for colocated voice services."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Iterator
from typing import Any, Callable, List, Sequence, TypeVar


T = TypeVar("T")


# CUDA libraries keep meaningful state in the calling host thread (for example
# cuDNN/cuBLAS handles and lazily prepared kernels). ``asyncio.to_thread`` uses a
# shared pool and can move consecutive requests between workers, making an
# otherwise warm model pay another multi-second first-use penalty. All local GPU
# phases are already protected by the application's asyncio GPU lock, so one
# dedicated worker preserves that state without reducing real concurrency.
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g3-whisper-gpu")

# Every GPU phase (ASR batch, ReDimNet slice, MMS_FA alignment, embedding,
# warmup, model reload) bumps this counter when it starts and when it ends. The
# batch worker's idle trim compares it, so the reserved pool is also released
# after diarization phases and embedding requests that never touch the ASR
# queue, and never in the middle of one. Only read and written on the event loop.
_gpu_activity_count = 0
_gpu_activity_listeners: List[Callable[[], None]] = []


def gpu_activity_count() -> int:
    return _gpu_activity_count


def add_gpu_activity_listener(listener: Callable[[], None]) -> None:
    if listener not in _gpu_activity_listeners:
        _gpu_activity_listeners.append(listener)


def remove_gpu_activity_listener(listener: Callable[[], None]) -> None:
    if listener in _gpu_activity_listeners:
        _gpu_activity_listeners.remove(listener)


def note_gpu_activity() -> None:
    global _gpu_activity_count
    _gpu_activity_count += 1
    for listener in list(_gpu_activity_listeners):
        try:
            listener()
        except Exception:
            # A listener is housekeeping; it must never fail a GPU phase.
            pass


@contextmanager
def shared_gpu_lease() -> Iterator[None]:
    """Serialize CUDA phases across processes when a shared lock path is set.

    The ordinary asyncio locks still serialize work inside each service.  This
    lease only adds host/container-wide coordination for deployments where the
    Whisper and DIA processes share one physical GPU and one filesystem volume.
    """

    configured_path = str(os.getenv("GENESIS_GPU_LEASE_PATH") or "").strip()
    if not configured_path:
        yield
        return

    from filelock import FileLock

    lock_path = Path(configured_path).expanduser().resolve(strict=False)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        yield


async def run_blocking_gpu_phase(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run every CUDA phase on one persistent host thread.

    Cancellation still waits for native work to finish before the caller can
    release its asyncio GPU lock. Counts as GPU activity for the idle trim.
    """

    note_gpu_activity()
    try:
        return await run_gpu_housekeeping(function, *args, **kwargs)
    finally:
        note_gpu_activity()


async def run_gpu_housekeeping(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """``run_blocking_gpu_phase`` without counting as activity (the idle trim itself)."""

    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(_gpu_executor, partial(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            # The client is already gone; the important invariant is that the
            # native thread has stopped touching CUDA before its async lock is
            # released. Runtime failures remain observable in server logs.
            pass
        raise


async def run_sliced_gpu_phase(
    lock: asyncio.Lock,
    items: Sequence[Any],
    function: Callable[..., T],
    slice_size: int,
    *args: Any,
) -> List[T]:
    """Run ``function(items[i:i+slice_size], *args)`` per slice, each under the lock.

    A long recording's alignment, speaker-cloud or verification phase used to be
    one lock hold of up to minutes, during which no live request could reach
    the GPU. Releasing the lock between slices lets the FIFO waiters (the ASR
    batch worker, live embeddings) run in between, so a bulk job never blocks a
    live request for longer than one slice. Results keep slice order.
    """

    slice_size = max(1, int(slice_size))
    results: List[T] = []
    for start in range(0, len(items), slice_size):
        async with lock:
            results.append(await run_blocking_gpu_phase(function, items[start : start + slice_size], *args))
        # Yield once so a waiter woken by the release really acquires the lock
        # before this phase queues up again behind it.
        await asyncio.sleep(0)
    return results


__all__ = [
    "add_gpu_activity_listener",
    "gpu_activity_count",
    "note_gpu_activity",
    "remove_gpu_activity_listener",
    "run_blocking_gpu_phase",
    "run_gpu_housekeeping",
    "run_sliced_gpu_phase",
    "shared_gpu_lease",
]
