"""Optional cross-process GPU coordination for colocated voice services."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Iterator
from typing import Any, Callable, Deque, Dict, List, Sequence, TypeVar

from . import genesis_whisper_server_progress as progress


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


# Two classes of GPU work. Interactive: everything that answers a request directly (the default:
# short transcripts and their language detection, embeddings, speaker turns of a live section, v1, the admin
# playground). Background: every phase of a diarization or a long transcript, including its ASR batches. The v2
# handler sets the class per request; it travels with the request's context into its tasks and worker threads.
INTERACTIVE = "interactive"
BACKGROUND = "background"
_gpu_class: ContextVar[str] = ContextVar("g3_gpu_class", default=INTERACTIVE)


def current_gpu_class() -> str:
    return _gpu_class.get()


def set_gpu_class(gpu_class: str) -> Any:
    """Make ``gpu_class`` the class of the current request's GPU work; returns the token for ``reset_gpu_class``."""

    if gpu_class not in (INTERACTIVE, BACKGROUND):
        raise ValueError(f"Unbekannte GPU-Klasse '{gpu_class}'.")
    return _gpu_class.set(gpu_class)


def reset_gpu_class(token: Any) -> None:
    _gpu_class.reset(token)


@contextmanager
def gpu_class_scope(gpu_class: str) -> Iterator[None]:
    token = set_gpu_class(gpu_class)
    try:
        yield
    finally:
        reset_gpu_class(token)


class PriorityGpuLock:
    """The process-wide local GPU lock: interactive waiters before background waiters, FIFO inside each class.

    A waiter's class is the caller's ``current_gpu_class()`` at ``acquire``. A release hands the lock straight to
    the first interactive waiter, else to the first background one, so a background waiter only gets the GPU
    while no interactive one waits. A running hold is never interrupted: bulk work stays preemptible only between
    its slices (run_sliced_gpu_phase). There is deliberately no ageing: background work waits as long as
    interactive work keeps arriving (owner decision). Drop-in for ``asyncio.Lock`` (``async with``,
    ``acquire``/``release``/``locked``).
    """

    def __init__(self) -> None:
        self._locked = False
        self._waiters: Dict[str, Deque[asyncio.Future]] = {INTERACTIVE: deque(), BACKGROUND: deque()}

    def locked(self) -> bool:
        return self._locked

    def waiting(self, gpu_class: str) -> int:
        return sum(1 for waiter in self._waiters[gpu_class] if not waiter.done())

    async def acquire(self) -> bool:
        if not self._locked:
            self._locked = True
            return True
        queue = self._waiters[current_gpu_class()]
        waiter = asyncio.get_running_loop().create_future()
        queue.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # Handed over just before the cancellation landed: pass the lock on.
                self.release()
            else:
                try:
                    queue.remove(waiter)
                except ValueError:
                    pass
            raise
        return True

    def release(self) -> None:
        if not self._locked:
            raise RuntimeError("GPU-Sperre ist nicht gehalten.")
        for gpu_class in (INTERACTIVE, BACKGROUND):
            queue = self._waiters[gpu_class]
            while queue:
                waiter = queue.popleft()
                if not waiter.done():
                    waiter.set_result(True)  # hand-off: the lock stays held, nobody can barge in between
                    return
        self._locked = False

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *_exc: Any) -> None:
        self.release()


@contextmanager
def shared_gpu_lease() -> Iterator[None]:
    """Serialize CUDA phases across processes when a shared lock path is set.

    The ordinary asyncio locks still serialize work inside each service.  This
    lease only adds host/container-wide coordination for deployments where the
    Whisper and another GPU process share one physical GPU and one filesystem volume.
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
    # Progress of the request this phase belongs to (docs/254): one unit per item, and the wait for a lock someone
    # else holds is reported as waiting. Both are no-ops outside a tracked request.
    progress.add_units(total=len(items))
    for start in range(0, len(items), slice_size):
        part = items[start : start + slice_size]
        locked = getattr(lock, "locked", None)
        wait = progress.begin_gpu_wait() if callable(locked) and locked() else None
        try:
            async with lock:
                progress.end_gpu_wait(wait)
                wait = None
                results.append(await run_blocking_gpu_phase(function, part, *args))
        finally:
            progress.end_gpu_wait(wait)
        progress.add_units(done=len(part))
        # Yield once so a waiter woken by the release really acquires the lock
        # before this phase queues up again behind it.
        await asyncio.sleep(0)
    return results


__all__ = [
    "BACKGROUND",
    "INTERACTIVE",
    "PriorityGpuLock",
    "add_gpu_activity_listener",
    "current_gpu_class",
    "gpu_class_scope",
    "gpu_activity_count",
    "note_gpu_activity",
    "remove_gpu_activity_listener",
    "reset_gpu_class",
    "run_blocking_gpu_phase",
    "run_gpu_housekeeping",
    "run_sliced_gpu_phase",
    "set_gpu_class",
    "shared_gpu_lease",
]
