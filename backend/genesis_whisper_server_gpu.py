"""Optional cross-process GPU coordination for colocated voice services."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from typing import Any, Callable, TypeVar


T = TypeVar("T")


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
    """Run a GPU thread without releasing the caller's async lock on cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
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


__all__ = ["run_blocking_gpu_phase", "shared_gpu_lease"]
