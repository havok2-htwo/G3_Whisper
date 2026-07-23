import asyncio
import threading
import time
import unittest

from backend.genesis_whisper_server_gpu import run_blocking_gpu_phase


class DedicatedGpuExecutorTests(unittest.TestCase):
    def test_gpu_phases_reuse_one_host_thread_and_run_serially(self) -> None:
        thread_ids: list[int] = []
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def phase() -> int:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            thread_id = threading.get_ident()
            thread_ids.append(thread_id)
            time.sleep(0.01)
            with state_lock:
                active -= 1
            return thread_id

        async def run() -> list[int]:
            return await asyncio.gather(*(run_blocking_gpu_phase(phase) for _ in range(6)))

        results = asyncio.run(run())

        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len(set(thread_ids)), 1)
        self.assertEqual(max_active, 1)

    def test_gpu_phase_forwards_args_kwargs_and_result(self) -> None:
        async def run() -> int:
            return await run_blocking_gpu_phase(lambda first, *, second: first + second, 7, second=5)

        self.assertEqual(asyncio.run(run()), 12)


if __name__ == "__main__":
    unittest.main()
