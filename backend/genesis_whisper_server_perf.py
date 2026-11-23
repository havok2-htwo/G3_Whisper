"""Performance graph data for the admin dashboard (shared G3 contract, schema 1: GET /api/admin/perf).

Service metrics:
- ``throughput``: realtime factor = audio seconds finished by the ASR batch worker per wall second, averaged over a
  sliding 30 s window. It falls to 0 within 30 s once work stops (never "the last batch's value").
- ``queue``: ASR chunks waiting, of active and of waiting (long) jobs.

GPU identity, utilisation and VRAM come from the shared ``g3_gpu`` helper (UUID matched, nvidia-smi numbering).
"""
from __future__ import annotations

from typing import Any, Optional

from .g3_gpu import RateWindow, perf_payload
from .genesis_whisper_server_globals import batch_runtime_state, batch_state_lock

SERVICE_NAME = "G3 Whisper"
RATE_WINDOW_SECONDS = 30

# Fed by WhisperBatchManager._process_batch with the audio seconds of every finished batch (all ASR requests,
# diarization transcripts, history retries and the admin benchmark go through it).
asr_audio_rate = RateWindow(RATE_WINDOW_SECONDS)


def queued_chunks(snapshot: Optional[dict[str, Any]] = None) -> int:
    """Chunks not yet in a batch: pending chunks of active jobs plus those of waiting long jobs."""

    if snapshot is None:
        with batch_state_lock:
            snapshot = dict(batch_runtime_state)
    return max(0, int(snapshot.get("queue_size") or 0)) + max(0, int(snapshot.get("pending_buffer_size") or 0))


def service_metrics(rate: Optional[RateWindow] = None, snapshot: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    window = rate if rate is not None else asr_audio_rate
    return [
        {
            "key": "throughput",
            "label": "Realtime factor",
            "unit": "x",
            "value": round(window.rate(), 3),
            "decimals": 1,
            "window_s": RATE_WINDOW_SECONDS,
        },
        {
            "key": "queue",
            "label": "Queue (chunks)",
            "unit": "",
            "value": queued_chunks(snapshot),
            "decimals": 0,
            "integer": True,
        },
    ]


def perf_snapshot(batch_manager: Any = None) -> dict[str, Any]:
    """Blocking (nvidia-smi / NVML): call through asyncio.to_thread."""

    error = None
    if batch_manager is None:
        error = "The ASR batch worker is not running."
    return perf_payload(SERVICE_NAME, service_metrics(), error)
