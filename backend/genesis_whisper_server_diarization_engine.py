"""Request-side driver of the DIA-free diarization and the live ``speaker_turns``.

The building blocks are pure: ``genesis_whisper_server_segmentation`` (seg-3.0 port), ``..._local_diarization``
(the V4 batch pipeline on features) and ``..._speaker_turns`` (one live section). This module runs them inside a
request the way every other GPU phase of the server runs:

- every CUDA phase goes through ``run_sliced_gpu_phase`` under ``app.state.local_gpu_lock`` (seg-3.0 in slices of
  ``SEGMENTATION_SLICE_CHUNKS`` chunks, ReDimNet in slices of ``EMBED_SLICE_WINDOWS`` windows), so a long recording
  never blocks a live request for longer than one slice, and the batch worker's idle trim sees the activity;
- the CPU work (window planning, clustering, Viterbi, the turn logic) runs in worker threads, never on the loop.

Settings: ``diarization_engine`` ("dia" = the G3_DIA server, default; "local" = this pipeline),
``speaker_turns_enabled`` (default on, effective only while the seg-3.0 checkpoint is usable) and
``segmentation_model_path``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .genesis_whisper_server_globals import current_settings, settings_lock
from .genesis_whisper_server_gpu import run_sliced_gpu_phase
from .genesis_whisper_server_local_diarization import (
    BATCH_WINDOW_SCALES,
    GRID_S,
    LOCAL_DIARIZATION_MODEL_ID,
    SAMPLE_RATE,
    Features,
    assemble_windows,
    diarize_features,
    plan_scale_starts,
    result_to_dia_payload,
    speech_frames_from_seg,
)
from .genesis_whisper_server_segmentation import (
    SEGMENTATION_MODEL_NAME,
    chunk_count,
    segment_audio_range,
    segmentation_available,
    segmentation_payload,
)
from .genesis_whisper_server_speaker_turns import (
    SECTION_WINDOW_SCALES,
    section_frames,
    section_speech_mask,
    speaker_turns,
)
from .genesis_whisper_server_vid import (
    REDIMNET_WINDOW_SAMPLES,
    VoiceWindow,
    embed_voice_windows,
    prepare_window_length,
)


DIARIZATION_ENGINE_DIA = "dia"
DIARIZATION_ENGINE_LOCAL = "local"
DIARIZATION_ENGINES = (DIARIZATION_ENGINE_DIA, DIARIZATION_ENGINE_LOCAL)

# One GPU-lock hold: 128 seg-3.0 chunks are four batches of 32 (~0.1 s on an RTX 2070S); 64 ReDimNet windows are
# eight batches of 8, the same slice the speaker clouds use.
SEGMENTATION_SLICE_CHUNKS = 128
EMBED_SLICE_WINDOWS = 64
# speaker_turns is made for live sections (Genesis3 cuts them at 15 s). Longer audio gets no turns (a warning), the
# transcript stays: one forced alignment pass over a whole minute is the upper bound kept for MMS_FA.
SPEAKER_TURNS_MAX_AUDIO_S = 60.0
# Window speech filter of the batch pipeline: "silero" as in the benchmark (window kept with >= 30 % Silero speech
# frames), or "seg" (the seg-3.0 speech mask, no Silero pass).
LOCAL_WINDOW_FILTER = "silero"


# ---------------------------------------------------------------------------
# Settings and capabilities
# ---------------------------------------------------------------------------

def configured_diarization_engine(settings: Optional[Mapping[str, Any]] = None) -> str:
    if settings is None:
        with settings_lock:
            value = current_settings.get("diarization_engine", DIARIZATION_ENGINE_DIA)
    else:
        value = settings.get("diarization_engine", DIARIZATION_ENGINE_DIA)
    return value if value in DIARIZATION_ENGINES else DIARIZATION_ENGINE_DIA


def speaker_turns_switch(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """The admin switch ``speaker_turns_enabled``; a settings dict without the key counts as on."""

    if settings is None:
        with settings_lock:
            return current_settings.get("speaker_turns_enabled", True) is True
    return settings.get("speaker_turns_enabled", True) is True


def speaker_turns_available() -> bool:
    """speaker_turns is offered: the switch is on and a seg-3.0 checkpoint is usable (no model load here)."""

    return speaker_turns_switch() and segmentation_available()


def segmentation_needed(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Some feature uses the seg-3.0 model: the local engine or the speaker_turns switch."""

    return configured_diarization_engine(settings) == DIARIZATION_ENGINE_LOCAL or speaker_turns_switch(settings)


def diarization_capability(dia_configured: bool) -> Dict[str, Any]:
    """The ``diarization`` block of ``GET /v2/capabilities``: the engine and whether it can run."""

    engine = configured_diarization_engine()
    if engine == DIARIZATION_ENGINE_LOCAL:
        return {"configured": segmentation_available(), "engine": engine, "model": LOCAL_DIARIZATION_MODEL_ID}
    return {"configured": dia_configured, "engine": engine}


# ---------------------------------------------------------------------------
# GPU phases (sliced under the request's GPU lock)
# ---------------------------------------------------------------------------

async def segment_sliced(lock: asyncio.Lock, audio: np.ndarray) -> Dict[str, Any]:
    """seg-3.0 arrays of ``audio``, ``SEGMENTATION_SLICE_CHUNKS`` chunks per lock hold."""

    total = chunk_count(len(audio))
    parts = await run_sliced_gpu_phase(lock, list(range(total)), segment_audio_range, SEGMENTATION_SLICE_CHUNKS, audio)
    return segmentation_payload(np.concatenate(parts) if parts else segment_audio_range([], audio))


def _embed_native(windows: Sequence[VoiceWindow], window_samples: int) -> List[np.ndarray]:
    """GPU unit: embed windows at their native length (warming that length's batch shapes once)."""

    if window_samples != REDIMNET_WINDOW_SAMPLES:
        prepare_window_length(window_samples)
    return [item.vector for item in embed_voice_windows(windows, window_samples=window_samples)]


async def embed_scales(
    lock: asyncio.Lock,
    audio: np.ndarray,
    starts: Mapping[str, np.ndarray],
    scales: Mapping[str, tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """ReDimNet2 vectors (N x 192) of the planned window starts of every scale, in plan order."""

    out: Dict[str, np.ndarray] = {}
    for name, (length_s, _) in scales.items():
        n = int(round(length_s * SAMPLE_RATE))
        windows = [
            VoiceWindow(audio=audio[int(s): int(s) + n], start_ms=round(int(s) * 1000 / SAMPLE_RATE),
                        end_ms=round((int(s) + n) * 1000 / SAMPLE_RATE), clean_duration_seconds=n / SAMPLE_RATE)
            for s in starts[name]
        ]
        if not windows:
            out[name] = np.zeros((0, 192), np.float32)
            continue
        parts = await run_sliced_gpu_phase(lock, windows, partial(_embed_native, window_samples=n), EMBED_SLICE_WINDOWS)
        out[name] = np.stack([vector for part in parts for vector in part]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Batch diarization (diarization_engine = "local")
# ---------------------------------------------------------------------------

async def run_local_diarization(
    lock: asyncio.Lock,
    audio: np.ndarray,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    speech_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """The G3_DIA ``/v2/diarize`` payload of ``audio`` computed in this process.

    ``speech_probs``: Silero probabilities per 512 samples when the caller already has them (the WXC path reuses
    the same array); computed here otherwise. RuntimeError (-> 503) when the seg-3.0 model cannot be loaded.
    """

    started = time.monotonic()
    samples = np.ascontiguousarray(np.asarray(audio, np.float32).reshape(-1))
    duration_s = len(samples) / SAMPLE_RATE
    segmentation = await segment_sliced(lock, samples)
    if LOCAL_WINDOW_FILTER == "silero":
        if speech_probs is None:
            from .genesis_whisper_server_wxc import silero_frame_probs

            speech_probs = await asyncio.to_thread(silero_frame_probs, samples)
        speech, frame_samples = np.asarray(speech_probs) > 0.5, 512
    else:
        speech = await asyncio.to_thread(speech_frames_from_seg, segmentation, duration_s)
        frame_samples = int(GRID_S * SAMPLE_RATE)
    starts = await asyncio.to_thread(plan_scale_starts, len(samples), speech, frame_samples, BATCH_WINDOW_SCALES)
    vectors = await embed_scales(lock, samples, starts, BATCH_WINDOW_SCALES)

    def _cpu() -> Dict[str, Any]:
        windows = assemble_windows(starts, vectors, BATCH_WINDOW_SCALES)
        result = diarize_features(Features(segmentation, windows, duration_s), num_speakers=num_speakers,
                                  min_speakers=min_speakers, max_speakers=max_speakers)
        return result_to_dia_payload(result, round(duration_s * 1000), num_speakers=num_speakers,
                                     min_speakers=min_speakers, max_speakers=max_speakers, started=started)

    payload = await asyncio.to_thread(_cpu)
    payload["model"] = {
        "id": LOCAL_DIARIZATION_MODEL_ID,
        "engine": DIARIZATION_ENGINE_LOCAL,
        "segmentation": SEGMENTATION_MODEL_NAME,
        "window_filter": LOCAL_WINDOW_FILTER,
    }
    return payload


# ---------------------------------------------------------------------------
# Live speaker_turns (transcript mode)
# ---------------------------------------------------------------------------

async def compute_section_turns(
    lock: asyncio.Lock,
    audio: np.ndarray,
    known_speakers: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Turns (without text) of one section: seg-3.0 mask, 1.5 s / 3 s windows, the frozen speaker_turns logic."""

    samples = np.ascontiguousarray(np.asarray(audio, np.float32).reshape(-1))
    segmentation = await segment_sliced(lock, samples)
    n = section_frames(len(samples))
    mask = await asyncio.to_thread(section_speech_mask, segmentation, n)
    starts = await asyncio.to_thread(
        plan_scale_starts, len(samples), mask, int(GRID_S * SAMPLE_RATE), SECTION_WINDOW_SCALES
    )
    vectors = await embed_scales(lock, samples, starts, SECTION_WINDOW_SCALES)

    def _cpu() -> List[Dict[str, Any]]:
        windows = assemble_windows(starts, vectors, SECTION_WINDOW_SCALES)
        section = {"dur": len(samples) / SAMPLE_RATE, "sm": mask, "w15": windows["w15"], "w30": windows["w30"]}
        return speaker_turns(section, known_speakers)

    return await asyncio.to_thread(_cpu)


def section_words(text: str, aligned: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every word of ``text`` with a time: the forced-alignment times where MMS_FA had them, else the time of the
    previous aligned word (the next one before the first). MMS_FA only aligns words with latin letters, so numbers
    and symbols ("21.", "%") would otherwise vanish from the turn texts. [] when nothing was aligned."""

    from .genesis_whisper_server_turn_gate import clean_segment_text

    tokens = clean_segment_text(text).split()
    if not aligned or not tokens:
        return []
    timed: List[Optional[tuple[float, float]]] = []
    j = 0
    for token in tokens:
        if j < len(aligned) and str(aligned[j].get("word")) == token:
            timed.append((float(aligned[j]["t0"]), float(aligned[j]["t1"])))
            j += 1
        else:
            timed.append(None)
    known = [t for t in timed if t is not None]
    if not known:
        return []
    out: List[Dict[str, Any]] = []
    last = known[0]
    for token, times in zip(tokens, timed):
        if times is not None:
            last = times
        out.append({"t0": last[0], "t1": last[1], "word": token})
    return out


def warmup_segmentation() -> bool:
    """Startup (GPU thread): load seg-3.0, run it once on 10 s of quiet noise and warm ReDimNet's 1.5 s shapes,
    only where a feature uses them. False when skipped or unavailable."""

    from .genesis_whisper_server_segmentation import load_segmentation_model

    if not segmentation_needed():
        print("[WARMUP-SEG] Lokale Diarisierung und speaker_turns aus -> uebersprungen.", file=sys.stderr)
        return False
    model = load_segmentation_model()
    if model is None:
        print("[WARMUP-SEG] Kein Segmentierungsmodell; speaker_turns bleibt aus, engine local antwortet 503.",
              file=sys.stderr)
        return False
    from .genesis_whisper_server_gpu import shared_gpu_lease

    rng = np.random.default_rng(0)
    with shared_gpu_lease():
        model.segment((rng.standard_normal(10 * SAMPLE_RATE) * 1e-3).astype(np.float32))
    prepare_window_length(int(SECTION_WINDOW_SCALES["w15"][0] * SAMPLE_RATE))
    print("[WARMUP-SEG] Segmentierung (seg-3.0) und ReDimNet 1,5 s sind bereit.", file=sys.stderr)
    return True


__all__ = [
    "DIARIZATION_ENGINES",
    "DIARIZATION_ENGINE_DIA",
    "DIARIZATION_ENGINE_LOCAL",
    "SPEAKER_TURNS_MAX_AUDIO_S",
    "compute_section_turns",
    "configured_diarization_engine",
    "diarization_capability",
    "embed_scales",
    "run_local_diarization",
    "section_words",
    "segment_sliced",
    "segmentation_needed",
    "speaker_turns_available",
    "speaker_turns_switch",
    "warmup_segmentation",
]
