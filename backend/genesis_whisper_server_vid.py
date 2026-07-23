"""ReDimNet2-B6 speaker embeddings used by every public voice-vector path."""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

import numpy as np
import torch

from .genesis_whisper_server_globals import (
    PROJECT_ROOT,
    current_settings,
    resolve_local_model_cache_path,
    settings_lock,
)
from .genesis_whisper_server_gpu import shared_gpu_lease


REDIMNET_MODEL_NAME = "ReDimNet2-B6"
REDIMNET_MODEL_VARIANT = "vb2+vox2+cnc2_v0-lm"
REDIMNET_MODEL_RELEASE = "v1.0.0"
# The v1.0.0 release asset below uses the post-tag ``agg_gnorm`` model
# configuration.  Pin the first MIT-licensed official source revision that can
# instantiate that released checkpoint; the original v1.0.0 tag predates this
# configuration and fails with ``unexpected keyword argument 'agg_gnorm'``.
REDIMNET_HUB_COMMIT = "2a8d15f65b1dfb5d73fede2f11ee42bcccca3035"
REDIMNET_HUB_REPOSITORY = f"PalabraAI/redimnet2:{REDIMNET_HUB_COMMIT}"
REDIMNET_CHECKPOINT_NAME = "b6-vb2+vox2+cnc2_v0-lm.pt"
REDIMNET_CHECKPOINT_URL = (
    "https://github.com/PalabraAI/redimnet2/releases/download/"
    f"{REDIMNET_MODEL_RELEASE}/{REDIMNET_CHECKPOINT_NAME}"
)
REDIMNET_CHECKPOINT_SHA256 = "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868"
REDIMNET_EMBEDDING_DIMENSION = 192
REDIMNET_SAMPLE_RATE = 16000
REDIMNET_WINDOW_SECONDS = 3.0
REDIMNET_WINDOW_SAMPLES = int(REDIMNET_SAMPLE_RATE * REDIMNET_WINDOW_SECONDS)
REDIMNET_MIN_WINDOW_SECONDS = 0.5
REDIMNET_MIN_WINDOW_SAMPLES = int(REDIMNET_SAMPLE_RATE * REDIMNET_MIN_WINDOW_SECONDS)
# Batch 16 is the measured throughput/memory sweet spot for B6 on the target
# GPU; larger batches consume substantially more VRAM without improving steady
# state throughput. OOM handling below still learns a smaller request-local
# size for constrained cards.
REDIMNET_DEFAULT_BATCH_SIZE = 16
REDIMNET_MIN_CUDA_BATCH_SIZE = 2
_ENERGY_VAD_FRAME_MS = 30
_ENERGY_VAD_BATCH_FRAMES = 8192
_ENERGY_VAD_MIN_RUN_MS = 150
_ENERGY_VAD_MERGE_GAP_MS = 300


@dataclass(frozen=True)
class VoiceWindow:
    audio: np.ndarray
    start_ms: int | None = None
    end_ms: int | None = None
    clean_duration_seconds: float = REDIMNET_WINDOW_SECONDS
    quality: float = 1.0
    stitched: bool = False
    source_spans: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class EmbeddedVoiceWindow:
    vector: np.ndarray
    start_ms: int | None
    end_ms: int | None
    clean_duration_seconds: float
    quality: float
    stitched: bool
    source_spans: tuple[tuple[int, int], ...] | None = None


_redimnet_lock = threading.Lock()
_redimnet_components: Dict[str, Any] = {
    "model": None,
    "device": None,
    "dtype": None,
    "cache_root": None,
    "cuda_batch_size": None,
}


def embedding_model_metadata() -> Dict[str, Any]:
    return {
        "id": REDIMNET_MODEL_NAME,
        "variant": REDIMNET_MODEL_VARIANT,
        "release": REDIMNET_MODEL_RELEASE,
        "source_commit": REDIMNET_HUB_COMMIT,
        "checkpoint_sha256": REDIMNET_CHECKPOINT_SHA256,
        "dimension": REDIMNET_EMBEDDING_DIMENSION,
        "normalization": "l2",
        "sample_rate": REDIMNET_SAMPLE_RATE,
    }


def _model_cache_root() -> Path:
    with settings_lock:
        configured = str(current_settings.get("local_model_cache_path", "")).strip()
    resolved = resolve_local_model_cache_path(configured)
    return Path(resolved) if resolved else Path(PROJECT_ROOT) / "models"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            block = file_obj.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _ensure_verified_checkpoint(hub_dir: Path) -> Path:
    checkpoint_dir = hub_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / REDIMNET_CHECKPOINT_NAME

    if checkpoint_path.is_file() and _sha256_file(checkpoint_path) == REDIMNET_CHECKPOINT_SHA256:
        return checkpoint_path

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="redimnet2-download-",
            suffix=".pt",
            dir=str(checkpoint_dir),
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            request = urllib.request.Request(
                REDIMNET_CHECKPOINT_URL,
                headers={"User-Agent": "G3-WHISPER/2.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    temp_file.write(block)

        actual_sha256 = _sha256_file(temp_path)
        if actual_sha256 != REDIMNET_CHECKPOINT_SHA256:
            raise RuntimeError(
                "ReDimNet2-Checkpoint hat eine unerwartete SHA256-Pruefsumme "
                f"({actual_sha256})."
            )
        os.replace(temp_path, checkpoint_path)
        temp_path = None
        return checkpoint_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def _resolve_device() -> torch.device:
    with settings_lock:
        configured = str(current_settings.get("local_gpu_device", "auto")).strip().lower()
    if torch.cuda.is_available() and configured != "cpu":
        if configured.startswith("cuda"):
            return torch.device(configured)
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _model_autocast(device: torch.device, dtype: torch.dtype):
    return (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )


def _run_model_warmup(
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    waveform = torch.zeros(
        (batch_size, REDIMNET_WINDOW_SAMPLES),
        dtype=torch.float32,
        device=device,
    )
    embedding = None
    try:
        with torch.inference_mode(), _model_autocast(device, dtype):
            embedding = model(waveform)
        if embedding.ndim != 2 or embedding.shape != (
            batch_size,
            REDIMNET_EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "ReDimNet2 lieferte beim Warmup eine unerwartete Batch-Form "
                f"({tuple(embedding.shape)})."
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        del waveform, embedding


def _warm_model_shapes(
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """Warm the only CUDA shapes used in production: one and the prepared max.

    ReDimNet/PyTorch incurs roughly 2.5 seconds of one-time kernel preparation for
    every previously unseen batch dimension on the target stack. Running arbitrary
    tail sizes therefore caused later short requests to spike even though the model
    itself was already resident. Production inference now uses batch 1 for the
    common single-window microphone case and pads every larger batch to one prepared
    maximum. That keeps the fast path to two stable CUDA shapes without compiling all
    16 possible dimensions or retaining their individual workspaces.
    """

    _run_model_warmup(model, device, dtype, 1)
    if device.type != "cuda" or REDIMNET_DEFAULT_BATCH_SIZE == 1:
        return REDIMNET_DEFAULT_BATCH_SIZE

    candidate = REDIMNET_DEFAULT_BATCH_SIZE
    while candidate >= REDIMNET_MIN_CUDA_BATCH_SIZE:
        try:
            _run_model_warmup(model, device, dtype, candidate)
            return candidate
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            candidate //= 2
            if torch.cuda.is_available():
                # Emergency-only recovery. Normal operation deliberately keeps the
                # prepared allocator state; an OOM cannot be retried while failed
                # allocations are still retained by the process cache.
                torch.cuda.empty_cache()
            print(
                "[WARNUNG-VID] ReDimNet2-Warmup hatte zu wenig CUDA-Speicher; "
                f"reduziere vorbereitete Batchgroesse auf {candidate}.",
                file=sys.stderr,
            )

    return 1


def _inference_batch_size(current_size: int, device_type: str, prepared_cuda_batch_size: int) -> int:
    """Return a stable model batch shape while preserving the one-window fast path."""

    if current_size <= 0:
        raise ValueError("current_size muss positiv sein.")
    if device_type != "cuda" or current_size == 1:
        return current_size
    prepared = max(1, int(prepared_cuda_batch_size))
    if current_size > prepared:
        raise ValueError("current_size darf die vorbereitete CUDA-Batchgroesse nicht ueberschreiten.")
    return prepared


def _prepare_waveform_batch(
    windows: Sequence[VoiceWindow],
    inference_batch_size: int,
) -> np.ndarray:
    if not windows or inference_batch_size < len(windows):
        raise ValueError("Ungueltige ReDimNet2-Inferenz-Batchgroesse.")
    batch = np.zeros(
        (inference_batch_size, REDIMNET_WINDOW_SAMPLES),
        dtype=np.float32,
    )
    for index, window in enumerate(windows):
        batch[index] = _fit_window(window.audio)
    return batch


def load_vid_model() -> bool:
    """Load the single pinned ReDimNet2 model once and verify its weights."""

    with _redimnet_lock:
        if _redimnet_components.get("model") is not None:
            return True

        try:
            cache_root = _model_cache_root()
            hub_dir = cache_root / ".torch" / "hub"
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))
            _ensure_verified_checkpoint(hub_dir)

            device = _resolve_device()
            cuda_allocated_before = 0
            cuda_reserved_before = 0
            if device.type == "cuda":
                cuda_allocated_before = int(torch.cuda.memory_allocated(device))
                cuda_reserved_before = int(torch.cuda.memory_reserved(device))
            print(
                f"[INFO-VID] Lade {REDIMNET_MODEL_NAME} ({REDIMNET_MODEL_VARIANT}) auf {device}...",
                file=sys.stderr,
            )
            with shared_gpu_lease():
                model = torch.hub.load(
                    REDIMNET_HUB_REPOSITORY,
                    "redimnet2",
                    model_name="b6",
                    train_type="lm",
                    dataset="vb2+vox2+cnc2_v0",
                    pretrained=True,
                    trust_repo=True,
                    skip_validation=True,
                    verbose=False,
                )
                model.eval()
                model.to(device)
                # ReDimNet2's waveform frontend intentionally runs in FP32 and
                # casts features back to the input dtype.  Keeping parameters in
                # FP32 while using CUDA autocast runs the neural backbone in
                # FP16 without breaking the frontend's FP32 pre-emphasis buffer.
                dtype = torch.float16 if device.type == "cuda" else torch.float32
                warmup_started = time.monotonic()
                cuda_batch_size = _warm_model_shapes(model, device, dtype)
                warmup_duration_ms = round((time.monotonic() - warmup_started) * 1000)

            _redimnet_components.update(
                {
                    "model": model,
                    "device": device,
                    "dtype": dtype,
                    "cache_root": str(cache_root),
                    "cuda_batch_size": cuda_batch_size,
                }
            )
            memory_note = ""
            if device.type == "cuda":
                allocated_delta = max(0, int(torch.cuda.memory_allocated(device)) - cuda_allocated_before)
                reserved_delta = max(0, int(torch.cuda.memory_reserved(device)) - cuda_reserved_before)
                memory_note = (
                    f" Zusatz-VRAM: {allocated_delta / (1024 ** 2):.0f} MiB allokiert, "
                    f"{reserved_delta / (1024 ** 2):.0f} MiB reserviert."
                )
            print(
                "[INFO-VID] ReDimNet2 erfolgreich fuer CUDA-Batchformen "
                f"1 und {cuda_batch_size} aufgewaermt ({warmup_duration_ms} ms).{memory_note}",
                file=sys.stderr,
            )
            return True
        except Exception as exc:
            _redimnet_components.update(
                {
                    "model": None,
                    "device": None,
                    "dtype": None,
                    "cache_root": None,
                    "cuda_batch_size": None,
                }
            )
            print(f"[FEHLER-VID] ReDimNet2 konnte nicht geladen werden: {exc}", file=sys.stderr)
            return False


def _fit_window(audio: np.ndarray, target_samples: int = REDIMNET_WINDOW_SAMPLES) -> np.ndarray:
    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(flattened) == target_samples:
        return flattened
    if len(flattened) > target_samples:
        return flattened[:target_samples]
    if len(flattened) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    repetitions = int(math.ceil(target_samples / len(flattened)))
    return np.tile(flattened, repetitions)[:target_samples].astype(np.float32, copy=False)


def _audio_quality(audio: np.ndarray) -> float | None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(samples) == 0 or not np.all(np.isfinite(samples)):
        return None
    # ``np.square`` used to allocate one temporary the size of every 3-second
    # window.  BLAS dot is allocation-free and substantially faster; FP32
    # accumulation changes only sub-ULP quality weights for normalized audio.
    clipped_count = int(np.count_nonzero(samples >= 0.999))
    clipped_count += int(np.count_nonzero(samples <= -0.999))
    clipped_ratio = clipped_count / len(samples)
    if clipped_ratio > 0.01:
        return None
    squared_sum = float(np.dot(samples, samples))
    if not np.isfinite(squared_sum) or squared_sum < 0.0:
        return None
    rms = math.sqrt(squared_sum / len(samples))
    if rms <= 10 ** (-50.0 / 20.0):
        return None
    # Preserve low-volume but valid speech with a smaller influence.
    loudness_weight = min(1.0, max(0.4, rms / 0.08))
    return loudness_weight


def iter_windows_from_audio(
    audio: np.ndarray,
    *,
    start_ms: int | None = None,
    stitched: bool = False,
    minimum_samples: int = REDIMNET_MIN_WINDOW_SAMPLES,
) -> Iterator[VoiceWindow]:
    """Yield fixed model windows without retaining a second audio-sized list."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    for offset in range(0, len(samples), REDIMNET_WINDOW_SAMPLES):
        chunk = samples[offset : offset + REDIMNET_WINDOW_SAMPLES]
        if len(chunk) < minimum_samples:
            continue
        quality = _audio_quality(chunk)
        if quality is None:
            continue
        duration_seconds = len(chunk) / float(REDIMNET_SAMPLE_RATE)
        chunk_start_ms = None if start_ms is None else start_ms + round(offset * 1000 / REDIMNET_SAMPLE_RATE)
        chunk_end_ms = None if chunk_start_ms is None else chunk_start_ms + round(duration_seconds * 1000)
        if stitched:
            quality *= 0.6
        yield VoiceWindow(
            audio=_fit_window(chunk),
            start_ms=chunk_start_ms,
            end_ms=chunk_end_ms,
            clean_duration_seconds=duration_seconds,
            quality=quality,
            stitched=stitched,
        )


def windows_from_audio(
    audio: np.ndarray,
    *,
    start_ms: int | None = None,
    stitched: bool = False,
    minimum_samples: int = REDIMNET_MIN_WINDOW_SAMPLES,
) -> list[VoiceWindow]:
    """Compatibility wrapper for callers that explicitly need a list."""

    return list(
        iter_windows_from_audio(
            audio,
            start_ms=start_ms,
            stitched=stitched,
            minimum_samples=minimum_samples,
        )
    )


def _detect_embedding_speech_segments(
    audio: np.ndarray,
    *,
    padding_ms: int = 120,
) -> list[tuple[int, int]]:
    """Energy VAD equivalent to the legacy extractor, but vectorized in blocks.

    A two-hour file has roughly 240k VAD frames.  Processing those frames one
    Python call at a time dominated the CPU phase and the old helper also made
    two audio-sized copies.  Blocks keep the largest temporary below 16 MiB.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(samples) == 0:
        return []
    frame_samples = int(REDIMNET_SAMPLE_RATE * _ENERGY_VAD_FRAME_MS / 1000)
    if len(samples) < frame_samples:
        if float(np.max(np.abs(samples))) < 0.0001:
            return []
        return [(0, len(samples))]

    frame_count = (len(samples) - frame_samples) // frame_samples + 1
    energies = np.empty(frame_count, dtype=np.float32)
    for first_frame in range(0, frame_count, _ENERGY_VAD_BATCH_FRAMES):
        last_frame = min(frame_count, first_frame + _ENERGY_VAD_BATCH_FRAMES)
        block = samples[first_frame * frame_samples : last_frame * frame_samples]
        frames = block.reshape(last_frame - first_frame, frame_samples)
        # Keep the original float32 square/mean semantics, just apply them to
        # many frames at once.  This is exactly equal for ordinary input in
        # NumPy, including the percentile-derived threshold below.
        energies[first_frame:last_frame] = np.sqrt(
            np.mean(np.square(frames), axis=1)
        )

    if len(energies) == 0 or float(np.max(energies)) < 0.0001:
        return []
    noise_floor = float(np.percentile(energies, 5))
    speech_peak = float(np.percentile(energies, 95))
    threshold = max(0.0001, noise_floor + (speech_peak - noise_floor) * 0.05)
    voiced = energies >= threshold

    # Find true runs in vectorized form.  ``ends`` are exclusive frame indices.
    bounded = np.empty(len(voiced) + 2, dtype=np.bool_)
    bounded[0] = False
    bounded[-1] = False
    bounded[1:-1] = voiced
    transitions = np.flatnonzero(bounded[1:] != bounded[:-1])
    run_starts = transitions[0::2]
    run_ends = transitions[1::2]
    min_run_frames = max(1, int(_ENERGY_VAD_MIN_RUN_MS / _ENERGY_VAD_FRAME_MS))
    padding_samples = int(padding_ms * REDIMNET_SAMPLE_RATE / 1000)
    merge_gap_samples = int(_ENERGY_VAD_MERGE_GAP_MS * REDIMNET_SAMPLE_RATE / 1000)

    merged: list[tuple[int, int]] = []
    for run_start, run_end in zip(run_starts.tolist(), run_ends.tolist()):
        if run_end - run_start < min_run_frames:
            continue
        start_sample = max(0, run_start * frame_samples - padding_samples)
        end_sample = min(len(samples), run_end * frame_samples + padding_samples)
        if merged and start_sample <= merged[-1][1] + merge_gap_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sample))
        elif end_sample > start_sample:
            merged.append((start_sample, end_sample))
    return merged


def _iter_concatenated_segment_windows(
    audio: np.ndarray,
    segments: Sequence[tuple[int, int]],
) -> Iterator[VoiceWindow]:
    """Window the logical concatenation of VAD spans with bounded memory."""

    segment_index = 0
    segment_position = segments[0][0] if segments else 0
    while segment_index < len(segments):
        pieces: list[np.ndarray] = []
        collected = 0
        while collected < REDIMNET_WINDOW_SAMPLES and segment_index < len(segments):
            segment_start, segment_end = segments[segment_index]
            segment_position = max(segment_position, segment_start)
            available = max(0, segment_end - segment_position)
            if available == 0:
                segment_index += 1
                if segment_index < len(segments):
                    segment_position = segments[segment_index][0]
                continue
            take = min(REDIMNET_WINDOW_SAMPLES - collected, available)
            pieces.append(audio[segment_position : segment_position + take])
            segment_position += take
            collected += take
            if segment_position >= segment_end:
                segment_index += 1
                if segment_index < len(segments):
                    segment_position = segments[segment_index][0]

        if collected < REDIMNET_MIN_WINDOW_SAMPLES:
            break
        chunk = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        quality = _audio_quality(chunk)
        if quality is None:
            continue
        yield VoiceWindow(
            audio=_fit_window(chunk),
            clean_duration_seconds=collected / float(REDIMNET_SAMPLE_RATE),
            quality=quality,
        )


def embed_voice_windows(
    windows: Iterable[VoiceWindow],
    *,
    batch_size: int = REDIMNET_DEFAULT_BATCH_SIZE,
) -> list[EmbeddedVoiceWindow]:
    effective_batch_size = max(1, int(batch_size))
    window_iterator = iter(windows)
    pending = list(itertools.islice(window_iterator, effective_batch_size))
    if not pending:
        return []
    if not load_vid_model():
        raise RuntimeError("ReDimNet2-B6 LM konnte nicht geladen werden.")

    model = _redimnet_components["model"]
    device: torch.device = _redimnet_components["device"]
    dtype: torch.dtype = _redimnet_components["dtype"]
    prepared_cuda_batch_size = max(
        1,
        int(_redimnet_components.get("cuda_batch_size") or REDIMNET_DEFAULT_BATCH_SIZE),
    )
    effective_batch_size = min(effective_batch_size, prepared_cuda_batch_size)
    output: list[EmbeddedVoiceWindow] = []

    with shared_gpu_lease():
        while pending:
            current_size = min(effective_batch_size, len(pending))
            current = pending[:current_size]
            model_batch_size = _inference_batch_size(
                current_size,
                device.type,
                prepared_cuda_batch_size,
            )
            batch_np = _prepare_waveform_batch(current, model_batch_size)
            waveform = None
            embeddings = None
            retry_after_oom = False
            try:
                waveform = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                with torch.inference_mode(), _model_autocast(device, dtype):
                    embeddings = model(waveform)
                embeddings_np = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and current_size > 1:
                    prepared_cuda_batch_size = max(1, model_batch_size // 2)
                    effective_batch_size = min(
                        max(1, current_size // 2),
                        prepared_cuda_batch_size,
                    )
                    _redimnet_components["cuda_batch_size"] = prepared_cuda_batch_size
                    retry_after_oom = True
                else:
                    raise
            finally:
                # Device tensors must not survive into the next batch.  This is
                # especially important after an OOM backoff on a shared GPU.
                del waveform, embeddings

            if retry_after_oom:
                # Run this after the exception context and device references
                # are gone; otherwise PyTorch cannot release all cached blocks.
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if embeddings_np.ndim != 2 or embeddings_np.shape[0] != model_batch_size:
                raise RuntimeError("ReDimNet2 lieferte eine unerwartete Batch-Form.")

            for window, vector in zip(current, embeddings_np[:current_size]):
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if len(vector) != REDIMNET_EMBEDDING_DIMENSION or not np.all(np.isfinite(vector)):
                    raise RuntimeError("ReDimNet2 lieferte einen ungueltigen Vektor.")
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    raise RuntimeError("ReDimNet2 lieferte einen Nullvektor.")
                output.append(
                    EmbeddedVoiceWindow(
                        vector=(vector / norm).astype(np.float32, copy=False),
                        start_ms=window.start_ms,
                        end_ms=window.end_ms,
                        clean_duration_seconds=window.clean_duration_seconds,
                        quality=window.quality,
                        stitched=window.stitched,
                        source_spans=window.source_spans,
                    )
                )
            del pending[:current_size]
            pending.extend(
                itertools.islice(
                    window_iterator,
                    max(0, effective_batch_size - len(pending)),
                )
            )

    return output


def _weighted_normalized_mean(vectors: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    vector_list = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    weight_array = np.asarray(list(weights), dtype=np.float32)
    if not vector_list or len(vector_list) != len(weight_array):
        raise ValueError("Keine gueltigen ReDimNet2-Vektoren vorhanden.")
    matrix = np.stack(vector_list, axis=0)
    mean = np.average(matrix, axis=0, weights=np.maximum(weight_array, 1e-6))
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("ReDimNet2-Vektoren konnten nicht zusammengefuehrt werden.")
    return (mean / norm).astype(np.float32, copy=False)


def generate_voice_vector(audio_data_np: np.ndarray) -> np.ndarray:
    """Generate the one public 192-D embedding used by legacy and v2 APIs."""

    audio = np.asarray(audio_data_np, dtype=np.float32).reshape(-1)
    speech_segments = _detect_embedding_speech_segments(audio)
    speech_sample_count = sum(end - start for start, end in speech_segments)
    if speech_sample_count < REDIMNET_MIN_WINDOW_SAMPLES:
        raise ValueError(
            "Zu wenig Sprache fuer ein Stimmembedding: mindestens "
            f"{REDIMNET_MIN_WINDOW_SECONDS:.1f} Sekunden erforderlich."
        )

    embedded = embed_voice_windows(
        _iter_concatenated_segment_windows(audio, speech_segments)
    )
    if not embedded:
        raise ValueError("Keine qualitativ geeigneten Sprachfenster fuer ein Stimmembedding gefunden.")
    return _weighted_normalized_mean(
        (item.vector for item in embedded),
        (item.quality * item.clean_duration_seconds for item in embedded),
    )


__all__ = [
    "EmbeddedVoiceWindow",
    "REDIMNET_EMBEDDING_DIMENSION",
    "REDIMNET_MIN_WINDOW_SAMPLES",
    "REDIMNET_SAMPLE_RATE",
    "REDIMNET_WINDOW_SAMPLES",
    "VoiceWindow",
    "embed_voice_windows",
    "embedding_model_metadata",
    "generate_voice_vector",
    "iter_windows_from_audio",
    "load_vid_model",
    "windows_from_audio",
]
