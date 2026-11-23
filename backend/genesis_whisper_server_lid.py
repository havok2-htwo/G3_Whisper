"""Spoken language identification (LID) for Cohere Transcribe with language "auto".

Cohere takes the language as its decoder prompt and cannot detect it, so "auto" used to mean German. A small
Whisper checkpoint answers the one question Cohere cannot: one encoder pass over a 30 s log-mel window, one
decoder step after <|startoftranscript|>, a softmax over only the 14 language tokens Cohere accepts. Nothing is
transcribed with it.

Measured on the owner's recordings (CPU, whisper-base): 100 % on 30 s windows, 95.5 % on 4 s crops, 83.6 % on
2 s crops; ECAPA (VoxLingua107) and AmberNet were clearly worse on German-accented English.

The model follows ReDimNet's pattern (genesis_whisper_server_vid): one module slot under a lock, lazy load,
the device of ``local_gpu_device``, FP16 on CUDA and FP32 on CPU, load and inference inside the cross-process
GPU lease, OOM back-off. Request handlers run it through the sliced GPU phase so a long file never holds the
GPU lock for all its windows at once. The download is a separate step (``prepare_lid_model``) on a worker
thread before any GPU lock is taken: a 1.6 GB first download must never stall the ASR queue, live embeddings
or another colocated GPU service, so the GPU phase itself only ever loads a complete local snapshot.

Detection is advisory: when the model cannot be loaded or a clip has too little speech, the request runs with
COHERE_FALLBACK_LANGUAGE exactly as before and reports why in its warnings. It never fails a request.
"""

from __future__ import annotations

import asyncio
import functools
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .genesis_whisper_server_globals import (
    COHERE_FALLBACK_LANGUAGE,
    COHERE_LANGUAGE_CODES,
    DEFAULT_LANGUAGE_DETECTION_MODEL,
    LANGUAGE_DETECTION_MODEL_OPTIONS,
    current_settings,
    resolve_local_model_cache_path,
    settings_lock,
    uses_cohere_backend,
)
from .genesis_whisper_server_gpu import run_sliced_gpu_phase, shared_gpu_lease
from .genesis_whisper_server_vid import resolve_local_device
from .genesis_whisper_server_wxc import (
    SUPERCHUNK_TARGET_S,
    build_language_superchunks,
    build_superchunks,
    padded_chunk_audio,
    region_pieces,
    superchunk_speech_regions,
)

SAMPLE_RATE = 16000
# Whisper's encoder always sees 30 s (shorter input is padded), so a window longer than that would be cut.
LID_WINDOW_SECONDS = 30.0
# Below this top probability a window, chunk or speaker does not decide on its own; it takes the file's
# duration-weighted majority instead. On 14 languages a wrong guess rarely reaches 0.5, a right one on clean
# speech is usually above 0.9.
LID_MIN_CONFIDENCE = 0.5
# Speech pooled per diarization speaker. One full Whisper window: more audio only costs time, 30 s was already
# 100 % right on the owner's files.
LID_MAX_POOL_SECONDS = 30.0
# Less speech than this is not detected at all (2 s crops were 84 % right, 1 s is mostly a guess).
LID_MIN_SPEECH_SECONDS = 1.0
# A whole-audio item longer than one window (31-60 s clips) is judged on at most this many windows.
LID_MAX_WINDOWS_PER_ITEM = 2
# Refinement below the prior (a diarization speaker's language, a long transcript's majority): a turn or speech
# piece of at least this length is detected on its own. Shorter ones keep the prior (1 s is mostly a guess,
# 2 s crops were 84 % right).
LID_TURN_MIN_SECONDS = 1.5
# A turn (or transcript piece) leaves its prior only with at least this top probability. Calibrated on
# whisper-base over 459 de/en clips of 2-8 s (owner crops, TTS, synthetic conversations; ground truth by hand):
#   threshold        0.60  0.70  0.80  0.85  0.90  0.95
#   wrong overrides  3.9%  2.8%  1.7%  1.3%  1.1%  0.2%   (confident but wrong, share of all clips)
#   caught switches  91%   89%   87%   86%   83%   75%    (confident and right)
# 0.85 keeps a wrong switch near 1 % (2 s: 3.2 %, 4-8 s: <=0.7 %) and still catches 86 % of real switches;
# beyond it every wrong switch avoided costs about ten caught ones. All 14 languages, 607 clips: 3.5 %
# wrong, 82 % caught (nl and zh TTS read as de with up to 0.99 carry most of the rest).
LID_TURN_OVERRIDE_CONFIDENCE = 0.85
# Long transcripts: a run of speech pieces shorter than LID_TURN_MIN_SECONDS is pooled into one detection
# window up to this much speech (8 s crops were about 95 % right), so choppy speech still counts.
LID_SHORT_POOL_SECONDS = 8.0
# Windows per encoder call. whisper-base at 8 x 30 s is about 0.2 GB of activations in FP16.
LID_BATCH_SIZE = 8
# Windows per GPU-lock hold in request handlers (two encoder batches), so live requests get in between.
LID_SLICE_WINDOWS = 16
# A failed load (missing download, no network) is not retried on every request.
LID_RETRY_AFTER_SECONDS = 300.0
# The files a detector needs. A snapshot without all of them (an interrupted first download) is completed by the
# next prepare step instead of being loaded offline and failing for good. All three Whisper options ship
# safetensors, so the PyTorch/Flax/TF duplicates are never fetched.
LID_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)
LID_DOWNLOAD_PATTERNS = ["*.json", "merges.txt", "model.safetensors"]

SOURCE_DETECTED = "detected"
SOURCE_MAJORITY = "majority"
SOURCE_FALLBACK = "fallback"


class LanguageDetectionUnavailable(RuntimeError):
    """The detection model could not be loaded; callers fall back to COHERE_FALLBACK_LANGUAGE."""


@dataclass(frozen=True)
class LidResult:
    """One detection: the most likely Cohere language, its probability and all 14 probabilities."""

    language: str
    probability: float
    probs: Dict[str, float]


_lid_lock = threading.Lock()
_lid_components: Dict[str, Any] = {
    "model": None,
    "feature_extractor": None,
    "model_id": None,
    "cache_root": None,
    "device": None,
    "dtype": None,
    "language_token_ids": None,
    "start_token_id": None,
    "batch_size": LID_BATCH_SIZE,
}
# The last failed load: key (model id, cache root), the error and when, so it is logged once and retried late.
_lid_failure: Dict[str, Any] = {"key": None, "error": None, "at": 0.0}
# Serializes downloads (worker threads, never the GPU thread) so parallel requests fetch a snapshot only once.
_lid_download_lock = threading.Lock()


def configured_lid_model() -> str:
    with settings_lock:
        configured = str(current_settings.get("language_detection_model") or "").strip()
    return configured if configured in LANGUAGE_DETECTION_MODEL_OPTIONS.values() else DEFAULT_LANGUAGE_DETECTION_MODEL


def _cache_root() -> str:
    """The ASR models' HF cache (local_model_cache_path); empty means the Hugging Face default cache."""

    with settings_lock:
        configured = str(current_settings.get("local_model_cache_path", "") or "").strip()
    return resolve_local_model_cache_path(configured)


def _lid_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def _release_cuda_pool() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _clear_components() -> None:
    _lid_components.update(
        {
            "model": None,
            "feature_extractor": None,
            "model_id": None,
            "cache_root": None,
            "device": None,
            "dtype": None,
            "language_token_ids": None,
            "start_token_id": None,
            "batch_size": LID_BATCH_SIZE,
        }
    )


def _snapshot_complete(path: Optional[str]) -> bool:
    return bool(path) and all(os.path.isfile(os.path.join(path, name)) for name in LID_REQUIRED_FILES)


def _download_snapshot(model_id: str, cache_root: str, local_only: bool) -> Optional[str]:
    """The snapshot folder of ``model_id`` in ``cache_root`` (HF default cache when empty); None offline-missing."""

    from huggingface_hub import snapshot_download

    from .genesis_whisper_server_local_asr_engine import _resolve_huggingface_token

    try:
        return snapshot_download(
            repo_id=model_id,
            cache_dir=cache_root or None,
            token=_resolve_huggingface_token(),
            allow_patterns=LID_DOWNLOAD_PATTERNS,
            local_files_only=local_only,
        )
    except Exception:
        if local_only:
            return None
        raise


def _local_snapshot(model_id: str, cache_root: str) -> Optional[str]:
    """A complete local snapshot, without network (HF links a file into the snapshot only once it is complete)."""

    path = _download_snapshot(model_id, cache_root, local_only=True)
    return path if _snapshot_complete(path) else None


def _record_failure(key: tuple, model_id: str, exc: BaseException) -> None:
    already_reported = _lid_failure["key"] == key
    _lid_failure.update({"key": key, "error": f"{type(exc).__name__}: {exc}", "at": time.monotonic()})
    if not already_reported:
        print(
            f"[WARNUNG-LID] Spracherkennung '{model_id}' nicht verfuegbar, Cohere nutzt "
            f"'{COHERE_FALLBACK_LANGUAGE}': {exc}",
            file=sys.stderr,
        )


def _retry_pending(key: tuple) -> bool:
    return _lid_failure["key"] == key and time.monotonic() - float(_lid_failure["at"]) < LID_RETRY_AFTER_SECONDS


def prepare_lid_model() -> bool:
    """Make sure the configured detector is complete on disk. Worker thread, never under a GPU lock.

    Resumes an interrupted download (only the missing files are fetched). False when it is unavailable; the
    failure is logged once and not retried for LID_RETRY_AFTER_SECONDS.
    """

    model_id = configured_lid_model()
    cache_root = _cache_root()
    if (
        _lid_components["model"] is not None
        and _lid_components["model_id"] == model_id
        and _lid_components["cache_root"] == cache_root
    ):
        return True
    key = (model_id, cache_root)
    with _lid_download_lock:
        if _local_snapshot(model_id, cache_root):
            return True
        if _retry_pending(key):
            return False
        started = time.monotonic()
        print(f"[INFO-LID] Lade Spracherkennung '{model_id}' herunter ...", file=sys.stderr)
        try:
            path = _download_snapshot(model_id, cache_root, local_only=False)
            if not _snapshot_complete(path):
                raise RuntimeError(f"Snapshot '{path}' lacks one of {', '.join(LID_REQUIRED_FILES)}.")
        except Exception as exc:
            _record_failure(key, model_id, exc)
            return False
        # A load that failed only for want of these files may run right away.
        _lid_failure.update({"key": None, "error": None, "at": 0.0})
        print(
            f"[INFO-LID] Spracherkennung '{model_id}' heruntergeladen "
            f"({round((time.monotonic() - started) * 1000)} ms).",
            file=sys.stderr,
        )
        return True


def _load_pretrained(model_id: str, cache_root: str):
    """(feature extractor, tokenizer, model) from the complete local snapshot only, never from the network.

    This runs inside the GPU lease; the download belongs to ``prepare_lid_model`` before it.
    """

    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    source = _local_snapshot(model_id, cache_root)
    if source is None:
        raise LanguageDetectionUnavailable(f"Language detection model '{model_id}' is not downloaded (yet).")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(source, local_files_only=True)
    tokenizer = WhisperTokenizer.from_pretrained(source, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(source, local_files_only=True)
    return feature_extractor, tokenizer, model


def _token_ids(tokenizer: Any) -> tuple[list[int], int]:
    unknown = getattr(tokenizer, "unk_token_id", None)
    language_ids = [tokenizer.convert_tokens_to_ids(f"<|{code}|>") for code in COHERE_LANGUAGE_CODES]
    start_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if any(token is None or token == unknown for token in [*language_ids, start_id]):
        raise RuntimeError("Tokenizer of the language detection model lacks Whisper's language tokens.")
    return [int(token) for token in language_ids], int(start_id)


def _move_model(target_device: torch.device) -> bool:
    """Move the loaded model to ``target_device`` (caller holds ``_lid_lock``, not the lease).

    Inside the lease like every other allocation, so another colocated GPU process at its VRAM peak is not hit by it. A failed
    move (OOM on the new card) would leave the weights split over two devices or dtypes, so the model is
    dropped instead and the next detection loads it cleanly. Returns whether the model is still loaded.
    """

    previous = _lid_components["device"]
    dtype = _lid_dtype(target_device)
    try:
        with shared_gpu_lease():
            _lid_components["model"].to(device=target_device, dtype=dtype)
    except Exception as exc:
        _clear_components()
        _release_cuda_pool()
        print(f"[WARNUNG-LID] Spracherkennung nicht nach '{target_device}' verschiebbar, entladen: {exc}", file=sys.stderr)
        return False
    _lid_components.update({"device": target_device, "dtype": dtype, "batch_size": LID_BATCH_SIZE})
    _release_cuda_pool()
    print(f"[INFO-LID] Spracherkennung von '{previous}' nach '{target_device}' verschoben.", file=sys.stderr)
    return True


def _ensure_loaded() -> bool:
    """Load or follow the configured model and device. Caller holds ``_lid_lock``."""

    model_id = configured_lid_model()
    cache_root = _cache_root()
    if _lid_components["model"] is not None:
        if _lid_components["model_id"] == model_id and _lid_components["cache_root"] == cache_root:
            target_device = resolve_local_device()
            if str(_lid_components["device"]) == str(target_device) or _move_model(target_device):
                return True
        else:
            _clear_components()
            _release_cuda_pool()

    key = (model_id, cache_root)
    if _retry_pending(key):
        return False

    device = resolve_local_device()
    dtype = _lid_dtype(device)
    started = time.monotonic()
    try:
        with shared_gpu_lease():
            feature_extractor, tokenizer, model = _load_pretrained(model_id, cache_root)
            language_ids, start_id = _token_ids(tokenizer)
            model.eval()
            model.to(device=device, dtype=dtype)
    except Exception as exc:
        _record_failure(key, model_id, exc)
        return False

    _lid_components.update(
        {
            "model": model,
            "feature_extractor": feature_extractor,
            "model_id": model_id,
            "cache_root": cache_root,
            "device": device,
            "dtype": dtype,
            "language_token_ids": language_ids,
            "start_token_id": start_id,
            "batch_size": LID_BATCH_SIZE,
        }
    )
    _lid_failure.update({"key": None, "error": None, "at": 0.0})
    print(
        f"[INFO-LID] Spracherkennung '{model_id}' auf '{device}' ({dtype}) geladen "
        f"({round((time.monotonic() - started) * 1000)} ms).",
        file=sys.stderr,
    )
    return True


def load_lid_model() -> bool:
    """Load the configured detection model once (GPU phase thread). False when it is unavailable.

    Loads only a complete local snapshot; ``prepare_lid_model`` (off the GPU thread) downloads it first.
    """

    with _lid_lock:
        return _ensure_loaded()


def lid_status() -> Dict[str, Any]:
    """Configured model, whether it is resident and the last load error for exactly that model (or None)."""

    model_id = configured_lid_model()
    key = (model_id, _cache_root())
    # No _lid_lock: this runs on the event loop (GET /v2/capabilities) and must not wait for a detection
    # that holds the lock on the GPU thread. Single dict reads are atomic; a stale answer is harmless.
    loaded = _lid_components["model"] is not None and _lid_components["model_id"] == model_id
    error = _lid_failure["error"] if _lid_failure["key"] == key else None
    return {"model": model_id, "loaded": loaded, "error": error}


def apply_lid_device_setting() -> bool:
    """Admin hook: follow a saved ``local_gpu_device`` / ``language_detection_model`` right away (GPU thread).

    A changed device moves the weights; a changed model, detection switched off or an ASR model other than Cohere
    unloads the old one so its VRAM is free (the next detection loads what is configured then). A not yet loaded model needs nothing.
    Returns True when something changed. Never touches the ASR model.
    """

    with settings_lock:
        enabled = current_settings.get("cohere_language_detection", True) is True
        # Only Cohere uses the detector; after a switch to Whisper or Parakeet its VRAM goes back to the card.
        enabled = enabled and uses_cohere_backend(str(current_settings.get("local_model") or ""))
    with _lid_lock:
        # A new choice gets a fresh load attempt instead of waiting out the previous model's retry pause.
        _lid_failure.update({"key": None, "error": None, "at": 0.0})
        if _lid_components["model"] is None:
            return False
        if not enabled or _lid_components["model_id"] != configured_lid_model():
            _clear_components()
            _release_cuda_pool()
            print("[INFO-LID] Spracherkennung entladen (Modell gewechselt oder abgeschaltet).", file=sys.stderr)
            return True
        target_device = resolve_local_device()
        if str(_lid_components["device"]) == str(target_device):
            return False
        _move_model(target_device)
        return True


def _logits_to_result(logits: np.ndarray) -> LidResult:
    """Softmax over the 14 Cohere language logits (in COHERE_LANGUAGE_CODES order)."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values = values - np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    best = int(np.argmax(probabilities))
    return LidResult(
        language=COHERE_LANGUAGE_CODES[best],
        probability=float(probabilities[best]),
        probs={code: float(value) for code, value in zip(COHERE_LANGUAGE_CODES, probabilities)},
    )


def _extract_features(feature_extractor: Any, windows: Sequence[np.ndarray]) -> torch.Tensor:
    return feature_extractor(
        [np.asarray(window, dtype=np.float32).reshape(-1) for window in windows],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features


@dataclass(frozen=True)
class LidFeatures:
    """Log-mel features of some windows, computed off the GPU lock by the extractor that made them."""

    extractor: Any
    values: torch.Tensor


def lid_features(windows: Sequence[np.ndarray]) -> Optional[LidFeatures]:
    """The windows' log-mel features on a worker thread, or None before the detector was ever loaded.

    The feature extraction (about 5 ms of CPU per 30 s window, more than the encoder itself) then no longer
    runs inside the GPU lock. Read without ``_lid_lock`` (a single atomic dict read); detect_languages uses
    the features only when the same extractor is still loaded, else it computes them itself as before.
    """

    extractor = _lid_components["feature_extractor"]
    if extractor is None or not windows:
        return None
    try:
        return LidFeatures(extractor=extractor, values=_extract_features(extractor, windows))
    except Exception:
        # The GPU phase extracts them itself and reports a real failure there.
        return None


def _run_batch(windows: Sequence[np.ndarray], precomputed: Optional[torch.Tensor] = None) -> np.ndarray:
    """Language logits [len(windows), 14] for one encoder batch. Caller holds ``_lid_lock`` and the lease."""

    model = _lid_components["model"]
    device: torch.device = _lid_components["device"]
    dtype: torch.dtype = _lid_components["dtype"]
    features = precomputed if precomputed is not None else _extract_features(_lid_components["feature_extractor"], windows)
    input_features = None
    decoder_input_ids = None
    logits = None
    try:
        input_features = features.to(device=device, dtype=dtype)
        decoder_input_ids = torch.full(
            (len(windows), 1),
            int(_lid_components["start_token_id"]),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            logits = model(input_features=input_features, decoder_input_ids=decoder_input_ids).logits
            selected = logits[:, -1, _lid_components["language_token_ids"]]
        return selected.detach().to(device="cpu", dtype=torch.float32).numpy()
    finally:
        del input_features, decoder_input_ids, logits


def detect_languages(windows: Sequence[np.ndarray], features: Optional[LidFeatures] = None) -> List[LidResult]:
    """One LidResult per 16 kHz mono window (at most 30 s each are seen), batched into few encoder calls.

    Runs on the GPU phase thread. ``features`` (lid_features of exactly these windows) are used when their
    extractor is still the loaded one. Raises LanguageDetectionUnavailable when the model cannot be loaded.
    """

    items = list(windows)
    if not items:
        return []
    with _lid_lock:
        if not _ensure_loaded():
            raise LanguageDetectionUnavailable(
                f"Language detection model '{configured_lid_model()}' is unavailable: {_lid_failure['error']}"
            )
        device: torch.device = _lid_components["device"]
        precomputed = (
            features.values
            if features is not None
            and features.extractor is _lid_components["feature_extractor"]
            and int(features.values.shape[0]) == len(items)
            else None
        )
        results: List[LidResult] = []
        with shared_gpu_lease():
            position = 0
            while position < len(items):
                batch_size = max(1, int(_lid_components["batch_size"]))
                current = items[position : position + batch_size]
                try:
                    logits = _run_batch(
                        current,
                        precomputed[position : position + len(current)] if precomputed is not None else None,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch_size > 1:
                        # Halve and retry; the smaller size sticks until the model is moved or reloaded.
                        _lid_components["batch_size"] = batch_size // 2
                        if device.type == "cuda" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                results.extend(_logits_to_result(row) for row in logits)
                position += len(current)
    return results


# ---------------------------------------------------------------------------
# Pooling and smoothing (pure, CPU)
# ---------------------------------------------------------------------------

def windows_for_item(audio: np.ndarray) -> List[np.ndarray]:
    """The LID windows of one ASR item: the item itself up to 30 s, else equal parts of at most 30 s.

    A 31-60 s whole-audio clip becomes two windows. Longer items exist only when Silero found no speech in a
    long file (it is then sent whole); its first LID_MAX_WINDOWS_PER_ITEM windows are enough to judge it.
    """

    return split_windows(audio)[:LID_MAX_WINDOWS_PER_ITEM]


def split_windows(audio: np.ndarray) -> List[np.ndarray]:
    """The audio itself up to 30 s, else all its equal parts of at most 30 s (a long turn, a long piece)."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    window_samples = int(LID_WINDOW_SECONDS * SAMPLE_RATE)
    if len(samples) <= window_samples:
        return [samples]
    count = math.ceil(len(samples) / window_samples)
    size = math.ceil(len(samples) / count)
    return [samples[offset : offset + size] for offset in range(0, len(samples), size)]


def pool_results(results: Sequence[Optional[LidResult]], weights: Sequence[float]) -> Optional[LidResult]:
    """One result from several windows: log-probabilities summed with each window's speech seconds as weight.

    The sum is divided by the total weight before renormalising, so the pooled probability stays a
    probability of one window's kind instead of saturating at 1.0 with many windows (same argmax).
    """

    total_weight = 0.0
    accumulated = np.zeros(len(COHERE_LANGUAGE_CODES), dtype=np.float64)
    for result, weight in zip(results, weights):
        if result is None or weight <= 0.0:
            continue
        probabilities = np.array([result.probs.get(code, 0.0) for code in COHERE_LANGUAGE_CODES], dtype=np.float64)
        accumulated += float(weight) * np.log(np.clip(probabilities, 1e-12, 1.0))
        total_weight += float(weight)
    if total_weight <= 0.0:
        return None
    return _logits_to_result(accumulated / total_weight)


def _weighted_majority(languages_and_seconds: Sequence[tuple[str, float]]) -> Optional[str]:
    totals: Counter[str] = Counter()
    for language, seconds in languages_and_seconds:
        totals[language] += max(0.0, float(seconds))
    if not totals:
        return None
    # Ties go to the language listed first among the Cohere codes, so the result is deterministic.
    return max(totals, key=lambda code: (totals[code], -COHERE_LANGUAGE_CODES.index(code)))


def _is_confident(result: Optional[LidResult], seconds: float) -> bool:
    return result is not None and seconds >= LID_MIN_SPEECH_SECONDS and result.probability >= LID_MIN_CONFIDENCE


def smooth_chunk_languages(
    results: Sequence[Optional[LidResult]],
    seconds: Sequence[float],
    fallback: str = COHERE_FALLBACK_LANGUAGE,
) -> List[str]:
    """Per-chunk languages for a transcript: confident chunks keep theirs, the rest take the file majority.

    The majority is weighted by the confident chunks' seconds. With no confident chunk the majority of every
    chunk that has a result decides; with no result at all (too little speech, detection unavailable) every
    chunk gets ``fallback``.
    """

    confident = [
        (result.language, chunk_seconds)
        for result, chunk_seconds in zip(results, seconds)
        if _is_confident(result, chunk_seconds)
    ]
    majority = _weighted_majority(confident) or _weighted_majority(
        [
            (result.language, chunk_seconds)
            for result, chunk_seconds in zip(results, seconds)
            if result is not None and chunk_seconds >= LID_MIN_SPEECH_SECONDS
        ]
    )
    if majority is None:
        return [fallback for _ in results]
    return [
        result.language if _is_confident(result, chunk_seconds) else majority
        for result, chunk_seconds in zip(results, seconds)
    ]


def resolve_speaker_languages(
    results: Mapping[str, Optional[LidResult]],
    seconds: Mapping[str, float],
    fallback: str = COHERE_FALLBACK_LANGUAGE,
) -> Dict[str, tuple[str, str]]:
    """(language, source) per diarization speaker; one speaker keeps one language for the whole file.

    A speaker with enough speech and a confident result is "detected". Everyone else takes the majority of
    the detected speakers, weighted by their speech seconds ("majority"), or ``fallback`` when nobody was
    detected ("fallback").
    """

    detected = {
        speaker: result.language
        for speaker, result in results.items()
        if _is_confident(result, float(seconds.get(speaker, 0.0)))
    }
    majority = _weighted_majority([(language, float(seconds.get(speaker, 0.0))) for speaker, language in detected.items()])
    resolved: Dict[str, tuple[str, str]] = {}
    for speaker in results:
        if speaker in detected:
            resolved[speaker] = (detected[speaker], SOURCE_DETECTED)
        elif majority is not None:
            resolved[speaker] = (majority, SOURCE_MAJORITY)
        else:
            resolved[speaker] = (fallback, SOURCE_FALLBACK)
    return resolved


def turn_language(result: Optional[LidResult], seconds: float, prior: str) -> tuple[str, bool]:
    """(language, overridden) of one diarization turn whose speaker was decided ``prior``.

    The speaker's language stays the prior: only a turn of at least LID_TURN_MIN_SECONDS whose own detection
    reaches LID_TURN_OVERRIDE_CONFIDENCE for another language takes that language (a German interjection in a
    diarization cluster labelled with an English speaker, a speaker who switches for a sentence).
    """

    if (
        result is not None
        and seconds >= LID_TURN_MIN_SECONDS
        and result.probability >= LID_TURN_OVERRIDE_CONFIDENCE
        and result.language != prior
    ):
        return result.language, True
    return prior, False


def smooth_piece_languages(
    results: Sequence[Optional[LidResult]],
    spans: Sequence[tuple[float, float]],
    fallback: str = COHERE_FALLBACK_LANGUAGE,
    seconds: Optional[Sequence[float]] = None,
) -> List[str]:
    """Per-piece languages for a long transcript's speech pieces or detection units (time order).

    The file majority (confident pieces, weighted by seconds) is the prior. An anchor is a piece of at least
    LID_TURN_MIN_SECONDS with a confident result (LID_MIN_CONFIDENCE) for the majority, or with
    LID_TURN_OVERRIDE_CONFIDENCE for another language, the same bar a diarization turn needs to leave its
    speaker's language. Every other piece (short, weak, or a doubtful deviation) takes the language of the
    nearest anchor, the earlier one on a tie; without any anchor everything is the majority, without any
    result at all ``fallback``. ``seconds`` is each one's speech (a pooled unit spans pauses); without it the
    span lengths.
    """

    if seconds is None:
        seconds = [max(0.0, float(end) - float(start)) for start, end in spans]
    usable = [
        result if result is not None and piece_seconds >= LID_TURN_MIN_SECONDS else None
        for result, piece_seconds in zip(results, seconds)
    ]
    confident = [
        (result.language, piece_seconds)
        for result, piece_seconds in zip(usable, seconds)
        if result is not None and result.probability >= LID_MIN_CONFIDENCE
    ]
    majority = _weighted_majority(confident) or _weighted_majority(
        [(result.language, piece_seconds) for result, piece_seconds in zip(usable, seconds) if result is not None]
    )
    if majority is None:
        return [fallback for _ in spans]
    anchors: List[Optional[str]] = [
        result.language
        if result is not None
        and result.probability >= LID_MIN_CONFIDENCE
        and (result.language == majority or result.probability >= LID_TURN_OVERRIDE_CONFIDENCE)
        else None
        for result in usable
    ]
    if not any(anchors):
        return [majority for _ in spans]
    # The nearest anchor is the previous or the next one (pieces are in time order): two linear scans.
    previous: List[Optional[int]] = []
    last: Optional[int] = None
    for index, anchor in enumerate(anchors):
        if anchor is not None:
            last = index
        previous.append(last)
    following: List[Optional[int]] = [None] * len(anchors)
    upcoming: Optional[int] = None
    for index in range(len(anchors) - 1, -1, -1):
        if anchors[index] is not None:
            upcoming = index
        following[index] = upcoming
    languages: List[str] = []
    for index, anchor in enumerate(anchors):
        if anchor is not None:
            languages.append(anchor)
            continue
        before, after = previous[index], following[index]
        gap_before = float(spans[index][0]) - float(spans[before][1]) if before is not None else math.inf
        gap_after = float(spans[after][0]) - float(spans[index][1]) if after is not None else math.inf
        languages.append(anchors[before] if gap_before <= gap_after else anchors[after])
    return languages


def dominant_language(languages_and_seconds: Sequence[tuple[str, float]], fallback: str) -> str:
    return _weighted_majority(languages_and_seconds) or fallback


# ---------------------------------------------------------------------------
# Request orchestration (async, used by /transcribe/ and /v2/audio/process)
# ---------------------------------------------------------------------------

@dataclass
class LanguageDecision:
    """What a request ran with: one language per ASR item, the dominant one and how it was decided."""

    languages: List[str]
    language: str
    source: str
    model: Optional[str] = None
    seconds: Dict[str, float] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def detection_ran(self) -> bool:
        return self.source in (SOURCE_DETECTED, SOURCE_FALLBACK) and self.model is not None

    def detection_block(self) -> Optional[Dict[str, Any]]:
        """models.asr.language_detection, only when detection ran for this request."""

        if not self.detection_ran:
            return None
        return {
            "model": self.model,
            "source": self.source,
            "languages": {code: round(value, 1) for code, value in sorted(self.seconds.items())},
        }


def unavailable_warning(message: str) -> Dict[str, Any]:
    return {
        "code": "LANGUAGE_DETECTION_UNAVAILABLE",
        "message": f"{message[:300]} Transcribed as '{COHERE_FALLBACK_LANGUAGE}'.",
    }


async def detect_windows(gpu_lock: asyncio.Lock, windows: Sequence[np.ndarray]) -> List[LidResult]:
    """detect_languages over many windows, one GPU-lock hold per LID_SLICE_WINDOWS windows.

    The download (first use, a new model choice, a resumed interrupted fetch) runs before, on a worker thread
    and outside every GPU lock. Raises LanguageDetectionUnavailable when the model is not available.
    """

    if not await asyncio.to_thread(prepare_lid_model):
        raise LanguageDetectionUnavailable(
            f"Language detection model '{configured_lid_model()}' is unavailable: {_lid_failure['error']}"
        )
    items = list(windows)
    results: List[LidResult] = []
    for start in range(0, len(items), LID_SLICE_WINDOWS):
        part = items[start : start + LID_SLICE_WINDOWS]
        # The log-mel features of this slice are computed on a worker thread, outside the GPU lock; only the
        # encoder step runs under it. Before the first load there is no extractor yet and the GPU phase makes
        # them itself.
        features = await asyncio.to_thread(lid_features, part)
        function = detect_languages if features is None else functools.partial(_detect_with_features, features=features)
        parts = await run_sliced_gpu_phase(gpu_lock, part, function, LID_SLICE_WINDOWS)
        results.extend(result for chunk in parts for result in chunk)
    return results


def _detect_with_features(windows: Sequence[np.ndarray], features: LidFeatures) -> List[LidResult]:
    return detect_languages(windows, features=features)


async def decide_item_languages(gpu_lock: asyncio.Lock, items: Sequence[np.ndarray]) -> LanguageDecision:
    """Detect the language of every ASR item (whole clip or superchunk) and smooth them over the file.

    Never raises: any detection failure becomes COHERE_FALLBACK_LANGUAGE plus a warning.
    """

    started = time.monotonic()
    model_id = configured_lid_model()
    item_seconds = [len(np.asarray(item).reshape(-1)) / SAMPLE_RATE for item in items]
    windows: List[np.ndarray] = []
    owners: List[int] = []
    for index, item in enumerate(items):
        if item_seconds[index] < LID_MIN_SPEECH_SECONDS:
            continue
        for window in windows_for_item(item):
            windows.append(window)
            owners.append(index)

    warnings: List[Dict[str, Any]] = []
    item_results: List[Optional[LidResult]] = [None] * len(items)
    if windows:
        try:
            window_results = await detect_windows(gpu_lock, windows)
            for index in range(len(items)):
                mine = [(result, len(window) / SAMPLE_RATE) for result, window, owner in zip(window_results, windows, owners) if owner == index]
                if mine:
                    item_results[index] = pool_results([result for result, _ in mine], [weight for _, weight in mine])
        except Exception as exc:
            print(f"[WARNUNG-LID] Spracherkennung fehlgeschlagen, nutze '{COHERE_FALLBACK_LANGUAGE}': {exc}", file=sys.stderr)
            warnings.append(unavailable_warning(str(exc)))
            item_results = [None] * len(items)

    languages = smooth_chunk_languages(item_results, item_seconds)
    detected = any(result is not None for result in item_results)
    seconds: Dict[str, float] = {}
    for language, value in zip(languages, item_seconds):
        seconds[language] = seconds.get(language, 0.0) + value
    decision = LanguageDecision(
        languages=languages,
        language=dominant_language(list(zip(languages, item_seconds)), COHERE_FALLBACK_LANGUAGE),
        source=SOURCE_DETECTED if detected else SOURCE_FALLBACK,
        model=model_id,
        seconds=seconds,
        warnings=warnings,
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    log_decision(
        decision,
        [
            f"{result.language}:{result.probability:.2f}" if result is not None else "-"
            for result in item_results
        ],
    )
    return decision


async def decide_transcript_items(
    gpu_lock: asyncio.Lock,
    audio: np.ndarray,
) -> tuple[List[np.ndarray], LanguageDecision]:
    """ASR items and their languages for a transcript with Cohere and an effective "auto". Never raises.

    Audio up to WHOLE_AUDIO_MAX_S (or without any Silero speech) stays one item with one decision, exactly as
    decide_item_languages. A longer file is judged per speech piece instead of per superchunk: every Silero
    region of split_audio_into_superchunks is one piece (longer ones in parts of at most 30 s, cut at the
    quietest moment near the even split), every piece of at least LID_TURN_MIN_SECONDS is one detection unit
    and runs of shorter pieces are pooled into units (piece_detection_units), all units go through one batched
    detection pass, smooth_piece_languages labels them, and the same builder as WXC diarization cuts
    superchunks that never mix two languages. One language gives exactly the superchunks of
    split_audio_into_superchunks. The CPU work around the detection runs on worker threads.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    regions = await asyncio.to_thread(superchunk_speech_regions, samples)
    if regions is None:
        items = [samples]
        return items, await decide_item_languages(gpu_lock, items)

    started = time.monotonic()
    model_id = configured_lid_model()
    pieces, units, windows, detectable = await asyncio.to_thread(_transcript_detection_plan, samples, regions)
    warnings: List[Dict[str, Any]] = []
    results: List[Optional[LidResult]] = [None] * len(units)
    if detectable:
        try:
            for index, result in zip(detectable, await detect_windows(gpu_lock, windows)):
                results[index] = result
        except Exception as exc:
            print(f"[WARNUNG-LID] Spracherkennung fehlgeschlagen, nutze '{COHERE_FALLBACK_LANGUAGE}': {exc}", file=sys.stderr)
            warnings.append(unavailable_warning(str(exc)))
            results = [None] * len(units)
    del windows

    items, chunks, languages = await asyncio.to_thread(_transcript_chunks, samples, regions, pieces, units, results)
    seconds: Dict[str, float] = {}
    for (start, end), language in zip(chunks, languages):
        seconds[language] = seconds.get(language, 0.0) + (end - start)
    decision = LanguageDecision(
        languages=languages,
        language=dominant_language(list(seconds.items()), COHERE_FALLBACK_LANGUAGE),
        source=SOURCE_DETECTED if any(result is not None for result in results) else SOURCE_FALLBACK,
        model=model_id,
        seconds=seconds,
        warnings=warnings,
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    log_decision(
        decision,
        [f"{result.language}:{result.probability:.2f}" if result is not None else "-" for result in results],
    )
    return items, decision


def piece_detection_units(pieces: Sequence[tuple[float, float]]) -> List[List[int]]:
    """Detection units over a long transcript's speech pieces (time order): lists of piece indices.

    A piece of at least LID_TURN_MIN_SECONDS is a unit of its own. A run of shorter pieces (choppy speech,
    dictated numbers, quick back-and-forth) is pooled, so its speech still counts: a unit grows until it holds
    LID_SHORT_POOL_SECONDS of speech or would span more than SUPERCHUNK_TARGET_S (one old superchunk, which
    was one detection window before). A pooled unit still under LID_TURN_MIN_SECONDS is not detected.
    """

    units: List[List[int]] = []
    pooling = False
    for index, (start, end) in enumerate(pieces):
        if end - start >= LID_TURN_MIN_SECONDS:
            units.append([index])
            pooling = False
            continue
        if pooling:
            unit = units[-1]
            speech = sum(pieces[k][1] - pieces[k][0] for k in unit)
            if speech < LID_SHORT_POOL_SECONDS and end - pieces[unit[0]][0] <= SUPERCHUNK_TARGET_S:
                unit.append(index)
                continue
        units.append([index])
        pooling = True
    return units


def _transcript_detection_plan(
    samples: np.ndarray,
    regions: Sequence[tuple[float, float]],
) -> tuple[List[tuple[float, float]], List[List[int]], List[np.ndarray], List[int]]:
    """(pieces, units, one window per detectable unit, the detectable unit indices). Worker thread.

    A single piece's window is a view into ``samples``; only pooled short pieces (at most
    LID_SHORT_POOL_SECONDS together) are copied.
    """

    pieces = region_pieces(regions, LID_WINDOW_SECONDS, audio=samples)
    units = piece_detection_units(pieces)
    windows: List[np.ndarray] = []
    detectable: List[int] = []
    for unit_index, unit in enumerate(units):
        if sum(pieces[k][1] - pieces[k][0] for k in unit) < LID_TURN_MIN_SECONDS:
            continue
        parts = [samples[max(0, round(pieces[k][0] * SAMPLE_RATE)) : round(pieces[k][1] * SAMPLE_RATE)] for k in unit]
        windows.append(parts[0] if len(parts) == 1 else np.concatenate(parts))
        detectable.append(unit_index)
    return pieces, units, windows, detectable


def _transcript_chunks(
    samples: np.ndarray,
    regions: Sequence[tuple[float, float]],
    pieces: Sequence[tuple[float, float]],
    units: Sequence[Sequence[int]],
    results: Sequence[Optional[LidResult]],
) -> tuple[List[np.ndarray], List[tuple[float, float]], List[str]]:
    """(ASR items, chunks, chunk languages) from the units' results. Worker thread."""

    unit_spans = [(pieces[unit[0]][0], pieces[unit[-1]][1]) for unit in units]
    unit_seconds = [sum(pieces[k][1] - pieces[k][0] for k in unit) for unit in units]
    unit_languages = smooth_piece_languages(results, unit_spans, seconds=unit_seconds)
    piece_languages: List[str] = [""] * len(pieces)
    for unit, language in zip(units, unit_languages):
        for k in unit:
            piece_languages[k] = language
    if len(set(piece_languages)) > 1:
        chunks, languages = build_language_superchunks(
            regions, [(start, end, language) for (start, end), language in zip(pieces, piece_languages)]
        )
        # A language change inside one long region leaves two touching chunks; padding both by CHUNK_PAD_S
        # would decode the speech at the seam twice (transcript mode has no word-level dedupe).
        items = padded_chunk_audio(samples, chunks, split_seams=True)
    else:
        chunks = build_superchunks(regions)
        languages = [piece_languages[0]] * len(chunks)
        items = padded_chunk_audio(samples, chunks)
    return items, chunks, languages


def log_decision(decision: LanguageDecision, confidences: Sequence[str]) -> None:
    shown = list(confidences[:12]) + (["..."] if len(confidences) > 12 else [])
    print(
        f"[INFO-LID] {decision.model}: {decision.source} -> {decision.language} "
        f"(Sekunden {', '.join(f'{code}={value:.1f}' for code, value in sorted(decision.seconds.items()))}; "
        f"Konfidenz {' '.join(shown)}; {decision.duration_ms} ms)",
        file=sys.stderr,
    )


def with_language(processing_key: tuple, language: str) -> tuple:
    return (processing_key[0], processing_key[1], processing_key[2], language, processing_key[4])


async def submit_by_language(
    batch_manager: Any,
    segments: Sequence[np.ndarray],
    processing_key: tuple,
    languages: Optional[Sequence[str]] = None,
    request_id: Optional[str] = None,
) -> List[Any]:
    """submit_job with one job per language, results back in segment order.

    A job carries one processing key and the scheduler batches only equal keys, so every language becomes its
    own job and its own batches. The jobs run concurrently. Without ``languages``, or with a single one, this is
    exactly the one submit_job call of before (the very same segment list). ``request_id`` names the job (the
    first one when there are several); without it every job gets a fresh token.
    """

    def token(index: int = 0) -> str:
        return request_id if request_id and index == 0 else uuid.uuid4().hex

    if not languages:
        return await batch_manager.submit_job(segments, token(), processing_key)
    distinct = list(dict.fromkeys(languages))
    if len(distinct) == 1:
        return await batch_manager.submit_job(segments, token(), with_language(processing_key, distinct[0]))

    groups = {language: [index for index, value in enumerate(languages) if value == language] for language in distinct}
    tasks = [
        asyncio.ensure_future(
            batch_manager.submit_job(
                [segments[index] for index in indices],
                token(position),
                with_language(processing_key, language),
            )
        )
        for position, (language, indices) in enumerate(groups.items())
    ]
    try:
        group_results = await asyncio.gather(*tasks)
    except BaseException:
        # One failed language must not leave the other jobs running for nobody.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    ordered: List[Any] = [None] * len(segments)
    for indices, results in zip(groups.values(), group_results):
        for index, result in zip(indices, results):
            ordered[index] = result
    return ordered


__all__ = [
    "LID_MAX_POOL_SECONDS",
    "LID_MIN_CONFIDENCE",
    "LID_MIN_SPEECH_SECONDS",
    "LID_SHORT_POOL_SECONDS",
    "LID_TURN_MIN_SECONDS",
    "LID_TURN_OVERRIDE_CONFIDENCE",
    "LanguageDecision",
    "LanguageDetectionUnavailable",
    "LidResult",
    "apply_lid_device_setting",
    "configured_lid_model",
    "decide_item_languages",
    "decide_transcript_items",
    "detect_languages",
    "detect_windows",
    "dominant_language",
    "lid_features",
    "lid_status",
    "load_lid_model",
    "piece_detection_units",
    "pool_results",
    "prepare_lid_model",
    "resolve_speaker_languages",
    "smooth_chunk_languages",
    "smooth_piece_languages",
    "split_windows",
    "submit_by_language",
    "turn_language",
    "windows_for_item",
    "with_language",
]
