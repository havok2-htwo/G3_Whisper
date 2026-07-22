"""ReDimNet2-B6 speaker embeddings used by every public voice-vector path."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import threading
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch

from .genesis_whisper_server_chunking import extract_speech_audio
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
REDIMNET_MIN_WINDOW_SECONDS = 1.0
REDIMNET_MIN_WINDOW_SAMPLES = int(REDIMNET_SAMPLE_RATE * REDIMNET_MIN_WINDOW_SECONDS)
REDIMNET_DEFAULT_BATCH_SIZE = 16


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
                warmup = torch.zeros(
                    (1, REDIMNET_WINDOW_SAMPLES),
                    dtype=torch.float32,
                    device=device,
                )
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    warmup_embedding = model(warmup)
                if warmup_embedding.shape[-1] != REDIMNET_EMBEDDING_DIMENSION:
                    raise RuntimeError(
                        "ReDimNet2 lieferte eine unerwartete Embedding-Dimension "
                        f"({warmup_embedding.shape[-1]})."
                    )

            _redimnet_components.update(
                {
                    "model": model,
                    "device": device,
                    "dtype": dtype,
                    "cache_root": str(cache_root),
                }
            )
            print("[INFO-VID] ReDimNet2 erfolgreich geladen und aufgewaermt.", file=sys.stderr)
            return True
        except Exception as exc:
            _redimnet_components.update({"model": None, "device": None, "dtype": None, "cache_root": None})
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
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 10 ** (-50.0 / 20.0):
        return None
    clipped_ratio = float(np.mean(np.abs(samples) >= 0.999))
    if clipped_ratio > 0.01:
        return None
    # Preserve low-volume but valid speech with a smaller influence.
    loudness_weight = min(1.0, max(0.4, rms / 0.08))
    return loudness_weight


def windows_from_audio(
    audio: np.ndarray,
    *,
    start_ms: int | None = None,
    stitched: bool = False,
    minimum_samples: int = REDIMNET_MIN_WINDOW_SAMPLES,
) -> list[VoiceWindow]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    windows: list[VoiceWindow] = []
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
        windows.append(
            VoiceWindow(
                audio=_fit_window(chunk),
                start_ms=chunk_start_ms,
                end_ms=chunk_end_ms,
                clean_duration_seconds=duration_seconds,
                quality=quality,
                stitched=stitched,
            )
        )
    return windows


def embed_voice_windows(
    windows: Sequence[VoiceWindow],
    *,
    batch_size: int = REDIMNET_DEFAULT_BATCH_SIZE,
) -> list[EmbeddedVoiceWindow]:
    if not windows:
        return []
    if not load_vid_model():
        raise RuntimeError("ReDimNet2-B6 LM konnte nicht geladen werden.")

    model = _redimnet_components["model"]
    device: torch.device = _redimnet_components["device"]
    dtype: torch.dtype = _redimnet_components["dtype"]
    output_vectors: list[np.ndarray] = []
    index = 0
    effective_batch_size = max(1, int(batch_size))

    with shared_gpu_lease():
        while index < len(windows):
            current_size = min(effective_batch_size, len(windows) - index)
            current = windows[index : index + current_size]
            batch_np = np.stack([_fit_window(window.audio) for window in current], axis=0)
            try:
                waveform = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    embeddings = model(waveform)
                embeddings_np = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
                del waveform, embeddings
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and current_size > 1:
                    effective_batch_size = max(1, current_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            for vector in embeddings_np:
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if len(vector) != REDIMNET_EMBEDDING_DIMENSION or not np.all(np.isfinite(vector)):
                    raise RuntimeError("ReDimNet2 lieferte einen ungueltigen Vektor.")
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    raise RuntimeError("ReDimNet2 lieferte einen Nullvektor.")
                output_vectors.append((vector / norm).astype(np.float32, copy=False))
            index += current_size

    return [
        EmbeddedVoiceWindow(
            vector=vector,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            clean_duration_seconds=window.clean_duration_seconds,
            quality=window.quality,
            stitched=window.stitched,
            source_spans=window.source_spans,
        )
        for window, vector in zip(windows, output_vectors)
    ]


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
    speech_only = extract_speech_audio(audio)
    if len(speech_only) < REDIMNET_MIN_WINDOW_SAMPLES:
        raise ValueError(
            "Zu wenig Sprache fuer ein Stimmembedding: mindestens 1,0 Sekunde erforderlich."
        )
    audio = speech_only

    windows = windows_from_audio(audio)
    embedded = embed_voice_windows(windows)
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
    "load_vid_model",
    "windows_from_audio",
]
