"""WXC processing path for v2 diarization: transcribe-first with word alignment.

Measured motivation (see .tmp/optimize benchmarks): DIA's exclusive partition cuts
audio into sub-word slivers that Cohere fills with subtitle hallucinations, while
~25s speech chunks transcribe clean (0 hallucination markers).  This module keeps
ASR on large Silero-derived chunks and re-attaches speakers afterwards:

  Silero VAD (batched streams) -> speech regions -> ~25s superchunks (+-250ms pad)
  -> [caller runs ASR per chunk] -> MMS_FA word timestamps -> seam dedup by word
  midpoint ownership -> word->speaker against the PR1-gated DIA skeleton ->
  sentence regroup at ASR punctuation -> iterative ReDimNet 192D verification
  (purified per-speaker cores, conservative reassignment, flags for the rest).

Heavy steps are synchronous on purpose; the v2 layer schedules them on the GPU
phase worker.  ASR itself stays in the existing batch queue.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .genesis_whisper_server_turn_gate import clean_segment_text
from .genesis_whisper_server_vid import VoiceWindow, embed_voice_windows

SAMPLE_RATE = 16000

# --- Silero segmentation (values validated on real meeting audio) ---
SILERO_WINDOW = 512                # 32ms native v5 frame
SILERO_STREAM_FRAMES = 1024        # ~32.8s per parallel stream
SILERO_THRESHOLD = 0.5
SILERO_DENSITY_WIN = 5
SILERO_MIN_DENSITY = 0.5
SILERO_MIN_REGION_MS = 200
SILERO_BRIDGE_MS = 300

SUPERCHUNK_TARGET_S = 25.0
CHUNK_PAD_S = 0.25                 # never clip a word at chunk edges

# --- sentence-level 192D verification (validated: words <0.6s are noise,
# sentences >=2s reach ~0.82 median cosine to their own speaker core) ---
VERIFY_MIN_SENT_S = 1.5
VERIFY_CORE_KEEP = 0.6
VERIFY_MARGIN = 0.15
VERIFY_MAX_ROUNDS = 3
VERIFY_TRIM_S = 0.25
VERIFY_TRIM_MIN_S = 1.2
SENTENCE_GAP_S = 1.5

_silero_lock = threading.Lock()
_silero_model: Any | None = None
_mms_lock = threading.Lock()
_mms_components: dict[str, Any] = {"model": None, "tokenizer": None, "aligner": None, "device": None}


def _default_user_hub() -> Path:
    return Path.home() / ".cache" / "torch" / "hub"


def _load_silero_model():
    global _silero_model
    with _silero_lock:
        if _silero_model is not None:
            return _silero_model
        candidates = [
            _default_user_hub() / "snakers4_silero-vad_master",
            Path(torch.hub.get_dir()) / "snakers4_silero-vad_master",
        ]
        local = next((c for c in candidates if c.is_dir()), None)
        if local is not None:
            model, _ = torch.hub.load(str(local), "silero_vad", source="local", trust_repo=True)
        else:
            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        model.eval()
        _silero_model = model
        print("[INFO-WXC] Silero VAD geladen.", file=sys.stderr)
        return model


def silero_frame_probs(audio: np.ndarray) -> np.ndarray:
    """Per-32ms speech probabilities via parallel Silero streams.

    Sequential per-frame calls are ~2000 python/model calls per audio minute; a
    two-hour file would take tens of minutes.  Splitting the file into ~33s
    streams and stepping them as one batch keeps Silero's recurrent state per
    stream row and reduces the loop to SILERO_STREAM_FRAMES steps total.
    """

    model = _load_silero_model()
    samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    n_frames = len(samples) // SILERO_WINDOW
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)

    n_streams = max(1, (n_frames + SILERO_STREAM_FRAMES - 1) // SILERO_STREAM_FRAMES)
    padded_frames = n_streams * SILERO_STREAM_FRAMES
    padded = np.zeros(padded_frames * SILERO_WINDOW, dtype=np.float32)
    padded[: n_frames * SILERO_WINDOW] = samples[: n_frames * SILERO_WINDOW]
    streams = torch.from_numpy(padded).reshape(n_streams, SILERO_STREAM_FRAMES * SILERO_WINDOW)

    model.reset_states()
    probs = np.empty((n_streams, SILERO_STREAM_FRAMES), dtype=np.float32)
    with torch.no_grad():
        for frame_index in range(SILERO_STREAM_FRAMES):
            chunk = streams[:, frame_index * SILERO_WINDOW : (frame_index + 1) * SILERO_WINDOW]
            out = model(chunk, SAMPLE_RATE)
            probs[:, frame_index] = out.reshape(-1).cpu().numpy()
    model.reset_states()
    return probs.reshape(-1)[:n_frames]


def speech_regions_from_probs(probs: np.ndarray) -> list[tuple[float, float]]:
    """User-designed density segmentation: fine grid, speech where speech is
    detected often enough; bridge small gaps, drop sub-200ms blips."""

    if probs.size == 0:
        return []
    spf = SILERO_WINDOW / SAMPLE_RATE
    voiced = (probs >= SILERO_THRESHOLD).astype(np.float32)
    kernel = np.ones(SILERO_DENSITY_WIN, dtype=np.float32) / SILERO_DENSITY_WIN
    density = np.convolve(voiced, kernel, mode="same")
    active = density >= SILERO_MIN_DENSITY

    regions: list[list[float]] = []
    index = 0
    total = len(active)
    while index < total:
        if active[index]:
            end = index
            while end < total and active[end]:
                end += 1
            regions.append([index * spf, end * spf])
            index = end
        else:
            index += 1

    bridged: list[list[float]] = []
    for region in regions:
        if bridged and region[0] - bridged[-1][1] <= SILERO_BRIDGE_MS / 1000.0:
            bridged[-1][1] = region[1]
        else:
            bridged.append(region)
    return [(a, b) for a, b in bridged if (b - a) * 1000.0 >= SILERO_MIN_REGION_MS]


def build_superchunks(regions: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    chunks: list[list[float]] = []
    for a, b in regions:
        if chunks and b - chunks[-1][0] <= SUPERCHUNK_TARGET_S:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
    return [(a, b) for a, b in chunks]


def padded_chunk_audio(audio: np.ndarray, chunks: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    total_s = len(audio) / SAMPLE_RATE
    out = []
    for a, b in chunks:
        pa = max(0.0, a - CHUNK_PAD_S)
        pb = min(total_s, b + CHUNK_PAD_S)
        out.append(np.ascontiguousarray(audio[int(pa * SAMPLE_RATE) : int(pb * SAMPLE_RATE)]))
    return out


# ---------------------------------------------------------------------------
# MMS_FA forced alignment
# ---------------------------------------------------------------------------

def _load_mms():
    with _mms_lock:
        if _mms_components["model"] is not None:
            return _mms_components
        # ReDimNet's loader redirects the global torch.hub dir into the model
        # cache; the MMS checkpoint usually already lives in the user hub, so
        # prefer whichever hub dir has it instead of downloading 1.2GB twice.
        previous_hub = torch.hub.get_dir()
        user_hub = _default_user_hub()
        try:
            if (user_hub / "checkpoints").is_dir():
                torch.hub.set_dir(str(user_hub))
            from torchaudio.pipelines import MMS_FA as bundle

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = bundle.get_model().to(device)
            model.eval()
            _mms_components.update(
                {
                    "model": model,
                    "tokenizer": bundle.get_tokenizer(),
                    "aligner": bundle.get_aligner(),
                    "device": device,
                }
            )
            print(f"[INFO-WXC] MMS_FA Alignment-Modell auf '{device}' geladen.", file=sys.stderr)
        finally:
            torch.hub.set_dir(previous_hub)
        return _mms_components


_WORD_KEEP_RE = re.compile(r"[a-z]")


def _normalize_word(word: str) -> str:
    lowered = (
        word.lower()
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    return re.sub(r"[^a-z]", "", lowered)


def align_chunk_words(
    audio: np.ndarray,
    chunks: Sequence[tuple[float, float]],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Word timestamps for every chunk text, deduped at seams by ownership.

    Ownership boundaries sit at the midpoints of the inter-chunk gaps, so a word
    that the +-250ms padding made audible in two neighbouring chunks is emitted
    exactly once.
    """

    components = _load_mms()
    model, tokenizer, aligner = components["model"], components["tokenizer"], components["aligner"]
    device = components["device"]
    total_s = len(audio) / SAMPLE_RATE

    bounds = [float("-inf")]
    for i in range(len(chunks) - 1):
        bounds.append((chunks[i][1] + chunks[i + 1][0]) / 2.0)
    bounds.append(float("inf"))

    words: list[dict[str, Any]] = []
    for index, ((a, b), text) in enumerate(zip(chunks, texts)):
        cleaned = clean_segment_text(text)
        pairs = [(w, _normalize_word(w)) for w in cleaned.split()]
        pairs = [(orig, norm) for orig, norm in pairs if norm and _WORD_KEEP_RE.search(norm)]
        if not pairs:
            continue
        pa = max(0.0, a - CHUNK_PAD_S)
        pb = min(total_s, b + CHUNK_PAD_S)
        clip = torch.from_numpy(
            np.ascontiguousarray(audio[int(pa * SAMPLE_RATE) : int(pb * SAMPLE_RATE)])
        ).unsqueeze(0).to(device)
        with torch.inference_mode():
            emission, _ = model(clip)
        try:
            spans = aligner(emission[0], tokenizer([norm for _, norm in pairs]))
        except Exception as exc:  # alignment must never kill the request
            print(f"[WARNUNG-WXC] Alignment von Chunk {index} fehlgeschlagen: {exc}", file=sys.stderr)
            continue
        ratio = clip.size(1) / emission.size(1)
        for (orig, _), span in zip(pairs, spans):
            t0 = pa + span[0].start * ratio / SAMPLE_RATE
            t1 = pa + span[-1].end * ratio / SAMPLE_RATE
            mid = (t0 + t1) / 2.0
            if bounds[index] < mid <= bounds[index + 1]:
                words.append({"t0": t0, "t1": t1, "word": orig})
    return words


# ---------------------------------------------------------------------------
# Word -> speaker skeleton, sentence regrouping
# ---------------------------------------------------------------------------

def assign_words_to_turns(
    words: Sequence[Mapping[str, Any]],
    gated_turns: Sequence[Mapping[str, Any]],
) -> None:
    turns = [
        {
            "a": float(turn["start_ms"]) / 1000.0,
            "b": float(turn["end_ms"]) / 1000.0,
            "spk": str(turn["speaker_id"]),
        }
        for turn in gated_turns
    ]
    for word in words:
        mid = (word["t0"] + word["t1"]) / 2.0
        speaker = None
        for turn in turns:
            if turn["a"] <= mid <= turn["b"]:
                speaker = turn["spk"]
                break
        if speaker is None and turns:
            speaker = min(
                turns,
                key=lambda t: (t["a"] - mid) if mid < t["a"] else (mid - t["b"]),
            )["spk"]
        word["speaker"] = speaker or "UNKNOWN"


_SENTENCE_END_RE = re.compile(r"[.!?]$")


def words_to_sentences(
    words: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    overlap_spans = [
        (float(o.get("start_ms", 0)) / 1000.0, float(o.get("end_ms", 0)) / 1000.0)
        for o in overlaps
    ]

    def in_overlap(a: float, b: float) -> bool:
        return any(oa < b and ob > a for oa, ob in overlap_spans)

    sentences: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in words:
        if current is not None and (
            word["speaker"] != current["speaker"]
            or word["t0"] - current["t1"] > SENTENCE_GAP_S
        ):
            sentences.append(current)
            current = None
        if current is None:
            current = {
                "t0": word["t0"],
                "t1": word["t1"],
                "speaker": word["speaker"],
                "text": word["word"],
            }
        else:
            current["t1"] = word["t1"]
            current["text"] += " " + word["word"]
        if _SENTENCE_END_RE.search(word["word"]):
            sentences.append(current)
            current = None
    if current is not None:
        sentences.append(current)

    for sentence in sentences:
        sentence["overlap"] = in_overlap(sentence["t0"], sentence["t1"])
    return sentences


# ---------------------------------------------------------------------------
# Iterative 192D verification (purified cores)
# ---------------------------------------------------------------------------

def _loo_cosines(vectors: np.ndarray) -> np.ndarray:
    total = vectors.sum(0)
    out = np.empty(len(vectors), dtype=np.float64)
    for index in range(len(vectors)):
        centroid = total - vectors[index]
        centroid /= np.linalg.norm(centroid) + 1e-12
        out[index] = float(vectors[index] @ centroid)
    return out


def _build_cores(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray]],
    labels: Sequence[str],
) -> dict[str, np.ndarray]:
    cores: dict[str, np.ndarray] = {}
    for speaker in sorted(set(labels)):
        eligible = [
            index
            for index, (unit, _) in enumerate(items)
            if labels[index] == speaker
            and not unit["overlap"]
            and (unit["t1"] - unit["t0"]) >= VERIFY_MIN_SENT_S
        ]
        if len(eligible) < 4:
            eligible = [
                index
                for index, (unit, _) in enumerate(items)
                if labels[index] == speaker and (unit["t1"] - unit["t0"]) >= 1.0
            ]
        if len(eligible) < 3:
            continue
        vectors = np.stack([items[index][1] for index in eligible])
        scores = _loo_cosines(vectors)
        keep = max(3, int(len(eligible) * VERIFY_CORE_KEEP))
        top = np.argsort(scores)[::-1][:keep]
        centroid = np.stack([items[eligible[j]][1] for j in top]).sum(0)
        cores[speaker] = centroid / np.linalg.norm(centroid)
    return cores


def verify_sentences(
    audio: np.ndarray,
    sentences: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Embed sentences (250ms core trim) and conservatively fix speaker labels.

    Only clear cases are changed: non-overlap sentences >= VERIFY_MIN_SENT_S
    whose embedding matches another purified speaker core by >= VERIFY_MARGIN.
    Everything else that looks suspicious is reported as a flag so downstream
    consumers can react without this stage guessing.
    """

    windows: list[VoiceWindow] = []
    kept: list[dict[str, Any]] = []
    for sentence in sentences:
        a, b = sentence["t0"], sentence["t1"]
        if (b - a) >= VERIFY_TRIM_MIN_S:
            a, b = a + VERIFY_TRIM_S, b - VERIFY_TRIM_S
        clip = audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)]
        if len(clip) < int(0.15 * SAMPLE_RATE):
            continue
        windows.append(
            VoiceWindow(
                audio=np.ascontiguousarray(clip),
                start_ms=round(a * 1000),
                end_ms=round(b * 1000),
                clean_duration_seconds=len(clip) / SAMPLE_RATE,
            )
        )
        kept.append(sentence)

    if not windows:
        return {"applied": [], "flags": [], "rounds": 0, "sentence_count": 0}

    embedded = embed_voice_windows(windows, batch_size=16)
    by_start = {item.start_ms: item.vector for item in embedded}
    items: list[tuple[dict[str, Any], np.ndarray]] = []
    for window, sentence in zip(windows, kept):
        vector = by_start.get(window.start_ms)
        if vector is not None:
            items.append((sentence, vector))

    labels = [sentence["speaker"] for sentence, _ in items]
    applied: list[dict[str, Any]] = []
    rounds_run = 0
    for round_number in range(1, VERIFY_MAX_ROUNDS + 1):
        cores = _build_cores(items, labels)
        if len(cores) < 2:
            break
        rounds_run = round_number
        changes = 0
        speakers = sorted(cores)
        for index, (sentence, vector) in enumerate(items):
            if labels[index] not in cores:
                continue
            if (sentence["t1"] - sentence["t0"]) < VERIFY_MIN_SENT_S or sentence["overlap"]:
                continue
            own = float(vector @ cores[labels[index]])
            best_other, best_sim = None, -1.0
            for other in speakers:
                if other == labels[index]:
                    continue
                similarity = float(vector @ cores[other])
                if similarity > best_sim:
                    best_other, best_sim = other, similarity
            if best_other is not None and best_sim >= own + VERIFY_MARGIN:
                applied.append(
                    {
                        "round": round_number,
                        "start_ms": round(sentence["t0"] * 1000),
                        "end_ms": round(sentence["t1"] * 1000),
                        "from": labels[index],
                        "to": best_other,
                        "own_cosine": round(own, 3),
                        "other_cosine": round(best_sim, 3),
                    }
                )
                labels[index] = best_other
                changes += 1
        if changes == 0:
            break

    flags: list[dict[str, Any]] = []
    cores = _build_cores(items, labels)
    speakers = sorted(cores)
    if len(cores) >= 2:
        for index, (sentence, vector) in enumerate(items):
            if labels[index] not in cores:
                continue
            own = float(vector @ cores[labels[index]])
            for other in speakers:
                if other == labels[index]:
                    continue
                similarity = float(vector @ cores[other])
                if similarity >= own + VERIFY_MARGIN:
                    flags.append(
                        {
                            "start_ms": round(sentence["t0"] * 1000),
                            "end_ms": round(sentence["t1"] * 1000),
                            "speaker": labels[index],
                            "suggested": other,
                            "own_cosine": round(own, 3),
                            "other_cosine": round(similarity, 3),
                            "reason": "overlap" if sentence["overlap"] else "too_short",
                        }
                    )
                    break

    for (sentence, _), label in zip(items, labels):
        if label != sentence["speaker"]:
            sentence["verified_from"] = sentence["speaker"]
            sentence["speaker"] = label

    matrix = {
        a: {b: round(float(cores[a] @ cores[b]), 3) for b in speakers}
        for a in speakers
    }
    return {
        "applied": applied,
        "flags": flags,
        "rounds": rounds_run,
        "sentence_count": len(items),
        "core_matrix": matrix,
    }
