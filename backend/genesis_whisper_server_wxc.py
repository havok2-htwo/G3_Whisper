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

from .genesis_whisper_server_gpu import shared_gpu_lease
from .genesis_whisper_server_turn_gate import clean_segment_text
from .genesis_whisper_server_vid import EmbeddedVoiceWindow, VoiceWindow, embed_voice_windows, resolve_local_device

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
# Audio up to this length stays one ASR item (no VAD pass, no seams); above it
# the recording is cut into superchunks so the scheduler can interleave other
# requests between them instead of holding the GPU for one monolithic call.
WHOLE_AUDIO_MAX_S = 60.0
# A language piece shorter than this joins its neighbour instead of becoming its
# own chunk: inside one Silero region (_absorb_language_slivers) and across
# regions (_absorb_short_language_runs).
WXC_LANGUAGE_MIN_PIECE_S = 0.5

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
# Guards inference, not loading: the model is stateful (see silero_frame_probs).
_silero_inference_lock = threading.Lock()
_silero_model: Any | None = None
_mms_lock = threading.Lock()
_mms_components: dict[str, Any] = {"model": None, "tokenizer": None, "aligner": None, "device": None, "dtype": None}
# MMS_FA weights on CUDA. FP16 halves the resident model (1.2 GB -> 0.6 GB) and
# the per-chunk peak; on the 10-min reference meeting all 1297 words, their order,
# speaker assignment and sentences were identical to FP32, mean deviation 0.06 ms,
# max 20 ms (one emission frame, 5 words). Emissions are cast back to FP32 for
# the CTC aligner.
MMS_CUDA_DTYPE = torch.float16


def _default_user_hub() -> Path:
    return Path.home() / ".cache" / "torch" / "hub"


def _silero_local_dir() -> Path | None:
    candidates = [
        _default_user_hub() / "snakers4_silero-vad_master",
        Path(torch.hub.get_dir()) / "snakers4_silero-vad_master",
    ]
    return next((c for c in candidates if c.is_dir()), None)


def silero_available_locally() -> bool:
    """Silero is loaded or its hub copy is on disk (no download needed to use it)."""

    return _silero_model is not None or _silero_local_dir() is not None


def _load_silero_model():
    global _silero_model
    with _silero_lock:
        if _silero_model is not None:
            return _silero_model
        local = _silero_local_dir()
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

    probs = np.empty((n_streams, SILERO_STREAM_FRAMES), dtype=np.float32)
    # Silero is causal, so one stream (a live section, any clip up to ~33 s)
    # stops at its last real frame instead of stepping through the zero padding;
    # the returned values are the same.
    steps = n_frames if n_streams == 1 else SILERO_STREAM_FRAMES
    # The one Silero model carries recurrent state per stream row between calls.
    # Two long requests run this on different asyncio.to_thread workers, so one
    # pass at a time: interleaved steps would corrupt each other's state (or
    # fail on a mismatched batch shape).
    with _silero_inference_lock:
        model.reset_states()
        try:
            with torch.no_grad():
                for frame_index in range(steps):
                    chunk = streams[:, frame_index * SILERO_WINDOW : (frame_index + 1) * SILERO_WINDOW]
                    out = model(chunk, SAMPLE_RATE)
                    probs[:, frame_index] = out.reshape(-1).cpu().numpy()
        finally:
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


def _language_at(moment: float, language_spans: Sequence[tuple[float, float, str]]) -> str:
    """Language of the span covering ``moment``, else of the nearest span (as words find their speaker)."""

    for start, end, language in language_spans:
        if start <= moment <= end:
            return language
    return min(
        language_spans,
        key=lambda span: (span[0] - moment) if moment < span[0] else (moment - span[1]),
    )[2]


def _absorb_language_slivers(pieces: list[list[Any]]) -> list[list[Any]]:
    """Fold sub-WXC_LANGUAGE_MIN_PIECE_S pieces into a neighbour of the same region.

    A turn boundary a few hundred ms inside a Silero region is DIA jitter, not a
    language change; as its own chunk it would reach Cohere as a sliver it fills
    with hallucinated text.
    """

    out: list[list[Any]] = []
    for piece in pieces:
        if out and piece[1] - piece[0] < WXC_LANGUAGE_MIN_PIECE_S:
            out[-1][1] = piece[1]
        elif len(out) == 1 and out[0][1] - out[0][0] < WXC_LANGUAGE_MIN_PIECE_S:
            out[0] = [out[0][0], piece[1], piece[2]]
        else:
            out.append(list(piece))
        if len(out) >= 2 and out[-2][2] == out[-1][2]:
            out[-2][1] = out[-1][1]
            out.pop()
    return out


def _absorb_short_language_runs(pieces: list[list[Any]]) -> list[list[Any]]:
    """Relabel a language run with under WXC_LANGUAGE_MIN_PIECE_S of speech to its nearer neighbour's language.

    _absorb_language_slivers only sees one Silero region. A whole short region of another speaker's language
    (a 0.35 s "yes" between two German regions) would otherwise become its own sub-second chunk that Cohere
    fills with hallucinated text, and it would also split the surrounding same-language speech in two. Runs
    span regions; the shortest run goes first, the neighbour with the smaller pause wins.
    """

    while True:
        runs: list[list[int]] = []
        for index, piece in enumerate(pieces):
            if runs and pieces[runs[-1][0]][2] == piece[2]:
                runs[-1][1] = index + 1
            else:
                runs.append([index, index + 1])
        if len(runs) < 2:
            return pieces
        short = [
            (sum(pieces[k][1] - pieces[k][0] for k in range(first, stop)), first, stop)
            for first, stop in runs
        ]
        short = [run for run in short if run[0] < WXC_LANGUAGE_MIN_PIECE_S]
        if not short:
            return pieces
        _, first, stop = min(short)
        gap_before = pieces[first][0] - pieces[first - 1][1] if first > 0 else float("inf")
        gap_after = pieces[stop][0] - pieces[stop - 1][1] if stop < len(pieces) else float("inf")
        language = pieces[first - 1][2] if gap_before <= gap_after else pieces[stop][2]
        for k in range(first, stop):
            pieces[k][2] = language


def build_language_superchunks(
    regions: Sequence[tuple[float, float]],
    language_spans: Sequence[tuple[float, float, str]],
) -> tuple[list[tuple[float, float]], list[str]]:
    """Superchunks that never mix two languages, plus each chunk's language, in time order.

    ``language_spans`` are labeled spans (start s, end s, language): the gated
    turns with their turn language in diarization (the speaker's language unless
    the turn's own detection overrode it), the labeled speech pieces of a long
    transcript. Every Silero region is cut where the covering span's language
    changes (speech outside any span belongs to the nearest one), pieces under
    WXC_LANGUAGE_MIN_PIECE_S fold into a neighbour, then the pieces merge to
    ~SUPERCHUNK_TARGET_S exactly like build_superchunks, but a chunk only grows
    while the language stays the same. With a single language the result equals
    build_superchunks(regions).
    """

    spans = sorted(language_spans)
    if not spans:
        chunks = build_superchunks(regions)
        return chunks, []
    pieces: list[list[Any]] = []
    for region_start, region_end in regions:
        cuts = sorted(
            {region_start, region_end}
            | {point for start, end, _ in spans for point in (start, end) if region_start < point < region_end}
        )
        region_pieces: list[list[Any]] = []
        for start, end in zip(cuts, cuts[1:]):
            if end <= start:
                continue
            language = _language_at((start + end) / 2.0, spans)
            if region_pieces and region_pieces[-1][2] == language:
                region_pieces[-1][1] = end
            else:
                region_pieces.append([start, end, language])
        pieces.extend(_absorb_language_slivers(region_pieces))
    pieces = _absorb_short_language_runs(pieces)

    chunks: list[list[float]] = []
    languages: list[str] = []
    for start, end, language in pieces:
        if chunks and languages[-1] == language and end - chunks[-1][0] <= SUPERCHUNK_TARGET_S:
            chunks[-1][1] = end
        else:
            chunks.append([start, end])
            languages.append(language)
    return [(a, b) for a, b in chunks], languages


def padded_chunk_audio(
    audio: np.ndarray,
    chunks: Sequence[tuple[float, float]],
    *,
    split_seams: bool = False,
) -> list[np.ndarray]:
    """Each chunk's audio, CHUNK_PAD_S wider on both sides.

    ``split_seams``: towards a neighbouring chunk closer than two pads, pad only half the gap, so touching
    chunks (a language change inside one Silero region) never share audio. For callers without a word-level
    dedupe at the seams (transcript mode); WXC keeps the full pad because its alignment dedupes.
    """

    total_s = len(audio) / SAMPLE_RATE
    out = []
    for index, (a, b) in enumerate(chunks):
        pad_before = pad_after = CHUNK_PAD_S
        if split_seams:
            if index > 0:
                pad_before = min(CHUNK_PAD_S, max(0.0, a - chunks[index - 1][1]) / 2.0)
            if index + 1 < len(chunks):
                pad_after = min(CHUNK_PAD_S, max(0.0, chunks[index + 1][0] - b) / 2.0)
        pa = max(0.0, a - pad_before)
        pb = min(total_s, b + pad_after)
        out.append(np.ascontiguousarray(audio[int(pa * SAMPLE_RATE) : int(pb * SAMPLE_RATE)]))
    return out


def split_audio_into_superchunks(
    audio: np.ndarray,
    *,
    whole_audio_max_seconds: float = WHOLE_AUDIO_MAX_S,
) -> list[np.ndarray]:
    """Scheduler-visible chunks for backends that accept a whole recording.

    Cohere chunks long audio internally, but the batch scheduler only sees the
    items it is handed: a two-hour file as one item is one GPU-lock hold for
    the entire transcription, during which no live request can be served.
    Long recordings therefore take the WXC superchunk path (Silero regions
    merged to ~25s, padded 250ms); short ones stay untouched.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    regions = superchunk_speech_regions(samples, whole_audio_max_seconds=whole_audio_max_seconds)
    if regions is None:
        return [samples]
    return padded_chunk_audio(samples, build_superchunks(regions))


def superchunk_speech_regions(
    audio: np.ndarray,
    *,
    whole_audio_max_seconds: float = WHOLE_AUDIO_MAX_S,
) -> list[tuple[float, float]] | None:
    """The Silero speech regions split_audio_into_superchunks merges, or None when the audio stays one item.

    None means: short enough to be one ASR item, or Silero found no speech (never drop audio silently: the
    model then decides what the file holds). Language detection reuses the regions to label speech pieces.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(samples) <= int(whole_audio_max_seconds * SAMPLE_RATE):
        return None
    regions = speech_regions_from_probs(silero_frame_probs(samples))
    return regions or None


# A long region is split near its even cut points, at the quietest 20 ms frame within this distance, so a
# language change there does not cut through a word.
REGION_CUT_SEARCH_S = 1.0
REGION_CUT_FRAME_S = 0.02


def _quietest_moment(audio: np.ndarray, lo: float, hi: float) -> float:
    """Centre (s) of the 20 ms frame with the least energy in [lo, hi]; on a tie the one nearest the middle."""

    frame = max(1, int(REGION_CUT_FRAME_S * SAMPLE_RATE))
    first = max(0, int(lo * SAMPLE_RATE))
    last = min(len(audio), int(hi * SAMPLE_RATE))
    count = (last - first) // frame
    if count < 1:
        return (lo + hi) / 2.0
    frames = np.asarray(audio[first : first + count * frame], dtype=np.float32).reshape(count, frame)
    centres = (first + np.arange(count) * frame + frame / 2.0) / SAMPLE_RATE
    energy = np.einsum("ij,ij->i", frames, frames)
    best = int(np.lexsort((np.abs(centres - (lo + hi) / 2.0), energy))[0])
    return float(centres[best])


def region_pieces(
    regions: Sequence[tuple[float, float]],
    max_seconds: float,
    audio: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Every region as one piece, a region longer than ``max_seconds`` as parts of at most that length.

    Without ``audio`` the parts are equal. With it each cut moves to the quietest moment within
    REGION_CUT_SEARCH_S of its even position (the parts are planned REGION_CUT_SEARCH_S shorter, so they stay
    within ``max_seconds``).
    """

    pieces: list[tuple[float, float]] = []
    search = REGION_CUT_SEARCH_S if audio is not None and max_seconds > 4 * REGION_CUT_SEARCH_S else 0.0
    for start, end in regions:
        if end - start <= max_seconds:
            pieces.append((start, end))
            continue
        count = max(1, int(np.ceil((end - start) / (max_seconds - 2 * search) - 1e-9)))
        size = (end - start) / count
        cuts = [start + index * size for index in range(1, count)]
        if search:
            cuts = [_quietest_moment(audio, cut - search, cut + search) for cut in cuts]
        bounds = [start, *cuts, end]
        pieces.extend(zip(bounds, bounds[1:]))
    return pieces


# ---------------------------------------------------------------------------
# MMS_FA forced alignment
# ---------------------------------------------------------------------------

def _mms_dtype(device: torch.device) -> torch.dtype:
    return MMS_CUDA_DTYPE if device.type == "cuda" else torch.float32


def _move_mms(target_device: torch.device) -> None:
    """Move the loaded MMS_FA model to ``target_device`` (caller holds ``_mms_lock``)."""

    previous = _mms_components["device"]
    dtype = _mms_dtype(target_device)
    _mms_components["model"].to(device=target_device, dtype=dtype)
    _mms_components.update({"device": target_device, "dtype": dtype})
    # The old card keeps the weights' blocks cached until the pool is released.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"[INFO-WXC] MMS_FA Alignment-Modell von '{previous}' nach '{target_device}' verschoben.", file=sys.stderr)


def apply_mms_device_setting() -> bool:
    """Admin hook: move a loaded MMS_FA model to the current ``local_gpu_device``.

    Returns True when it moved. A not yet loaded model needs nothing; its first
    load resolves the device itself.
    """

    with _mms_lock:
        if _mms_components["model"] is None:
            return False
        target_device = resolve_local_device()
        if str(_mms_components["device"]) == str(target_device):
            return False
        _move_mms(target_device)
        return True


# torch.hub caches the MMS_FA weights as <hub>/checkpoints/<url basename>; torchaudio's MMS_FA URL is
# .../mms/torchaudio/ctc_alignment_mling_uroman/model.pt. A constant, so the capability check imports nothing.
MMS_CHECKPOINT_FILE = "model.pt"


def _mms_hub_choice() -> tuple[Path, bool]:
    """(hub dir to load MMS_FA from, whether its checkpoint is already there).

    ReDimNet's loader redirects the global torch.hub dir into the model cache;
    the MMS checkpoint usually lives in the user hub. Whichever hub has the file
    wins; without one the user hub is used when it has a checkpoints folder
    (as before), else the current hub dir.
    """

    name = MMS_CHECKPOINT_FILE
    user_hub = _default_user_hub()
    current_hub = Path(torch.hub.get_dir())
    for hub in (user_hub, current_hub):
        if (hub / "checkpoints" / name).is_file():
            return hub, True
    return (user_hub if (user_hub / "checkpoints").is_dir() else current_hub), False


def mms_available_locally() -> bool:
    """MMS_FA is loaded or its checkpoint is on disk: using it downloads nothing (cheap, no model load)."""

    return _mms_components["model"] is not None or _mms_hub_choice()[1]


def _load_mms(*, allow_download: bool = True):
    with _mms_lock:
        if _mms_components["model"] is not None:
            # Follow a changed local_gpu_device instead of staying on the old card.
            target_device = resolve_local_device()
            if str(_mms_components["device"]) != str(target_device):
                _move_mms(target_device)
            return _mms_components
        # Prefer whichever hub dir has the checkpoint instead of downloading
        # 1.2GB twice (see _mms_hub_choice).
        previous_hub = torch.hub.get_dir()
        hub, present = _mms_hub_choice()
        if not present and not allow_download:
            raise RuntimeError("MMS_FA-Checkpoint fehlt lokal; ohne Download keine Wortzeiten.")
        try:
            torch.hub.set_dir(str(hub))
            from torchaudio.pipelines import MMS_FA as bundle

            # Same card as the ASR engine (local_gpu_device). torch.device("cuda")
            # would be cuda:0, which on the 3090 + 2070S box can be the DIA card.
            device = resolve_local_device()
            dtype = _mms_dtype(device)
            model = bundle.get_model().to(device=device, dtype=dtype)
            model.eval()
            _mms_components.update(
                {
                    "model": model,
                    "tokenizer": bundle.get_tokenizer(),
                    "aligner": bundle.get_aligner(),
                    "device": device,
                    "dtype": dtype,
                }
            )
            print(f"[INFO-WXC] MMS_FA Alignment-Modell auf '{device}' ({dtype}) geladen.", file=sys.stderr)
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


def alignment_bounds(chunks: Sequence[tuple[float, float]]) -> list[float]:
    """Ownership boundaries at the midpoints of the inter-chunk gaps."""

    bounds = [float("-inf")]
    for i in range(len(chunks) - 1):
        bounds.append((chunks[i][1] + chunks[i + 1][0]) / 2.0)
    bounds.append(float("inf"))
    return bounds


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

    return align_chunk_slice(list(range(len(chunks))), audio, chunks, texts, alignment_bounds(chunks))


def align_chunk_slice(
    indices: Sequence[int],
    audio: np.ndarray,
    chunks: Sequence[tuple[float, float]],
    texts: Sequence[str],
    bounds: Sequence[float],
    allow_download: bool = True,
) -> list[dict[str, Any]]:
    """Align only the chunks in ``indices``; the GPU unit of the sliced driver.

    ``allow_download`` False (the live speaker_turns path) raises instead of
    fetching a missing MMS_FA checkpoint under the GPU lock.

    Seam ownership only needs the neighbouring chunk boundaries, so any subset
    of chunks can be aligned independently and the per-slice word lists simply
    concatenate in chunk order. Holds the cross-container GPU lease like every
    other CUDA phase, so a colocated DIA never runs pyannote next to MMS_FA.
    """

    total_s = len(audio) / SAMPLE_RATE
    words: list[dict[str, Any]] = []
    with shared_gpu_lease():
        components = _load_mms() if allow_download else _load_mms(allow_download=False)
        model, tokenizer, aligner = components["model"], components["tokenizer"], components["aligner"]
        device = components["device"]
        dtype = components.get("dtype") or torch.float32

        for index in indices:
            (a, b), text = chunks[index], texts[index]
            cleaned = clean_segment_text(text)
            pairs = [(w, _normalize_word(w)) for w in cleaned.split()]
            pairs = [(orig, norm) for orig, norm in pairs if norm and _WORD_KEEP_RE.search(norm)]
            if not pairs:
                continue
            pa = max(0.0, a - CHUNK_PAD_S)
            pb = min(total_s, b + CHUNK_PAD_S)
            clip = torch.from_numpy(
                np.ascontiguousarray(audio[int(pa * SAMPLE_RATE) : int(pb * SAMPLE_RATE)])
            ).unsqueeze(0).to(device=device, dtype=dtype)
            with torch.inference_mode():
                emission, _ = model(clip)
                emission = emission.float()
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


def plan_verification_windows(
    audio: np.ndarray,
    sentences: Sequence[dict[str, Any]],
) -> tuple[list[VoiceWindow], list[dict[str, Any]]]:
    """CPU part of verification: one 250ms-core-trimmed window per usable sentence."""

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
    return windows, kept


def verify_sentences(
    audio: np.ndarray,
    sentences: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Embed sentences (250ms core trim) and conservatively fix speaker labels.

    Only clear cases are changed: non-overlap sentences >= VERIFY_MIN_SENT_S
    whose embedding matches another purified speaker core by >= VERIFY_MARGIN.
    Everything else that looks suspicious is reported as a flag so downstream
    consumers can react without this stage guessing.

    The request path runs the same three steps with the embedding sliced under
    the GPU lock (see ``plan_verification_windows`` / ``finish_verification``).
    """

    windows, kept = plan_verification_windows(audio, sentences)
    if not windows:
        return finish_verification(windows, kept, [])
    return finish_verification(windows, kept, embed_voice_windows(windows))


def finish_verification(
    windows: Sequence[VoiceWindow],
    kept: Sequence[dict[str, Any]],
    embedded: Sequence[EmbeddedVoiceWindow],
) -> dict[str, Any]:
    """CPU part of verification: purified cores, relabel rounds and flags."""

    if not windows:
        return {"applied": [], "flags": [], "rounds": 0, "sentence_count": 0}

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
