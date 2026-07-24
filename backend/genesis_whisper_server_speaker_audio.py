"""Build compact listening samples for unknown diarization speakers."""

from __future__ import annotations

import base64
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .genesis_whisper_server_speaker_matching import SpeakerCloud
from .genesis_whisper_server_vid import REDIMNET_SAMPLE_RATE


UNKNOWN_SPEAKER_AUDIO_MIME_TYPE = "audio/mpeg"
UNKNOWN_SPEAKER_AUDIO_BITRATE = "64k"
UNKNOWN_SPEAKER_AUDIO_MAX_DURATION_MS = 30_000
UNKNOWN_SPEAKER_AUDIO_MIN_SNIPPET_MS = 5_000
UNKNOWN_SPEAKER_AUDIO_MIN_QUALITY = 0.60
UNKNOWN_SPEAKER_AUDIO_FADE_MS = 10
UNKNOWN_SPEAKER_AUDIO_CONTIGUOUS_TOLERANCE_MS = 1
UNKNOWN_SPEAKER_AUDIO_FFMPEG_TIMEOUT_SECONDS = 60

# A listening reference must always exist ("no unknown speaker without audio").
# When the strict clean-inlier selection finds nothing (typically mixed_cluster
# speakers), the relaxed tier lowers the bars, and as a last resort the sample
# is cut straight from the speaker's exclusive DIA turns.
UNKNOWN_SPEAKER_AUDIO_RELAXED_MIN_QUALITY = 0.35
UNKNOWN_SPEAKER_AUDIO_RELAXED_MIN_SNIPPET_MS = 2_000
UNKNOWN_SPEAKER_AUDIO_TURN_MIN_PIECE_MS = 700
UNKNOWN_SPEAKER_AUDIO_TURN_MAX_DURATION_MS = 15_000


@dataclass(frozen=True)
class _ScoredWindow:
    start_ms: int
    end_ms: int
    centrality: float
    quality: float


@dataclass(frozen=True)
class _SourceRun:
    start_ms: int
    end_ms: int
    windows: tuple[_ScoredWindow, ...]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class _SelectedSnippet:
    start_ms: int
    end_ms: int
    centrality: float
    quality: float

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left).reshape(-1)
    right_array = np.asarray(right).reshape(-1)
    if left_array.shape != right_array.shape:
        return None
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if not np.isfinite(left_norm) or not np.isfinite(right_norm):
        return None
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    if not np.isfinite(similarity):
        return None
    return min(1.0, max(-1.0, similarity))


def _eligible_windows(
    cloud: SpeakerCloud,
    *,
    audio_duration_ms: int,
    min_quality: float = UNKNOWN_SPEAKER_AUDIO_MIN_QUALITY,
) -> list[_ScoredWindow]:
    if cloud.prototype is None:
        return []

    by_interval: dict[tuple[int, int], _ScoredWindow] = {}
    for sample in cloud.inliers:
        if sample.stitched or sample.start_ms is None or sample.end_ms is None:
            continue
        if not np.isfinite(sample.quality) or sample.quality < min_quality:
            continue
        start_ms = max(0, int(sample.start_ms))
        end_ms = min(audio_duration_ms, int(sample.end_ms))
        if end_ms <= start_ms:
            continue
        centrality = _cosine_similarity(sample.vector, cloud.prototype)
        if centrality is None:
            continue
        candidate = _ScoredWindow(
            start_ms=start_ms,
            end_ms=end_ms,
            centrality=centrality,
            quality=float(sample.quality),
        )
        key = (start_ms, end_ms)
        previous = by_interval.get(key)
        if previous is None or (candidate.centrality, candidate.quality) > (
            previous.centrality,
            previous.quality,
        ):
            by_interval[key] = candidate

    return sorted(
        by_interval.values(),
        key=lambda item: (item.start_ms, item.end_ms, -item.centrality, -item.quality),
    )


def _contiguous_runs(
    windows: Sequence[_ScoredWindow],
    min_snippet_ms: int = UNKNOWN_SPEAKER_AUDIO_MIN_SNIPPET_MS,
) -> list[_SourceRun]:
    runs: list[_SourceRun] = []
    current: list[_ScoredWindow] = []
    current_start = 0
    current_end = 0
    for window in windows:
        if not current:
            current = [window]
            current_start = window.start_ms
            current_end = window.end_ms
            continue
        if window.start_ms <= current_end + UNKNOWN_SPEAKER_AUDIO_CONTIGUOUS_TOLERANCE_MS:
            current.append(window)
            current_end = max(current_end, window.end_ms)
            continue
        if current_end - current_start >= min_snippet_ms:
            runs.append(_SourceRun(current_start, current_end, tuple(current)))
        current = [window]
        current_start = window.start_ms
        current_end = window.end_ms

    if current and current_end - current_start >= min_snippet_ms:
        runs.append(_SourceRun(current_start, current_end, tuple(current)))
    return runs


def _span_score(
    windows: Sequence[_ScoredWindow],
    start_ms: int,
    end_ms: int,
) -> tuple[float, float]:
    """Return duration/quality-weighted centrality and mean quality."""

    centrality_sum = 0.0
    quality_sum = 0.0
    weight_sum = 0.0
    duration_sum = 0.0
    for window in windows:
        overlap_ms = max(0, min(end_ms, window.end_ms) - max(start_ms, window.start_ms))
        if overlap_ms <= 0:
            continue
        weighted_duration = overlap_ms * window.quality
        centrality_sum += window.centrality * weighted_duration
        weight_sum += weighted_duration
        quality_sum += window.quality * overlap_ms
        duration_sum += overlap_ms
    if weight_sum <= 0.0 or duration_sum <= 0.0:
        return -1.0, 0.0
    return centrality_sum / weight_sum, quality_sum / duration_sum


def _best_slice(run: _SourceRun, duration_ms: int) -> _SelectedSnippet:
    duration_ms = min(duration_ms, run.duration_ms)
    if duration_ms == run.duration_ms:
        centrality, quality = _span_score(run.windows, run.start_ms, run.end_ms)
        return _SelectedSnippet(run.start_ms, run.end_ms, centrality, quality)

    latest_start = run.end_ms - duration_ms
    starts = {run.start_ms, latest_start}
    for window in run.windows:
        starts.add(min(max(window.start_ms, run.start_ms), latest_start))
        starts.add(min(max(window.end_ms - duration_ms, run.start_ms), latest_start))

    candidates: list[_SelectedSnippet] = []
    for start_ms in starts:
        end_ms = start_ms + duration_ms
        centrality, quality = _span_score(run.windows, start_ms, end_ms)
        candidates.append(_SelectedSnippet(start_ms, end_ms, centrality, quality))
    return min(
        candidates,
        key=lambda item: (-item.centrality, -item.quality, item.start_ms, item.end_ms),
    )


def _select_snippets(
    cloud: SpeakerCloud,
    audio_duration_ms: int,
    *,
    min_quality: float = UNKNOWN_SPEAKER_AUDIO_MIN_QUALITY,
    min_snippet_ms: int = UNKNOWN_SPEAKER_AUDIO_MIN_SNIPPET_MS,
) -> list[_SelectedSnippet]:
    runs = _contiguous_runs(
        _eligible_windows(cloud, audio_duration_ms=audio_duration_ms, min_quality=min_quality),
        min_snippet_ms,
    )
    remaining_ms = UNKNOWN_SPEAKER_AUDIO_MAX_DURATION_MS
    selected: list[_SelectedSnippet] = []
    remaining_runs = list(runs)
    while remaining_ms >= min_snippet_ms and remaining_runs:
        candidates = [
            (
                _best_slice(run, min(run.duration_ms, remaining_ms)),
                run,
            )
            for run in remaining_runs
            if min(run.duration_ms, remaining_ms) >= min_snippet_ms
        ]
        if not candidates:
            break
        snippet, chosen_run = min(
            candidates,
            key=lambda item: (
                -item[0].centrality,
                -item[0].quality,
                -item[0].duration_ms,
                item[0].start_ms,
                item[0].end_ms,
            ),
        )
        selected.append(snippet)
        remaining_ms -= snippet.duration_ms
        remaining_runs.remove(chosen_run)

    return sorted(selected, key=lambda item: (item.start_ms, item.end_ms))


def _pcm_for_snippets(
    audio: np.ndarray,
    snippets: Sequence[_SelectedSnippet],
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    fade_samples = round(UNKNOWN_SPEAKER_AUDIO_FADE_MS * REDIMNET_SAMPLE_RATE / 1000)
    for snippet in snippets:
        start_sample = round(snippet.start_ms * REDIMNET_SAMPLE_RATE / 1000)
        end_sample = round(snippet.end_ms * REDIMNET_SAMPLE_RATE / 1000)
        piece = np.asarray(audio[start_sample:end_sample], dtype=np.float32).copy()
        if piece.size == 0:
            continue
        if not np.all(np.isfinite(piece)):
            raise RuntimeError("Ausgewaehltes Sprecher-Audio enthaelt ungueltige Sample-Werte.")
        np.clip(piece, -1.0, 1.0, out=piece)
        applied_fade = min(fade_samples, len(piece) // 2)
        if applied_fade > 0:
            fade_in = np.linspace(0.0, 1.0, applied_fade, endpoint=True, dtype=np.float32)
            piece[:applied_fade] *= fade_in
            piece[-applied_fade:] *= fade_in[::-1]
        pieces.append(piece)
    if not pieces:
        return np.empty(0, dtype=np.float32)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _encode_mp3(pcm: np.ndarray) -> bytes:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg ist fuer unbekannte Sprecher-Hoerproben nicht verfuegbar.")

    pcm_s16 = np.rint(pcm * 32767.0).astype("<i2", copy=False)
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(REDIMNET_SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        UNKNOWN_SPEAKER_AUDIO_BITRATE,
        "-f",
        "mp3",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=pcm_s16.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=UNKNOWN_SPEAKER_AUDIO_FFMPEG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ffmpeg konnte die Sprecher-Hoerprobe nicht kodieren: {exc}") from exc
    if process.returncode != 0 or not process.stdout:
        error_text = process.stderr.decode("utf-8", errors="replace").strip()
        if len(error_text) > 1000:
            error_text = error_text[:1000] + "..."
        raise RuntimeError(
            "ffmpeg konnte die Sprecher-Hoerprobe nicht kodieren: "
            + (error_text or "Unbekannter ffmpeg-Fehler.")
        )
    return process.stdout


def _turn_fallback_snippets(
    exclusive_turns: Sequence[Mapping[str, Any]],
    speaker_id: str,
    audio_duration_ms: int,
) -> list[_SelectedSnippet]:
    """Last-resort sample straight from the speaker's exclusive DIA turns.

    A mixed_cluster speaker has no clean embedding windows, but the human on the
    other end still needs SOMETHING to listen to.  Longest turns first, small
    pieces skipped, capped below the clean-tier length so the lower fidelity is
    also visible in the duration.
    """

    pieces: list[_SelectedSnippet] = []
    turns = [
        turn
        for turn in exclusive_turns
        if str(turn.get("speaker_id", "")).strip() == speaker_id
    ]
    turns.sort(key=lambda t: int(t["end_ms"]) - int(t["start_ms"]), reverse=True)
    remaining_ms = UNKNOWN_SPEAKER_AUDIO_TURN_MAX_DURATION_MS
    for turn in turns:
        start_ms = max(0, int(turn["start_ms"]))
        end_ms = min(audio_duration_ms, int(turn["end_ms"]))
        duration = end_ms - start_ms
        if duration < UNKNOWN_SPEAKER_AUDIO_TURN_MIN_PIECE_MS:
            continue
        take = min(duration, remaining_ms)
        pieces.append(_SelectedSnippet(start_ms, start_ms + take, 0.0, 0.0))
        remaining_ms -= take
        if remaining_ms < UNKNOWN_SPEAKER_AUDIO_TURN_MIN_PIECE_MS:
            break
    return sorted(pieces, key=lambda item: (item.start_ms, item.end_ms))


def build_unknown_speaker_audio_assets(
    audio: np.ndarray,
    speaker_clouds: Mapping[str, SpeakerCloud],
    speaker_ids: Iterable[str],
    exclusive_turns: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return base64 MP3 listening samples for unknown speakers.

    Three tiers guarantee "no unknown speaker without audio": the strict
    clean-inlier selection, a relaxed variant for weak clouds, and finally raw
    exclusive-turn audio.  ``quality_tier`` tells the client which one it got.
    """

    audio_samples = np.asarray(audio)
    if audio_samples.ndim != 1:
        raise ValueError("Sprecher-Hoerproben benoetigen einkanaliges Audio.")
    audio_duration_ms = len(audio_samples) * 1000 // REDIMNET_SAMPLE_RATE
    assets: dict[str, dict[str, Any]] = {}
    raw_speaker_ids: Iterable[str] = [speaker_ids] if isinstance(speaker_ids, str) else speaker_ids
    normalized_speaker_ids = {
        str(item).strip()
        for item in raw_speaker_ids
        if item is not None and str(item).strip()
    }
    for speaker_id in sorted(normalized_speaker_ids):
        cloud = speaker_clouds.get(speaker_id)
        if cloud is None:
            continue
        snippets = _select_snippets(cloud, audio_duration_ms)
        quality_tier = "clean"
        if not snippets:
            snippets = _select_snippets(
                cloud,
                audio_duration_ms,
                min_quality=UNKNOWN_SPEAKER_AUDIO_RELAXED_MIN_QUALITY,
                min_snippet_ms=UNKNOWN_SPEAKER_AUDIO_RELAXED_MIN_SNIPPET_MS,
            )
            quality_tier = "relaxed"
        if not snippets and exclusive_turns:
            snippets = _turn_fallback_snippets(exclusive_turns, speaker_id, audio_duration_ms)
            quality_tier = "turns_fallback"
        if not snippets:
            continue
        pcm = _pcm_for_snippets(audio_samples, snippets)
        if pcm.size == 0:
            continue
        if len(pcm) > round(UNKNOWN_SPEAKER_AUDIO_MAX_DURATION_MS * REDIMNET_SAMPLE_RATE / 1000):
            raise RuntimeError("Interner Fehler: Sprecher-Hoerprobe ueberschreitet 30 Sekunden.")
        mp3_bytes = _encode_mp3(pcm)
        duration_ms = round(len(pcm) * 1000 / REDIMNET_SAMPLE_RATE)
        assets[speaker_id] = {
            "mime_type": UNKNOWN_SPEAKER_AUDIO_MIME_TYPE,
            "encoding": "base64",
            "data": base64.b64encode(mp3_bytes).decode("ascii"),
            "duration_ms": duration_ms,
            "quality_tier": quality_tier,
            "snippets": [
                {
                    "start_ms": snippet.start_ms,
                    "end_ms": snippet.end_ms,
                    "duration_ms": snippet.duration_ms,
                    "centrality": round(snippet.centrality, 6),
                }
                for snippet in snippets
            ],
        }
    return assets
