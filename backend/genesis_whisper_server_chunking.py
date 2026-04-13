from __future__ import annotations

from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np


TARGET_SAMPLE_RATE = 16000


def extract_speech_audio(
    audio_data_np: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    padding_ms: int = 120,
) -> np.ndarray:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Speech-Extraction erwartet {TARGET_SAMPLE_RATE}Hz, erhielt aber {sample_rate}Hz.")

    if audio_data_np is None or len(audio_data_np) == 0:
        return np.asarray([], dtype=np.float32)

    audio = np.asarray(audio_data_np, dtype=np.float32).flatten()
    segments = _detect_speech_segments(audio, sample_rate, padding_ms=padding_ms)
    if not segments:
        return np.asarray([], dtype=np.float32)

    speech_parts = [audio[start_sample:end_sample] for start_sample, end_sample in segments if end_sample > start_sample]
    if not speech_parts:
        return np.asarray([], dtype=np.float32)
    if len(speech_parts) == 1:
        return speech_parts[0].copy()
    return np.concatenate(speech_parts).astype(np.float32, copy=False)


def split_audio_for_whisper(
    audio_data_np: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_chunk_seconds: float = 30.0,
    padding_ms: int = 180,
    overlap_ms: int = 250,
) -> List[np.ndarray]:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Whisper-Chunking erwartet {TARGET_SAMPLE_RATE}Hz, erhielt aber {sample_rate}Hz.")

    if audio_data_np is None or len(audio_data_np) == 0:
        return []

    audio = np.asarray(audio_data_np, dtype=np.float32).flatten()
    max_chunk_samples = max(int(max_chunk_seconds * sample_rate), sample_rate)

    # Short audio (≤ one Whisper window): skip VAD entirely to avoid
    # cutting speech at edges where TTS fades in/out quietly.
    if len(audio) <= max_chunk_samples:
        return [audio.copy()]

    # Longer audio: use VAD to find speech regions, then chunk each one.
    segments = _detect_speech_segments(audio, sample_rate, padding_ms=padding_ms)

    # If VAD finds nothing at all, treat the whole file as one speech region
    # This guarantees we NEVER silently drop audio segments. Whisper will figure it out.
    if not segments:
        segments = [(0, len(audio))]

    overlap_samples = max(0, int(overlap_ms * sample_rate / 1000))
    chunks: List[np.ndarray] = []

    for start_sample, end_sample in segments:
        segment = audio[start_sample:end_sample]
        if len(segment) <= max_chunk_samples:
            chunks.append(segment.copy())
            continue

        step = max(max_chunk_samples - overlap_samples, sample_rate)
        for offset in range(0, len(segment), step):
            chunk = segment[offset:offset + max_chunk_samples]
            if len(chunk) == 0:
                continue
            chunks.append(chunk.copy())
            if offset + max_chunk_samples >= len(segment):
                break

    return chunks


def combine_transcription_chunks(text_chunks: Sequence[str]) -> str:
    return " ".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip()).strip()


def _detect_speech_segments(audio: np.ndarray, sample_rate: int, padding_ms: int) -> List[Tuple[int, int]]:
    # WebRTC VAD completely disabled as it fails on telephone/filtered voice.
    # We strictly rely on the energy (volume) based detection.
    segments = _detect_segments_with_energy(audio, sample_rate, padding_ms)
    # Only split segments if the silence between them is at least 300ms (0.3 seconds)
    return _merge_segments(segments, merge_gap_samples=int(0.30 * sample_rate))


def _detect_segments_with_webrtcvad(audio: np.ndarray, sample_rate: int, padding_ms: int, vad_aggressiveness: int = 0) -> List[Tuple[int, int]]:
    try:
        import webrtcvad
    except Exception:
        return []

    frame_ms = 30
    frame_samples = int(sample_rate * frame_ms / 1000)
    if len(audio) < frame_samples:
        return []

    # vad_aggressiveness: 0 (least aggressive / most sensitive) to 3 (most aggressive)
    vad = webrtcvad.Vad(vad_aggressiveness)
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    frames = []
    for start in range(0, len(pcm16) - frame_samples + 1, frame_samples):
        frame = pcm16[start:start + frame_samples]
        frames.append((start, start + frame_samples, vad.is_speech(frame.tobytes(), sample_rate)))

    padding_frames = max(1, padding_ms // frame_ms)
    ring_buffer: deque[Tuple[int, int, bool]] = deque(maxlen=padding_frames)
    voiced_segments: List[Tuple[int, int]] = []
    triggered = False
    start_sample = 0

    for frame_start, frame_end, is_speech in frames:
        if not triggered:
            ring_buffer.append((frame_start, frame_end, is_speech))
            voiced_count = sum(1 for _, _, voiced in ring_buffer if voiced)
            if voiced_count >= max(1, int(0.7 * ring_buffer.maxlen)):
                triggered = True
                start_sample = ring_buffer[0][0]
                ring_buffer.clear()
        else:
            ring_buffer.append((frame_start, frame_end, is_speech))
            unvoiced_count = sum(1 for _, _, voiced in ring_buffer if not voiced)
            if unvoiced_count >= max(1, int(0.7 * ring_buffer.maxlen)):
                end_sample = frame_end
                voiced_segments.append((start_sample, end_sample))
                ring_buffer.clear()
                triggered = False

    if triggered:
        voiced_segments.append((start_sample, len(audio)))

    return _apply_padding(voiced_segments, len(audio), int(padding_ms * sample_rate / 1000))


def _detect_segments_with_energy(audio: np.ndarray, sample_rate: int, padding_ms: int) -> List[Tuple[int, int]]:
    frame_ms = 30
    frame_samples = int(sample_rate * frame_ms / 1000)
    if len(audio) < frame_samples:
        if float(np.max(np.abs(audio))) < 0.0001:
            return []
        return [(0, len(audio))]

    frame_starts = list(range(0, len(audio) - frame_samples + 1, frame_samples))
    energies = np.array([float(np.sqrt(np.mean(np.square(audio[start:start + frame_samples])))) for start in frame_starts], dtype=np.float32)

    if len(energies) == 0 or float(np.max(energies)) < 0.0001:
        return []

    p05_noise_floor = float(np.percentile(energies, 5))
    p95_speech_peak = float(np.percentile(energies, 95))

    # The threshold is dynamically calculated based on the audio's own volume distribution.
    # We take the background noise (p05) and add 5% of the dynamic range up to the peak (p95).
    # This means anything that is 5% "louder" than the pure background noise counts as speech.
    dynamic_threshold = p05_noise_floor + (p95_speech_peak - p05_noise_floor) * 0.05
    
    # We ensure a hard minimum to avoid false positives on totally silent files.
    threshold = max(0.0001, dynamic_threshold)
    voiced_flags = energies >= threshold
    min_run_frames = max(1, int(150 / frame_ms))

    segments: List[Tuple[int, int]] = []
    run_start = None
    for index, is_voiced in enumerate(voiced_flags):
        if is_voiced and run_start is None:
            run_start = index
        elif not is_voiced and run_start is not None:
            if index - run_start >= min_run_frames:
                segments.append((frame_starts[run_start], frame_starts[index - 1] + frame_samples))
            run_start = None

    if run_start is not None and len(voiced_flags) - run_start >= min_run_frames:
        segments.append((frame_starts[run_start], frame_starts[-1] + frame_samples))

    return _apply_padding(segments, len(audio), int(padding_ms * sample_rate / 1000))


def _apply_padding(segments: Iterable[Tuple[int, int]], audio_length: int, padding_samples: int) -> List[Tuple[int, int]]:
    padded: List[Tuple[int, int]] = []
    for start_sample, end_sample in segments:
        start_sample = max(0, start_sample - padding_samples)
        end_sample = min(audio_length, end_sample + padding_samples)
        if end_sample > start_sample:
            padded.append((start_sample, end_sample))
    return padded


def _merge_segments(segments: Iterable[Tuple[int, int]], merge_gap_samples: int) -> List[Tuple[int, int]]:
    sorted_segments = sorted((start, end) for start, end in segments if end > start)
    if not sorted_segments:
        return []

    merged: List[Tuple[int, int]] = [sorted_segments[0]]
    for start_sample, end_sample in sorted_segments[1:]:
        previous_start, previous_end = merged[-1]
        if start_sample <= previous_end + merge_gap_samples:
            merged[-1] = (previous_start, max(previous_end, end_sample))
        else:
            merged.append((start_sample, end_sample))
    return merged
