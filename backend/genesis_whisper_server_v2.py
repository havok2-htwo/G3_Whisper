"""Versioned multimode audio-processing API."""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException, MultiPartParser

from .genesis_whisper_server_audio import get_audio_duration_seconds, load_audio_file
from .genesis_whisper_server_auth import authorize_api_key, get_auth_store, require_admin
from .genesis_whisper_server_batching import enqueue_audio_segments_bounded
from .genesis_whisper_server_chunking import combine_transcription_chunks, split_audio_for_whisper
from .genesis_whisper_server_dia_client import DiaClientError, diarize_v2
from .genesis_whisper_server_globals import (
    current_settings,
    get_effective_transcription_language,
    settings_lock,
    uses_cohere_backend,
)
from .genesis_whisper_server_history import append_history_entry, capture_history_audio
from .genesis_whisper_server_gpu import run_blocking_gpu_phase
from .genesis_whisper_server_local_asr_engine import get_last_local_asr_load_error, load_local_asr_model
from .genesis_whisper_server_repetition import (
    REPETITION_FILTER_HEADER,
    filter_repeated_patterns,
    repetition_filter_enabled,
)
from .genesis_whisper_server_speaker_matching import (
    SpeakerProfileValidationError,
    extract_speaker_clouds,
    match_known_speakers,
    validate_known_speakers,
)
from .genesis_whisper_server_speaker_audio import build_unknown_speaker_audio_assets
from .genesis_whisper_server_speaker_refinement import refine_speaker_turns
from .genesis_whisper_server_storage import log_transcription
from .genesis_whisper_server_turn_gate import finalize_segment_text, prefilter_turns
from .genesis_whisper_server_wxc import (
    align_chunk_words,
    assign_words_to_turns,
    build_superchunks,
    padded_chunk_audio,
    silero_frame_probs,
    speech_regions_from_probs,
    verify_sentences,
    words_to_sentences,
)
from .genesis_whisper_server_vid import embedding_model_metadata, generate_voice_vector


V2_SCHEMA_VERSION = "2.0"
V2_REQUEST_JSON_MAX_BYTES = 16 * 1024 * 1024
V2_MODES = {"embedding", "transcript", "transcript_embedding", "diarization"}
SPEAKER_REFINEMENT_MODES = {"off", "shadow", "conservative"}
MAX_EXPECTED_SPEAKERS = 64
MAX_TURN_CHUNK_SECONDS = 30.0
TURN_MERGE_GAP_MS = 250


@dataclass
class V2ApiError(RuntimeError):
    status_code: int
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class _BoundedV2MultipartParser(MultiPartParser):
    """Apply the metadata limit to text *and* file-style request parts.

    Starlette's ``max_part_size`` intentionally applies only to ordinary form
    fields.  A client may nevertheless send the JSON part with a filename, in
    which case the base parser would spool the complete value before our later
    length check.  Count ``request`` bytes in the streaming callback so at most
    16 MiB ever reaches either memory or the temporary-file spool.
    """

    def __init__(self, *args: Any, request_part_max_bytes: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_part_max_bytes = request_part_max_bytes
        self._request_part_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.field_name == "request":
            self._request_part_bytes += end - start
            if self._request_part_bytes > self._request_part_max_bytes:
                raise MultiPartException("Request part exceeded maximum size of 16384KB.")
        super().on_part_data(data, start, end)


def _processing_key() -> tuple[str, str, str, str, str]:
    with settings_lock:
        return (
            str(current_settings["local_model"]),
            str(current_settings["local_gpu_device"]),
            str(current_settings["local_model_cache_path"]),
            str(current_settings.get("transcription_language", "auto")),
            str(current_settings.get("local_model_precision", "fp16")),
        )


def _validate_object_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise V2ApiError(
            422,
            "INVALID_REQUEST",
            f"Unbekannte Felder in {path}: {', '.join(unknown)}",
            details={"field": path},
        )


def _parse_request_json(raw_value: str | None) -> dict[str, Any]:
    if raw_value is None:
        raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Feld 'request' fehlt.")
    if len(raw_value.encode("utf-8")) > V2_REQUEST_JSON_MAX_BYTES:
        raise V2ApiError(413, "REQUEST_METADATA_TOO_LARGE", "Request-JSON ist groesser als 16 MiB.")
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise V2ApiError(400, "INVALID_JSON", "Multipart-Feld 'request' enthaelt kein gueltiges JSON.") from exc
    if not isinstance(payload, dict):
        raise V2ApiError(422, "INVALID_REQUEST", "Request-JSON muss ein Objekt sein.")
    _validate_object_keys(payload, {"schema_version", "mode", "diarization"}, "request")
    if payload.get("schema_version") != V2_SCHEMA_VERSION:
        raise V2ApiError(422, "INVALID_REQUEST", "schema_version muss '2.0' sein.")
    mode = str(payload.get("mode") or "").strip()
    if mode not in V2_MODES:
        raise V2ApiError(422, "INVALID_REQUEST", f"Unbekannter Modus '{mode}'.")

    diarization = payload.get("diarization")
    if mode != "diarization":
        if diarization is not None:
            raise V2ApiError(422, "INVALID_REQUEST", "diarization ist nur im Modus 'diarization' erlaubt.")
        return {"schema_version": V2_SCHEMA_VERSION, "mode": mode}
    if diarization is None:
        diarization = {}
    if not isinstance(diarization, dict):
        raise V2ApiError(422, "INVALID_REQUEST", "diarization muss ein Objekt sein.")
    _validate_object_keys(
        diarization,
        {
            "expected_speakers",
            "known_speakers",
            "speaker_refinement",
            "unknown_speaker_audio",
        },
        "request.diarization",
    )

    expected = diarization.get("expected_speakers")
    if expected is not None:
        if isinstance(expected, bool) or not isinstance(expected, int) or not 1 <= expected <= MAX_EXPECTED_SPEAKERS:
            raise V2ApiError(
                422,
                "INVALID_REQUEST",
                f"expected_speakers muss zwischen 1 und {MAX_EXPECTED_SPEAKERS} liegen.",
            )
    known = diarization.get("known_speakers", [])
    if not isinstance(known, list) or len(known) > MAX_EXPECTED_SPEAKERS:
        raise V2ApiError(422, "INVALID_REQUEST", "known_speakers muss eine Liste mit maximal 64 Eintraegen sein.")
    for index, profile in enumerate(known):
        if not isinstance(profile, dict):
            raise V2ApiError(422, "INVALID_REQUEST", f"known_speakers[{index}] muss ein Objekt sein.")
        _validate_object_keys(profile, {"id", "embeddings"}, f"known_speakers[{index}]")
    try:
        validate_known_speakers(known)
    except SpeakerProfileValidationError as exc:
        raise V2ApiError(422, "INVALID_REQUEST", str(exc)) from exc
    if expected is not None and len(known) > expected:
        raise V2ApiError(
            422,
            "INVALID_REQUEST",
            "expected_speakers darf nicht kleiner als die Anzahl bekannter Sprecher sein.",
        )
    speaker_refinement = diarization.get("speaker_refinement", "off")
    if not isinstance(speaker_refinement, str) or speaker_refinement not in SPEAKER_REFINEMENT_MODES:
        raise V2ApiError(
            422,
            "INVALID_REQUEST",
            "speaker_refinement muss 'off', 'shadow' oder 'conservative' sein.",
        )
    unknown_speaker_audio = diarization.get("unknown_speaker_audio", False)
    if not isinstance(unknown_speaker_audio, bool):
        raise V2ApiError(
            422,
            "INVALID_REQUEST",
            "unknown_speaker_audio muss ein boolescher Wert sein.",
        )
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "mode": mode,
        "diarization": {
            "expected_speakers": expected,
            "known_speakers": known,
            "speaker_refinement": speaker_refinement,
            "unknown_speaker_audio": unknown_speaker_audio,
        },
    }


async def _parse_multipart_parts(http_request: Request) -> tuple[FormData, StarletteUploadFile, str]:
    content_type = http_request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise V2ApiError(400, "INVALID_MULTIPART", "Content-Type muss multipart/form-data sein.")
    parser_options: dict[str, Any] = {
        "headers": http_request.headers,
        "stream": http_request.stream(),
        "max_files": 2,
        "max_fields": 2,
        "request_part_max_bytes": V2_REQUEST_JSON_MAX_BYTES,
    }
    # Starlette added ``max_part_size`` after the minimum version supported by
    # this service.  Our streaming callback independently enforces the same
    # 16 MiB bound for the JSON part, so omitting the upstream convenience
    # option on older installations does not weaken the request limit.
    if "max_part_size" in inspect.signature(MultiPartParser.__init__).parameters:
        parser_options["max_part_size"] = V2_REQUEST_JSON_MAX_BYTES
    parser = _BoundedV2MultipartParser(**parser_options)
    try:
        form = await parser.parse()
    except (MultiPartException, StarletteHTTPException) as exc:
        message = str(getattr(exc, "detail", exc)).lower()
        if "maximum size" in message or "too large" in message:
            raise V2ApiError(
                413,
                "REQUEST_METADATA_TOO_LARGE",
                "Request-JSON ist groesser als 16 MiB.",
            ) from exc
        raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Daten sind ungueltig.") from exc
    except Exception as exc:
        for pending_file in getattr(parser, "_files_to_close_on_error", ()):
            pending_file.close()
        raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Daten konnten nicht gelesen werden.") from exc

    try:
        if set(form.keys()) - {"file", "request"}:
            raise V2ApiError(422, "INVALID_REQUEST", "Unbekannte Multipart-Felder sind nicht erlaubt.")
        if len(form.getlist("file")) != 1 or len(form.getlist("request")) != 1:
            raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Felder 'file' und 'request' werden genau einmal erwartet.")

        audio_part = form.get("file")
        request_part = form.get("request")
        if not isinstance(audio_part, StarletteUploadFile):
            raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Feld 'file' fehlt oder ist keine Datei.")

        if isinstance(request_part, str):
            raw_request = request_part
        elif isinstance(request_part, StarletteUploadFile):
            metadata = await request_part.read(V2_REQUEST_JSON_MAX_BYTES + 1)
            if len(metadata) > V2_REQUEST_JSON_MAX_BYTES:
                raise V2ApiError(413, "REQUEST_METADATA_TOO_LARGE", "Request-JSON ist groesser als 16 MiB.")
            try:
                raw_request = metadata.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise V2ApiError(400, "INVALID_JSON", "Multipart-Feld 'request' ist nicht UTF-8-kodiert.") from exc
        else:
            raise V2ApiError(400, "INVALID_MULTIPART", "Multipart-Feld 'request' fehlt.")

        if len(raw_request.encode("utf-8")) > V2_REQUEST_JSON_MAX_BYTES:
            raise V2ApiError(413, "REQUEST_METADATA_TOO_LARGE", "Request-JSON ist groesser als 16 MiB.")
        return form, audio_part, raw_request
    except Exception:
        await form.close()
        raise


def _error_response(
    request_id: str,
    error: V2ApiError,
    *,
    mode: str | None = None,
    request_started: float | None = None,
) -> JSONResponse:
    error_body: dict[str, Any] = {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
    }
    if error.details:
        error_body["details"] = error.details
    return JSONResponse(
        status_code=error.status_code,
        headers={"X-Request-ID": request_id},
        content={
            "schema_version": V2_SCHEMA_VERSION,
            "request_id": request_id,
            "status": "failed",
            "mode": mode,
            "models": {},
            "timings_ms": (
                {"total": round((time.monotonic() - request_started) * 1000)}
                if request_started is not None
                else {}
            ),
            "result": None,
            "warnings": [],
            "error": error_body,
        },
    )


def _success_response(request_id: str, payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        headers={"X-Request-ID": request_id},
        content={"schema_version": V2_SCHEMA_VERSION, "request_id": request_id, **payload},
    )


def _authorize_v2_request(request: Request) -> str | None:
    """Accept the public API key or an authenticated admin UI session.

    Client API keys remain the normal machine-to-machine credential.  The
    same-origin admin dashboard already has a protected, HTTP-only session and
    must not require operators to copy a client key back into their browser
    merely to exercise the v2 test panel.
    """

    # An explicitly supplied key is always authoritative.  This keeps the
    # admin tester useful for validating client credentials instead of
    # silently accepting a bad key through the browser session fallback.
    if request.headers.get("x-api-key") is not None:
        return authorize_api_key(request)

    try:
        return authorize_api_key(request)
    except HTTPException as api_key_error:
        try:
            require_admin(request)
        except HTTPException:
            raise api_key_error
        return None


def _ensure_asr_model(processing_key: tuple[str, str, str, str, str]) -> None:
    model_id, device, cache_path, _, precision = processing_key
    if load_local_asr_model(model_id, device, cache_path, precision):
        return
    detail = "Lokales ASR-Modell konnte nicht geladen werden."
    load_error = get_last_local_asr_load_error()
    if load_error:
        detail = f"{detail} Ursache: {load_error}"
    raise RuntimeError(detail)


async def _transcribe_audio(request: Request, audio: np.ndarray) -> tuple[str, int, int, str]:
    processing_key = _processing_key()
    model_id = processing_key[0]
    batch_manager = request.app.state.whisper_batch_manager
    request_token = uuid.uuid4().hex
    started = time.monotonic()
    if uses_cohere_backend(model_id):
        segments = [audio]
    else:
        segments = await asyncio.to_thread(split_audio_for_whisper, audio)
    if not segments:
        return "", 0, 0, model_id
    results = await enqueue_audio_segments_bounded(
        batch_manager,
        segments,
        request_token,
        processing_key,
    )
    text = combine_transcription_chunks([result.text for result in results])
    return text, round((time.monotonic() - started) * 1000), len(segments), model_id


def _normalize_dia_segments(items: Sequence[Mapping[str, Any]], duration_ms: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        speaker_id = str(item.get("speaker_id") or "").strip()
        try:
            start_ms = max(0, int(item.get("start_ms", 0)))
            end_ms = min(duration_ms, int(item.get("end_ms", 0)))
        except (TypeError, ValueError):
            continue
        if speaker_id and end_ms > start_ms:
            normalized.append({"start_ms": start_ms, "end_ms": end_ms, "speaker_id": speaker_id})
    normalized.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["speaker_id"]))
    return normalized


def _merge_exclusive_turns(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        current = dict(item)
        if (
            merged
            and merged[-1]["speaker_id"] == current["speaker_id"]
            and str(merged[-1].get("original_speaker_id", merged[-1]["speaker_id"]))
            == str(current.get("original_speaker_id", current["speaker_id"]))
            and current["start_ms"] <= merged[-1]["end_ms"] + TURN_MERGE_GAP_MS
        ):
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], current["end_ms"])
        else:
            merged.append(current)
    return merged


def _turn_audio_chunks(audio: np.ndarray, start_ms: int, end_ms: int) -> list[np.ndarray]:
    sample_rate = 16000
    start_sample = max(0, round(start_ms * sample_rate / 1000))
    end_sample = min(len(audio), round(end_ms * sample_rate / 1000))
    turn_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    max_samples = round(MAX_TURN_CHUNK_SECONDS * sample_rate)
    return [turn_audio[offset : offset + max_samples] for offset in range(0, len(turn_audio), max_samples) if len(turn_audio[offset : offset + max_samples])]


def _segment_has_overlap(segment: Mapping[str, Any], overlaps: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        int(overlap.get("end_ms", 0)) > int(segment["start_ms"])
        and int(overlap.get("start_ms", 0)) < int(segment["end_ms"])
        for overlap in overlaps
    )


async def _transcribe_turns(
    request: Request,
    audio: np.ndarray,
    turns: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    apply_repetition_filter: bool,
    gate_enabled: bool = True,
) -> tuple[list[dict[str, Any]], int, str]:
    processing_key = _processing_key()
    model_id = processing_key[0]
    turn_chunk_counts: list[int] = []
    prepared_chunks: list[np.ndarray] = []
    for turn in turns:
        chunks = _turn_audio_chunks(audio, int(turn["start_ms"]), int(turn["end_ms"]))
        turn_chunk_counts.append(len(chunks))
        prepared_chunks.extend(chunks)
    started = time.monotonic()
    request_token = uuid.uuid4().hex
    batch_manager = request.app.state.whisper_batch_manager
    batch_results = await enqueue_audio_segments_bounded(
        batch_manager,
        prepared_chunks,
        request_token,
        processing_key,
    )
    result_index = 0
    transcript_segments: list[dict[str, Any]] = []
    for turn_index, (turn, chunk_count) in enumerate(zip(turns, turn_chunk_counts)):
        chunk_texts = [batch_results[result_index + offset].text for offset in range(chunk_count)]
        result_index += chunk_count
        text = combine_transcription_chunks(chunk_texts)
        if apply_repetition_filter:
            text = filter_repeated_patterns(text)
        disposition = None
        if gate_enabled:
            duration_ms = int(turn["end_ms"]) - int(turn["start_ms"])
            text, disposition = finalize_segment_text(text, duration_ms)
            if disposition == "drop":
                continue
        if not text:
            continue
        segment = {
            "index": turn_index,
            "start_ms": int(turn["start_ms"]),
            "end_ms": int(turn["end_ms"]),
            "diarization_speaker_id": str(
                turn.get("original_speaker_id", turn["speaker_id"])
            ),
            **(
                {"refined_diarization_speaker_id": str(turn["speaker_id"])}
                if "original_speaker_id" in turn
                else {}
            ),
            "text": text,
            "overlap": _segment_has_overlap(turn, overlaps),
        }
        if disposition == "asr_failure":
            segment["asr_failure"] = True
        transcript_segments.append(segment)
    return transcript_segments, round((time.monotonic() - started) * 1000), model_id


async def _wxc_transcribe_segments(
    request: Request,
    audio: np.ndarray,
    gated_turns: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    apply_repetition_filter: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any], str]:
    """Transcribe-first WXC path: large Silero chunks -> ASR -> word alignment ->
    gated speaker skeleton -> sentence-level 192D verification.

    Raises WxcNoSpeechError when Silero finds no usable speech so the caller can
    fall back to the legacy per-turn path instead of returning silence.
    """

    processing_key = _processing_key()
    timings: dict[str, int] = {}

    vad_started = time.monotonic()
    probs = await asyncio.to_thread(silero_frame_probs, audio)
    regions = speech_regions_from_probs(probs)
    timings["vad"] = round((time.monotonic() - vad_started) * 1000)
    if not regions:
        raise WxcNoSpeechError("Silero fand keine Sprachregionen.")

    chunks = build_superchunks(regions)
    chunk_audio = padded_chunk_audio(audio, chunks)

    asr_started = time.monotonic()
    batch_manager = request.app.state.whisper_batch_manager
    results = await enqueue_audio_segments_bounded(
        batch_manager,
        chunk_audio,
        uuid.uuid4().hex,
        processing_key,
    )
    texts = [result.text for result in results]
    timings["transcription"] = round((time.monotonic() - asr_started) * 1000)

    align_started = time.monotonic()
    async with request.app.state.local_gpu_lock:
        words = await run_blocking_gpu_phase(align_chunk_words, audio, chunks, texts)
    timings["alignment"] = round((time.monotonic() - align_started) * 1000)

    assign_words_to_turns(words, gated_turns)
    sentences = words_to_sentences(words, overlaps)

    verify_started = time.monotonic()
    async with request.app.state.local_gpu_lock:
        verification = await run_blocking_gpu_phase(verify_sentences, audio, sentences)
    timings["verification"] = round((time.monotonic() - verify_started) * 1000)

    transcript_segments: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        text = sentence["text"].strip()
        if apply_repetition_filter:
            text = filter_repeated_patterns(text)
        if not text:
            continue
        segment = {
            "index": index,
            "start_ms": round(sentence["t0"] * 1000),
            "end_ms": round(sentence["t1"] * 1000),
            "diarization_speaker_id": str(sentence["speaker"]),
            "text": text,
            "overlap": bool(sentence["overlap"]),
        }
        if "verified_from" in sentence:
            segment["verified_from"] = str(sentence["verified_from"])
        transcript_segments.append(segment)

    speech_seconds = sum(b - a for a, b in regions)
    diagnostics = {
        "speech_regions": len(regions),
        "speech_seconds": round(speech_seconds, 1),
        "superchunks": len(chunks),
        "words_aligned": len(words),
        "verification": verification,
    }
    return transcript_segments, timings, diagnostics, processing_key[0]


class WxcNoSpeechError(RuntimeError):
    """Raised when the WXC path cannot find speech and the caller should fall back."""


async def _generate_embedding(request: Request, audio: np.ndarray) -> tuple[list[float], int]:
    started = time.monotonic()
    async with request.app.state.local_gpu_lock:
        vector = await run_blocking_gpu_phase(generate_voice_vector, audio)
    return vector.astype(float).tolist(), round((time.monotonic() - started) * 1000)


def _record_log(
    request: Request,
    history_id: str,
    mode: str,
    model_id: str | None,
    total_duration_ms: int,
    transcription_duration_ms: int | None,
    embedding_duration_ms: int | None,
    transcript: str,
    *,
    retry_of: str | None = None,
    existing_blob_id: str | None = None,
) -> None:
    entry = {
        "history_id": history_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": request.client.host if request.client else "unknown",
        "engine": "v2",
        "mode": mode,
        "model_id": model_id,
        "total_duration_ms": total_duration_ms,
        "transcription_duration_ms": transcription_duration_ms,
        "voice_vector_duration_ms": embedding_duration_ms,
        "transcript": transcript,
        "retry_of": retry_of,
        "retry_mode": mode if mode in {"transcript", "transcript_embedding"} else None,
    }
    append_history_entry(entry, existing_blob_id=existing_blob_id)
    log_transcription(entry)


async def _process_diarization(
    http_request: Request,
    file: UploadFile,
    audio: np.ndarray,
    configuration: Mapping[str, Any],
    apply_repetition_filter: bool,
    gate_enabled: bool = True,
    pipeline_mode: str = "wxc",
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], list[dict[str, Any]], str]:
    expected_speakers = configuration.get("expected_speakers")
    known_speakers = list(configuration.get("known_speakers") or [])
    speaker_refinement_mode = str(configuration.get("speaker_refinement") or "off")
    include_unknown_speaker_audio = bool(configuration.get("unknown_speaker_audio", False))
    dia_started = time.monotonic()
    dia_response = await diarize_v2(
        file.file,
        file.filename or "upload",
        file.content_type,
        num_speakers=expected_speakers,
        min_speakers=len(known_speakers) if expected_speakers is None and known_speakers else None,
    )
    dia_duration_ms = round((time.monotonic() - dia_started) * 1000)
    duration_ms = round(get_audio_duration_seconds(audio) * 1000)
    standard = _normalize_dia_segments(dia_response.get("diarization", []), duration_ms)
    exclusive = _normalize_dia_segments(dia_response.get("exclusive_diarization", []), duration_ms)
    overlaps = [
        {
            "start_ms": max(0, int(item.get("start_ms", 0))),
            "end_ms": min(duration_ms, int(item.get("end_ms", 0))),
            "speaker_ids": [str(value) for value in item.get("speaker_ids", [])],
        }
        for item in dia_response.get("overlaps", [])
        if int(item.get("end_ms", 0)) > int(item.get("start_ms", 0))
    ]

    embedding_started = time.monotonic()
    async with http_request.app.state.local_gpu_lock:
        # Standard diarization preserves all speaker activity and therefore
        # defines the safe, overlap-free enrollment regions. Exclusive turns
        # remain reserved for ASR so overlapping speech is transcribed once.
        speaker_clouds = await run_blocking_gpu_phase(extract_speaker_clouds, audio, standard, overlaps)
    embedding_ms = round((time.monotonic() - embedding_started) * 1000)

    refinement_diagnostics: dict[str, Any] | None = None
    effective_exclusive = exclusive
    if speaker_refinement_mode != "off":
        refinement = await asyncio.to_thread(
            refine_speaker_turns,
            speaker_refinement_mode,
            exclusive,
            standard,
            overlaps,
            speaker_clouds,
        )
        effective_exclusive = refinement.turns
        speaker_clouds = refinement.speaker_clouds
        refinement_diagnostics = refinement.diagnostics

    if gate_enabled:
        turns, gate_diagnostics = prefilter_turns(effective_exclusive)
    else:
        turns = _merge_exclusive_turns(effective_exclusive)
        gate_diagnostics = {"enabled": False}

    wxc_diagnostics: dict[str, Any] | None = None
    wxc_fallback_warning: dict[str, Any] | None = None
    extra_timings: dict[str, int] = {}
    if pipeline_mode == "wxc":
        try:
            transcript_segments, extra_timings, wxc_diagnostics, asr_model_id = (
                await _wxc_transcribe_segments(
                    http_request,
                    audio,
                    turns,
                    overlaps,
                    apply_repetition_filter,
                )
            )
            transcription_ms = extra_timings.get("transcription", 0)
        except WxcNoSpeechError:
            wxc_fallback_warning = {"code": "WXC_NO_SPEECH_FALLBACK"}
        except Exception as exc:
            # The legacy per-turn path is the safety net: a WXC failure must
            # degrade the result quality, never the request.
            print(f"[WARNUNG-WXC] Pipeline fehlgeschlagen, Fallback auf Turn-Pfad: {exc}", file=sys.stderr)
            wxc_fallback_warning = {"code": "WXC_FALLBACK", "message": str(exc)[:300]}
    if pipeline_mode != "wxc" or wxc_fallback_warning is not None:
        transcript_segments, transcription_ms, asr_model_id = await _transcribe_turns(
            http_request,
            audio,
            turns,
            overlaps,
            apply_repetition_filter,
            gate_enabled,
        )

    try:
        assignments, unresolved_profile_ids, unresolved_clusters = match_known_speakers(
            known_speakers,
            speaker_clouds,
        )
    except SpeakerProfileValidationError as exc:
        raise V2ApiError(422, "INVALID_REQUEST", str(exc)) from exc

    detected_speakers = sorted(speaker_clouds)
    known_ids = {str(profile["id"]) for profile in known_speakers}
    reserved_public_ids = set(known_ids)
    unmatched_public_ids: dict[str, str] = {}
    all_unmatched_labels = sorted(
        set(detected_speakers)
        | {
            str(
                segment.get(
                    "refined_diarization_speaker_id",
                    segment["diarization_speaker_id"],
                )
            )
            for segment in transcript_segments
        }
    )
    for dia_speaker_id in all_unmatched_labels:
        if dia_speaker_id in assignments:
            continue
        candidate = dia_speaker_id
        if candidate in reserved_public_ids:
            base = f"unknown-{dia_speaker_id}"
            candidate = base
            suffix = 1
            while candidate in reserved_public_ids:
                candidate = f"{base}-{suffix}"
                suffix += 1
        unmatched_public_ids[dia_speaker_id] = candidate
        reserved_public_ids.add(candidate)

    for segment in transcript_segments:
        dia_speaker_id = str(
            segment.get(
                "refined_diarization_speaker_id",
                segment["diarization_speaker_id"],
            )
        )
        assignment = assignments.get(dia_speaker_id)
        if assignment:
            segment["speaker_id"] = assignment["speaker_id"]
            segment["speaker_kind"] = "known"
        elif dia_speaker_id in unresolved_clusters:
            segment["speaker_id"] = unmatched_public_ids[dia_speaker_id]
            segment["speaker_kind"] = "unresolved"
        else:
            segment["speaker_id"] = unmatched_public_ids[dia_speaker_id]
            segment["speaker_kind"] = "unknown"

    transcript_text = " ".join(segment["text"] for segment in transcript_segments).strip()
    speaker_assignments: list[dict[str, Any]] = []
    unknown_speakers: list[dict[str, Any]] = []
    unresolved_speakers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if wxc_fallback_warning is not None:
        warnings.append(wxc_fallback_warning)
    unknown_audio_assets: dict[str, dict[str, Any]] = {}
    unknown_audio_ms: int | None = None
    unidentified_dia_ids = sorted(
        speaker_id for speaker_id in detected_speakers if speaker_id not in assignments
    )
    if include_unknown_speaker_audio and unidentified_dia_ids:
        unknown_audio_started = time.monotonic()
        try:
            unknown_audio_assets = await asyncio.to_thread(
                build_unknown_speaker_audio_assets,
                audio,
                speaker_clouds,
                unidentified_dia_ids,
                effective_exclusive,
            )
        except (RuntimeError, ValueError):
            warnings.append({"code": "UNKNOWN_SPEAKER_AUDIO_ENCODING_FAILED"})
        unknown_audio_ms = round((time.monotonic() - unknown_audio_started) * 1000)

    for dia_speaker_id in detected_speakers:
        cloud = speaker_clouds[dia_speaker_id]
        assignment = assignments.get(dia_speaker_id)
        if assignment:
            speaker_assignments.append(
                {
                    "diarization_speaker_id": dia_speaker_id,
                    "speaker_id": assignment["speaker_id"],
                    "kind": "known",
                    "embedding_status": cloud.status,
                    **{key: value for key, value in assignment.items() if key != "speaker_id"},
                }
            )
            continue
        unresolved_match = unresolved_clusters.get(dia_speaker_id)
        speaker_kind = "unresolved" if unresolved_match else "unknown"
        speaker_assignments.append(
            {
                "diarization_speaker_id": dia_speaker_id,
                "speaker_id": unmatched_public_ids[dia_speaker_id],
                "kind": speaker_kind,
                "embedding_status": cloud.status,
                "cosine_similarity": None,
                **(unresolved_match or {}),
            }
        )
        public_embeddings = cloud.public_embeddings()
        public_speaker = {
            "speaker_id": unmatched_public_ids[dia_speaker_id],
            "diarization_speaker_id": dia_speaker_id,
            "speaker_kind": speaker_kind,
            "embedding_status": cloud.status,
            "embeddings": public_embeddings,
            "embeddings_truncated": len(cloud.inliers) > max(0, len(public_embeddings) - 1),
            "candidate_count": cloud.candidate_count,
            "retained_count": len(cloud.inlier_indices),
            "discarded_outliers": cloud.discarded_outliers,
            "purity": round(cloud.purity, 6) if cloud.purity is not None else None,
            **(unresolved_match or {}),
        }
        if dia_speaker_id in unknown_audio_assets:
            public_speaker["audio"] = unknown_audio_assets[dia_speaker_id]
        if unresolved_match:
            unresolved_speakers.append(public_speaker)
        else:
            unknown_speakers.append(public_speaker)
        if cloud.status != "ready":
            warnings.append(
                {
                    "code": "SPEAKER_EMBEDDING_QUALITY",
                    "speaker_id": dia_speaker_id,
                    "status": cloud.status,
                }
            )

    if include_unknown_speaker_audio and unidentified_dia_ids and not any(
        warning.get("code") == "UNKNOWN_SPEAKER_AUDIO_ENCODING_FAILED" for warning in warnings
    ):
        unavailable_audio_ids = [
            unmatched_public_ids[speaker_id]
            for speaker_id in unidentified_dia_ids
            if speaker_id not in unknown_audio_assets
        ]
        if unavailable_audio_ids:
            warnings.append(
                {
                    "code": "UNKNOWN_SPEAKER_AUDIO_UNAVAILABLE",
                    "speaker_ids": unavailable_audio_ids,
                }
            )

    if unresolved_profile_ids:
        warnings.append(
            {
                "code": "KNOWN_SPEAKERS_UNRESOLVED",
                "speaker_ids": unresolved_profile_ids,
            }
        )
    if expected_speakers is not None and len(detected_speakers) != expected_speakers:
        warnings.append(
            {
                "code": "EXPECTED_SPEAKER_COUNT_MISMATCH",
                "expected": expected_speakers,
                "detected": len(detected_speakers),
            }
        )

    result = {
        "transcript": {"text": transcript_text, "segments": transcript_segments},
        "speaker_counts": {
            "expected": expected_speakers,
            "detected": len(detected_speakers),
            "known_provided": len(known_speakers),
            "known_assigned": len(assignments),
            "unknown": len(unknown_speakers),
            "unresolved": len(unresolved_speakers),
        },
        "speaker_assignments": speaker_assignments,
        "unknown_speakers": unknown_speakers,
        "unresolved_speakers": unresolved_speakers,
        "unresolved_known_speakers": unresolved_profile_ids,
    }
    if refinement_diagnostics is not None:
        result["speaker_refinement"] = refinement_diagnostics
    result["turn_gate"] = gate_diagnostics
    if wxc_diagnostics is not None:
        result["speaker_verification"] = wxc_diagnostics.pop("verification", None)
        result["chunking"] = wxc_diagnostics
    timings = {
        "diarization": dia_duration_ms,
        "transcription": transcription_ms,
        "embedding": embedding_ms,
        **extra_timings,
    }
    if refinement_diagnostics is not None:
        timings["speaker_refinement"] = int(refinement_diagnostics["processing_ms"])
    if unknown_audio_ms is not None:
        timings["unknown_speaker_audio"] = unknown_audio_ms
    models = {
        "asr": {"id": asr_model_id},
        "diarization": dia_response.get("model") or {},
        "embedding": embedding_model_metadata(),
    }
    return result, timings, models, warnings, transcript_text


def create_v2_api(app: FastAPI) -> FastAPI:
    @app.post(
        "/v2/audio/process",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file", "request"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "request": {
                                    "type": "string",
                                    "description": "UTF-8 JSON object with schema_version 2.0 and mode (max. 16 MiB).",
                                },
                            },
                        },
                        "encoding": {"request": {"contentType": "application/json"}},
                    }
                },
            }
        },
    )
    async def process_audio_v2(http_request: Request):
        request_id = uuid.uuid4().hex
        request_started = time.monotonic()
        form_data: FormData | None = None
        mode: str | None = None
        try:
            try:
                api_key_id = _authorize_v2_request(http_request)
            except HTTPException as exc:
                raise V2ApiError(exc.status_code, "INVALID_API_KEY", str(exc.detail), False) from exc
            form_data, file, request_json = await _parse_multipart_parts(http_request)
            parsed = _parse_request_json(request_json)
            mode = parsed["mode"]
            filename = file.filename or "upload"
            decode_started = time.monotonic()
            try:
                audio = await asyncio.to_thread(load_audio_file, file.file, filename)
            except HTTPException as exc:
                raise V2ApiError(400, "AUDIO_DECODE_FAILED", str(exc.detail), False) from exc
            except Exception as exc:
                raise V2ApiError(400, "AUDIO_DECODE_FAILED", f"Audiodatei konnte nicht dekodiert werden: {exc}") from exc
            decode_ms = round((time.monotonic() - decode_started) * 1000)
            audio_duration_ms = round(get_audio_duration_seconds(audio) * 1000)
            filter_enabled = repetition_filter_enabled(http_request.headers.get(REPETITION_FILTER_HEADER))
            gate_enabled = http_request.headers.get("x-g3-turn-gate", "").strip().lower() not in ("off", "0", "false", "no")
            pipeline_mode = (
                "turns"
                if http_request.headers.get("x-g3-pipeline", "").strip().lower() in ("turns", "legacy", "off")
                else "wxc"
            )
            timings: dict[str, int] = {"decode": decode_ms}
            warnings: list[dict[str, Any]] = []
            result: dict[str, Any]
            models: dict[str, Any] = {}
            transcript_text = ""
            asr_model_id: str | None = None
            embedding_ms: int | None = None
            transcription_ms: int | None = None

            if mode == "embedding":
                vector, embedding_ms = await _generate_embedding(http_request, audio)
                timings["embedding"] = embedding_ms
                models["embedding"] = embedding_model_metadata()
                result = {"embedding": {"vector": vector}}
            elif mode == "transcript":
                transcript_text, transcription_ms, _, asr_model_id = await _transcribe_audio(http_request, audio)
                if filter_enabled:
                    transcript_text = filter_repeated_patterns(transcript_text)
                timings["transcription"] = transcription_ms
                models["asr"] = {"id": asr_model_id}
                result = {"transcript": {"text": transcript_text}}
            elif mode == "transcript_embedding":
                transcript_text, transcription_ms, _, asr_model_id = await _transcribe_audio(http_request, audio)
                if filter_enabled:
                    transcript_text = filter_repeated_patterns(transcript_text)
                embedding_started = time.monotonic()
                try:
                    vector, embedding_ms = await _generate_embedding(http_request, audio)
                    embedding_result: dict[str, Any] | None = {"vector": vector}
                except ValueError as exc:
                    # A transcript remains useful even when a short/quiet live
                    # microphone window cannot support reliable speaker
                    # identity.  Keep the combined mode successful-but-partial
                    # instead of discarding the ASR result with HTTP 422.
                    embedding_ms = round((time.monotonic() - embedding_started) * 1000)
                    embedding_result = None
                    warnings.append(
                        {
                            "code": "VOICE_EMBEDDING_UNAVAILABLE",
                            "message": str(exc),
                        }
                    )
                timings.update({"transcription": transcription_ms, "embedding": embedding_ms})
                models.update({"asr": {"id": asr_model_id}, "embedding": embedding_model_metadata()})
                result = {"transcript": {"text": transcript_text}, "embedding": embedding_result}
            else:
                result, mode_timings, models, warnings, transcript_text = await _process_diarization(
                    http_request,
                    file,
                    audio,
                    parsed["diarization"],
                    filter_enabled,
                    gate_enabled,
                    pipeline_mode,
                )
                timings.update(mode_timings)
                transcription_ms = mode_timings.get("transcription")
                embedding_ms = mode_timings.get("embedding")
                asr_model_id = (models.get("asr") or {}).get("id")

            total_ms = round((time.monotonic() - request_started) * 1000)
            timings["total"] = total_ms
            if api_key_id:
                get_auth_store().record_api_key_usage(api_key_id, get_audio_duration_seconds(audio))
            _record_log(
                http_request,
                request_id,
                mode,
                asr_model_id,
                total_ms,
                transcription_ms,
                embedding_ms,
                transcript_text,
            )
            with settings_lock:
                retain_history_audio = current_settings.get("debug_retain_history_audio", False) is True
            if retain_history_audio:
                try:
                    await asyncio.to_thread(
                        capture_history_audio,
                        request_id,
                        file.file,
                        filename,
                        file.content_type,
                    )
                except Exception as exc:
                    # Retention is diagnostic-only and must never invalidate a
                    # successful production request.
                    print(f"[V2-WARNUNG] Debug-Audio konnte nicht gespeichert werden: {exc}", file=sys.stderr)
            return _success_response(
                request_id,
                {
                    "status": "partial" if warnings else "completed",
                    "mode": mode,
                    "audio": {"duration_ms": audio_duration_ms},
                    "models": models,
                    "timings_ms": timings,
                    "result": result,
                    "warnings": warnings,
                },
            )
        except V2ApiError as exc:
            return _error_response(request_id, exc, mode=mode, request_started=request_started)
        except DiaClientError as exc:
            return _error_response(
                request_id,
                V2ApiError(exc.status_code, exc.code, exc.message, exc.retryable),
                mode=mode,
                request_started=request_started,
            )
        except SpeakerProfileValidationError as exc:
            return _error_response(
                request_id,
                V2ApiError(422, "INVALID_REQUEST", str(exc)),
                mode=mode,
                request_started=request_started,
            )
        except ValueError as exc:
            return _error_response(
                request_id,
                V2ApiError(422, "INSUFFICIENT_SPEECH", str(exc)),
                mode=mode,
                request_started=request_started,
            )
        except RuntimeError as exc:
            return _error_response(
                request_id,
                V2ApiError(503, "MODEL_UNAVAILABLE", str(exc), True),
                mode=mode,
                request_started=request_started,
            )
        except Exception as exc:
            return _error_response(
                request_id,
                V2ApiError(500, "INTERNAL_ERROR", f"Audio-Verarbeitung fehlgeschlagen: {exc}", False),
                mode=mode,
                request_started=request_started,
            )
        finally:
            if form_data is not None:
                await form_data.close()

    return app


__all__ = ["create_v2_api"]
