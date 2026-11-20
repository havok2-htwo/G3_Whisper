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
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException, MultiPartParser

from .genesis_whisper_server_audio import get_audio_duration_seconds, load_audio_file
from .genesis_whisper_server_auth import authorize_api_key, get_auth_store, require_admin
from .genesis_whisper_server_batching import served_model_id
from .genesis_whisper_server_chunking import combine_transcription_chunks, split_audio_for_whisper
from .genesis_whisper_server_dia_client import DiaClientError, dia_is_configured, diarize_v2
from .genesis_whisper_server_globals import (
    SUPPORTED_LANGUAGE_OPTIONS,
    SUPPORTED_LANGUAGE_VALUES,
    current_settings,
    get_effective_transcription_language,
    get_local_model_backend,
    get_local_model_label,
    is_local_model_loaded,
    language_detection_applies,
    language_detection_fallback,
    settings_lock,
    uses_parakeet_backend,
    uses_whole_audio_backend,
)
from .genesis_whisper_server_history import append_history_entry, capture_history_audio
from .genesis_whisper_server_keywords import (
    Glossary,
    KeywordValidationError,
    capabilities_block,
    keywords_block,
    parse_request_keywords,
    resolve_glossary,
    sanitize_keyword_setting,
    truncation_warnings,
    with_keywords,
)
from .genesis_whisper_server_lid import (
    LID_MAX_POOL_SECONDS,
    LID_MIN_SPEECH_SECONDS,
    LID_TURN_MIN_SECONDS,
    SOURCE_DETECTED,
    SOURCE_FALLBACK,
    LanguageDecision,
    configured_lid_model,
    decide_item_languages,
    decide_transcript_items,
    detect_windows,
    dominant_language,
    lid_status,
    log_decision,
    pool_results,
    resolve_speaker_languages,
    split_windows,
    submit_by_language,
    turn_language,
    unavailable_warning,
)
from .genesis_whisper_server_gpu import run_blocking_gpu_phase, run_sliced_gpu_phase
from .genesis_whisper_server_local_asr_engine import get_last_local_asr_load_error, load_local_asr_model
from .genesis_whisper_server_repetition import (
    REPETITION_FILTER_HEADER,
    filter_repeated_patterns,
    repetition_filter_enabled,
)
from .genesis_whisper_server_speaker_matching import (
    SpeakerProfileValidationError,
    build_speaker_clouds,
    match_known_speakers,
    plan_speaker_windows,
    validate_known_speakers,
)
from .genesis_whisper_server_speaker_audio import build_unknown_speaker_audio_assets
from .genesis_whisper_server_speaker_refinement import refine_speaker_turns
from .genesis_whisper_server_storage import log_transcription, normalize_settings
from .genesis_whisper_server_turn_gate import finalize_segment_text, prefilter_turns
from .genesis_whisper_server_wxc import (
    align_chunk_slice,
    alignment_bounds,
    assign_words_to_turns,
    build_language_superchunks,
    build_superchunks,
    finish_verification,
    padded_chunk_audio,
    plan_verification_windows,
    silero_frame_probs,
    speech_regions_from_probs,
    split_audio_into_superchunks,
    words_to_sentences,
)
from .genesis_whisper_server_vid import EmbeddedVoiceWindow, VoiceWindow, embed_voice_windows, embedding_model_metadata, generate_voice_vector


V2_SCHEMA_VERSION = "2.0"
V2_REQUEST_JSON_MAX_BYTES = 16 * 1024 * 1024
V2_MODES = ("embedding", "transcript", "transcript_embedding", "diarization")
SPEAKER_REFINEMENT_MODES = {"off", "shadow", "conservative"}
MAX_EXPECTED_SPEAKERS = 64
MAX_TURN_CHUNK_SECONDS = 30.0
TURN_MERGE_GAP_MS = 250
# segment "language_source" in diarization: the segment ran in its speaker's language, or in its own turn's.
LANGUAGE_SOURCE_SPEAKER = "speaker"
LANGUAGE_SOURCE_TURN = "turn"


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
    _validate_object_keys(payload, {"schema_version", "mode", "diarization", "language", "keywords"}, "request")
    if payload.get("schema_version") != V2_SCHEMA_VERSION:
        raise V2ApiError(422, "INVALID_REQUEST", "schema_version muss '2.0' sein.")
    mode = str(payload.get("mode") or "").strip()
    if mode not in V2_MODES:
        raise V2ApiError(422, "INVALID_REQUEST", f"Unbekannter Modus '{mode}'.")

    language_value = payload.get("language")
    normalized_language: str | None = None
    if language_value is not None:
        if not isinstance(language_value, str) or not language_value.strip():
            raise V2ApiError(422, "INVALID_REQUEST", "language muss ein nicht-leerer String sein.")
        if mode == "embedding":
            raise V2ApiError(422, "INVALID_REQUEST", "language ist im Modus 'embedding' nicht erlaubt.")
        normalized_language = language_value.strip().lower()
        if normalized_language not in SUPPORTED_LANGUAGE_VALUES:
            raise V2ApiError(
                422,
                "INVALID_REQUEST",
                f"language '{normalized_language}' wird nicht unterstuetzt.",
            )

    try:
        keywords = parse_request_keywords(payload.get("keywords"))
    except KeywordValidationError as exc:
        raise V2ApiError(422, "INVALID_REQUEST", str(exc), details={"field": "request.keywords"}) from exc
    if keywords is not None and mode == "embedding":
        raise V2ApiError(422, "INVALID_REQUEST", "keywords ist im Modus 'embedding' nicht erlaubt.")

    diarization = payload.get("diarization")
    if mode != "diarization":
        if diarization is not None:
            raise V2ApiError(422, "INVALID_REQUEST", "diarization ist nur im Modus 'diarization' erlaubt.")
        return {"schema_version": V2_SCHEMA_VERSION, "mode": mode, "language": normalized_language, "keywords": keywords}
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
        "language": normalized_language,
        "keywords": keywords,
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


def _request_processing_key(language_override: str | None, glossary: Glossary | None = None) -> tuple:
    """Server settings key, with the per-request language override and glossary applied.

    The batch worker groups requests by this key, so an overridden language
    or a glossary simply forms its own batch group and reaches the engine
    unchanged. Without glossary terms the key is the plain 5-tuple.
    """

    key = _processing_key()
    if language_override:
        key = (key[0], key[1], key[2], language_override, key[4])
    return with_keywords(key, glossary)


def _glossary_or_server(glossary: Glossary | None) -> Glossary:
    """The request's resolved glossary; a caller without one (history retry) gets the server glossary."""

    return glossary if glossary is not None else resolve_glossary()


# One GPU-lock hold of a bulk job may not exceed roughly a live request's own
# GPU time. 64 ReDimNet windows are eight batches of 8 (~0.3 s); four MMS
# alignment chunks of ~25 s are ~0.8 s on an RTX 4090.
EMBED_SLICE_WINDOWS = 64
ALIGN_SLICE_CHUNKS = 4


async def _embed_windows_sliced(request: Request, windows: Sequence[VoiceWindow]) -> list[EmbeddedVoiceWindow]:
    """Embed many windows without holding the GPU lock for the whole list."""

    parts = await run_sliced_gpu_phase(
        request.app.state.local_gpu_lock,
        list(windows),
        embed_voice_windows,
        EMBED_SLICE_WINDOWS,
    )
    return [item for part in parts for item in part]


async def _extract_speaker_clouds(
    request: Request,
    audio: np.ndarray,
    standard: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Speaker clouds with the ReDimNet pass sliced under the GPU lock."""

    window_plan = await asyncio.to_thread(plan_speaker_windows, audio, standard, overlaps)
    embedded_windows = await _embed_windows_sliced(request, window_plan.windows)
    return await asyncio.to_thread(build_speaker_clouds, window_plan, embedded_windows)


async def _align_words_sliced(
    request: Request,
    audio: np.ndarray,
    chunks: Sequence[tuple[float, float]],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Forced alignment chunk-group by chunk-group, releasing the lock in between."""

    bounds = alignment_bounds(chunks)
    parts = await run_sliced_gpu_phase(
        request.app.state.local_gpu_lock,
        list(range(len(chunks))),
        align_chunk_slice,
        ALIGN_SLICE_CHUNKS,
        audio,
        chunks,
        texts,
        bounds,
    )
    return [word for part in parts for word in part]


def _language_settings(language_override: str | None) -> tuple[str, str]:
    """(model id, requested language): the request's override, else the server setting.

    Reads with defaults instead of the strict processing key, so a diarization fixture or a settings file
    without these keys never fails here before the ASR path itself has run.
    """

    with settings_lock:
        model_id = str(current_settings.get("local_model") or "")
        configured = str(current_settings.get("transcription_language", "auto") or "auto")
    return model_id, (language_override or configured).strip().lower()


def _static_language_decision(language_override: str | None) -> LanguageDecision:
    """The language of a request that runs no detection: explicit, Whisper's own "auto", or the de fallback."""

    model_id, language = _language_settings(language_override)
    applied = get_effective_transcription_language(model_id, language)
    return LanguageDecision(
        languages=[],
        language=applied,
        source="request" if language_override else "setting",
    )


async def _transcribe_audio(
    request: Request,
    audio: np.ndarray,
    language_override: str | None = None,
    glossary: Glossary | None = None,
) -> tuple[str, int, int, str, LanguageDecision]:
    """(text, transcription ms, ASR items, served model, language decision).

    With Cohere or Granite and an effective "auto" the language is detected per ASR item before any job is submitted, so
    "auto" never reaches the engine: one item (up to WHOLE_AUDIO_MAX_S) gets one decision; a long file is
    detected per Silero speech piece in one batched pass, cut into superchunks that never mix two languages and
    submitted as one job per language (decide_transcript_items). The glossary rides in the processing key of every
    job.
    """

    processing_key = _request_processing_key(language_override, _glossary_or_server(glossary))
    model_id = processing_key[0]
    batch_manager = request.app.state.whisper_batch_manager
    started = time.monotonic()
    detect = language_detection_applies(model_id, processing_key[3])
    decision: LanguageDecision | None = None
    if detect and uses_whole_audio_backend(model_id):
        # The language pieces decide where the superchunks are cut, so detection builds them itself.
        segments, decision = await decide_transcript_items(request.app.state.local_gpu_lock, audio)
    elif uses_whole_audio_backend(model_id):
        # Cohere accepts whole recordings, but a long one must still reach the
        # scheduler as chunks so live requests can run between them.
        segments = await asyncio.to_thread(split_audio_into_superchunks, audio)
    else:
        segments = await asyncio.to_thread(split_audio_for_whisper, audio)
    if decision is None:
        if detect and segments:
            decision = await decide_item_languages(request.app.state.local_gpu_lock, segments)
        else:
            decision = _static_language_decision(language_override)
    if not segments:
        return "", 0, 0, model_id, decision
    results = await submit_by_language(batch_manager, segments, processing_key, decision.languages or None)
    text = combine_transcription_chunks([result.text for result in results])
    model_id = served_model_id(results, model_id)
    transcription_ms = round((time.monotonic() - started) * 1000) - decision.duration_ms
    return text, max(0, transcription_ms), len(segments), model_id, decision


def _apply_language_to_models(models: dict[str, Any], decision: LanguageDecision) -> None:
    """models.asr.language (what really ran, the dominant one) plus the detection block when detection ran."""

    asr = models.get("asr")
    if not isinstance(asr, dict):
        return
    asr["language"] = decision.language
    block = decision.detection_block()
    if block is not None:
        asr["language_detection"] = block


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


def _turn_key(turn: Mapping[str, Any]) -> tuple[int, int, str]:
    return (int(turn["start_ms"]), int(turn["end_ms"]), str(turn["speaker_id"]))


@dataclass
class SpeakerLanguages:
    """Detected language per diarization label; a label without one (no cloud, no turns) gets ``default``.

    ``by_turn`` holds only the turns whose own detection overrode their speaker's language (keyed by
    start_ms, end_ms, speaker_id of the gated turn); every other turn is decoded in its speaker's language.
    """

    by_speaker: dict[str, str]
    sources: dict[str, str]
    default: str
    by_turn: dict[tuple[int, int, str], str] = field(default_factory=dict)

    def for_speaker(self, label: Any) -> str:
        return self.by_speaker.get(str(label), self.default)

    def source_for(self, label: Any) -> str:
        return self.sources.get(str(label), "majority" if self.by_speaker else SOURCE_FALLBACK)

    def for_turn(self, turn: Mapping[str, Any]) -> str:
        return self.by_turn.get(_turn_key(turn)) or self.for_speaker(turn["speaker_id"])

    def turn_source(self, turn: Mapping[str, Any]) -> str:
        return LANGUAGE_SOURCE_TURN if _turn_key(turn) in self.by_turn else LANGUAGE_SOURCE_SPEAKER

    def segment_source(self, label: Any, language: str) -> str:
        """LANGUAGE_SOURCE_TURN when a segment ran in another language than its speaker's."""

        return LANGUAGE_SOURCE_SPEAKER if language == self.for_speaker(label) else LANGUAGE_SOURCE_TURN


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in sorted(spans):
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def _span_seconds(spans: Sequence[tuple[int, int]]) -> float:
    return sum(end_ms - start_ms for start_ms, end_ms in spans) / 1000.0


def _speaker_lid_spans(
    cloud: Any,
    turns: Sequence[Mapping[str, Any]],
    max_seconds: float = LID_MAX_POOL_SECONDS,
) -> list[tuple[int, int]]:
    """Up to ``max_seconds`` of one speaker: the cloud's best inlier windows first, then its longest turns.

    Inliers are the overlap-free, margin-trimmed windows that also define the speaker's voice; a stitched window
    contributes its source spans. A speaker whose cloud is too small (short contributions) is topped up from
    its exclusive turns, so even a few seconds of speech still get a language of their own.
    """

    candidates: list[tuple[int, int]] = []
    inliers = sorted(getattr(cloud, "inliers", []) or [], key=lambda item: -float(item.quality))
    for window in inliers:
        if window.source_spans:
            candidates.extend((int(a), int(b)) for a, b in window.source_spans)
        elif window.start_ms is not None and window.end_ms is not None:
            candidates.append((int(window.start_ms), int(window.end_ms)))
    candidates.extend(
        (int(turn["start_ms"]), int(turn["end_ms"]))
        for turn in sorted(turns, key=lambda item: int(item["start_ms"]) - int(item["end_ms"]))
    )
    # The budget is spent in priority order: each candidate adds only its not yet chosen audio, clipped to what
    # is left. Trimming the merged list in time order instead could drop every inlier for an early raw turn.
    chosen: list[tuple[int, int]] = []
    remaining_ms = int(max_seconds * 1000)
    for start_ms, end_ms in candidates:
        if remaining_ms <= 0:
            break
        cursor = start_ms
        for chosen_start, chosen_end in [*chosen, (end_ms, end_ms)]:
            if remaining_ms <= 0 or cursor >= end_ms:
                break
            if chosen_end <= cursor:
                continue
            gap_end = min(chosen_start, end_ms)
            if gap_end > cursor:
                take = min(gap_end - cursor, remaining_ms)
                chosen = _merge_spans([*chosen, (cursor, cursor + take)])
                remaining_ms -= take
            cursor = max(cursor, chosen_end)
    return chosen


def _spans_audio(audio: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    pieces = [
        audio[max(0, round(start_ms * 16000 / 1000)) : min(len(audio), round(end_ms * 16000 / 1000))]
        for start_ms, end_ms in spans
    ]
    pieces = [piece for piece in pieces if len(piece)]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(pieces).astype(np.float32, copy=False))


async def _detect_speaker_languages(
    request: Request,
    audio: np.ndarray,
    speaker_clouds: Mapping[str, Any],
    exclusive_turns: Sequence[Mapping[str, Any]],
    asr_turns: Sequence[Mapping[str, Any]] = (),
) -> tuple[SpeakerLanguages, LanguageDecision]:
    """One language per diarization speaker as the prior, refined per ASR turn. Never raises.

    Every speaker's pooled speech is one LID window. Weak speakers take the majority of the confident ones,
    weighted by speech seconds. Every ``asr_turns`` turn (the gated/merged turns ASR will see) of at least
    LID_TURN_MIN_SECONDS is detected on its own as well (a turn over 30 s in parts, pooled); speakers and
    turns share one batched detection pass. A turn keeps its speaker's language unless its own result reaches
    LID_TURN_OVERRIDE_CONFIDENCE for another one (turn_language): a cluster that mixes an English speaker
    with German interjections of others, or a speaker switching for a sentence.
    """

    started = time.monotonic()
    turns_by_speaker: dict[str, list[Mapping[str, Any]]] = {}
    for turn in exclusive_turns:
        turns_by_speaker.setdefault(str(turn["speaker_id"]), []).append(turn)
    labels = sorted(set(speaker_clouds) | set(turns_by_speaker))
    speech_seconds = {
        label: _span_seconds(
            _merge_spans([(int(turn["start_ms"]), int(turn["end_ms"])) for turn in turns_by_speaker.get(label, [])])
        )
        for label in labels
    }
    pooled: dict[str, np.ndarray] = {}
    for label in labels:
        spans = _speaker_lid_spans(speaker_clouds.get(label), turns_by_speaker.get(label, []))
        if _span_seconds(spans) >= LID_MIN_SPEECH_SECONDS:
            pooled[label] = _spans_audio(audio, spans)
    # Turn windows ride in the same pass as the speakers: (turn index, window seconds) per window.
    windows: list[np.ndarray] = list(pooled.values())
    turn_owners: list[tuple[int, float]] = []
    for index, turn in enumerate(asr_turns):
        if (int(turn["end_ms"]) - int(turn["start_ms"])) / 1000.0 < LID_TURN_MIN_SECONDS:
            continue
        # A view, not a copy: on a long meeting the turns cover most of the file.
        turn_audio = audio[max(0, round(int(turn["start_ms"]) * 16)) : min(len(audio), round(int(turn["end_ms"]) * 16))]
        for window in split_windows(turn_audio):
            if len(window):
                windows.append(window)
                turn_owners.append((index, len(window) / 16000.0))

    warnings: list[dict[str, Any]] = []
    results: dict[str, Any] = {label: None for label in labels}
    turn_results: dict[int, Any] = {}
    if windows:
        try:
            detected = await detect_windows(request.app.state.local_gpu_lock, windows)
            results.update(dict(zip(pooled, detected[: len(pooled)])))
            per_turn: dict[int, list[tuple[Any, float]]] = {}
            for (index, window_seconds), result in zip(turn_owners, detected[len(pooled) :]):
                per_turn.setdefault(index, []).append((result, window_seconds))
            turn_results = {
                index: pool_results([result for result, _ in items], [weight for _, weight in items])
                for index, items in per_turn.items()
            }
        except Exception as exc:
            print(f"[WARNUNG-LID] Sprecher-Spracherkennung fehlgeschlagen, nutze '{language_detection_fallback()}': {exc}", file=sys.stderr)
            warnings.append(unavailable_warning(str(exc)))
            results = {label: None for label in labels}
            turn_results = {}

    resolved = resolve_speaker_languages(results, speech_seconds)
    by_speaker = {label: language for label, (language, _source) in resolved.items()}
    sources = {label: source for label, (_language, source) in resolved.items()}
    dominant = dominant_language(
        [(language, speech_seconds.get(label, 0.0)) for label, language in by_speaker.items()],
    )
    seconds: dict[str, float] = {}
    for label, language in by_speaker.items():
        seconds[language] = seconds.get(language, 0.0) + speech_seconds.get(label, 0.0)
    by_turn: dict[tuple[int, int, str], str] = {}
    overridden_log: list[str] = []
    for index, result in turn_results.items():
        turn = asr_turns[index]
        prior = by_speaker.get(str(turn["speaker_id"]), dominant)
        turn_seconds = (int(turn["end_ms"]) - int(turn["start_ms"])) / 1000.0
        language, overridden = turn_language(result, turn_seconds, prior)
        if overridden:
            by_turn[_turn_key(turn)] = language
            # The reported seconds follow what really runs: the turn's speech moves to its own language.
            seconds[prior] = max(0.0, seconds.get(prior, 0.0) - turn_seconds)
            seconds[language] = seconds.get(language, 0.0) + turn_seconds
            overridden_log.append(
                f"{turn['speaker_id']}@{int(turn['start_ms']) / 1000:.1f}s={language}:{result.probability:.2f}"
            )
    seconds = {code: value for code, value in seconds.items() if value > 0.0} or seconds
    decision = LanguageDecision(
        languages=[by_speaker[label] for label in labels],
        language=dominant_language(list(seconds.items()), dominant) if by_turn else dominant,
        source=SOURCE_DETECTED if any(result is not None for result in results.values()) else SOURCE_FALLBACK,
        model=configured_lid_model(),
        seconds=seconds,
        warnings=warnings,
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    log_decision(
        decision,
        [
            f"{label}={result.language}:{result.probability:.2f}" if result is not None else f"{label}=-"
            for label, result in results.items()
        ]
        + ([f"Turns {len(turn_results)}, umgestellt {len(overridden_log)}"] if turn_results else [])
        + overridden_log,
    )
    return SpeakerLanguages(by_speaker=by_speaker, sources=sources, default=dominant, by_turn=by_turn), decision


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
    language_override: str | None = None,
    speaker_languages: SpeakerLanguages | None = None,
    glossary: Glossary | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    processing_key = _request_processing_key(language_override, _glossary_or_server(glossary))
    model_id = processing_key[0]
    turn_chunk_counts: list[int] = []
    prepared_chunks: list[np.ndarray] = []
    chunk_languages: list[str] | None = [] if speaker_languages is not None else None
    for turn in turns:
        chunks = _turn_audio_chunks(audio, int(turn["start_ms"]), int(turn["end_ms"]))
        turn_chunk_counts.append(len(chunks))
        prepared_chunks.extend(chunks)
        if chunk_languages is not None:
            chunk_languages.extend([speaker_languages.for_turn(turn)] * len(chunks))
    started = time.monotonic()
    batch_manager = request.app.state.whisper_batch_manager
    # One job per language (the speaker's, or the turn's own where it overrode it); results come back in turn order.
    batch_results = await submit_by_language(batch_manager, prepared_chunks, processing_key, chunk_languages)
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
        if speaker_languages is not None:
            # Exactly the language this turn was decoded with.
            segment["language"] = speaker_languages.for_turn(turn)
            segment["language_source"] = speaker_languages.turn_source(turn)
        transcript_segments.append(segment)
    return transcript_segments, round((time.monotonic() - started) * 1000), served_model_id(batch_results, model_id)


async def _wxc_transcribe_segments(
    request: Request,
    audio: np.ndarray,
    gated_turns: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    apply_repetition_filter: bool,
    language_override: str | None = None,
    speaker_languages: SpeakerLanguages | None = None,
    glossary: Glossary | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any], str]:
    """Transcribe-first WXC path: large Silero chunks -> ASR -> word alignment ->
    gated speaker skeleton -> sentence-level 192D verification.

    With detected languages and more than one of them among the gated turns
    (each turn in its speaker's language unless its own detection overrode
    it), the superchunks are cut at every language change so no chunk mixes
    two languages, and every language is its own ASR job. The chunk list stays
    in time order, so alignment, seams and speaker assignment see what they
    saw before. One language keeps exactly the chunking of before. A sentence
    reports the language of the chunk that owns its midpoint.

    Raises WxcNoSpeechError when Silero finds no usable speech so the caller can
    fall back to the legacy per-turn path instead of returning silence.
    """

    processing_key = _request_processing_key(language_override, _glossary_or_server(glossary))
    timings: dict[str, int] = {}

    vad_started = time.monotonic()
    probs = await asyncio.to_thread(silero_frame_probs, audio)
    regions = speech_regions_from_probs(probs)
    timings["vad"] = round((time.monotonic() - vad_started) * 1000)
    if not regions:
        raise WxcNoSpeechError("Silero fand keine Sprachregionen.")

    chunk_languages: list[str] | None = None
    language_spans = (
        [
            (
                float(turn["start_ms"]) / 1000.0,
                float(turn["end_ms"]) / 1000.0,
                speaker_languages.for_turn(turn),
            )
            for turn in gated_turns
        ]
        if speaker_languages is not None
        else []
    )
    if len({language for _, _, language in language_spans}) > 1:
        chunks, chunk_languages = build_language_superchunks(regions, language_spans)
    else:
        chunks = build_superchunks(regions)
        if speaker_languages is not None:
            single = language_spans[0][2] if language_spans else speaker_languages.default
            chunk_languages = [single] * len(chunks)
    chunk_audio = padded_chunk_audio(audio, chunks)

    asr_started = time.monotonic()
    batch_manager = request.app.state.whisper_batch_manager
    results = await submit_by_language(batch_manager, chunk_audio, processing_key, chunk_languages)
    texts = [result.text for result in results]
    # Collapse ASR repetition loops on the full chunk text, BEFORE alignment
    # splits it into words. Word-level alignment plus sentence splitting at "."
    # would otherwise scatter a loop like "So. So. So." into single-word
    # sentences, where the per-sentence filter can never see the repetition.
    if apply_repetition_filter:
        texts = [filter_repeated_patterns(text) for text in texts]
    timings["transcription"] = round((time.monotonic() - asr_started) * 1000)

    align_started = time.monotonic()
    words = await _align_words_sliced(request, audio, chunks, texts)
    timings["alignment"] = round((time.monotonic() - align_started) * 1000)

    assign_words_to_turns(words, gated_turns)
    sentences = words_to_sentences(words, overlaps)
    bounds = alignment_bounds(chunks)

    verify_started = time.monotonic()
    verify_windows, verify_kept = await asyncio.to_thread(plan_verification_windows, audio, sentences)
    verify_embedded = await _embed_windows_sliced(request, verify_windows) if verify_windows else []
    verification = await asyncio.to_thread(finish_verification, verify_windows, verify_kept, verify_embedded)
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
        if chunk_languages and speaker_languages is not None:
            language = _chunk_language_at((sentence["t0"] + sentence["t1"]) / 2.0, bounds, chunk_languages)
            segment["language"] = language
            segment["language_source"] = speaker_languages.segment_source(sentence["speaker"], language)
        transcript_segments.append(segment)

    speech_seconds = sum(b - a for a, b in regions)
    diagnostics = {
        "speech_regions": len(regions),
        "speech_seconds": round(speech_seconds, 1),
        "superchunks": len(chunks),
        **(
            {"superchunk_languages": {code: chunk_languages.count(code) for code in sorted(set(chunk_languages))}}
            if chunk_languages
            else {}
        ),
        "words_aligned": len(words),
        "verification": verification,
    }
    return transcript_segments, timings, diagnostics, served_model_id(results, processing_key[0])


def _chunk_language_at(moment: float, bounds: Sequence[float], chunk_languages: Sequence[str]) -> str:
    """Language of the chunk that owns ``moment`` (the seam ownership of the word alignment)."""

    for index, language in enumerate(chunk_languages):
        if bounds[index] < moment <= bounds[index + 1]:
            return language
    return chunk_languages[-1]


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
    request_keywords: Sequence[str] | None = None,
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
    # The request's own glossary terms (normalised, capped), so a history retry sends them again.
    stored_keywords = sanitize_keyword_setting(list(request_keywords or []))
    if stored_keywords:
        entry["request_keywords"] = stored_keywords
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
    language_override: str | None = None,
    glossary: Glossary | None = None,
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
    # Standard diarization preserves all speaker activity and therefore
    # defines the safe, overlap-free enrollment regions. Exclusive turns
    # remain reserved for ASR so overlapping speech is transcribed once.
    speaker_clouds = await _extract_speaker_clouds(http_request, audio, standard, overlaps)
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

    # Per-speaker language, decided once the speaker clouds are final and before
    # any ASR: Cohere/Granite + an effective "auto" only. Everything else keeps its one
    # explicit (or Whisper-detected) language for the whole file.
    # The gated/merged turns ASR will see are refined per turn in the same
    # detection pass (the speaker's language stays the prior).
    if gate_enabled:
        turns, gate_diagnostics = prefilter_turns(effective_exclusive)
    else:
        turns = _merge_exclusive_turns(effective_exclusive)
        gate_diagnostics = {"enabled": False}

    speaker_languages: SpeakerLanguages | None = None
    lid_ms: int | None = None
    if language_detection_applies(*_language_settings(language_override)):
        speaker_languages, language_decision = await _detect_speaker_languages(
            http_request, audio, speaker_clouds, effective_exclusive, turns
        )
        lid_ms = language_decision.duration_ms
    else:
        language_decision = _static_language_decision(language_override)

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
                    language_override,
                    speaker_languages,
                    glossary,
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
            language_override,
            speaker_languages,
            glossary,
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
        # The ASR paths set the language each segment was decoded with; anything else (no detection, a WXC
        # chunk list without languages) runs in the speaker's or the request's one language.
        if speaker_languages is None:
            segment["language"] = language_decision.language
            segment["language_source"] = language_decision.source
        elif "language" not in segment:
            segment["language"] = speaker_languages.for_speaker(dia_speaker_id)
            segment["language_source"] = LANGUAGE_SOURCE_SPEAKER

    def _speaker_language_fields(dia_speaker_id: str) -> dict[str, str]:
        if speaker_languages is None:
            return {"language": language_decision.language, "language_source": language_decision.source}
        return {
            "language": speaker_languages.for_speaker(dia_speaker_id),
            "language_source": speaker_languages.source_for(dia_speaker_id),
        }

    transcript_text = " ".join(segment["text"] for segment in transcript_segments).strip()
    speaker_assignments: list[dict[str, Any]] = []
    unknown_speakers: list[dict[str, Any]] = []
    unresolved_speakers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if wxc_fallback_warning is not None:
        warnings.append(wxc_fallback_warning)
    warnings.extend(language_decision.warnings)
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
                    **_speaker_language_fields(dia_speaker_id),
                }
            )
            if cloud.status != "ready":
                # The identity was accepted under the stricter imperfect-cluster bar;
                # keep the audio-quality caveat visible instead of hiding it behind a
                # confident "known" label.
                warnings.append(
                    {
                        "code": "KNOWN_SPEAKER_IMPERFECT_CLUSTER",
                        "speaker_id": assignment["speaker_id"],
                        "diarization_speaker_id": dia_speaker_id,
                        "status": cloud.status,
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
                **_speaker_language_fields(dia_speaker_id),
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
        "transcript": {"text": transcript_text, "language": language_decision.language, "segments": transcript_segments},
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
    if lid_ms is not None:
        timings["language_detection"] = lid_ms
    models = {
        "asr": {"id": asr_model_id},
        "diarization": dia_response.get("model") or {},
        "embedding": embedding_model_metadata(),
    }
    _apply_language_to_models(models, language_decision)
    return result, timings, models, warnings, transcript_text


def create_v2_capabilities_api(app: FastAPI) -> FastAPI:
    @app.get("/v2/capabilities")
    async def capabilities_v2():
        with settings_lock:
            settings_snapshot = current_settings.copy()
        settings_snapshot = normalize_settings(settings_snapshot)
        model_id = settings_snapshot["local_model"]
        configured_language = settings_snapshot["transcription_language"]
        detection = lid_status()
        detection_enabled = settings_snapshot["cohere_language_detection"] is True
        # "auto" stays "auto" for Cohere/Granite while detection is on and its model is not known to be broken.
        detection_effective = detection_enabled and detection["error"] is None
        return {
            "schema_version": V2_SCHEMA_VERSION,
            "service": "asr",
            "model": {
                "id": model_id,
                "label": get_local_model_label(model_id),
                "backend": get_local_model_backend(model_id),
                "loaded": is_local_model_loaded(model_id),
            },
            "language": {
                "configured": configured_language,
                "effective": get_effective_transcription_language(
                    model_id, configured_language, detection=detection_effective
                ),
                "applied": not uses_parakeet_backend(model_id),
                "detection": {
                    "enabled": detection_enabled,
                    "model": detection["model"],
                    "loaded": detection["loaded"],
                },
                # The list a request is validated against. It stays global, per-model lists would turn requests
                # that work today into 422.
                "options": list(SUPPORTED_LANGUAGE_OPTIONS.values()),
            },
            "modes": list(V2_MODES),
            "embedding": {"id": embedding_model_metadata()["id"]},
            "diarization": {"configured": dia_is_configured()},
            # Always present: clients detect the request field "keywords" by this key.
            "keywords": capabilities_block(model_id),
        }

    return app


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
            # Server glossary first, the request's terms after it; resolved once, sent with every ASR job.
            glossary = resolve_glossary(parsed.get("keywords")) if mode != "embedding" else None
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
            language_decision: LanguageDecision | None = None
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
                transcript_text, transcription_ms, _, asr_model_id, language_decision = await _transcribe_audio(
                    http_request, audio, parsed.get("language"), glossary
                )
                if filter_enabled:
                    transcript_text = filter_repeated_patterns(transcript_text)
                timings["transcription"] = transcription_ms
                models["asr"] = {"id": asr_model_id}
                result = {"transcript": {"text": transcript_text, "language": language_decision.language}}
            elif mode == "transcript_embedding":
                transcript_text, transcription_ms, _, asr_model_id, language_decision = await _transcribe_audio(
                    http_request, audio, parsed.get("language"), glossary
                )
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
                result = {
                    "transcript": {"text": transcript_text, "language": language_decision.language},
                    "embedding": embedding_result,
                }
            else:
                result, mode_timings, models, warnings, transcript_text = await _process_diarization(
                    http_request,
                    file,
                    audio,
                    parsed["diarization"],
                    filter_enabled,
                    gate_enabled,
                    pipeline_mode,
                    parsed.get("language"),
                    glossary,
                )
                timings.update(mode_timings)
                transcription_ms = mode_timings.get("transcription")
                embedding_ms = mode_timings.get("embedding")
                asr_model_id = (models.get("asr") or {}).get("id")

            if language_decision is not None:
                # What the request really ran with (detected or explicit), not a recomputation from settings.
                _apply_language_to_models(models, language_decision)
                warnings.extend(language_decision.warnings)
                if language_decision.detection_ran:
                    timings["language_detection"] = language_decision.duration_ms
            elif asr_model_id and isinstance(models.get("asr"), dict) and "language" not in models["asr"]:
                models["asr"]["language"] = _static_language_decision(parsed.get("language")).language
            if glossary is not None:
                # applied follows the model that really served the request (a failover may have changed it).
                asr_block = models.setdefault("asr", {})
                if isinstance(asr_block, dict):
                    asr_block["keywords"] = keywords_block(glossary, asr_model_id)
                warnings.extend(truncation_warnings(glossary))
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
                request_keywords=parsed.get("keywords") if glossary is not None else None,
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


__all__ = ["create_v2_api", "create_v2_capabilities_api"]
