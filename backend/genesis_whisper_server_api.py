import asyncio
import datetime
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .genesis_whisper_server_audio import get_audio_duration_seconds, load_audio_file
from .genesis_whisper_server_auth import authorize_api_key, get_auth_store
from .genesis_whisper_server_batching import enqueue_audio_segments_bounded
from .genesis_whisper_server_chunking import combine_transcription_chunks, split_audio_for_whisper
from .genesis_whisper_server_globals import (
    current_settings,
    get_effective_transcription_language,
    history_lock,
    settings_lock,
    transcription_history,
    uses_cohere_backend,
)
from .genesis_whisper_server_local_asr_engine import (
    get_last_local_asr_load_error,
    load_local_asr_model,
    transcribe_local_asr_batch,
)
from .genesis_whisper_server_storage import log_transcription
from .genesis_whisper_server_vid import generate_voice_vector
from .genesis_whisper_server_gpu import run_blocking_gpu_phase, shared_gpu_lease
from .genesis_whisper_server_repetition import (
    REPETITION_FILTER_HEADER,
    filter_repeated_patterns,
    repetition_filter_enabled,
)


def _normalize_engine(engine: str) -> str:
    normalized = (engine or "local").strip().lower()
    if normalized in ("local", "lokal"):
        return "local"
    raise HTTPException(status_code=400, detail="Es wird nur noch die lokale ASR-Engine unterstuetzt.")


def _get_local_processing_key() -> Tuple[str, str, str, str, str]:
    with settings_lock:
        model_id = current_settings["local_model"]
        device = current_settings["local_gpu_device"]
        cache_path = current_settings["local_model_cache_path"]
        language = current_settings.get("transcription_language", "auto")
        precision = current_settings.get("local_model_precision", "fp16")
    return model_id, device, cache_path, language, precision


def process_local_asr_batch(audio_batch: List[np.ndarray], processing_key: Tuple[str, str, str, str, str]) -> List[str]:
    model_id, device, cache_path, language, precision = processing_key
    with shared_gpu_lease():
        if not load_local_asr_model(model_id, device, cache_path, precision):
            load_error = get_last_local_asr_load_error()
            detail = "Lokales ASR-Modell konnte nicht geladen werden."
            if load_error:
                detail = f"{detail} Ursache: {load_error}"
            raise RuntimeError(detail)
        with settings_lock:
            batch_size = max(1, int(current_settings.get("batch_max_segments", len(audio_batch) or 1)))
        return transcribe_local_asr_batch(audio_batch, language=language, batch_size=batch_size)


def _should_use_batch(voice_ident: bool) -> bool:
    # Voice embeddings are a separate serial phase. ASR must always retain the
    # normal chunk/batch path so long recordings are never truncated to a
    # model's single ~30-second processor window.
    _ = voice_ident
    return True


def create_api(app: FastAPI) -> FastAPI:
    @app.post("/transcribe/")
    async def transcribe_endpoint(
        request: Request,
        file: UploadFile = File(..., description="Die zu transkribierende Audiodatei (z.B. WAV, MP3, FLAC)."),
        engine: str = Form("local", description="Es wird nur noch die lokale ASR-Engine unterstuetzt."),
        voice_ident: bool = Form(False, description="Wenn True, wird zusaetzlich ein Stimm-Vektor (Embedding) generiert."),
    ):
        api_key_id = authorize_api_key(request)
        request_start_time = time.monotonic()
        source_ip = request.client.host if request.client else "unknown"
        engine = _normalize_engine(engine)
        filename = file.filename or "upload"

        try:
            audio_data = await asyncio.to_thread(load_audio_file, file.file, filename)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Konnte Audiodatei nicht verarbeiten: {exc}") from exc

        transcription_text = ""
        voice_vector: Optional[List[float]] = None
        transcription_duration_ms: Optional[int] = None
        voice_vector_duration_ms: Optional[int] = None
        batch_ids: List[str] = []
        segment_count = 1
        used_batching = False

        local_processing_key = _get_local_processing_key()
        model_id = local_processing_key[0]
        effective_language = get_effective_transcription_language(model_id, local_processing_key[3])

        # ASR always uses the ordinary chunk/batch path.  Voice embeddings are
        # generated afterwards in their own serialized GPU phase, so enabling
        # ``voice_ident`` can never truncate a long recording to one model
        # processor window.
        used_batching = _should_use_batch(voice_ident)
        batch_manager = request.app.state.whisper_batch_manager
        request_id = uuid.uuid4().hex
        batch_start = time.monotonic()
        if uses_cohere_backend(model_id):
            segment_count = 1
            batch_result = await batch_manager.enqueue(
                audio_data=audio_data,
                request_id=request_id,
                segment_index=0,
                total_segments=1,
                processing_key=local_processing_key,
            )
            transcription_text = batch_result.text
            batch_ids = [batch_result.batch_id]
            transcription_duration_ms = round((time.monotonic() - batch_start) * 1000)
        else:
            segmented_audio = await asyncio.to_thread(split_audio_for_whisper, audio_data)
            segment_count = len(segmented_audio)

            if segment_count == 0:
                transcription_text = ""
                transcription_duration_ms = 0
            else:
                batch_results = await enqueue_audio_segments_bounded(
                    batch_manager,
                    segmented_audio,
                    request_id,
                    local_processing_key,
                )
                transcription_text = combine_transcription_chunks([result.text for result in batch_results])
                batch_ids = sorted({result.batch_id for result in batch_results})
                transcription_duration_ms = round((time.monotonic() - batch_start) * 1000)

        if voice_ident and used_batching:
            local_gpu_lock = request.app.state.local_gpu_lock
            v_start = time.monotonic()
            try:
                async with local_gpu_lock:
                    print("[API-INFO] Generiere ReDimNet2-Stimm-Vektor...", file=sys.stderr)
                    vector_np = await run_blocking_gpu_phase(generate_voice_vector, audio_data)
                voice_vector = vector_np.tolist()
                voice_vector_duration_ms = round((time.monotonic() - v_start) * 1000)
                print(
                    f"[API-INFO] Stimm-Vektor erfolgreich generiert in {voice_vector_duration_ms}ms.",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[API-FEHLER] Stimm-Vektor-Generierung fehlgeschlagen: {exc}", file=sys.stderr)
                voice_vector = None
                voice_vector_duration_ms = round((time.monotonic() - v_start) * 1000)

        if repetition_filter_enabled(request.headers.get(REPETITION_FILTER_HEADER)):
            transcription_text = filter_repeated_patterns(transcription_text)

        total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": source_ip,
            "engine": engine,
            "model_id": model_id,
            "transcription_language": effective_language,
            "total_duration_ms": total_duration_ms,
            "transcription_duration_ms": transcription_duration_ms,
            "voice_vector_duration_ms": voice_vector_duration_ms,
            "transcript": transcription_text,
            "voice_ident_requested": voice_ident,
            "batched": used_batching,
            "segment_count": segment_count,
            "batch_ids": batch_ids,
        }

        with history_lock:
            transcription_history.appendleft(log_entry)
        log_transcription(log_entry)

        if api_key_id:
            get_auth_store().record_api_key_usage(api_key_id, get_audio_duration_seconds(audio_data))

        response_data = {
            "transcription": transcription_text,
            "total_duration_ms": total_duration_ms,
            "transcription_duration_ms": transcription_duration_ms,
        }
        if voice_ident:
            response_data["voice_vector"] = voice_vector
            response_data["voice_vector_duration_ms"] = voice_vector_duration_ms

        if await request.is_disconnected():
            print("[API WARNUNG] Client hat die Verbindung getrennt, bevor die Antwort gesendet werden konnte.", file=sys.stderr)
            return

        return response_data

    @app.get("/v1/models")
    async def list_models_openai():
        with settings_lock:
            local_model = current_settings.get("local_model", "whisper-1")
        return {
            "object": "list",
            "data": [
                {
                    "id": local_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "genesis"
                },
                {
                    "id": "whisper-1",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "genesis"
                }
            ]
        }

    @app.post("/v1/audio/transcriptions")
    async def transcribe_openai_endpoint(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0.0),
    ):
        # We call the existing endpoint logic to avoid duplication
        response = await transcribe_endpoint(request, file=file, engine="local", voice_ident=False)
        transcription_text = response.get("transcription", "")
        
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(transcription_text)
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": current_settings.get("transcription_language", "auto"),
                "duration": response.get("transcription_duration_ms", 0) / 1000.0,
                "text": transcription_text,
                "segments": []
            }
            
        return {"text": transcription_text}

    return app
