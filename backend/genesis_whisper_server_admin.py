from __future__ import annotations

import asyncio
import time
import uuid
from statistics import mean
from typing import Any, Dict, List

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from .genesis_whisper_server_audio import get_audio_duration_seconds, load_audio_bytes
from .genesis_whisper_server_auth import (
    get_admin_key_store,
    require_admin,
)
from .genesis_whisper_server_chunking import combine_transcription_chunks, split_audio_for_whisper
from .genesis_whisper_server_globals import (
    AVAILABLE_DEVICES,
    DEVICE_MAP_UI_TO_INTERNAL,
    current_settings,
    get_effective_transcription_language,
    history_lock,
    local_model_components,
    LOCAL_ASR_MODEL_MAP,
    SUPPORTED_LANGUAGE_OPTIONS,
    SUPPORTED_MODEL_PRECISION_OPTIONS,
    settings_lock,
    transcription_history,
    uses_cohere_backend,
)
from .genesis_whisper_server_local_asr_engine import get_last_local_asr_load_error, load_local_asr_model
from .genesis_whisper_server_model_manager import (
    delete_model_cache,
    list_model_statuses,
    queue_model_download,
)
from .genesis_whisper_server_storage import normalize_settings, save_settings


class AdminSettingsPayload(BaseModel):
    local_model: str
    local_gpu_device: str
    local_model_precision: str = "fp16"
    local_model_cache_path: str
    transcription_language: str
    batch_wait_time_ms: int
    batch_max_segments: int
    batch_max_audio_seconds: float
    huggingface_token: str


class AdminModelActionPayload(BaseModel):
    model_id: str
    storage_path: str | None = None
    huggingface_token: str | None = None


def _serialize_settings() -> Dict[str, Any]:
    with settings_lock:
        settings_copy = current_settings.copy()
    return normalize_settings(settings_copy)


def _settings_options() -> Dict[str, List[Dict[str, str]]]:
    models = [{"label": label, "value": model_id} for label, model_id in LOCAL_ASR_MODEL_MAP.items()]
    devices = [{"label": label, "value": DEVICE_MAP_UI_TO_INTERNAL[label]} for label in AVAILABLE_DEVICES]
    languages = [{"label": label, "value": value} for label, value in SUPPORTED_LANGUAGE_OPTIONS.items()]
    precisions = [{"label": label, "value": value} for label, value in SUPPORTED_MODEL_PRECISION_OPTIONS.items()]
    return {"models": models, "devices": devices, "languages": languages, "precisions": precisions}


def _load_model_for_settings(settings_snapshot: Dict[str, Any]) -> bool:
    model_id = settings_snapshot["local_model"]
    device = settings_snapshot["local_gpu_device"]
    precision = settings_snapshot["local_model_precision"]
    cache_path = settings_snapshot["local_model_cache_path"]
    return load_local_asr_model(model_id, device, cache_path, precision)


def _get_local_processing_key() -> tuple[str, str, str, str, str]:
    with settings_lock:
        model_id = current_settings["local_model"]
        device = current_settings["local_gpu_device"]
        cache_path = current_settings["local_model_cache_path"]
        language = current_settings.get("transcription_language", "auto")
        precision = current_settings.get("local_model_precision", "fp16")
    return model_id, device, cache_path, language, precision


def _effective_model_storage_path(storage_path: str | None = None) -> str:
    if storage_path is not None:
        return str(storage_path)
    return str(_serialize_settings().get("local_model_cache_path", ""))


def _get_loaded_model_cuda_index() -> int | None:
    model = local_model_components.get("model")
    if model is None:
        return None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return None
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _reset_peak_vram_tracking(cuda_index: int | None) -> None:
    if cuda_index is None:
        return
    torch.cuda.synchronize(cuda_index)
    torch.cuda.reset_peak_memory_stats(cuda_index)


def _read_peak_vram_metrics(cuda_index: int | None) -> Dict[str, float | None]:
    if cuda_index is None:
        return {
            "peak_vram_reserved_mb": None,
            "peak_vram_allocated_mb": None,
        }

    torch.cuda.synchronize(cuda_index)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(cuda_index)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(cuda_index)
    return {
        "peak_vram_reserved_mb": round(peak_reserved_bytes / (1024 * 1024), 2),
        "peak_vram_allocated_mb": round(peak_allocated_bytes / (1024 * 1024), 2),
    }


async def _run_admin_benchmark(request: Request, audio_data, repeat_count: int) -> Dict[str, Any]:
    processing_key = _get_local_processing_key()
    model_id, _, _, configured_language, _ = processing_key
    effective_language = get_effective_transcription_language(model_id, configured_language)

    if not await asyncio.to_thread(_load_model_for_settings, _serialize_settings()):
        load_error = get_last_local_asr_load_error()
        detail = "Lokales ASR-Modell konnte fuer den Benchmark nicht geladen werden."
        if load_error:
            detail = f"{detail} Ursache: {load_error}"
        raise HTTPException(status_code=500, detail=detail)

    cuda_index = _get_loaded_model_cuda_index()
    _reset_peak_vram_tracking(cuda_index)
    batch_started_at = time.perf_counter()

    audio_seconds = round(get_audio_duration_seconds(audio_data), 3)
    if uses_cohere_backend(model_id):
        segments_per_run = [audio_data]
    else:
        segments_per_run = await asyncio.to_thread(split_audio_for_whisper, audio_data)

    chunks_per_run = len(segments_per_run)
    total_chunks = chunks_per_run * repeat_count
    total_audio_seconds = round(audio_seconds * repeat_count, 3)
    batch_ids: List[str] = []

    if chunks_per_run == 0:
        peak_metrics = _read_peak_vram_metrics(cuda_index)
        return {
            "ok": True,
            "workflow": "whisper_chunk_queue" if not uses_cohere_backend(model_id) else "cohere_audio_batch",
            "model_id": model_id,
            "transcription_language": effective_language,
            "repeat_count": repeat_count,
            "audio_seconds": audio_seconds,
            "total_audio_seconds": total_audio_seconds,
            "chunks_per_run": 0,
            "total_chunks": 0,
            "batches_used": 0,
            "total_wall_time_ms": 0,
            "avg_wall_time_per_run_ms": 0,
            "rtf": None,
            "transcripts_match": True,
            "transcript": "",
            **peak_metrics,
        }

    batch_manager = request.app.state.whisper_batch_manager
    benchmark_id = uuid.uuid4().hex[:10]

    pending_results = []
    for repeat_index in range(repeat_count):
        request_id = f"benchmark-{benchmark_id}-{repeat_index}"
        for segment_index, segment in enumerate(segments_per_run):
            pending_results.append(
                batch_manager.enqueue(
                    audio_data=segment,
                    request_id=request_id,
                    segment_index=segment_index,
                    total_segments=chunks_per_run,
                    processing_key=processing_key,
                )
            )

    try:
        batch_results = await asyncio.gather(*pending_results)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_wall_time_ms = round((time.perf_counter() - batch_started_at) * 1000)
    transcripts: List[str] = []
    result_index = 0
    for _ in range(repeat_count):
        segment_texts: List[str] = []
        for _ in range(chunks_per_run):
            segment_result = batch_results[result_index]
            result_index += 1
            segment_texts.append(segment_result.text)
            batch_ids.append(segment_result.batch_id)
        transcripts.append(combine_transcription_chunks(segment_texts))

    normalized_transcripts = {transcript.strip() for transcript in transcripts}
    wall_seconds = total_wall_time_ms / 1000 if total_wall_time_ms > 0 else 0.0
    peak_metrics = _read_peak_vram_metrics(cuda_index)

    return {
        "ok": True,
        "workflow": "whisper_chunk_queue" if not uses_cohere_backend(model_id) else "cohere_audio_batch",
        "model_id": model_id,
        "transcription_language": effective_language,
        "repeat_count": repeat_count,
        "audio_seconds": audio_seconds,
        "total_audio_seconds": total_audio_seconds,
        "chunks_per_run": chunks_per_run,
        "total_chunks": total_chunks,
        "batches_used": len(set(batch_ids)),
        "total_wall_time_ms": total_wall_time_ms,
        "avg_wall_time_per_run_ms": round(total_wall_time_ms / repeat_count, 2),
        "rtf": round(total_audio_seconds / wall_seconds, 3) if wall_seconds > 0 else None,
        "transcripts_match": len(normalized_transcripts) <= 1,
        "transcript": transcripts[0] if transcripts else "",
        **peak_metrics,
    }


def create_admin_api(app: FastAPI) -> FastAPI:
    @app.get("/api/admin/keys")
    async def admin_get_keys(_: dict[str, str] = Depends(require_admin)):
        return get_admin_key_store().list_keys()

    @app.post("/api/admin/keys")
    async def admin_rotate_key(_: dict[str, str] = Depends(require_admin)):
        return {
            "key": get_admin_key_store().rotate_admin_key(),
            "keys": get_admin_key_store().list_keys(),
        }

    @app.get("/api/admin/settings")
    async def admin_get_settings(_: dict[str, str] = Depends(require_admin)):
        model_identifier = local_model_components.get("model_identifier")
        settings_snapshot = _serialize_settings()
        return {
            "settings": settings_snapshot,
            "options": _settings_options(),
            "models": list_model_statuses(settings_snapshot.get("local_model_cache_path", "")),
            "loaded_model_identifier": list(model_identifier) if model_identifier else None,
        }

    @app.put("/api/admin/settings")
    async def admin_update_settings(payload: AdminSettingsPayload, _: dict[str, str] = Depends(require_admin)):
        payload_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        normalized = normalize_settings(payload_data)
        with settings_lock:
            previous_settings = current_settings.copy()
            current_settings.update(normalized)
            saved_settings = save_settings(current_settings.copy())
            current_settings.clear()
            current_settings.update(saved_settings)

        model_settings_changed = any(
            previous_settings.get(key) != saved_settings.get(key)
            for key in ("local_model", "local_gpu_device", "local_model_precision", "local_model_cache_path")
        )
        model_loaded = None
        if model_settings_changed:
            model_loaded = await asyncio.to_thread(_load_model_for_settings, saved_settings)

        return {
            "ok": True,
            "settings": _serialize_settings(),
            "model_reloaded": model_settings_changed,
            "model_loaded": model_loaded,
            "options": _settings_options(),
            "models": list_model_statuses(saved_settings.get("local_model_cache_path", "")),
        }

    @app.get("/api/admin/models")
    async def admin_get_models(storage_path: str | None = None, _: dict[str, str] = Depends(require_admin)):
        return {"models": list_model_statuses(_effective_model_storage_path(storage_path))}

    @app.post("/api/admin/models/download")
    async def admin_download_model(payload: AdminModelActionPayload, _: dict[str, str] = Depends(require_admin)):
        try:
            job = queue_model_download(
                payload.model_id,
                _effective_model_storage_path(payload.storage_path),
                payload.huggingface_token,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "job": job,
            "models": list_model_statuses(_effective_model_storage_path(payload.storage_path)),
        }

    @app.post("/api/admin/models/delete")
    async def admin_delete_model(payload: AdminModelActionPayload, _: dict[str, str] = Depends(require_admin)):
        try:
            result = delete_model_cache(payload.model_id, _effective_model_storage_path(payload.storage_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **result,
            "models": list_model_statuses(_effective_model_storage_path(payload.storage_path)),
        }

    @app.get("/api/admin/stats")
    async def admin_stats(_: dict[str, str] = Depends(require_admin)):
        with history_lock:
            history_items = list(transcription_history)

        recent_history = history_items[:25]
        total_requests = len(history_items)
        total_duration_values = [entry.get("total_duration_ms", 0) for entry in history_items if entry.get("total_duration_ms") is not None]
        transcription_values = [entry.get("transcription_duration_ms", 0) for entry in history_items if entry.get("transcription_duration_ms") is not None]

        return {
            "summary": {
                "total_requests": total_requests,
                "avg_total_duration_ms": round(mean(total_duration_values), 2) if total_duration_values else None,
                "avg_transcription_duration_ms": round(mean(transcription_values), 2) if transcription_values else None,
            },
            "history": recent_history,
        }

    @app.get("/api/admin/queue")
    async def admin_queue(request: Request, _: dict[str, str] = Depends(require_admin)):
        batch_manager = request.app.state.whisper_batch_manager
        return batch_manager.snapshot()

    @app.post("/api/admin/benchmark")
    async def admin_benchmark(
        request: Request,
        file: UploadFile = File(..., description="Audio- oder Video-Datei fuer den Benchmark."),
        repeat_count: int = Form(1),
        _: dict[str, str] = Depends(require_admin),
    ):
        if repeat_count < 1 or repeat_count > 64:
            raise HTTPException(status_code=400, detail="Wiederholungen muessen zwischen 1 und 64 liegen.")

        filename = file.filename or "benchmark-audio"
        try:
            audio_bytes = await file.read()
            audio_data = load_audio_bytes(audio_bytes, filename)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Konnte Audiodatei nicht verarbeiten: {exc}") from exc

        benchmark_result = await _run_admin_benchmark(request, audio_data, repeat_count)
        benchmark_result["file_name"] = filename
        return benchmark_result

    return app
