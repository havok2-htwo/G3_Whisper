import json
import os
import sys
from typing import Any, Dict, Optional

from .genesis_whisper_server_globals import (
    LOCAL_ASR_MODEL_MAP,
    LOG_FILE,
    LOGS_DIR,
    SETTINGS_FILE,
    SUPPORTED_LANGUAGE_VALUES,
)


DEFAULT_SETTINGS: Dict[str, Any] = {
    "local_model": "openai/whisper-large-v3-turbo",
    "local_gpu_device": "auto",
    "local_model_cache_path": ".\\models",
    "transcription_language": "auto",
    "batch_wait_time_ms": 500,
    "batch_max_segments": 32,
    "batch_max_audio_seconds": 300.0,
    "huggingface_token": "",
}


def normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = settings or {}
    normalized: Dict[str, Any] = {}
    valid_models = set(LOCAL_ASR_MODEL_MAP.values())

    local_model = str(source.get("local_model", DEFAULT_SETTINGS["local_model"])).strip() or DEFAULT_SETTINGS["local_model"]
    normalized["local_model"] = local_model if local_model in valid_models else DEFAULT_SETTINGS["local_model"]
    normalized["local_gpu_device"] = str(source.get("local_gpu_device", DEFAULT_SETTINGS["local_gpu_device"])).strip() or DEFAULT_SETTINGS["local_gpu_device"]
    normalized["local_model_cache_path"] = str(source.get("local_model_cache_path", "")).strip()
    configured_language = str(source.get("transcription_language", DEFAULT_SETTINGS["transcription_language"])).strip().lower()
    normalized["transcription_language"] = (
        configured_language if configured_language in SUPPORTED_LANGUAGE_VALUES else DEFAULT_SETTINGS["transcription_language"]
    )
    normalized["huggingface_token"] = str(source.get("huggingface_token", "")).strip()

    try:
        normalized["batch_wait_time_ms"] = max(0, int(source.get("batch_wait_time_ms", DEFAULT_SETTINGS["batch_wait_time_ms"])))
    except (TypeError, ValueError):
        normalized["batch_wait_time_ms"] = DEFAULT_SETTINGS["batch_wait_time_ms"]

    try:
        normalized["batch_max_segments"] = max(1, int(source.get("batch_max_segments", DEFAULT_SETTINGS["batch_max_segments"])))
    except (TypeError, ValueError):
        normalized["batch_max_segments"] = DEFAULT_SETTINGS["batch_max_segments"]

    try:
        normalized["batch_max_audio_seconds"] = max(1.0, float(source.get("batch_max_audio_seconds", DEFAULT_SETTINGS["batch_max_audio_seconds"])))
    except (TypeError, ValueError):
        normalized["batch_max_audio_seconds"] = DEFAULT_SETTINGS["batch_max_audio_seconds"]

    return normalized


def load_settings() -> Dict[str, Any]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file_obj:
            return normalize_settings(json.load(file_obj))
    except (json.JSONDecodeError, IOError) as exc:
        print(f"[FEHLER] Konnte Einstellungsdatei nicht laden: {exc}", file=sys.stderr)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_settings(settings)
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(normalized, file_obj, indent=4)
        print(f"[INFO] Einstellungen in '{SETTINGS_FILE}' gespeichert.")
    except IOError as exc:
        print(f"[FEHLER] Konnte Einstellungen nicht speichern: {exc}", file=sys.stderr)
    return normalized


def log_transcription(log_data: Dict[str, Any]):
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(log_data) + "\n")
    except IOError as exc:
        print(f"[FEHLER] Konnte Log-Eintrag nicht schreiben: {exc}", file=sys.stderr)
