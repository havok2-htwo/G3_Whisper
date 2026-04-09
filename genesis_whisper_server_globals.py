# genesis_whisper_server_globals.py
# Zentraler Ort fuer Konfigurationen, Konstanten und globale,
# thread-sichere Variablen fuer den lokalen Transkriptions-Server.

import os
import threading
from collections import deque
from typing import Any, Dict, List

import torch

# --- Pfade und Konstanten ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
SETTINGS_FILE = os.path.join(LOGS_DIR, "genesis_whisper_settings.json")
LOG_FILE = os.path.join(LOGS_DIR, "transcription_log.jsonl")
HISTORY_MAX_LEN = 100
BATCH_HISTORY_MAX_LEN = 50
COHERE_FALLBACK_LANGUAGE = "de"

LOCAL_ASR_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "Cohere Transcribe 03/2026": {
        "value": "CohereLabs/cohere-transcribe-03-2026",
        "backend": "cohere_transcribe",
        "default_language": COHERE_FALLBACK_LANGUAGE,
    },
    "Whisper Large v3 Turbo": {
        "value": "openai/whisper-large-v3-turbo",
        "backend": "whisper",
        "default_language": "auto",
    },
    "Whisper Large v3": {
        "value": "openai/whisper-large-v3",
        "backend": "whisper",
        "default_language": "auto",
    },
    "Whisper Medium": {
        "value": "openai/whisper-medium",
        "backend": "whisper",
        "default_language": "auto",
    },
    "Whisper Small": {
        "value": "openai/whisper-small",
        "backend": "whisper",
        "default_language": "auto",
    },
    "Whisper Base": {
        "value": "openai/whisper-base",
        "backend": "whisper",
        "default_language": "auto",
    },
    "Whisper Tiny": {
        "value": "openai/whisper-tiny",
        "backend": "whisper",
        "default_language": "auto",
    },
}
LOCAL_ASR_MODEL_MAP: Dict[str, str] = {
    label: spec["value"] for label, spec in LOCAL_ASR_MODEL_SPECS.items()
}

SUPPORTED_LANGUAGE_OPTIONS: Dict[str, str] = {
    "Auto detect (Whisper only)": "auto",
    "German (de)": "de",
    "English (en)": "en",
    "French (fr)": "fr",
    "Italian (it)": "it",
    "Spanish (es)": "es",
    "Portuguese (pt)": "pt",
    "Greek (el)": "el",
    "Dutch (nl)": "nl",
    "Polish (pl)": "pl",
    "Arabic (ar)": "ar",
    "Vietnamese (vi)": "vi",
    "Chinese / Mandarin (zh)": "zh",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
}
SUPPORTED_LANGUAGE_VALUES = set(SUPPORTED_LANGUAGE_OPTIONS.values())


def get_local_model_spec(model_id: str) -> Dict[str, str]:
    for spec in LOCAL_ASR_MODEL_SPECS.values():
        if spec["value"] == model_id:
            return spec
    return {"value": model_id, "backend": "whisper", "default_language": "auto"}


def get_local_model_backend(model_id: str) -> str:
    return get_local_model_spec(model_id)["backend"]


def uses_whisper_backend(model_id: str) -> bool:
    return get_local_model_backend(model_id) == "whisper"


def uses_cohere_backend(model_id: str) -> bool:
    return get_local_model_backend(model_id) == "cohere_transcribe"


def get_effective_transcription_language(model_id: str, configured_language: str) -> str:
    normalized_language = (configured_language or "auto").strip().lower()
    if uses_cohere_backend(model_id) and normalized_language == "auto":
        return COHERE_FALLBACK_LANGUAGE
    if normalized_language:
        return normalized_language
    return get_local_model_spec(model_id).get("default_language", "auto")


# --- GPU-Geraeteerkennung ---
AVAILABLE_DEVICES: List[str] = ["Auto (Empfohlen)", "CPU"]
DEVICE_MAP_UI_TO_INTERNAL: Dict[str, str] = {"Auto (Empfohlen)": "auto", "CPU": "cpu"}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = f"GPU {i} ({torch.cuda.get_device_name(i)})"
        AVAILABLE_DEVICES.append(device_name)
        DEVICE_MAP_UI_TO_INTERNAL[device_name] = f"cuda:{i}"

# --- Geteilter, thread-sicherer Zustand ---
settings_lock = threading.Lock()
model_load_lock = threading.Lock()
history_lock = threading.Lock()
batch_state_lock = threading.Lock()

current_settings: Dict[str, Any] = {}
local_model_components: Dict[str, Any] = {
    "model": None,
    "processor": None,
    "model_identifier": None,
    "model_backend": None,
}
transcription_history = deque(maxlen=HISTORY_MAX_LEN)
batch_history = deque(maxlen=BATCH_HISTORY_MAX_LEN)
batch_runtime_state: Dict[str, Any] = {
    "worker_running": False,
    "queue_size": 0,
    "pending_buffer_size": 0,
    "active_batch_id": None,
    "active_batch_size": 0,
    "active_batch_audio_seconds": 0.0,
    "active_batch_started_at": None,
    "last_batch_completed_at": None,
    "last_batch_duration_ms": None,
    "last_error": None,
    "total_batches_processed": 0,
    "total_segments_processed": 0,
}
