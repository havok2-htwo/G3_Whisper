"""Named settings presets: built-in GPU tiers plus presets the operator saves from the admin dashboard.

Same model as G3_OmniVoice and G3_RERANK: a preset is a bag of settings values. Loading one fills the settings form;
the operator reviews it and saves through the normal validated PUT /api/admin/settings, so a preset never bypasses
validation and never reloads the model by itself.

Built-in tiers (one name per target machine, identical across the G3 services):
- "32GB VRAM"  RTX 5090, every G3 service on one card, whole-card peak <= 30 GB.
- "24GB VRAM"  RTX 4090, every G3 service on one card, whole-card peak <= 23 GB.
- "24+8 VRAM"  3090 (24 GB) + 2070S (8 GB) box, this service on the 24 GB card.
The number in front names the card the service runs on. Whisper always runs on the 24 GB card of the two-GPU box
(DIA alone takes the 8 GB card), so there is no "8+24 VRAM" tier here. Presets with a target card size pick
local_gpu_device when the list is built: the CUDA device whose memory matches (machines number their cards
differently), see resolve_presets.

User presets live in logs/genesis_whisper_presets.json next to the settings file; they cannot reuse a built-in name.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import genesis_whisper_server_globals as whisper_globals
from . import genesis_whisper_server_storage as storage
from .genesis_whisper_server_globals import LOGS_DIR
from .genesis_whisper_server_storage import normalize_settings

PRESETS_FILE = os.path.join(LOGS_DIR, "genesis_whisper_presets.json")
# A fresh install (no saved settings yet) starts with this tier; the same in every G3 service. It is a single-card
# tier, so it never sets local_gpu_device; the two-card tier is never applied automatically.
FIRST_START_PRESET = "24GB VRAM"
PRESET_NAME_MAX_LENGTH = 64

# Keys a preset may set: model choice, device, precision, batching, scheduler and VRAM behaviour. Machine-specific
# values (local_model_cache_path), secrets (huggingface_token, dia_api_key), the DIA URL, the transcription language
# and the debug audio switch stay out, so loading a preset never touches them.
PRESET_KEYS: Tuple[str, ...] = (
    "local_model",
    "local_gpu_device",
    "local_model_precision",
    "batch_wait_time_ms",
    "batch_max_segments",
    "batch_max_audio_seconds",
    "scheduler_long_job_min_chunks",
    "scheduler_max_parallel_long_jobs",
    "scheduler_first_chunk_fast_path",
    "cuda_memory_trim_after_batch",
    "cohere_cuda_graphs",
    "redimnet_max_batch_size",
)

# Tolerance when matching a preset's card size to a detected card (a "24 GB" card reports ~23.6-24.0 GiB).
CARD_MATCH_TOLERANCE_MIB = 2048

# =====================================================================================================================
# BUILT-IN TIER VALUES
# Measured on the RTX 5090 (2026-10-16, in-process, Cohere Transcribe fp16 + ReDimNet2-B6 + MMS_FA), see
# README.md "GPU Tier Presets" for the per-point numbers.
# - batch_max_audio_seconds 400: 60-min transcription 393x real time vs 283x at the default 100 (200: 336x, 800: 398x,
#   batches then hit the 16-segment cap); ASR peak +0 MB (6.56 GB at 200 s, 6.58 GB at 400 s). The cost is latency: a
#   live request can wait about 0.9 s behind one bulk batch instead of about 0.37 s.
# - cohere_cuda_graphs on: about +1.0 GB idle and peak for 2.8x faster 60-min ASR and about 10x lower clip latency.
# - fp16: bf16 measured the same VRAM and speed within 2-3 %; fp16 is the measured default.
# - redimnet_max_batch_size: the largest single VRAM item after the ASR weights. 8 reserves 2816 MiB, 4 1442 MiB; a
#   60-min diarization takes 19.6 s at 8 vs 23.2 s at 4. Peak 9.7 GB at 8, 8.3 GB at 4.
# - cuda_memory_trim_after_batch on: idle drops back to about 6.3 GB 5 s after the last GPU phase.
# batch_wait_time_ms stays out: the 500 ms default adds up to 0.5 s to every fresh live request, a lower value was not
# measured yet (see CHANGELOG).
# local_gpu_device is never listed here: single-card tiers leave it alone, the two-card tier gets it from card matching.
# Whisper does not fit the 8 GB 2070S (weights 4.4 + graphs 1.0 + ReDimNet + MMS_FA + context > 9 GB, no bf16), so
# there is no "8+24 VRAM" tier.
# =====================================================================================================================
# The model, batch_wait_time_ms and the scheduler keys stay as the operator set them (measured with their defaults).
_TIER_COMMON: Dict[str, Any] = {
    "local_model_precision": "fp16",
    "cohere_cuda_graphs": True,
    "batch_max_segments": 16,
    "batch_max_audio_seconds": 400.0,
    "cuda_memory_trim_after_batch": True,
}
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "32GB VRAM": {
        "description": "RTX 5090, all G3 services on one card (whole-card peak <= 30 GB). Cohere fp16 with CUDA graphs, "
        "400 s ASR batches, ReDimNet batch 8: about 6.3 GB idle, 9.7 GB peak for this process (measured on the RTX 5090).",
        "target_vram_gb": None,
        "values": {**_TIER_COMMON, "redimnet_max_batch_size": 8},
    },
    "24GB VRAM": {
        "description": "RTX 4090, all G3 services on one card (whole-card peak <= 23 GB). Cohere fp16 with CUDA graphs, "
        "400 s ASR batches, ReDimNet batch 4: about 6.3 GB idle, 8.3 GB peak for this process (measured on the RTX 5090).",
        "target_vram_gb": None,
        "values": {**_TIER_COMMON, "redimnet_max_batch_size": 4},
    },
    "24+8 VRAM": {
        "description": "3090 + 2070S box, Whisper on the 3090 (24 GB) with OmniVoice, EMBED and RERANK; DIA runs alone "
        "on the 2070S. Same values as 24GB VRAM (ReDimNet batch 4): about 6.3 GB idle, 8.3 GB peak for this process "
        "(measured on the RTX 5090).",
        "target_vram_gb": 24,
        "values": {**_TIER_COMMON, "redimnet_max_batch_size": 4},
    },
}

_lock = threading.RLock()


def _atomic_write_text(path: str, text: str) -> None:
    """Write via a temp file in the same directory and os.replace, so a crash never leaves a half-written file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".presets-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            file_obj.write(text)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _clean_name(name: str) -> str:
    clean = " ".join(str(name or "").split())
    if not clean:
        raise ValueError("Preset name is required.")
    if len(clean) > PRESET_NAME_MAX_LENGTH:
        raise ValueError(f"Preset names have at most {PRESET_NAME_MAX_LENGTH} characters.")
    return clean


def preset_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Only PRESET_KEYS, each normalised exactly like a saved setting (unknown keys are dropped)."""
    sent = {key: values[key] for key in PRESET_KEYS if key in (values or {})}
    normalized = normalize_settings(sent)
    return {key: normalized[key] for key in sent}


def _load_user_presets() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[FEHLER] Konnte Presets nicht laden: {exc}", file=sys.stderr)
        return {}
    presets: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("values"), dict) and name not in BUILTIN_PRESETS:
                presets[str(name)] = {
                    "name": str(name),
                    "values": preset_values(entry["values"]),
                    "created_at": entry.get("created_at"),
                }
    return presets


def _write_user_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    _atomic_write_text(PRESETS_FILE, json.dumps(presets, indent=2, ensure_ascii=True))


def _cuda_cards_mib() -> List[int]:
    """Total memory of each CUDA card in MiB, index = cuda:N (read once at startup, no GPU work here)."""
    return list(whisper_globals.CUDA_DEVICE_TOTAL_MIB)


def _match_card(target_gb: Optional[int], cards_mib: List[int]) -> Tuple[Optional[str], Optional[str]]:
    """local_gpu_device for a preset's card size on this machine, plus a note when no card matches."""
    if not target_gb:
        return None, None
    target_mib = target_gb * 1024
    for index, total_mib in enumerate(cards_mib):
        if abs(total_mib - target_mib) <= CARD_MATCH_TOLERANCE_MIB:
            return f"cuda:{index}", None
    detected = ", ".join(f"GPU {index}: {total / 1024:.0f} GB" for index, total in enumerate(cards_mib)) or "none"
    return None, f"No {target_gb} GB card detected (cards: {detected}); the GPU setting is left unchanged."


def resolve_presets() -> List[Dict[str, Any]]:
    """Built-ins first (in tier order), then user presets by name. Built-ins with a card size get local_gpu_device."""
    cards_mib = _cuda_cards_mib()
    result: List[Dict[str, Any]] = []
    for name, preset in BUILTIN_PRESETS.items():
        values = dict(preset["values"])
        device, note = _match_card(preset.get("target_vram_gb"), cards_mib)
        if device:
            values["local_gpu_device"] = device
        result.append(
            {
                "name": name,
                "built_in": True,
                "description": preset.get("description", ""),
                "target_vram_gb": preset.get("target_vram_gb"),
                "note": note,
                "values": values,
            }
        )
    with _lock:
        user = _load_user_presets()
    for name in sorted(user, key=str.lower):
        entry = user[name]
        result.append(
            {
                "name": name,
                "built_in": False,
                "description": "",
                "target_vram_gb": None,
                "created_at": entry.get("created_at"),
                "note": None,
                "values": entry["values"],
            }
        )
    return result


def save_user_preset(name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    clean = _clean_name(name)
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset; save under another name.")
    cleaned = preset_values(values)
    if not cleaned:
        raise ValueError("The preset contains no settings.")
    with _lock:
        presets = _load_user_presets()
        created = (presets.get(clean) or {}).get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        presets[clean] = {"name": clean, "values": cleaned, "created_at": created}
        _write_user_presets(presets)
    return {
        "name": clean,
        "built_in": False,
        "description": "",
        "target_vram_gb": None,
        "created_at": created,
        "note": None,
        "values": cleaned,
    }


def delete_user_preset(name: str) -> bool:
    clean = " ".join(str(name or "").split())
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset and cannot be deleted.")
    with _lock:
        presets = _load_user_presets()
        if clean not in presets:
            return False
        presets.pop(clean)
        _write_user_presets(presets)
    return True


def load_settings_or_first_start_preset() -> Tuple[Dict[str, Any], bool]:
    """The saved settings; without a settings file (first start) the defaults with FIRST_START_PRESET applied, then
    the machine-specific seed values (docker: the model cache path), saved so the dashboard shows what runs.
    Returns (settings, applied)."""
    if os.path.exists(storage.SETTINGS_FILE):
        return storage.load_settings(), False
    values = dict(BUILTIN_PRESETS[FIRST_START_PRESET]["values"])
    return storage.save_settings({**storage.DEFAULT_SETTINGS, **values, **storage.load_settings_seed()}), True
