#!/bin/sh
set -e

# The GENESIS Whisper server's built-in default model cache path is the Windows
# literal ".\\models", which is meaningless on Linux. On first boot (before any
# settings file exists) seed one that pins the cache to the mounted /app/models
# volume. Existing settings (written by the admin dashboard) are left untouched.

SETTINGS_DIR="/app/logs"
SETTINGS_FILE="${SETTINGS_DIR}/genesis_whisper_settings.json"

mkdir -p "${SETTINGS_DIR}" /app/models

if [ ! -f "${SETTINGS_FILE}" ]; then
  echo "[entrypoint] Seeding ${SETTINGS_FILE} (cache -> /app/models)"
  cat > "${SETTINGS_FILE}" <<'JSON'
{
    "local_model": "openai/whisper-large-v3-turbo",
    "local_gpu_device": "auto",
    "local_model_precision": "fp16",
    "local_model_cache_path": "/app/models",
    "transcription_language": "auto",
    "batch_wait_time_ms": 500,
    "batch_max_segments": 16,
    "batch_max_audio_seconds": 300.0,
    "cuda_memory_trim_after_batch": false,
    "debug_retain_history_audio": false,
    "huggingface_token": ""
}
JSON
fi

# Admin access is username/password (default admin/admin, forced change on first login),
# so no startup admin key is generated here anymore.
exec "$@"
