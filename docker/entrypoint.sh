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
    "batch_max_segments": 32,
    "batch_max_audio_seconds": 300.0,
    "huggingface_token": ""
}
JSON
fi

# Print a temporary startup admin key on boot (start.sh normally does this, but the
# container runs the module directly and skips it). When no admin key is configured,
# generate one and log it so you can reach the /admin dashboard, then set a persistent
# key there. Set GENESIS_ADMIN_KEY for a fixed key instead (which disables this).
if [ -z "${GENESIS_ADMIN_KEY:-}" ] && [ -z "${GENESIS_STARTUP_ADMIN_KEY:-}" ]; then
  key="$(python /app/tools/generate_startup_admin_key.py 2>/dev/null || true)"
  if [ -n "$key" ]; then
    GENESIS_STARTUP_ADMIN_KEY="$key"
    export GENESIS_STARTUP_ADMIN_KEY
    : "${GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS:=1800}"
    export GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS
    echo "============================================================" >&2
    echo "GENESIS Whisper - temporary startup admin key:" >&2
    echo "  ${GENESIS_STARTUP_ADMIN_KEY}" >&2
    echo "Valid ~${GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS}s after startup. Use it as the" >&2
    echo "X-Admin-Key at /admin, then set a persistent key there." >&2
    echo "(Set GENESIS_ADMIN_KEY for a fixed key and disable this.)" >&2
    echo "============================================================" >&2
  fi
fi

exec "$@"
