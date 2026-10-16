#!/bin/sh
set -e

# The GENESIS Whisper server's built-in default model cache path is the Windows
# literal ".\\models", which is meaningless on Linux. On first boot (before any
# settings file exists) seed the cache path of the mounted /app/models volume into
# the first-start seed file, NOT into the settings file: the server only treats a
# start without a settings file as a first start, applies the "24GB VRAM" GPU tier
# on top of the defaults, merges these seed values and saves the result. Writing
# the settings file here would hide the first start from the server. Existing
# settings (written by the server or the admin dashboard) are left untouched.

SETTINGS_DIR="/app/logs"
SETTINGS_FILE="${SETTINGS_DIR}/genesis_whisper_settings.json"
SEED_FILE="${SETTINGS_DIR}/genesis_whisper_settings.seed.json"

mkdir -p "${SETTINGS_DIR}" /app/models

if [ ! -f "${SETTINGS_FILE}" ]; then
  echo "[entrypoint] First start: seeding ${SEED_FILE} (cache -> /app/models)"
  cat > "${SEED_FILE}" <<'JSON'
{
    "local_model_cache_path": "/app/models"
}
JSON
fi

# Admin access is username/password (default admin/admin, forced change on first login),
# so no startup admin key is generated here anymore.
exec "$@"
