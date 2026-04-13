#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
TMP_DIR="$PROJECT_ROOT/.tmp"
STARTUP_ADMIN_KEY_FILE="$TMP_DIR/startup_admin_key.txt"

fail() {
  echo "FEHLER: $1" >&2
  exit 1
}

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  bash "$PROJECT_ROOT/install.sh"
fi

[[ -x "$VENV_PYTHON" ]] || fail "Die Linux-Installation ist noch nicht abgeschlossen. Bitte zuerst 'bash ./install.sh' ausfuehren."

export GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS="${GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS:-300}"
export GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS="${GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS:-15}"
export GENESIS_STARTUP_ADMIN_KEY=""

mkdir -p "$TMP_DIR"
rm -f "$STARTUP_ADMIN_KEY_FILE"
if "$VENV_PYTHON" "$PROJECT_ROOT/tools/generate_startup_admin_key.py" > "$STARTUP_ADMIN_KEY_FILE" 2>/dev/null; then
  if [[ -s "$STARTUP_ADMIN_KEY_FILE" ]]; then
    GENESIS_STARTUP_ADMIN_KEY="$(head -n 1 "$STARTUP_ADMIN_KEY_FILE")"
    export GENESIS_STARTUP_ADMIN_KEY
  fi
fi
rm -f "$STARTUP_ADMIN_KEY_FILE"

if [[ -n "$GENESIS_STARTUP_ADMIN_KEY" ]]; then
  echo
  echo "============================================================"
  echo "Temporary startup admin key (valid for $GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS seconds after server start):"
  echo "$GENESIS_STARTUP_ADMIN_KEY"
  echo "Copy it now if you need emergency admin access in the browser."
  echo "This screen clears automatically in $GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS seconds..."
  echo "============================================================"
  sleep "$GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS"
  if command -v clear >/dev/null 2>&1; then
    clear
  fi
else
  echo "WARNUNG: Temporarer Startup-Admin-Key konnte nicht erzeugt werden."
fi

echo "Nutze lokale venv unter \"$VENV_DIR\" ..."
echo "Starte den GENESIS Whisper Server..."
"$VENV_PYTHON" -m backend.genesis_whisper_server
