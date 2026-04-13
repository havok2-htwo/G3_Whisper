#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
REQ_STAMP="$VENV_DIR/.requirements_installed"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_NODE_MODULES="$FRONTEND_DIR/node_modules"
FRONTEND_NPM_STAMP="$FRONTEND_DIR/.node_modules_installed"

NEEDS_SETUP=0
NEEDS_PY_DEPS=0
NEEDS_FRONTEND_DEPS=0
NEEDS_FRONTEND_BUILD=0

fail() {
  echo "FEHLER: $1" >&2
  exit 1
}

choose_python() {
  local candidate
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

touch_stamp() {
  printf 'installed\n' > "$1"
}

install_torch_stack() {
  local torch_index_url

  echo "Installiere PyTorch-Stack ..."
  if [[ -n "${TORCH_WHEEL_DIR:-}" ]]; then
    echo "Verwende lokale Wheels aus \"$TORCH_WHEEL_DIR\"."
    "$VENV_PIP" install \
      "$TORCH_WHEEL_DIR"/torch-*.whl \
      "$TORCH_WHEEL_DIR"/torchvision-*.whl \
      "$TORCH_WHEEL_DIR"/torchaudio-*.whl
    return
  fi

  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo "Verwende benutzerdefinierten PyTorch-Index: $TORCH_INDEX_URL"
    "$VENV_PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    return
  fi

  case "$(uname -s)" in
    Darwin)
      echo "Verwende PyTorch-Wheels von PyPI."
      "$VENV_PIP" install torch torchvision torchaudio
      ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1; then
        torch_index_url="https://download.pytorch.org/whl/cu128"
        echo "NVIDIA-GPU erkannt. Verwende PyTorch CUDA 12.8 Wheels von download.pytorch.org."
        "$VENV_PIP" install torch torchvision torchaudio --index-url "$torch_index_url"
      else
        echo "Keine NVIDIA-GPU erkannt. Verwende PyTorch-Wheels von PyPI."
        "$VENV_PIP" install torch torchvision torchaudio
      fi
      ;;
    *)
      echo "Unbekanntes System. Verwende PyTorch-Wheels von PyPI."
      "$VENV_PIP" install torch torchvision torchaudio
      ;;
  esac
}

if [[ ! -x "$VENV_PYTHON" ]]; then
  NEEDS_SETUP=1
  NEEDS_PY_DEPS=1
fi

if [[ "$NEEDS_SETUP" -eq 1 ]]; then
  PYTHON_BOOTSTRAP="$(choose_python)" || fail "Kein Python-Interpreter gefunden."
  echo "Lokale venv nicht gefunden. Setup wird gestartet..."
  echo "Erstelle virtuelle Umgebung in \"$VENV_DIR\" ..."
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR" || fail "Konnte keine lokale venv erstellen."

  echo "Aktualisiere pip, setuptools und wheel ..."
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel || fail "pip-Update fehlgeschlagen."

  install_torch_stack || fail "PyTorch-Installation fehlgeschlagen."
fi

if [[ -x "$VENV_PYTHON" && "$NEEDS_PY_DEPS" -eq 0 ]]; then
  if [[ ! -f "$REQ_STAMP" || requirements.txt -nt "$REQ_STAMP" ]]; then
    NEEDS_PY_DEPS=1
  fi
fi

if [[ "$NEEDS_PY_DEPS" -eq 1 ]]; then
  echo "Installiere/Aktualisiere Python-Abhaengigkeiten ..."
  "$VENV_PIP" install -r requirements.txt || fail "Installation aus requirements.txt fehlgeschlagen."
  touch_stamp "$REQ_STAMP"
fi

if [[ ! -d "$FRONTEND_NODE_MODULES" ]]; then
  NEEDS_FRONTEND_DEPS=1
elif [[ ! -f "$FRONTEND_NPM_STAMP" || "$FRONTEND_DIR/package.json" -nt "$FRONTEND_NPM_STAMP" || "$FRONTEND_DIR/package-lock.json" -nt "$FRONTEND_NPM_STAMP" ]]; then
  NEEDS_FRONTEND_DEPS=1
fi

if [[ "$NEEDS_FRONTEND_DEPS" -eq 1 ]]; then
  command -v npm >/dev/null 2>&1 || fail "npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren."
  echo "Frontend-Abhaengigkeiten fehlen oder sind veraltet. Fuehre npm install aus..."
  (
    cd "$FRONTEND_DIR"
    npm install
  ) || fail "npm install fehlgeschlagen."
  touch_stamp "$FRONTEND_NPM_STAMP"
fi

DIST_INDEX="$FRONTEND_DIR/dist/index.html"
if [[ ! -f "$DIST_INDEX" ]]; then
  NEEDS_FRONTEND_BUILD=1
else
  while IFS= read -r path; do
    if [[ -e "$path" && "$path" -nt "$DIST_INDEX" ]]; then
      NEEDS_FRONTEND_BUILD=1
      break
    fi
  done <<'EOF'
frontend/index.html
frontend/package.json
frontend/package-lock.json
frontend/tsconfig.app.json
frontend/tsconfig.json
frontend/tsconfig.node.json
frontend/vite.config.ts
frontend/vite.config.js
frontend/vite.config.d.ts
EOF

  if [[ "$NEEDS_FRONTEND_BUILD" -eq 0 && -d "$FRONTEND_DIR/src" ]]; then
    if [[ -n "$(find "$FRONTEND_DIR/src" -type f -newer "$DIST_INDEX" -print -quit 2>/dev/null)" ]]; then
      NEEDS_FRONTEND_BUILD=1
    fi
  fi
fi

if [[ "$NEEDS_FRONTEND_BUILD" -eq 1 ]]; then
  command -v npm >/dev/null 2>&1 || fail "npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren."
  echo "Frontend-Build ist veraltet oder fehlt. Fuehre npm run build aus..."
  (
    cd "$FRONTEND_DIR"
    npm run build
  ) || fail "Frontend-Build fehlgeschlagen."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNUNG: ffmpeg wurde nicht gefunden. Einige Audio-/Videoformate koennen ohne ffmpeg nicht dekodiert werden."
fi

echo "Linux-Setup abgeschlossen."
echo "Starten mit: bash ./start.sh"
