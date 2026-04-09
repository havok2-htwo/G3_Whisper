@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =================================================================
REM == Startskript fuer GENESIS Whisper Server
REM == richtet bei Bedarf lokale virtuelle Umgebung .\venv selbst ein
REM =================================================================

cd /D "%~dp0"
chcp 65001 > nul

set "VENV_DIR=%CD%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "REQ_STAMP=%VENV_DIR%\.requirements_installed"
set "NEEDS_SETUP=0"
set "NEEDS_PY_DEPS=0"

if not exist "%VENV_PYTHON%" (
    set "NEEDS_SETUP=1"
    set "NEEDS_PY_DEPS=1"
)

if "%NEEDS_SETUP%"=="1" (
    echo Lokale venv nicht gefunden. Setup wird gestartet...
    echo Erstelle virtuelle Umgebung in "%VENV_DIR%" ...
    py -3.10 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 (
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo FEHLER: Konnte keine lokale venv erstellen.
        pause
        exit /b 1
    )

    echo Aktualisiere pip, setuptools und wheel ...
    call "%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 (
        echo FEHLER: pip-Update fehlgeschlagen.
        pause
        exit /b 1
    )

    echo Installiere PyTorch-Stack ...
    if defined TORCH_WHEEL_DIR (
        echo Verwende lokale Wheels aus "%TORCH_WHEEL_DIR%".
        call "%VENV_PIP%" install "%TORCH_WHEEL_DIR%\torch-*.whl" "%TORCH_WHEEL_DIR%\torchvision-*.whl" "%TORCH_WHEEL_DIR%\torchaudio-*.whl"
    ) else (
        echo Verwende PyTorch CUDA 12.8 Wheels von download.pytorch.org.
        call "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    )
    if errorlevel 1 (
        echo FEHLER: PyTorch-Installation fehlgeschlagen.
        pause
        exit /b 1
    )
)

if exist "%VENV_PYTHON%" if "%NEEDS_PY_DEPS%"=="0" (
    call "%VENV_PYTHON%" -c "import os, sys; req='requirements.txt'; stamp=r'%REQ_STAMP%'; sys.exit(0 if os.path.exists(stamp) and os.path.getmtime(stamp) >= os.path.getmtime(req) else 1)"
    if errorlevel 1 (
        set "NEEDS_PY_DEPS=1"
    )
)

if "%NEEDS_PY_DEPS%"=="1" (
    echo Installiere/Aktualisiere Python-Abhaengigkeiten ...
    call "%VENV_PIP%" install -r requirements.txt
    if errorlevel 1 (
        echo FEHLER: Installation aus requirements.txt fehlgeschlagen.
        pause
        exit /b 1
    )
    call "%VENV_PYTHON%" -c "from pathlib import Path; Path(r'%REQ_STAMP%').write_text('ok', encoding='utf-8')"
    if errorlevel 1 (
        echo FEHLER: Konnte Dependency-Status nicht speichern.
        pause
        exit /b 1
    )
)

if not exist "%CD%\frontend\node_modules" (
    echo Frontend-Abhaengigkeiten fehlen. Fuehre npm install aus...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo FEHLER: npm install fehlgeschlagen.
        pause
        exit /b 1
    )
    popd
)

if not exist "%CD%\frontend\dist\index.html" (
    echo Frontend-Build fehlt. Fuehre npm run build aus...
    pushd frontend
    call npm run build
    if errorlevel 1 (
        popd
        echo FEHLER: Frontend-Build fehlgeschlagen.
        pause
        exit /b 1
    )
    popd
)

echo Nutze lokale venv unter "%CD%\venv" ...
echo Starte den GENESIS Whisper Server...
call "%VENV_PYTHON%" genesis_whisper_server.py

echo.
echo =================================================================
echo == Der GENESIS Whisper Server wurde beendet.
echo == Das Fenster bleibt offen, damit du den Log lesen kannst.
echo =================================================================
pause
