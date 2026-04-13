@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =================================================================
REM == Startskript fuer GENESIS 3 Whisper Server
REM == richtet bei Bedarf lokale virtuelle Umgebung .\venv selbst ein
REM =================================================================

cd /D "%~dp0"
chcp 65001 > nul
title G3 WHISPER Server Launcher

set "VENV_DIR=%CD%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "REQ_STAMP=%VENV_DIR%\.requirements_installed"
set "FRONTEND_DIR=%CD%\frontend"
set "FRONTEND_NODE_MODULES=%FRONTEND_DIR%\node_modules"
set "FRONTEND_NPM_STAMP=%FRONTEND_DIR%\.node_modules_installed"
set "NEEDS_SETUP=0"
set "NEEDS_PY_DEPS=0"
set "NEEDS_FRONTEND_DEPS=0"

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

if not exist "%FRONTEND_NODE_MODULES%" (
    set "NEEDS_FRONTEND_DEPS=1"
) else (
    powershell -NoProfile -Command "$stamp = '%FRONTEND_NPM_STAMP%'; if (-not (Test-Path $stamp)) { exit 1 }; $stampItem = Get-Item $stamp; $items = @(); foreach ($path in @('frontend\\package.json','frontend\\package-lock.json')) { if (Test-Path $path) { $items += Get-Item $path } }; if ($items | Where-Object { $_.LastWriteTime -gt $stampItem.LastWriteTime } | Select-Object -First 1) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEEDS_FRONTEND_DEPS=1"
)

if "%NEEDS_FRONTEND_DEPS%"=="1" (
    where npm > nul 2>&1
    if errorlevel 1 (
        echo FEHLER: npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren.
        pause
        exit /b 1
    )
    echo Frontend-Abhaengigkeiten fehlen oder sind veraltet. Fuehre npm install aus...
    pushd "%FRONTEND_DIR%"
    call npm install
    if errorlevel 1 (
        popd
        echo FEHLER: npm install fehlgeschlagen.
        pause
        exit /b 1
    )
    > "%FRONTEND_NPM_STAMP%" echo installed
    popd
)

set "NEEDS_FRONTEND_BUILD=0"
if not exist "%FRONTEND_DIR%\dist\index.html" (
    set "NEEDS_FRONTEND_BUILD=1"
) else (
    powershell -NoProfile -Command "$dist = Get-Item 'frontend\\dist\\index.html'; $items = @(); foreach ($path in @('frontend\\index.html','frontend\\package.json','frontend\\package-lock.json','frontend\\tsconfig.app.json','frontend\\tsconfig.json','frontend\\tsconfig.node.json','frontend\\vite.config.ts','frontend\\vite.config.js','frontend\\vite.config.d.ts')) { if (Test-Path $path) { $items += Get-Item $path } }; if (Test-Path 'frontend\\src') { $items += Get-ChildItem 'frontend\\src' -Recurse -File }; if ($items | Where-Object { $_.LastWriteTime -gt $dist.LastWriteTime } | Select-Object -First 1) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEEDS_FRONTEND_BUILD=1"
)

if "%NEEDS_FRONTEND_BUILD%"=="1" (
    where npm > nul 2>&1
    if errorlevel 1 (
        echo FEHLER: npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren.
        pause
        exit /b 1
    )
    echo Frontend-Build ist veraltet oder fehlt. Fuehre npm run build aus...
    pushd "%FRONTEND_DIR%"
    call npm run build
    if errorlevel 1 (
        popd
        echo FEHLER: Frontend-Build fehlgeschlagen.
        pause
        exit /b 1
    )
    popd
)

where ffmpeg > nul 2>&1
if errorlevel 1 (
    echo WARNUNG: ffmpeg wurde nicht gefunden. Einige Audio-/Videoformate koennen ohne ffmpeg nicht dekodiert werden.
)

if not defined GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS set "GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS=300"
if not defined GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS set "GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS=15"
set "GENESIS_STARTUP_ADMIN_KEY="
set "TMP_DIR=%CD%\.tmp"
set "STARTUP_ADMIN_KEY_FILE=%TMP_DIR%\startup_admin_key.txt"
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%" > nul 2>&1
del /q "%STARTUP_ADMIN_KEY_FILE%" > nul 2>&1
call "%VENV_PYTHON%" "%CD%\tools\generate_startup_admin_key.py" > "%STARTUP_ADMIN_KEY_FILE%" 2>nul
if exist "%STARTUP_ADMIN_KEY_FILE%" (
    set /p GENESIS_STARTUP_ADMIN_KEY=<"%STARTUP_ADMIN_KEY_FILE%"
    del /q "%STARTUP_ADMIN_KEY_FILE%" > nul 2>&1
)

if defined GENESIS_STARTUP_ADMIN_KEY (
    echo.
    echo ============================================================
    echo Temporary startup admin key ^(valid for %GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS% seconds after server start^):
    echo %GENESIS_STARTUP_ADMIN_KEY%
    echo Copy it now if you need emergency admin access in the browser.
    echo This screen clears automatically in %GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% seconds...
    echo ============================================================
    timeout /t %GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% /nobreak > nul
    cls
) else (
    echo WARNUNG: Temporarer Startup-Admin-Key konnte nicht erzeugt werden.
)

echo Nutze lokale venv unter "%CD%\venv" ...
echo Starte den GENESIS Whisper Server...
call "%VENV_PYTHON%" -m backend.genesis_whisper_server

echo.
echo =================================================================
echo == Der GENESIS Whisper Server wurde beendet.
echo == Das Fenster bleibt offen, damit du den Log lesen kannst.
echo =================================================================
pause
