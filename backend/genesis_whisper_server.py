# genesis_whisper_server.py
# Haupt-Startdatei fuer den GENESIS Whisper Server.

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from .genesis_whisper_server_admin import create_admin_api
from .genesis_whisper_server_api import create_api, process_local_asr_batch
from .genesis_whisper_server_batching import WhisperBatchManager
from .genesis_whisper_server_globals import current_settings
from .genesis_whisper_server_storage import load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
WARMUP_SAMPLE = PROJECT_ROOT / "testaudio" / "Testaudio_02.wav"


def _warmup_local_asr() -> None:
    """Eager-load the configured ASR model and run one transcription on a sample clip so
    the first real request is already warm (model + cuDNN/compile kernels primed), then
    release the reserved CUDA cache pool so idle VRAM stays low. Best-effort: a warmup
    failure (e.g. an unsupported precision) must never block server startup."""
    import gc

    try:
        from .genesis_whisper_server_audio import load_audio_bytes
        from .genesis_whisper_server_local_asr_engine import load_local_asr_model, transcribe_local_asr

        model_id = current_settings.get("local_model")
        if not model_id:
            print("[WARMUP] kein lokales ASR-Modell konfiguriert -> uebersprungen.", file=sys.stderr)
            return
        if not WARMUP_SAMPLE.is_file():
            print(f"[WARMUP] Testaudio nicht gefunden: {WARMUP_SAMPLE} -> uebersprungen.", file=sys.stderr)
            return

        device = current_settings.get("local_gpu_device")
        cache = current_settings.get("local_model_cache_path")
        precision = current_settings.get("local_model_precision", "fp16")
        language = current_settings.get("transcription_language") or "auto"

        print(f"[WARMUP] Lade Modell '{model_id}' (precision={precision}) und waerme mit {WARMUP_SAMPLE.name} ...", file=sys.stderr)
        if not load_local_asr_model(model_id, device, cache, precision):
            print("[WARMUP] Modell-Load fehlgeschlagen -> Warmup uebersprungen.", file=sys.stderr)
            return

        audio = load_audio_bytes(WARMUP_SAMPLE.read_bytes(), WARMUP_SAMPLE.name)
        text = transcribe_local_asr(audio, language=language)
        print(f"[WARMUP] fertig. Beispiel-Transkript: {str(text)[:80]!r}", file=sys.stderr)
    except Exception as exc:
        print(f"[WARMUP] nicht-kritischer Warmup-Fehler ({type(exc).__name__}: {exc}).", file=sys.stderr)
    finally:
        # Release the reserved CUDA pool the warmup just sized (keeps idle VRAM low).
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass


async def startup_server(app: FastAPI):
    print("--- GENESIS Transcription Server wird gestartet (FastAPI + React + lokales ASR-Batching) ---", file=sys.stderr)

    current_settings.clear()
    current_settings.update(load_settings())
    print(
        f"Geladene Start-Konfiguration: "
        f"Lokales ASR-Modell='{current_settings['local_model']}', "
        f"Geraet='{current_settings['local_gpu_device']}', "
        f"Precision='{current_settings['local_model_precision']}', "
        f"Sprache='{current_settings['transcription_language']}', "
        f"Cache-Pfad='{current_settings['local_model_cache_path'] or 'Standard'}'",
        file=sys.stderr,
    )

    local_gpu_lock = asyncio.Lock()
    whisper_batch_manager = WhisperBatchManager(process_local_asr_batch, local_gpu_lock)
    app.state.local_gpu_lock = local_gpu_lock
    app.state.whisper_batch_manager = whisper_batch_manager
    await whisper_batch_manager.start()

    # Eager-load + warm the ASR model on a sample clip so the first real request is fast,
    # then trim the CUDA cache. Best-effort: a warmup failure must not block startup.
    await asyncio.to_thread(_warmup_local_asr)


async def shutdown_server(app: FastAPI):
    batch_manager = getattr(app.state, "whisper_batch_manager", None)
    if batch_manager is not None:
        await batch_manager.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_server(app)
    try:
        yield
    finally:
        await shutdown_server(app)


app = FastAPI(title="GENESIS Transcription Server API", lifespan=lifespan)
app = create_api(app)
app = create_admin_api(app)


def _frontend_index_response() -> Union[HTMLResponse, FileResponse]:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return HTMLResponse(
        """
        <html>
          <head><title>GENESIS Transcription Server</title></head>
          <body style="font-family: sans-serif; padding: 32px;">
            <h1>Frontend-Build fehlt</h1>
            <p>Bitte im Ordner <code>frontend</code> erst <code>npm install</code> und danach <code>npm run build</code> ausfuehren.</p>
          </body>
        </html>
        """,
        status_code=503,
    )


def _landing_response() -> HTMLResponse:
    return HTMLResponse(
        """
        <html lang="de">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>GENESIS Whisper Server</title>
            <style>
              :root {
                color-scheme: light;
                font-family: "Bahnschrift", "Segoe UI Variable", "Trebuchet MS", sans-serif;
              }
              body {
                margin: 0;
                min-height: 100vh;
                color: #ebf0df;
                background:
                  radial-gradient(circle at top left, rgba(255, 194, 117, 0.24), transparent 28%),
                  radial-gradient(circle at top right, rgba(139, 230, 192, 0.22), transparent 24%),
                  linear-gradient(145deg, #182028 0%, #10161c 45%, #17131c 100%);
              }
              main {
                width: min(980px, calc(100vw - 40px));
                margin: 0 auto;
                padding: 56px 0;
              }
              .panel {
                background: rgba(15, 23, 29, 0.82);
                border: 1px solid rgba(255, 248, 224, 0.12);
                border-radius: 28px;
                padding: 28px;
                box-shadow: 0 28px 60px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(18px);
              }
              .eyebrow {
                display: inline-block;
                margin-bottom: 0.8rem;
                color: #95f2c7;
                font-size: 0.75rem;
                font-weight: 800;
                letter-spacing: 0.18em;
                text-transform: uppercase;
              }
              h1 {
                margin: 0 0 1rem;
                font-family: "Rockwell", "Bahnschrift", serif;
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 1.02;
              }
              p {
                color: #a9b2b8;
                line-height: 1.6;
              }
              code {
                color: #f8e7c3;
              }
              .button-row {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 24px;
              }
              a {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 220px;
                border-radius: 999px;
                text-decoration: none;
                padding: 0.9rem 1.25rem;
                font-weight: 700;
                letter-spacing: 0.02em;
              }
              .primary {
                background: linear-gradient(135deg, #ffc16c, #ff8b47);
                color: #1b1510;
                box-shadow: 0 12px 28px rgba(255, 139, 71, 0.25);
              }
              .secondary {
                color: #f4f6ee;
                border: 1px solid rgba(255, 255, 255, 0.12);
                background: rgba(255, 255, 255, 0.06);
              }
            </style>
          </head>
          <body>
            <main>
              <section class="panel">
                <span class="eyebrow">GENESIS Transcription Server</span>
                <h1>Lokale Transkription offen, Admin-Dashboard geschuetzt.</h1>
                <p>
                  Die oeffentliche Upload-API unter <code>POST /transcribe/</code> ist ohne Key erreichbar,
                  solange keine API-Keys angelegt sind. Sobald im Admin-Dashboard mindestens ein API-Key
                  existiert, wird ein <code>X-API-Key</code>-Header verlangt.
                </p>
                <p>
                  Das Admin-Dashboard unter <code>/admin</code> ist per Login (Benutzername/Passwort) geschuetzt.
                  Standardzugang <code>admin</code> / <code>admin</code> &ndash; das Passwort muss beim ersten
                  Login geaendert werden.
                </p>
                <div class="button-row">
                  <a class="primary" href="/admin">Admin-Dashboard oeffnen</a>
                  <a class="secondary" href="/docs">OpenAPI / Docs</a>
                </div>
              </section>
            </main>
          </body>
        </html>
        """,
        status_code=200,
    )


@app.get("/")
async def serve_landing():
    return _landing_response()


@app.get("/admin")
@app.get("/admin/")
async def serve_frontend_index():
    return _frontend_index_response()


@app.get("/{full_path:path}")
async def serve_frontend_assets(full_path: str):
    if full_path.startswith("api/"):
        return HTMLResponse(status_code=404, content="Not Found")

    if full_path.startswith("admin/"):
        return _frontend_index_response()

    asset_path = FRONTEND_DIST_DIR / full_path
    if asset_path.exists() and asset_path.is_file():
        return FileResponse(asset_path)
    return _landing_response()


if __name__ == "__main__":
    print("\n--- Server ist bereit ---", file=sys.stderr)
    print("Oeffne die Startseite in deinem Browser: http://127.0.0.1:7861/", file=sys.stderr)
    print("Das Admin-Dashboard ist erreichbar unter: http://127.0.0.1:7861/admin", file=sys.stderr)
    print("Der API-Endpunkt ist erreichbar unter: POST http://127.0.0.1:7861/transcribe/", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=7861)
