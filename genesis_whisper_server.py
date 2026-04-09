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

from genesis_whisper_server_admin import create_admin_api
from genesis_whisper_server_api import create_api, process_local_asr_batch
from genesis_whisper_server_batching import WhisperBatchManager
from genesis_whisper_server_globals import current_settings
from genesis_whisper_server_storage import load_settings

SCRIPT_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = SCRIPT_DIR / "frontend" / "dist"


async def startup_server(app: FastAPI):
    print("--- GENESIS Transcription Server wird gestartet (FastAPI + React + lokales ASR-Batching) ---", file=sys.stderr)

    current_settings.clear()
    current_settings.update(load_settings())
    print(
        f"Geladene Start-Konfiguration: "
        f"Lokales ASR-Modell='{current_settings['local_model']}', "
        f"Geraet='{current_settings['local_gpu_device']}', "
        f"Sprache='{current_settings['transcription_language']}', "
        f"Cache-Pfad='{current_settings['local_model_cache_path'] or 'Standard'}'",
        file=sys.stderr,
    )

    local_gpu_lock = asyncio.Lock()
    whisper_batch_manager = WhisperBatchManager(process_local_asr_batch, local_gpu_lock)
    app.state.local_gpu_lock = local_gpu_lock
    app.state.whisper_batch_manager = whisper_batch_manager
    await whisper_batch_manager.start()


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


@app.get("/")
async def serve_frontend_index():
    return _frontend_index_response()


@app.get("/{full_path:path}")
async def serve_frontend_assets(full_path: str):
    if full_path.startswith("api/"):
        return HTMLResponse(status_code=404, content="Not Found")

    asset_path = FRONTEND_DIST_DIR / full_path
    if asset_path.exists() and asset_path.is_file():
        return FileResponse(asset_path)
    return _frontend_index_response()


if __name__ == "__main__":
    print("\n--- Server ist bereit ---", file=sys.stderr)
    print("Oeffne das Admin-Dashboard in deinem Browser: http://127.0.0.1:7861/", file=sys.stderr)
    print("Der API-Endpunkt ist erreichbar unter: POST http://127.0.0.1:7861/transcribe/", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=7861)
