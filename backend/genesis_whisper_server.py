# genesis_whisper_server.py
# Entry point of the G3 Whisper server.

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .genesis_whisper_server_admin import create_admin_api
from .genesis_whisper_server_api import create_api, process_local_asr_batch
from .genesis_whisper_server_batching import WhisperBatchManager
from .genesis_whisper_server_globals import current_settings
from .genesis_whisper_server_history import reset_history_audio_store, shutdown_history_audio_store
from .genesis_whisper_server_gpu import run_blocking_gpu_phase
from .genesis_whisper_server_presets import FIRST_START_PRESET, load_settings_or_first_start_preset
from .genesis_whisper_server_v2 import create_v2_api, create_v2_capabilities_api

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
WARMUP_SAMPLE = PROJECT_ROOT / "testaudio" / "Testaudio_02.wav"
# Without the sample file (testaudio/ is not shipped everywhere) the warmup runs on
# a synthetic clip: a few seconds of quiet noise plus a low tone. The transcript is
# irrelevant; what matters is that model load, Cohere graph capture and the first
# inference kernels happen at startup instead of in the first real request.
WARMUP_SYNTHETIC_SECONDS = 4.0
WARMUP_SAMPLE_RATE = 16000


def _synthetic_warmup_audio(seconds: float = WARMUP_SYNTHETIC_SECONDS, sample_rate: int = WARMUP_SAMPLE_RATE):
    """Deterministic low-level noise + 220 Hz tone, float32 mono in [-1, 1] at the model rate."""
    import numpy as np

    samples = max(1, int(seconds * sample_rate))
    rng = np.random.default_rng(0)
    timeline = np.arange(samples, dtype=np.float32) / float(sample_rate)
    tone = 0.05 * np.sin(2.0 * np.pi * 220.0 * timeline)
    noise = 0.005 * rng.standard_normal(samples)
    return (tone + noise).astype(np.float32)


def _warmup_audio():
    """(audio, label): the repo sample when present, otherwise the synthetic clip."""

    if WARMUP_SAMPLE.is_file():
        from .genesis_whisper_server_audio import load_audio_bytes

        return load_audio_bytes(WARMUP_SAMPLE.read_bytes(), WARMUP_SAMPLE.name), WARMUP_SAMPLE.name
    return _synthetic_warmup_audio(), f"synthetischem {WARMUP_SYNTHETIC_SECONDS:.0f}-s-Clip (kein {WARMUP_SAMPLE.name})"


def _warmup_local_asr() -> None:
    """Eager-load the configured ASR model and run one transcription on a warmup clip so
    the first real request is already warm (model + cuDNN/compile kernels primed), then
    release the reserved CUDA cache pool so idle VRAM stays low. Best-effort: a warmup
    failure (e.g. an unsupported precision) must never block server startup."""
    import gc

    try:
        from .genesis_whisper_server_gpu import shared_gpu_lease
        from .genesis_whisper_server_local_asr_engine import (
            get_cohere_inference_patch_stats,
            load_local_asr_model,
            transcribe_local_asr,
        )

        model_id = current_settings.get("local_model")
        if not model_id:
            print("[WARMUP] kein lokales ASR-Modell konfiguriert -> uebersprungen.", file=sys.stderr)
            return
        device = current_settings.get("local_gpu_device")
        cache = current_settings.get("local_model_cache_path")
        precision = current_settings.get("local_model_precision", "fp16")
        language = current_settings.get("transcription_language") or "auto"
        audio, audio_label = _warmup_audio()

        print(f"[WARMUP] Lade Modell '{model_id}' (precision={precision}) und waerme mit {audio_label} ...", file=sys.stderr)
        with shared_gpu_lease():
            if not load_local_asr_model(model_id, device, cache, precision):
                print("[WARMUP] Modell-Load fehlgeschlagen -> Warmup uebersprungen.", file=sys.stderr)
                return

            text = transcribe_local_asr(audio, language=language)
        print(f"[WARMUP] fertig. Beispiel-Transkript: {str(text)[:80]!r}", file=sys.stderr)
        # Die CUDA-Graphen (Decoder B=1/2/4, Encoder B=1 alle Frame-Buckets) werden bereits in
        # load_local_asr_model auf diesem GPU-Thread gecaptured; hier nur der Status.
        graph_stats = get_cohere_inference_patch_stats()
        if graph_stats and graph_stats.get("graphs"):
            print(
                f"[WARMUP] Cohere CUDA-Graphs: {graph_stats['graphs']} Graphen, "
                f"{graph_stats['graph_memory_mb']} MB, Replays Encoder/Decoder "
                f"{graph_stats['encoder_graph']}/{graph_stats['decoder_graph']}.",
                file=sys.stderr,
            )
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


def _warmup_redimnet() -> None:
    """Eager-load every ReDimNet CUDA batch bucket (1/2/4/8) after the ASR cache trim."""

    try:
        from .genesis_whisper_server_vid import load_vid_model

        print("[WARMUP-VID] Lade und waerme ReDimNet2 fuer niedrige Tail-Latenz...", file=sys.stderr)
        if not load_vid_model():
            print("[WARMUP-VID] ReDimNet2-Warmup fehlgeschlagen; lazy Retry bleibt aktiv.", file=sys.stderr)
            return
        print("[WARMUP-VID] ReDimNet2 ist bereit.", file=sys.stderr)
    except Exception as exc:
        print(
            f"[WARMUP-VID] nicht-kritischer Warmup-Fehler ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )


def _lid_warmup_wanted() -> bool:
    """Only Cohere needs detection (Whisper detects itself, Parakeet ignores the language), and only while
    cohere_language_detection is on."""

    from .genesis_whisper_server_globals import cohere_language_detection_enabled, uses_cohere_backend

    return uses_cohere_backend(str(current_settings.get("local_model") or "")) and cohere_language_detection_enabled()


def _prepare_lid() -> None:
    """Download the detector before its GPU warmup (worker thread): no GPU lock or lease is held meanwhile."""

    try:
        from .genesis_whisper_server_lid import prepare_lid_model

        if _lid_warmup_wanted():
            prepare_lid_model()
    except Exception as exc:
        print(f"[WARMUP-LID] Download fehlgeschlagen ({type(exc).__name__}: {exc}).", file=sys.stderr)


def _warmup_lid() -> None:
    """Eager-load the language detection model and run it once, only where requests will use it.

    Best-effort like the other warmups: a failure leaves the lazy load (and the COHERE_FALLBACK_LANGUAGE
    fallback) in place. It loads only what ``_prepare_lid`` put on disk.
    """

    try:
        from .genesis_whisper_server_globals import uses_cohere_backend
        from .genesis_whisper_server_lid import configured_lid_model, detect_languages, load_lid_model

        if not _lid_warmup_wanted():
            if uses_cohere_backend(str(current_settings.get("local_model") or "")):
                print("[WARMUP-LID] Spracherkennung abgeschaltet -> uebersprungen.", file=sys.stderr)
            return
        print(f"[WARMUP-LID] Lade Spracherkennung '{configured_lid_model()}' ...", file=sys.stderr)
        if not load_lid_model():
            print("[WARMUP-LID] Spracherkennung nicht verfuegbar; lazy Retry bleibt aktiv.", file=sys.stderr)
            return
        detect_languages([_synthetic_warmup_audio()])
        print("[WARMUP-LID] Spracherkennung ist bereit.", file=sys.stderr)
    except Exception as exc:
        print(f"[WARMUP-LID] nicht-kritischer Warmup-Fehler ({type(exc).__name__}: {exc}).", file=sys.stderr)


def _warmup_segmentation() -> None:
    """Eager-load seg-3.0, warm ReDimNet's 1.5 s shapes and the batch diarization (FP16 windows, Viterbi); MMS_FA
    and Silero only with speaker_turns on. Best-effort like the other warmups: a failure leaves the lazy load in
    place (without a seg-3.0 checkpoint diarization answers 503 and speaker_turns is not offered)."""

    try:
        from .genesis_whisper_server_diarization_engine import warmup_segmentation

        warmup_segmentation()
    except Exception as exc:
        print(f"[WARMUP-SEG] nicht-kritischer Warmup-Fehler ({type(exc).__name__}: {exc}).", file=sys.stderr)


async def startup_server(app: FastAPI):
    print("--- GENESIS Transcription Server wird gestartet (FastAPI + React + lokales ASR-Batching) ---", file=sys.stderr)

    loaded_settings, first_start = load_settings_or_first_start_preset()
    current_settings.clear()
    current_settings.update(loaded_settings)
    if first_start:
        print(
            f"Erster Start: keine gespeicherten Einstellungen, Preset '{FIRST_START_PRESET}' angewendet und gespeichert.",
            file=sys.stderr,
        )
    reset_history_audio_store(
        enabled=current_settings.get("debug_retain_history_audio", False) is True,
    )
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
    await run_blocking_gpu_phase(_warmup_local_asr)
    # ReDimNet is warmed after the ASR-only trim. Its prepared CUDA shapes stay
    # warm; the allocator pool its warmup reserved (2.8 GB at batch 8) is released
    # by the batch worker's idle trim 5 s later, like after every other GPU phase.
    await run_blocking_gpu_phase(_warmup_redimnet)
    # The language detector (about 0.15 GB with whisper-base) comes last, after the ASR trim like ReDimNet.
    # Its first download runs on a worker thread first, so the GPU phase only loads from disk.
    await asyncio.to_thread(_prepare_lid)
    await run_blocking_gpu_phase(_warmup_lid)
    # seg-3.0 (about 6 MB of weights) and the 1.5 s ReDimNet buckets last; the idle trim releases their pool.
    await run_blocking_gpu_phase(_warmup_segmentation)


async def shutdown_server(app: FastAPI):
    batch_manager = getattr(app.state, "whisper_batch_manager", None)
    if batch_manager is not None:
        await batch_manager.stop()
    shutdown_history_audio_store()
    # Pending per-key usage counters (debounced writes).
    from .genesis_whisper_server_auth import get_auth_store

    get_auth_store().flush()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_server(app)
    try:
        yield
    finally:
        await shutdown_server(app)


app = FastAPI(title="G3 Whisper API", lifespan=lifespan)
app = create_api(app)
app = create_v2_api(app)
app = create_v2_capabilities_api(app)
app = create_admin_api(app)


def _frontend_index_response() -> Union[HTMLResponse, FileResponse]:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return HTMLResponse(
        """<!doctype html>
<html lang="en">
  <head><meta charset="utf-8" /><title>G3 Whisper · Admin</title></head>
  <body style="font-family: system-ui, sans-serif; padding: 32px;">
    <h1>The admin UI is not built</h1>
    <p>Run <code>npm install</code> and then <code>npm run build</code> in the <code>frontend</code> folder.</p>
  </body>
</html>
""",
        status_code=503,
    )


_LANDING_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>G3 Whisper</title>
    <style>
      :root {
        color-scheme: dark;
        --text: #ebf0df; --text-strong: #faf7ef; --text-dim: #a9b2b8; --accent: #ffc16c; --accent-strong: #ff8b47;
        --panel: rgba(15, 23, 29, .82); --panel-border: rgba(255, 248, 224, .12); --line-strong: rgba(255, 255, 255, .16);
        --surface-hover: rgba(255, 255, 255, .08); --on-accent: #1b1510;
        --bg: #10161c;
        --bg-image: radial-gradient(circle at top left, rgba(255, 194, 117, .20), transparent 28%),
          radial-gradient(circle at top right, rgba(139, 230, 192, .16), transparent 24%),
          linear-gradient(145deg, #182028, #10161c 45%, #17131c);
      }
      @media (prefers-color-scheme: light) {
        :root {
          color-scheme: light;
          --text: #1f272d; --text-strong: #11171b; --text-dim: #56626b; --accent: #ffb24d; --accent-strong: #e8661f;
          --panel: rgba(255, 255, 255, .86); --panel-border: rgba(31, 39, 45, .12); --line-strong: rgba(31, 39, 45, .20);
          --surface-hover: rgba(31, 39, 45, .07);
          --bg: #f5f1e8;
          --bg-image: radial-gradient(circle at top left, rgba(255, 170, 90, .22), transparent 30%),
            radial-gradient(circle at top right, rgba(90, 200, 160, .18), transparent 26%),
            linear-gradient(145deg, #faf6ee, #f2eee5 45%, #f4eff3);
        }
      }
      html { background: var(--bg); }
      body {
        margin: 0; min-height: 100vh; color: var(--text); background: var(--bg-image); background-attachment: fixed;
        font: 1rem/1.5 Bahnschrift, "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
      }
      main { width: min(960px, calc(100vw - 32px)); margin: 0 auto; padding: 56px 0; }
      .panel {
        display: grid; gap: 16px; padding: 24px; border-radius: 24px;
        background: var(--panel); border: 1px solid var(--panel-border); box-shadow: 0 28px 60px rgba(0, 0, 0, .28);
        backdrop-filter: blur(18px);
      }
      .eyebrow { margin: 0; color: var(--accent); font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
      h1 {
        margin: 0; color: var(--text-strong); line-height: 1.2;
        font: 700 clamp(1.75rem, 1.25rem + 1.5vw, 2.25rem) Rockwell, "Roboto Slab", Bahnschrift, system-ui, sans-serif;
      }
      p { margin: 0; color: var(--text-dim); }
      code { color: var(--text-strong); font-family: "Cascadia Code", Consolas, "DejaVu Sans Mono", monospace; font-size: .9em; }
      .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
      a {
        display: inline-flex; align-items: center; justify-content: center; min-height: 40px; padding: 0 20px;
        border-radius: 999px; border: 1px solid transparent; text-decoration: none; font-size: .875rem; font-weight: 700;
      }
      a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
      .primary { background: linear-gradient(135deg, var(--accent), var(--accent-strong)); color: var(--on-accent); }
      .secondary { background: var(--surface-hover); color: var(--text-strong); border-color: var(--line-strong); }
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">G3 GPU service</p>
        <h1>G3 Whisper</h1>
        <p>
          Local speech recognition, voice embeddings and diarization. The transcription API
          (<code>POST /transcribe/</code>, <code>POST /v1/audio/transcriptions</code>, <code>GET /v1/models</code>,
          <code>POST /v2/audio/process</code>) is open while no API key exists. Once an administrator creates a key,
          every request needs <code>Authorization: Bearer &lt;key&gt;</code> or <code>X-API-Key: &lt;key&gt;</code>.
        </p>
        <p>The admin pages under <code>/admin</code> require a username and password.</p>
        <div class="actions">
          <a class="primary" href="/admin">Open admin</a>
          <a class="secondary" href="/docs">API docs</a>
        </div>
      </section>
    </main>
  </body>
</html>
"""


def _landing_response() -> HTMLResponse:
    return HTMLResponse(_LANDING_HTML, status_code=200)


def _resolve_frontend_asset(full_path: str) -> Path | None:
    """File under frontend/dist for full_path, or None (same rules as G3_RERANK).

    The path is checked lexically before the filesystem is touched: on Windows, dist / "//host/share/x" is a UNC
    path, and resolving it would open an SMB connection. Backslashes, drive letters / colons, absolute paths and
    '..' never reach the filesystem; symlinks pointing outside dist are never served.
    """
    if not full_path or "\x00" in full_path or "\\" in full_path or ":" in full_path:
        return None
    relative = PurePosixPath(full_path)
    if relative.is_absolute() or full_path.startswith("/") or ".." in relative.parts:
        return None
    parts = [part for part in relative.parts if part not in {"", "."}]
    if not parts:
        return None
    try:
        dist_root = FRONTEND_DIST_DIR.resolve(strict=True)
        candidate = dist_root.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_relative_to(dist_root) or not candidate.is_file():
        return None
    return candidate


@app.get("/", include_in_schema=False)
async def serve_landing():
    return _landing_response()


@app.get("/admin", include_in_schema=False)
@app.get("/admin/", include_in_schema=False)
async def serve_frontend_index():
    return _frontend_index_response()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_assets(full_path: str):
    if full_path.startswith(("api/", "v1/", "v2/")):
        # JSON 404 instead of the landing page for mistyped API paths.
        raise HTTPException(status_code=404, detail="Not Found")

    if full_path.startswith("admin/"):
        asset_path = _resolve_frontend_asset(full_path.removeprefix("admin/"))
        return FileResponse(asset_path) if asset_path is not None else _frontend_index_response()

    asset_path = _resolve_frontend_asset(full_path)
    if asset_path is not None:
        return FileResponse(asset_path)
    return _landing_response()


if __name__ == "__main__":
    print("\n--- G3 Whisper is ready ---", file=sys.stderr)
    print("Start page: http://127.0.0.1:7861/", file=sys.stderr)
    print("Admin: http://127.0.0.1:7861/admin", file=sys.stderr)
    print("API: POST http://127.0.0.1:7861/transcribe/", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=7861)
