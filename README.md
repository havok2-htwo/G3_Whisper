# G3_WHISPER

**Repository:** [https://github.com/havok2-htwo/G3_Whisper](https://github.com/havok2-htwo/G3_Whisper)

## Purpose

`G3_WHISPER` is a local transcription server based on FastAPI + React/Vite.
It provides a simple upload API for audio and video files, features a protected admin dashboard, and currently supports two local ASR paths:

- Whisper models via Hugging Face `transformers`
- `CohereLabs/cohere-transcribe-03-2026`

Optionally, a voice vector can also be generated based on `pyannote/embedding`.
For gated Hugging Face models, the runtime loader can use the saved admin-setting token or `HUGGINGFACE_TOKEN` / `HF_TOKEN`.

This file acts as the central functional and technical reference for this repository. If the code and the README diverge, the following rule applies: The README must be updated in the same work step until it accurately reflects the current state again.

Endpoint-level request/response details live in [API_DOCUMENTATION.md](x:/dev/G3_WHISPER/API_DOCUMENTATION.md). This README remains the central operational and architectural overview.

## Maintenance Requirements

For any functional work on the project, the following applies:

1. Change code, scripts, configuration, and UI.
2. Then check whether `README.md` needs to be adapted or expanded.
3. Then update `CHANGELOG.md`.

This applies even if the change seems small. The README is deliberately the "source of truth" here for humans and AI agents.

## Current Status

Active is a single local server that:

- serves the React frontend and the API over the same port
- loads local ASR models
- processes Whisper requests via an internal batch queue system
- provides settings, queue status, history, and benchmarks in the admin area
- optionally generates a voice vector

Old Gradio, OpenAI, Voxtral, and side-server paths are no longer active. Such legacy code has largely been moved to [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete).

## Core Features

- Transcription via `POST /transcribe/`
- OpenAI-compatible routes on `GET /v1/models` and `POST /v1/audio/transcriptions`
- public landing page on `GET /` and admin SPA on `GET /admin`
- OpenAPI docs on `GET /docs` and schema export on `GET /openapi.json`
- admin dashboard protected by `X-Admin-Key`
- persistent hashed admin key plus temporary startup admin key on launch
- Model switching, Hugging Face Token input, and runtime settings in the admin panel
- Cache Manager with `missing`, `partial`, `downloading`, `ready`, and `error` states
- Batch queue for Whisper
- Benchmark in the admin panel including:
  - Runtime
  - Chunks
  - Total audio seconds
  - RTF (Real-Time Factor)
  - Peak VRAM
  - Transcript display
- Optional voice vector generation
- Pause/VAD-based segmentation for Whisper
- Speech-only pre-filtering for the voice vector

## Architecture

### Backend

The backend now lives under [`backend`](x:/dev/G3_WHISPER/backend) as a Python package.

Core backend files:

- [backend/genesis_whisper_server.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server.py)
  - creates the FastAPI app
  - mounts API and admin routes
  - defines the lifespan startup/shutdown
  - serves the built frontend
- [backend/genesis_whisper_server_api.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_api.py)
  - unprotected upload API
  - dispatch to local ASR path
  - optional voice vector generation
- [backend/genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_admin.py)
  - protected admin endpoints
  - settings, stats, queue, benchmark
- [backend/genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_local_asr_engine.py)
  - local loading and inference for Whisper and Cohere
- [backend/genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_batching.py)
  - queuing and batch processing for Whisper
- [backend/genesis_whisper_server_chunking.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_chunking.py)
  - speech segment detection
  - Whisper chunking
  - speech-only extraction for Voice ID
- [backend/genesis_whisper_server_audio.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_audio.py)
  - reading and normalizing audio/video files
- [backend/genesis_whisper_server_vid.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_vid.py)
  - optional voice vector via `pyannote/embedding`
- [backend/genesis_whisper_server_auth.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_auth.py)
  - admin key storage, rotation, and header-based authentication
- [backend/genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_storage.py)
  - loading/saving settings
  - JSONL logging
- [backend/genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_globals.py)
  - central constants, model lists, runtime state

### Frontend

The frontend is located under [`frontend`](x:/dev/G3_WHISPER/frontend) and uses:

- React 18
- TypeScript
- Vite

The build target is [`frontend/dist`](x:/dev/G3_WHISPER/frontend/dist). This directory is served directly by FastAPI.

## Active Runtime Logic

### Transcription

The native upload path is `POST /transcribe/`.

Inputs:

- `file`: Audio or video file
- `engine`: currently only `local`
- `voice_ident`: `true` or `false`

Behavior:

- If `voice_ident=false`, the request is allowed to enter the Whisper batch worker.
- If `voice_ident=true`, the request is intentionally processed serially under a GPU lock.
- If the active model is a Whisper model, audio is split into speech segments prior to batch inference.
- If the active model is Cohere, the local ASR path currently processes the audio as a single item.
- For gated Cohere loads, the runtime path now completes an incomplete Hugging Face snapshot first and only then loads the model from the finished local snapshot.
- Only the native `POST /transcribe/` route can return a `voice_vector`.

### OpenAI-Compatible Routes

The server also exposes:

- `GET /v1/models`
- `POST /v1/audio/transcriptions`

Compatibility notes:

- `GET /v1/models` returns the currently active local model plus the compatibility alias `whisper-1`.
- `POST /v1/audio/transcriptions` routes through the same local transcription backend as `POST /transcribe/`.
- `response_format` currently changes the output shape (`json`, `text`, `verbose_json`).
- `model`, `language`, `prompt`, and `temperature` are accepted for client compatibility, but the current implementation does not yet apply them independently from the saved server settings.

### Voice Vector

The voice vector is optional.

Important rules:

- It is only generated if `voice_ident=true`.
- Prior to embedding creation, only recognized speech segments are used whenever possible.
- If no speech is reliably detected, the code falls back once to the original audio.

Thus, the voice vector is now intentionally "speech-first" and not "silence-first".

### Whisper Chunking

Whisper does not use blind pure 30s slicing.

Instead:

- first detect speech segments
- skip large silences as much as possible
- apply padding to segments
- only further subdivide into larger Whisper-compatible chunks when necessary
- apply overlap between chunks

Detection path:

- active runtime path: energy-based speech detection only
- `webrtcvad` helper code still exists in the file but is intentionally disabled for the current main path because telephone/filtered speech performed less reliably there

### Benchmark

In the admin panel, there is a benchmark workflow. An audio or video file can be uploaded there and the number of repetitions can be set.

The benchmark shows:

- Number of runs
- Number of chunks per run
- Total number of chunks
- Audio seconds per run
- Total audio seconds
- Total time
- Average time per run
- RTF (Real-Time Factor)
- Batch history used
- Peak VRAM
- Generated transcript

Number display in the dashboard:

- Benchmark and metric values are displayed in the frontend using the German number format.
- Example: `78,578` means `seventy-eight point five hundred seventy-eight`, not `78.578` as a thousands value.

## Active Models

The current model list is maintained centrally in [genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/genesis_whisper_server_globals.py).

Currently available:

- `CohereLabs/cohere-transcribe-03-2026`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-large-v3`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `openai/whisper-tiny`

The Cache Manager in the UI also additionally supports downloading and deleting the `pyannote/embedding` model for Voice Vectors.

Note:

- The `openai/whisper-*` names are Hugging Face model IDs.
- No OpenAI Cloud API is used anymore.
- The `engine` value `openai` is no longer supported.

## Optimizations

The local ASR path already utilizes several optimizations:

- GPU dtype optimization, preferring `bfloat16`, else `float16`
- `torch.compile(...)`, if available and sensible
- `sdpa` as the attention standard
- `flash_attention_2` only if `flash_attn` is installed
- Lazy loading for models
- Batch queue for Whisper

Backend-specific exception:

- The Cohere ASR model is currently intentionally loaded with `attn_implementation="eager"` because the current Transformers integration for this model does not properly support `sdpa` or `flash_attention_2` yet.
- On Windows, the internal Cohere `transcribe()` compile path is disabled if no `cl.exe` is found.
- For gated Cohere models, the runtime loader now reuses the configured Hugging Face token for both direct `from_pretrained(...)` calls and a full fallback `snapshot_download(...)` when the local cache is incomplete.

Currently not included as a fixed integrated standard:

- explicit Triton-for-Windows setup
- mandatory Flash Attention setup
- explicit GPU warmup immediately after model loading

Important runtime decision:

- The model loader is no longer dependent on `accelerate` as a mandatory dependency for standard operation.
- Models are moved directly to the target device in the single-GPU/CPU path instead of enforcing `device_map`.

## Admin Key Workflow

The admin area is now protected only by `X-Admin-Key`.

Principle:

- all `/api/admin/...` routes require the request header `X-Admin-Key`
- the frontend stores the entered admin key in local browser storage (`localStorage` and `sessionStorage`) and sends it with each admin request
- the persistent admin key is stored hashed in [`logs/genesis_whisper_secrets.json`](x:/dev/G3_WHISPER/logs/genesis_whisper_secrets.json)
- `GET /api/admin/keys` returns metadata for the active admin key
- `POST /api/admin/keys` rotates the admin key and returns the new plaintext token exactly once
- the startup script also prints a temporary startup admin key that expires server-side after a short TTL

## Environment Variables

The most important environment variables:

- `HUGGINGFACE_TOKEN`
  - required for `pyannote/embedding`
  - required for gated Hugging Face models such as `CohereLabs/cohere-transcribe-03-2026` unless the token is saved in the admin UI
  - **Note:** Can now optionally be configured directly via the Admin Settings UI, which takes precedence over `.env`.
  - **Important:** A token entered only during a manual cache download is not sufficient for later runtime loading. The token must be persisted in Admin Settings or `.env`.
  - **Important:** The same saved token is now also used by the runtime loader itself when Cohere needs to complete missing snapshot files such as `tokenizer.model`.
- `GENESIS_ADMIN_KEY`
  - optional bootstrap value for the persistent admin key on first run
- `GENESIS_STARTUP_ADMIN_KEY`
  - optional temporary recovery key, usually generated by the startup script
- `GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS`
  - validity window for the temporary startup key after server start
- `GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS`
  - how long the batch script keeps the key visible before clearing the screen

## Persistent Data and Logs

The server stores runtime data under [`logs`](x:/dev/G3_WHISPER/logs).

Important files:

- `logs/genesis_whisper_settings.json`
  - persisted server settings
- `logs/genesis_whisper_secrets.json`
  - hashed persistent admin key metadata
- `logs/transcription_log.jsonl`
  - JSONL log of transcription requests

## Default Settings

The active default values come from [backend/genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_storage.py):

- `local_model`: `openai/whisper-large-v3-turbo`
- `local_gpu_device`: `auto`
- `local_model_cache_path`: `.\models`
- `transcription_language`: `auto`
- `batch_wait_time_ms`: `500`
- `batch_max_segments`: `32`
- `batch_max_audio_seconds`: `300.0`
- `huggingface_token`: empty string

## Admin Endpoints

Active admin routes:

- all of the following require `X-Admin-Key`
- `GET /api/admin/keys`
- `POST /api/admin/keys`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/models`
- `POST /api/admin/models/download`
- `POST /api/admin/models/delete`
- `GET /api/admin/stats`
- `GET /api/admin/queue`
- `POST /api/admin/benchmark`

## API Endpoints

Active open routes:

- `POST /transcribe/`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `GET /docs`
- `GET /openapi.json`

Frontend delivery:

- `GET /`
- `GET /admin`
- `GET /admin/{full_path:path}`
- `GET /{full_path:path}`

SPA Behavior:

- `GET /` serves a public landing page that explains the open transcription API and the protected admin dashboard
- `/admin` and `/admin/...` serve the admin SPA
- static frontend assets are served directly
- unknown non-API paths fall back to the public landing page
- paths under `/api/` are not accidentally passed through to the frontend

## Startup and Operation

### One-Click Start

The standard launcher entry points are:

```bat
start.bat
```

```bash
bash ./install.sh
bash ./start.sh
```

Platform split:

- `start.bat` is the Windows-first launcher and performs setup plus start in one step.
- `install.sh` is the dedicated Linux/Unix setup step.
- `start.sh` is the Linux/Unix launcher and calls `install.sh` automatically unless `SKIP_INSTALL=1` is set.
- both launchers start the server via `python -m backend.genesis_whisper_server`

What the launcher/install flow handles:

- creating a local `venv` under [`venv`](x:/dev/G3_WHISPER/venv) if not already present
- updating `pip`, `setuptools`, `wheel`
- installing PyTorch
- installing Python dependencies from [requirements.txt](x:/dev/G3_WHISPER/requirements.txt) or updating them if the file changed
- generating `frontend/node_modules` via `npm install` if missing or if `frontend/package*.json` changed since the last dependency install
- building the frontend via `npm run build` if the build is missing or stale
- warning if `ffmpeg` is not available on `PATH`
- generating and displaying a temporary startup admin key before launch
- starting the server afterwards

Note:

- The project intentionally uses `venv` and not `.venv`.
- `start.bat` is the Windows-first default launcher.
- On Linux/Unix, `bash ./install.sh` is the safest invocation because executable bits are not always preserved in ZIP/Git-on-Windows workflows.
- `install.sh` auto-selects PyTorch from PyPI on CPU-only Linux and switches to CUDA 12.8 wheels if `nvidia-smi` is available.
- `TORCH_INDEX_URL` can be set to force a specific PyTorch wheel index on Linux/Unix.
- `start.sh` performs setup first by default and then starts the server.

### Manual Frontend Commands

```powershell
cd frontend
npm install
npm run build
```

### Manual Python Dependencies

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

Linux / Unix equivalent:

```bash
./venv/bin/pip install -r requirements.txt
```

## Important Repository Areas

Actively relevant:

- [`frontend`](x:/dev/G3_WHISPER/frontend)
- [`logs`](x:/dev/G3_WHISPER/logs)
- [`models`](x:/dev/G3_WHISPER/models)
- [`testaudio`](x:/dev/G3_WHISPER/testaudio)
- the active [`backend`](x:/dev/G3_WHISPER/backend) package
- [start.bat](x:/dev/G3_WHISPER/start.bat)
- [install.sh](x:/dev/G3_WHISPER/install.sh)
- [start.sh](x:/dev/G3_WHISPER/start.sh)
- [requirements.txt](x:/dev/G3_WHISPER/requirements.txt)

Do not consider as active runtime source:

- [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete)

This directory deliberately contains legacy files, backup/alternate paths, and discarded helper scripts.

## Available Helper Scripts

### Startup Admin Key Generator

[tools/generate_startup_admin_key.py](x:/dev/G3_WHISPER/tools/generate_startup_admin_key.py)

Purpose:

- generates a one-time temporary admin key for the startup script

### Legacy Password Hash Helper

[backend/genesis_whisper_password_hash.py](x:/dev/G3_WHISPER/backend/genesis_whisper_password_hash.py)

Purpose:

- generic PBKDF2-SHA256 hash generator
- not part of the active admin-key-only authentication path

### Load Test

[backend/genesis_whisper_batch_load_test.py](x:/dev/G3_WHISPER/backend/genesis_whisper_batch_load_test.py)

Purpose:

- sends many parallel requests to the running server
- can validate admin access by passing `--admin-key`
- helps measure throughput, batch behavior, and stability

## File Formats

The server accepts audio files and, via decoding, video files as sources. The exact robustness depends on the installed local audio/codec dependencies.

Notes:

- Formats directly readable by `soundfile` work without external tools.
- For many video/container formats and some audio codecs, `ffmpeg` on `PATH` is required for the fallback decode path.

The admin benchmark also accepts audio or video and uses the same audio loading path.

## Known Quirks and Specifics

- `python-multipart` is required for FastAPI form uploads and is part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `librosa` is required by parts of the local ASR stack and is therefore explicitly part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `omegaconf` is explicitly part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt) because `pyannote/embedding` can otherwise fail at runtime even when `pyannote.audio` is already installed.
- The server has been migrated to FastAPI Lifespan. Old event handler calls should not be reintroduced.
- If a model cannot be loaded, the API and benchmark will bubble up the specific loader error instead of just a generic message.
- For models wrapped with `torch.compile(...)`, do not use truthiness checks like `if not model`. Always explicitly check `is None` instead.
- The admin frontend intentionally uses the German number format for metrics so that RTF, seconds, and VRAM values remain unambiguously readable.
- The active speech detection path is energy-based. The `webrtcvad` helper implementation is currently not used in the main runtime flow.
- The OpenAI-compatible `POST /v1/audio/transcriptions` route currently uses the active saved server model/settings and mainly varies the response shape for client compatibility.
- The Cohere model relies on `trust_remote_code`; the dynamic modules required for this are cached locally within the project under [`models/hf_modules`](x:/dev/G3_WHISPER/models/hf_modules) instead of in the global user cache.
- Cohere is only treated as cache-ready when its local snapshot contains the expected remote-code files, tokenizer assets including `tokenizer.model`, and model weights. Incomplete snapshots are intentionally surfaced as `partial`.
- If a gated Cohere snapshot is incomplete but a valid Hugging Face token is configured, the runtime loader now performs a full `snapshot_download(...)` before loading the local snapshot.
- On Windows, if no `cl.exe` is available, Cohere continues to run without the optional internal compile path.
- On Windows, Hugging Face may warn about degraded cache behavior without symlink support. Developer Mode or elevated execution improves this, but the cache still works without it.
- `pyannote/embedding` is treated as cache-ready when its snapshot contains `config.yaml` and `pytorch_model.bin`; unlike Whisper models it does not rely on `preprocessor_config.json`.
- The voice-vector loader prefers a complete local snapshot under the configured `local_model_cache_path` before falling back to the Hugging Face Hub cache.
- `__pycache__` might reappear locally. This is normal and not source code.
- [backend/genesis_whisper_server_diarization_engine.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_diarization_engine.py) is in the repo but is not currently part of the active main path.

## Git & AI Agents: Saving Code (Push)

This repository is connected to GitHub. Since AI assistants often run in the background, they generally cannot interact with interactive login prompts from the *Windows Git Credential Manager*. The authentication process would stall.
Therefore, if an AI agent is tasked with managing code and pushing, the following workflow applies:

1. AI agent checks the status and keeps the `.gitignore` clean.
2. AI agent commits local changes via `git add -A` and `git commit -m "..."`.
3. The AI agent then instructs the user to manually execute the command `git push` (or `git push -u origin main`) in their **own visible terminal**.
4. The user executes the push and logs in - if prompted by Git - via the pop-up UI window.

## Working Rules for Future Changes

- Do not reactivate old OpenAI/Voxtral/Gradio paths unless explicitly requested.
- Reconcile `README.md` against real code after every relevant change.
- Update `CHANGELOG.md` after every relevant change.
- New runtime paths, files, or endpoints must be documented in this README.
- If something is only tentatively being removed, move it to [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete) first instead of deleting it immediately.

## Quick Overview for New Sessions

If someone needs to understand the repo quickly, the most important entry points are:

1. [README.md](x:/dev/G3_WHISPER/README.md)
2. [API_DOCUMENTATION.md](x:/dev/G3_WHISPER/API_DOCUMENTATION.md)
3. [backend/genesis_whisper_server.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server.py)
4. [backend/genesis_whisper_server_api.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_api.py)
5. [backend/genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_admin.py)
6. [backend/genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_local_asr_engine.py)
7. [backend/genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_batching.py)
