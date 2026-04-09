# G3_WHISPER

**Repository:** [https://github.com/havok2-htwo/G3_Whisper](https://github.com/havok2-htwo/G3_Whisper)

## Purpose

`G3_WHISPER` is a local transcription server based on FastAPI + React/Vite.
It provides a simple upload API for audio and video files, features a protected admin dashboard, and currently supports two local ASR paths:

- Whisper models via Hugging Face `transformers`
- `CohereLabs/cohere-transcribe-03-2026`

Optionally, a voice vector can also be generated based on `pyannote/embedding`.

This file acts as the central functional and technical reference for this repository. If the code and the README diverge, the following rule applies: The README must be updated in the same work step until it accurately reflects the current state again.

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
- Admin login via signed HTTP-only session cookie
- Model switching and runtime settings in the admin panel
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

The backend is split into multiple Python files:

- [genesis_whisper_server.py](x:/dev/G3_WHISPER/genesis_whisper_server.py)
  - creates the FastAPI app
  - mounts API and admin routes
  - defines the lifespan startup/shutdown
  - serves the built frontend
- [genesis_whisper_server_api.py](x:/dev/G3_WHISPER/genesis_whisper_server_api.py)
  - unprotected upload API
  - dispatch to local ASR path
  - optional voice vector generation
- [genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/genesis_whisper_server_admin.py)
  - protected admin endpoints
  - settings, stats, queue, benchmark
- [genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_local_asr_engine.py)
  - local loading and inference for Whisper and Cohere
- [genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/genesis_whisper_server_batching.py)
  - queuing and batch processing for Whisper
- [genesis_whisper_server_chunking.py](x:/dev/G3_WHISPER/genesis_whisper_server_chunking.py)
  - speech segment detection
  - Whisper chunking
  - speech-only extraction for Voice ID
- [genesis_whisper_server_audio.py](x:/dev/G3_WHISPER/genesis_whisper_server_audio.py)
  - reading and normalizing audio/video files
- [genesis_whisper_server_vid.py](x:/dev/G3_WHISPER/genesis_whisper_server_vid.py)
  - optional voice vector via `pyannote/embedding`
- [genesis_whisper_server_auth.py](x:/dev/G3_WHISPER/genesis_whisper_server_auth.py)
  - admin authentication via cookie + HMAC signature
- [genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/genesis_whisper_server_storage.py)
  - loading/saving settings
  - JSONL logging
- [genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/genesis_whisper_server_globals.py)
  - central constants, model lists, runtime state

### Frontend

The frontend is located under [`frontend`](x:/dev/G3_WHISPER/frontend) and uses:

- React 18
- TypeScript
- Vite

The build target is [`frontend/dist`](x:/dev/G3_WHISPER/frontend/dist). This directory is served directly by FastAPI.

## Active Runtime Logic

### Transcription

The standard path is `POST /transcribe/`.

Inputs:

- `file`: Audio or video file
- `engine`: currently only `local`
- `voice_ident`: `true` or `false`

Behavior:

- If `voice_ident=false`, the request is allowed to enter the Whisper batch worker.
- If `voice_ident=true`, the request is intentionally processed serially under a GPU lock.
- If the active model is a Whisper model, audio is split into speech segments prior to batch inference.
- If the active model is Cohere, the local ASR path currently processes the audio as a single item.

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

- preferred: `webrtcvad`
- fallback: energy-based

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

Currently not included as a fixed integrated standard:

- explicit Triton-for-Windows setup
- mandatory Flash Attention setup
- explicit GPU warmup immediately after model loading

Important runtime decision:

- The model loader is no longer dependent on `accelerate` as a mandatory dependency for standard operation.
- Models are moved directly to the target device in the single-GPU/CPU path instead of enforcing `device_map`.

## Authentication

The admin area uses a lightweight custom session solution instead of a large 3rd-party auth framework.

Principle:

- Username and password hash come from the `.env`
- on a successful login, a signed token is generated
- the token is stored as an HTTP-only cookie in the browser
- protected routes use `require_admin`

Hash formats:

- `pbkdf2_sha256$...`
- `sha256$...`
- `plain$...`
- or raw SHA256 hash as a fallback

## `.env`

The most important environment variables:

- `HUGGINGFACE_TOKEN`
  - required for `pyannote/embedding`
  - may also be relevant for gated Hugging Face models
- `GENESIS_ADMIN_USERNAME`
- `GENESIS_ADMIN_PASSWORD_HASH`
- `GENESIS_SESSION_SECRET`

If the three `GENESIS_ADMIN_*` or session variables are missing, the admin login is not properly configured.

## Persistent Data and Logs

The server stores runtime data under [`logs`](x:/dev/G3_WHISPER/logs).

Important files:

- `logs/genesis_whisper_settings.json`
  - persisted server settings
- `logs/transcription_log.jsonl`
  - JSONL log of transcription requests

## Default Settings

The active default values come from [genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/genesis_whisper_server_storage.py):

- `local_model`: `openai/whisper-base`
- `local_gpu_device`: `auto`
- `local_model_cache_path`: empty
- `transcription_language`: `auto`
- `batch_wait_time_ms`: `250`
- `batch_max_segments`: `8`
- `batch_max_audio_seconds`: `120.0`

## Admin Endpoints

Active admin routes:

- `POST /api/admin/login`
- `POST /api/admin/logout`
- `GET /api/admin/session`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/stats`
- `GET /api/admin/queue`
- `POST /api/admin/benchmark`

## API Endpoints

Active open route:

- `POST /transcribe/`

Frontend delivery:

- `GET /`
- `GET /{full_path:path}`

SPA Behavior:

- static files are served directly
- unknown frontend paths fall back to `index.html`
- paths under `/api/` are not accidentally passed through to the frontend

## Startup and Operation

### One-Click Start

The standard entry point is:

```bat
genesis2_whisper_server.bat
```

The startup script handles:

- creating a local `venv` under [`venv`](x:/dev/G3_WHISPER/venv) if not already present
- updating `pip`, `setuptools`, `wheel`
- installing PyTorch
- installing Python dependencies from [requirements.txt](x:/dev/G3_WHISPER/requirements.txt) or updating them if the file changed
- generating `frontend/node_modules` via `npm install` if needed
- building the frontend via `npm run build` if the build is missing
- starting the server afterwards

Note:

- The project intentionally uses `venv` and not `.venv`.
- The batch startup is the preferred method for normal use.

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

## Important Repository Areas

Actively relevant:

- [`frontend`](x:/dev/G3_WHISPER/frontend)
- [`logs`](x:/dev/G3_WHISPER/logs)
- [`models`](x:/dev/G3_WHISPER/models)
- [`testaudio`](x:/dev/G3_WHISPER/testaudio)
- the active `genesis_whisper_server_*.py` files in the root
- [genesis2_whisper_server.bat](x:/dev/G3_WHISPER/genesis2_whisper_server.bat)
- [requirements.txt](x:/dev/G3_WHISPER/requirements.txt)

Do not consider as active runtime source:

- [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete)

This directory deliberately contains legacy files, backup/alternate paths, and discarded helper scripts.

## Available Helper Scripts

### Password Hash

[genesis_whisper_password_hash.py](x:/dev/G3_WHISPER/genesis_whisper_password_hash.py)

Purpose:

- Generating a password hash for the `.env`

### Load Test

[genesis_whisper_batch_load_test.py](x:/dev/G3_WHISPER/genesis_whisper_batch_load_test.py)

Purpose:

- sends many parallel requests to the running server
- helps measure throughput, batch behavior, and stability

## File Formats

The server accepts audio files and, via decoding, video files as sources. The exact robustness depends on the installed local audio/codec dependencies.

The admin benchmark also accepts audio or video and uses the same audio loading path.

## Known Quirks and Specifics

- `python-multipart` is required for FastAPI form uploads and is part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `librosa` is required by parts of the local ASR stack and is therefore explicitly part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- The server has been migrated to FastAPI Lifespan. Old event handler calls should not be reintroduced.
- If a model cannot be loaded, the API and benchmark will bubble up the specific loader error instead of just a generic message.
- For models wrapped with `torch.compile(...)`, do not use truthiness checks like `if not model`. Always explicitly check `is None` instead.
- The admin frontend intentionally uses the German number format for metrics so that RTF, seconds, and VRAM values remain unambiguously readable.
- The Cohere model relies on `trust_remote_code`; the dynamic modules required for this are cached locally within the project under [`models/hf_modules`](x:/dev/G3_WHISPER/models/hf_modules) instead of in the global user cache.
- On Windows, if no `cl.exe` is available, Cohere continues to run without the optional internal compile path.
- `__pycache__` might reappear locally. This is normal and not source code.
- [genesis_whisper_server_diarization_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_diarization_engine.py) is in the repo but is not currently part of the active main path.

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
2. [genesis_whisper_server.py](x:/dev/G3_WHISPER/genesis_whisper_server.py)
3. [genesis_whisper_server_api.py](x:/dev/G3_WHISPER/genesis_whisper_server_api.py)
4. [genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/genesis_whisper_server_admin.py)
5. [genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_local_asr_engine.py)
6. [genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/genesis_whisper_server_batching.py)
