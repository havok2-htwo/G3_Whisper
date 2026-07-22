# G3_WHISPER

**Repository:** [https://github.com/havok2-htwo/G3_Whisper](https://github.com/havok2-htwo/G3_Whisper)

## Purpose

`G3_WHISPER` is a local transcription server based on FastAPI + React/Vite.
It provides a simple upload API for audio and video files, features a protected admin dashboard, and currently supports two local ASR paths:

- Whisper models via Hugging Face `transformers`
- `CohereLabs/cohere-transcribe-03-2026`

Voice vectors are generated exclusively with the pinned, open-weight ReDimNet2-B6 LM
`vb2+vox2+cnc2_v0` model and are always L2-normalized 192-dimensional vectors.
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
- exposes the versioned multi-mode `POST /v2/audio/process` API
- generates fixed ReDimNet2-B6 voice embeddings and, in `diarization` mode only, delegates speaker separation to G3_DIA

Old Gradio, OpenAI, Voxtral, and side-server paths are no longer active. Such legacy code has largely been moved to [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete).

## Core Features

- Transcription via `POST /transcribe/`
- Versioned processing via `POST /v2/audio/process` with `embedding`, `transcript`, `transcript_embedding`, and `diarization` modes
- OpenAI-compatible routes on `GET /v1/models` and `POST /v1/audio/transcriptions`
- public landing page on `GET /` and admin SPA on `GET /admin`
- OpenAPI docs on `GET /docs` and schema export on `GET /openapi.json`
- admin dashboard protected by username/password login
- client API keys for locking down the public transcription endpoints when desired
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
- Interactive v2 pipeline tester in the admin panel with drag-and-drop upload, all four
  processing modes, optional DIA speaker count/profiles, conservative speaker refinement,
  unknown-speaker listening samples, phase timings, speaker turns, warnings, and the
  complete JSON response
- Optional voice vector generation
- ReDimNet2 profile matching and return of unknown-speaker embeddings plus optional MP3
  reference samples after DIA diarization
- Conservative exact-pattern repetition filtering, enabled by default on every transcript-producing route and the admin benchmark
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
  - legacy upload API
  - dispatch to local ASR path
  - optional voice vector generation
- [backend/genesis_whisper_server_v2.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_v2.py)
  - versioned multi-mode processing API
  - DIA orchestration, turn transcription, and unified responses
- [backend/genesis_whisper_server_dia_client.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_dia_client.py)
  - authenticated streaming client for the G3_DIA v2 API
- [backend/genesis_whisper_server_speaker_matching.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_speaker_matching.py)
  - clean speaker-window extraction, robust embedding clouds, and one-to-one profile matching
- [backend/genesis_whisper_server_repetition.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_repetition.py)
  - exact Unicode-token repetition filtering
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
  - the single pinned ReDimNet2-B6 embedding implementation
- [backend/genesis_whisper_server_gpu.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_gpu.py)
  - optional cross-process file lease for shared-GPU deployments
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

### Versioned v2 Processing

`POST /v2/audio/process` accepts `multipart/form-data` with an audio/video `file` and a
JSON string in the `request` part. The JSON always uses `schema_version: "2.0"` and one
of four modes:

- `embedding`: one L2-normalized ReDimNet2 192-D embedding, without ASR or DIA
- `transcript`: one cleaned transcript, without DIA
- `transcript_embedding`: the transcript plus one recording-level ReDimNet2 embedding, without DIA; if no suitable embedding window exists, the transcript is still returned with `status: partial`, `embedding: null`, and a warning
- `diarization`: G3_DIA speaker separation, ASR per exclusive speaker turn, ReDimNet2 profile matching, and unknown-speaker embeddings

Only `diarization` contacts G3_DIA. The other three modes do not pay a DIA round trip and
do not load or expose the DIA model. The DIA `community-1` pipeline still uses its own
internal 256-D representation, but that representation is never serialized or used for
identity matching. Public voice vectors use only ReDimNet2.

The authenticated admin SPA can call this route with its existing same-origin session.
Machine clients continue to use `X-API-Key` whenever client keys are configured. If the
tester explicitly supplies an API key, that key is validated normally and an invalid key
is not hidden by the admin session.

In `diarization` mode, clients may provide an exact `expected_speakers` value from `1` to
`64` and zero or more known profiles. Every profile has a unique application-level `id`
and one or more 192-D ReDimNet2 vectors. Vectors must contain only finite values and have
a non-zero norm; dimensions such as the former 512-D vectors or DIA-internal 256-D
vectors are rejected with HTTP 422. If `expected_speakers` is omitted, DIA selects the
speaker count automatically; supplied known profiles set only the lower bound. Known
profiles are expected to represent speakers that are present in the recording.

Two independent, optional DIA result enhancements are available. `speaker_refinement`
accepts `off` (the default), `shadow`, or `conservative` and uses the already-computed
ReDimNet2 windows to detect high-confidence whole-turn label mistakes before ASR. The
default only disables this additional correction pass; DIA itself still runs normally.
`unknown_speaker_audio` is a boolean with default `false`. When enabled, it adds a short
MP3 listening reference to eligible unknown and unresolved speakers. It does not enable,
disable, or otherwise alter DIA or speaker refinement.

The result retains both the assigned application `speaker_id` and the original
`diarization_speaker_id`, millisecond timecodes, speaker kind, text, and overlap flag. If
refinement was requested, `refined_diarization_speaker_id` records the effective DIA label
while the original field remains unchanged. Refinement diagnostics report proposals,
applied turns, moved duration, evidence, runtime, truncation, and any rollback reason.
Unknown speakers return a robust prototype plus at most 63 diverse time-coded
representatives, for a deterministic maximum of 64 vectors per speaker. Weak or
ambiguous evidence is not forced into a known identity and is reported through
unresolved profile IDs, assignment/quality information, and warnings.

Requested listening references are returned as raw base64-encoded `audio/mpeg` data in
the corresponding `unknown_speakers` or `unresolved_speakers` item. Each reference is at
most 30 seconds and concatenates only source snippets of at least five contiguous seconds,
selected from central, timed, non-stitched final cloud inliers. The response also lists
the original time range and centrality of every source snippet. Known speakers never
receive this audio field. If no suitable source exists, or MP3 encoding fails, embeddings
remain available and the response carries a warning instead.

These DIA options leave Cohere transcription unchanged. No glossary, hotword, vocabulary,
prompt-biasing, or additional ASR-model behavior is introduced.

There is intentionally no embedding-model selector and no client-supplied
`embedding_space_id`. API version 2 is permanently tied to this exact ReDimNet2 model
space; changing the checkpoint later requires a new API/embedding version.

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
- The legacy response keys remain unchanged, but `voice_vector` now contains exactly 192 ReDimNet2 values instead of the former 512 values.

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

- Legacy `POST /transcribe/` generates it only when `voice_ident=true`; v2 also exposes it through `embedding` and `transcript_embedding`.
- VAD removes silence, the remaining speech is split into fixed three-second windows, windows are embedded in batches, and a quality-weighted mean is L2-normalized.
- Invalid, silent, or heavily clipped windows are discarded. At least 0.5 seconds of usable input is required for a recording-level embedding; DIA speaker clouds deliberately keep their stricter two-second clean-speech requirement.
- In v2 `transcript_embedding`, ASR runs first. If no window survives the embedding checks, the request remains successful with `status: partial`, the complete transcript, `embedding: null`, and warning code `VOICE_EMBEDDING_UNAVAILABLE`. Pure `embedding` requests still fail with HTTP 422 because they have no independent result to return.
- Multi-speaker input outside `diarization` mode intentionally produces a mixed recording-level vector; no identity assignment is attempted.
- The model is lazy-loaded once, warmed once, uses FP16 on CUDA and FP32 on CPU, and is deliberately not passed through `torch.compile`.

The single public model is ReDimNet2-B6 LM `vb2+vox2+cnc2_v0`, 192-D, from the
[official MIT-licensed repository](https://github.com/PalabraAI/redimnet2). Runtime code pins release `v1.0.0`, source commit
`2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`, and checkpoint SHA-256
`287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868`.
The source pin is newer than the original release tag because this release asset uses the
post-tag `agg_gnorm` configuration; the tag itself cannot instantiate this checkpoint.

### Diarization and Profile Matching

In `diarization` mode, Whisper streams the original upload to G3_DIA. DIA returns both
standard and exclusive speaker turns plus overlap intervals. Exclusive turns drive ASR;
standard turns/overlap intervals identify regions that are unsafe for embeddings.

ReDimNet embeddings per detected DIA speaker are created only from clean speech:

- a 200 ms safety margin is removed at speaker boundaries
- overlap regions are excluded
- clean regions need at least two seconds and target three-second windows
- loudness, clipping, and invalid-sample checks reject unsafe material
- all valid candidates participate; cosine components remove mixed clusters and near-duplicate/outlier filtering stabilizes the cloud
- Hungarian matching enforces a global one-to-one relationship between known IDs and detected speaker clusters

Initial matching thresholds are cosine `0.60`, support `0.60`, and stability margin
`0.04`; speaker-cloud components start at cosine `0.45` and need a dominant component
share of `0.60`. Quality states are `ready`, `low_support`, `mixed_cluster`, and
`insufficient_clean_speech`.

Optional `speaker_refinement=shadow` evaluates suspicious original exclusive DIA turns
without changing them. `conservative` can reassign a complete turn only when multiple
quality, centrality, vote, and overlap gates agree. It uses frozen speaker seeds and no
second embedding inference. The complete proposal set is applied synchronously and then
validated; speaker-count, seed, cloud-quality, timeline, or moved-duration violations roll
the whole pass back. Profile matching happens afterwards, so client-provided identities
cannot steer the correction.

Optional `unknown_speaker_audio=true` selects only timed, non-stitched final inliers with
quality at least `0.60`, ranks their contiguous source runs by prototype centrality, and
encodes up to 30 seconds as 16 kHz mono MP3 at 64 kbit/s. Every selected source run is at
least five seconds. The `audio.data` value is plain base64 without a data-URL prefix.

### Transcript Repetition Filter

After chunk joining, transcript-producing paths collapse exact consecutive ASR loops to
the first original occurrence. This applies by default to `POST /transcribe/`,
`POST /v1/audio/transcriptions`, the transcript-producing v2 modes, and the admin
benchmark. Diarization filters each speaker turn independently so it never removes text
across speaker boundaries.

Matching uses Unicode NFKC normalization and case-folding and ignores whitespace and
terminal punctuation. A single word must repeat at least five times; a pattern of 2
through 32 tokens must repeat at least three times. There are no fuzzy matches, markers,
counters, or response-shape changes. Clients can disable the filter for a request with
`X-G3-Repetition-Filter: off`.

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

### Pipeline Tester and Benchmark

The admin panel first exposes the production v2 pipeline itself: drop an audio/video file,
select `embedding`, `transcript`, `transcript_embedding`, or `diarization`, and inspect
client/server runtime, every server phase, model metadata, speaker counts, time-coded turns,
warnings, and the raw response. Diarization additionally accepts `expected_speakers` and a
strictly validated JSON array of known 192-D ReDimNet2 profiles. It also exposes the
independent speaker-refinement mode and unknown-speaker-audio switch, renders refinement
diagnostics, and provides an audio player for returned listening references.

The same file can then be reused in the collapsible parallel ASR benchmark, where the number
of repetitions can be set.

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

The current ASR model list is maintained centrally in [backend/genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_globals.py).

Currently available:

- `CohereLabs/cohere-transcribe-03-2026`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-large-v3`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `openai/whisper-tiny`

Voice embedding is not an admin-selectable ASR model. Every public embedding path uses
the fixed ReDimNet2-B6 LM checkpoint described above; it is downloaded lazily into the
configured model cache and verified against its pinned SHA-256 before use.

Note:

- The `openai/whisper-*` names are Hugging Face model IDs.
- No OpenAI Cloud API is used anymore.
- The `engine` value `openai` is no longer supported.
- There is no public embedding-model selection and no `embedding_space_id` request field.

## Optimizations

The local ASR path already utilizes several optimizations:

- configurable GPU model precision via admin settings: `bf16`, `fp16`, `int8_bnb`, `fp8`, or `fp32`
  - `bf16` is the recommended precision.
  - `fp8` is experimental (see "Experimental fp8 Precision" below).
- optional 8-bit Transformers loading through `bitsandbytes` for lower model VRAM usage
- `torch.compile(...)`, if available and sensible
- `sdpa` as the attention standard
- `flash_attention_2` only if `flash_attn` is installed
- Batch queue for Whisper
- bounded enqueue windows (`2 * batch_max_segments`, minimum 2, maximum 256) on legacy,
  v2, diarization, and benchmark paths; this keeps the GPU worker fed without retaining
  thousands of request tasks for long recordings
- real batched inference: Whisper uses one processor/model forward per worker batch;
  Cohere uses sub-batches of at most 16 audio items and performs its own long-audio chunking
- startup warmup of the configured ASR model with a sample clip (see "Startup Warmup" below)
- optional idle CUDA-cache trimming, disabled by default to preserve low tail latency (see "Idle VRAM Trimming" below)
- ReDimNet2 window batching (default 16) with automatic CUDA-OOM batch-size reduction,
  one shared batch stream across DIA speakers, block-vectorized VAD, and lazy audio windows
  that avoid full recording/speech copies
- serial DIA, ASR, and ReDim phases on the local GPU; an optional cross-process file lease coordinates Whisper and DIA when they share a physical GPU

### Startup Warmup

The configured ASR model is now eager-loaded at startup and warmed with `testaudio/Testaudio_02.wav`
(previously the model was loaded lazily on the first request). After the warmup transcription the
CUDA cache is trimmed, so the first real request is already warm without leaving an oversized
reserved pool behind.

The warmup is best-effort: any warmup failure (for example an unsupported precision) is logged and
never blocks server startup.

### Idle VRAM Trimming

The `cuda_memory_trim_after_batch` admin checkbox mirrors OmniVoice's "Auto VRAM trim after batch"
setting and defaults to `false`. With the low-latency default, the Whisper batch worker does not call
the process-wide `torch.cuda.empty_cache()` after the queue drains, preserving the warm allocator
state used by both Cohere and ReDimNet. If an operator explicitly enables the option, cleanup waits
until the queue has been empty for 250 ms, is serialized with the local GPU lock, and yields to newly
queued ASR work. Explicit model unload/free-memory operations and emergency CUDA-OOM recovery remain
separate from this automatic setting. The `start.bat` launcher additionally sets
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256` before launch to
reduce reserved-pool fragmentation (note: `expandable_segments` is ignored on Windows).

### Experimental fp8 Precision

The precision setting accepts an experimental `fp8` value that loads the model with
`FineGrainedFP8Config` while keeping `bf16` compute.

- `fp8` is gated on the Hugging Face `kernels` package being importable. If `kernels` is not
  installed, the path falls back to `bf16` (logged), because without `kernels` the model would
  load but inference would fail.
- The `kernels` package is intentionally not installed automatically (it can break model loading);
  install it manually with `pip install -U kernels` if you want real fp8.
- Because `fp8` uses `bf16` compute, it also avoids the fp16 attention-mask overflow that
  `int8_bnb` would otherwise hit on the Cohere ASR model.

The Cohere ASR model hardcodes a `-1e9` attention-mask fill value, which overflows fp16 (the
compute dtype `int8_bnb` forces) and used to raise `RuntimeError: value cannot be converted to
type c10::Half without overflow`. A guard on `torch.Tensor.masked_fill` now clamps such scalar
fill values to the tensor dtype's `finfo` bounds, so **`int8_bnb` works** (and saves the most
weight VRAM). In-range values and `fp32`/`bf16` are unaffected. `bf16` remains the most robust
choice for maximum compatibility.

Backend-specific exception:

- The Cohere ASR model is currently intentionally loaded with `attn_implementation="eager"` because the current Transformers integration for this model does not properly support `sdpa` or `flash_attention_2` yet.
- On Windows, the internal Cohere `transcribe()` compile path is disabled if no `cl.exe` is found.
- For gated Cohere models, the runtime loader now reuses the configured Hugging Face token for both direct `from_pretrained(...)` calls and a full fallback `snapshot_download(...)` when the local cache is incomplete.

Currently not included as a fixed integrated standard:

- explicit Triton-for-Windows setup
- mandatory Flash Attention setup
- the optional Hugging Face `kernels` package required for real `fp8` (intentionally not auto-installed)

Important runtime decision:

- The model loader is no longer dependent on `accelerate` as a mandatory dependency for standard operation.
- Models are moved directly to the target device in the single-GPU/CPU path instead of enforcing `device_map`.

## Admin Login and Client API Keys

The admin area is protected by an httpOnly browser session cookie after username/password login.

Principle:

- the default account on a fresh deploy is `admin` / `admin`
- first login requires a password change
- `/api/admin/auth/login`, `/logout`, `/whoami`, and `/change-password` manage the browser session
- all operational `/api/admin/...` routes require a valid session and block access until the first password change is complete
- `/api/admin/api-keys` lets admins create or delete client API keys
- the public transcription endpoints stay open while no client API keys exist; once at least one key exists, clients must send `X-API-Key`
- users, sessions, and client API keys are stored hashed/serialized in [`logs/genesis_whisper_secrets.json`](x:/dev/G3_WHISPER/logs/genesis_whisper_secrets.json)

## Environment Variables

The most important environment variables:

- `HUGGINGFACE_TOKEN`
  - required for gated Hugging Face models such as `CohereLabs/cohere-transcribe-03-2026` unless the token is saved in the admin UI
  - **Note:** Can now optionally be configured directly via the Admin Settings UI, which takes precedence over `.env`.
  - **Important:** A token entered only during a manual cache download is not sufficient for later runtime loading. The token must be persisted in Admin Settings or `.env`.
  - **Important:** The same saved token is now also used by the runtime loader itself when Cohere needs to complete missing snapshot files such as `tokenizer.model`.
- `DIA_SERVER_BASE_URL`
  - optional fallback for the DIA server URL; use `http://dia:7864` in the bundled Compose network
  - a URL saved in the Whisper admin UI takes precedence
- `DIA_SERVER_API_KEY`
  - optional fallback for a DIA client key created in the G3_DIA admin UI
  - the value is treated as write-only and a key saved in the Whisper admin UI takes precedence
  - it is sent only to DIA as `X-API-Key` and is never returned or logged by Whisper
- `GENESIS_GPU_LEASE_PATH`
  - optional path to a file lock shared by Whisper and DIA, for example `/app/gpu-coordination/gpu.lock` in Compose
  - when unset, each process keeps only its existing in-process GPU serialization

## Persistent Data and Logs

The server stores runtime data under [`logs`](x:/dev/G3_WHISPER/logs).

Important files:

- `logs/genesis_whisper_settings.json`
  - persisted server settings
- `logs/genesis_whisper_secrets.json`
  - admin users, browser sessions, and client API key metadata
- `logs/transcription_log.jsonl`
  - JSONL log of transcription requests

## Default Settings

The active default values come from [backend/genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_storage.py):

- `local_model`: `openai/whisper-large-v3-turbo`
- `local_gpu_device`: `auto`
- `local_model_precision`: `fp16`
- `local_model_cache_path`: `.\models`
- `transcription_language`: `auto`
- `batch_wait_time_ms`: `500`
- `batch_max_segments`: `16`
- `batch_max_audio_seconds`: `300.0`
- `cuda_memory_trim_after_batch`: `false`
- `huggingface_token`: empty string
- `dia_server_base_url`: empty string (falls back to `DIA_SERVER_BASE_URL`)
- `dia_api_key`: empty string (write-only; falls back to `DIA_SERVER_API_KEY`)

## Admin Endpoints

Active admin routes:

- `POST /api/admin/auth/login`
- `POST /api/admin/auth/logout`
- `GET /api/admin/auth/whoami`
- `POST /api/admin/auth/change-password`
- all of the following require a completed admin login session
- `GET /api/admin/api-keys`
- `POST /api/admin/api-keys`
- `DELETE /api/admin/api-keys/{key_id}`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `DELETE /api/admin/settings/dia-api-key`
- `POST /api/admin/dia/test`
- `GET /api/admin/models`
- `POST /api/admin/models/download`
- `POST /api/admin/models/delete`
- `GET /api/admin/stats`
- `GET /api/admin/queue`
- `POST /api/admin/benchmark`

Admin settings updates are partial merges: omitted fields retain their saved values, so
older clients do not erase newer DIA settings. The DIA server URL and API key can be
entered in the admin UI. The saved URL/key take precedence over their environment
fallbacks. The key is write-only in responses, a blank update preserves it, and only
`DELETE /api/admin/settings/dia-api-key` removes the saved value explicitly. The
connection test calls the configured DIA `GET /v2/capabilities` endpoint.

## API Endpoints

Active open routes:

- `POST /transcribe/`
- `POST /v2/audio/process`
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

### Legacy Password Hash Helper

[backend/genesis_whisper_password_hash.py](x:/dev/G3_WHISPER/backend/genesis_whisper_password_hash.py)

Purpose:

- generic PBKDF2-SHA256 hash generator
- not part of the active browser login flow

### Load Test

[backend/genesis_whisper_batch_load_test.py](x:/dev/G3_WHISPER/backend/genesis_whisper_batch_load_test.py)

Purpose:

- sends many parallel requests to the running server
- can validate admin access and include Queue snapshots by passing `--admin-password`
- helps measure throughput, batch behavior, and stability

## File Formats

The server accepts audio files and, via decoding, video files as sources. The exact robustness depends on the installed local audio/codec dependencies.

Notes:

- Formats directly readable by `soundfile` work without external tools.
- For many video/container formats and some audio codecs, `ffmpeg` on `PATH` is required for the fallback decode path.
- The `ffmpeg` fallback stages its input as a seekable temporary file. This also supports MP4/M4A/MOV files whose `moov` metadata is stored after the media payload instead of at the beginning.
- Public uploads and admin benchmark uploads are decoded from the spooled upload stream in a worker thread, avoiding an additional full-file RAM copy and blocking work on the async request loop.
- In v2 `diarization` mode, Whisper rewinds and streams that original spooled upload to DIA instead of materializing another complete in-memory copy.

The admin benchmark also accepts audio or video and uses the same audio loading path.

## Known Quirks and Specifics

- `python-multipart` is required for FastAPI form uploads and is part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `librosa` is required by parts of the local ASR stack and is therefore explicitly part of [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `httpx` provides the streamed DIA upstream client and `filelock` provides optional cross-process GPU coordination.
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
- The Cohere ASR model hardcodes a `-1e9` attention-mask value that overflows fp16; a `masked_fill` fp16 guard now clamps it, so `int8_bnb` (max weight-VRAM save) works. `bf16` is still the most robust choice; experimental `fp8` needs the optional `kernels` package.
- On Windows, Hugging Face may warn about degraded cache behavior without symlink support. Developer Mode or elevated execution improves this, but the cache still works without it.
- Existing 512-D voice profiles are incompatible with the new 192-D model space and are not converted automatically. Recreate each profile from its reference audio before using it with v2.
- Existing 512-D model/cache files are left on disk deliberately; removing them is an operator decision and not part of startup migration.
- `__pycache__` might reappear locally. This is normal and not source code.

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
