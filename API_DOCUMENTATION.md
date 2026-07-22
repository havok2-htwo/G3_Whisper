# API Documentation

This document describes the API surface that is currently implemented in the repository. If this file and the code diverge, the code wins until the documentation is updated in the same work step.

Primary server entry point:

- [backend/genesis_whisper_server.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server.py)

Relevant route implementations:

- [backend/genesis_whisper_server_api.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_api.py)
- [backend/genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_admin.py)
- [backend/genesis_whisper_server_auth.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_auth.py)

## Base URL

Default local server URL:

- `http://127.0.0.1:7861`

OpenAPI / interactive docs:

- `GET /docs`
- `GET /openapi.json`

## Authentication

Open/public routes:

- no API key is required while no client API keys exist
- once at least one client API key exists, public transcription clients must send `X-API-Key`

Protected admin routes:

- use a username/password login and an httpOnly same-origin session cookie
- the default fresh-deploy account is `admin` / `admin`
- the first login forces a password change before operational admin routes are available
- users, sessions, and client API keys are stored in [`logs/genesis_whisper_secrets.json`](x:/dev/G3_WHISPER/logs/genesis_whisper_secrets.json)

Client API key example:

```http
X-API-Key: genesis_whisper_xxxxxxxxxxxxxxxxxxxxxxxx
```

## Public Transcription API

### `POST /transcribe/`

Purpose:

- native local upload/transcription endpoint
- optionally returns a voice embedding

Content type:

- `multipart/form-data`

Form fields:

- `file` (required): audio or video upload
- `engine` (optional): defaults to `local`; only `local` / `lokal` are accepted
- `voice_ident` (optional): boolean, defaults to `false`

Runtime behavior:

- `voice_ident=false` allows the request to use the internal batch queue
- Whisper models split the audio into speech-based chunks before queueing
- the Cohere model stays on a single whole-audio queue item
- `voice_ident=true` bypasses batching and runs under the local GPU lock
- only this native route can return `voice_vector`

Success response without voice embedding:

```json
{
  "transcription": "Hallo Welt",
  "total_duration_ms": 842,
  "transcription_duration_ms": 611
}
```

Success response with voice embedding:

```json
{
  "transcription": "Hallo Welt",
  "total_duration_ms": 1568,
  "transcription_duration_ms": 822,
  "voice_vector": [0.0123, -0.0441, 0.2319],
  "voice_vector_duration_ms": 731
}
```

Typical errors:

- `400`: upload could not be decoded or validated
- `500`: model load or runtime error; the server currently propagates the concrete loader message

### `GET /v1/models`

Purpose:

- lightweight OpenAI-style model listing for compatibility clients

Behavior:

- returns the currently active local model
- also returns the alias `whisper-1`

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/whisper-large-v3-turbo",
      "object": "model",
      "created": 1760000000,
      "owned_by": "genesis"
    },
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1760000000,
      "owned_by": "genesis"
    }
  ]
}
```

### `POST /v1/audio/transcriptions`

Purpose:

- OpenAI-compatible transcription route for clients that expect the Whisper API shape

Content type:

- `multipart/form-data`

Accepted form fields:

- `file` (required)
- `model` (optional, default `whisper-1`)
- `language` (optional)
- `prompt` (optional)
- `response_format` (optional, default `json`)
- `temperature` (optional, default `0.0`)

Compatibility notes:

- internally this route calls the native transcription path with `engine=local` and `voice_ident=false`
- it always uses the active local server model
- `response_format` affects the response body
- `model`, `language`, `prompt`, and `temperature` are currently accepted for compatibility but are not applied independently from the saved server settings yet

Response shapes:

- `response_format=json`

```json
{
  "text": "Hallo Welt"
}
```

- `response_format=text`

```text
Hallo Welt
```

- `response_format=verbose_json`

```json
{
  "task": "transcribe",
  "language": "de",
  "duration": 0.611,
  "text": "Hallo Welt",
  "segments": []
}
```

## Admin API

### `POST /api/admin/auth/login`

Purpose:

- verifies an admin username/password
- sets the `g3_whisper_session` httpOnly cookie
- reports whether a password change is required

Request:

```json
{
  "username": "admin",
  "password": "admin"
}
```

Response:

```json
{
  "username": "admin",
  "must_change_password": true
}
```

### `POST /api/admin/auth/change-password`

Purpose:

- changes the password for the logged-in admin user
- issues a fresh session cookie after the password change

Request:

```json
{
  "current_password": "admin",
  "new_password": "new-password"
}
```

Response:

```json
{
  "ok": true,
  "must_change_password": false
}
```

### `GET /api/admin/auth/whoami`

Purpose:

- validates the current session cookie
- returns the active admin user and password-change state

### `POST /api/admin/auth/logout`

Purpose:

- deletes the current admin session and clears the session cookie

### `GET /api/admin/api-keys`

Purpose:

- lists client API key metadata
- plaintext tokens are never returned after creation

### `POST /api/admin/api-keys`

Purpose:

- creates a client API key for public transcription callers
- returns the plaintext token exactly once

### `DELETE /api/admin/api-keys/{key_id}`

Purpose:

- deletes a client API key

All operational endpoints below require a completed admin login session.

### `GET /api/admin/settings`

Purpose:

- returns current saved settings
- returns UI options for models, devices, precisions, and languages
- returns the currently loaded model identifier if a model is already in memory

Response shape:

```json
{
  "settings": {
    "local_model": "openai/whisper-large-v3-turbo",
    "local_gpu_device": "auto",
    "local_model_precision": "fp16",
    "local_model_cache_path": ".\\models",
    "transcription_language": "auto",
    "batch_wait_time_ms": 500,
    "batch_max_segments": 32,
    "batch_max_audio_seconds": 300.0,
    "huggingface_token": "hf_xxx"
  },
  "options": {
    "models": [
      { "label": "Whisper Large v3 Turbo", "value": "openai/whisper-large-v3-turbo" }
    ],
    "devices": [
      { "label": "Auto (Empfohlen)", "value": "auto" }
    ],
    "precisions": [
      { "label": "INT8 (bitsandbytes)", "value": "int8_bnb" },
      { "label": "FP16", "value": "fp16" }
    ],
    "languages": [
      { "label": "German (de)", "value": "de" }
    ]
  },
  "models": [
    {
      "id": "CohereLabs/cohere-transcribe-03-2026",
      "label": "Cohere Transcribe 03/2026",
      "backend": "cohere_transcribe",
      "status": "partial",
      "local_path": null,
      "cache_path": "X:\\dev\\G3_WHISPER\\models\\models--CohereLabs--cohere-transcribe-03-2026",
      "storage_root": "X:\\dev\\G3_WHISPER\\models",
      "approx_size_gb": null,
      "size_on_disk_gb": 0.0,
      "error": null,
      "updated_at": null
    }
  ],
  "loaded_model_identifier": [
    "openai/whisper-large-v3-turbo",
    "auto",
    ".\\models",
    "fp16"
  ]
}
```

### `PUT /api/admin/settings`

Purpose:

- validates and persists admin settings
- reloads the model if model-related settings changed, including precision

Request body:

```json
{
  "local_model": "openai/whisper-large-v3-turbo",
  "local_gpu_device": "auto",
  "local_model_precision": "fp16",
  "local_model_cache_path": ".\\models",
  "transcription_language": "auto",
  "batch_wait_time_ms": 500,
  "batch_max_segments": 32,
  "batch_max_audio_seconds": 300.0,
  "huggingface_token": "hf_xxx"
}
```

Response fields:

- `ok`
- `settings`
- `model_reloaded`
- `model_loaded`
- `options`
- `models`

Notes:

- `huggingface_token` can be stored via this route and is then reused by later manual cache downloads and runtime model loads.

### `GET /api/admin/models`

Purpose:

- returns the cache-manager status list for the selected storage path

Query params:

- `storage_path` (optional): overrides the configured model storage root for the status query

Response shape:

```json
{
  "models": [
    {
      "id": "CohereLabs/cohere-transcribe-03-2026",
      "label": "Cohere Transcribe 03/2026",
      "backend": "cohere_transcribe",
      "status": "partial",
      "local_path": null,
      "cache_path": "X:\\dev\\G3_WHISPER\\models\\models--CohereLabs--cohere-transcribe-03-2026",
      "storage_root": "X:\\dev\\G3_WHISPER\\models",
      "approx_size_gb": null,
      "size_on_disk_gb": 0.0,
      "error": null,
      "updated_at": null
    }
  ]
}
```

Status values currently used:

- `missing`
- `partial`
- `downloading`
- `ready`
- `error`

Notes:

- A gated Cohere snapshot is only considered `ready` if the required remote-code files, tokenizer assets including `tokenizer.model`, and model weights are all present locally.

### `POST /api/admin/models/download`

Purpose:

- starts or resumes a background model download for the selected cache path

Request body:

```json
{
  "model_id": "CohereLabs/cohere-transcribe-03-2026",
  "storage_path": ".\\models",
  "huggingface_token": "hf_xxx"
}
```

Response fields:

- `job`
- `models`

Notes:

- If `huggingface_token` is omitted, the server falls back to the saved admin settings token and then to `HUGGINGFACE_TOKEN` / `HF_TOKEN`.

### `POST /api/admin/models/delete`

Purpose:

- removes the cached repository directory for a managed model from the selected storage path

Request body:

```json
{
  "model_id": "CohereLabs/cohere-transcribe-03-2026",
  "storage_path": ".\\models"
}
```

Response fields:

- `ok`
- `removed`
- `removed_path`
- `storage_root`
- `models`

### `GET /api/admin/stats`

Purpose:

- returns aggregate summary metrics and the recent transcription history

Response fields:

- `summary.total_requests`
- `summary.avg_total_duration_ms`
- `summary.avg_transcription_duration_ms`
- `history` with up to 25 recent log entries

History entry fields currently include:

- `timestamp`
- `source_ip`
- `engine`
- `model_id`
- `transcription_language`
- `total_duration_ms`
- `transcription_duration_ms`
- `voice_vector_duration_ms`
- `transcript`
- `voice_ident_requested`
- `batched`
- `segment_count`
- `batch_ids`

### `GET /api/admin/queue`

Purpose:

- returns the current queue/worker snapshot plus recent batch history

Response fields currently include:

- `worker_running`
- `queue_size`
- `pending_buffer_size`
- `active_batch_id`
- `active_batch_size`
- `active_batch_audio_seconds`
- `active_batch_started_at`
- `last_batch_completed_at`
- `last_batch_duration_ms`
- `last_error`
- `total_batches_processed`
- `total_segments_processed`
- `recent_batches`

Recent batch entry fields currently include:

- `batch_id`
- `timestamp`
- `batch_size`
- `audio_seconds`
- `duration_ms`
- `request_ids`
- `status`
- `error` (only on failed batches)

### `POST /api/admin/benchmark`

Purpose:

- runs repeated benchmark passes through the active ASR pipeline
- uses the same audio decode path as the public API

Content type:

- `multipart/form-data`

Form fields:

- `file` (required): audio or video upload
- `repeat_count` (optional): integer from `1` to `64`, default `1`

Response fields:

- `ok`
- `file_name`
- `workflow`
- `model_id`
- `transcription_language`
- `repeat_count`
- `audio_seconds`
- `total_audio_seconds`
- `chunks_per_run`
- `total_chunks`
- `batches_used`
- `total_wall_time_ms`
- `avg_wall_time_per_run_ms`
- `rtf`
- `transcripts_match`
- `transcript`
- `peak_vram_reserved_mb`
- `peak_vram_allocated_mb`

Workflow values currently used:

- `whisper_chunk_queue`
- `cohere_audio_batch`

## Frontend and SPA Delivery

The server also serves the frontend and public landing page:

- `GET /` returns the public landing page
- `GET /admin`
- `GET /admin/`
- `GET /admin/{full_path:path}` return the admin SPA entry
- `GET /{full_path:path}` serves built frontend assets if they exist, otherwise falls back to the landing page

Special case:

- paths under `/api/` are not forwarded to the frontend fallback

## Runtime Notes

- audio is normalized to mono `float32` at `16 kHz`
- direct `soundfile` decode is preferred; `ffmpeg` is the fallback for unsupported formats
- the `ffmpeg` fallback uses a seekable temporary input, so MP4/M4A/MOV containers with metadata at the end are supported
- upload decoding runs outside the async request loop and does not first copy the complete upload into a second in-memory bytes object
- Whisper chunking currently uses the energy-based speech detector
- Cohere uses the active saved language, and when the server setting is `auto`, it falls back to `de`
- For gated Cohere models, the runtime loader reuses the saved Hugging Face token and completes incomplete local snapshots before loading from the finished local snapshot.
- admin benchmark and public transcription log into [`logs`](x:/dev/G3_WHISPER/logs)
