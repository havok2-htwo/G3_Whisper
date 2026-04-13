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

- no API key required

Protected admin routes:

- require the request header `X-Admin-Key`
- the persistent admin key is stored hashed in [`logs/genesis_whisper_secrets.json`](x:/dev/G3_WHISPER/logs/genesis_whisper_secrets.json)
- a temporary startup admin key can also be valid for a short TTL if the launcher generated one

Example header:

```http
X-Admin-Key: genesis_admin_xxxxxxxxxxxxxxxxxxxxxxxx
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
- `500`: model load or runtime error

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
      "id": "openai/whisper-base",
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

All endpoints in this section require `X-Admin-Key`.

### `GET /api/admin/keys`

Purpose:

- returns metadata for the active persistent admin key

Response:

```json
{
  "admin_key": {
    "id": "admin",
    "label": "Master Admin Key",
    "created_at": "2026-04-13T09:00:00+00:00",
    "last_used_at": "2026-04-13T09:05:00+00:00"
  }
}
```

### `POST /api/admin/keys`

Purpose:

- rotates the persistent admin key
- returns the new plaintext key exactly once

Response:

```json
{
  "key": {
    "id": "admin",
    "label": "Master Admin Key",
    "token": "genesis_admin_xxxxxxxxxxxxxxxxxxxxxxxx",
    "created_at": "2026-04-13T09:10:00+00:00"
  },
  "keys": {
    "admin_key": {
      "id": "admin",
      "label": "Master Admin Key",
      "created_at": "2026-04-13T09:10:00+00:00",
      "last_used_at": null
    }
  }
}
```

### `GET /api/admin/settings`

Purpose:

- returns current saved settings
- returns UI options for models, devices, and languages
- returns the currently loaded model identifier if a model is already in memory

Response shape:

```json
{
  "settings": {
    "local_model": "openai/whisper-base",
    "local_gpu_device": "auto",
    "local_model_cache_path": "",
    "transcription_language": "auto",
    "batch_wait_time_ms": 250,
    "batch_max_segments": 8,
    "batch_max_audio_seconds": 120.0
  },
  "options": {
    "models": [
      { "label": "Whisper Base", "value": "openai/whisper-base" }
    ],
    "devices": [
      { "label": "Auto (Empfohlen)", "value": "auto" }
    ],
    "languages": [
      { "label": "German (de)", "value": "de" }
    ]
  },
  "loaded_model_identifier": [
    "openai/whisper-base",
    "auto",
    ""
  ]
}
```

### `PUT /api/admin/settings`

Purpose:

- validates and persists admin settings
- reloads the model if model-related settings changed

Request body:

```json
{
  "local_model": "openai/whisper-base",
  "local_gpu_device": "auto",
  "local_model_cache_path": "",
  "transcription_language": "auto",
  "batch_wait_time_ms": 250,
  "batch_max_segments": 8,
  "batch_max_audio_seconds": 120.0
}
```

Response fields:

- `ok`
- `settings`
- `model_reloaded`
- `model_loaded`
- `options`

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
- Whisper chunking currently uses the energy-based speech detector
- Cohere uses the active saved language, and when the server setting is `auto`, it falls back to `de`
- admin benchmark and public transcription log into [`logs`](x:/dev/G3_WHISPER/logs)
