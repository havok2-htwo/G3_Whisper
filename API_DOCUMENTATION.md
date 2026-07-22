# API Documentation

This document describes the API surface that is currently implemented in the repository. If this file and the code diverge, the code wins until the documentation is updated in the same work step.

Primary server entry point:

- [backend/genesis_whisper_server.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server.py)

Relevant route implementations:

- [backend/genesis_whisper_server_api.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_api.py)
- [backend/genesis_whisper_server_v2.py](x:/dev/G3_WHISPER/backend/genesis_whisper_server_v2.py)
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

## Versioned Audio Processing API

### `POST /v2/audio/process`

This is the versioned API for transcription and voice processing. It accepts
`multipart/form-data` with these parts:

- `file` (required): an audio or video upload
- `request` (required): UTF-8 JSON with `schema_version: "2.0"` and a `mode`; the JSON part is limited to 16 MiB

The four modes are:

- `embedding`: returns exactly one 192-D recording-level ReDimNet2 embedding; it does not run ASR or DIA
- `transcript`: returns a cleaned transcript; it does not run DIA
- `transcript_embedding`: returns a cleaned transcript and exactly one 192-D recording-level embedding; it does not run DIA
- `diarization`: runs G3_DIA, transcribes exclusive speaker turns, matches optional known ReDimNet2 profiles, and returns unknown/unresolved speaker vectors

Only `diarization` contacts the configured DIA server. Recording-level embeddings in
the other modes use VAD, fixed three-second windows, batched inference, quality
weighting, and final L2 normalization. If such a recording contains multiple speakers,
the returned value is intentionally a mixed vector; these modes do not identify people.

All success responses contain `schema_version`, a generated `request_id`, `status`,
`mode`, `audio`, informative `models`, millisecond `timings_ms`, `result`, and
`warnings`. The same request ID is returned in the `X-Request-ID` response header.
`status` is `completed` when there are no warnings and `partial` when usable output has
warnings.

#### Embedding model contract

Every public voice vector, including legacy `voice_vector`, is an L2-normalized 192-D
vector from ReDimNet2-B6 LM `vb2+vox2+cnc2_v0`. The model contract is fixed to:

- official MIT-licensed repository: [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)
- release: `v1.0.0`
- compatible pinned source commit: `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`
- checkpoint: `b6-vb2+vox2+cnc2_v0-lm.pt`
- checkpoint SHA-256: `287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868`
- sample rate: 16 kHz
- output normalization: L2

There is no model-choice field and no client-managed `embedding_space_id`. The v2 API
is permanently bound to this model space. A future checkpoint/model-space change must
use a new API or embedding version rather than silently changing v2.

Existing 512-D profiles are not compatible. They must be regenerated from reference
audio; neither 512-D legacy vectors nor DIA-internal 256-D vectors are accepted as known
profiles.

#### `embedding`

Request:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@Sprache.m4a' \
  -F 'request={"schema_version":"2.0","mode":"embedding"};type=application/json'
```

Response (the vector is shortened here only for readability; the real array always has
192 values):

```json
{
  "schema_version": "2.0",
  "request_id": "c921f58d32894484b44699e24fccdf28",
  "status": "completed",
  "mode": "embedding",
  "audio": { "duration_ms": 12840 },
  "models": {
    "embedding": {
      "id": "ReDimNet2-B6",
      "variant": "vb2+vox2+cnc2_v0-lm",
      "release": "v1.0.0",
      "source_commit": "2a8d15f65b1dfb5d73fede2f11ee42bcccca3035",
      "checkpoint_sha256": "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
      "dimension": 192,
      "normalization": "l2",
      "sample_rate": 16000
    }
  },
  "timings_ms": { "decode": 93, "embedding": 217, "total": 314 },
  "result": { "embedding": { "vector": [0.0123, -0.0441, 0.0317] } },
  "warnings": []
}
```

#### `transcript`

Request:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@Sprache.m4a' \
  -F 'request={"schema_version":"2.0","mode":"transcript"};type=application/json'
```

Mode-specific result:

```json
{
  "transcript": { "text": "Hallo Welt." }
}
```

The full response wraps this object in the common success envelope and identifies the
active ASR model in `models.asr.id`.

#### `transcript_embedding`

Request:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@Sprache.m4a' \
  -F 'request={"schema_version":"2.0","mode":"transcript_embedding"};type=application/json'
```

Mode-specific result (vector shortened in this documentation):

```json
{
  "transcript": { "text": "Hallo Welt." },
  "embedding": { "vector": [0.0123, -0.0441, 0.0317] }
}
```

#### `diarization` request

The optional `diarization` object accepts:

- `expected_speakers`: exact DIA speaker count, integer `1..64`
- `known_speakers`: at most 64 profiles; each has a unique non-empty `id` (at most 128 UTF-8 bytes) and one or more `embeddings`

Every supplied embedding must contain exactly 192 finite numeric values and have a
non-zero norm. The server normalizes valid inputs before comparison. Duplicate IDs,
empty embedding lists, NaN/infinity, null vectors, 512-D vectors, and 256-D vectors are
HTTP 422 errors. Profile count and vectors are otherwise bounded by the 16 MiB request
JSON limit.

`expected_speakers` is the exact count passed to DIA. If omitted, DIA chooses the count
automatically; the number of known profiles becomes only the minimum speaker count.
Consequently `expected_speakers`, when supplied, cannot be lower than the number of
known profiles. A supplied profile is a client assertion that the person occurs in the
recording, but weak matching evidence is still never forced.

Example request JSON (`vector-...` denotes an actual array of exactly 192 numbers):

```json
{
  "schema_version": "2.0",
  "mode": "diarization",
  "diarization": {
    "expected_speakers": 5,
    "known_speakers": [
      {
        "id": "person-17",
        "embeddings": [
          ["vector-192-values-from-ReDimNet2"]
        ]
      }
    ]
  }
}
```

Copy-paste call with an exact count and no known profiles:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@meeting.m4a' \
  -F 'request={"schema_version":"2.0","mode":"diarization","diarization":{"expected_speakers":5,"known_speakers":[]}};type=application/json'
```

To send real profiles, save valid JSON containing full 192-value arrays as
`diarization-request.json` and send it as a form part:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@meeting.m4a' \
  -F 'request=@diarization-request.json;type=application/json'
```

Whisper streams the original spooled upload to `POST <DIA base URL>/v2/diarize`. The
DIA key is sent as `X-API-Key`. DIA supplies standard turns, exclusive turns, and
overlap intervals; its internal 256-D `community-1` embeddings are neither serialized
nor used for identity matching. Exclusive turns drive ASR. ReDimNet speaker vectors are
created only from non-overlapping speech after a 200 ms boundary margin, a minimum two
seconds of clean speech, three-second target windows, and loudness/clipping/finite-value
quality checks.

All valid candidate windows participate. Cosine components and iterative median/MAD
filtering remove near duplicates and outliers. Starting thresholds are component cosine `0.45`, dominant-component share
`0.60`, profile cosine `0.60`, support `0.60`, and stability margin `0.04`. Hungarian
matching enforces a global one-to-one assignment. Quality is one of `ready`,
`low_support`, `mixed_cluster`, or `insufficient_clean_speech`.

#### `diarization` response

Each transcript segment contains:

- `start_ms`, `end_ms`
- assigned `speaker_id`
- original `diarization_speaker_id`
- `speaker_kind`: `known`, `unknown`, or `unresolved`
- `text`
- `overlap`

Known matches use the supplied profile ID. Unknown and unresolved DIA speakers retain a
stable DIA-derived ID and return their cleaned vector collection for later enrollment.
Each such collection contains at most 64 entries: one `prototype`, followed by at most
63 diverse time-coded `representative` entries. The cap and representative selection
are deterministic.

Abridged response example (all vectors are shortened in this documentation):

```json
{
  "schema_version": "2.0",
  "request_id": "f17f383a917a4a9eac066d79f001f2cc",
  "status": "partial",
  "mode": "diarization",
  "audio": { "duration_ms": 90421 },
  "models": {
    "asr": { "id": "openai/whisper-large-v3-turbo" },
    "diarization": { "id": "pyannote/speaker-diarization-community-1" },
    "embedding": { "id": "ReDimNet2-B6", "dimension": 192, "normalization": "l2" }
  },
  "timings_ms": {
    "decode": 128,
    "diarization": 2100,
    "transcription": 3310,
    "embedding": 483,
    "total": 6044
  },
  "result": {
    "transcript": {
      "text": "Guten Morgen. Hallo zusammen.",
      "segments": [
        {
          "index": 0,
          "start_ms": 440,
          "end_ms": 3120,
          "speaker_id": "person-17",
          "diarization_speaker_id": "SPEAKER_00",
          "speaker_kind": "known",
          "text": "Guten Morgen.",
          "overlap": false
        },
        {
          "index": 1,
          "start_ms": 3310,
          "end_ms": 5900,
          "speaker_id": "SPEAKER_04",
          "diarization_speaker_id": "SPEAKER_04",
          "speaker_kind": "unknown",
          "text": "Hallo zusammen.",
          "overlap": false
        }
      ]
    },
    "speaker_counts": {
      "expected": 5,
      "detected": 5,
      "known_provided": 4,
      "known_assigned": 4,
      "unknown": 1,
      "unresolved": 0
    },
    "speaker_assignments": [
      {
        "diarization_speaker_id": "SPEAKER_00",
        "speaker_id": "person-17",
        "kind": "known",
        "embedding_status": "ready",
        "cosine_similarity": 0.812,
        "support": 0.91,
        "stability_margin": 0.15
      }
    ],
    "unknown_speakers": [
      {
        "speaker_id": "SPEAKER_04",
        "speaker_kind": "unknown",
        "embedding_status": "ready",
        "embeddings": [
          { "kind": "prototype", "vector": [0.03, -0.01, 0.04] },
          {
            "kind": "representative",
            "start_ms": 3310,
            "end_ms": 5900,
            "clean_duration_seconds": 2.59,
            "quality": 0.94,
            "stitched": false,
            "vector": [0.02, -0.02, 0.05]
          }
        ],
        "embeddings_truncated": false,
        "candidate_count": 3,
        "retained_count": 3,
        "discarded_outliers": 0,
        "purity": 1.0
      }
    ],
    "unresolved_speakers": [],
    "unresolved_known_speakers": []
  },
  "warnings": [
    {
      "code": "SPEAKER_EMBEDDING_QUALITY",
      "speaker_id": "SPEAKER_03",
      "status": "low_support"
    }
  ]
}
```

#### v2 errors

Errors use the same schema/version/request ID but `status: "failed"`:

```json
{
  "schema_version": "2.0",
  "request_id": "c8105099563840e28ee7ceba8dab6db8",
  "status": "failed",
  "mode": "diarization",
  "models": {},
  "timings_ms": { "total": 2 },
  "result": null,
  "warnings": [],
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Embedding muss genau 192 Werte enthalten.",
    "retryable": false
  }
}
```

`mode` is `null` when validation fails before a valid mode can be established; the
remaining envelope fields are still present.

Relevant status codes:

- `400`: missing multipart parts, invalid JSON, or audio decode failure
- `401`: missing/invalid Whisper client API key after public API protection is enabled
- `413`: `request` JSON exceeds 16 MiB
- `422`: invalid schema/mode, IDs, counts, profile vectors, or insufficient usable speech
- `503`: DIA is not configured for a `diarization` request, or a required local model is unavailable
- `502`: DIA rejected Whisper authentication, returned an upstream error, or violated the v2 response contract
- `504`: DIA request timeout
- `500`: unexpected local processing failure

The main DIA error codes are `DIA_NOT_CONFIGURED`, `DIA_AUTH_FAILED`,
`DIA_UPSTREAM_ERROR`, `DIA_INVALID_RESPONSE`, and `DIA_TIMEOUT`. Secrets are never
included in error bodies or logs.

#### Repetition filtering

Exact consecutive repetition filtering is enabled by default on this route whenever the
mode produces text. It is also enabled on legacy `/transcribe/`, OpenAI-compatible
`/v1/audio/transcriptions`, and the admin benchmark. Single-token patterns collapse
from five repetitions; 2-32-token patterns collapse from three repetitions. Matching is
Unicode token based, case-insensitive, whitespace/terminal-punctuation insensitive, and
never fuzzy. Each run is reduced to its first original occurrence without response
metadata or markers. Diarization applies it only inside each speaker turn.

Opt out for one request with:

```http
X-G3-Repetition-Filter: off
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
- legacy field names and response shape are unchanged, but `voice_vector` is now always a normalized 192-D ReDimNet2-B6 vector rather than a 512-D legacy vector
- this route never contacts DIA; multi-speaker audio therefore yields a mixed recording-level vector

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

The example vector is shortened for readability; the actual `voice_vector` array has
exactly 192 values. Exact-pattern repetition filtering is enabled by default and can be
disabled for this request with `X-G3-Repetition-Filter: off`.

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
- exact-pattern repetition filtering is enabled by default; `X-G3-Repetition-Filter: off` disables it without changing the selected response shape

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
    "huggingface_token": "hf_xxx",
    "dia_server_base_url": "http://dia:7864",
    "dia_server_base_url_effective": "http://dia:7864",
    "dia_server_base_url_source": "settings",
    "dia_api_key": "",
    "dia_api_key_configured": true,
    "dia_api_key_source": "settings"
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
  "huggingface_token": "hf_xxx",
  "dia_server_base_url": "http://dia:7864",
  "dia_api_key": "dia_xxx"
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
- updates are partial merges; omitted fields keep their saved value, so older admin clients cannot remove newer settings
- `dia_server_base_url` configures the DIA service used by diarization requests
- `dia_api_key` is write-only: a non-empty value replaces the saved key, while an omitted or empty value preserves it
- settings responses always return `dia_api_key` as an empty string and expose only `dia_api_key_configured` plus its `settings` / `environment` / `none` source
- saved DIA values take precedence over the `DIA_SERVER_BASE_URL` and `DIA_SERVER_API_KEY` environment fallbacks
- the key is sent upstream only in `X-API-Key` and is never returned or written to application logs
- DIA settings affect only v2 requests whose mode is `diarization`

### `DELETE /api/admin/settings/dia-api-key`

Purpose:

- explicitly deletes only the DIA API key saved in the settings file
- reports whether a `DIA_SERVER_API_KEY` environment fallback remains active
- cannot modify or reveal environment secrets

### `POST /api/admin/dia/test`

Purpose:

- tests the saved/environment DIA configuration or unsaved URL/key values from the admin form
- requests `<DIA base URL>/v2/capabilities` with the key in `X-API-Key`
- does not follow redirects and never includes the key or the upstream response body in its response

Optional request body:

```json
{
  "dia_server_base_url": "http://dia:7864",
  "dia_api_key": "dia_xxx"
}
```

Success response:

```json
{
  "ok": true,
  "base_url": "http://dia:7864",
  "status_code": 200,
  "message": "DIA server connection succeeded."
}
```

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
- applies the exact-pattern repetition filter after chunk joining; `X-G3-Repetition-Filter: off` disables it for this benchmark request

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
- v2 diarization rewinds and streams the original spooled upload to DIA without creating another full-file RAM copy
- Whisper chunking currently uses the energy-based speech detector
- Cohere uses the active saved language, and when the server setting is `auto`, it falls back to `de`
- For gated Cohere models, the runtime loader reuses the saved Hugging Face token and completes incomplete local snapshots before loading from the finished local snapshot.
- admin benchmark and public transcription log into [`logs`](x:/dev/G3_WHISPER/logs)
- the optional `GENESIS_GPU_LEASE_PATH` file lock can serialize CUDA phases with a colocated DIA process; without it only each service's in-process locks apply
