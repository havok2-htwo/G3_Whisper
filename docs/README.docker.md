# G3 Voice — Docker deployment

`docker-compose.yml` in this repo runs **two** GPU services:

| Service     | Description                                | URL                  | Port |
|-------------|--------------------------------------------|----------------------|------|
| `whisper`   | GENESIS ASR + speaker diarization + ReDimNet2 embeddings | `http://<host>:7861` | 7861 |
| `omnivoice` | G3 OmniVoice TTS + voice cloning            | `http://<host>:8091` | 8091 |

Whisper uses **CUDA 12.8** wheels, OmniVoice the pinned **CUDA 13.0** stack — each
container is self-contained, so the differing CUDA versions do not conflict. Both GPU
services share the single NVIDIA GPU (verified on an RTX 5090 / `sm_120`).

The speaker diarization (`POST /v2/audio/process` with mode `diarization`) runs inside the
Whisper container (pyannote segmentation-3.0 network + ReDimNet2 windows + clustering).
The former separate `dia` service (G3_DIA) is no longer part of the stack; an old `.env`
that still sets `DIA_SERVER_BASE_URL` / `DIA_SERVER_API_KEY` is ignored. Diarization needs
the segmentation-3.0 checkpoint, see [Diarization checkpoint](#diarization-checkpoint).

Each admin dashboard (`/admin`) is behind a **username/password login** — default
`admin` / `admin`, with a forced password change on first login. The public processing
APIs stay open until you create an API key in the dashboard; once a key exists callers
must send a valid `X-API-Key` header (usage is tracked per key).

## Layout — two sibling repos

This compose builds all services and expects the `g3_omnivoice` repo **next to** this one:

```
<parent>/
├─ g3_whisper/      <- this repo; run docker compose from here
│  ├─ docker-compose.yml
│  └─ .env
└─ g3_omnivoice/    <- clone of the g3_omnivoice repo
```

## Host prerequisites (Ubuntu)

1. **Docker Engine + Compose v2 (>= 2.24.0)** — the `.env` `required:` syntax needs it;
   check with `docker compose version`. Install: `curl -fsSL https://get.docker.com | sh`
2. **NVIDIA driver** supporting your GPU (RTX 5090 needs a recent one).
3. **NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi   # verify
   ```

## First run

```bash
git clone -b main https://dev.it-breitenstein.de/ai-jointventure/g3_whisper.git
git clone -b main https://dev.it-breitenstein.de/ai-jointventure/g3_omnivoice.git
cp g3_whisper/.env.example g3_whisper/.env      # set HUGGINGFACE_TOKEN
cd g3_whisper
docker compose up -d --build
docker compose logs -f
```

## Updating

```bash
for r in g3_whisper g3_omnivoice; do git -C "$r" checkout main && git -C "$r" pull --ff-only; done
cd g3_whisper
docker compose up -d --build --remove-orphans   # --remove-orphans stops the old g3-dia container once
```

Saved admin settings live in the `*_logs` / `*_data` volumes and survive the rebuild.

> **First start is slow.** No models are baked into the images; each service downloads
> them into a named volume on first boot (the default ASR model Cohere Transcribe and
> ReDimNet2-B6 during startup warmup; OmniVoice
> `k2-fsa/OmniVoice` ~3 GB). The
> server only accepts connections after warmup, which is why the healthchecks use a long
> `start_period`.
>
> Cohere Transcribe is gated: set `HUGGINGFACE_TOKEN` in `.env` (or in the admin settings)
> and accept the license at <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>
> with that account. Without it the warmup only logs a hint and the server still starts;
> transcription works once the token is saved.

## Diarization checkpoint

The diarization reads the pyannote **segmentation-3.0** checkpoint (about 6 MB) from the
Whisper model cache. It is **never downloaded automatically**, and the `whisper_models`
volume does not contain it after a fresh deploy: until it is provided, diarization requests
answer `503 MODEL_UNAVAILABLE`, `GET /v2/capabilities` reports
`"diarization": {"configured": false, ...}` and live `speaker_turns` are not offered. All
other modes work.

Accept the license of the gated `pyannote/speaker-diarization-community-1` (or
`pyannote/segmentation-3.0`) on Hugging Face with the account of `HUGGINGFACE_TOKEN`, then
fetch the file once into the volume (the container's model cache is `/app/models`):

```bash
docker compose exec whisper python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('pyannote/speaker-diarization-community-1', 'segmentation/pytorch_model.bin', cache_dir='/app/models'))"
docker compose restart whisper   # optional: the next request finds the file anyway, a restart also warms it up
```

`HF_TOKEN` from `.env` is used for the download. Alternatively copy a
`pytorch_model.bin` into the volume and set its path as **Segmentation checkpoint**
(`segmentation_model_path`) in the admin settings.

Settings files from the time with the separate DIA service still contain
`diarization_engine` / `dia_server_base_url` / `dia_api_key`; they load normally, the keys are
ignored and the next save drops them.

The public embedding model is fixed; there is no model selector and no
`embedding_space_id`. All new and legacy public voice vectors are 192-D L2-normalized
ReDimNet2-B6 LM values. Existing 512-D profiles must be regenerated from reference
audio. Old model files are not deleted automatically from an existing volume.

## Volumes (persist across rebuilds)

| Volume             | Mount         | Contents                                 |
|--------------------|---------------|------------------------------------------|
| `whisper_models`   | `/app/models` | ASR and verified ReDimNet2 model cache   |
| `whisper_logs`     | `/app/logs`   | Settings JSON + transcription log        |
| `omnivoice_models` | `/app/models` | OmniVoice HF model cache                  |
| `omnivoice_data`   | `/app/data`   | Runtime settings, voice profiles, secrets|

The volumes `dia_models`, `dia_logs` and `gpu_coordination` of the former DIA service are no
longer used; after the update `docker volume rm g3-voice_dia_models g3-voice_dia_logs
g3-voice_gpu_coordination` frees them (optional).

Pre-seeding a volume from an existing local cache (skips the first download):

```bash
docker volume create g3-voice_whisper_models
docker run --rm -v g3-voice_whisper_models:/dest -v /path/to/models:/src \
  alpine sh -c "cp -a /src/. /dest/ && chown -R 1000:1000 /dest"
```

> The `chown -R 1000:1000` is required — a pre-populated volume keeps the host files'
> ownership, and the containers run as the non-root uid 1000, which must write the HF
> cache/locks. Apply the same chown to any pre-seeded volume (`omnivoice_models`,
> `omnivoice_data`, …).

## Common commands

```bash
docker compose up -d --build     # build + start
docker compose ps                # status (incl. health)
docker compose logs -f whisper   # follow one service
docker compose down              # stop (keeps volumes/models)
docker compose down -v           # stop + delete volumes (re-download next start)
```

## Notes

- **torch.compile**: both images ship `build-essential`, so the Cohere transcribe path and
  OmniVoice's `compile_model` (on by default) can JIT their CUDA kernels. Expect a one-time
  compile delay on the first request after a cold start.
- **GPU sharing**: both services share the same GPU. Whisper serializes its diarization,
  ASR, and ReDim phases under one GPU lock. OmniVoice
  enforces a VRAM budget (`OMNIVOICE_TTS_VRAM_BUDGET_MB`, default 10000); lower it if
  you run other GPU workloads.
- **`could not select device driver "nvidia"`** → NVIDIA Container Toolkit not installed or
  Docker not restarted after `nvidia-ctk runtime configure`.
