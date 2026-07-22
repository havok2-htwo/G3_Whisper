# G3 Voice — Docker deployment

`docker-compose.yml` in this repo runs **three** GPU services:

| Service     | Description                                | URL                  | Port |
|-------------|--------------------------------------------|----------------------|------|
| `whisper`   | GENESIS ASR + ReDimNet2 embedding/orchestration | `http://<host>:7861` | 7861 |
| `omnivoice` | G3 OmniVoice TTS + voice cloning            | `http://<host>:8091` | 8091 |
| `dia`       | GENESIS DIA speaker diarization             | `http://<host>:7864` | 7864 |

Whisper + DIA use **CUDA 12.8** wheels, OmniVoice the pinned **CUDA 13.0** stack — each
container is self-contained, so the differing CUDA versions do not conflict. The three
GPU services share the single NVIDIA GPU (verified on an RTX 5090 / `sm_120`).

Whisper calls DIA only for `POST /v2/audio/process` requests whose mode is
`diarization`. The `embedding`, `transcript`, and `transcript_embedding` modes do not
contact DIA. Whisper and DIA mount the same `gpu_coordination` volume and use
`GENESIS_GPU_LEASE_PATH=/app/gpu-coordination/gpu.lock`, so their CUDA-heavy phases are
serialized across the two containers.

Each admin dashboard (`/admin`) is behind a **username/password login** — default
`admin` / `admin`, with a forced password change on first login. The public processing
APIs stay open until you create an API key in the dashboard; once a key exists callers
must send a valid `X-API-Key` header (usage is tracked per key).

## Layout — two sibling repos

This compose builds all services and expects the `g3_omnivoice` and `g3_dia`
repos **next to** this one:

```
<parent>/
├─ g3_whisper/      <- this repo; run docker compose from here
│  ├─ docker-compose.yml
│  └─ .env
├─ g3_omnivoice/    <- clone of the g3_omnivoice repo
└─ g3_dia/          <- clone of the g3_dia repo
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
git clone https://dev.it-breitenstein.de/ai-jointventure/g3_whisper.git
git clone https://dev.it-breitenstein.de/ai-jointventure/g3_omnivoice.git
git clone https://dev.it-breitenstein.de/ai-jointventure/g3_dia.git
cp g3_whisper/.env.example g3_whisper/.env      # set HUGGINGFACE_TOKEN and optional DIA_SERVER_API_KEY
cd g3_whisper
docker compose up -d --build
docker compose logs -f
```

> **First start is slow.** No models are baked into the images; each service downloads
> them into a named volume on first boot (Whisper `whisper-large-v3-turbo` ~1.6 GB at
> warmup, ReDimNet2-B6 on first embedding, and Cohere lazily; OmniVoice
> `k2-fsa/OmniVoice` ~3 GB). The
> server only accepts connections after warmup, which is why the healthchecks use a long
> `start_period`.

## Whisper to DIA configuration

Compose defaults `DIA_SERVER_BASE_URL` to `http://dia:7864`. Create a client API key in
the G3_DIA admin UI and either enter the DIA URL/key in the Whisper admin settings or set
these environment values in `.env`:

```dotenv
DIA_SERVER_BASE_URL=http://dia:7864
DIA_SERVER_API_KEY=dia_xxx
```

Saved Whisper settings take precedence over environment fallbacks. The DIA key is
write-only: it is sent upstream as `X-API-Key`, is never returned or logged, and has a
separate delete action. The admin connection test calls `GET /v2/capabilities` on DIA.
Settings updates use partial-merge semantics so older admin clients cannot accidentally
erase the DIA fields.

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
| `dia_models`       | `/app/models` | DIA / pyannote HF model cache            |
| `dia_logs`         | `/app/logs`   | Settings JSON + diarization log + secrets|
| `gpu_coordination` | `/app/gpu-coordination` | Shared Whisper/DIA GPU lock file |

Pre-seeding a volume from an existing local cache (skips the first download):

```bash
docker volume create g3-voice_whisper_models
docker run --rm -v g3-voice_whisper_models:/dest -v /path/to/models:/src \
  alpine sh -c "cp -a /src/. /dest/ && chown -R 1000:1000 /dest"
```

> The `chown -R 1000:1000` is required — a pre-populated volume keeps the host files'
> ownership, and the containers run as the non-root uid 1000, which must write the HF
> cache/locks. Apply the same chown to any pre-seeded volume (`omnivoice_models`,
> `omnivoice_data`, `dia_models`, …).

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
  OmniVoice's optional `compile_model` can JIT their CUDA kernels. Expect a one-time
  compile delay on the first request after a cold start.
- **GPU sharing**: all services share the same GPU. Whisper serializes DIA, ASR, and
  ReDim phases, and Whisper/DIA additionally share the Compose file lease. OmniVoice
  enforces a VRAM budget (`OMNIVOICE_TTS_VRAM_BUDGET_MB`, default 24000); lower it if
  you run other GPU workloads.
- **`could not select device driver "nvidia"`** → NVIDIA Container Toolkit not installed or
  Docker not restarted after `nvidia-ctk runtime configure`.
