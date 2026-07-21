# G3 Voice — Docker deployment

`docker-compose.yml` in this repo runs **three** GPU services:

| Service     | Description                                | URL                  | Port |
|-------------|--------------------------------------------|----------------------|------|
| `whisper`   | GENESIS Whisper ASR + speaker diarization  | `http://<host>:7861` | 7861 |
| `omnivoice` | G3 OmniVoice TTS + voice cloning            | `http://<host>:8091` | 8091 |
| `dia`       | GENESIS DIA speaker diarization             | `http://<host>:7864` | 7864 |

Whisper + DIA use **CUDA 12.8** wheels, OmniVoice the pinned **CUDA 13.0** stack — each
container is self-contained, so the differing CUDA versions do not conflict. The three
GPU services share the single NVIDIA GPU (verified on an RTX 5090 / `sm_120`).

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
cp g3_whisper/.env.example g3_whisper/.env      # set HUGGINGFACE_TOKEN (admin login is admin/admin)
cd g3_whisper
docker compose up -d --build
docker compose logs -f
```

> **First start is slow.** No models are baked into the images; each service downloads
> them from Hugging Face into a named volume on first boot (Whisper `whisper-large-v3-turbo`
> ~1.6 GB at warmup + diarization/Cohere lazily; OmniVoice `k2-fsa/OmniVoice` ~3 GB). The
> server only accepts connections after warmup, which is why the healthchecks use a long
> `start_period`.

## Volumes (persist across rebuilds)

| Volume             | Mount         | Contents                                 |
|--------------------|---------------|------------------------------------------|
| `whisper_models`   | `/app/models` | Whisper / diarization HF model cache     |
| `whisper_logs`     | `/app/logs`   | Settings JSON + transcription log        |
| `omnivoice_models` | `/app/models` | OmniVoice HF model cache                  |
| `omnivoice_data`   | `/app/data`   | Runtime settings, voice profiles, secrets|
| `dia_models`       | `/app/models` | DIA / pyannote HF model cache            |
| `dia_logs`         | `/app/logs`   | Settings JSON + diarization log + secrets|

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
- **GPU sharing**: both models live on the same GPU. OmniVoice enforces a VRAM budget
  (`OMNIVOICE_TTS_VRAM_BUDGET_MB`, default 24000); lower it if you run other GPU workloads.
- **`could not select device driver "nvidia"`** → NVIDIA Container Toolkit not installed or
  Docker not restarted after `nvidia-ctk runtime configure`.
