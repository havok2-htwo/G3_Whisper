# syntax=docker/dockerfile:1.7
#
# GENESIS Whisper Server — GPU transcription/diarization service.
# Target host: Linux + NVIDIA driver + nvidia-container-toolkit (RTX 5090 / sm_120 OK).
# The image ships NO models: they are downloaded from Hugging Face into the mounted
# /app/models volume on first use (whisper-large-v3-turbo eagerly at warmup, the
# diarization / Cohere models lazily on the first request that needs them).

############################  1) Frontend build  ############################
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend

# Install deps from the lockfile first (better layer caching).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the React/Vite admin dashboard -> /build/frontend/dist
COPY frontend/ ./
RUN npm run build


############################  2) Python runtime  ############################
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Send Hugging Face caches (incl. the diarization pipeline that ignores the
    # explicit cache_dir) into the mounted models volume so they persist.
    HF_HOME=/app/models/.hf

# System libraries:
#   ffmpeg         -> decode mp3/m4a/video containers SoundFile cannot read; also brings the
#                     libav* shared libs that pyannote.audio 4.x's torchcodec dependency needs
#   libsndfile1    -> soundfile / librosa
#   libsamplerate0 -> the `samplerate` resampler
#   build-essential-> g++/gcc toolchain torch.compile / Inductor uses to JIT-compile the CUDA
#                     kernel wrappers on first inference (Cohere transcribe() + the optional
#                     Whisper torch.compile). Without it Inductor raises "No working C++ compiler".
#   curl           -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsamplerate0 \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8 wheels (bundle their own CUDA + cuDNN runtime; the host driver is
# injected by nvidia-container-toolkit). Pinned to the verified dev stack
# (torch 2.11.0+cu128 / Python 3.13); cu128 supports Blackwell / sm_120 (RTX 5090).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Application Python dependencies.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Application code + the built frontend.
COPY backend/ ./backend/
COPY testaudio/ ./testaudio/
COPY --from=frontend /build/frontend/dist ./frontend/dist

# Entrypoint seeds a Linux-correct settings file on first boot (the built-in default
# cache path is the Windows literal ".\\models").
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run as non-root. Pre-create + own the volume mountpoints so the named volumes
# inherit `app` ownership on first mount.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/logs /app/frontend \
    && chown -R app:app /app
USER app

EXPOSE 7861
VOLUME ["/app/models", "/app/logs"]

# First boot downloads the warmup model, so give startup a long grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1200s --retries=5 \
    CMD curl -fsS http://localhost:7861/openapi.json >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "backend.genesis_whisper_server"]
