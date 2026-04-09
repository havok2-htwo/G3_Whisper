import io
import shutil
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException


def _decode_audio_with_ffmpeg(audio_bytes: bytes, filename: str) -> np.ndarray:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dateiformat von '{filename}' wird von SoundFile nicht erkannt und ffmpeg ist nicht verfuegbar."
            ),
        )

    process = subprocess.run(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.returncode != 0 or not process.stdout:
        error_text = process.stderr.decode("utf-8", errors="replace").strip() or "Unbekannter ffmpeg-Fehler."
        raise HTTPException(status_code=400, detail=f"ffmpeg konnte '{filename}' nicht dekodieren: {error_text}")

    audio_data = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio_data.size == 0:
        raise HTTPException(status_code=400, detail=f"ffmpeg lieferte keine Audiodaten fuer '{filename}'.")
    return audio_data


def load_audio_bytes(audio_bytes: bytes, filename: str, target_sample_rate: int = 16000) -> np.ndarray:
    audio_stream = io.BytesIO(audio_bytes)
    try:
        audio_data, samplerate = sf.read(audio_stream, dtype="float32")
    except Exception:
        return _decode_audio_with_ffmpeg(audio_bytes, filename)

    if samplerate != target_sample_rate:
        try:
            import samplerate as src

            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            ratio = target_sample_rate / samplerate
            audio_data = src.resample(audio_data, ratio, "sinc_best")
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="Audio hat falsche Samplerate. Bitte 'samplerate' (`pip install samplerate`) installieren.",
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fehler beim Resampling: {exc}") from exc

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return np.asarray(audio_data, dtype=np.float32)


def get_audio_duration_seconds(audio_data: Optional[np.ndarray], sample_rate: int = 16000) -> float:
    if audio_data is None or len(audio_data) == 0:
        return 0.0
    return len(audio_data) / float(sample_rate)
