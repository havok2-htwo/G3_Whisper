import io
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException


def _normalize_audio_data(audio_data: np.ndarray) -> np.ndarray:
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.size == 0:
        return audio_data

    # Avoid np.abs(audio_data), which temporarily doubles memory usage for long
    # recordings.  The returned array is also normalized in place for the same
    # reason.
    min_val = float(np.min(audio_data))
    max_val = float(np.max(audio_data))
    max_val = max(abs(min_val), abs(max_val))
    if not np.isfinite(max_val):
        raise HTTPException(status_code=400, detail="Audiodatei enthaelt ungueltige Sample-Werte.")
    if max_val > 0.0:
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()
        audio_data /= max_val
    return audio_data


def _safe_temp_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if 1 < len(suffix) <= 11 and suffix[1:].isalnum():
        return suffix
    return ".bin"


@contextmanager
def _seekable_ffmpeg_input(audio_file: BinaryIO, filename: str) -> Iterator[str]:
    """Copy an upload to a named file so container demuxers can seek in it.

    FastAPI's UploadFile is seekable to Python, but rolled temporary files cannot
    reliably be reopened by ffmpeg on Windows.  A named, closed temporary copy is
    portable and fixes MP4/M4A/MOV files whose metadata is stored after the media.
    """

    temp_path: Optional[str] = None
    try:
        try:
            audio_file.seek(0, os.SEEK_SET)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="genesis-whisper-input-",
                suffix=_safe_temp_suffix(filename),
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                shutil.copyfileobj(audio_file, temp_file, length=1024 * 1024)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Konnte Upload '{filename}' nicht fuer ffmpeg vorbereiten: {exc}",
            ) from exc

        yield temp_path
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                # A failed cleanup must not discard an otherwise valid
                # transcription result. The OS temp cleanup can reclaim it.
                pass


def _decode_audio_with_ffmpeg(
    audio_file: BinaryIO,
    filename: str,
    target_sample_rate: int,
) -> np.ndarray:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dateiformat von '{filename}' wird von SoundFile nicht erkannt und ffmpeg ist nicht verfuegbar."
            ),
        )

    with _seekable_ffmpeg_input(audio_file, filename) as input_path:
        process = subprocess.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-v",
                "error",
                "-i",
                input_path,
                "-map",
                "0:a:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        return_code = process.returncode
        pcm_bytes = process.stdout
        error_text = process.stderr.decode("utf-8", errors="replace").strip()
        error_text = error_text.replace(input_path, filename).replace(input_path.replace("\\", "/"), filename)
        del process

    if return_code != 0 or not pcm_bytes:
        error_text = error_text or "Unbekannter ffmpeg-Fehler."
        raise HTTPException(status_code=400, detail=f"ffmpeg konnte '{filename}' nicht dekodieren: {error_text}")

    # Convert once and scale in place. This avoids another full-size float array
    # for long recordings.
    audio_data = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    del pcm_bytes
    audio_data /= 32768.0
    if audio_data.size == 0:
        raise HTTPException(status_code=400, detail=f"ffmpeg lieferte keine Audiodaten fuer '{filename}'.")
    return _normalize_audio_data(audio_data)


def load_audio_file(audio_file: BinaryIO, filename: str, target_sample_rate: int = 16000) -> np.ndarray:
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate muss groesser als 0 sein.")

    try:
        audio_file.seek(0, os.SEEK_SET)
        audio_data, samplerate = sf.read(audio_file, dtype="float32")
    except Exception:
        return _decode_audio_with_ffmpeg(audio_file, filename, target_sample_rate)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if samplerate != target_sample_rate:
        try:
            import samplerate as src

            ratio = target_sample_rate / samplerate
            audio_data = src.resample(audio_data, ratio, "sinc_best")
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="Audio hat falsche Samplerate. Bitte 'samplerate' (`pip install samplerate`) installieren.",
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fehler beim Resampling: {exc}") from exc

    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.size == 0:
        raise HTTPException(status_code=400, detail=f"Audiodatei '{filename}' enthaelt keine Audiodaten.")
    return _normalize_audio_data(audio_data)


def load_audio_bytes(audio_bytes: bytes, filename: str, target_sample_rate: int = 16000) -> np.ndarray:
    return load_audio_file(io.BytesIO(audio_bytes), filename, target_sample_rate)


def get_audio_duration_seconds(audio_data: Optional[np.ndarray], sample_rate: int = 16000) -> float:
    if audio_data is None or len(audio_data) == 0:
        return 0.0
    return len(audio_data) / float(sample_rate)
