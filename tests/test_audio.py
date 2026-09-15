import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException

from backend import genesis_whisper_server_audio as audio


class _TrackingBytesIO(io.BytesIO):
    def __init__(self, initial_bytes: bytes):
        super().__init__(initial_bytes)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("The whole upload must not be read into one bytes object.")
        return super().read(size)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for container decoder tests")
class AudioDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-tests-")
        cls.m4a_path = Path(cls.temp_dir.name) / "tail-moov.m4a"
        subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=0.25",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-y",
                str(cls.m4a_path),
            ],
            check=True,
        )
        cls.m4a_bytes = cls.m4a_path.read_bytes()
        if cls.m4a_bytes.find(b"mdat") >= cls.m4a_bytes.find(b"moov"):
            raise AssertionError("The regression fixture must store moov after mdat.")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_tail_moov_m4a_decodes_from_bytes(self) -> None:
        decoded = audio.load_audio_bytes(self.m4a_bytes, "tail-moov.m4a")

        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.ndim, 1)
        self.assertGreater(len(decoded), 3000)
        self.assertLess(len(decoded), 5000)
        peak = float(np.max(np.abs(decoded)))
        self.assertGreater(peak, 0.001)
        self.assertLess(peak, 1.0)

    def test_ffmpeg_fallback_honors_target_sample_rate(self) -> None:
        decoded = audio.load_audio_bytes(self.m4a_bytes, "tail-moov.m4a", target_sample_rate=8000)

        self.assertGreater(len(decoded), 1500)
        self.assertLess(len(decoded), 2500)

    def test_ffmpeg_fallback_copies_upload_in_bounded_chunks(self) -> None:
        upload = _TrackingBytesIO(self.m4a_bytes)

        with mock.patch.object(audio.sf, "read", side_effect=RuntimeError("force ffmpeg fallback")):
            decoded = audio.load_audio_file(upload, "tail-moov.m4a")

        self.assertGreater(len(decoded), 0)
        self.assertTrue(upload.read_sizes)
        self.assertLessEqual(max(upload.read_sizes), 1024 * 1024)

    def test_invalid_container_error_uses_original_filename(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            audio.load_audio_bytes(b"not an audio container", "broken.mov")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("broken.mov", raised.exception.detail)
        self.assertNotIn("genesis-whisper-input-", raised.exception.detail)


class ResamplingPathTests(unittest.TestCase):
    @staticmethod
    def _wav_bytes(samples: np.ndarray, source_rate: int) -> io.BytesIO:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, samples, source_rate, format="WAV", subtype="FLOAT")
        buffer.seek(0)
        return buffer

    def test_native_rate_wav_stays_on_the_soundfile_path(self) -> None:
        samples = np.random.default_rng(1).uniform(-0.3, 0.3, 16000 * 3).astype(np.float32)

        with mock.patch.object(audio, "_decode_audio_with_ffmpeg", side_effect=AssertionError("ffmpeg must not run")):
            decoded = audio.load_audio_file(self._wav_bytes(samples, 16000), "clip.wav")

        self.assertEqual(len(decoded), len(samples))
        self.assertLess(float(np.max(np.abs(decoded - samples))), 1e-6)

    def test_wav_with_other_rate_is_resampled_by_ffmpeg_when_available(self) -> None:
        samples = np.random.default_rng(2).uniform(-0.3, 0.3, 24000 * 3).astype(np.float32)
        expected = np.zeros(16000 * 3, dtype=np.float32)

        with (
            mock.patch.object(audio.shutil, "which", return_value="ffmpeg"),
            mock.patch.object(audio, "_decode_audio_with_ffmpeg", return_value=expected) as ffmpeg_decode,
        ):
            decoded = audio.load_audio_file(self._wav_bytes(samples, 24000), "clip.wav")

        self.assertIs(decoded, expected)
        ffmpeg_decode.assert_called_once()
        self.assertEqual(ffmpeg_decode.call_args.args[2], 16000)

    def test_chunked_resampling_fallback_matches_the_one_shot_converter_exactly(self) -> None:
        import samplerate as src

        source_rate = 24000
        samples = np.random.default_rng(7).uniform(-0.3, 0.3, source_rate * 13 + 777).astype(np.float32)

        with mock.patch.object(audio.shutil, "which", return_value=None):
            decoded = audio.load_audio_file(self._wav_bytes(samples, source_rate), "clip.wav")

        reference = src.resample(samples, 16000 / source_rate, audio.RESAMPLE_CONVERTER)
        self.assertEqual(len(decoded), len(reference))
        self.assertEqual(float(np.max(np.abs(decoded - reference))), 0.0)

    def test_resampling_releases_the_gil_between_chunks(self) -> None:
        import threading
        import time

        # 180 s -> 36 chunks; a one-shot call would give this thread ~1 tick.
        samples = np.random.default_rng(3).uniform(-0.3, 0.3, 24000 * 180).astype(np.float32)
        finished = threading.Event()

        def work() -> None:
            audio._resample_in_chunks(samples, 24000, 16000)
            finished.set()

        worker = threading.Thread(target=work)
        worker.start()
        ticks = 0
        while not finished.is_set():
            ticks += 1
            time.sleep(0.002)
        worker.join()

        # A one-shot libsamplerate call would freeze this thread for the whole
        # conversion; chunking must let it run between chunks.
        self.assertGreater(ticks, 18)


if __name__ == "__main__":
    unittest.main()
