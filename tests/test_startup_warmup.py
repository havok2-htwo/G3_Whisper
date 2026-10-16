"""Startup warmup: a synthetic clip replaces the missing testaudio sample, so model load and graph capture still
happen at startup. Runs without a GPU: the ASR engine calls are mocked.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import genesis_whisper_server as server
from backend import genesis_whisper_server_local_asr_engine as engine
from backend.genesis_whisper_server_globals import current_settings

WARMUP_SETTINGS = {
    "local_model": "CohereLabs/cohere-transcribe-03-2026",
    "local_gpu_device": "cpu",
    "local_model_cache_path": "",
    "local_model_precision": "fp16",
    "transcription_language": "de",
}


class StartupWarmupTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(current_settings, WARMUP_SETTINGS, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _missing_sample(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-warmup-tests-")
        self.addCleanup(temp_dir.cleanup)
        return mock.patch.object(server, "WARMUP_SAMPLE", Path(temp_dir.name) / "testaudio" / "Testaudio_02.wav")

    def test_synthetic_clip_is_quiet_mono_float32_at_16_khz(self) -> None:
        audio = server._synthetic_warmup_audio()

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(len(audio), int(server.WARMUP_SYNTHETIC_SECONDS * 16000))
        self.assertLess(float(np.max(np.abs(audio))), 0.1)
        self.assertGreater(float(np.sqrt(np.mean(audio**2))), 0.01)
        np.testing.assert_array_equal(audio, server._synthetic_warmup_audio())

    def test_missing_sample_falls_back_to_the_synthetic_clip(self) -> None:
        with self._missing_sample():
            audio, label = server._warmup_audio()
        self.assertEqual(len(audio), int(server.WARMUP_SYNTHETIC_SECONDS * 16000))
        self.assertIn("synthetisch", label)

    def test_present_sample_is_still_used(self) -> None:
        temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-warmup-tests-")
        self.addCleanup(temp_dir.cleanup)
        sample = Path(temp_dir.name) / "Testaudio_02.wav"
        sample.write_bytes(b"RIFF")
        decoded = np.ones(1600, dtype=np.float32)
        with (
            mock.patch.object(server, "WARMUP_SAMPLE", sample),
            mock.patch("backend.genesis_whisper_server_audio.load_audio_bytes", return_value=decoded) as load,
        ):
            audio, label = server._warmup_audio()
        load.assert_called_once_with(b"RIFF", "Testaudio_02.wav")
        self.assertIs(audio, decoded)
        self.assertEqual(label, "Testaudio_02.wav")

    def test_warmup_loads_and_transcribes_without_the_sample_file(self) -> None:
        with (
            self._missing_sample(),
            mock.patch.object(engine, "load_local_asr_model", return_value=True) as load,
            mock.patch.object(engine, "transcribe_local_asr", return_value="") as transcribe,
            mock.patch.object(engine, "get_cohere_inference_patch_stats", return_value=None),
        ):
            server._warmup_local_asr()

        load.assert_called_once_with("CohereLabs/cohere-transcribe-03-2026", "cpu", "", "fp16")
        transcribe.assert_called_once()
        audio = transcribe.call_args.args[0]
        self.assertEqual(len(audio), int(server.WARMUP_SYNTHETIC_SECONDS * 16000))
        self.assertEqual(transcribe.call_args.kwargs, {"language": "de"})

    def test_failed_model_load_skips_the_transcription(self) -> None:
        with (
            self._missing_sample(),
            mock.patch.object(engine, "load_local_asr_model", return_value=False),
            mock.patch.object(engine, "transcribe_local_asr") as transcribe,
        ):
            server._warmup_local_asr()
        transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
