import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_api as legacy_api
from backend import genesis_whisper_server_vid as vid


def _unit_vector(index: int = 0) -> np.ndarray:
    vector = np.zeros(vid.REDIMNET_EMBEDDING_DIMENSION, dtype=np.float32)
    vector[index] = 1.0
    return vector


class _AsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _BatchManager:
    async def enqueue(self, **_kwargs):
        return type("BatchResult", (), {"text": "Hallo Welt", "batch_id": "batch-test"})()


class _CountingBatchManager:
    def __init__(self) -> None:
        self.segments: list[int] = []

    async def enqueue(self, **kwargs):
        index = int(kwargs["segment_index"])
        self.segments.append(index)
        return type(
            "BatchResult",
            (),
            {"text": f"Abschnitt {index + 1}.", "batch_id": "batch-long"},
        )()


class RedimNetVoicePathTests(unittest.TestCase):
    def test_public_model_metadata_is_pinned_192_dimensional_b6(self) -> None:
        metadata = vid.embedding_model_metadata()

        self.assertEqual(metadata["id"], "ReDimNet2-B6")
        self.assertEqual(metadata["variant"], "vb2+vox2+cnc2_v0-lm")
        self.assertEqual(metadata["release"], "v1.0.0")
        self.assertEqual(metadata["dimension"], 192)
        self.assertEqual(
            metadata["checkpoint_sha256"],
            "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
        )

    def test_generate_voice_vector_returns_one_l2_normalized_192_vector(self) -> None:
        audio = np.full(vid.REDIMNET_SAMPLE_RATE * 3, 0.1, dtype=np.float32)
        windows = [
            vid.VoiceWindow(audio=audio, clean_duration_seconds=3.0, quality=1.0),
            vid.VoiceWindow(audio=audio, clean_duration_seconds=3.0, quality=0.5),
        ]
        embedded = [
            vid.EmbeddedVoiceWindow(
                vector=_unit_vector(0),
                start_ms=0,
                end_ms=3000,
                clean_duration_seconds=3.0,
                quality=1.0,
                stitched=False,
            ),
            vid.EmbeddedVoiceWindow(
                vector=_unit_vector(1),
                start_ms=3000,
                end_ms=6000,
                clean_duration_seconds=3.0,
                quality=0.5,
                stitched=False,
            ),
        ]

        with (
            mock.patch.object(vid, "extract_speech_audio", return_value=audio),
            mock.patch.object(vid, "windows_from_audio", return_value=windows),
            mock.patch.object(vid, "embed_voice_windows", return_value=embedded),
        ):
            result = vid.generate_voice_vector(audio)

        self.assertEqual(result.shape, (192,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)
        self.assertGreater(float(result[0]), float(result[1]))

    def test_legacy_voice_ident_keeps_shape_but_returns_192_values(self) -> None:
        app = FastAPI()
        app.state.local_gpu_lock = _AsyncLock()
        app.state.whisper_batch_manager = _BatchManager()
        legacy_api.create_api(app)
        vector = np.full(192, 1.0 / math.sqrt(192), dtype=np.float32)
        audio = np.full(32000, 0.1, dtype=np.float32)
        settings = {
            "local_model": "openai/whisper-tiny",
            "local_gpu_device": "cpu",
            "local_model_cache_path": "",
            "transcription_language": "de",
            "local_model_precision": "fp32",
        }

        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=audio),
            mock.patch.object(legacy_api, "split_audio_for_whisper", return_value=[audio]),
            mock.patch.object(legacy_api, "generate_voice_vector", return_value=vector),
            mock.patch.object(legacy_api, "log_transcription"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/transcribe/",
                    files={"file": ("voice.wav", b"fixture", "audio/wav")},
                    data={"engine": "local", "voice_ident": "true"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["transcription"], "Hallo Welt")
        self.assertIn("voice_vector", payload)
        self.assertIn("voice_vector_duration_ms", payload)
        self.assertEqual(len(payload["voice_vector"]), 192)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in payload["voice_vector"])),
            1.0,
            places=5,
        )

    def test_legacy_voice_ident_keeps_the_full_chunked_asr_path(self) -> None:
        app = FastAPI()
        app.state.local_gpu_lock = _AsyncLock()
        batch_manager = _CountingBatchManager()
        app.state.whisper_batch_manager = batch_manager
        legacy_api.create_api(app)
        vector = _unit_vector(3)
        full_audio = np.full(vid.REDIMNET_SAMPLE_RATE * 65, 0.1, dtype=np.float32)
        chunks = [full_audio[: vid.REDIMNET_SAMPLE_RATE * 30], full_audio[vid.REDIMNET_SAMPLE_RATE * 30 :]]
        settings = {
            "local_model": "openai/whisper-tiny",
            "local_gpu_device": "cpu",
            "local_model_cache_path": "",
            "transcription_language": "de",
            "local_model_precision": "fp32",
        }

        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=full_audio),
            mock.patch.object(legacy_api, "split_audio_for_whisper", return_value=chunks),
            mock.patch.object(legacy_api, "generate_voice_vector", return_value=vector) as generate,
            mock.patch.object(legacy_api, "log_transcription"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/transcribe/",
                    files={"file": ("long.wav", b"fixture", "audio/wav")},
                    data={"engine": "local", "voice_ident": "true"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(batch_manager.segments, [0, 1])
        self.assertIn("Abschnitt 1", response.json()["transcription"])
        self.assertIn("Abschnitt 2", response.json()["transcription"])
        generate.assert_called_once_with(full_audio)

    def test_whisper_backend_contains_no_legacy_pyannote_embedding_loader(self) -> None:
        backend_root = Path(__file__).resolve().parents[1] / "backend"
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(backend_root.glob("*.py"))
        ).lower()

        self.assertNotIn("pyannote/embedding", source)
        self.assertNotIn("pyannote.audio", source)


if __name__ == "__main__":
    unittest.main()
