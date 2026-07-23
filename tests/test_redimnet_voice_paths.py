import math
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import torch
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
        self.assertEqual(vid.REDIMNET_MIN_WINDOW_SECONDS, 0.5)

    def test_recording_embedding_accepts_half_second_of_clean_speech(self) -> None:
        audio = np.full(round(vid.REDIMNET_SAMPLE_RATE * 0.5), 0.1, dtype=np.float32)

        def embed(windows):
            prepared = list(windows)
            self.assertEqual(len(prepared), 1)
            self.assertAlmostEqual(prepared[0].clean_duration_seconds, 0.5)
            return [
                vid.EmbeddedVoiceWindow(
                    vector=_unit_vector(0),
                    start_ms=None,
                    end_ms=None,
                    clean_duration_seconds=0.5,
                    quality=1.0,
                    stitched=False,
                )
            ]

        with (
            mock.patch.object(vid, "_detect_embedding_speech_segments", return_value=[(0, len(audio))]),
            mock.patch.object(vid, "embed_voice_windows", side_effect=embed),
        ):
            result = vid.generate_voice_vector(audio)

        self.assertEqual(result.shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)

    def test_recording_embedding_rejects_less_than_half_second_of_speech(self) -> None:
        audio = np.full(round(vid.REDIMNET_SAMPLE_RATE * 0.49), 0.1, dtype=np.float32)
        with mock.patch.object(
            vid,
            "_detect_embedding_speech_segments",
            return_value=[(0, len(audio))],
        ):
            with self.assertRaisesRegex(ValueError, "0.5 Sekunden"):
                vid.generate_voice_vector(audio)

    def test_generate_voice_vector_returns_one_l2_normalized_192_vector(self) -> None:
        audio = np.full(vid.REDIMNET_SAMPLE_RATE * 6, 0.1, dtype=np.float32)
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
            mock.patch.object(
                vid,
                "_detect_embedding_speech_segments",
                return_value=[(0, len(audio))],
            ),
            mock.patch.object(
                vid,
                "_iter_concatenated_segment_windows",
                return_value=iter(windows),
            ),
            mock.patch.object(vid, "embed_voice_windows", return_value=embedded),
        ):
            result = vid.generate_voice_vector(audio)

        self.assertEqual(result.shape, (192,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)
        self.assertGreater(float(result[0]), float(result[1]))

    def test_embedding_vad_matches_expected_padded_energy_regions(self) -> None:
        audio = np.zeros(vid.REDIMNET_SAMPLE_RATE * 10, dtype=np.float32)
        audio[10_000:40_000] = 0.1
        audio[70_000:120_000] = 0.05

        segments = vid._detect_embedding_speech_segments(audio)

        self.assertEqual(segments, [(7680, 42240), (67680, 121920)])

    def test_concatenated_vad_windows_match_materialized_audio_without_full_copy(self) -> None:
        rng = np.random.default_rng(13)
        audio = rng.normal(0.0, 0.08, 200_000).astype(np.float32)
        segments = [(100, 60_100), (80_000, 150_000), (160_000, 190_000)]
        materialized = np.concatenate([audio[start:end] for start, end in segments])

        streaming = list(vid._iter_concatenated_segment_windows(audio, segments))
        expected = vid.windows_from_audio(materialized)

        self.assertEqual(len(streaming), len(expected))
        for actual, reference in zip(streaming, expected):
            np.testing.assert_array_equal(actual.audio, reference.audio)
            self.assertEqual(actual.quality, reference.quality)
        # The first complete window lives inside one VAD segment and remains a
        # view of the decoded input instead of an audio-sized speech copy.
        self.assertTrue(np.shares_memory(streaming[0].audio, audio))

    def test_embedding_batches_lazy_iterable_and_preserves_order(self) -> None:
        generated = 0
        calls: list[tuple[int, int]] = []

        def windows():
            nonlocal generated
            for index in range(10):
                generated += 1
                yield vid.VoiceWindow(
                    audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.01 * (index + 1), dtype=np.float32),
                    start_ms=index * 3000,
                    end_ms=(index + 1) * 3000,
                )

        class FakeModel:
            def __call__(self, waveform):
                calls.append((len(waveform), generated))
                result = torch.zeros((len(waveform), vid.REDIMNET_EMBEDDING_DIMENSION))
                result[:, 0] = 1.0
                return result

        components = {
            "model": FakeModel(),
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "cache_root": None,
        }
        with (
            mock.patch.object(vid, "load_vid_model", return_value=True),
            mock.patch.dict(vid._redimnet_components, components, clear=True),
            mock.patch.object(vid, "shared_gpu_lease", return_value=nullcontext()),
        ):
            embedded = vid.embed_voice_windows(windows(), batch_size=4)

        self.assertEqual([size for size, _ in calls], [4, 4, 2])
        self.assertEqual(calls[0], (4, 4))
        self.assertEqual([item.start_ms for item in embedded], [index * 3000 for index in range(10)])

    def test_cuda_inference_uses_only_single_or_prepared_batch_shape(self) -> None:
        self.assertEqual(vid._inference_batch_size(1, "cuda", 16), 1)
        for current_size in range(2, 17):
            with self.subTest(current_size=current_size):
                self.assertEqual(vid._inference_batch_size(current_size, "cuda", 16), 16)
        self.assertEqual(vid._inference_batch_size(3, "cpu", 16), 3)

    def test_cuda_batch_padding_preserves_real_windows_and_zero_fills_tail(self) -> None:
        windows = [
            vid.VoiceWindow(
                audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.01 * (index + 1), dtype=np.float32),
                start_ms=index,
                end_ms=index + 1,
            )
            for index in range(3)
        ]

        prepared = vid._prepare_waveform_batch(windows, 16)

        self.assertEqual(prepared.shape, (16, vid.REDIMNET_WINDOW_SAMPLES))
        np.testing.assert_array_equal(prepared[0], windows[0].audio)
        np.testing.assert_array_equal(prepared[2], windows[2].audio)
        self.assertTrue(np.all(prepared[3:] == 0.0))

    def test_embedding_oom_backoff_retries_in_order_and_keeps_smaller_batch(self) -> None:
        calls: list[int] = []

        class OomAboveTwoModel:
            def __call__(self, waveform):
                calls.append(len(waveform))
                if len(waveform) > 2:
                    raise RuntimeError("CUDA out of memory")
                result = torch.zeros((len(waveform), vid.REDIMNET_EMBEDDING_DIMENSION))
                result[:, 1] = 1.0
                return result

        windows = [
            vid.VoiceWindow(
                audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.05, dtype=np.float32),
                start_ms=index,
                end_ms=index + 1,
            )
            for index in range(5)
        ]
        components = {
            "model": OomAboveTwoModel(),
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "cache_root": None,
        }
        with (
            mock.patch.object(vid, "load_vid_model", return_value=True),
            mock.patch.dict(vid._redimnet_components, components, clear=True),
            mock.patch.object(vid, "shared_gpu_lease", return_value=nullcontext()),
        ):
            embedded = vid.embed_voice_windows(iter(windows), batch_size=4)

        self.assertEqual(calls, [4, 2, 2, 1])
        self.assertEqual([item.start_ms for item in embedded], list(range(5)))
        self.assertTrue(all(float(item.vector[1]) == 1.0 for item in embedded))

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
