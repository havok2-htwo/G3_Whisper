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
    async def submit_job(self, audio_segments, request_id, processing_key):
        return [
            type("BatchResult", (), {"text": "Hallo Welt", "batch_id": "batch-test"})()
            for _ in audio_segments
        ]


class _CountingBatchManager:
    def __init__(self) -> None:
        self.segments: list[int] = []

    async def submit_job(self, audio_segments, request_id, processing_key):
        results = []
        for index, _segment in enumerate(audio_segments):
            self.segments.append(index)
            results.append(
                type(
                    "BatchResult",
                    (),
                    {"text": f"Abschnitt {index + 1}.", "batch_id": "batch-long"},
                )()
            )
        return results


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

    def test_cuda_inference_pads_to_next_power_of_two_bucket(self) -> None:
        expected = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 7: 8, 8: 8}
        for current_size, bucket in expected.items():
            with self.subTest(current_size=current_size):
                self.assertEqual(vid._inference_batch_size(current_size, "cuda", 8), bucket)
        # After an OOM backoff the prepared maximum caps the bucket.
        self.assertEqual(vid._inference_batch_size(2, "cuda", 2), 2)
        self.assertEqual(vid._inference_batch_size(3, "cpu", 8), 3)
        with self.assertRaises(ValueError):
            vid._inference_batch_size(9, "cuda", 8)

    def test_batch_buckets_and_default_maximum(self) -> None:
        self.assertEqual(vid.REDIMNET_MAX_BATCH_SIZE, 8)
        self.assertEqual(vid.redimnet_batch_buckets(8), [1, 2, 4, 8])
        self.assertEqual(vid.redimnet_batch_buckets(4), [1, 2, 4])
        self.assertEqual(vid.redimnet_batch_buckets(1), [1])

    def test_cuda_warmup_prepares_every_bucket_largest_first(self) -> None:
        warmed: list[int] = []
        with mock.patch.object(vid, "_run_model_warmup", side_effect=lambda m, d, t, b: warmed.append(b)):
            prepared = vid._warm_model_shapes(object(), torch.device("cuda"), torch.float16)
        self.assertEqual(prepared, 8)
        self.assertEqual(warmed, [8, 4, 2, 1])

    def test_cuda_warmup_oom_halves_maximum_and_warms_smaller_buckets(self) -> None:
        warmed: list[int] = []

        def warm(model, device, dtype, batch):
            if batch > 4:
                raise RuntimeError("CUDA out of memory")
            warmed.append(batch)

        with (
            mock.patch.object(vid, "_run_model_warmup", side_effect=warm),
            mock.patch.object(vid.torch.cuda, "is_available", return_value=False),
        ):
            prepared = vid._warm_model_shapes(object(), torch.device("cuda"), torch.float16)
        self.assertEqual(prepared, 4)
        self.assertEqual(warmed, [4, 2, 1])

    def test_cuda_warmup_still_prepares_single_window_when_every_batch_ooms(self) -> None:
        warmed: list[int] = []

        def warm(model, device, dtype, batch):
            if batch > 1:
                raise RuntimeError("CUDA out of memory")
            warmed.append(batch)

        with (
            mock.patch.object(vid, "_run_model_warmup", side_effect=warm),
            mock.patch.object(vid.torch.cuda, "is_available", return_value=False),
        ):
            prepared = vid._warm_model_shapes(object(), torch.device("cuda"), torch.float16)
        self.assertEqual(prepared, 1)
        self.assertEqual(warmed, [1])

    def test_cuda_warmup_honours_a_smaller_configured_maximum(self) -> None:
        warmed: list[int] = []
        with mock.patch.object(vid, "_run_model_warmup", side_effect=lambda m, d, t, b: warmed.append(b)):
            prepared = vid._warm_model_shapes(object(), torch.device("cuda"), torch.float16, 4)
        self.assertEqual(prepared, 4)
        self.assertEqual(warmed, [4, 2, 1])

    def test_max_batch_setting_is_normalised_to_a_bucket(self) -> None:
        from backend import genesis_whisper_server_storage as storage

        self.assertEqual(storage.DEFAULT_SETTINGS["redimnet_max_batch_size"], 8)
        cases = [(8, 8), (4, 4), (2, 2), (1, 1), ("4", 4), (6, 4), (3, 2), (16, 8), (0, 1), (-2, 1), ("x", 8), (None, 8), (True, 8)]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(storage.normalize_settings({"redimnet_max_batch_size": value})["redimnet_max_batch_size"], expected)
        self.assertEqual(storage.normalize_settings({})["redimnet_max_batch_size"], 8)
        with mock.patch.dict(vid.current_settings, {"redimnet_max_batch_size": 4}, clear=True):
            self.assertEqual(vid.configured_max_batch_size(), 4)
        with mock.patch.dict(vid.current_settings, {}, clear=True):
            self.assertEqual(vid.configured_max_batch_size(), 8)

    def test_embedding_default_batch_follows_the_setting(self) -> None:
        calls: list[int] = []

        class FakeModel:
            def __call__(self, waveform):
                calls.append(len(waveform))
                result = torch.zeros((len(waveform), vid.REDIMNET_EMBEDDING_DIMENSION))
                result[:, 0] = 1.0
                return result

        windows = [
            vid.VoiceWindow(audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.05, dtype=np.float32), start_ms=index)
            for index in range(10)
        ]
        components = {"model": FakeModel(), "device": torch.device("cpu"), "dtype": torch.float32, "cuda_batch_size": 8}
        with (
            mock.patch.object(vid, "load_vid_model", return_value=True),
            mock.patch.dict(vid._redimnet_components, components, clear=True),
            mock.patch.object(vid, "shared_gpu_lease", return_value=nullcontext()),
            mock.patch.dict(vid.current_settings, {"redimnet_max_batch_size": 4}, clear=True),
        ):
            vid.embed_voice_windows(windows)

        self.assertEqual(calls, [4, 4, 2])

    def _loaded_components(self, model, device="cuda:0", maximum=8) -> dict:
        return {
            "model": model,
            "device": torch.device(device),
            "dtype": torch.float16,
            "cache_root": None,
            "cuda_batch_size": maximum,
            "max_batch_size": maximum,
        }

    def test_changed_max_batch_releases_the_pool_and_rewarms_the_loaded_model(self) -> None:
        model = mock.Mock()
        events: list[str] = []
        with (
            mock.patch.dict(vid._redimnet_components, self._loaded_components(model), clear=True),
            mock.patch.dict(vid.current_settings, {"redimnet_max_batch_size": 4, "local_gpu_device": "cuda:0"}, clear=True),
            mock.patch.object(vid, "resolve_local_device", return_value=torch.device("cuda:0")),
            mock.patch.object(vid, "_release_cuda_pool", side_effect=lambda: events.append("release")),
            mock.patch.object(vid, "_warm_model_shapes", side_effect=lambda m, d, t, n: events.append(f"warm{n}") or n),
        ):
            self.assertTrue(vid.load_vid_model())
            self.assertEqual(vid._redimnet_components["max_batch_size"], 4)
            self.assertEqual(vid._redimnet_components["cuda_batch_size"], 4)
            # Unchanged settings: no second re-warm.
            self.assertTrue(vid.load_vid_model())
            self.assertFalse(vid.apply_vid_settings())

        self.assertEqual(events, ["release", "warm4"])
        model.to.assert_not_called()

    def test_oom_backoff_alone_does_not_trigger_a_rewarm(self) -> None:
        components = self._loaded_components(mock.Mock())
        components["cuda_batch_size"] = 2
        with (
            mock.patch.dict(vid._redimnet_components, components, clear=True),
            mock.patch.dict(vid.current_settings, {"redimnet_max_batch_size": 8}, clear=True),
            mock.patch.object(vid, "resolve_local_device", return_value=torch.device("cuda:0")),
            mock.patch.object(vid, "_warm_model_shapes") as warm,
        ):
            self.assertFalse(vid.apply_vid_settings())
        warm.assert_not_called()

    def test_changed_gpu_device_moves_the_loaded_model(self) -> None:
        model = mock.Mock()
        with (
            mock.patch.dict(vid._redimnet_components, self._loaded_components(model), clear=True),
            mock.patch.dict(vid.current_settings, {"redimnet_max_batch_size": 8}, clear=True),
            mock.patch.object(vid, "resolve_local_device", return_value=torch.device("cuda:1")),
            mock.patch.object(vid, "_release_cuda_pool") as release,
            mock.patch.object(vid, "_warm_model_shapes", return_value=8),
        ):
            self.assertTrue(vid.apply_vid_settings())
            self.assertEqual(vid._redimnet_components["device"], torch.device("cuda:1"))
        model.to.assert_called_once_with(torch.device("cuda:1"))
        release.assert_called_once_with()

    def test_apply_settings_without_a_loaded_model_does_nothing(self) -> None:
        with (
            mock.patch.dict(vid._redimnet_components, {"model": None}, clear=True),
            mock.patch.object(vid, "_warm_model_shapes") as warm,
        ):
            self.assertFalse(vid.apply_vid_settings())
        warm.assert_not_called()

    def test_local_device_follows_the_gpu_setting(self) -> None:
        cases = {"cuda:1": "cuda:1", "cuda": "cuda:0", "auto": "cuda:0", "cpu": "cpu"}
        for configured, expected in cases.items():
            with (
                self.subTest(configured=configured),
                mock.patch.dict(vid.current_settings, {"local_gpu_device": configured}, clear=True),
                mock.patch.object(vid.torch.cuda, "is_available", return_value=True),
                mock.patch.object(vid.torch.cuda, "current_device", return_value=0),
            ):
                self.assertEqual(str(vid.resolve_local_device()), expected)
        with (
            mock.patch.dict(vid.current_settings, {"local_gpu_device": "cuda:1"}, clear=True),
            mock.patch.object(vid.torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(str(vid.resolve_local_device()), "cpu")

    def test_long_stream_runs_in_chunks_of_eight_with_padded_tail(self) -> None:
        calls: list[int] = []

        class FakeModel:
            def __call__(self, waveform):
                calls.append(len(waveform))
                result = torch.zeros((len(waveform), vid.REDIMNET_EMBEDDING_DIMENSION))
                result[:, 2] = 1.0
                return result

        windows = [
            vid.VoiceWindow(
                audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.05, dtype=np.float32),
                start_ms=index,
                end_ms=index + 1,
            )
            for index in range(19)
        ]
        components = {
            "model": FakeModel(),
            "device": torch.device("cuda"),
            "dtype": torch.float16,
            "cache_root": None,
            "cuda_batch_size": 8,
        }
        real_from_numpy = torch.from_numpy

        class _HostTensor:
            def __init__(self, array):
                self._tensor = real_from_numpy(array)

            def to(self, device=None, dtype=None):
                return self._tensor.to(dtype=dtype)

        with (
            mock.patch.object(vid, "load_vid_model", return_value=True),
            mock.patch.dict(vid._redimnet_components, components, clear=True),
            mock.patch.object(vid, "shared_gpu_lease", return_value=nullcontext()),
            mock.patch.object(vid, "_model_autocast", return_value=nullcontext()),
            mock.patch.object(vid.torch, "from_numpy", side_effect=_HostTensor),
        ):
            embedded = vid.embed_voice_windows(windows)

        self.assertEqual(calls, [8, 8, 4])
        self.assertEqual([item.start_ms for item in embedded], list(range(19)))

    def test_cuda_batch_padding_preserves_real_windows_and_zero_fills_tail(self) -> None:
        windows = [
            vid.VoiceWindow(
                audio=np.full(vid.REDIMNET_WINDOW_SAMPLES, 0.01 * (index + 1), dtype=np.float32),
                start_ms=index,
                end_ms=index + 1,
            )
            for index in range(3)
        ]

        prepared = vid._prepare_waveform_batch(windows, 4)

        self.assertEqual(prepared.shape, (4, vid.REDIMNET_WINDOW_SAMPLES))
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
