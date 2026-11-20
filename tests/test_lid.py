"""Language detection for Cohere + "auto" (genesis_whisper_server_lid). CPU only: the Whisper model is faked."""

from __future__ import annotations

import asyncio
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from backend import genesis_whisper_server_lid as lid
from backend import genesis_whisper_server_wxc as wxc
from backend.genesis_whisper_server_globals import COHERE_LANGUAGE_CODES, current_settings

# A Whisper-like vocabulary: the 14 Cohere languages plus languages Cohere cannot take.
_VOCAB = ["<|startoftranscript|>", "<|ru|>", "<|tr|>", *[f"<|{code}|>" for code in COHERE_LANGUAGE_CODES]]
_TOKEN_IDS = {token: index for index, token in enumerate(_VOCAB)}


class _FakeTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        return _TOKEN_IDS.get(token, self.unk_token_id)


class _FakeFeatureExtractor:
    def __call__(self, windows, sampling_rate, return_tensors):
        assert sampling_rate == 16000 and return_tensors == "pt"
        return SimpleNamespace(input_features=torch.zeros((len(windows), 80, 10)))


class _FakeModel:
    """Logits favour Russian (not a Cohere language), then ``preferred``; records batches and moves."""

    def __init__(self, preferred: str = "de", oom_above: int | None = None) -> None:
        self.preferred = preferred
        self.oom_above = oom_above
        self.batch_sizes: list[int] = []
        self.moves: list[tuple] = []

    def eval(self):
        return self

    def to(self, device=None, dtype=None):
        self.moves.append((str(device), dtype))
        return self

    def __call__(self, input_features, decoder_input_ids):
        batch = int(input_features.shape[0])
        if self.oom_above is not None and batch > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert decoder_input_ids.tolist() == [[_TOKEN_IDS["<|startoftranscript|>"]]] * batch
        self.batch_sizes.append(batch)
        logits = torch.zeros((batch, 1, len(_VOCAB)))
        logits[:, 0, _TOKEN_IDS["<|ru|>"]] = 20.0
        logits[:, 0, _TOKEN_IDS[f"<|{self.preferred}|>"]] = 6.0
        logits[:, 0, _TOKEN_IDS["<|en|>"]] = 2.0
        return SimpleNamespace(logits=logits)


def _result(language: str, probability: float) -> lid.LidResult:
    rest = (1.0 - probability) / (len(COHERE_LANGUAGE_CODES) - 1)
    probs = {code: (probability if code == language else rest) for code in COHERE_LANGUAGE_CODES}
    return lid.LidResult(language=language, probability=probability, probs=probs)


class _LidCase(unittest.TestCase):
    def setUp(self) -> None:
        settings = mock.patch.dict(
            current_settings,
            {
                "local_model_cache_path": "",
                "local_gpu_device": "cpu",
                "cohere_language_detection": True,
                "language_detection_model": "openai/whisper-base",
            },
            clear=True,
        )
        settings.start()
        self.addCleanup(settings.stop)
        components = dict(lid._lid_components)
        failure = dict(lid._lid_failure)
        lid._clear_components()
        lid._lid_failure.update({"key": None, "error": None, "at": 0.0})

        def restore() -> None:
            lid._lid_components.clear()
            lid._lid_components.update(components)
            lid._lid_failure.clear()
            lid._lid_failure.update(failure)

        self.addCleanup(restore)
        self.device = torch.device("cpu")
        device_patch = mock.patch.object(lid, "resolve_local_device", side_effect=lambda: self.device)
        device_patch.start()
        self.addCleanup(device_patch.stop)

    def use_model(self, model: _FakeModel):
        loader = mock.patch.object(
            lid, "_load_pretrained", return_value=(_FakeFeatureExtractor(), _FakeTokenizer(), model)
        )
        started = loader.start()
        self.addCleanup(loader.stop)
        return started


class RestrictedDetectionTests(_LidCase):
    def test_only_the_14_cohere_languages_can_win(self) -> None:
        self.use_model(_FakeModel(preferred="fr"))
        [result] = lid.detect_languages([np.zeros(16000 * 3, dtype=np.float32)])

        # Russian has by far the largest logit, but Cohere cannot be prompted with it.
        self.assertEqual(result.language, "fr")
        self.assertEqual(set(result.probs), set(COHERE_LANGUAGE_CODES))
        self.assertAlmostEqual(sum(result.probs.values()), 1.0, places=6)
        self.assertGreater(result.probability, 0.9)

    def test_windows_are_batched_into_few_encoder_calls(self) -> None:
        model = _FakeModel()
        self.use_model(model)
        results = lid.detect_languages([np.zeros(16000, dtype=np.float32)] * 10)
        self.assertEqual(len(results), 10)
        self.assertEqual(model.batch_sizes, [lid.LID_BATCH_SIZE, 10 - lid.LID_BATCH_SIZE])

    def test_out_of_memory_halves_the_batch_and_keeps_every_result(self) -> None:
        model = _FakeModel(oom_above=2)
        self.use_model(model)
        results = lid.detect_languages([np.zeros(16000, dtype=np.float32)] * 5)
        self.assertEqual([result.language for result in results], ["de"] * 5)
        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual(lid._lid_components["batch_size"], 2)

    def test_cpu_runs_fp32_and_the_model_follows_the_device_setting(self) -> None:
        model = _FakeModel()
        self.use_model(model)
        self.assertTrue(lid.load_lid_model())
        self.assertEqual(model.moves, [("cpu", torch.float32)])

        self.device = torch.device("cuda:1")
        with mock.patch.object(lid, "_release_cuda_pool"):
            self.assertTrue(lid.apply_lid_device_setting())
        self.assertEqual(model.moves[-1], ("cuda:1", torch.float16))
        self.assertEqual(str(lid._lid_components["device"]), "cuda:1")
        self.assertIs(lid._lid_components["dtype"], torch.float16)

    def test_a_new_model_choice_or_switching_off_unloads_without_loading(self) -> None:
        loader = self.use_model(_FakeModel())
        self.assertTrue(lid.load_lid_model())
        current_settings["language_detection_model"] = "openai/whisper-small"
        self.assertTrue(lid.apply_lid_device_setting())
        self.assertIsNone(lid._lid_components["model"])
        self.assertEqual(loader.call_count, 1)

        self.assertTrue(lid.load_lid_model())
        self.assertEqual(loader.call_args.args[0], "openai/whisper-small")
        current_settings["cohere_language_detection"] = False
        self.assertTrue(lid.apply_lid_device_setting())
        self.assertIsNone(lid._lid_components["model"])


class LoadFailureTests(_LidCase):
    def test_failed_load_warns_once_and_is_not_retried_on_every_request(self) -> None:
        loader = mock.patch.object(lid, "_load_pretrained", side_effect=OSError("no snapshot and offline"))
        load = loader.start()
        self.addCleanup(loader.stop)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            for _ in range(3):
                with self.assertRaises(lid.LanguageDetectionUnavailable):
                    lid.detect_languages([np.zeros(16000 * 2, dtype=np.float32)])
        self.assertEqual(stderr.getvalue().count("[WARNUNG-LID]"), 1)
        self.assertEqual(load.call_count, 1)
        self.assertIn("no snapshot", lid.lid_status()["error"])

    def test_request_decision_falls_back_to_german_with_a_warning_and_never_raises(self) -> None:
        loader = mock.patch.object(lid, "_load_pretrained", side_effect=OSError("offline"))
        loader.start()
        self.addCleanup(loader.stop)
        with contextlib.redirect_stderr(io.StringIO()):
            decision = asyncio.run(
                lid.decide_item_languages(asyncio.Lock(), [np.zeros(16000 * 5, dtype=np.float32)] * 2)
            )
        self.assertEqual(decision.languages, ["de", "de"])
        self.assertEqual((decision.language, decision.source), ("de", "fallback"))
        self.assertEqual([warning["code"] for warning in decision.warnings], ["LANGUAGE_DETECTION_UNAVAILABLE"])
        self.assertEqual(decision.detection_block()["source"], "fallback")

    def test_any_detection_error_is_contained(self) -> None:
        with (
            mock.patch.object(lid, "detect_languages", side_effect=RuntimeError("CUDA error: device-side assert")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            decision = asyncio.run(lid.decide_item_languages(asyncio.Lock(), [np.zeros(16000 * 5, dtype=np.float32)]))
        self.assertEqual((decision.language, decision.source), ("de", "fallback"))
        self.assertEqual(len(decision.warnings), 1)


class AggregationTests(unittest.TestCase):
    def test_pooling_weights_log_probabilities_by_seconds(self) -> None:
        pooled = lid.pool_results([_result("en", 0.9), _result("de", 0.6)], [2.0, 20.0])
        self.assertEqual(pooled.language, "de")
        self.assertAlmostEqual(sum(pooled.probs.values()), 1.0, places=6)
        self.assertIsNone(lid.pool_results([None, _result("en", 0.9)], [3.0, 0.0]))

    def test_windows_of_one_item(self) -> None:
        self.assertEqual(len(lid.windows_for_item(np.zeros(16000 * 20, dtype=np.float32))), 1)
        two = lid.windows_for_item(np.zeros(16000 * 45, dtype=np.float32))
        self.assertEqual([len(window) for window in two], [16000 * 45 // 2] * 2)
        self.assertEqual(len(lid.windows_for_item(np.zeros(16000 * 200, dtype=np.float32))), 2)

    def test_low_confidence_chunks_take_the_duration_weighted_majority(self) -> None:
        results = [_result("en", 0.95), _result("fr", 0.4), _result("de", 0.9), _result("de", 0.92), None]
        seconds = [25.0, 25.0, 10.0, 10.0, 0.5]
        # en 25 s beats de 20 s among the confident chunks.
        self.assertEqual(lid.smooth_chunk_languages(results, seconds), ["en", "en", "de", "de", "en"])

    def test_without_any_confident_chunk_the_majority_of_all_still_decides(self) -> None:
        results = [_result("en", 0.4), _result("fr", 0.45), _result("en", 0.3)]
        self.assertEqual(lid.smooth_chunk_languages(results, [10.0, 15.0, 10.0]), ["en", "en", "en"])

    def test_no_result_at_all_falls_back(self) -> None:
        self.assertEqual(lid.smooth_chunk_languages([None, _result("en", 0.99)], [5.0, 0.5]), ["de", "de"])

    def test_speakers_detected_majority_and_fallback(self) -> None:
        resolved = lid.resolve_speaker_languages(
            {"A": _result("en", 0.97), "B": _result("de", 0.9), "C": _result("fr", 0.3), "D": None},
            {"A": 120.0, "B": 40.0, "C": 30.0, "D": 0.4},
        )
        self.assertEqual(
            resolved,
            {"A": ("en", "detected"), "B": ("de", "detected"), "C": ("en", "majority"), "D": ("en", "majority")},
        )
        self.assertEqual(
            lid.resolve_speaker_languages({"A": _result("en", 0.3), "B": None}, {"A": 30.0, "B": 9.0}),
            {"A": ("de", "fallback"), "B": ("de", "fallback")},
        )


class SubmitByLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_job_per_language_and_results_in_segment_order(self) -> None:
        async def submit_job(segments, request_id, key):
            return [SimpleNamespace(text=f"{key[3]}:{int(segment[0])}") for segment in segments]

        manager = SimpleNamespace(submit_job=mock.AsyncMock(side_effect=submit_job))
        segments = [np.full(10, index, dtype=np.float32) for index in range(4)]
        key = ("CohereLabs/cohere-transcribe-03-2026", "cuda:0", "", "auto", "fp16")
        results = await lid.submit_by_language(manager, segments, key, ["de", "en", "de", "en"])
        self.assertEqual([result.text for result in results], ["de:0", "en:1", "de:2", "en:3"])
        self.assertEqual(sorted(call.args[2][3] for call in manager.submit_job.await_args_list), ["de", "en"])

    async def test_a_single_language_is_one_job_with_the_same_segment_list(self) -> None:
        manager = SimpleNamespace(submit_job=mock.AsyncMock(return_value=[]))
        segments = [np.zeros(10, dtype=np.float32)]
        key = ("m", "cpu", "", "auto", "fp16")
        await lid.submit_by_language(manager, segments, key, ["fr"], "request-1")
        manager.submit_job.assert_awaited_once()
        self.assertIs(manager.submit_job.await_args.args[0], segments)
        self.assertEqual(manager.submit_job.await_args.args[1:], ("request-1", ("m", "cpu", "", "fr", "fp16")))

    async def test_a_failed_language_job_cancels_the_other(self) -> None:
        cancelled = asyncio.Event()

        async def submit_job(segments, request_id, key):
            if key[3] == "de":
                raise RuntimeError("Batch-Worker laeuft nicht.")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        manager = SimpleNamespace(submit_job=submit_job)
        with self.assertRaises(RuntimeError):
            await lid.submit_by_language(
                manager, [np.zeros(1)] * 2, ("m", "cpu", "", "auto", "fp16"), ["en", "de"]
            )
        self.assertTrue(cancelled.is_set())


class WxcLanguageChunkTests(unittest.TestCase):
    def test_one_language_is_exactly_the_old_chunking(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(20):
            starts = np.cumsum(rng.uniform(0.3, 9.0, size=30))
            regions = [(float(a), float(a + rng.uniform(0.2, 6.0))) for a in starts]
            regions = [region for index, region in enumerate(regions) if index == 0 or region[0] > regions[index - 1][1]]
            spans = [(a - 0.1, b + 0.05, "en") for a, b in regions[::2]]
            chunks, languages = wxc.build_language_superchunks(regions, spans)
            self.assertEqual(chunks, wxc.build_superchunks(regions))
            self.assertEqual(set(languages), {"en"})

    def test_no_chunk_crosses_a_language_change(self) -> None:
        regions = [(0.0, 9.0), (9.4, 40.0)]
        spans = [(0.0, 4.0, "de"), (4.0, 20.0, "en"), (20.0, 40.0, "de")]
        chunks, languages = wxc.build_language_superchunks(regions, spans)
        self.assertEqual(chunks, [(0.0, 4.0), (4.0, 20.0), (20.0, 40.0)])
        self.assertEqual(languages, ["de", "en", "de"])
        for (start, end), language in zip(chunks, languages):
            covering = {lang for a, b, lang in spans if a < end and b > start}
            self.assertEqual(covering, {language})

    def test_a_long_language_run_still_splits_at_the_target_length(self) -> None:
        regions = [(float(index), index + 0.8) for index in range(0, 60)]
        chunks, languages = wxc.build_language_superchunks(regions, [(0.0, 60.0, "fr")])
        self.assertEqual(chunks, wxc.build_superchunks(regions))
        self.assertTrue(all(end - start <= wxc.SUPERCHUNK_TARGET_S for start, end in chunks))

    def test_dia_slivers_and_uncovered_speech_join_a_neighbour(self) -> None:
        # A 0.2 s "en" sliver inside a German region and speech after the last turn.
        regions = [(0.0, 10.0)]
        spans = [(0.0, 5.0, "de"), (5.0, 5.2, "en"), (5.2, 8.0, "de")]
        chunks, languages = wxc.build_language_superchunks(regions, spans)
        self.assertEqual((chunks, languages), ([(0.0, 10.0)], ["de"]))


if __name__ == "__main__":
    unittest.main()
