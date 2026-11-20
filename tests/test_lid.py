"""Language detection for Cohere + "auto" (genesis_whisper_server_lid). CPU only: the Whisper model is faked."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from backend import genesis_whisper_server_lid as lid
from backend import genesis_whisper_server_wxc as wxc
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_globals import COHERE_LANGUAGE_CODES, COHERE_MODEL_ID, current_settings

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
                "local_model": COHERE_MODEL_ID,
                "local_model_cache_path": "",
                "local_gpu_device": "cpu",
                "cohere_language_detection": True,
                "language_detection_model": "openai/whisper-base",
            },
            clear=True,
        )
        settings.start()
        self.addCleanup(settings.stop)
        # The download step is covered by PrepareTests; here the snapshot is "on disk".
        self.prepare_patch = mock.patch.object(lid, "prepare_lid_model", return_value=True)
        self.prepare_patch.start()
        self.addCleanup(self.prepare_patch.stop)
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

    def test_switching_the_asr_model_away_from_cohere_unloads_the_detector(self) -> None:
        self.use_model(_FakeModel())
        self.assertTrue(lid.load_lid_model())
        current_settings["local_model"] = "openai/whisper-large-v3-turbo"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(lid.apply_lid_device_setting())
        self.assertIsNone(lid._lid_components["model"])

    def test_a_failed_move_unloads_instead_of_leaving_a_split_model(self) -> None:
        model = _FakeModel()
        self.use_model(model)
        self.assertTrue(lid.load_lid_model())
        leases = []

        @contextlib.contextmanager
        def lease():
            leases.append("in")
            yield

        self.device = torch.device("cuda:0")
        with (
            mock.patch.object(lid, "shared_gpu_lease", lease),
            mock.patch.object(model, "to", side_effect=RuntimeError("CUDA out of memory")),
            mock.patch.object(lid, "_release_cuda_pool"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            lid.apply_lid_device_setting()
        # The move ran inside the cross-process lease, and its failure left nothing half-moved behind.
        self.assertEqual(leases, ["in"])
        self.assertIsNone(lid._lid_components["model"])
        self.assertIsNone(lid._lid_components["device"])


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


class PrepareTests(_LidCase):
    """The detector's download: outside every GPU lock, resumed when a snapshot is incomplete."""

    def setUp(self) -> None:
        super().setUp()
        self.prepare_patch.stop()
        temp = tempfile.TemporaryDirectory(prefix="lid-snapshot-")
        self.addCleanup(temp.cleanup)
        self.snapshot = os.path.join(temp.name, "snapshot")
        os.makedirs(self.snapshot)
        self.calls: list[bool] = []

        def download(model_id, cache_root, local_only):
            self.calls.append(local_only)
            if not local_only:
                self.write(*lid.LID_REQUIRED_FILES)
            return self.snapshot

        patcher = mock.patch.object(lid, "_download_snapshot", side_effect=download)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, *names: str) -> None:
        for name in names:
            with open(os.path.join(self.snapshot, name), "w", encoding="utf-8") as handle:
                handle.write("x")

    def test_an_interrupted_download_is_resumed_not_loaded_offline_for_good(self) -> None:
        # The feature extractor's config arrived, the weights did not (process restarted mid-download).
        self.write("preprocessor_config.json", "config.json")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(lid.prepare_lid_model())
        self.assertEqual(self.calls, [True, False])
        self.assertEqual(lid._local_snapshot("openai/whisper-base", ""), self.snapshot)

    def test_a_complete_snapshot_needs_no_network(self) -> None:
        self.write(*lid.LID_REQUIRED_FILES)
        self.assertTrue(lid.prepare_lid_model())
        self.assertEqual(self.calls, [True])

    def test_the_gpu_phase_never_downloads(self) -> None:
        self.write("preprocessor_config.json")
        with self.assertRaises(lid.LanguageDetectionUnavailable):
            lid._load_pretrained("openai/whisper-base", "")
        self.assertEqual(self.calls, [True])

    def test_detection_downloads_before_it_takes_the_gpu_lock(self) -> None:
        gpu_lock = asyncio.Lock()
        seen: list[tuple[str, bool]] = []

        def prepare() -> bool:
            seen.append(("prepare", gpu_lock.locked()))
            return True

        async def sliced(lock, items, function, size):
            seen.append(("gpu", lock.locked()))
            return [[]]

        with (
            mock.patch.object(lid, "prepare_lid_model", side_effect=prepare),
            mock.patch.object(lid, "run_sliced_gpu_phase", side_effect=sliced),
        ):
            asyncio.run(lid.detect_windows(gpu_lock, [np.zeros(16000, dtype=np.float32)]))
        self.assertEqual(seen, [("prepare", False), ("gpu", False)])

    def test_a_failed_download_is_reported_once_and_not_retried_at_once(self) -> None:
        def offline(model_id, cache_root, local_only):
            self.calls.append(local_only)
            if local_only:
                return None
            raise OSError("no network")

        stderr = io.StringIO()
        with mock.patch.object(lid, "_download_snapshot", side_effect=offline), contextlib.redirect_stderr(stderr):
            self.assertFalse(lid.prepare_lid_model())
            self.assertFalse(lid.prepare_lid_model())
            with self.assertRaises(lid.LanguageDetectionUnavailable):
                asyncio.run(lid.detect_windows(asyncio.Lock(), [np.zeros(16000, dtype=np.float32)]))
        # One network attempt; later calls only look at the disk until LID_RETRY_AFTER_SECONDS.
        self.assertEqual(self.calls, [True, False, True, True])
        self.assertEqual(stderr.getvalue().count("[WARNUNG-LID]"), 1)
        self.assertIn("no network", lid.lid_status()["error"])


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

    def test_a_short_region_of_another_language_joins_its_neighbours(self) -> None:
        # A 0.35 s English backchannel is its own Silero region between two German ones.
        regions = [(0.0, 12.0), (12.4, 12.75), (13.1, 20.0)]
        spans = [(0.0, 12.0, "de"), (12.4, 12.75, "en"), (13.1, 20.0, "de")]
        chunks, languages = wxc.build_language_superchunks(regions, spans)
        self.assertEqual((chunks, languages), (wxc.build_superchunks(regions), ["de"]))
        # A real English turn keeps its own chunk.
        regions = [(0.0, 12.0), (12.4, 15.0), (15.3, 20.0)]
        spans = [(0.0, 12.0, "de"), (12.4, 15.0, "en"), (15.3, 20.0, "de")]
        self.assertEqual(wxc.build_language_superchunks(regions, spans)[1], ["de", "en", "de"])


class SpeakerPoolTests(unittest.TestCase):
    def test_the_budget_keeps_the_inliers_before_a_long_raw_turn(self) -> None:
        inliers = [
            SimpleNamespace(quality=0.9, source_spans=None, start_ms=300000 + i * 3000, end_ms=303000 + i * 3000)
            for i in range(6)
        ]
        turns = [{"start_ms": 0, "end_ms": 45000}, {"start_ms": 300000, "end_ms": 318000}]
        spans = v2._speaker_lid_spans(SimpleNamespace(inliers=inliers), turns)
        # All 18 s of inliers, then 12 s of the longest turn; returned in time order.
        self.assertEqual(spans, [(0, 12000), (300000, 318000)])
        self.assertEqual(v2._span_seconds(spans), 30.0)

    def test_overlapping_candidates_count_once(self) -> None:
        inliers = [SimpleNamespace(quality=0.5, source_spans=[(1000, 3000), (5000, 6000)], start_ms=None, end_ms=None)]
        spans = v2._speaker_lid_spans(SimpleNamespace(inliers=inliers), [{"start_ms": 0, "end_ms": 8000}])
        self.assertEqual(spans, [(0, 8000)])


def _result(language: str, probability: float) -> lid.LidResult:
    rest = (1.0 - probability) / (len(COHERE_LANGUAGE_CODES) - 1)
    return lid.LidResult(language, probability, {code: (probability if code == language else rest) for code in COHERE_LANGUAGE_CODES})


class TurnRefinementTests(unittest.TestCase):
    def test_a_turn_overrides_its_speaker_only_when_long_and_confident_enough(self) -> None:
        cutoff = lid.LID_TURN_OVERRIDE_CONFIDENCE
        self.assertEqual(lid.turn_language(_result("de", cutoff), 3.0, "en"), ("de", True))
        self.assertEqual(lid.turn_language(_result("de", cutoff - 0.01), 3.0, "en"), ("en", False))
        self.assertEqual(lid.turn_language(_result("de", 0.99), lid.LID_TURN_MIN_SECONDS - 0.01, "en"), ("en", False))
        self.assertEqual(lid.turn_language(_result("en", 0.99), 3.0, "en"), ("en", False))
        self.assertEqual(lid.turn_language(None, 3.0, "en"), ("en", False))

    def test_the_calibrated_constants(self) -> None:
        self.assertEqual(lid.LID_TURN_MIN_SECONDS, 1.5)
        self.assertGreater(lid.LID_TURN_OVERRIDE_CONFIDENCE, lid.LID_MIN_CONFIDENCE)

    def test_long_audio_is_split_into_windows_of_at_most_30_s(self) -> None:
        windows = lid.split_windows(np.zeros(16000 * 70, dtype=np.float32))
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(len(window) <= 16000 * 30 for window in windows))
        self.assertEqual(sum(len(window) for window in windows), 16000 * 70)
        self.assertEqual(len(lid.split_windows(np.zeros(16000 * 10, dtype=np.float32))), 1)
        # Whole-audio items keep their cap.
        self.assertEqual(len(lid.windows_for_item(np.zeros(16000 * 100, dtype=np.float32))), lid.LID_MAX_WINDOWS_PER_ITEM)

    def test_region_pieces_split_only_long_regions(self) -> None:
        self.assertEqual(wxc.region_pieces([(0.0, 10.0), (12.0, 72.0)], 30.0), [(0.0, 10.0), (12.0, 42.0), (42.0, 72.0)])
        pieces = wxc.region_pieces([(0.0, 61.0)], 30.0)
        self.assertEqual(len(pieces), 3)
        self.assertEqual((pieces[0][0], pieces[-1][1]), (0.0, 61.0))


class PieceSmoothingTests(unittest.TestCase):
    SPANS = [(0.0, 10.0), (10.5, 12.0), (12.5, 13.5), (14.0, 24.0)]

    def test_confident_pieces_keep_their_language_others_take_the_nearest_anchor(self) -> None:
        results = [_result("de", 0.97), _result("en", 0.3), None, _result("en", 0.95)]
        # Piece 1 is weak (gap 0.5 s to de, 2.0 s to en), piece 2 is short (2.0 s to de, 0.5 s to en).
        self.assertEqual(lid.smooth_piece_languages(results, self.SPANS), ["de", "de", "en", "en"])

    def test_a_deviation_from_the_majority_needs_the_override_confidence(self) -> None:
        spans = [(0.0, 20.0), (20.5, 24.0), (24.5, 40.0)]
        doubtful = [_result("de", 0.97), _result("en", lid.LID_TURN_OVERRIDE_CONFIDENCE - 0.05), _result("de", 0.97)]
        sure = [_result("de", 0.97), _result("en", lid.LID_TURN_OVERRIDE_CONFIDENCE), _result("de", 0.97)]
        self.assertEqual(lid.smooth_piece_languages(doubtful, spans), ["de", "de", "de"])
        self.assertEqual(lid.smooth_piece_languages(sure, spans), ["de", "en", "de"])

    def test_without_anchors_the_majority_and_without_results_the_fallback(self) -> None:
        weak = [_result("fr", 0.3), _result("fr", 0.4), None, _result("it", 0.2)]
        self.assertEqual(lid.smooth_piece_languages(weak, self.SPANS), ["fr"] * 4)
        self.assertEqual(lid.smooth_piece_languages([None] * 4, self.SPANS), ["de"] * 4)


class SuperchunkRegionTests(unittest.TestCase):
    def test_short_audio_and_no_speech_stay_one_item(self) -> None:
        self.assertIsNone(wxc.superchunk_speech_regions(np.zeros(16000 * 30, dtype=np.float32)))
        with (
            mock.patch.object(wxc, "silero_frame_probs", return_value=np.zeros(4)),
            mock.patch.object(wxc, "speech_regions_from_probs", return_value=[]),
        ):
            self.assertIsNone(wxc.superchunk_speech_regions(np.zeros(16000 * 90, dtype=np.float32)))

    def test_a_long_transcript_without_a_language_change_is_chunked_exactly_as_before(self) -> None:
        regions = [(1.0, 24.0), (30.0, 52.0), (70.0, 82.0), (82.5, 94.0)]
        audio = np.full(16000 * 100, 0.2, dtype=np.float32)
        with (
            mock.patch.object(wxc, "silero_frame_probs", return_value=np.zeros(4)),
            mock.patch.object(wxc, "speech_regions_from_probs", return_value=regions),
            mock.patch.object(lid, "detect_languages", side_effect=lambda windows: [_result("de", 0.97)] * len(windows)),
            mock.patch.object(lid, "prepare_lid_model", return_value=True),
        ):
            before = wxc.split_audio_into_superchunks(audio)
            items, decision = asyncio.run(lid.decide_transcript_items(asyncio.Lock(), audio))
        self.assertEqual([len(item) for item in items], [len(item) for item in before])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(items, before)))
        self.assertEqual(decision.languages, ["de"] * len(before))

    def test_unavailable_detection_keeps_the_old_chunks_in_german_with_a_warning(self) -> None:
        regions = [(1.0, 24.0), (30.0, 52.0)]
        audio = np.full(16000 * 100, 0.2, dtype=np.float32)
        with (
            mock.patch.object(lid, "superchunk_speech_regions", return_value=regions),
            mock.patch.object(lid, "prepare_lid_model", return_value=False),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            items, decision = asyncio.run(lid.decide_transcript_items(asyncio.Lock(), audio))
        self.assertEqual(len(items), len(wxc.build_superchunks(regions)))
        self.assertEqual((decision.languages, decision.source), (["de", "de"], "fallback"))
        self.assertEqual(decision.warnings[0]["code"], "LANGUAGE_DETECTION_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
