"""ASR glossary ("keywords"): validation, merge with the server glossary, processing keys, Granite prompt and KV sizing.

Runs without a GPU or model: the batch manager, the processor and the model are faked.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_api as legacy_api
from backend import genesis_whisper_server_batching as batching
from backend import genesis_whisper_server_granite_speech as granite
from backend import genesis_whisper_server_keywords as keywords
from backend import genesis_whisper_server_lid as lid
from backend import genesis_whisper_server_local_asr_engine as engine
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_batching import WhisperBatchManager
from backend.genesis_whisper_server_globals import (
    COHERE_MODEL_ID,
    GRANITE_SPEECH_MODEL_ID,
    batch_history,
    current_settings,
    job_history,
    settings_lock,
)
from tests.test_granite_backend import _FakeModel, _FakeProcessor, _SpeedStateCase, clip
from tests.test_v2_api import _CapturingApp, _request, _upload

KEY = (GRANITE_SPEECH_MODEL_ID, "cpu", "", "de", "bf16")


def _glossary(*terms, truncated=False):
    return keywords.Glossary(terms=tuple(terms), truncated=truncated)


def _v2_settings(model=GRANITE_SPEECH_MODEL_ID, **values):
    return {
        "local_model": model,
        "local_gpu_device": "cpu",
        "local_model_cache_path": "",
        "transcription_language": "de",
        "local_model_precision": "bf16",
        "cohere_language_detection": False,
        "asr_keywords": [],
        **values,
    }


class _KeyRecordingManager:
    """submit_job fake: records every processing key, the text names the key's glossary."""

    def __init__(self):
        self.keys = []

    async def submit_job(self, segments, request_id, key):
        self.keys.append(key)
        terms = keywords.key_keywords(key)
        return [SimpleNamespace(text="|".join(terms) or "-", batch_id="b", model_id=key[0]) for _ in segments]


class NormalisationTests(unittest.TestCase):
    def test_trim_drop_empty_and_dedupe_case_insensitively_keeping_the_first_spelling(self):
        terms, dropped = keywords.normalize_keywords(["  Genesis ", "", "   ", "genesis", "OmniVoice", "GENESIS", "omnivoice"])
        self.assertEqual(terms, ["Genesis", "OmniVoice"])
        self.assertFalse(dropped)

    def test_entries_over_100_chars_are_dropped_and_count_as_truncated(self):
        terms, dropped = keywords.normalize_keywords(["a" * 100, "b" * 101, "c"])
        self.assertEqual(terms, ["a" * 100, "c"])
        self.assertTrue(dropped)

    def test_limits_cut_from_the_end(self):
        many = [f"t{index}" for index in range(1005)]
        kept, cut = keywords.limit_keywords(many)
        self.assertEqual(kept, many[:1000])
        self.assertTrue(cut)
        long_terms = ["x" * 100 for _ in range(170)]  # 17000 chars
        kept, cut = keywords.limit_keywords(long_terms)
        self.assertEqual(len(kept), 160)
        self.assertTrue(cut)
        self.assertEqual(keywords.limit_keywords(["a", "b"]), (["a", "b"], False))

    def test_server_glossary_comes_first_and_wins_duplicates(self):
        with mock.patch.dict(current_settings, {"asr_keywords": ["Genesis", "Optimus"]}):
            glossary = keywords.resolve_glossary(["genesis", "Whisper", "optimus", "Granite"])
        self.assertEqual(glossary.terms, ("Genesis", "Optimus", "Whisper", "Granite"))
        self.assertFalse(glossary.truncated)

    def test_merge_over_the_limit_cuts_request_terms_and_flags_truncated(self):
        server = [f"s{index}" for index in range(990)]
        with mock.patch.dict(current_settings, {"asr_keywords": server}):
            glossary = keywords.resolve_glossary([f"r{index}" for index in range(20)])
        self.assertEqual(glossary.count, 1000)
        self.assertEqual(glossary.terms[:990], tuple(server))
        self.assertEqual(glossary.terms[-1], "r9")
        self.assertTrue(glossary.truncated)
        self.assertEqual(keywords.truncation_warnings(glossary), [{"code": "KEYWORDS_TRUNCATED"}])

    def test_form_text_splits_on_newlines_and_commas(self):
        self.assertEqual(
            keywords.resolve_glossary(keywords.parse_form_keywords("Genesis, Optimus\nWhisper\r\n\n,Granite")).terms,
            ("Genesis", "Optimus", "Whisper", "Granite"),
        )
        self.assertEqual(keywords.parse_form_keywords(None), [])

    def test_setting_is_sanitised_on_save_without_an_error(self):
        self.assertEqual(storage.DEFAULT_SETTINGS["asr_keywords"], [])
        self.assertIn("asr_keywords", storage.SERVICE_SETTINGS_FIELDS)
        self.assertNotIn("asr_keywords", storage.PRESET_KEYS)
        normalized = storage.normalize_settings({"asr_keywords": [" A ", "a", "", "B" * 101, 7, "C"]})
        self.assertEqual(normalized["asr_keywords"], ["A", "C"])
        self.assertEqual(storage.normalize_settings({"asr_keywords": "x\ny, z"})["asr_keywords"], ["x", "y", "z"])
        self.assertEqual(storage.normalize_settings({"asr_keywords": None})["asr_keywords"], [])
        capped = storage.normalize_settings({"asr_keywords": [f"t{index}" for index in range(1200)]})["asr_keywords"]
        self.assertEqual(len(capped), 1000)

    def test_admin_payload_and_response_carry_the_field(self):
        self.assertIn("asr_keywords", admin.AdminSettingsPayload.model_fields)
        payload = admin.AdminSettingsPayload(asr_keywords=["Genesis", "Optimus"])
        self.assertEqual(payload.model_dump(exclude_none=True)["asr_keywords"], ["Genesis", "Optimus"])
        with mock.patch.dict(current_settings, {"asr_keywords": ["Genesis"]}):
            response = admin._settings_response()
        self.assertEqual(response["settings"]["asr_keywords"], ["Genesis"])
        self.assertIn("asr_keywords", response["service_fields"])


class RequestValidationTests(unittest.TestCase):
    def parse(self, **values):
        return v2._parse_request_json(json.dumps({"schema_version": "2.0", **values}))

    def test_keywords_are_accepted_in_every_asr_mode(self):
        for mode in ("transcript", "transcript_embedding", "diarization"):
            with self.subTest(mode=mode):
                self.assertEqual(self.parse(mode=mode, keywords=["Genesis", " x "])["keywords"], ["Genesis", " x "])
        self.assertIsNone(self.parse(mode="transcript")["keywords"])

    def test_keywords_in_embedding_mode_are_422(self):
        with self.assertRaises(v2.V2ApiError) as raised:
            self.parse(mode="embedding", keywords=["Genesis"])
        self.assertEqual((raised.exception.status_code, raised.exception.code), (422, "INVALID_REQUEST"))

    def test_wrong_types_are_422(self):
        for value in ("Genesis", {"a": 1}, 3, ["ok", 5], ["ok", None], [["nested"]]):
            with self.subTest(value=value):
                with self.assertRaises(v2.V2ApiError) as raised:
                    self.parse(mode="transcript", keywords=value)
                self.assertEqual((raised.exception.status_code, raised.exception.code), (422, "INVALID_REQUEST"))


class ProcessingKeyTests(unittest.TestCase):
    def test_no_keywords_keep_todays_key(self):
        self.assertEqual(keywords.with_keywords(KEY, None), KEY)
        self.assertEqual(keywords.with_keywords(KEY, _glossary()), KEY)
        with mock.patch.dict(v2.current_settings, _v2_settings(), clear=True):
            self.assertEqual(v2._request_processing_key(None, _glossary()), KEY)
            self.assertEqual(v2._request_processing_key("en", None), KEY[:3] + ("en",) + KEY[4:])

    def test_equal_glossaries_share_a_key_and_different_ones_do_not(self):
        first = keywords.with_keywords(KEY, _glossary("Genesis", "Optimus"))
        same = keywords.with_keywords(KEY, _glossary("Genesis", "Optimus"))
        other = keywords.with_keywords(KEY, _glossary("Genesis"))
        self.assertEqual(len(first), 6)
        self.assertEqual(first, same)
        self.assertEqual(hash(first), hash(same))
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, KEY)
        self.assertEqual(keywords.key_keywords(first), ("Genesis", "Optimus"))
        self.assertEqual(keywords.key_keywords(KEY), ())
        # Short and stable: the digest is all the key compares.
        self.assertEqual(first[5].digest, keywords.keywords_digest(["Genesis", "Optimus"]))
        self.assertEqual(len(first[5].digest), 16)

    def test_language_split_and_failover_keep_the_glossary(self):
        key = keywords.with_keywords(KEY, _glossary("Genesis"))
        self.assertEqual(keywords.key_keywords(lid.with_language(key, "en")), ("Genesis",))
        self.assertEqual(lid.with_language(key, "en")[3], "en")
        self.assertEqual(lid.with_language(KEY, "en"), KEY[:3] + ("en",) + KEY[4:])

    def test_submit_by_language_passes_the_glossary_to_every_job(self):
        manager = _KeyRecordingManager()
        key = keywords.with_keywords(KEY, _glossary("Genesis"))
        asyncio.run(lid.submit_by_language(manager, [clip(1), clip(2), clip(3)], key, ["de", "en", "de"]))
        self.assertEqual(sorted(k[3] for k in manager.keys), ["de", "en"])
        self.assertTrue(all(keywords.key_keywords(k) == ("Genesis",) for k in manager.keys))

    def test_the_batch_function_passes_the_terms_to_the_engine(self):
        key = keywords.with_keywords(KEY, _glossary("Genesis"))
        with (
            mock.patch.object(legacy_api, "load_local_asr_model", return_value=True),
            mock.patch.object(legacy_api, "transcribe_local_asr_batch", return_value=["x"]) as transcribe,
        ):
            legacy_api.process_local_asr_batch([clip(1)], key)
            legacy_api.process_local_asr_batch([clip(1)], KEY)
        self.assertEqual(transcribe.call_args_list[0].kwargs["keywords"], ("Genesis",))
        self.assertEqual(transcribe.call_args_list[0].kwargs["language"], "de")
        self.assertNotIn("keywords", transcribe.call_args_list[1].kwargs)

    def test_engine_hands_keywords_to_granite_only(self):
        processor, model = _FakeProcessor(), _FakeModel()
        components = {"model": model, "processor": processor, "model_backend": "granite_speech"}
        with mock.patch.dict(engine.local_model_components, components, clear=True), mock.patch.dict(
            current_settings, {"granite_compile": "off"}
        ):
            engine.transcribe_local_asr_batch([clip(1)], language="de", batch_size=8, keywords=("Genesis",))
        self.assertTrue(processor.calls[0]["text"][0].endswith("capitalization. Keywords: Genesis\n ASSISTANT:"))


class SchedulerSeparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._previous = current_settings.copy()
        with settings_lock:
            current_settings.update(
                {
                    "batch_wait_time_ms": 50,
                    "batch_max_segments": 16,
                    "batch_max_audio_seconds": 1000.0,
                    "scheduler_long_job_min_chunks": 50,
                    "scheduler_max_parallel_long_jobs": 0,
                    "scheduler_first_chunk_fast_path": False,
                    "cuda_memory_trim_after_batch": False,
                }
            )
        batch_history.clear()
        job_history.clear()
        self.batches = []

        def process(audio_batch, key):
            self.batches.append((key, len(audio_batch)))
            return ["|".join(keywords.key_keywords(key))] * len(audio_batch)

        async def gpu_phase(function, audio_batch, key):
            return function(audio_batch, key)

        self.manager = WhisperBatchManager(process, asyncio.Lock())
        self._patch = mock.patch.object(batching, "run_blocking_gpu_phase", new=gpu_phase)
        self._patch.start()
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.stop()
        self._patch.stop()
        with settings_lock:
            current_settings.clear()
            current_settings.update(self._previous)

    async def test_two_glossaries_never_share_a_batch_and_equal_ones_do(self):
        a = keywords.with_keywords(KEY, _glossary("Genesis"))
        a_again = keywords.with_keywords(KEY, _glossary("Genesis"))
        b = keywords.with_keywords(KEY, _glossary("Optimus"))
        tasks = [
            asyncio.create_task(self.manager.submit_job([clip(1)] * 2, "a", a)),
            asyncio.create_task(self.manager.submit_job([clip(1)] * 2, "a2", a_again)),
            asyncio.create_task(self.manager.submit_job([clip(1)] * 2, "b", b)),
            asyncio.create_task(self.manager.submit_job([clip(1)] * 2, "plain", KEY)),
        ]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        self.assertEqual([[item.text for item in result] for result in results], [["Genesis"] * 2, ["Genesis"] * 2, ["Optimus"] * 2, [""] * 2])
        sizes = {}
        for key, size in self.batches:
            sizes[keywords.key_keywords(key)] = sizes.get(keywords.key_keywords(key), 0) + size
        self.assertEqual(sizes, {("Genesis",): 4, ("Optimus",): 2, (): 2})
        # The two equal glossaries ran in one batch.
        self.assertIn((a, 4), self.batches)
        self.assertIn((KEY, 2), self.batches)


class GranitePromptTests(_SpeedStateCase):
    def test_keywords_follow_the_punctuation_prompt_with_and_without_a_language_hint(self):
        self.assertEqual(
            granite.prompt_content("auto", ("Genesis", "Optimus")),
            "<|audio|>transcribe the speech with proper punctuation and capitalization. Keywords: Genesis, Optimus",
        )
        self.assertEqual(
            granite.prompt_content("de", ["Genesis"]),
            "<|audio|>The speech is in German. Transcribe it exactly; do not translate. "
            "transcribe the speech with proper punctuation and capitalization. Keywords: Genesis",
        )
        self.assertEqual(granite.prompt_content("de", ()), granite.prompt_content("de"))
        self.assertEqual(
            granite.build_prompt(_FakeProcessor(), "de", ["Genesis"]),
            f"USER: {granite.prompt_content('de', ['Genesis'])}\n ASSISTANT:",
        )

    def test_batch_uses_the_keyword_prompt_for_every_row(self):
        processor, model = _FakeProcessor(), _FakeModel()
        granite.transcribe_granite_batch(processor, model, [clip(2), clip(3)], "de", 16, keywords=("Genesis",))
        self.assertEqual(processor.calls[0]["text"], [granite.build_prompt(processor, "de", ("Genesis",))] * 2)

    def test_prompt_tokens_count_with_the_tokenizer_or_estimate(self):
        self.assertEqual(granite.prompt_tokens(_FakeProcessor(), ()), granite.GRANITE_PROMPT_TOKENS)
        # The fake tokenizer cannot encode: ASCII chars / 3, rounded up.
        suffix = " Keywords: Genesis, Optimus"
        self.assertEqual(granite.prompt_tokens(_FakeProcessor(), ("Genesis", "Optimus")), 64 + -(-len(suffix) // 3))
        # Non-ASCII counts one token per char.
        self.assertEqual(granite.prompt_tokens(_FakeProcessor(), ("日本",)), 64 + -(-len(" Keywords: ") // 3) + 2)
        counting = _FakeProcessor()
        counting.tokenizer.encode = lambda text, add_special_tokens: list(range(7))
        self.assertEqual(granite.prompt_tokens(counting, ("Genesis",)), 71)

    def test_kv_budget_shrinks_sub_batches_for_long_glossaries(self):
        model = _FakeModel(dtype=torch.bfloat16)
        seconds = [45.0] * 16
        plain = granite.sub_batch_sizes(seconds, 16, "de", model, False)
        long_prompt = granite.sub_batch_sizes(seconds, 16, "de", model, False, 4000)
        self.assertEqual(plain, [16])
        self.assertGreater(len(long_prompt), 1)
        self.assertEqual(sum(long_prompt), 16)
        for prompt in (64, 1000, 4000, 16000):
            for fast in (False, True):
                start = 0
                for size in granite.sub_batch_sizes(seconds, 16, "de", model, fast, prompt):
                    tokens = granite.estimated_total_tokens(seconds[start], "de", prompt)
                    rows = size
                    if fast and granite.fits_static_cache(tokens):
                        tokens, rows = granite.bucket_cache_length(tokens), granite.bucket_batch_size(size)
                    if size > 1:
                        self.assertLessEqual(granite.kv_cache_mb(model, rows, tokens), granite.GRANITE_KV_BUDGET_MB)
                    start += size
        self.assertEqual(granite.estimated_total_tokens(10.0, "de", 1000) - granite.estimated_total_tokens(10.0, "de"), 936)

    def test_eager_batch_with_a_long_glossary_is_cut_into_smaller_sub_batches(self):
        processor, model = _FakeProcessor(), _FakeModel(dtype=torch.bfloat16)
        terms = tuple(f"Begriff{index:04d}" for index in range(1000))
        texts = granite.transcribe_granite_batch(processor, model, [clip(45.0)] * 16, "de", 16, keywords=terms)
        self.assertEqual(len(texts), 16)
        self.assertGreater(len(processor.calls), 1)
        self.assertEqual(sum(len(call["lengths"]) for call in processor.calls), 16)

    def test_fast_mode_runs_a_prompt_beyond_the_largest_cache_bucket_eager(self):
        current_settings["granite_compile"] = "static_cache"
        processor, model = _FakeProcessor(), _FakeModel(dtype=torch.bfloat16)
        terms = tuple(f"Begriff{index:04d}" for index in range(1000))
        granite.transcribe_granite_batch(processor, model, [clip(2.0)] * 3, "de", 16, keywords=terms)
        self.assertTrue(all("cache_implementation" not in call for call in model.generate_calls))
        # No padding rows for an eager sub-batch.
        self.assertEqual([len(call["lengths"]) for call in processor.calls], [3])
        self.assertGreaterEqual(granite.speed_stats()["length_skips"], 1)
        # A short glossary still uses the static cache.
        model = _FakeModel(dtype=torch.bfloat16)
        granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2.0)], "de", 16, keywords=("Genesis",))
        self.assertEqual(model.generate_calls[0]["cache_implementation"], "static")

    def test_a_real_input_beyond_the_largest_bucket_falls_back_to_eager(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel()
        inputs = {"input_ids": torch.ones(1, granite.GRANITE_MAX_CACHE_LENGTH, dtype=torch.long)}
        with contextlib.redirect_stderr(io.StringIO()):
            granite._generate_with_fallback(model, inputs, 32, "static_cache", 1)
        self.assertNotIn("cache_implementation", model.generate_calls[0])
        self.assertEqual(granite.speed_stats()["length_skips"], 1)

    def test_largest_bucket_covers_the_plain_prompt_and_bounds_compiled_shapes(self):
        longest = granite.estimated_total_tokens(granite.GRANITE_MAX_ITEM_SECONDS, "ja")
        self.assertTrue(granite.fits_static_cache(longest))
        shapes = len(granite.GRANITE_BATCH_BUCKETS) * granite.GRANITE_MAX_CACHE_LENGTH // granite.GRANITE_CACHE_LENGTH_BUCKET
        self.assertLessEqual(shapes, granite.GRANITE_COMPILE_RECOMPILE_LIMIT)


class V2EndpointTests(unittest.TestCase):
    def setUp(self):
        self.manager = _KeyRecordingManager()
        self.app = _CapturingApp()
        self.app.state.whisper_batch_manager = self.manager
        v2.create_v2_api(self.app)
        self.endpoint = self.app.routes[("POST", "/v2/audio/process")]
        self.audio = np.full(16000 * 3, 0.1, dtype=np.float32)

    def call(self, request, settings):
        form = mock.Mock()
        form.close = mock.AsyncMock()
        with (
            mock.patch.dict(v2.current_settings, settings, clear=True),
            mock.patch.object(v2, "_authorize_v2_request", return_value=None),
            mock.patch.object(v2, "_parse_multipart_parts", new=mock.AsyncMock(return_value=(form, _upload(), json.dumps(request)))),
            mock.patch.object(v2, "load_audio_file", return_value=self.audio),
            mock.patch.object(v2, "split_audio_into_superchunks", return_value=[self.audio]),
            mock.patch.object(v2, "_generate_embedding", new=mock.AsyncMock(return_value=([0.0] * 192, 3))),
            mock.patch.object(v2, "_record_log"),
        ):
            response = asyncio.run(self.endpoint(_request(self.app)))
        return response.status_code, json.loads(response.body.decode("utf-8"))

    def test_granite_applies_the_merged_glossary(self):
        status, body = self.call(
            {"schema_version": "2.0", "mode": "transcript", "keywords": ["Optimus", " genesis ", ""]},
            _v2_settings(asr_keywords=["Genesis"]),
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["models"]["asr"]["keywords"], {"applied": True, "count": 2, "truncated": False})
        self.assertEqual(body["result"]["transcript"]["text"], "Genesis|Optimus")
        self.assertEqual(keywords.key_keywords(self.manager.keys[0]), ("Genesis", "Optimus"))
        self.assertEqual(body["warnings"], [])

    def test_cohere_accepts_but_does_not_apply(self):
        status, body = self.call(
            {"schema_version": "2.0", "mode": "transcript_embedding", "keywords": ["Genesis"]},
            _v2_settings(model=COHERE_MODEL_ID),
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["models"]["asr"]["keywords"], {"applied": False, "count": 1, "truncated": False})

    def test_no_keywords_report_count_zero_and_keep_the_plain_key(self):
        status, body = self.call({"schema_version": "2.0", "mode": "transcript"}, _v2_settings())
        self.assertEqual(status, 200, body)
        self.assertEqual(body["models"]["asr"]["keywords"], {"applied": False, "count": 0, "truncated": False})
        self.assertEqual(self.manager.keys, [KEY])

    def test_truncation_adds_the_warning(self):
        status, body = self.call(
            {"schema_version": "2.0", "mode": "transcript", "keywords": ["ok", "x" * 101]}, _v2_settings()
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["models"]["asr"]["keywords"], {"applied": True, "count": 1, "truncated": True})
        self.assertIn({"code": "KEYWORDS_TRUNCATED"}, body["warnings"])

    def test_embedding_mode_rejects_keywords_and_carries_no_block(self):
        status, body = self.call({"schema_version": "2.0", "mode": "embedding", "keywords": ["x"]}, _v2_settings())
        self.assertEqual((status, body["error"]["code"]), (422, "INVALID_REQUEST"))
        status, body = self.call({"schema_version": "2.0", "mode": "embedding"}, _v2_settings())
        self.assertEqual(status, 200, body)
        self.assertNotIn("asr", body["models"])

    def test_diarization_passes_the_glossary_to_both_asr_paths(self):
        glossary = _glossary("Genesis")
        dia = mock.AsyncMock(
            return_value={
                "model": {"id": "community-1"},
                "diarization": [{"start_ms": 0, "end_ms": 3000, "speaker_id": "A"}],
                "exclusive_diarization": [{"start_ms": 0, "end_ms": 3000, "speaker_id": "A"}],
                "overlaps": [],
            }
        )
        for pipeline in ("wxc", "turns"):
            with self.subTest(pipeline=pipeline):
                manager = _KeyRecordingManager()
                app = _CapturingApp()
                app.state.whisper_batch_manager = manager
                with (
                    mock.patch.dict(v2.current_settings, _v2_settings(), clear=True),
                    mock.patch.object(v2, "diarize_v2", new=dia),
                    mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value={})),
                    mock.patch.object(v2, "silero_frame_probs", return_value=np.ones(4)),
                    mock.patch.object(v2, "speech_regions_from_probs", return_value=[(0.0, 3.0)]),
                    mock.patch.object(v2, "_align_words_sliced", new=mock.AsyncMock(return_value=[])),
                    mock.patch.object(v2, "plan_verification_windows", return_value=([], [])),
                    mock.patch.object(v2, "finish_verification", return_value={}),
                ):
                    asyncio.run(
                        v2._process_diarization(
                            _request(app),
                            _upload(),
                            self.audio,
                            {"expected_speakers": None, "known_speakers": []},
                            False,
                            False,
                            pipeline,
                            None,
                            glossary,
                        )
                    )
                self.assertTrue(manager.keys)
                self.assertTrue(all(keywords.key_keywords(key) == ("Genesis",) for key in manager.keys))

    def test_internal_callers_without_a_glossary_get_the_server_glossary(self):
        request = _request(self.app)
        with (
            mock.patch.dict(v2.current_settings, _v2_settings(asr_keywords=["Genesis"]), clear=True),
            mock.patch.object(v2, "split_audio_into_superchunks", return_value=[self.audio]),
        ):
            text, *_ = asyncio.run(v2._transcribe_audio(request, self.audio))
        self.assertEqual(text, "Genesis")


class CapabilitiesTests(unittest.TestCase):
    def capabilities(self, settings):
        app = FastAPI()
        v2.create_v2_capabilities_api(app)
        with mock.patch.dict(current_settings, storage.normalize_settings({**storage.DEFAULT_SETTINGS, **settings}), clear=True):
            return TestClient(app).get("/v2/capabilities").json()

    def test_keywords_key_is_always_present_and_supported_only_on_granite(self):
        granite_body = self.capabilities({"local_model": GRANITE_SPEECH_MODEL_ID, "asr_keywords": ["Genesis", "Optimus"]})
        self.assertEqual(
            granite_body["keywords"],
            {"supported": True, "max_entries": 1000, "max_chars": 16000, "server_glossary_count": 2},
        )
        for model in (COHERE_MODEL_ID, "openai/whisper-large-v3-turbo"):
            with self.subTest(model=model):
                self.assertFalse(self.capabilities({"local_model": model})["keywords"]["supported"])


class V1EndpointTests(unittest.TestCase):
    def post(self, endpoint, form, model=GRANITE_SPEECH_MODEL_ID, server_terms=()):
        audio = np.full(16000 * 2, 0.1, dtype=np.float32)
        manager = _KeyRecordingManager()
        app = FastAPI()
        app.state.whisper_batch_manager = manager
        app.state.local_gpu_lock = asyncio.Lock()
        legacy_api.create_api(app)
        settings = {**_v2_settings(model=model), "asr_keywords": list(server_terms)}
        with (
            mock.patch.dict(legacy_api.current_settings, settings, clear=True),
            mock.patch.object(legacy_api, "authorize_api_key", return_value=None),
            mock.patch.object(legacy_api, "load_audio_file", return_value=audio),
            mock.patch.object(legacy_api, "split_audio_into_superchunks", return_value=[audio]),
            mock.patch.object(legacy_api, "split_audio_for_whisper", return_value=[audio]),
            mock.patch.object(legacy_api, "log_transcription"),
            mock.patch.object(legacy_api, "append_history_entry"),
        ):
            with TestClient(app) as client:
                response = client.post(endpoint, files={"file": ("a.wav", b"fixture", "audio/wav")}, data=form)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json(), manager.keys

    def test_transcribe_form_field_with_newlines_and_commas(self):
        body, keys = self.post("/transcribe/", {"engine": "local", "keywords": "Optimus, Whisper\nGranite"}, server_terms=["Genesis"])
        self.assertEqual(keywords.key_keywords(keys[0]), ("Genesis", "Optimus", "Whisper", "Granite"))
        self.assertEqual(body["keywords"], {"applied": True, "count": 4, "truncated": False})

    def test_transcribe_without_keywords_keeps_the_plain_key(self):
        body, keys = self.post("/transcribe/", {"engine": "local"})
        self.assertEqual(keys, [KEY])
        self.assertEqual(body["keywords"], {"applied": False, "count": 0, "truncated": False})

    def test_openai_endpoint_reports_keywords_only_in_verbose_json(self):
        body, keys = self.post("/v1/audio/transcriptions", {"response_format": "verbose_json", "keywords": "Genesis", "prompt": "ignored"})
        self.assertEqual(keywords.key_keywords(keys[0]), ("Genesis",))
        self.assertEqual(body["keywords"], {"applied": True, "count": 1, "truncated": False})
        body, _ = self.post("/v1/audio/transcriptions", {"keywords": "Genesis"})
        self.assertEqual(set(body), {"text"})

    def test_whisper_accepts_but_does_not_apply(self):
        body, keys = self.post("/transcribe/", {"engine": "local", "keywords": "Genesis"}, model="openai/whisper-large-v3-turbo")
        self.assertEqual(body["keywords"], {"applied": False, "count": 1, "truncated": False})


if __name__ == "__main__":
    unittest.main()
