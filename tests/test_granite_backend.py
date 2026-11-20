"""IBM Granite Speech 5.0 2B as a local ASR backend.

Runs without a GPU or model download: the processor, tokenizer and model are faked. The fake processor encodes each
clip's length in its first prompt token and the fake model echoes it, so every text names the clip it came from.
"""
import asyncio
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BatchFeature

from backend import genesis_whisper_server as server
from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_granite_speech as granite
from backend import genesis_whisper_server_lid as lid
from backend import genesis_whisper_server_local_asr_engine as engine
from backend import genesis_whisper_server_model_manager as model_manager
from backend import genesis_whisper_server_storage as storage
from backend.genesis_whisper_server_globals import (
    COHERE_MODEL_ID,
    GRANITE_SPEECH_MODEL_ID,
    LOCAL_ASR_MODEL_SPECS,
    current_settings,
    get_effective_transcription_language,
    get_local_model_backend,
    language_detection_applies,
    language_detection_fallback,
    uses_cohere_backend,
    uses_granite_backend,
    uses_language_detection_backend,
    uses_whisper_backend,
    uses_whole_audio_backend,
)

PROMPT_LENGTH = 4
EOS = 9999
PAD = 0


class _FakeTokenizer:
    padding_side = "left"
    eos_token_id = EOS
    pad_token_id = PAD

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"USER: {messages[0]['content']}\n ASSISTANT:"

    def batch_decode(self, rows, skip_special_tokens):
        assert skip_special_tokens
        return [" ".join(f"t{int(token)}" for token in row if int(token) not in (EOS, PAD)) + " " for row in rows]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls = []

    def __call__(self, text, audio, device, return_tensors):
        assert isinstance(audio, list) and all(clip.ndim == 1 and clip.dtype == np.float32 for clip in audio)
        assert len(text) == len(audio) and return_tensors == "pt"
        self.calls.append({"text": list(text), "lengths": [len(clip) for clip in audio], "device": device})
        input_ids = torch.full((len(audio), PROMPT_LENGTH), 7, dtype=torch.long)
        input_ids[:, 0] = torch.tensor([len(clip) // 160 for clip in audio])
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "input_features": torch.zeros(len(audio), 3, 160),
                "input_features_mask": torch.ones(len(audio), 3, dtype=torch.bool),
            }
        )


class _FakeModel(torch.nn.Module):
    def __init__(self, fail_static=False, dtype=torch.float32, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.config = mock.Mock(
            text_config=mock.Mock(num_hidden_layers=40, num_attention_heads=16, num_key_value_heads=4, head_dim=128)
        )
        self.fail_static = fail_static
        self.error = error or RuntimeError("static cache broke")
        self.generate_calls = []
        self.batch_rows = []

    def generate(self, **kwargs):
        self.generate_calls.append({key: value for key, value in kwargs.items() if not torch.is_tensor(value)})
        self.batch_rows.append(int(kwargs["input_ids"].shape[0]))
        if self.fail_static and kwargs.get("cache_implementation") == "static":
            self._cache = object()
            raise self.error
        input_ids = kwargs["input_ids"]
        new = torch.stack([input_ids[:, 0], torch.full_like(input_ids[:, 0], EOS)], dim=1)
        return torch.cat([input_ids, new], dim=1)


def clip(seconds):
    return np.full(int(round(seconds * 16000)), 0.01, dtype=np.float32)


def name(seconds):
    return f"t{int(round(seconds * 16000)) // 160}"


class _SpeedStateCase(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(granite._speed_state)
        granite._speed_state.clear()
        granite._speed_state.update(granite._fresh_speed_state())
        self.settings = mock.patch.dict(current_settings, {"granite_compile": "off"})
        self.settings.start()

    def tearDown(self):
        self.settings.stop()
        granite._speed_state.clear()
        granite._speed_state.update(self.saved_state)


class GraniteRegistryTests(unittest.TestCase):
    def test_model_is_registered_as_granite_backend(self):
        spec = LOCAL_ASR_MODEL_SPECS["Granite Speech 5.0 2B"]
        self.assertEqual(spec["value"], "ibm-granite/granite-speech-5.0-2b")
        self.assertEqual(spec["default_language"], "auto")
        self.assertEqual(get_local_model_backend(GRANITE_SPEECH_MODEL_ID), "granite_speech")
        self.assertTrue(uses_granite_backend(GRANITE_SPEECH_MODEL_ID))
        self.assertFalse(uses_cohere_backend(GRANITE_SPEECH_MODEL_ID))
        self.assertFalse(uses_whisper_backend(GRANITE_SPEECH_MODEL_ID))

    def test_granite_takes_the_whole_audio_path_like_cohere(self):
        self.assertTrue(uses_whole_audio_backend(GRANITE_SPEECH_MODEL_ID))

    def test_download_requires_the_chat_template_and_skips_the_unused_ctc_head(self):
        spec = model_manager._supported_model_specs()[GRANITE_SPEECH_MODEL_ID]
        for required in ("chat_template.jinja", "processor_config.json", "preprocessor_config.json", "model.safetensors"):
            self.assertIn(required, spec["required_files"])
        self.assertIn("out_llm.safetensors", spec["ignore_patterns"])

    def test_download_passes_the_ignore_patterns(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(model_manager, "snapshot_download") as download, mock.patch.object(
            model_manager.threading, "Thread"
        ) as thread:
            model_manager.queue_model_download(GRANITE_SPEECH_MODEL_ID, root, None)
            thread.call_args.kwargs["target"]()
        self.assertIn("out_llm.safetensors", download.call_args.kwargs["ignore_patterns"])

    def test_benchmark_workflow_label(self):
        self.assertEqual(admin._benchmark_workflow(GRANITE_SPEECH_MODEL_ID), "granite_audio_batch")

    def test_snapshot_is_ready_only_with_the_chat_template(self):
        with tempfile.TemporaryDirectory() as snapshot:
            for filename in (
                "config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model.safetensors",
            ):
                open(os.path.join(snapshot, filename), "w").close()
            self.assertFalse(engine._is_snapshot_ready_for_model(GRANITE_SPEECH_MODEL_ID, snapshot))
            open(os.path.join(snapshot, "chat_template.jinja"), "w").close()
            self.assertTrue(engine._is_snapshot_ready_for_model(GRANITE_SPEECH_MODEL_ID, snapshot))

    def test_attention_is_left_to_transformers_even_when_flash_attention_is_installed(self):
        # An explicit "sdpa" makes the Q-Former projector raise; without a value transformers picks per sub-model.
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(engine._resolve_backend_attention_implementation(GRANITE_SPEECH_MODEL_ID, "flash_attention_2"))
        self.assertIsNone(engine._resolve_backend_attention_implementation(GRANITE_SPEECH_MODEL_ID, "sdpa"))
        self.assertNotIn("attn_implementation", engine._build_model_kwargs("bf16", True, "cuda:0", torch.bfloat16, None))


class GraniteLanguageTests(unittest.TestCase):
    def test_language_detection_applies_to_granite_auto_only(self):
        with mock.patch.dict(current_settings, {"cohere_language_detection": True}):
            self.assertTrue(uses_language_detection_backend(GRANITE_SPEECH_MODEL_ID))
            self.assertTrue(language_detection_applies(GRANITE_SPEECH_MODEL_ID, "auto"))
            self.assertTrue(language_detection_applies(GRANITE_SPEECH_MODEL_ID, None))
            self.assertFalse(language_detection_applies(GRANITE_SPEECH_MODEL_ID, "de"))
        with mock.patch.dict(current_settings, {"cohere_language_detection": False}):
            self.assertFalse(language_detection_applies(GRANITE_SPEECH_MODEL_ID, "auto"))

    def test_auto_stays_auto_for_granite_without_detection(self):
        # Unlike Cohere, Granite transcribes without a hint, so "auto" never becomes German.
        self.assertEqual(get_effective_transcription_language(GRANITE_SPEECH_MODEL_ID, "auto"), "auto")
        self.assertEqual(get_effective_transcription_language(COHERE_MODEL_ID, "auto"), "de")

    def test_detection_fallback_is_auto_for_granite_and_german_for_cohere(self):
        self.assertEqual(language_detection_fallback(GRANITE_SPEECH_MODEL_ID), "auto")
        self.assertEqual(language_detection_fallback(COHERE_MODEL_ID), "de")
        with mock.patch.dict(current_settings, {"local_model": GRANITE_SPEECH_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([None, None], [10.0, 10.0]), ["auto", "auto"])
            self.assertEqual(lid.resolve_speaker_languages({"A": None}, {"A": 10.0}), {"A": ("auto", "fallback")})
            self.assertEqual(lid.dominant_language([]), "auto")
            self.assertEqual(lid.dominant_language([("auto", 3.0)]), "auto")
            self.assertEqual(lid.smooth_piece_languages([None], [(0.0, 10.0)]), ["auto"])
        with mock.patch.dict(current_settings, {"local_model": COHERE_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([None], [10.0]), ["de"])

    def test_weak_detections_give_granite_no_hint(self):
        weak = lid.LidResult(language="de", probability=0.3, probs={})
        strong = lid.LidResult(language="en", probability=0.9, probs={})
        with mock.patch.dict(current_settings, {"local_model": GRANITE_SPEECH_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([weak, weak], [10.0, 10.0]), ["auto", "auto"])
            self.assertEqual(lid.smooth_piece_languages([weak], [(0.0, 10.0)]), ["auto"])
            # A confident majority still covers the weak chunks.
            self.assertEqual(lid.smooth_chunk_languages([weak, strong], [10.0, 10.0]), ["en", "en"])
        with mock.patch.dict(current_settings, {"local_model": COHERE_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([weak, weak], [10.0, 10.0]), ["de", "de"])
            self.assertEqual(lid.smooth_piece_languages([weak], [(0.0, 10.0)]), ["de"])

    def test_a_language_granite_cannot_take_is_reported_as_auto(self):
        italian = lid.LidResult(language="it", probability=0.95, probs={})
        self.assertEqual(get_effective_transcription_language(GRANITE_SPEECH_MODEL_ID, "it"), "auto")
        self.assertEqual(get_effective_transcription_language(GRANITE_SPEECH_MODEL_ID, " JA "), "ja")
        self.assertEqual(get_effective_transcription_language(COHERE_MODEL_ID, "it"), "it")
        with mock.patch.dict(current_settings, {"local_model": GRANITE_SPEECH_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([italian], [10.0]), ["auto"])
            self.assertEqual(lid.smooth_piece_languages([italian], [(0.0, 10.0)]), ["auto"])
            self.assertEqual(lid.resolve_speaker_languages({"A": italian}, {"A": 10.0}), {"A": ("auto", "detected")})
            self.assertEqual(lid.turn_language(italian, 5.0, "de"), ("auto", True))
        with mock.patch.dict(current_settings, {"local_model": COHERE_MODEL_ID}):
            self.assertEqual(lid.smooth_chunk_languages([italian], [10.0]), ["it"])

    def test_prompt_with_hint_for_the_six_granite_languages(self):
        expected = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "ja": "Japanese"}
        for code, language in expected.items():
            with self.subTest(code=code):
                self.assertEqual(
                    granite.prompt_content(code),
                    f"<|audio|>The speech is in {language}. Transcribe it exactly; do not translate. "
                    "transcribe the speech with proper punctuation and capitalization.",
                )

    def test_plain_prompt_for_auto_unsupported_and_missing_languages(self):
        plain = "<|audio|>transcribe the speech with proper punctuation and capitalization."
        for language in ("auto", "it", "nl", "zh", "ko", "", None, " AUTO "):
            with self.subTest(language=language):
                self.assertEqual(granite.prompt_content(language), plain)

    def test_prompt_goes_through_the_chat_template(self):
        self.assertEqual(
            granite.build_prompt(_FakeProcessor(), "de"),
            f"USER: {granite.prompt_content('de')}\n ASSISTANT:",
        )

    def test_prompt_without_a_template_uses_the_snapshot_format(self):
        processor = _FakeProcessor()
        processor.tokenizer.apply_chat_template = mock.Mock(side_effect=ValueError("no chat template"))
        self.assertEqual(granite.build_prompt(processor, "auto"), f"USER: {granite.prompt_content('auto')}\n ASSISTANT:")

    def test_lid_warmup_is_wanted_for_granite(self):
        with mock.patch.dict(server.current_settings, {"local_model": GRANITE_SPEECH_MODEL_ID, "cohere_language_detection": True}):
            self.assertTrue(server._lid_warmup_wanted())

    def test_languages_become_separate_jobs_with_their_own_processing_key(self):
        submitted = []

        class _Manager:
            async def submit_job(self, segments, token, processing_key):
                submitted.append((processing_key, len(segments)))
                return [mock.Mock(text=processing_key[3]) for _ in segments]

        key = (GRANITE_SPEECH_MODEL_ID, "auto", "", "auto", "bf16")
        results = asyncio.run(lid.submit_by_language(_Manager(), [clip(1), clip(2), clip(3)], key, ["de", "auto", "de"]))
        self.assertEqual([result.text for result in results], ["de", "auto", "de"])
        self.assertEqual(sorted((processing_key[3], count) for processing_key, count in submitted), [("auto", 1), ("de", 2)])


class GraniteBatchTests(_SpeedStateCase):
    def test_one_processor_call_and_one_generate_for_a_batch(self):
        processor, model = _FakeProcessor(), _FakeModel()
        texts = granite.transcribe_granite_batch(processor, model, [clip(2), clip(5), clip(3)], "de", 16)

        self.assertEqual(texts, [name(2), name(5), name(3)])
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(processor.calls[0]["text"], [granite.build_prompt(processor, "de")] * 3)
        self.assertEqual(processor.calls[0]["device"], "cpu")
        generate = model.generate_calls[0]
        self.assertEqual((generate["do_sample"], generate["num_beams"]), (False, 1))
        self.assertNotIn("cache_implementation", generate)

    def test_clips_are_sorted_by_length_and_results_come_back_in_input_order(self):
        processor, model = _FakeProcessor(), _FakeModel()
        durations = [1.0, 4.0, 2.0, 3.0, 0.5]
        texts = granite.transcribe_granite_batch(processor, model, [clip(value) for value in durations], "auto", 2)

        self.assertEqual(texts, [name(value) for value in durations])
        self.assertEqual([call["lengths"] for call in processor.calls], [[64000, 48000], [32000, 16000], [8000]])

    def test_sub_batches_are_capped_at_sixteen(self):
        processor, model = _FakeProcessor(), _FakeModel()
        granite.transcribe_granite_batch(processor, model, [clip(1 + index / 100) for index in range(20)], "en", 64)
        self.assertEqual([len(call["lengths"]) for call in processor.calls], [16, 4])

    def test_empty_and_tiny_clips_keep_their_slot_and_never_reach_the_model(self):
        processor, model = _FakeProcessor(), _FakeModel()
        texts = granite.transcribe_granite_batch(
            processor, model, [np.zeros(0, dtype=np.float32), clip(2), np.zeros(100, dtype=np.float32)], "de", 8
        )
        self.assertEqual(texts, ["", name(2), ""])
        self.assertEqual(processor.calls[0]["lengths"], [32000])

        empty_model = _FakeModel()
        self.assertEqual(granite.transcribe_granite_batch(_FakeProcessor(), empty_model, [np.zeros(0, dtype=np.float32)], "de", 8), [""])
        self.assertEqual(empty_model.generate_calls, [])

    def test_prompt_tokens_are_stripped_and_text_is_trimmed(self):
        texts = granite.transcribe_granite_batch(_FakeProcessor(), _FakeModel(), [clip(1)], "de", 1)
        self.assertEqual(texts, ["t100"])
        self.assertNotIn("t7", texts[0])

    def test_max_new_tokens_follow_the_longest_clip_of_the_sub_batch(self):
        self.assertEqual(granite.max_new_tokens_for(0.0, "de"), 32)
        self.assertEqual(granite.max_new_tokens_for(25.0, "de"), 282)
        self.assertEqual(granite.max_new_tokens_for(60.0, "en"), 632)
        self.assertEqual(granite.max_new_tokens_for(1000.0, "de"), granite.GRANITE_MAX_NEW_TOKENS_CAP)
        model = _FakeModel()
        granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2), clip(25)], "de", 16)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 282)

    def test_japanese_and_auto_get_the_dense_token_rate(self):
        # Fast Japanese runs ~11 tokens/s; "auto" may be Japanese.
        for language in ("ja", "auto", None, "it"):
            with self.subTest(language=language):
                self.assertEqual(granite.max_new_tokens_for(60.0, language), 60 * 16 + 32)
        self.assertEqual(granite.max_new_tokens_for(100.0, "ja"), granite.GRANITE_MAX_NEW_TOKENS_CAP)

    def test_a_row_cut_at_the_token_limit_is_counted_and_warned(self):
        class _Looping(_FakeModel):
            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                new = torch.full((input_ids.shape[0], kwargs["max_new_tokens"]), 5, dtype=torch.long)
                new[1:, 0] = EOS
                return torch.cat([input_ids, new], dim=1)

        with contextlib.redirect_stderr(io.StringIO()) as log:
            granite.transcribe_granite_batch(_FakeProcessor(), _Looping(), [clip(3), clip(2)], "de", 8)
        self.assertEqual(granite.speed_stats()["truncated_rows"], 1)
        self.assertIn("1 von 2 Clips am Token-Limit 62", log.getvalue())

    def test_rows_that_end_with_eos_are_not_counted_as_cut(self):
        granite.transcribe_granite_batch(_FakeProcessor(), _FakeModel(), [clip(3), clip(2)], "de", 8)
        self.assertEqual(granite.speed_stats()["truncated_rows"], 0)

    def test_long_clips_are_split_at_quiet_moments_and_joined(self):
        processor, model = _FakeProcessor(), _FakeModel()
        long_clip = np.concatenate([clip(40), np.zeros(16000, dtype=np.float32), clip(40)])
        texts = granite.transcribe_granite_batch(processor, model, [clip(2), long_clip], "de", 16)

        lengths = processor.calls[0]["lengths"]
        self.assertEqual(len(lengths), 3)
        self.assertTrue(all(length <= granite.GRANITE_MAX_ITEM_SECONDS * 16000 for length in lengths))
        self.assertEqual(sum(lengths), len(long_clip) + 32000)
        parts = granite.split_long_item(long_clip)
        self.assertEqual(texts[1], " ".join(f"t{len(part) // 160}" for part in parts))
        # The cut lands in the silence between the two speech halves.
        self.assertTrue(40 * 16000 <= len(parts[0]) <= 41 * 16000)

    def test_dispatch_through_the_engine(self):
        processor, model = _FakeProcessor(), _FakeModel()
        components = {"model": model, "processor": processor, "model_backend": "granite_speech"}
        with mock.patch.dict(engine.local_model_components, components, clear=True):
            texts = engine.transcribe_local_asr_batch([clip(1), clip(2)], language="fr", batch_size=8)
        self.assertEqual(texts, [name(1), name(2)])
        self.assertIn("The speech is in French.", processor.calls[0]["text"][0])

    def test_left_padding_is_enforced(self):
        processor = _FakeProcessor()
        processor.tokenizer.padding_side = "right"
        with contextlib.redirect_stderr(io.StringIO()):
            granite.ensure_left_padding(processor)
        self.assertEqual(processor.tokenizer.padding_side, "left")


class GraniteSpeedModeTests(_SpeedStateCase):
    def test_static_cache_mode_uses_the_static_cache_without_compile_and_a_bucketed_length(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel()
        texts = granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertEqual(texts, [name(2)])
        generate = model.generate_calls[0]
        self.assertEqual(generate["cache_implementation"], "static")
        self.assertTrue(generate["disable_compile"])
        # The cache gets the bucketed length, generation stops at the eager budget.
        self.assertEqual((PROMPT_LENGTH + generate["max_new_tokens"]) % granite.GRANITE_CACHE_LENGTH_BUCKET, 0)
        (criterion,) = generate["stopping_criteria"]
        self.assertEqual(criterion.max_length, PROMPT_LENGTH + granite.max_new_tokens_for(2, "de"))
        self.assertTrue(bool(criterion(torch.zeros(2, criterion.max_length, dtype=torch.long), None).all()))
        self.assertFalse(bool(criterion(torch.zeros(2, criterion.max_length - 1, dtype=torch.long), None).any()))
        stats = granite.speed_stats()
        self.assertEqual((stats["fast_calls"], stats["last_mode"]), (1, "static_cache"))

    def test_fast_modes_pad_a_sub_batch_to_a_row_bucket_and_drop_the_padding_rows(self):
        current_settings["granite_compile"] = "static_cache"
        processor, model = _FakeProcessor(), _FakeModel()
        texts = granite.transcribe_granite_batch(processor, model, [clip(2), clip(5), clip(3)], "de", 16)
        self.assertEqual(texts, [name(2), name(5), name(3)])
        self.assertEqual(processor.calls[0]["lengths"], [80000, 48000, 32000, 32000])
        self.assertEqual(model.batch_rows, [4])

    def test_eager_fallback_runs_only_the_real_rows(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel(fail_static=True)
        with contextlib.redirect_stderr(io.StringIO()):
            texts = granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2), clip(5), clip(3)], "de", 16)
        self.assertEqual(texts, [name(2), name(5), name(3)])
        self.assertEqual(model.batch_rows, [4, 3])

    def test_a_static_cache_failure_falls_back_to_eager_for_the_process_lifetime(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel(fail_static=True)
        with contextlib.redirect_stderr(io.StringIO()) as log:
            first = granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
            second = granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(3)], "de", 8)
        self.assertEqual((first, second), ([name(2)], [name(3)]))
        self.assertEqual([call.get("cache_implementation") for call in model.generate_calls], ["static", None, None])
        self.assertEqual(log.getvalue().count("fehlgeschlagen"), 1)
        self.assertIn("static cache broke", granite.speed_stats()["failure"])
        self.assertFalse(hasattr(model, "_cache"))

    def test_a_cuda_oom_runs_that_batch_eager_and_keeps_the_mode(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel(fail_static=True, error=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        model._compiled_call = object()
        with contextlib.redirect_stderr(io.StringIO()):
            first = granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertEqual(first, [name(2)])
        stats = granite.speed_stats()
        self.assertIsNone(stats["failure"])
        self.assertEqual((stats["oom_fallbacks"], stats["eager_calls"]), (1, 1))
        self.assertFalse(hasattr(model, "_cache"))
        self.assertTrue(hasattr(model, "_compiled_call"))
        model.fail_static = False
        granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertEqual(model.generate_calls[-1]["cache_implementation"], "static")

    def test_compile_without_cuda_runs_eager(self):
        current_settings["granite_compile"] = "compile"
        model = _FakeModel()
        with contextlib.redirect_stderr(io.StringIO()) as log:
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertEqual([call.get("cache_implementation") for call in model.generate_calls], [None, None])
        self.assertEqual(log.getvalue().count("kein CUDA-Geraet"), 1)

    def test_compile_mode_passes_a_reduce_overhead_compile_config(self):
        current_settings["granite_compile"] = "compile"
        model = _FakeModel()
        with mock.patch.object(granite, "_inductor_unavailable_reason", return_value=None):
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        generate = model.generate_calls[0]
        self.assertEqual(generate["cache_implementation"], "static")
        self.assertFalse(generate["disable_compile"])
        self.assertEqual(generate["compile_config"].mode, "reduce-overhead")
        self.assertFalse(generate["compile_config"].fullgraph)
        self.assertGreaterEqual(torch._dynamo.config.recompile_limit, granite.GRANITE_COMPILE_RECOMPILE_LIMIT)
        self.assertEqual(granite.speed_stats()["compiled_shapes"], 1)

    def test_compile_shapes_are_bounded_by_the_row_and_length_buckets(self):
        current_settings["granite_compile"] = "compile"
        model = _FakeModel(dtype=torch.bfloat16)
        with mock.patch.object(granite, "_inductor_unavailable_reason", return_value=None):
            for rows in range(1, 17):
                granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)] * rows, "de", 16)
        # 16 batch sizes, one length bucket: five row buckets.
        self.assertEqual(sorted(set(model.batch_rows)), [1, 2, 4, 8, 16])
        self.assertEqual(granite.speed_stats()["compiled_shapes"], 5)
        buckets = len(granite.GRANITE_BATCH_BUCKETS)
        longest = granite.bucket_cache_length(granite.estimated_total_tokens(granite.GRANITE_MAX_ITEM_SECONDS, "ja"))
        self.assertLessEqual(buckets * longest // granite.GRANITE_CACHE_LENGTH_BUCKET, granite.GRANITE_COMPILE_RECOMPILE_LIMIT)

    def test_compile_on_an_int8_model_is_reported_as_static_cache(self):
        current_settings["granite_compile"] = "compile"
        model = _FakeModel()
        model.hf_quantizer = mock.Mock(is_compileable=False)
        with mock.patch.object(granite, "_inductor_unavailable_reason", return_value=None), contextlib.redirect_stderr(io.StringIO()):
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertTrue(model.generate_calls[0]["disable_compile"])
        self.assertEqual(granite.speed_stats()["last_mode"], "static_cache")

    def test_a_static_cache_over_the_budget_runs_eager(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel()
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(granite, "GRANITE_KV_BUDGET_MB", 1):
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertNotIn("cache_implementation", model.generate_calls[0])
        self.assertEqual(granite.speed_stats()["budget_skips"], 1)

    def test_a_budget_skip_drops_the_cache_but_keeps_the_compiled_step(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel()
        model._cache = object()
        model._compiled_call = object()
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(granite, "GRANITE_KV_BUDGET_MB", 1):
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertFalse(hasattr(model, "_cache"))
        self.assertTrue(hasattr(model, "_compiled_call"))

    def test_kv_sizing_with_the_real_budget(self):
        # 40 layers x 4 KV heads x 128 x 2 (K and V) x 2 bytes = 80 KiB per token.
        model = _FakeModel(dtype=torch.bfloat16)
        self.assertEqual(granite._kv_bytes_per_token(model), 80 * 1024)
        self.assertEqual(granite.kv_cache_mb(model, 16, 1280), 1600.0)
        self.assertEqual(granite.kv_cache_mb(model, 1, 256), 20.0)
        # A live clip and sixteen 45 s German clips fit one sub-batch in every mode; sixteen 60 s clips do not.
        self.assertEqual(granite.sub_batch_sizes([5.0], 16, "de", model, True), [1])
        self.assertEqual(granite.sub_batch_sizes([45.0] * 16, 16, "de", model, False), [16])
        self.assertEqual(granite.sub_batch_sizes([45.0] * 16, 16, "de", model, True), [16])
        self.assertEqual(granite.sub_batch_sizes([60.0] * 16, 16, "de", model, False), [15, 1])
        self.assertEqual(granite.sub_batch_sizes([60.0] * 16, 16, "de", model, True), [8, 8])
        for fast in (False, True):
            start = 0
            seconds = [60.0] * 6 + [2.0] * 10
            for size in granite.sub_batch_sizes(seconds, 16, "auto", model, fast):
                tokens = granite.estimated_total_tokens(seconds[start], "auto")
                rows = size
                if fast:
                    tokens, rows = granite.bucket_cache_length(tokens), granite.bucket_batch_size(size)
                self.assertLessEqual(granite.kv_cache_mb(model, rows, tokens), granite.GRANITE_KV_BUDGET_MB)
                start += size
            self.assertEqual(start, 16)

    def test_eager_sub_batches_follow_the_kv_budget(self):
        processor, model = _FakeProcessor(), _FakeModel(dtype=torch.bfloat16)
        clips = [clip(59.0)] * 6 + [clip(2.0)] * 10
        texts = granite.transcribe_granite_batch(processor, model, clips, "auto", 16)
        self.assertEqual(len(texts), 16)
        self.assertEqual([len(call["lengths"]) for call in processor.calls], [12, 4])
        self.assertTrue(all(call.get("cache_implementation") is None for call in model.generate_calls))

    def test_switching_off_releases_a_kept_static_cache_and_the_compiled_graphs(self):
        model = _FakeModel()
        model._cache = object()
        model._compiled_call = object()
        with mock.patch.object(granite, "_reset_compiled_graphs") as reset:
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertFalse(hasattr(model, "_cache"))
        self.assertFalse(hasattr(model, "_compiled_call"))
        reset.assert_called_once()

    def test_release_without_a_compiled_step_leaves_dynamo_alone(self):
        model = _FakeModel()
        model._cache = object()
        with mock.patch.object(granite, "_reset_compiled_graphs") as reset:
            granite.release_speed_state(model)
        self.assertFalse(hasattr(model, "_cache"))
        reset.assert_not_called()

    def test_idle_trim_frees_a_static_cache_but_not_one_with_captured_graphs(self):
        model = _FakeModel()
        model._cache = object()
        components = {"model": model, "processor": _FakeProcessor(), "model_backend": "granite_speech"}
        with mock.patch.dict(engine.local_model_components, components, clear=True):
            self.assertTrue(engine.release_idle_model_caches())
            self.assertFalse(hasattr(model, "_cache"))
            model._cache = object()
            model._compiled_call = object()
            self.assertFalse(engine.release_idle_model_caches())
            self.assertTrue(hasattr(model, "_cache"))
        with mock.patch.dict(engine.local_model_components, {"model": model, "model_backend": "cohere_transcribe"}, clear=True):
            self.assertFalse(engine.release_idle_model_caches())

    def test_the_scheduler_trim_releases_idle_model_caches(self):
        from backend import genesis_whisper_server_batching as batching

        with mock.patch.object(engine, "release_idle_model_caches") as release:
            batching.WhisperBatchManager._trim_cuda_cache()
        release.assert_called_once()

    def test_speed_stats_reach_admin_stats_and_the_benchmark(self):
        model = _FakeModel(dtype=torch.bfloat16)
        model._cache = mock.Mock(max_batch_size=4, max_cache_len=512)
        components = {"model": model, "processor": _FakeProcessor(), "model_backend": "granite_speech"}
        with mock.patch.dict(engine.local_model_components, components, clear=True):
            before = engine.granite_speed_stats()
            granite._speed_state["eager_calls"] += 3
            block = admin._granite_speed_delta(before)["granite_speed"]
        self.assertEqual(block["eager_calls"], 3)
        self.assertEqual(block["kept_cache_mb"], 160.0)
        self.assertEqual(block["requested"], "off")
        with mock.patch.dict(engine.local_model_components, {"model": None, "model_backend": "whisper"}, clear=True):
            self.assertIsNone(engine.granite_speed_stats())
            self.assertEqual(admin._granite_speed_delta(None), {})

    def test_setting_is_normalised(self):
        self.assertEqual(storage.DEFAULT_SETTINGS["granite_compile"], "off")
        for value, expected in (("compile", "compile"), (" Static_Cache ", "static_cache"), ("fast", "off"), (None, "off")):
            with self.subTest(value=value):
                self.assertEqual(storage.normalize_settings({"granite_compile": value})["granite_compile"], expected)

    def test_admin_form_shows_granite_compile_only_for_granite(self):
        response = admin._settings_response()
        fields = response["model_fields"]
        self.assertIn("granite_compile", fields["granite_speech"])
        for backend in ("whisper", "cohere_transcribe", "parakeet"):
            self.assertNotIn("granite_compile", fields[backend])
        self.assertIn("granite_compile", response["settings"])


class GraniteLoaderTests(unittest.TestCase):
    def test_loader_leaves_attention_to_transformers_enforces_left_padding_and_stores_the_backend(self):
        processor = _FakeProcessor()
        processor.tokenizer.padding_side = "right"
        model = _FakeModel()
        with (
            mock.patch.dict(engine.local_model_components, {"model": None}, clear=True),
            mock.patch.object(engine, "_prepare_model_loading_options", return_value=(False, "cpu", torch.float32, "flash_attention_2")),
            mock.patch.object(engine, "_resolve_granite_pretrained_source", return_value=("/snap", {"local_files_only": True})) as source,
            mock.patch.object(engine, "_resolve_huggingface_token", return_value=None),
            mock.patch.object(engine.AutoProcessor, "from_pretrained", return_value=processor) as processor_load,
            mock.patch.object(engine, "_load_auto_speech_model", return_value=model) as model_load,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertTrue(engine.load_local_asr_model(GRANITE_SPEECH_MODEL_ID, "cpu", "", "bf16"))
            stored = dict(engine.local_model_components)

        source.assert_called_once()
        self.assertNotIn("trust_remote_code", processor_load.call_args.kwargs)
        model_kwargs = model_load.call_args.args[2]
        self.assertNotIn("attn_implementation", model_kwargs)
        self.assertNotIn("trust_remote_code", model_kwargs)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(stored["model_backend"], "granite_speech")
        self.assertIs(stored["model"], model)

    def test_resolver_downloads_a_missing_snapshot_without_the_ctc_head(self):
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.object(engine, "snapshot_download", return_value="/snap") as download,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            source, args = engine._resolve_granite_pretrained_source(GRANITE_SPEECH_MODEL_ID, cache, None)
        self.assertEqual((source, args), ("/snap", {"local_files_only": True}))
        self.assertIn("out_llm.safetensors", download.call_args.kwargs["ignore_patterns"])

    def test_int8_keeps_audio_path_and_head_unquantized(self):
        fake_config = mock.Mock()
        with mock.patch.object(engine, "_require_bitsandbytes_quantization", return_value=fake_config), contextlib.redirect_stderr(io.StringIO()):
            engine._build_model_kwargs("int8_bnb", True, "cuda:0", torch.float16, "sdpa", int8_skip_modules=granite.GRANITE_INT8_SKIP_MODULES)
        self.assertEqual(fake_config.call_args.kwargs["llm_int8_skip_modules"], ["encoder", "projector", "language_model.lm_head"])

    def test_unloading_granite_releases_the_static_cache(self):
        model = _FakeModel()
        model._cache = object()
        components = {"model": model, "processor": _FakeProcessor(), "model_identifier": ("x",), "model_backend": "granite_speech"}
        with mock.patch.dict(engine.local_model_components, components, clear=True), contextlib.redirect_stderr(io.StringIO()):
            engine._cleanup_previous_model()
        self.assertFalse(hasattr(model, "_cache"))


if __name__ == "__main__":
    unittest.main()
