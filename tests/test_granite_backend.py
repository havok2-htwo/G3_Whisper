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
    def __init__(self, fail_static=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = mock.Mock(
            text_config=mock.Mock(num_hidden_layers=40, num_attention_heads=16, num_key_value_heads=4, head_dim=128)
        )
        self.fail_static = fail_static
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append({key: value for key, value in kwargs.items() if not torch.is_tensor(value)})
        if self.fail_static and kwargs.get("cache_implementation") == "static":
            self._cache = object()
            raise RuntimeError("static cache broke")
        input_ids = kwargs["input_ids"]
        new = torch.stack([input_ids[:, 0], torch.full_like(input_ids[:, 0], EOS)], dim=1)
        return torch.cat([input_ids, new], dim=1)


def clip(seconds):
    return np.full(int(round(seconds * 16000)), 0.01, dtype=np.float32)


def name(seconds):
    return f"t{int(round(seconds * 16000)) // 160}"


class _SpeedStateCase(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(granite._speed_state, logged=set(granite._speed_state["logged"]))
        granite._speed_state.update({"failure": None, "logged": set(), "fast_calls": 0, "eager_calls": 0, "budget_skips": 0})
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

    def test_attention_is_sdpa_even_when_flash_attention_is_installed(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(engine._resolve_backend_attention_implementation(GRANITE_SPEECH_MODEL_ID, "flash_attention_2"), "sdpa")
        self.assertEqual(engine._resolve_backend_attention_implementation(GRANITE_SPEECH_MODEL_ID, "sdpa"), "sdpa")


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
        self.assertEqual(granite.max_new_tokens_for(0.0), 32)
        self.assertEqual(granite.max_new_tokens_for(25.0), 282)
        self.assertEqual(granite.max_new_tokens_for(60.0), 632)
        self.assertEqual(granite.max_new_tokens_for(1000.0), granite.GRANITE_MAX_NEW_TOKENS_CAP)
        model = _FakeModel()
        granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2), clip(25)], "de", 16)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 282)

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
        self.assertEqual((PROMPT_LENGTH + generate["max_new_tokens"]) % granite.GRANITE_CACHE_LENGTH_BUCKET, 0)
        self.assertGreaterEqual(generate["max_new_tokens"], granite.max_new_tokens_for(2))
        self.assertEqual(granite.speed_stats()["fast_calls"], 1)

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

    def test_a_static_cache_over_the_budget_runs_eager(self):
        current_settings["granite_compile"] = "static_cache"
        model = _FakeModel()
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(granite, "GRANITE_STATIC_CACHE_BUDGET_MB", 1):
            granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertNotIn("cache_implementation", model.generate_calls[0])
        self.assertEqual(granite.speed_stats()["budget_skips"], 1)

    def test_switching_off_releases_a_kept_static_cache(self):
        model = _FakeModel()
        model._cache = object()
        model._compiled_call = object()
        granite.transcribe_granite_batch(_FakeProcessor(), model, [clip(2)], "de", 8)
        self.assertFalse(hasattr(model, "_cache"))
        self.assertFalse(hasattr(model, "_compiled_call"))

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
    def test_loader_uses_sdpa_left_padding_and_stores_the_granite_backend(self):
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
        self.assertEqual(model_kwargs["attn_implementation"], "sdpa")
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
