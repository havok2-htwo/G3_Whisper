import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from backend import genesis_whisper_server_cohere_cuda_graphs as graphs
from backend import genesis_whisper_server_local_asr_engine as engine
from backend import genesis_whisper_server_storage as storage
from backend.genesis_whisper_server_globals import current_settings


class _FakeEncoder:
    """Stands in for the Cohere ConformerEncoder; records the cuDNN state of each call."""

    def __init__(self) -> None:
        self.cudnn_states = []
        self.pre_encode = SimpleNamespace(conv=[SimpleNamespace(weight=torch.zeros(1, dtype=torch.float16))])

    def forward(self, input_features=None, length=None, return_dict=False, **kwargs):
        self.cudnn_states.append(torch.backends.cudnn.enabled)
        return "eager-encoder", input_features

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakeModel:
    """Only the attributes CohereCudaGraphs reads; never reaches CUDA in these tests."""

    def __init__(self) -> None:
        self.encoder = _FakeEncoder()
        attention = SimpleNamespace(num_heads=2, head_dim=4, hidden_size=8)
        layer = SimpleNamespace(first_sub_layer=attention)
        self.transf_decoder = SimpleNamespace(
            _embedding=SimpleNamespace(position_embedding=SimpleNamespace(max_sequence_length=1024)),
            _decoder=SimpleNamespace(layers=[layer], final_layer_norm=None),
        )
        self.log_softmax = None
        self.encoder_decoder_proj = None
        self.generation_config = SimpleNamespace(eos_token_id=3, pad_token_id=2)
        self.generate_calls = []

    def generate(self, input_features=None, input_ids=None, length=None, attention_mask=None, **kwargs):
        self.generate_calls.append({"input_features": input_features, "length": length, **kwargs})
        return "original-generate"

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float16)


def _generate_kwargs(batch=1, frames=200, prompt_len=9, **overrides):
    kwargs = {
        "input_features": torch.zeros(batch, 128, frames),
        "length": torch.full((batch,), frames, dtype=torch.long),
        "decoder_input_ids": torch.ones(batch, prompt_len, dtype=torch.long),
        "decoder_attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "decoder_start_token_id": 7,
        "use_cache": True,
    }
    kwargs.update(overrides)
    return kwargs


class BucketMathTests(unittest.TestCase):
    def test_bucket_for_picks_smallest_fitting_bucket(self) -> None:
        self.assertEqual(graphs.bucket_for(1, (1, 2, 4)), 1)
        self.assertEqual(graphs.bucket_for(3, (1, 2, 4)), 4)
        self.assertEqual(graphs.bucket_for(9, (1, 2, 4, 8, 16)), 16)
        self.assertIsNone(graphs.bucket_for(17, (1, 2, 4, 8, 16)))

    def test_frame_buckets_cover_35_seconds_in_28_steps(self) -> None:
        self.assertEqual(graphs.frame_bucket_for(1), 128)
        self.assertEqual(graphs.frame_bucket_for(128), 128)
        self.assertEqual(graphs.frame_bucket_for(129), 256)
        # 35 s at 10 ms hop -> 3501 frames -> last bucket
        self.assertEqual(graphs.frame_bucket_for(3501), graphs.ENCODER_MAX_FRAMES)
        self.assertEqual(graphs.ENCODER_MAX_FRAMES // graphs.ENCODER_FRAME_BUCKET, 28)

    def test_encoder_output_len_matches_three_stride_two_convs(self) -> None:
        for frames in (1, 2, 7, 8, 9, 200, 1401, 3501, 3584):
            expected = frames
            for _ in range(3):
                expected = (expected + 2 - 3) // 2 + 1
            self.assertEqual(graphs.encoder_output_len(frames), expected)
        self.assertEqual(graphs.encoder_output_len(graphs.ENCODER_MAX_FRAMES), 448)

    def test_decoder_cache_len_fits_transcribe_prompt_plus_256_tokens(self) -> None:
        self.assertEqual(graphs.decoder_cache_len(9, 256), 320)
        self.assertLessEqual(graphs.decoder_cache_len(10, 256), graphs.DECODER_CACHE_LEN)
        self.assertEqual(graphs.decoder_cache_len(64, 256), 320)
        self.assertEqual(graphs.decoder_cache_len(65, 256), 384)

    def test_default_caps_bound_the_capturable_set(self) -> None:
        self.assertEqual(graphs.ENCODER_BATCH_BUCKETS, (1, 2, 4))
        self.assertEqual(graphs.DECODER_BATCH_BUCKETS, (1, 2, 4, 8, 16))
        full_set = len(graphs.ENCODER_BATCH_BUCKETS) * 28 + len(graphs.DECODER_BATCH_BUCKETS)
        self.assertLessEqual(graphs.DEFAULT_MAX_GRAPHS, full_set)
        self.assertIsNotNone(graphs.DEFAULT_MEMORY_BUDGET_MB)


class CudnnDisabledTests(unittest.TestCase):
    def test_restores_previous_state_even_on_error(self) -> None:
        before = torch.backends.cudnn.enabled
        with self.assertRaises(ValueError):
            with graphs.cudnn_disabled():
                self.assertFalse(torch.backends.cudnn.enabled)
                raise ValueError("boom")
        self.assertEqual(torch.backends.cudnn.enabled, before)


class FallbackRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _FakeModel()
        self.handle = graphs.install_cohere_cuda_graphs(self.model)

    def tearDown(self) -> None:
        self.handle.uninstall()

    def _route(self, **kwargs):
        kwargs = dict(kwargs)
        input_features = kwargs.pop("input_features")
        length = kwargs.pop("length")
        attention_mask = kwargs.pop("attention_mask", None)
        input_ids = kwargs.pop("input_ids", None)
        reason, _, _ = self.handle._decoder_route(input_features, input_ids, length, attention_mask, kwargs)
        return reason

    def test_transcribe_style_call_only_misses_cuda(self) -> None:
        self.assertEqual(self._route(**_generate_kwargs()), "device")

    def test_non_greedy_or_unknown_kwargs_route_to_original_generate(self) -> None:
        cases = {
            "sampling": _generate_kwargs(do_sample=True),
            "beams": _generate_kwargs(num_beams=4),
            "generate_kwargs": _generate_kwargs(temperature=0.7),
            "no_cache": _generate_kwargs(use_cache=False),
            "max_new_tokens": {k: v for k, v in _generate_kwargs().items() if k != "max_new_tokens"},
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(self._route(**kwargs), reason)
        output_kwargs = _generate_kwargs(output_scores=True, return_dict_in_generate=True)
        self.assertEqual(self._route(**output_kwargs), "generate_kwargs")

    def test_shapes_outside_the_buckets_route_to_original_generate(self) -> None:
        self.assertEqual(self._route(**_generate_kwargs(batch=17)), "batch")
        self.assertEqual(self._route(**_generate_kwargs(frames=3600)), "audio_len")
        self.assertEqual(self._route(**_generate_kwargs(max_new_tokens=400)), "cache_len")
        ragged = torch.ones(2, 9, dtype=torch.long)
        ragged[1, -2:] = 0
        self.assertEqual(self._route(**_generate_kwargs(batch=2, decoder_attention_mask=ragged)), "decoder_mask")
        mismatch = _generate_kwargs(batch=2)
        mismatch["decoder_input_ids"] = torch.ones(3, 9, dtype=torch.long)
        self.assertEqual(self._route(**mismatch), "inputs")

    def test_largest_supported_shapes_still_qualify(self) -> None:
        # 16 rows x 3584 frames is the largest graph shape; only the device check remains.
        self.assertEqual(self._route(**_generate_kwargs(batch=16, frames=3584)), "device")

    def test_fallback_forwards_every_argument_unchanged(self) -> None:
        kwargs = _generate_kwargs(do_sample=True, top_k=5)
        result = self.model.generate(**kwargs)
        self.assertEqual(result, "original-generate")
        call = self.model.generate_calls[-1]
        self.assertTrue(call["do_sample"])
        self.assertEqual(call["top_k"], 5)
        self.assertIs(call["input_features"], kwargs["input_features"])
        stats = self.handle.stats()
        self.assertEqual(stats["decoder_fallback"], 1)
        self.assertEqual(stats["decoder_graph"], 0)
        self.assertEqual(stats["fallback_reasons"], {"generate_kwargs": 1})

    def test_cpu_encoder_call_runs_eager_without_cudnn(self) -> None:
        before = torch.backends.cudnn.enabled
        features = torch.zeros(1, 128, 100)
        result = self.model.encoder(features, torch.tensor([100]))
        self.assertEqual(result[0], "eager-encoder")
        self.assertEqual(self.model.encoder.cudnn_states, [False])
        self.assertEqual(torch.backends.cudnn.enabled, before)
        self.assertEqual(self.handle.stats()["encoder_eager"], 1)

    def test_uninstall_restores_class_methods(self) -> None:
        self.assertIn("generate", self.model.__dict__)
        self.assertIn("forward", self.model.encoder.__dict__)
        self.handle.uninstall()
        self.assertNotIn("generate", self.model.__dict__)
        self.assertNotIn("forward", self.model.encoder.__dict__)
        self.assertEqual(self.model.generate(**_generate_kwargs()), "original-generate")
        self.assertEqual(self.handle.stats()["decoder_fallback"], 0)
        self.assertFalse(self.handle.installed)


class CaptureGuardTests(unittest.TestCase):
    def _on_other_thread(self, function):
        box = {}

        def run():
            try:
                box["result"] = function()
            except Exception as exc:  # noqa: BLE001 - surfaced through the box
                box["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        return box

    def test_capture_is_refused_on_any_thread_but_the_installer(self) -> None:
        handle = graphs.install_cohere_cuda_graphs(_FakeModel())
        try:
            self.assertIsNone(handle._capture_block_reason(0))
            box = self._on_other_thread(lambda: handle._capture_block_reason(0))
            self.assertEqual(box["result"], "fremder Thread")
            box = self._on_other_thread(handle._require_owner_thread)
            self.assertIsInstance(box["error"], RuntimeError)
            box = self._on_other_thread(handle.warmup)
            self.assertIsInstance(box["error"], RuntimeError)
            # A non-owner thread asking for a new shape gets "no graph" (eager), never a capture.
            box = self._on_other_thread(lambda: handle._encoder_graph_state(1, 256, 128, torch.device("cpu")))
            self.assertIsNone(box["result"])
            box = self._on_other_thread(lambda: handle._decoder_graph_state(1, torch.device("cpu"), torch.float16))
            self.assertIsNone(box["result"])
            self.assertEqual(handle.stats()["captures"], 0)
            self.assertGreaterEqual(handle.stats()["capture_skipped"], 2)
        finally:
            handle.uninstall()

    def test_graph_count_and_memory_budget_are_hard_ceilings(self) -> None:
        handle = graphs.install_cohere_cuda_graphs(_FakeModel(), max_graphs=0)
        try:
            self.assertIn("Graph-Limit", handle._capture_block_reason(0))
            self.assertIsNone(handle._decoder_graph_state(1, torch.device("cpu"), torch.float16))
        finally:
            handle.uninstall()

        handle = graphs.install_cohere_cuda_graphs(_FakeModel(), memory_budget_mb=1)
        try:
            self.assertIsNone(handle._capture_block_reason(512 * 1024))
            self.assertIn("Speicherbudget", handle._capture_block_reason(2 * 1024 * 1024))
            with mock.patch.object(handle, "_pool_bytes", 1024 * 1024):
                self.assertIn("Speicherbudget", handle._capture_block_reason(1))
        finally:
            handle.uninstall()

    def test_uninstalled_handle_never_captures(self) -> None:
        handle = graphs.install_cohere_cuda_graphs(_FakeModel())
        handle.uninstall()
        self.assertEqual(handle._capture_block_reason(0), "deinstalliert")


class SupportGateTests(unittest.TestCase):
    def _model_on(self, device_type: str, **attributes):
        parameter = SimpleNamespace(device=torch.device(device_type, 0) if device_type == "cuda" else torch.device("cpu"))
        parameter.is_floating_point = lambda: True
        model = _FakeModel()
        model.parameters = lambda: iter([parameter])
        for name, value in attributes.items():
            setattr(model, name, value)
        return model

    def test_cpu_model_is_not_supported(self) -> None:
        with mock.patch.object(graphs.torch.cuda, "is_available", return_value=True):
            supported, reason = graphs.cohere_cuda_graphs_supported(self._model_on("cpu"), "fp16")
        self.assertFalse(supported)
        self.assertIn("cpu", reason)

    def test_no_cuda_is_not_supported(self) -> None:
        with mock.patch.object(graphs.torch.cuda, "is_available", return_value=False):
            supported, _ = graphs.cohere_cuda_graphs_supported(self._model_on("cuda"), "fp16")
        self.assertFalse(supported)

    def test_quantized_precisions_are_not_supported(self) -> None:
        with mock.patch.object(graphs.torch.cuda, "is_available", return_value=True):
            for precision in ("int8_bnb", "fp8", "fp32"):
                with self.subTest(precision=precision):
                    supported, _ = graphs.cohere_cuda_graphs_supported(self._model_on("cuda"), precision)
                    self.assertFalse(supported)
            supported, reason = graphs.cohere_cuda_graphs_supported(self._model_on("cuda", is_quantized=True), "fp16")
            self.assertFalse(supported)
            self.assertIn("quantisiert", reason)
            for precision in ("fp16", "bf16"):
                with self.subTest(precision=precision):
                    self.assertTrue(graphs.cohere_cuda_graphs_supported(self._model_on("cuda"), precision)[0])


class _FakePatch:
    def __init__(self, events=None, warmup_error=None, model=None) -> None:
        self.events = events if events is not None else []
        self.warmup_error = warmup_error
        self.installed = True
        self.model = model

    def warmup(self):
        if self.warmup_error:
            raise self.warmup_error
        return {"graphs": 31, "graph_memory_mb": 550.0, "warmup_seconds": 4.2, "capture_failures": 0}

    def uninstall(self):
        self.events.append("uninstall")
        self.installed = False

    def stats(self):
        return {"graphs": 31}


class EngineInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_settings = current_settings.copy()
        engine._COHERE_INFERENCE_PATCH = None
        self.cuda_device = mock.patch.object(
            engine, "_get_model_device_and_dtype", return_value=(torch.device("cuda", 0), torch.float16)
        )

    def tearDown(self) -> None:
        engine._COHERE_INFERENCE_PATCH = None
        current_settings.clear()
        current_settings.update(self.original_settings)

    def test_cpu_model_gets_no_patch(self) -> None:
        with mock.patch.object(engine, "install_cohere_cuda_graphs") as install:
            engine._install_cohere_inference_patch(_FakeModel(), "fp16")
        install.assert_not_called()
        self.assertIsNone(engine._COHERE_INFERENCE_PATCH)

    def test_disabled_setting_installs_only_the_cudnn_free_encoder(self) -> None:
        current_settings["cohere_cuda_graphs"] = False
        sentinel = _FakePatch()
        with self.cuda_device, mock.patch.object(engine, "install_cohere_cuda_graphs") as install, mock.patch.object(
            engine, "install_cohere_cudnn_free_encoder", return_value=sentinel
        ):
            engine._install_cohere_inference_patch(_FakeModel(), "fp16")
        install.assert_not_called()
        self.assertIs(engine._COHERE_INFERENCE_PATCH, sentinel)

    def test_quantized_precision_skips_graphs(self) -> None:
        current_settings["cohere_cuda_graphs"] = True
        sentinel = _FakePatch()
        with self.cuda_device, mock.patch.object(
            engine, "cohere_cuda_graphs_supported", return_value=(False, "Precision 'int8_bnb' wird nicht unterstuetzt")
        ), mock.patch.object(engine, "install_cohere_cuda_graphs") as install, mock.patch.object(
            engine, "install_cohere_cudnn_free_encoder", return_value=sentinel
        ):
            engine._install_cohere_inference_patch(_FakeModel(), "int8_bnb")
        install.assert_not_called()
        self.assertIs(engine._COHERE_INFERENCE_PATCH, sentinel)

    def test_failed_warmup_falls_back_to_eager_and_uninstalls(self) -> None:
        current_settings["cohere_cuda_graphs"] = True
        events = []
        failing = _FakePatch(events, warmup_error=RuntimeError("capture broke"))
        eager = _FakePatch()
        with self.cuda_device, mock.patch.object(
            engine, "cohere_cuda_graphs_supported", return_value=(True, "")
        ), mock.patch.object(engine, "install_cohere_cuda_graphs", return_value=failing), mock.patch.object(
            engine, "install_cohere_cudnn_free_encoder", return_value=eager
        ), mock.patch.object(engine.torch.cuda, "empty_cache"):
            engine._install_cohere_inference_patch(_FakeModel(), "fp16")
        self.assertEqual(events, ["uninstall"])
        self.assertIs(engine._COHERE_INFERENCE_PATCH, eager)

    def test_successful_install_keeps_graph_handle(self) -> None:
        current_settings["cohere_cuda_graphs"] = True
        model = _FakeModel()
        handle = _FakePatch(model=model)
        with self.cuda_device, mock.patch.object(
            engine, "cohere_cuda_graphs_supported", return_value=(True, "")
        ), mock.patch.object(engine, "install_cohere_cuda_graphs", return_value=handle) as install:
            engine._install_cohere_inference_patch(model, "fp16")
        install.assert_called_once_with(model)
        self.assertIs(engine._COHERE_INFERENCE_PATCH, handle)
        self.assertTrue(engine.cohere_cuda_graphs_active(model))
        self.assertFalse(engine.cohere_cuda_graphs_active(_FakeModel()))

    def test_cleanup_uninstalls_graphs_before_emptying_the_cache(self) -> None:
        events = []
        engine._COHERE_INFERENCE_PATCH = _FakePatch(events)
        saved_components = dict(engine.local_model_components)
        engine.local_model_components.update({"model": object(), "processor": object()})
        try:
            with mock.patch.object(engine.torch.cuda, "is_available", return_value=True), mock.patch.object(
                engine.torch.cuda, "synchronize"
            ), mock.patch.object(
                engine.torch.cuda, "empty_cache", side_effect=lambda: events.append("empty_cache")
            ), mock.patch.object(engine, "DYNAMO_AVAILABLE", False):
                engine._cleanup_previous_model()
        finally:
            engine.local_model_components.clear()
            engine.local_model_components.update(saved_components)
        self.assertEqual(events, ["uninstall", "empty_cache"])
        self.assertIsNone(engine._COHERE_INFERENCE_PATCH)


class _TranscribeModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return ["text"] * len(kwargs["audio_arrays"])


class CompileGateTests(unittest.TestCase):
    def setUp(self) -> None:
        engine._COHERE_INFERENCE_PATCH = None
        engine._LOGGED_COMPILE_SKIP_REASONS.clear()

    def tearDown(self) -> None:
        engine._COHERE_INFERENCE_PATCH = None

    def _compile_flag(self, *, graphs_active: bool, has_cl: bool, has_triton: bool) -> bool:
        model = _TranscribeModel()
        if graphs_active:
            engine._COHERE_INFERENCE_PATCH = _FakePatch(model=model)
        audio = [torch.zeros(16000).numpy()]
        with mock.patch.object(
            engine, "_get_model_device_and_dtype", return_value=(torch.device("cuda", 0), torch.float16)
        ), mock.patch.object(engine.shutil, "which", return_value="cl.exe" if has_cl else None), mock.patch.object(
            engine, "_has_triton", return_value=has_triton
        ), mock.patch.object(engine.os, "name", "nt"):
            engine._cohere_transcribe_via_transcribe_api(object(), model, audio, "de", 1)
        return model.calls[-1]["compile"]

    def test_graphs_disable_compile_even_with_full_toolchain(self) -> None:
        self.assertFalse(self._compile_flag(graphs_active=True, has_cl=True, has_triton=True))

    def test_compile_requires_triton_as_well_as_a_compiler(self) -> None:
        self.assertFalse(self._compile_flag(graphs_active=False, has_cl=True, has_triton=False))
        self.assertFalse(self._compile_flag(graphs_active=False, has_cl=False, has_triton=True))
        self.assertTrue(self._compile_flag(graphs_active=False, has_cl=True, has_triton=True))


class CudaGraphSettingTests(unittest.TestCase):
    def test_setting_defaults_on_and_is_strict_bool(self) -> None:
        self.assertTrue(storage.DEFAULT_SETTINGS["cohere_cuda_graphs"])
        self.assertTrue(storage.normalize_settings({})["cohere_cuda_graphs"])
        self.assertTrue(storage.normalize_settings({"cohere_cuda_graphs": "no"})["cohere_cuda_graphs"])
        self.assertTrue(storage.normalize_settings({"cohere_cuda_graphs": 0})["cohere_cuda_graphs"])
        self.assertFalse(storage.normalize_settings({"cohere_cuda_graphs": False})["cohere_cuda_graphs"])
        self.assertTrue(storage.normalize_settings({"cohere_cuda_graphs": True})["cohere_cuda_graphs"])

    def test_engine_reads_the_setting(self) -> None:
        original = current_settings.copy()
        try:
            current_settings.pop("cohere_cuda_graphs", None)
            self.assertTrue(engine._cohere_cuda_graphs_enabled())
            current_settings["cohere_cuda_graphs"] = False
            self.assertFalse(engine._cohere_cuda_graphs_enabled())
        finally:
            current_settings.clear()
            current_settings.update(original)


if __name__ == "__main__":
    unittest.main()
