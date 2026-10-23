"""Parakeet TDT 0.6B v3 (ONNX via onnx-asr) as a local ASR backend.

Runs without a GPU or model download: onnx-asr, the Hugging Face snapshot and the loaded session are faked.
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_local_asr_engine as engine
from backend import genesis_whisper_server_model_manager as model_manager
from backend.genesis_whisper_server_globals import (
    LOCAL_ASR_MODEL_SPECS,
    PARAKEET_MODEL_ID,
    get_local_model_backend,
    uses_cohere_backend,
    uses_parakeet_backend,
    uses_whole_audio_backend,
    uses_whisper_backend,
)


class _FakeParakeetSession:
    """Mimics onnx-asr: a list in, a list of texts out, one entry per waveform."""

    def __init__(self):
        self.calls = []

    def recognize(self, waveforms, sample_rate):
        self.calls.append((len(waveforms), sample_rate))
        return [f"text-{len(wave)}" for wave in waveforms]


class ParakeetRegistryTests(unittest.TestCase):
    def test_model_is_registered_as_parakeet_backend(self):
        self.assertIn("Parakeet TDT 0.6B v3 (ONNX)", LOCAL_ASR_MODEL_SPECS)
        self.assertEqual(get_local_model_backend(PARAKEET_MODEL_ID), "parakeet")
        self.assertTrue(uses_parakeet_backend(PARAKEET_MODEL_ID))
        self.assertFalse(uses_cohere_backend(PARAKEET_MODEL_ID))
        self.assertFalse(uses_whisper_backend(PARAKEET_MODEL_ID))

    def test_parakeet_and_cohere_share_the_whole_audio_path_whisper_does_not(self):
        self.assertTrue(uses_whole_audio_backend(PARAKEET_MODEL_ID))
        self.assertTrue(uses_whole_audio_backend("CohereLabs/cohere-transcribe-03-2026"))
        self.assertFalse(uses_whole_audio_backend("openai/whisper-large-v3-turbo"))

    def test_download_requires_the_onnx_files_not_the_transformers_ones(self):
        specs = model_manager._supported_model_specs()
        required = specs[PARAKEET_MODEL_ID]["required_files"]
        self.assertIn("encoder-model.onnx", required)
        self.assertIn("decoder_joint-model.onnx", required)
        self.assertIn("vocab.txt", required)
        self.assertNotIn("preprocessor_config.json", required)
        self.assertEqual(specs[PARAKEET_MODEL_ID]["required_any_groups"], ())


class ParakeetBatchTests(unittest.TestCase):
    def test_batch_keeps_order_and_skips_empty_chunks(self):
        session = _FakeParakeetSession()
        audio = [
            np.zeros(1600, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(3200, dtype=np.float32).reshape(-1, 1),  # column vector is flattened
        ]

        texts = engine._transcribe_parakeet_batch(session, audio)

        self.assertEqual(texts, ["text-1600", "", "text-3200"])
        # One batched call for the non-empty chunks only, at the model's sample rate.
        self.assertEqual(session.calls, [(2, 16000)])

    def test_all_empty_batch_never_calls_the_model(self):
        session = _FakeParakeetSession()
        self.assertEqual(engine._transcribe_parakeet_batch(session, [np.zeros(0, dtype=np.float32)]), [""])
        self.assertEqual(session.calls, [])

    def test_dispatch_uses_parakeet_without_a_processor(self):
        session = _FakeParakeetSession()
        components = {"model": session, "processor": None, "model_backend": "parakeet"}
        with mock.patch.dict(engine.local_model_components, components, clear=True):
            texts = engine.transcribe_local_asr_batch([np.zeros(800, dtype=np.float32)], language="de", batch_size=8)
        self.assertEqual(texts, ["text-800"])


class ParakeetLoaderTests(unittest.TestCase):
    def test_loader_builds_onnx_session_from_the_cached_snapshot(self):
        fake_onnx_asr = types.SimpleNamespace(load_model=mock.Mock(return_value=_FakeParakeetSession()))
        with mock.patch.dict(sys.modules, {"onnx_asr": fake_onnx_asr}), mock.patch.object(
            engine, "snapshot_download", return_value="/cache/snap"
        ) as download, mock.patch.object(engine, "_register_parakeet_cuda_libraries") as register:
            session = engine._load_parakeet_model(PARAKEET_MODEL_ID, True, "/cache", None)

        self.assertIsInstance(session, _FakeParakeetSession)
        register.assert_called_once_with()
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        fake_onnx_asr.load_model.assert_called_once_with(
            "nemo-parakeet-tdt-0.6b-v3",
            path="/cache/snap",
            providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"],
        )

    def test_cpu_load_does_not_register_cuda_or_request_cuda(self):
        fake_onnx_asr = types.SimpleNamespace(load_model=mock.Mock(return_value=_FakeParakeetSession()))
        with mock.patch.dict(sys.modules, {"onnx_asr": fake_onnx_asr}), mock.patch.object(
            engine, "snapshot_download", return_value="/cache/snap"
        ), mock.patch.object(engine, "_register_parakeet_cuda_libraries") as register:
            engine._load_parakeet_model(PARAKEET_MODEL_ID, False, "", None)

        register.assert_not_called()
        self.assertEqual(fake_onnx_asr.load_model.call_args.kwargs["providers"], ["CPUExecutionProvider"])


class ParakeetVramTests(unittest.TestCase):
    def test_vram_lookup_does_not_assume_torch_parameters(self):
        # ONNX sessions have no .parameters(); the admin benchmark must not crash on them.
        with mock.patch.dict(admin.local_model_components, {"model": _FakeParakeetSession()}, clear=True), mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertIsNone(admin._get_loaded_model_cuda_index())

    def test_benchmark_workflow_label_for_parakeet(self):
        self.assertEqual(admin._benchmark_workflow(PARAKEET_MODEL_ID), "parakeet_audio_batch")
        self.assertEqual(admin._benchmark_workflow("openai/whisper-small"), "whisper_chunk_queue")


if __name__ == "__main__":
    unittest.main()
