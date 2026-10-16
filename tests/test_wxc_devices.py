"""WXC model placement and shared-model safety: MMS_FA follows local_gpu_device, Silero runs one pass at a time.

Runs without a GPU: devices are faked (no tensor ever reaches CUDA) and torchaudio's MMS_FA bundle is replaced.
"""
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
import torch

from backend import genesis_whisper_server_wxc as wxc


class _FakeMmsModel:
    def __init__(self) -> None:
        self.moves: list[tuple[str, torch.dtype]] = []

    def to(self, device=None, dtype=None):
        self.moves.append((str(device), dtype))
        return self

    def eval(self):
        return self


def _fake_torchaudio(model: _FakeMmsModel) -> dict:
    bundle = types.SimpleNamespace(
        get_model=lambda: model,
        get_tokenizer=lambda: "tokenizer",
        get_aligner=lambda: "aligner",
    )
    pipelines = types.ModuleType("torchaudio.pipelines")
    pipelines.MMS_FA = bundle
    package = types.ModuleType("torchaudio")
    package.pipelines = pipelines
    return {"torchaudio": package, "torchaudio.pipelines": pipelines}


class MmsDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        empty = {"model": None, "tokenizer": None, "aligner": None, "device": None, "dtype": None}
        patcher = mock.patch.dict(wxc._mms_components, empty, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The release after a move must not need a real GPU.
        cuda_patch = mock.patch.object(wxc.torch.cuda, "is_available", return_value=False)
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)

    def _device(self, name: str):
        patcher = mock.patch.object(wxc, "resolve_local_device", return_value=torch.device(name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_mms_loads_on_the_asr_card_not_cuda0(self) -> None:
        model = _FakeMmsModel()
        self._device("cuda:1")
        with mock.patch.dict(sys.modules, _fake_torchaudio(model)):
            components = wxc._load_mms()

        self.assertEqual(components["device"], torch.device("cuda:1"))
        self.assertEqual(components["dtype"], wxc.MMS_CUDA_DTYPE)
        self.assertEqual(model.moves, [("cuda:1", wxc.MMS_CUDA_DTYPE)])

    def test_cpu_setting_keeps_mms_in_fp32_on_cpu(self) -> None:
        model = _FakeMmsModel()
        self._device("cpu")
        with mock.patch.dict(sys.modules, _fake_torchaudio(model)):
            components = wxc._load_mms()

        self.assertEqual(components["device"], torch.device("cpu"))
        self.assertEqual(model.moves, [("cpu", torch.float32)])

    def test_loaded_mms_moves_when_the_gpu_setting_changes(self) -> None:
        model = _FakeMmsModel()
        wxc._mms_components.update(
            {"model": model, "tokenizer": "t", "aligner": "a", "device": torch.device("cuda:0"), "dtype": torch.float16}
        )
        self._device("cuda:1")

        self.assertTrue(wxc.apply_mms_device_setting())
        self.assertFalse(wxc.apply_mms_device_setting())
        self.assertEqual(wxc._mms_components["device"], torch.device("cuda:1"))
        self.assertEqual(model.moves, [("cuda:1", wxc.MMS_CUDA_DTYPE)])

    def test_lazy_load_path_also_follows_the_setting(self) -> None:
        model = _FakeMmsModel()
        wxc._mms_components.update(
            {"model": model, "tokenizer": "t", "aligner": "a", "device": torch.device("cuda:0"), "dtype": torch.float16}
        )
        self._device("cuda:1")

        self.assertEqual(wxc._load_mms()["device"], torch.device("cuda:1"))
        self.assertEqual(wxc._load_mms()["device"], torch.device("cuda:1"))
        self.assertEqual(len(model.moves), 1)

    def test_apply_without_a_loaded_model_does_nothing(self) -> None:
        self._device("cuda:1")
        self.assertFalse(wxc.apply_mms_device_setting())


class _StatefulSilero:
    """Records overlapping inference: a second pass entering while one runs would corrupt the stream state."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def reset_states(self) -> None:
        pass

    def __call__(self, chunk, sample_rate):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.0005)
        with self.guard:
            self.active -= 1
        return torch.zeros(chunk.shape[0])


class SileroConcurrencyTests(unittest.TestCase):
    def test_concurrent_callers_never_interleave_on_the_shared_model(self) -> None:
        model = _StatefulSilero()
        audio = np.zeros(wxc.SILERO_WINDOW * 4, dtype=np.float32)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                wxc.silero_frame_probs(audio)
            except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
                errors.append(exc)

        with (
            mock.patch.object(wxc, "_load_silero_model", return_value=model),
            mock.patch.object(wxc, "SILERO_STREAM_FRAMES", 32),
        ):
            threads = [threading.Thread(target=run) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(model.max_active, 1)


if __name__ == "__main__":
    unittest.main()
