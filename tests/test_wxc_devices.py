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


class MmsLocalCheckpointTests(unittest.TestCase):
    """speaker_turns may only use MMS_FA from disk: the check is a file test, the no-download load refuses."""

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        empty = {"model": None, "tokenizer": None, "aligner": None, "device": None, "dtype": None}
        patcher = mock.patch.dict(wxc._mms_components, empty, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.user_hub = Path(tmp.name) / "user"
        self.model_hub = Path(tmp.name) / "models"
        self.user_hub.mkdir()
        self.model_hub.mkdir()
        previous = torch.hub.get_dir()
        torch.hub.set_dir(str(self.model_hub))
        self.addCleanup(torch.hub.set_dir, previous)
        hub_patch = mock.patch.object(wxc, "_default_user_hub", return_value=self.user_hub)
        hub_patch.start()
        self.addCleanup(hub_patch.stop)
        device_patch = mock.patch.object(wxc, "resolve_local_device", return_value=torch.device("cpu"))
        device_patch.start()
        self.addCleanup(device_patch.stop)

    def _put(self, hub) -> None:
        (hub / "checkpoints").mkdir(parents=True, exist_ok=True)
        (hub / "checkpoints" / wxc.MMS_CHECKPOINT_FILE).write_bytes(b"x")

    def test_availability_follows_the_checkpoint_file_in_either_hub(self) -> None:
        self.assertFalse(wxc.mms_available_locally())
        self._put(self.model_hub)
        self.assertTrue(wxc.mms_available_locally())
        self.assertEqual(wxc._mms_hub_choice(), (self.model_hub, True))
        self._put(self.user_hub)
        self.assertEqual(wxc._mms_hub_choice(), (self.user_hub, True))

    def test_an_empty_user_checkpoints_folder_does_not_hide_the_model_hub_copy(self) -> None:
        (self.user_hub / "checkpoints").mkdir()
        self._put(self.model_hub)
        self.assertEqual(wxc._mms_hub_choice(), (self.model_hub, True))

    def test_no_download_load_refuses_without_the_file_and_never_builds_the_model(self) -> None:
        model = _FakeMmsModel()
        get_model = mock.Mock(return_value=model)
        modules = _fake_torchaudio(model)
        modules["torchaudio.pipelines"].MMS_FA.get_model = get_model
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError):
                wxc._load_mms(allow_download=False)
            get_model.assert_not_called()
            self._put(self.user_hub)
            components = wxc._load_mms(allow_download=False)
        self.assertIs(components["model"], model)
        get_model.assert_called_once()
        self.assertTrue(wxc.mms_available_locally())

    def test_slice_passes_the_no_download_flag(self) -> None:
        from contextlib import contextmanager

        @contextmanager
        def lease():
            yield

        seen = []

        def fake_load(**kwargs):
            seen.append(kwargs)
            return {"model": None, "tokenizer": None, "aligner": None, "device": "cpu"}

        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(wxc, "shared_gpu_lease", new=lease), mock.patch.object(wxc, "_load_mms", side_effect=fake_load):
            wxc.align_chunk_slice([0], audio, [(0.0, 1.0)], ["..."], [float("-inf"), float("inf")], allow_download=False)
            wxc.align_chunk_slice([0], audio, [(0.0, 1.0)], ["..."], [float("-inf"), float("inf")])
        self.assertEqual(seen, [{"allow_download": False}, {}])


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


class _CausalSilero:
    """Stateful per stream row like Silero: each output depends on the frames seen so far."""

    def __init__(self) -> None:
        self.calls = 0
        self.state = None

    def reset_states(self) -> None:
        self.state = None

    def __call__(self, chunk, sample_rate):
        self.calls += 1
        energy = chunk.abs().mean(dim=1)
        self.state = energy if self.state is None else 0.5 * self.state + energy
        return self.state.clone()


class SileroSingleStreamTests(unittest.TestCase):
    def test_one_stream_stops_at_its_last_frame_with_the_same_values(self) -> None:
        rng = np.random.default_rng(5)
        audio = rng.standard_normal(wxc.SILERO_WINDOW * 300).astype(np.float32)
        short = _CausalSilero()
        with mock.patch.object(wxc, "_load_silero_model", return_value=short):
            probs = wxc.silero_frame_probs(audio)
        self.assertEqual(short.calls, 300)
        # reference: step through the whole zero-padded stream as before
        full = _CausalSilero()
        padded = np.zeros(wxc.SILERO_STREAM_FRAMES * wxc.SILERO_WINDOW, np.float32)
        padded[: len(audio)] = audio
        stream = torch.from_numpy(padded).reshape(1, -1)
        expected = []
        for index in range(wxc.SILERO_STREAM_FRAMES):
            out = full(stream[:, index * wxc.SILERO_WINDOW:(index + 1) * wxc.SILERO_WINDOW], 16000)
            expected.append(float(out[0]))
        np.testing.assert_array_equal(probs, np.array(expected[:300], np.float32))

    def test_several_streams_still_step_the_full_stream_length(self) -> None:
        audio = np.ones(wxc.SILERO_WINDOW * (wxc.SILERO_STREAM_FRAMES + 10), np.float32)
        model = _CausalSilero()
        with mock.patch.object(wxc, "_load_silero_model", return_value=model):
            probs = wxc.silero_frame_probs(audio)
        self.assertEqual(model.calls, wxc.SILERO_STREAM_FRAMES)
        self.assertEqual(len(probs), wxc.SILERO_STREAM_FRAMES + 10)


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
