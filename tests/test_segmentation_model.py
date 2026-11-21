import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend import genesis_whisper_server_segmentation as seg


def _random_model(seed: int = 0) -> seg.PyanNet:
    torch.manual_seed(seed)
    model = seg.PyanNet()
    with torch.no_grad():
        # spread the sinc bands so the filters are well defined (asteroid init values, slightly perturbed)
        model.sincnet.conv1d[0].filterbank.low_hz_.mul_(1.0 + 0.01 * torch.rand_like(model.sincnet.conv1d[0].filterbank.low_hz_))
    return model.eval()


class _Payload:
    """Stands for a pyannote class (Specifications etc.) pickled next to the weights."""

    def __init__(self) -> None:
        self.problem = "multilabel"


class _Exploit:
    """A pickle that would run a shell command on load with the stock unpickler."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"echo pwned > \"{self.marker}\"",))


_CALLS: list = []


def _record_call(value: str) -> str:
    _CALLS.append(value)
    return value


class _Probe:
    """Pickles as a call of _record_call: proves the stock unpickler would run the callable, ours does not."""

    def __reduce__(self):
        return (_record_call, ("called",))


class PyanNetShapeTests(unittest.TestCase):
    def test_frame_grid_matches_pyannote_receptive_field(self) -> None:
        start, step, duration = seg.frame_grid()
        self.assertEqual(start, 0.0)
        self.assertAlmostEqual(step, 0.016875)
        self.assertAlmostEqual(duration, 0.0619375)
        self.assertEqual(seg.num_output_frames(160000), 589)

    def test_forward_shape_and_log_probabilities(self) -> None:
        model = _random_model()
        with torch.inference_mode():
            out = model(torch.randn(2, 1, 160000) * 0.1)
        self.assertEqual(tuple(out.shape), (2, 589, 7))
        np.testing.assert_allclose(out.exp().sum(-1).numpy(), 1.0, atol=1e-5)

    def test_state_dict_names_follow_pyannote(self) -> None:
        keys = set(seg.PyanNet().state_dict())
        for key in (
            "sincnet.wav_norm1d.weight",
            "sincnet.conv1d.0.filterbank.low_hz_",
            "sincnet.conv1d.0.filterbank.band_hz_",
            "sincnet.conv1d.1.weight",
            "sincnet.norm1d.2.bias",
            "lstm.weight_ih_l3_reverse",
            "linear.1.weight",
            "classifier.bias",
        ):
            self.assertIn(key, keys)
        # window_ / n_ are derived buffers, never expected from the checkpoint
        self.assertNotIn("sincnet.conv1d.0.filterbank.window_", keys)

    def test_powerset_mapping_and_hard_conversion(self) -> None:
        mapping = seg.powerset_mapping()
        self.assertEqual(mapping.tolist(), [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        ])
        logits = torch.full((1, 3, 7), -10.0)
        logits[0, 0, 0] = 0.0
        logits[0, 1, 5] = 0.0
        logits[0, 2, 2] = 0.0
        multi = seg.powerset_to_multilabel(logits, torch.from_numpy(mapping))
        self.assertEqual(multi[0].tolist(), [[0, 0, 0], [1, 0, 1], [0, 1, 0]])


class ChunkingTests(unittest.TestCase):
    def test_chunks_like_pyannote_inference(self) -> None:
        sr = 16000
        cases = {5.0: 1, 10.0: 1, 11.0: 2, 25.5: 17, 30.0: 21}
        for seconds, expected in cases.items():
            audio = np.arange(int(seconds * sr), dtype=np.float32)
            chunks = seg.chunk_audio(audio)
            self.assertEqual(chunks.shape, (expected, 160000), seconds)
            self.assertEqual(chunks[1 if expected > 1 else 0, 0], (16000.0 if expected > 1 else 0.0))
        # the remainder chunk starts right after the last full chunk and is zero-padded
        audio = np.ones(int(25.5 * sr), np.float32)
        chunks = seg.chunk_audio(audio)
        last = chunks[-1]
        self.assertEqual(int(last.sum()), int(25.5 * sr) - 16 * sr)
        self.assertEqual(last[-1], 0.0)

    def test_segment_returns_seg_arrays(self) -> None:
        model = seg.SegmentationModel(_random_model(), torch.device("cpu"))
        audio = (np.random.default_rng(0).standard_normal(16000 * 12) * 0.05).astype(np.float32)
        out = model.segment(audio)
        self.assertEqual(out["data"].shape, (3, 589, 3))
        self.assertTrue(set(np.unique(out["data"]).tolist()) <= {0.0, 1.0})
        self.assertLessEqual(out["data"].sum(-1).max(), 2.0)
        self.assertEqual((out["chunk_step"], out["chunk_dur"]), (1.0, 10.0))
        self.assertAlmostEqual(out["frame_step"], 0.016875)


class CheckpointLoadingTests(unittest.TestCase):
    def test_round_trip_through_restricted_unpickler(self) -> None:
        model = _random_model(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save({
                "state_dict": model.state_dict(),
                "hyper_parameters": {"sincnet": {"stride": 10, "sample_rate": 16000}, "sample_rate": 16000,
                                     "lstm": {"num_layers": 4}},
                "pyannote.audio": {"specifications": _Payload()},
            }, path)
            loaded = seg.SegmentationModel.from_checkpoint(path)
        x = torch.randn(1, 1, 160000) * 0.1
        with torch.inference_mode():
            np.testing.assert_array_equal(model(x).numpy(), loaded.model(x).numpy())

    def test_dangerous_globals_are_never_called(self) -> None:
        model = _random_model(1)
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "pwned.txt"
            path = Path(tmp) / "evil.bin"
            torch.save({"state_dict": model.state_dict(), "callback": _Exploit(str(marker)),
                        "specs": _Payload()}, path)
            state, hparams = seg.load_checkpoint_state(path)
            raw = torch.load(str(path), map_location="cpu", pickle_module=seg._restricted_pickle, weights_only=False)
            self.assertFalse(marker.exists())
            self.assertIsInstance(raw["callback"], seg._InertObject)
            self.assertIsInstance(raw["specs"], seg._InertObject)
            self.assertNotIsInstance(raw["specs"], _Payload)
        self.assertIn("classifier.weight", state)
        self.assertEqual(hparams, {})

    def test_unpickler_resolves_only_the_allow_list(self) -> None:
        import io
        import pickle

        unpickler = seg._RestrictedUnpickler(io.BytesIO(b""))
        self.assertIs(unpickler.find_class("collections", "OrderedDict"), __import__("collections").OrderedDict)
        for module, name in (("os", "system"), ("builtins", "eval"), ("subprocess", "Popen"), ("builtins", "getattr")):
            stub = unpickler.find_class(module, name)
            self.assertTrue(issubclass(stub, seg._InertObject), (module, name))
        payload = pickle.dumps(_Exploit("never"))
        self.assertIsInstance(seg._restricted_pickle.loads(payload), seg._InertObject)

    def test_counter_check_stock_pickle_would_call(self) -> None:
        import pickle

        _CALLS.clear()
        payload = pickle.dumps(_Probe())
        self.assertIsInstance(seg._restricted_pickle.loads(payload), seg._InertObject)
        self.assertEqual(_CALLS, [])
        self.assertEqual(pickle.loads(payload), "called")  # the stock unpickler runs it
        self.assertEqual(_CALLS, ["called"])

    def test_rejects_non_segmentation_checkpoint(self) -> None:
        state = seg.PyanNet(num_classes=4).state_dict()
        with self.assertRaises(ValueError):
            seg.build_model(state)
        state = dict(seg.PyanNet().state_dict())
        del state["classifier.weight"]
        with self.assertRaises((ValueError, KeyError)):
            seg.build_model(state)


class _ScriptedSideEffect(torch.nn.Module):
    """A scripted module whose __setstate__ writes a marker file: runs inside torch.jit.load."""

    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight

    @torch.jit.export
    def __getstate__(self):
        return (self.marker, self.weight, self.training)

    @torch.jit.export
    def __setstate__(self, state: tuple[str, torch.Tensor, bool]) -> None:
        self.marker = state[0]
        self.weight = state[1]
        self.training = state[2]
        torch.save(torch.zeros(1), self.marker)


class TorchScriptRefusalTests(unittest.TestCase):
    def _scripted(self, tmp: Path) -> tuple[Path, Path]:
        marker = tmp / "pwned.pt"
        path = tmp / "pytorch_model.bin"
        torch.jit.save(torch.jit.script(_ScriptedSideEffect(str(marker))), str(path))
        if marker.exists():
            marker.unlink()  # scripting itself does not run __setstate__; make sure the check starts clean
        return path, marker

    def test_counter_check_torch_load_runs_the_scripted_setstate(self) -> None:
        # Proves the attack is real: torch.load ignores pickle_module for a TorchScript zip.
        with tempfile.TemporaryDirectory() as tmp:
            path, marker = self._scripted(Path(tmp))
            torch.load(str(path), map_location="cpu", pickle_module=seg._restricted_pickle, weights_only=False)
            self.assertTrue(marker.exists())

    def test_torchscript_archive_is_refused_before_any_code_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, marker = self._scripted(Path(tmp))
            with self.assertRaises(ValueError):
                seg.load_checkpoint_state(path)
            with self.assertRaises(ValueError):
                seg.SegmentationModel.from_checkpoint(path)
            self.assertFalse(marker.exists())

    def test_marker_records_are_refused_in_a_plain_zip_too(self) -> None:
        import io
        import zipfile

        for name in ("archive/constants.pkl", "constants.pkl", "archive/code/__torch__.py", "code/x.py"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("archive/data.pkl", b"x")
                archive.writestr(name, b"x")
            with self.assertRaises(ValueError, msg=name):
                seg._refuse_torchscript(buffer.getvalue())
        with self.assertRaises(ValueError):
            seg._refuse_torchscript(b"PK\x03\x04 not really a zip")

    def test_a_plain_torch_save_checkpoint_passes_the_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save({"state_dict": _random_model(2).state_dict()}, path)
            seg._refuse_torchscript(path.read_bytes())
            state, _ = seg.load_checkpoint_state(path)
        self.assertIn("classifier.weight", state)


class LoadLockTests(unittest.TestCase):
    def setUp(self) -> None:
        state = dict(seg._segmentation_state)
        self.addCleanup(lambda: (seg._segmentation_state.clear(), seg._segmentation_state.update(state)))
        seg._segmentation_state.update({"model": None, "path": None, "device": None, "failed_path": None,
                                        "error": None})

    def test_a_running_load_does_not_block_the_availability_check(self) -> None:
        import threading
        from unittest import mock

        started, release = threading.Event(), threading.Event()
        checkpoint = Path("/m/pytorch_model.bin")

        def slow_load(_path, _device):
            started.set()
            release.wait(10)  # stands for the GPU lease wait / checkpoint read
            return mock.Mock(device=torch.device("cpu"))

        results: dict = {}
        with (
            mock.patch.object(seg, "resolve_segmentation_checkpoint", return_value=checkpoint),
            mock.patch.object(seg, "_target_device", return_value=torch.device("cpu")),
            mock.patch.object(seg.SegmentationModel, "from_checkpoint", side_effect=slow_load),
        ):
            loader = threading.Thread(target=lambda: results.setdefault("model", seg.load_segmentation_model()))
            loader.start()
            self.assertTrue(started.wait(5))
            probe = threading.Thread(target=lambda: results.setdefault("available", seg.segmentation_available()))
            probe.start()
            probe.join(2)
            status_alive = probe.is_alive()
            status = seg.segmentation_status()
            release.set()
            loader.join(5)
        self.assertFalse(status_alive, "segmentation_available waited for the running load")
        self.assertTrue(results["available"])
        self.assertFalse(status["loaded"])
        self.assertIsNotNone(results["model"])

    def test_concurrent_loads_load_once(self) -> None:
        import threading
        from unittest import mock

        calls: list = []
        gate = threading.Event()

        def load(_path, _device):
            calls.append(1)
            gate.wait(5)
            return mock.Mock(device=torch.device("cpu"))

        with (
            mock.patch.object(seg, "resolve_segmentation_checkpoint", return_value=Path("/m/pytorch_model.bin")),
            mock.patch.object(seg, "_target_device", return_value=torch.device("cpu")),
            mock.patch.object(seg.SegmentationModel, "from_checkpoint", side_effect=load),
        ):
            out: list = []
            threads = [threading.Thread(target=lambda: out.append(seg.load_segmentation_model())) for _ in range(3)]
            for thread in threads:
                thread.start()
            gate.set()
            for thread in threads:
                thread.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len({id(m) for m in out}), 1)


class DiscoveryTests(unittest.TestCase):
    def test_explicit_path_then_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "my.bin"
            explicit.write_bytes(b"x")
            self.assertEqual(seg.resolve_segmentation_checkpoint({"segmentation_model_path": str(explicit)}), explicit.resolve())
            self.assertIsNone(seg.resolve_segmentation_checkpoint({"segmentation_model_path": str(root / "missing.bin")}))
            snap = root / "models--pyannote--speaker-diarization-community-1" / "snapshots" / "abc" / "segmentation"
            snap.mkdir(parents=True)
            (snap / "pytorch_model.bin").write_bytes(b"x")
            found = seg.resolve_segmentation_checkpoint({"local_model_cache_path": str(root)})
            # settings without local_model_cache_path fall back to the live settings; call the cache helper directly
            self.assertEqual(seg._snapshot_file(root, "pyannote/speaker-diarization-community-1",
                                                "segmentation/pytorch_model.bin"), snap / "pytorch_model.bin")
            self.assertTrue(found is None or found.is_file())


if __name__ == "__main__":
    unittest.main()
