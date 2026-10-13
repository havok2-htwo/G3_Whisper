"""Opt-in GPU equivalence check: Cohere CUDA graphs vs. the untouched eager model.

Loads the real CohereLabs/cohere-transcribe model once through the engine (so the install,
warmup and owner-thread logic under test is the production one) and compares transcripts
byte-for-byte against the original eager path (cuDNN on, HF generate) on random slices of the
test audio, a batch of 16, a > 35 s clip (transcribe() chunking path), mixed batches, a lazy
capture of new buckets and a non-owner thread. Needs ~6 GB free VRAM and ~2 minutes.

Run: set G3_GPU_TESTS=1 and run this module, e.g.
    venv\\Scripts\\python.exe -m unittest tests.test_cohere_cuda_graphs_gpu -v
"""

import os
import random
import shutil
import subprocess
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_RATE = 16000
VOICES_DIR = Path(os.getenv("G3_GPU_TEST_AUDIO_DIR", r"C:\Users\Worker\Desktop\voices"))
GPU_TESTS_ENABLED = os.getenv("G3_GPU_TESTS") == "1" and torch.cuda.is_available()


def _load_audio(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if rate == SAMPLE_RATE:
            return np.ascontiguousarray(mono)
    if shutil.which("ffmpeg") is None:
        raise unittest.SkipTest(f"ffmpeg fehlt fuer {path.name}")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _audio_sources() -> dict:
    paths = sorted(PROJECT_ROOT.glob("testaudio/*.wav"))
    if VOICES_DIR.is_dir():
        paths += sorted(p for p in VOICES_DIR.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac"})
    return {p.name: _load_audio(p) for p in paths}


@unittest.skipUnless(GPU_TESTS_ENABLED, "GPU-Test: nur mit CUDA und G3_GPU_TESTS=1")
class CohereCudaGraphEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend import genesis_whisper_server_local_asr_engine as engine
        from backend.genesis_whisper_server_globals import current_settings

        cls.engine = engine
        cls.settings = current_settings
        cls.saved_settings = current_settings.copy()
        current_settings["cohere_cuda_graphs"] = True
        loaded = engine.load_local_asr_model(MODEL_ID, "cuda:0", str(PROJECT_ROOT / "models"), "fp16")
        if not loaded:
            raise unittest.SkipTest(f"Cohere-Modell nicht ladbar: {engine.get_last_local_asr_load_error()}")
        cls.model = engine.local_model_components["model"]
        if not engine.cohere_cuda_graphs_active(cls.model):
            raise AssertionError("CUDA-Graphen wurden beim Laden nicht installiert.")

        sources = _audio_sources()
        if not sources:
            raise unittest.SkipTest("keine Testaudios gefunden")
        rng = random.Random(1234)
        names = sorted(sources)
        cls.cases = []
        for _ in range(24):
            name = rng.choice(names)
            audio = sources[name]
            duration = rng.uniform(0.4, min(20.0, len(audio) / SAMPLE_RATE))
            start = rng.uniform(0.0, max(0.0, len(audio) / SAMPLE_RATE - duration))
            clip = audio[int(start * SAMPLE_RATE) : int((start + duration) * SAMPLE_RATE)]
            language = "en" if "EN" in name or name.startswith("Jarvis") else "de"
            cls.cases.append((f"{name}[{start:.2f}+{duration:.2f}]", language, clip))
        cls.long_clip = np.concatenate([sources[name] for name in names])
        cls.batch16 = [clip for _, _, clip in cls.cases[:16]]
        cls.batch3 = [cls.cases[0][2], cls.cases[5][2], cls.cases[9][2]]

        # Graph run first (installed by the engine on this thread), then the untouched model.
        cls.graph_texts = cls._run_all()
        cls.graph_stats = engine.get_cohere_inference_patch_stats()
        engine._uninstall_cohere_inference_patch()
        assert "generate" not in cls.model.__dict__ and "forward" not in cls.model.encoder.__dict__
        cls.eager_texts = cls._run_all()
        # Back to graphs for the remaining tests (fresh install + warmup on this thread).
        engine._install_cohere_inference_patch(cls.model, "fp16")

    @classmethod
    def tearDownClass(cls) -> None:
        engine = cls.engine
        engine._cleanup_previous_model()
        engine.local_model_components.clear()
        engine.local_model_components.update(
            {"model": None, "processor": None, "model_identifier": None, "model_backend": None}
        )
        cls.settings.clear()
        cls.settings.update(cls.saved_settings)
        torch.cuda.empty_cache()

    @classmethod
    def _transcribe(cls, clips, language="de"):
        return cls.engine.transcribe_local_asr_batch(list(clips), language=language, batch_size=len(clips))

    @classmethod
    def _run_all(cls) -> dict:
        texts = {name: cls._transcribe([clip], language)[0] for name, language, clip in cls.cases}
        texts["batch16"] = cls._transcribe(cls.batch16)
        texts["batch3"] = cls._transcribe(cls.batch3)
        texts["long"] = cls._transcribe([cls.long_clip])[0]
        return texts

    def test_graph_transcripts_are_byte_identical_to_eager(self) -> None:
        self.assertGreater(len(self.long_clip) / SAMPLE_RATE, 35.0)
        diffs = {k: (self.eager_texts[k], self.graph_texts[k]) for k in self.eager_texts if self.eager_texts[k] != self.graph_texts[k]}
        self.assertEqual(diffs, {})
        self.assertTrue(any(self.eager_texts[name] for name, _, _ in self.cases))

    def test_graph_path_was_really_used(self) -> None:
        stats = self.graph_stats
        self.assertGreater(stats["decoder_graph"], 0)
        self.assertGreater(stats["encoder_graph"], 0)
        self.assertEqual(stats["capture_failures"], 0)
        self.assertNotIn("decoder_no_graph", stats["fallback_reasons"])
        # batch16 needs decoder B=16 and encoder B=4 graphs that the warmup does not prepare.
        self.assertIn(16, stats["decoder_graphs"])
        self.assertTrue(any(batch == 4 for batch, _ in stats["encoder_graphs"]))
        self.assertLessEqual(stats["graph_memory_mb"], stats["memory_budget_mb"])

    def test_non_owner_thread_never_captures_and_matches(self) -> None:
        handle = self.engine._COHERE_INFERENCE_PATCH
        captures_before = handle.stats()["captures"]
        name, language, clip = self.cases[3]
        box = {}

        def run():
            try:
                box["single"] = self._transcribe([clip], language)[0]
                # batch 8 -> decoder B=8 is not warmed: must run without capture on this thread
                box["batch8"] = self._transcribe([c for _, _, c in self.cases[16:24]])
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        self.assertNotIn("error", box)
        self.assertEqual(box["single"], self.eager_texts[name])
        self.assertEqual(handle.stats()["captures"], captures_before)
        # Same batch on the owner thread may capture and must give the same text.
        self.assertEqual(self._transcribe([c for _, _, c in self.cases[16:24]]), box["batch8"])

    def test_tiny_budget_falls_back_to_eager_with_same_text(self) -> None:
        from backend.genesis_whisper_server_cohere_cuda_graphs import install_cohere_cuda_graphs

        self.engine._uninstall_cohere_inference_patch()
        # 0.5 MB is below even the smallest state estimate, so nothing may be captured.
        limited = install_cohere_cuda_graphs(self.model, memory_budget_mb=0.5)
        try:
            name, language, clip = self.cases[7]
            self.assertEqual(self._transcribe([clip], language)[0], self.eager_texts[name])
            stats = limited.stats()
            self.assertEqual(stats["graphs"], 0)
            self.assertGreater(stats["decoder_fallback"], 0)
            self.assertGreater(stats["encoder_eager"], 0)
        finally:
            limited.uninstall()
            torch.cuda.empty_cache()
            self.engine._install_cohere_inference_patch(self.model, "fp16")


if __name__ == "__main__":
    unittest.main()
