from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend import genesis_whisper_server_gpu as gpu
from backend import genesis_whisper_server_speaker_matching as matching
from backend import genesis_whisper_server_v2 as v2
from backend import genesis_whisper_server_wxc as wxc
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow, VoiceWindow


def _unit(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=192).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _fake_embed(windows, **_kwargs):
    return [
        EmbeddedVoiceWindow(
            vector=_unit(window.start_ms),
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            clean_duration_seconds=window.clean_duration_seconds,
            quality=1.0,
            stitched=False,
            source_spans=None,
        )
        for window in windows
    ]


class SlicedGpuPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_lock_is_released_between_slices_and_order_is_kept(self) -> None:
        lock = asyncio.Lock()
        events: list[str] = []

        async def fake_phase(function, items, *args):
            events.append(f"slice:{items[0]}-{items[-1]}")
            await asyncio.sleep(0.01)  # a real GPU slice suspends the coroutine
            return function(items, *args)

        async def live_request() -> None:
            # Arrives while slice one holds the lock and must run before slice two.
            async with lock:
                events.append("live")

        with mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase):
            phase = asyncio.create_task(gpu.run_sliced_gpu_phase(lock, list(range(10)), lambda part: sum(part), 4))
            await asyncio.sleep(0)  # the phase now holds the lock inside slice one
            self.assertTrue(lock.locked())
            live = asyncio.create_task(live_request())
            results = await phase
            await live

        self.assertEqual(results, [6, 22, 17])
        self.assertEqual(events, ["slice:0-3", "live", "slice:4-7", "slice:8-9"])

    async def test_empty_input_never_touches_the_lock(self) -> None:
        lock = asyncio.Lock()
        with mock.patch.object(gpu, "run_blocking_gpu_phase", side_effect=AssertionError("no slices expected")):
            self.assertEqual(await gpu.run_sliced_gpu_phase(lock, [], lambda part: part, 8), [])


class AlignmentSliceTests(unittest.TestCase):
    def test_whole_call_equals_concatenated_slices(self) -> None:
        audio = np.zeros(16000 * 60, dtype=np.float32)
        chunks = [(0.0, 20.0), (20.5, 40.0), (40.5, 60.0)]
        texts = ["eins zwei", "drei", "vier fuenf sechs"]
        seen: list[list[int]] = []

        def fake_slice(indices, _audio, _chunks, _texts, bounds):
            seen.append(list(indices))
            self.assertEqual(bounds, wxc.alignment_bounds(chunks))
            return [{"t0": float(index), "t1": float(index) + 1, "word": _texts[index]} for index in indices]

        with mock.patch.object(wxc, "align_chunk_slice", side_effect=fake_slice):
            words = wxc.align_chunk_words(audio, chunks, texts)

        self.assertEqual(seen, [[0, 1, 2]])
        self.assertEqual([word["word"] for word in words], texts)

    def test_alignment_runs_inside_the_shared_gpu_lease(self) -> None:
        events: list[str] = []

        @contextmanager
        def fake_lease():
            events.append("lease:enter")
            yield
            events.append("lease:exit")

        def fake_load():
            events.append("load_mms")
            return {"model": None, "tokenizer": None, "aligner": None, "device": "cpu"}

        audio = np.zeros(16000 * 2, dtype=np.float32)
        with (
            mock.patch.object(wxc, "shared_gpu_lease", new=fake_lease),
            mock.patch.object(wxc, "_load_mms", side_effect=fake_load),
        ):
            # Punctuation-only text has no alignable words, so no model call happens.
            words = wxc.align_chunk_slice([0], audio, [(0.0, 2.0)], ["..."], [float("-inf"), float("inf")])

        self.assertEqual(words, [])
        self.assertEqual(events, ["lease:enter", "load_mms", "lease:exit"])

    def test_bounds_sit_at_gap_midpoints(self) -> None:
        self.assertEqual(wxc.alignment_bounds([(0.0, 10.0), (11.0, 20.0)]), [float("-inf"), 10.5, float("inf")])


class VerificationSplitTests(unittest.TestCase):
    def test_plan_embed_finish_matches_the_single_call(self) -> None:
        audio = np.random.default_rng(1).uniform(-0.2, 0.2, 16000 * 40).astype(np.float32)
        sentences = [
            {"t0": index * 4.0, "t1": index * 4.0 + 3.5, "speaker": f"S{index % 2}", "overlap": False, "text": "x"}
            for index in range(10)
        ]

        with mock.patch.object(wxc, "embed_voice_windows", side_effect=_fake_embed):
            whole = wxc.verify_sentences(audio, sentences)
            windows, kept = wxc.plan_verification_windows(audio, sentences)
            split = wxc.finish_verification(windows, kept, _fake_embed(windows))

        self.assertEqual(len(windows), 10)
        self.assertEqual(whole, split)

    def test_no_usable_sentences_yield_an_empty_verification(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        windows, kept = wxc.plan_verification_windows(audio, [{"t0": 0.0, "t1": 0.05, "speaker": "S", "overlap": False}])
        self.assertEqual(windows, [])
        self.assertEqual(wxc.finish_verification(windows, kept, []), {"applied": [], "flags": [], "rounds": 0, "sentence_count": 0})


class SpeakerCloudSplitTests(unittest.TestCase):
    def test_plan_embed_build_matches_the_single_call(self) -> None:
        audio = np.random.default_rng(2).uniform(-0.2, 0.2, 16000 * 30).astype(np.float32)
        segments = [
            {"start_ms": 0, "end_ms": 12000, "speaker_id": "A"},
            {"start_ms": 12000, "end_ms": 30000, "speaker_id": "B"},
        ]

        with mock.patch.object(matching, "embed_voice_windows", side_effect=_fake_embed):
            whole = matching.extract_speaker_clouds(audio, segments, [])
            plan = matching.plan_speaker_windows(audio, segments, [])
            split = matching.build_speaker_clouds(plan, _fake_embed(plan.windows))

        self.assertEqual(plan.speaker_ids, ["A", "B"])
        self.assertEqual(len(plan.windows), len(plan.owners))
        self.assertGreater(len(plan.windows), 0)
        self.assertEqual(sorted(whole), sorted(split))
        for speaker_id in whole:
            self.assertEqual(whole[speaker_id].status, split[speaker_id].status)
            self.assertEqual(whole[speaker_id].inlier_indices, split[speaker_id].inlier_indices)
            if whole[speaker_id].prototype is not None:
                np.testing.assert_allclose(whole[speaker_id].prototype, split[speaker_id].prototype)

    def test_build_rejects_mismatched_embeddings(self) -> None:
        plan = matching.SpeakerWindowPlan(speaker_ids=["A"], owners=["A"], windows=[
            VoiceWindow(audio=np.zeros(48000, dtype=np.float32), start_ms=0, end_ms=3000, clean_duration_seconds=3.0)
        ])
        with self.assertRaises(RuntimeError):
            matching.build_speaker_clouds(plan, [])


class V2SlicedDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_driver_slices_windows_and_keeps_order(self) -> None:
        lock = asyncio.Lock()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(local_gpu_lock=lock)))
        windows = [
            VoiceWindow(audio=np.zeros(48000, dtype=np.float32), start_ms=index * 3000, end_ms=index * 3000 + 3000, clean_duration_seconds=3.0)
            for index in range(150)
        ]
        slice_sizes: list[int] = []

        async def fake_phase(function, items, *args):
            slice_sizes.append(len(items))
            return function(items, *args)

        with (
            mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase),
            mock.patch.object(v2, "embed_voice_windows", side_effect=_fake_embed),
        ):
            embedded = await v2._embed_windows_sliced(request, windows)

        self.assertEqual(slice_sizes, [64, 64, 22])
        self.assertEqual([item.start_ms for item in embedded], [window.start_ms for window in windows])

    async def test_alignment_driver_slices_chunks_and_keeps_order(self) -> None:
        lock = asyncio.Lock()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(local_gpu_lock=lock)))
        audio = np.zeros(16000 * 100, dtype=np.float32)
        chunks = [(float(index * 10), float(index * 10 + 9)) for index in range(9)]
        texts = [f"wort{index}" for index in range(9)]
        seen: list[list[int]] = []

        async def fake_phase(function, items, *args):
            return function(items, *args)

        def fake_slice(indices, _audio, _chunks, _texts, _bounds):
            seen.append(list(indices))
            return [{"t0": float(index), "t1": float(index) + 1, "word": _texts[index]} for index in indices]

        with (
            mock.patch.object(gpu, "run_blocking_gpu_phase", new=fake_phase),
            mock.patch.object(v2, "align_chunk_slice", side_effect=fake_slice),
        ):
            words = await v2._align_words_sliced(request, audio, chunks, texts)

        self.assertEqual(seen, [[0, 1, 2, 3], [4, 5, 6, 7], [8]])
        self.assertEqual([word["word"] for word in words], texts)


if __name__ == "__main__":
    unittest.main()
