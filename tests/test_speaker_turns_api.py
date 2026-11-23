"""speaker_turns in v2 transcript mode, the diarization engine switch and their settings/capabilities.

No model files and no GPU: the GPU units (seg-3.0 slices, ReDimNet windows, MMS_FA) are replaced by fakes, the CPU
logic is the real one. The driver tests compare the sliced async path with the phase-1 functions on identical fake
features, so the wiring cannot change a result.
"""

from __future__ import annotations

import asyncio
import json
import math
import types
import unittest
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import FormData

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_diarization_engine as engine
from backend import genesis_whisper_server_local_diarization as ld
from backend import genesis_whisper_server_storage as storage
from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_dia_client import _validate_dia_payload
from backend.genesis_whisper_server_lid import LanguageDecision
from backend.genesis_whisper_server_speaker_turns import section_features, speaker_turns

from tests.test_v2_api import _CapturingApp, _payload, _request, _upload

FSTEP = 0.016875
NF = 589
_GERMAN = LanguageDecision(languages=[], language="de", source="setting")


def _voice(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


def _known(speaker_id: str = "s1", seed: int = 1, weight: float = 6.0) -> dict:
    return {"id": speaker_id, "vector": _voice(seed).astype(float).tolist(), "weight": weight}


class _Meeting:
    """Fake seg-3.0 activations and ReDimNet vectors of a schedule [(start s, end s, voice seed)]."""

    def __init__(self, schedule, duration_s: float, noise: float = 0.03) -> None:
        self.schedule = schedule
        self.duration_s = duration_s
        self.noise = noise
        self.audio = np.full(int(duration_s * 16000), 0.01, np.float32)
        total, window, step = len(self.audio), 160000, 16000
        full = (total - window) // step + 1 if total >= window else 0
        nc = full + int(total < window or (total - window) % step > 0)
        slots = {seed: index % 3 for index, seed in enumerate(sorted({s for _, _, s in schedule}))}
        self.data = np.zeros((nc, NF, 3), np.float32)
        for c in range(nc):
            for f in range(NF):
                t = c + f * FSTEP + FSTEP / 2
                for a, b, s in schedule:
                    if a <= t < b:
                        self.data[c, f, slots[s]] = 1.0
        self.seg_calls: list[list[int]] = []

    def vector_at(self, start_s: float, length_s: float) -> np.ndarray:
        mid = start_s + length_s / 2
        who = [s for a, b, s in self.schedule if a <= mid < b] or [999]
        rng = np.random.default_rng(int(round(start_s * 1000)) * 7 + int(length_s * 10))
        v = _voice(who[0]) + self.noise * rng.standard_normal(192).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    # fakes of the GPU units
    def segment_range(self, indices, _audio):
        indices = list(indices)
        self.seg_calls.append(indices)
        return self.data[indices[0]: indices[0] + len(indices)]

    def overlap_prob(self) -> np.ndarray:
        # soft P(overlap) as seg-3.0 would give it: high where two local speakers are active
        return np.where(self.data.sum(-1) >= 2, 0.9, 0.02).astype(np.float32)

    def segment_range_with_overlap(self, indices, audio):
        indices = list(indices)
        return self.segment_range(indices, audio), self.overlap_prob()[indices[0]: indices[0] + len(indices)]

    def embed_native(self, windows, window_samples):
        return [self.vector_at(w.start_ms / 1000.0, window_samples / 16000.0) for w in windows]

    def embed_reference(self, _audio, starts, length_s):
        return np.stack([self.vector_at(int(s) / 16000.0, length_s) for s in starts])

    def segmentation(self) -> dict:
        from backend.genesis_whisper_server_segmentation import segmentation_payload

        return segmentation_payload(self.data, self.overlap_prob())


class _CountingLock:
    """asyncio lock that records how often it was taken and whether it is held."""

    def __init__(self) -> None:
        self.inner = asyncio.Lock()
        self.acquired = 0

    async def __aenter__(self):
        await self.inner.acquire()
        self.acquired += 1
        return self

    async def __aexit__(self, *_exc):
        self.inner.release()
        return False

    def locked(self) -> bool:
        return self.inner.locked()


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

class SpeakerTurnsParsingTests(unittest.TestCase):
    def _parse(self, body: dict) -> dict:
        return v2._parse_request_json(json.dumps({"schema_version": "2.0", "mode": "transcript", **body}))

    def test_without_the_field_the_parsed_request_is_unchanged(self) -> None:
        self.assertEqual(self._parse({}), {"schema_version": "2.0", "mode": "transcript", "language": None})

    def test_known_speakers_are_parsed_and_unit_normalised(self) -> None:
        raw = _known()
        raw["vector"] = [2.0 * x for x in raw["vector"]]
        parsed = self._parse({"speaker_turns": {"known_speakers": [raw, _known("s2", 2, 1.5)]}})
        known = parsed["speaker_turns"]["known_speakers"]
        self.assertEqual([entry["id"] for entry in known], ["s1", "s2"])
        self.assertAlmostEqual(float(np.linalg.norm(known[0]["vector"])), 1.0, places=5)
        self.assertEqual(self._parse({"speaker_turns": {}})["speaker_turns"], {"known_speakers": []})

    def test_invalid_fields_are_422_invalid_request_with_the_field_named(self) -> None:
        nan_vector = _known()
        nan_vector["vector"][3] = float("nan")
        cases = {
            "not an object": [],
            "unknown key": {"known_speakers": [], "extra": 1},
            "17 speakers": {"known_speakers": [_known(f"s{i}", i) for i in range(17)]},
            "191 dims": {"known_speakers": [{**_known(), "vector": [0.1] * 191}]},
            "NaN": {"known_speakers": [nan_vector]},
            "id not a string": {"known_speakers": [{**_known(), "id": 7}]},
            "zero vector": {"known_speakers": [{**_known(), "vector": [0.0] * 192}]},
            "negative weight": {"known_speakers": [{**_known(), "weight": -1}]},
            "duplicate id": {"known_speakers": [_known(), _known()]},
        }
        for name, value in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(v2.V2ApiError) as raised:
                    self._parse({"speaker_turns": value})
                self.assertEqual((raised.exception.status_code, raised.exception.code), (422, "INVALID_REQUEST"))
                self.assertEqual(raised.exception.details, {"field": "request.speaker_turns"})

    def test_only_transcript_mode_may_carry_the_field(self) -> None:
        for mode in ("embedding", "transcript_embedding", "diarization"):
            with self.subTest(mode=mode):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(
                        json.dumps({"schema_version": "2.0", "mode": mode, "speaker_turns": {"known_speakers": []}})
                    )
                self.assertEqual(raised.exception.code, "INVALID_REQUEST")


# ---------------------------------------------------------------------------
# Words with times
# ---------------------------------------------------------------------------

class SectionWordsTests(unittest.TestCase):
    def test_words_without_letters_take_the_time_of_their_neighbour(self) -> None:
        aligned = [
            {"t0": 0.5, "t1": 0.8, "word": "Am"},
            {"t0": 1.4, "t1": 1.9, "word": "November"},
            {"t0": 2.1, "t1": 2.4, "word": "sind"},
        ]
        words = v2.section_words("21. Am 21. November sind 30 %", aligned)
        self.assertEqual([w["word"] for w in words], ["21.", "Am", "21.", "November", "sind", "30", "%"])
        self.assertEqual([w["t0"] for w in words], [0.5, 0.5, 0.5, 1.4, 2.1, 2.1, 2.1])

    def test_nothing_aligned_gives_no_words(self) -> None:
        self.assertEqual(v2.section_words("Hallo Welt", []), [])
        self.assertEqual(v2.section_words("", [{"t0": 0, "t1": 1, "word": "x"}]), [])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

_TURNS = [
    {"start_ms": 0, "end_ms": 4000, "text": "", "speaker_ref": "s1", "cos": 0.91,
     "vector": _voice(1), "support_s": 3.9},
    {"start_ms": 5000, "end_ms": 9000, "text": "", "speaker_ref": "new:1", "cos": 0.88,
     "vector": _voice(2), "support_s": 3.8},
]
_ALIGNED = [
    {"t0": 0.5, "t1": 0.9, "word": "Guten"},
    {"t0": 1.0, "t1": 1.6, "word": "Morgen."},
    {"t0": 5.5, "t1": 6.0, "word": "Hallo"},
    {"t0": 6.1, "t1": 6.6, "word": "zusammen."},
]


class SpeakerTurnsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _CapturingApp()
        v2.create_v2_api(self.app)
        self.endpoint = self.app.routes[("POST", "/v2/audio/process")]
        self.audio = np.full(16000 * 10, 0.1, dtype=np.float32)

    def _call(self, body: dict, *, available: bool = True, audio: np.ndarray | None = None,
              turns=None, aligned=None, transcribe=None, mms_local: bool = True):
        upload = _upload()
        raw_request = json.dumps({"schema_version": "2.0", "mode": "transcript", **body})
        form = FormData([("file", upload), ("request", raw_request)])
        self.decode = mock.Mock(return_value=self.audio if audio is None else audio)
        self.transcribe = transcribe or mock.AsyncMock(
            return_value=("Guten Morgen. Hallo zusammen.", 30, 1, "asr-model", _GERMAN)
        )
        self.compute = turns if turns is not None else mock.AsyncMock(return_value=[dict(t) for t in _TURNS])
        self.align = mock.AsyncMock(return_value=list(_ALIGNED) if aligned is None else aligned)
        with (
            mock.patch.object(v2, "_parse_multipart_parts", new=mock.AsyncMock(return_value=(form, upload, raw_request))),
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", new=self.decode),
            mock.patch.object(v2, "_record_log"),
            mock.patch.object(v2, "speaker_turns_available", return_value=available),
            mock.patch.object(v2, "_transcribe_audio", new=self.transcribe),
            mock.patch.object(v2, "compute_section_turns", new=self.compute),
            mock.patch.object(v2, "_align_words_sliced", new=self.align),
            mock.patch.object(v2, "mms_available_locally", return_value=mms_local),
        ):
            return asyncio.run(self.endpoint(_request(self.app)))

    def test_turns_carry_the_contract_fields_and_the_words_of_their_time(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": [_known()]}})
        payload = _payload(response)
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["transcript"], {"text": "Guten Morgen. Hallo zusammen.", "language": "de"})
        turns = payload["result"]["speaker_turns"]
        self.assertEqual([t["text"] for t in turns], ["Guten Morgen.", "Hallo zusammen."])
        self.assertEqual([t["speaker_ref"] for t in turns], ["s1", "new:1"])
        for turn in turns:
            self.assertEqual(set(turn), {"start_ms", "end_ms", "text", "speaker_ref", "cos", "vector", "support_s"})
            self.assertEqual(len(turn["vector"]), 192)
            self.assertTrue(all(math.isfinite(x) for x in turn["vector"]))
        self.assertEqual(payload["models"]["embedding"]["id"], "ReDimNet2-B6")
        self.assertEqual(payload["models"]["segmentation"], {"id": "pyannote/segmentation-3.0"})
        self.assertIn("speaker_turns", payload["timings_ms"])
        self.assertIn("alignment", payload["timings_ms"])
        # The known speakers reach the turn logic parsed; one alignment pass over the whole section.
        known = self.compute.await_args.args[2]
        self.assertEqual(known[0]["id"], "s1")
        self.assertEqual(self.align.await_args.args[2], [(0.0, 10.0)])
        self.assertEqual(self.align.await_args.args[3], ["Guten Morgen. Hallo zusammen."])
        # never a download of MMS_FA under the GPU lock on this path
        self.assertIs(self.align.await_args.kwargs.get("allow_download"), False)

    def test_missing_mms_checkpoint_skips_the_alignment_and_warns(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": []}}, mms_local=False)
        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["transcript"]["text"], "Guten Morgen. Hallo zusammen.")
        self.assertEqual([t["text"] for t in payload["result"]["speaker_turns"]], ["", ""])
        self.assertEqual([w["code"] for w in payload["warnings"]], ["SPEAKER_TURNS_WORDS_UNALIGNED"])
        self.align.assert_not_awaited()

    def test_not_offered_is_422_before_decode_and_asr(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": []}}, available=False)
        payload = _payload(response)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(payload["error"]["details"]["reason"], "not_supported")
        self.decode.assert_not_called()
        self.transcribe.assert_not_awaited()

    def test_invalid_field_is_422_before_decode(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": [{"id": "s1", "vector": [1.0] * 12}]}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(_payload(response)["error"]["code"], "INVALID_REQUEST")
        self.decode.assert_not_called()

    def test_a_failing_speaker_part_keeps_the_transcript(self) -> None:
        response = self._call(
            {"speaker_turns": {"known_speakers": []}},
            turns=mock.AsyncMock(side_effect=RuntimeError("Kein Segmentierungsmodell")),
        )
        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["result"]["transcript"]["text"], "Guten Morgen. Hallo zusammen.")
        self.assertIsNone(payload["result"]["speaker_turns"])
        self.assertEqual(payload["warnings"][0]["code"], "SPEAKER_TURNS_UNAVAILABLE")
        self.align.assert_not_awaited()

    def test_unaligned_words_leave_the_turns_without_text_and_warn(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": []}}, aligned=[])
        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([t["text"] for t in payload["result"]["speaker_turns"]], ["", ""])
        self.assertEqual([w["code"] for w in payload["warnings"]], ["SPEAKER_TURNS_WORDS_UNALIGNED"])

    def test_audio_over_a_minute_gets_no_turns_but_its_transcript(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": []}}, audio=np.zeros(16000 * 61, np.float32))
        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(payload["result"]["speaker_turns"])
        self.assertEqual(payload["warnings"][0]["code"], "SPEAKER_TURNS_AUDIO_TOO_LONG")
        self.compute.assert_not_awaited()

    def test_section_without_speech_puts_all_words_into_one_turn_without_speaker(self) -> None:
        response = self._call({"speaker_turns": {"known_speakers": []}}, turns=mock.AsyncMock(return_value=[]))
        turns = _payload(response)["result"]["speaker_turns"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["text"], "Guten Morgen. Hallo zusammen.")
        self.assertIsNone(turns[0]["speaker_ref"])
        self.assertIsNone(turns[0]["vector"])

    def test_without_the_field_transcript_mode_is_unchanged(self) -> None:
        response = self._call({})
        payload = _payload(response)
        self.assertEqual(payload["result"], {"transcript": {"text": "Guten Morgen. Hallo zusammen.", "language": "de"}})
        self.assertEqual(payload["models"], {"asr": {"id": "asr-model", "language": "de"}})
        self.compute.assert_not_awaited()
        self.align.assert_not_awaited()


# ---------------------------------------------------------------------------
# Drivers: sliced async path == phase-1 functions
# ---------------------------------------------------------------------------

class SectionTurnsDriverTests(unittest.TestCase):
    def test_driver_equals_section_features_plus_speaker_turns(self) -> None:
        meeting = _Meeting([(0.3, 4.2, 11), (5.0, 9.6, 12)], 10.0)
        known = [{"id": "s1", "vector": _voice(11), "weight": 8.0}]
        lock = _CountingLock()
        with (
            mock.patch.object(engine, "segment_audio_range", side_effect=meeting.segment_range),
            mock.patch.object(engine, "_embed_native", side_effect=meeting.embed_native),
            mock.patch.object(engine._wxc, "silero_available_locally", return_value=False),
        ):
            turns = asyncio.run(engine.compute_section_turns(lock, meeting.audio, known))
        reference = speaker_turns(section_features(meeting.audio, meeting.segmentation(), meeting.embed_reference), known)
        strip = lambda items: [(t["start_ms"], t["end_ms"], t["speaker_ref"], t["support_s"]) for t in items]
        self.assertEqual(strip(turns), strip(reference))
        self.assertEqual([t["speaker_ref"] for t in turns], ["s1", "new:1"])
        # one seg-3.0 slice and one ReDimNet slice per window scale, each under the lock
        self.assertEqual(lock.acquired, 3)

    def _planned_starts(self, meeting, probs=None, silero_error=None):
        planned: dict = {}
        original = engine.plan_scale_starts

        def spy(num_samples, speech, frame_samples, scales):
            planned["frame_samples"] = frame_samples
            planned["starts"] = original(num_samples, speech, frame_samples, scales)
            return planned["starts"]

        silero = mock.Mock(return_value=probs, side_effect=silero_error)
        with (
            mock.patch.object(engine, "segment_audio_range", side_effect=meeting.segment_range),
            mock.patch.object(engine, "_embed_native", side_effect=meeting.embed_native),
            mock.patch.object(engine._wxc, "silero_available_locally", return_value=True),
            mock.patch.object(engine._wxc, "silero_frame_probs", new=silero),
            mock.patch.object(engine, "plan_scale_starts", side_effect=spy),
        ):
            turns = asyncio.run(engine.compute_section_turns(_CountingLock(), meeting.audio, []))
        return planned, turns, silero

    def test_windows_are_chosen_by_silero_like_the_live_bench(self) -> None:
        meeting = _Meeting([(0.3, 4.2, 11), (5.0, 9.6, 12)], 10.0)
        frames = len(meeting.audio) // 512
        t = (np.arange(frames) * 512 + 256) / 16000.0
        # Silero hears only the first voice: no window over the second may be planned, although seg-3.0 marks it.
        probs = np.where((t >= 0.3) & (t < 4.2), 0.9, 0.1).astype(np.float32)
        planned, turns, silero = self._planned_starts(meeting, probs)
        silero.assert_called_once()
        self.assertEqual(planned["frame_samples"], 512)
        for name, (length_s, hop_s) in engine.SECTION_WINDOW_SCALES.items():
            expected = ld.plan_window_starts(len(meeting.audio), length_s, hop_s, probs > 0.5, 512)
            np.testing.assert_array_equal(planned["starts"][name], expected)
        self.assertTrue(all(int(s) / 16000.0 < 4.2 for s in planned["starts"]["w15"]))
        self.assertTrue(turns)

    def test_failing_silero_falls_back_to_the_seg_mask(self) -> None:
        meeting = _Meeting([(0.3, 4.2, 11), (5.0, 9.6, 12)], 10.0)
        planned, turns, _ = self._planned_starts(meeting, silero_error=RuntimeError("kaputt"))
        self.assertEqual(planned["frame_samples"], 320)
        self.assertEqual([t["speaker_ref"] for t in turns], ["new:1", "new:2"])


class LocalDiarizationDriverTests(unittest.TestCase):
    def test_driver_equals_diarize_audio_and_passes_the_dia_validation(self) -> None:
        schedule = [(0.0, 14.0, 21), (14.5, 30.0, 22), (30.5, 44.0, 21), (44.5, 60.0, 22)]
        meeting = _Meeting(schedule, 60.0)
        probs = np.zeros(len(meeting.audio) // 512, np.float32)
        t = (np.arange(len(probs)) + 0.5) * 512 / 16000
        for a, b, _ in schedule:
            probs[(t >= a) & (t < b)] = 0.9
        lock = _CountingLock()
        with (
            mock.patch.object(engine, "SEGMENTATION_SLICE_CHUNKS", 16),
            mock.patch.object(engine, "segment_audio_range", side_effect=AssertionError("ohne Overlap-Werte")),
            mock.patch.object(engine, "segment_audio_range_with_overlap",
                              side_effect=meeting.segment_range_with_overlap),
            mock.patch.object(engine, "_embed_diarization", side_effect=meeting.embed_native),
            mock.patch.object(engine, "_embed_native", side_effect=AssertionError("FP16-Kopie erwartet")),
        ):
            payload = asyncio.run(engine.run_local_diarization(lock, meeting.audio, speech_probs=probs))
        reference = ld.diarize_audio(meeting.audio, segmentation=meeting.segmentation(), speech_probs=probs,
                                     embed=meeting.embed_reference)
        for key in ("diarization", "exclusive_diarization", "overlaps", "counts"):
            self.assertEqual(payload[key], reference[key], key)
        self.assertEqual(payload["counts"]["speakers"], 2)
        self.assertEqual(payload["model"]["engine"], "local")
        _validate_dia_payload(payload)
        # 51 chunks in slices of 16 -> 4 seg-3.0 lock holds; the slices are consecutive and cover every chunk
        self.assertEqual([c[0] for c in meeting.seg_calls], [0, 16, 32, 48])
        self.assertEqual(sum(len(c) for c in meeting.seg_calls), 51)
        self.assertGreater(lock.acquired, 4)

    def _run_driver(self, meeting, probs=None, *, params=None, half_embed=True, silero=None, seg_side=None):
        calls: list[tuple[int, int]] = []

        def unit(windows, window_samples):
            calls.append((window_samples, len(windows)))
            return meeting.embed_native(windows, window_samples)

        patches = [
            mock.patch.object(engine, "segment_audio_range_with_overlap",
                              side_effect=seg_side or meeting.segment_range_with_overlap),
            mock.patch.object(engine, "_embed_diarization" if half_embed else "_embed_native", side_effect=unit),
            mock.patch.object(engine, "_embed_native" if half_embed else "_embed_diarization",
                              side_effect=AssertionError("falscher Einbettungsweg")),
        ]
        if silero is not None:
            patches.append(mock.patch.object(engine._wxc, "silero_frame_probs", side_effect=silero))
        for patch in patches:
            patch.start()
        try:
            payload = asyncio.run(engine.run_local_diarization(_CountingLock(), meeting.audio, speech_probs=probs,
                                                               params=params, half_embed=half_embed))
        finally:
            for patch in reversed(patches):
                patch.stop()
        return payload, calls

    @staticmethod
    def _schedule_probs(meeting, schedule):
        probs = np.zeros(len(meeting.audio) // 512, np.float32)
        t = (np.arange(len(probs)) + 0.5) * 512 / 16000
        for a, b, _ in schedule:
            probs[(t >= a) & (t < b)] = 0.9
        return probs

    def test_short_windows_are_embedded_after_clustering_and_only_where_unsure(self) -> None:
        schedule = [(0.0, 14.0, 21), (14.5, 30.0, 22), (30.5, 44.0, 21), (44.5, 60.0, 22)]
        meeting = _Meeting(schedule, 60.0)
        probs = self._schedule_probs(meeting, schedule)
        planned = ld.plan_scale_starts(len(meeting.audio), probs > 0.5, 512)
        payload, calls = self._run_driver(meeting, probs)
        per_length: dict = {}
        for samples, count in calls:
            per_length[samples] = per_length.get(samples, 0) + count
        self.assertEqual(per_length[48000], len(planned["s3.0"]))
        self.assertEqual(per_length[24000], len(planned["s1.5"]))
        self.assertTrue(0 < per_length[12000] < len(planned["s0.75"]), per_length)
        # the 0.75 s windows come last (after the clustering) and in slices of at most 48 s of audio
        first75 = next(i for i, (samples, _) in enumerate(calls) if samples == 12000)
        self.assertTrue(all(samples == 12000 for samples, _ in calls[first75:]))
        self.assertTrue(all(count <= {48000: 16, 24000: 32, 12000: 64}[samples] for samples, count in calls))
        self.assertEqual(payload["counts"]["speakers"], 2)
        # V5 (V5_PARAMS + the public model): every 0.75 s window, the same speakers here
        v5, calls = self._run_driver(meeting, probs, params=ld.V5_PARAMS, half_embed=False)
        self.assertEqual(sum(c for samples, c in calls if samples == 12000), len(planned["s0.75"]))
        self.assertEqual(v5["counts"]["speakers"], 2)
        reference = ld.diarize_audio(meeting.audio, segmentation=meeting.segmentation(), speech_probs=probs,
                                     embed=meeting.embed_reference, params=ld.V5_PARAMS)
        for key in ("diarization", "exclusive_diarization", "overlaps"):
            self.assertEqual(v5[key], reference[key], key)

    def test_silero_runs_next_to_segmentation(self) -> None:
        import threading

        schedule = [(0.0, 14.0, 21), (14.5, 30.0, 22)]
        meeting = _Meeting(schedule, 30.0)
        probs = self._schedule_probs(meeting, schedule)
        silero_started = threading.Event()

        def silero(_samples):
            silero_started.set()
            return probs

        def seg(indices, audio):
            # seg-3.0 waits for Silero: serial phases (seg first) would time out here
            if not silero_started.wait(timeout=10):
                raise AssertionError("Silero laeuft nicht neben seg-3.0")
            return meeting.segment_range_with_overlap(indices, audio)

        payload, _ = self._run_driver(meeting, None, silero=silero, seg_side=seg)
        self.assertEqual(payload["counts"]["speakers"], 2)
        same, _ = self._run_driver(meeting, probs)
        self.assertEqual(payload["diarization"], same["diarization"])

    def test_sliced_segmentation_carries_the_overlap_probability_in_chunk_order(self) -> None:
        meeting = _Meeting([(0.0, 20.0, 21), (15.0, 40.0, 22)], 40.0)
        with (
            mock.patch.object(engine, "SEGMENTATION_SLICE_CHUNKS", 8),
            mock.patch.object(engine, "segment_audio_range_with_overlap",
                              side_effect=meeting.segment_range_with_overlap),
        ):
            seg = asyncio.run(engine.segment_sliced(_CountingLock(), meeting.audio, overlap=True))
        np.testing.assert_array_equal(seg["data"], meeting.data)
        np.testing.assert_array_equal(seg["overlap_prob"], meeting.overlap_prob())
        self.assertEqual(seg["overlap_prob"].shape, meeting.data.shape[:2])
        self.assertGreater(float(seg["overlap_prob"].max()), 0.5)  # the 15-20 s overlap is there

    def test_without_a_checkpoint_the_driver_raises_runtime_error(self) -> None:
        with mock.patch.object(engine, "segment_audio_range_with_overlap",
                               side_effect=RuntimeError("Kein Segmentierungsmodell")):
            with self.assertRaises(RuntimeError):
                asyncio.run(engine.run_local_diarization(_CountingLock(), np.zeros(16000 * 5, np.float32),
                                                         speech_probs=np.zeros(156, np.float32)))


# ---------------------------------------------------------------------------
# _process_diarization engine switch
# ---------------------------------------------------------------------------

class DiarizationEngineSwitchTests(unittest.TestCase):
    def _run(self, *, local_result=None, local_error=None):
        turns = [
            {"start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_00"},
            {"start_ms": 6000, "end_ms": 12000, "speaker_id": "SPEAKER_01"},
        ]
        dia_payload = local_result or {
            "model": {"id": ld.LOCAL_DIARIZATION_MODEL_ID, "engine": "local"},
            "diarization": turns,
            "exclusive_diarization": turns,
            "overlaps": [],
        }
        local = mock.AsyncMock(return_value=dia_payload, side_effect=local_error)
        dia = mock.AsyncMock(side_effect=AssertionError("the DIA server must not be called"))

        async def transcribe(_request, _audio, gated, *_rest):
            return [
                {"index": i, "start_ms": t["start_ms"], "end_ms": t["end_ms"],
                 "diarization_speaker_id": t["speaker_id"], "text": f"Satz {i}", "overlap": False}
                for i, t in enumerate(gated)
            ], 9, "asr-model"

        app = _CapturingApp()
        with (
            mock.patch.object(v2, "configured_diarization_engine", return_value="local"),
            mock.patch.object(v2, "run_local_diarization", new=local),
            mock.patch.object(v2, "silero_frame_probs", return_value=np.zeros(375, np.float32)),
            mock.patch.object(v2, "diarize_v2", new=dia),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value={})),
            mock.patch.object(v2, "_transcribe_turns", new=mock.AsyncMock(side_effect=transcribe)),
            mock.patch.object(v2, "language_detection_applies", return_value=False),
        ):
            result = asyncio.run(
                v2._process_diarization(
                    _request(app), _upload(), np.full(16000 * 12, 0.1, np.float32),
                    {"expected_speakers": None, "known_speakers": [], "speaker_refinement": "off",
                     "unknown_speaker_audio": False},
                    False, False, "turns",
                )
            )
        return result, local, dia

    def test_local_engine_replaces_the_dia_call_with_the_same_downstream(self) -> None:
        (result, timings, models, _warnings, text), local, dia = self._run()
        dia.assert_not_awaited()
        local.assert_awaited_once()
        self.assertEqual(local.await_args.kwargs["num_speakers"], None)
        self.assertIsNotNone(local.await_args.kwargs["speech_probs"])
        self.assertEqual(text, "Satz 0 Satz 1")
        self.assertEqual(models["diarization"]["engine"], "local")
        self.assertIn("diarization", timings)
        self.assertEqual([s["diarization_speaker_id"] for s in result["transcript"]["segments"]],
                         ["SPEAKER_00", "SPEAKER_01"])

    def test_unloadable_local_engine_is_503_model_unavailable_without_dia_fallback(self) -> None:
        with self.assertRaises(v2.V2ApiError) as raised:
            self._run(local_error=RuntimeError("Kein Segmentierungsmodell (segmentation-3.0) gefunden."))
        self.assertEqual((raised.exception.status_code, raised.exception.code), (503, "MODEL_UNAVAILABLE"))
        self.assertFalse(raised.exception.retryable)


# ---------------------------------------------------------------------------
# Settings and capabilities
# ---------------------------------------------------------------------------

class SettingsTests(unittest.TestCase):
    def test_defaults_keep_existing_installs_on_dia(self) -> None:
        settings = storage.normalize_settings({})
        self.assertEqual(settings["diarization_engine"], "dia")
        self.assertIs(settings["speaker_turns_enabled"], True)
        self.assertEqual(settings["segmentation_model_path"], "")
        for key in ("diarization_engine", "speaker_turns_enabled", "segmentation_model_path"):
            self.assertIn(key, storage.SERVICE_SETTINGS_FIELDS)
            self.assertNotIn(key, storage.PRESET_KEYS)

    def test_values_are_normalised(self) -> None:
        self.assertEqual(storage.normalize_settings({"diarization_engine": " LOCAL "})["diarization_engine"], "local")
        self.assertEqual(storage.normalize_settings({"diarization_engine": "pyannote"})["diarization_engine"], "dia")
        self.assertIs(storage.normalize_settings({"speaker_turns_enabled": "no"})["speaker_turns_enabled"], True)
        self.assertIs(storage.normalize_settings({"speaker_turns_enabled": False})["speaker_turns_enabled"], False)

    def test_segmentation_is_needed_by_either_feature(self) -> None:
        self.assertTrue(engine.segmentation_needed({"diarization_engine": "dia", "speaker_turns_enabled": True}))
        self.assertTrue(engine.segmentation_needed({"diarization_engine": "local", "speaker_turns_enabled": False}))
        self.assertFalse(engine.segmentation_needed({"diarization_engine": "dia", "speaker_turns_enabled": False}))

    def test_admin_lists_the_engine_options(self) -> None:
        values = [option["value"] for option in admin._settings_options()["diarization_engines"]]
        self.assertEqual(values, ["dia", "local"])


class CapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = admin.current_settings.copy()
        app = FastAPI()
        v2.create_v2_capabilities_api(app)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        admin.current_settings.clear()
        admin.current_settings.update(self.original)

    def _caps(self, checkpoint: bool, mms_local: bool = True, **values) -> dict:
        admin.current_settings.clear()
        admin.current_settings.update(storage.normalize_settings({**storage.DEFAULT_SETTINGS, **values}))
        with (
            mock.patch.object(engine, "segmentation_available", return_value=checkpoint),
            mock.patch.object(engine._wxc, "mms_available_locally", return_value=mms_local),
        ):
            response = self.client.get("/v2/capabilities")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_speaker_turns_offered_with_checkpoint_and_switch(self) -> None:
        self.assertEqual(
            self._caps(True)["speaker_turns"],
            {"supported": True, "embedding_model": "ReDimNet2-B6", "dimension": 192, "max_known_speakers": 16},
        )

    def test_speaker_turns_not_offered_without_checkpoint_or_switch(self) -> None:
        self.assertEqual(self._caps(False)["speaker_turns"], {"supported": False})
        self.assertEqual(self._caps(True, speaker_turns_enabled=False)["speaker_turns"], {"supported": False})

    def test_speaker_turns_not_offered_without_a_local_mms_checkpoint(self) -> None:
        # The turn texts need MMS_FA word times; offering the field would mean a download under the GPU lock.
        self.assertEqual(self._caps(True, mms_local=False)["speaker_turns"], {"supported": False})

    def test_diarization_reports_the_engine(self) -> None:
        self.assertEqual(self._caps(True)["diarization"]["engine"], "dia")
        local = self._caps(True, diarization_engine="local")["diarization"]
        self.assertEqual(local, {"configured": True, "engine": "local", "model": ld.LOCAL_DIARIZATION_MODEL_ID})
        self.assertFalse(self._caps(False, diarization_engine="local")["diarization"]["configured"])


class SegmentationAvailabilityTests(unittest.TestCase):
    def test_a_failed_load_withdraws_the_offer_until_the_settings_change(self) -> None:
        from backend import genesis_whisper_server_segmentation as seg

        state = dict(seg._segmentation_state)
        self.addCleanup(lambda: (seg._segmentation_state.clear(), seg._segmentation_state.update(state)))
        with mock.patch.object(seg, "resolve_segmentation_checkpoint", return_value=seg.Path("/m/pytorch_model.bin")):
            seg._segmentation_state.update({"model": None, "failed_path": None})
            self.assertTrue(seg.segmentation_available())
            seg._segmentation_state["failed_path"] = str(seg.Path("/m/pytorch_model.bin"))
            self.assertFalse(seg.segmentation_available())
            seg.apply_segmentation_settings(True)
            self.assertTrue(seg.segmentation_available())

    def test_chunk_ranges_cut_like_the_whole_signal(self) -> None:
        from backend import genesis_whisper_server_segmentation as seg

        audio = np.random.default_rng(3).standard_normal(int(16000 * 23.4)).astype(np.float32)
        whole = seg.chunk_audio(audio)
        self.assertEqual(len(whole), seg.chunk_count(len(audio)))
        parts = np.concatenate([seg.chunk_audio(audio, first=i, count=5) for i in range(0, len(whole), 5)])
        np.testing.assert_array_equal(parts, whole)


if __name__ == "__main__":
    unittest.main()
