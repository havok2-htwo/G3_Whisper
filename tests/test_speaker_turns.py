import json
import unittest

import numpy as np

from backend import genesis_whisper_server_speaker_turns as st

FSTEP = 0.016875


def _voice(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


def _section(schedule, duration_s, noise=0.03, seed=0):
    """A section with speech per ``schedule`` [(start, end, voice_seed)] and windows at the production hops."""

    rng = np.random.default_rng(seed)
    n = int(np.ceil(duration_s / 0.02))
    t = (np.arange(n) + 0.5) * 0.02
    sm = np.zeros(n, bool)
    for a, b, _ in schedule:
        sm |= (t >= a) & (t < b)
    out = {"sm": sm}
    for key, (length, hop) in st.SECTION_WINDOW_SCALES.items():
        starts, vecs = [], []
        x = 0.0
        while x + length <= duration_s + 1e-9:
            mid = x + length / 2
            who = [s for a, b, s in schedule if a <= mid < b]
            if who:
                v = _voice(who[0]) + noise * rng.standard_normal(192).astype(np.float32)
                starts.append(x)
                vecs.append(v / np.linalg.norm(v))
            x += hop
        starts = np.array(starts)
        out[key] = (starts, starts + length, np.array(vecs, np.float32).reshape(-1, 192))
    return out


class RequestTests(unittest.TestCase):
    def _known(self, **over):
        entry = {"id": "s1", "vector": [0.1] * 192, "weight": 6.5}
        entry.update(over)
        return entry

    def test_valid_request_is_normalized(self) -> None:
        known = st.parse_speaker_turns_request({"known_speakers": [self._known(), self._known(id="s2", weight=2)]})
        self.assertEqual([k["id"] for k in known], ["s1", "s2"])
        self.assertAlmostEqual(float(np.linalg.norm(known[0]["vector"])), 1.0, places=5)
        self.assertEqual(st.parse_speaker_turns_request({}), [])

    def test_invalid_requests_raise(self) -> None:
        bad = [
            [],
            {"known_speakers": "x"},
            {"known_speakers": [self._known()], "extra": 1},
            {"known_speakers": [self._known(id=f"s{i}") for i in range(17)]},
            {"known_speakers": [self._known(vector=[0.1] * 191)]},
            {"known_speakers": [self._known(vector=[0.0] * 192)]},
            {"known_speakers": [self._known(vector=[float("nan")] + [0.1] * 191)]},
            {"known_speakers": [self._known(vector=[True] * 192)]},
            {"known_speakers": [self._known(weight=0)]},
            {"known_speakers": [self._known(weight="3")]},
            {"known_speakers": [self._known(id="")]},
            {"known_speakers": [self._known(id="new:1")]},
            {"known_speakers": [self._known(), self._known()]},
            {"known_speakers": [{**self._known(), "name": "Anna"}]},
        ]
        for value in bad:
            with self.assertRaises(ValueError, msg=str(value)[:80]):
                st.parse_speaker_turns_request(value)

    def test_capability_block(self) -> None:
        self.assertEqual(st.speaker_turns_capability(True), {
            "supported": True, "embedding_model": "ReDimNet2-B6", "dimension": 192, "max_known_speakers": 16,
        })
        self.assertEqual(st.speaker_turns_capability(False), {"supported": False})


class TurnTests(unittest.TestCase):
    def test_known_and_new_voice(self) -> None:
        section = _section([(0.2, 6.0, 1), (6.4, 12.8, 2)], 13.0)
        known = [{"id": "s3", "vector": _voice(1), "weight": 12.0}]
        turns = st.speaker_turns(section, known)
        again = st.speaker_turns(section, known)
        self.assertEqual([t["speaker_ref"] for t in turns], [t["speaker_ref"] for t in again])
        self.assertEqual([t["speaker_ref"] for t in turns], ["s3", "new:1"])
        first, second = turns
        self.assertLessEqual(first["end_ms"], second["start_ms"] + 20)
        self.assertGreater(first["cos"], 0.6)
        for turn in turns:
            self.assertEqual(set(turn), {"start_ms", "end_ms", "text", "speaker_ref", "cos", "vector", "support_s"})
            self.assertEqual(turn["vector"].shape, (192,))
            self.assertAlmostEqual(float(np.linalg.norm(turn["vector"])), 1.0, places=4)
            self.assertGreater(turn["support_s"], 0)

    def test_without_known_speakers_new_refs_are_numbered(self) -> None:
        section = _section([(0.2, 5.0, 1), (5.4, 10.0, 2), (10.4, 14.8, 1)], 15.0)
        refs = [t["speaker_ref"] for t in st.speaker_turns(section, [])]
        self.assertEqual(len(refs), 3)
        self.assertEqual(refs[0], refs[2])
        self.assertNotEqual(refs[0], refs[1])
        self.assertEqual(sorted(set(refs)), ["new:1", "new:2"])

    def test_too_little_speech_gives_one_turn_without_speaker(self) -> None:
        section = _section([(0.3, 1.0, 1)], 1.2)
        self.assertEqual(len(section["w15"][2]), 0)
        turns = st.speaker_turns(section, [])
        self.assertEqual(len(turns), 1)
        self.assertIsNone(turns[0]["speaker_ref"])
        self.assertIsNone(turns[0]["vector"])
        self.assertEqual(st.speaker_turns({"sm": np.zeros(50, bool), "w15": section["w15"], "w30": section["w30"]}, []), [])

    def test_speech_mask_from_segmentation(self) -> None:
        n = st.section_frames(16000 * 4)
        self.assertEqual(n, 200)
        data = np.zeros((1, 589, 3), np.float32)
        frames_t = np.arange(589) * FSTEP + FSTEP / 2
        data[0, (frames_t > 1.0) & (frames_t < 2.0), 0] = 1.0
        mask = st.section_speech_mask({"data": data, "chunk_step": 1.0, "frame_step": FSTEP}, n)
        self.assertEqual(mask.shape, (200,))
        self.assertTrue(mask[75])
        self.assertFalse(mask[25])
        self.assertFalse(mask[150])


class WordTests(unittest.TestCase):
    def test_words_go_to_the_turn_of_their_midpoint(self) -> None:
        turns = [
            {"start_ms": 0, "end_ms": 4000, "text": "", "speaker_ref": "s1", "cos": 0.9, "vector": _voice(1), "support_s": 3.5},
            {"start_ms": 4200, "end_ms": 9000, "text": "", "speaker_ref": "new:1", "cos": 0.8, "vector": _voice(2), "support_s": 4.0},
        ]
        words = [
            {"t0": 0.5, "t1": 0.9, "word": "Hallo"},
            {"t0": 3.8, "t1": 4.3, "word": "zusammen."},   # mid 4.05 -> nearest is the first turn
            {"t0": 5.0, "t1": 5.4, "word": "Ja,"},
            {"t0": 9.2, "t1": 9.6, "word": "genau."},       # after the last turn
            {"t0": 6.0, "t1": 6.2, "word": "  "},
        ]
        out = st.assign_words_to_turns(turns, words)
        self.assertEqual([t["text"] for t in out], ["Hallo zusammen.", "Ja, genau."])
        self.assertEqual(turns[0]["text"], "")  # input untouched

    def test_words_without_turns_are_kept(self) -> None:
        out = st.assign_words_to_turns([], [{"t0": 0.1, "t1": 0.4, "word": "Okay."}, {"t0": 0.5, "t1": 0.7, "word": "Hm."}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "Okay. Hm.")
        self.assertIsNone(out[0]["speaker_ref"])
        self.assertEqual((out[0]["start_ms"], out[0]["end_ms"]), (100, 700))

    def test_wire_format_is_json(self) -> None:
        section = _section([(0.2, 6.0, 1)], 6.5)
        turns = st.assign_words_to_turns(st.speaker_turns(section, []), [{"t0": 1.0, "t1": 1.5, "word": "Test"}])
        wire = st.turns_to_wire(turns)
        decoded = json.loads(json.dumps(wire))
        self.assertEqual(decoded[0]["text"], "Test")
        self.assertEqual(len(decoded[0]["vector"]), 192)
        self.assertEqual(decoded[0]["speaker_ref"], "new:1")


if __name__ == "__main__":
    unittest.main()
