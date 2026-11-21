import unittest

import numpy as np

from backend import genesis_whisper_server_local_diarization as ld
from backend.genesis_whisper_server_dia_client import _validate_dia_payload
from backend.genesis_whisper_server_vid import VoiceWindow, _prepare_waveform_batch

FSTEP = 0.016875
NF = 589


def _voice(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


def _synthetic(schedule, duration_s, *, overlap=(), noise=0.03, seed=0):
    """Features of a fake meeting: ``schedule`` = [(start, end, speaker)], disjoint; ``overlap`` = extra
    [(start, end, speaker)] spans where a second speaker talks at the same time."""

    rng = np.random.default_rng(seed)
    voices = {k: _voice(100 + k) for k in {s for _, _, s in schedule}}

    def who(t):
        out = [s for a, b, s in schedule if a <= t < b]
        out += [s for a, b, s in overlap if a <= t < b]
        return out

    # seg-3.0: chunk c starts at c s; local speaker slot = global id (<= 3 speakers in these tests)
    total, window, step = int(duration_s * 16000), 160000, 16000
    full = (total - window) // step + 1 if total >= window else 0
    nc = full + int(total < window or (total - window) % step > 0)
    data = np.zeros((nc, NF, 3), np.float32)
    for c in range(nc):
        for f in range(NF):
            for s in who(c + f * FSTEP + FSTEP / 2):
                data[c, f, s % 3] = 1.0
    seg = {"data": data, "chunk_step": 1.0, "frame_step": FSTEP}

    windows = {}
    for name, (length, hop) in ld.BATCH_WINDOW_SCALES.items():
        st, vec = [], []
        t = 0.0
        while t + length <= duration_s + 1e-9:
            mid = t + length / 2
            speakers = who(mid)
            if speakers:
                v = voices[speakers[0]] + noise * rng.standard_normal(192).astype(np.float32)
                st.append(t)
                vec.append(v / np.linalg.norm(v))
            t += hop
        st = np.array(st)
        windows[name] = (st, st + length, np.array(vec, np.float32).reshape(-1, 192))
    return ld.Features(seg, windows, duration_s)


class ViterbiTests(unittest.TestCase):
    def test_penalty_suppresses_short_switches(self) -> None:
        score = np.zeros((10, 2), np.float32)
        score[:, 0] = 1.0
        score[4, 1] = 3.0  # one frame clearly better for label 1
        self.assertEqual(ld.viterbi(score, 3.0).tolist(), [0] * 10)
        self.assertEqual(ld.viterbi(score, 0.5).tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    def test_free_frames_switch_without_penalty(self) -> None:
        score = np.zeros((6, 2), np.float32)
        score[:3, 0] = 1.0
        score[3:, 1] = 0.4
        self.assertEqual(ld.viterbi(score, 3.0).tolist(), [0] * 6)
        free = np.array([False, False, False, True, False, False])
        self.assertEqual(ld.viterbi(score, 3.0, free).tolist(), [0, 0, 0, 1, 1, 1])

    def test_free_switch_frames_after_pauses(self) -> None:
        sm = np.array([1, 1, 0, 0, 0, 1, 0, 1], bool)
        self.assertEqual(ld.free_switch_frames(sm, 3).tolist(), [False] * 5 + [True, False, False])


class AuditTests(unittest.TestCase):
    def test_low_purity_cluster_is_flagged(self) -> None:
        votes = [
            (0, 0, np.array([90, 0, 10])),
            (0, 1, np.array([0, 80, 20])),
            (1, 0, np.array([40, 30, 30])),  # cluster 2 never dominates a local speaker
        ]
        pur, tot = ld._audit(votes, 3, 0.5)
        self.assertEqual(tot.tolist(), [130, 110, 60])
        self.assertEqual(int(np.argmin(pur)), 2)
        self.assertEqual(pur[2], 0.0)
        self.assertAlmostEqual(pur[0], 90 / 130)  # the 40-vote group is not dominated (< 50 %)


class PipelineTests(unittest.TestCase):
    SCHEDULE = [(0.5, 12.0, 0), (12.5, 25.0, 1), (25.5, 38.0, 0), (38.5, 50.0, 1)]

    def test_two_speakers_found_and_deterministic(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0)
        res = ld.diarize_features(feats)
        again = ld.diarize_features(feats)
        np.testing.assert_array_equal(res.active, again.active)
        self.assertEqual(res.active.shape[1], 2)
        # the primary label follows the schedule away from the turn boundaries
        lab_a = res.primary[int(5 / ld.GRID_S)]
        lab_b = res.primary[int(18 / ld.GRID_S)]
        self.assertNotEqual(lab_a, lab_b)
        self.assertEqual(res.primary[int(30 / ld.GRID_S)], lab_a)
        self.assertEqual(res.primary[int(45 / ld.GRID_S)], lab_b)
        self.assertEqual(res.primary[int(12.2 / ld.GRID_S)], -1)  # silence

    def test_overlap_gets_a_second_speaker(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        res = ld.diarize_features(feats)
        t = int(21 / ld.GRID_S)
        self.assertEqual(int(res.active[t].sum()), 2)
        payload = ld.result_to_dia_payload(res, 52000)
        self.assertTrue(any(o["start_ms"] <= 21000 < o["end_ms"] for o in payload["overlaps"]))

    def test_speaker_bounds(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0)
        one = ld.diarize_features(feats, num_speakers=1)
        self.assertEqual(one.active.shape[1], 1)
        capped = ld.diarize_features(feats, max_speakers=1)
        self.assertEqual(capped.active.shape[1], 1)
        three = _synthetic([(0.5, 15.0, 0), (15.5, 30.0, 1), (30.5, 45.0, 2)], 46.0)
        self.assertEqual(ld.diarize_features(three).active.shape[1], 3)
        self.assertLessEqual(ld.diarize_features(three, max_speakers=2).active.shape[1], 2)

    def test_payload_is_a_valid_dia_v2_answer(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        payload = ld.result_to_dia_payload(ld.diarize_features(feats), 52000, num_speakers=None)
        self.assertIs(_validate_dia_payload(payload), payload)
        self.assertEqual(payload["counts"]["speakers"], 2)
        self.assertEqual(payload["exclusive_diarization"][0]["speaker_id"], "SPEAKER_00")
        for seg in payload["exclusive_diarization"]:
            self.assertEqual(seg["start_ms"] % 20, 0)
        # exclusive turns never overlap
        ex = payload["exclusive_diarization"]
        for a, b in zip(ex, ex[1:]):
            self.assertLessEqual(a["end_ms"], b["start_ms"])

    def test_silence_and_tiny_inputs(self) -> None:
        silent = ld.Features({"data": np.zeros((1, NF, 3), np.float32), "chunk_step": 1.0, "frame_step": FSTEP},
                             {k: (np.zeros(0), np.zeros(0), np.zeros((0, 192), np.float32)) for k in ("s3.0", "s1.5")},
                             4.0)
        res = ld.diarize_features(silent)
        payload = ld.result_to_dia_payload(res, 4000)
        _validate_dia_payload(payload)
        self.assertEqual(payload["diarization"], [])
        short = _synthetic([(0.2, 1.2, 0)], 1.4)
        res = ld.diarize_features(short)
        self.assertEqual(res.active.shape[1], 1)
        self.assertTrue(res.active[int(0.8 / ld.GRID_S), 0])


class WindowPlanTests(unittest.TestCase):
    def test_plan_keeps_windows_with_enough_speech(self) -> None:
        speech = np.zeros(500, bool)  # 10 s on the 20 ms grid
        speech[100:200] = True        # 2-4 s
        starts = ld.plan_window_starts(160000, 3.0, 0.5, speech, 320)
        self.assertEqual((starts / 16000).tolist(), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertEqual(len(ld.plan_window_starts(16000, 3.0, 0.5, speech, 320)), 0)

    def test_compute_window_features_uses_native_length(self) -> None:
        seen = []

        def fake_embed(audio, starts, length_s):
            seen.append((len(starts), length_s))
            return np.tile(_voice(1), (len(starts), 1))

        audio = np.zeros(16000 * 6, np.float32)
        speech = np.ones(300, bool)
        out = ld.compute_window_features(audio, speech, 320, embed=fake_embed)
        self.assertEqual(seen, [(7, 3.0), (19, 1.5)])
        self.assertEqual(out["s1.5"][2].shape, (19, 192))
        np.testing.assert_allclose(out["s1.5"][1] - out["s1.5"][0], 1.5)

    def test_waveform_batch_can_use_other_window_lengths(self) -> None:
        windows = [VoiceWindow(audio=np.ones(24000, np.float32)), VoiceWindow(audio=np.ones(10000, np.float32))]
        batch = _prepare_waveform_batch(windows, 4, 24000)
        self.assertEqual(batch.shape, (4, 24000))
        self.assertEqual(float(batch[1].sum()), 24000.0)  # short window tiled to the length
        self.assertEqual(float(batch[2:].sum()), 0.0)
        self.assertEqual(_prepare_waveform_batch(windows, 2).shape, (2, 48000))


if __name__ == "__main__":
    unittest.main()
