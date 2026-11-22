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


def _synthetic(schedule, duration_s, *, overlap=(), noise=0.03, seed=0, voices=None):
    """Features of a fake meeting: ``schedule`` = [(start, end, speaker)], disjoint; ``overlap`` = extra
    [(start, end, speaker)] spans where a second speaker talks at the same time; ``voices``: speaker -> vector."""

    rng = np.random.default_rng(seed)
    voices = voices or {k: _voice(100 + k) for k in {s for _, _, s in schedule}}

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
        self.assertEqual(seen, [(7, 3.0), (19, 1.5), (22, 0.75)])
        self.assertEqual(out["s1.5"][2].shape, (19, 192))
        np.testing.assert_allclose(out["s1.5"][1] - out["s1.5"][0], 1.5)

    def test_waveform_batch_can_use_other_window_lengths(self) -> None:
        windows = [VoiceWindow(audio=np.ones(24000, np.float32)), VoiceWindow(audio=np.ones(10000, np.float32))]
        batch = _prepare_waveform_batch(windows, 4, 24000)
        self.assertEqual(batch.shape, (4, 24000))
        self.assertEqual(float(batch[1].sum()), 24000.0)  # short window tiled to the length
        self.assertEqual(float(batch[2:].sum()), 0.0)
        self.assertEqual(_prepare_waveform_batch(windows, 2).shape, (2, 48000))


def _naive_reseg_score(s, C, scales):
    """The V4 frame score (one slice add per window), the reference for the vectorized ``_reseg``."""

    score = np.zeros((s.n, len(C)), np.float64)
    cnt = np.zeros(s.n)
    for scale, wgt in scales:
        st, en, X = s.windows[scale]
        S = X @ C.T
        fa, fb = ld.window_frames(st, en, s.n)
        mid = (fa + fb) // 2
        half = np.maximum(1, ((fb - fa) * s.p["core"]).astype(int) // 2)
        lo, hi = np.maximum(0, mid - half), np.minimum(s.n, mid + half)
        for i in range(len(X)):
            score[lo[i]:hi[i]] += wgt * S[i]
            cnt[lo[i]:hi[i]] += wgt
    has = cnt > 0
    score[has] /= cnt[has, None]
    return score, has


def _state(feats, **params):
    p = dict(ld.DEFAULT_PARAMS, **params)
    n = ld.grid_frames(feats.duration_s)
    info = ld.seg_info(feats.seg, n, p["seg_off"], bool(p["ham"]))
    count = np.minimum(ld.seg_count(info, n), p["max_overlap"])
    sm = count >= 1
    return ld._State(n=n, sm=sm, count=count, info=info, windows=ld._unit_windows(feats.windows),
                     free=ld.free_switch_frames(sm, int(p["pause_free"] / ld.GRID_S)), p=p)


V4_PARAMS = dict(min_spk_s=6.0, w075=0.0, marg_thr=0.0, split_ratio=0.0, seg_off=None, ham=1, map_mode="vote",
                 beta=0.0, ov_blend=0.0, ov_prune_pov=0.0)


class V5BlockTests(unittest.TestCase):
    SCHEDULE = PipelineTests.SCHEDULE

    def test_seg_info_frame_offset_and_uniform_weights(self) -> None:
        seg = {"data": np.zeros((2, NF, 3), np.float32), "chunk_step": 1.0, "frame_step": FSTEP}
        n = ld.grid_frames(11.0)
        v4 = ld.seg_info(seg, n)
        v5 = ld.seg_info(seg, n, 0.04, hamming=False)
        np.testing.assert_allclose(v4.window, np.hamming(NF), rtol=1e-6)
        self.assertTrue(bool((v5.window == 1.0).all()))
        # frame 3 of chunk 1: V4 at 1 + 3.5 * FSTEP = 1.059 s (grid 52), V5 at 1 + 3 * FSTEP + 0.04 = 1.091 s (54)
        self.assertEqual((int(v4.gis[1, 3]), int(v5.gis[1, 3])), (52, 54))

    def test_aggregate_frames_is_the_seg_count_overlap_add(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        n = ld.grid_frames(52.0)
        for hamming in (True, False):
            info = ld.seg_info(feats.seg, n, 0.04, hamming)
            mean = ld.aggregate_frames(info, info.data.sum(2), n)
            np.testing.assert_array_equal(np.rint(mean).astype(np.int32), ld.seg_count(info, n))
        # two chunks with different values: 1 where only chunk 0 covers, 0 where only chunk 1, a mix in between
        seg = {"data": np.zeros((2, NF, 3), np.float32), "chunk_step": 1.0, "frame_step": FSTEP}
        n = ld.grid_frames(11.0)
        info = ld.seg_info(seg, n, 0.04, hamming=False)
        mixed = ld.aggregate_frames(info, np.stack([np.ones(NF), np.zeros(NF)]), n)
        self.assertEqual(float(mixed[int(0.5 / ld.GRID_S)]), 1.0)
        self.assertEqual(float(mixed[int(10.5 / ld.GRID_S)]), 0.0)
        self.assertTrue(0.3 < float(mixed[int(5.0 / ld.GRID_S)]) < 0.7)
        self.assertEqual(len(ld.aggregate_frames(info, np.zeros((0, NF)), n)), n)

    def test_vectorized_reseg_equals_the_slice_loop(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)], noise=0.2, seed=3)
        s = _state(feats)
        C = ld.unit(np.stack([_voice(100), _voice(101), _voice(7)]))
        for final, scales in ((False, (("s1.5", 1.0), ("s3.0", 0.5))),
                              (True, (("s1.5", 1.0), ("s3.0", 0.5), ("s0.75", 8.0)))):
            path, score = ld._reseg(s, C, final=final)
            ref, has = _naive_reseg_score(s, C, scales)
            np.testing.assert_allclose(score[has], ref[has], atol=1e-5)
            self.assertGreater(float(has.mean()), 0.9)
        # without the final flag the 0.75 s windows never count
        no_short = dict(s.windows)
        no_short.pop("s0.75")
        s2 = _state(feats)
        s2.windows = no_short
        np.testing.assert_array_equal(ld._reseg(s2, C)[1], ld._reseg(s, C)[1])
        # extra is added to the scaled score: a large bonus forces its label
        bonus = np.zeros((s.n, 3), np.float32)
        bonus[:, 2] = 100.0
        path, _ = ld._reseg(s, C, extra=bonus)
        self.assertTrue(bool((path == 2).all()))

    def test_margins_flag_a_duplicate_centroid(self) -> None:
        a = _voice(5)
        b = ld.unit(a + 0.3 * _voice(6))  # cos(a, b) about 0.96
        feats = _synthetic([(0.0, 10.0, 0), (10.0, 20.0, 1)], 20.0, voices={0: a, 1: b}, noise=0.0)
        s = _state(feats)
        path = np.where(np.arange(s.n) < s.n // 2, 0, 1).astype(np.int32)
        marg, secs = ld._margins(s, np.stack([a, b]), path)
        self.assertTrue(bool((marg < ld.DEFAULT_PARAMS["marg_thr"]).all()), marg)
        self.assertAlmostEqual(float(secs.sum()), float(s.sm.sum() * ld.GRID_S), places=6)
        c = _voice(9)
        far = _synthetic([(0.0, 10.0, 0), (10.0, 20.0, 1)], 20.0, voices={0: a, 1: c}, noise=0.0)
        marg, _ = ld._margins(_state(far), np.stack([a, c]), path)
        self.assertTrue(bool((marg > 0.5).all()), marg)

    def test_split_two_separates_two_groups_and_keeps_one(self) -> None:
        rng = np.random.default_rng(1)
        a, b = _voice(11), _voice(12)
        two = ld.unit(np.concatenate([a + 0.05 * rng.standard_normal((20, 192)),
                                      b + 0.05 * rng.standard_normal((14, 192))]).astype(np.float32))
        ratio, minor, lab = ld._split_two(two)
        self.assertLess(ratio, 0.5)
        self.assertEqual(minor, 14)
        self.assertEqual(len(set(lab[:20].tolist())), 1)
        one = ld.unit((a + 0.05 * rng.standard_normal((30, 192))).astype(np.float32))
        split = ld._split_two(one)
        self.assertTrue(split is None or split[0] > ld.DEFAULT_PARAMS["split_ratio"])
        self.assertIsNone(ld._split_two(two[:5]))

    def test_split_test_finds_two_similar_voices_that_ahc_merges(self) -> None:
        a = _voice(21)
        orth = _voice(22) - float(_voice(22) @ a) * a
        b = ld.unit(0.65 * a + 0.76 * orth / np.linalg.norm(orth))  # cos(a, b) = 0.65
        feats = _synthetic(self.SCHEDULE, 52.0, voices={0: a, 1: b}, noise=0.02)
        self.assertEqual(ld.diarize_features(feats, params=V4_PARAMS).active.shape[1], 1)
        res = ld.diarize_features(feats)
        self.assertEqual(res.active.shape[1], 2)
        self.assertEqual(res.debug["nsplit"], 1)
        lab_a, lab_b = res.primary[int(5 / ld.GRID_S)], res.primary[int(18 / ld.GRID_S)]
        self.assertNotEqual(lab_a, lab_b)
        self.assertEqual(res.primary[int(30 / ld.GRID_S)], lab_a)
        self.assertEqual(res.primary[int(45 / ld.GRID_S)], lab_b)
        # a speaker hint caps the split
        self.assertEqual(ld.diarize_features(feats, max_speakers=1).active.shape[1], 1)

    def test_soft_overlap_probability_prunes_the_second_speaker(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        t = int(21 / ld.GRID_S)
        seg = dict(feats.seg)
        for value, speakers in ((0.9, 2), (0.1, 1)):
            seg["overlap_prob"] = np.full(seg["data"].shape[:2], value, np.float32)
            res = ld.diarize_features(ld.Features(seg, feats.windows, feats.duration_s))
            self.assertTrue(res.debug["ov_prune"])
            self.assertEqual(int(res.active[t].sum()), speakers, value)
        # without the probability (precomputed seg arrays of an old caller) nothing is pruned, and the debug says so
        res = ld.diarize_features(feats)
        self.assertFalse(res.debug["ov_prune"])
        self.assertEqual(int(res.active[t].sum()), 2)

    def test_final_pass_runs_only_with_the_short_windows(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0)
        self.assertTrue(ld.diarize_features(feats).debug["final_pass"])
        self.assertFalse(ld.diarize_features(feats, params={"w075": 0}).debug["final_pass"])
        two = {k: v for k, v in feats.windows.items() if k != "s0.75"}
        res = ld.diarize_features(ld.Features(feats.seg, two, feats.duration_s))
        self.assertFalse(res.debug["final_pass"])
        self.assertEqual(res.active.shape[1], 2)

    def test_v4_parameters_still_run(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        res = ld.diarize_features(feats, params=V4_PARAMS)
        self.assertEqual(res.active.shape[1], 2)
        self.assertFalse(res.debug["final_pass"])
        self.assertEqual(int(res.active[int(21 / ld.GRID_S)].sum()), 2)

    def test_speech_frames_use_the_pipeline_aggregation(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0)
        n = ld.grid_frames(52.0)
        expected = ld.seg_count(ld.seg_info(feats.seg, n, 0.04, hamming=False), n) >= 1
        np.testing.assert_array_equal(ld.speech_frames_from_seg(feats.seg, 52.0), expected)
        v4 = ld.speech_frames_from_seg(feats.seg, 52.0, {"seg_off": None, "ham": 1})
        np.testing.assert_array_equal(v4, ld.seg_count(ld.seg_info(feats.seg, n), n) >= 1)


if __name__ == "__main__":
    unittest.main()
