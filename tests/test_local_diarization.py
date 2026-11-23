import unittest

import numpy as np

from backend import genesis_whisper_server_local_diarization as ld
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
        payload = ld.result_to_diarization_payload(res, 52000)
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

    def test_payload_is_a_valid_diarization_payload(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        payload = ld.result_to_diarization_payload(ld.diarize_features(feats), 52000, num_speakers=None)
        self.assertIs(ld.validate_diarization_payload(payload), payload)
        self.assertEqual(payload["counts"]["speakers"], 2)
        self.assertEqual(payload["exclusive_diarization"][0]["speaker_id"], "SPEAKER_00")
        for seg in payload["exclusive_diarization"]:
            self.assertEqual(seg["start_ms"] % 20, 0)
        # exclusive turns never overlap
        ex = payload["exclusive_diarization"]
        for a, b in zip(ex, ex[1:]):
            self.assertLessEqual(a["end_ms"], b["start_ms"])

    def test_validator_rejects_a_broken_payload(self) -> None:
        # Counter-check for the payload tests: the validator is not a pass-through.
        feats = _synthetic(self.SCHEDULE, 52.0)
        good = ld.result_to_diarization_payload(ld.diarize_features(feats), 52000)
        broken = [
            {**good, "schema_version": "1.0"},
            {**good, "overlaps": None},
            {**good, "exclusive_diarization": [{"start_ms": 500, "end_ms": 400, "speaker_id": "SPEAKER_00"}]},
            {**good, "overlaps": [{"start_ms": 0, "end_ms": 20, "speaker_ids": ["SPEAKER_00"]}]},
        ]
        for payload in broken:
            with self.subTest(payload=payload.get("schema_version")):
                with self.assertRaises(ValueError):
                    ld.validate_diarization_payload(payload)

    def test_silence_and_tiny_inputs(self) -> None:
        silent = ld.Features({"data": np.zeros((1, NF, 3), np.float32), "chunk_step": 1.0, "frame_step": FSTEP},
                             {k: (np.zeros(0), np.zeros(0), np.zeros((0, 192), np.float32)) for k in ("s3.0", "s1.5")},
                             4.0)
        res = ld.diarize_features(silent)
        payload = ld.result_to_diarization_payload(res, 4000)
        ld.validate_diarization_payload(payload)
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
        C = ld.unit(np.stack([_voice(100), _voice(101), _voice(7)]))
        for fast in (False, True):
            s = _state(feats, loc_w=0.0, fast_sums=fast)
            for final, scales in ((False, (("s1.5", 1.0), ("s3.0", 0.5))),
                                  (True, (("s1.5", 1.0), ("s3.0", 0.5), ("s0.75", 8.0)))):
                path, score = ld._reseg(s, C, final=final)
                ref, has = _naive_reseg_score(s, C, scales)
                np.testing.assert_allclose(score[has], ref[has], atol=1e-5)
                self.assertGreater(float(has.mean()), 0.9)
                # counter-check: the reference sees a different weighting
                wrong, _ = _naive_reseg_score(s, C, tuple((k, 1.0) for k, _ in scales))
                self.assertGreater(float(np.abs(score[has] - wrong[has]).max()), 1e-3)
        s = _state(feats)
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


def _viterbi_loop_free_frames(sm, pause_frames):
    """The V5 loop of ``free_switch_frames`` (reference for the vectorised version)."""

    free = np.zeros(len(sm), bool)
    run = 0
    for t in range(len(sm)):
        if not sm[t]:
            run += 1
        else:
            if run >= pause_frames:
                free[t] = True
            run = 0
    return free


class V5FastTests(unittest.TestCase):
    """V5-fast: the exact speed-ups give the V5 result, the new terms only act where they should."""

    SCHEDULE = PipelineTests.SCHEDULE

    def test_numba_viterbi_is_bit_identical_to_the_numpy_loop(self) -> None:
        rng = np.random.default_rng(5)
        for T, K in ((2, 1), (50, 2), (3000, 4), (800, 7)):
            score = rng.standard_normal((T, K)).astype(np.float32)
            score[::7] = np.round(score[::7])  # ties
            free = rng.random(T) < 0.05
            for pen in (0.0, 1.5, 30.0):
                np.testing.assert_array_equal(ld.viterbi_nb(score, pen, free), ld.viterbi(score, pen, free))
                np.testing.assert_array_equal(ld.viterbi_nb(score, pen), ld.viterbi(score, pen))
        # counter-check: the penalty matters on this input, so equal paths are not trivially equal
        score = rng.standard_normal((400, 3)).astype(np.float32)
        self.assertFalse(np.array_equal(ld.viterbi_nb(score, 0.0), ld.viterbi_nb(score, 5.0)))
        # float64 scores take the numpy loop
        np.testing.assert_array_equal(ld.viterbi_nb(score.astype(np.float64), 1.0), ld.viterbi(score, 1.0))

    def test_free_switch_frames_equals_the_loop(self) -> None:
        rng = np.random.default_rng(1)
        for sm in (rng.random(5000) < 0.7, np.zeros(10, bool), np.ones(10, bool), np.zeros(0, bool)):
            for pause in (0, 1, 25):
                np.testing.assert_array_equal(ld.free_switch_frames(sm, pause), _viterbi_loop_free_frames(sm, pause))

    def test_fast_audit_and_activity_equal_the_loops(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)], noise=0.2, seed=3)
        s = _state(feats)
        C = ld.unit(np.stack([_voice(100), _voice(101), _voice(7)]))
        path, score = ld._reseg(s, C)
        K = len(C)
        fast = ld._audit_fast(ld._local_vote_matrix(s, path, K), K, 0.5)
        loop = ld._audit(ld._local_votes(s, path, K), K, 0.5)
        for a, b in zip(fast, loop):
            np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(ld._overlap_activity_hung_fast(s, path, K, score),
                                   ld._overlap_activity_hung(s, path, K, score), atol=1e-5)
        # counter-check: another path gives other purities
        other = ld._audit_fast(ld._local_vote_matrix(s, (path + 1) % K, K), K, 0.5)
        self.assertFalse(np.allclose(other[0], fast[0]))

    def test_exact_speedups_alone_give_the_v5_result(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)], noise=0.2, seed=3)
        v5 = ld.diarize_features(feats, params=ld.V5_PARAMS)
        fast = ld.diarize_features(feats, params=dict(ld.V5_PARAMS, numba_viterbi=True, fast_sums=True))
        np.testing.assert_array_equal(fast.active, v5.active)
        np.testing.assert_array_equal(fast.primary, v5.primary)

    def _counting(self, feats, transform=None):
        calls = []
        lookup = {round(float(t), 3): v for t, v in zip(feats.windows["s0.75"][0], feats.windows["s0.75"][2])}

        def embed075(starts_s):
            calls.append(np.asarray(starts_s).copy())
            X = np.stack([lookup[round(float(t), 3)] for t in starts_s]) if len(starts_s) else np.zeros((0, 192))
            return transform(X) if transform else X

        return calls, embed075

    def test_short_windows_are_embedded_only_at_unsure_frames(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)], noise=0.25, seed=4)
        st75 = feats.windows["s0.75"][0]
        placeholders = dict(feats.windows, **{"s0.75": (st75, st75 + 0.75, np.zeros((len(st75), 192), np.float32))})
        calls, embed075 = self._counting(feats)
        lazy = ld.diarize_features(ld.Features(feats.seg, placeholders, feats.duration_s, embed075=embed075))
        self.assertEqual(len(calls), 1)
        asked = calls[0]
        self.assertTrue(0 < len(asked) < len(st75), (len(asked), len(st75)))
        self.assertEqual(lazy.debug["n075"], len(asked))
        self.assertTrue(set(np.round(asked, 3)) <= set(np.round(st75, 3)))
        # the overlap (20-22 s) is unsure: every window whose core lies inside it is asked for
        core_mid = st75 + 0.375
        inside = st75[(core_mid > 20.3) & (core_mid < 21.7)]
        self.assertTrue(set(np.round(inside, 3)) <= set(np.round(asked, 3)))
        # with the vectors already there (no callback) the same windows give the same result
        eager = ld.diarize_features(feats)
        np.testing.assert_array_equal(lazy.active, eager.active)
        self.assertEqual(eager.debug["n075"], len(asked))
        # counter-check: the asked vectors are really used (flipped vectors change the final pass)
        _, flipped = self._counting(feats, transform=lambda X: -X)
        bad = ld.diarize_features(ld.Features(feats.seg, placeholders, feats.duration_s, embed075=flipped))
        self.assertFalse(np.array_equal(bad.active, eager.active))
        # V5: every 0.75 s window, asked for once, same result as the stored vectors
        calls, embed075 = self._counting(feats)
        v5_lazy = ld.diarize_features(ld.Features(feats.seg, placeholders, feats.duration_s, embed075=embed075),
                                      params=ld.V5_PARAMS)
        self.assertEqual([len(c) for c in calls], [len(st75)])
        np.testing.assert_array_equal(v5_lazy.active, ld.diarize_features(feats, params=ld.V5_PARAMS).active)
        # no final pass: nothing is asked
        calls, embed075 = self._counting(feats)
        ld.diarize_features(ld.Features(feats.seg, placeholders, feats.duration_s, embed075=embed075),
                            params={"w075": 0})
        self.assertEqual(calls, [])

    def test_unsure_frames_follow_margin_overlap_and_changes(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)])
        st75, en75, _ = feats.windows["s0.75"]
        C = ld.unit(np.stack([_voice(100), _voice(101)]))
        s = _state(feats)
        path, score = ld._reseg(s, C)
        base = ld._unsure_075(s, path, score, st75, en75)
        none = ld._unsure_075(_state(feats, sel075_margin=0.0, sel075_overlap=False), path, score, st75, en75)
        every = ld._unsure_075(_state(feats, sel075_margin=10.0), path, score, st75, en75)
        wide = ld._unsure_075(_state(feats, sel075_dilate_s=1.0), path, score, st75, en75)
        self.assertTrue(bool(every.all()))
        self.assertTrue(bool((none <= base).all()) and bool((base <= wide).all()))
        self.assertGreater(int(wide.sum()), int(base.sum()))
        # without margin and overlap only label changes count; here they fall into the pauses, where no window sits,
        # and widened by 0.5 s they reach exactly the windows next to the changes
        changes = (np.flatnonzero(path[1:] != path[:-1]) + 1) * ld.GRID_S
        self.assertEqual(len(changes), 3)
        self.assertEqual(int(none.sum()), 0)
        near = ld._unsure_075(_state(feats, sel075_margin=0.0, sel075_overlap=False, sel075_dilate_s=0.5),
                              path, score, st75, en75)
        self.assertTrue(0 < int(near.sum()) < len(st75))
        mids = (st75 + en75) / 2
        dist = np.abs(mids[:, None] - changes[None, :]).min(1)
        self.assertTrue(bool((dist[near] <= 0.5 + 0.2 + ld.GRID_S).all()))
        self.assertTrue(bool((dist[~near] > 0.5).all()))

    def test_local_speaker_vectors_are_means_of_windows_inside_one_local_speaker(self) -> None:
        feats = _synthetic(self.SCHEDULE, 52.0, noise=0.05, seed=2)
        s = _state(feats)
        L = ld._local_speaker_vectors(s)
        self.assertIsNotNone(L)
        np.testing.assert_allclose(np.linalg.norm(L["E"], axis=1), 1.0, atol=1e-5)
        self.assertEqual(len(L["U"]), len(L["G"]))
        # each unit's vector is close to the voice of the speaker who talks on its frames
        voices = {k: _voice(100 + k) for k in (0, 1)}
        t = L["G"] * ld.GRID_S
        for u in range(len(L["E"])):
            mid = float(np.median(t[L["U"] == u]))
            who = [k for a, b, k in self.SCHEDULE if a <= mid < b][0]
            self.assertGreater(float(L["E"][u] @ voices[who]), float(L["E"][u] @ voices[1 - who]) + 0.5)
        # the term only enters the final pass, and only with loc_w > 0
        C = ld.unit(np.stack([_voice(100), _voice(101)]))
        for final in (False, True):
            with_loc = ld._reseg(_state(feats), C, final=final)[1]
            without = ld._reseg(_state(feats, loc_w=0.0), C, final=final)[1]
            if final:
                self.assertFalse(np.allclose(with_loc, without))
            else:
                np.testing.assert_array_equal(with_loc, without)
        # counter-check: a speaker below loc_min_f sole frames gets no vector
        self.assertIsNone(ld._local_speaker_vectors(_state(feats, loc_min_f=10 ** 6)))

    def test_v5_params_turn_every_fast_path_off(self) -> None:
        p = dict(ld.DEFAULT_PARAMS, **ld.V5_PARAMS)
        self.assertEqual((p["numba_viterbi"], p["fast_sums"], p["loc_w"], p["sel075"]), (False, False, 0.0, False))
        d = ld.DEFAULT_PARAMS
        self.assertEqual((d["numba_viterbi"], d["fast_sums"], d["loc_w"], d["loc_scales"], d["sel075"],
                          d["sel075_margin"], d["sel075_overlap"], d["sel075_dilate_s"]),
                         (True, True, 0.2, "s1.5+s0.75", True, 0.3, True, 0.0))

    def test_diarize_audio_embeds_the_short_windows_on_demand(self) -> None:
        sr = ld.SAMPLE_RATE
        feats = _synthetic(self.SCHEDULE, 52.0, overlap=[(20.0, 22.0, 0)], noise=0.2, seed=3)
        audio = np.zeros(52 * sr, np.float32)
        speech = np.ones(len(audio) // 512, bool)
        asked = {}

        def embed(_audio, starts, length_s):
            asked.setdefault(length_s, []).append(len(starts))
            rng = np.random.default_rng(len(starts))
            X = rng.standard_normal((len(starts), 192)).astype(np.float32)
            return X / np.linalg.norm(X, axis=1, keepdims=True)

        ld.diarize_audio(audio, segmentation=feats.seg, speech_probs=speech.astype(np.float32), embed=embed)
        planned = ld.plan_scale_starts(len(audio), speech, 512)
        self.assertEqual(asked[3.0], [len(planned["s3.0"])])
        self.assertEqual(asked[1.5], [len(planned["s1.5"])])
        self.assertEqual(len(asked[0.75]), 1)
        self.assertLess(asked[0.75][0], len(planned["s0.75"]))

if __name__ == "__main__":
    unittest.main()
