"""DIA-free batch diarization: seg-3.0 speech/overlap + ReDimNet2 windows + AHC + Viterbi resegmentation.

Production version of the "V4" research pipeline (bench: DER on synthetic DE/EN meetings and AMI equal to or better
than pyannote community-1 through the G3_DIA server, every speaker count correct on the held-out AMI meetings):

1. seg-3.0 (``genesis_whisper_server_segmentation``) gives per 20 ms frame the number of active speakers (0/1/2,
   pyannote-style Hamming overlap-add over the 10 s chunks). ``count >= 1`` is the speech mask.
2. ReDimNet2-B6 embeds 3 s windows (hop 0.5 s) and 1.5 s windows (hop 0.25 s) that hold enough speech.
3. Clean 3 s windows (>= 70 % speech, <= 10 % overlap) are clustered with average-linkage AHC (cosine 0.55);
   clusters below 6 s of support are dissolved into the nearest big one.
4. Viterbi resegmentation over the frame grid with the window/centroid cosines (both scales, switch penalty, free
   switches after pauses), two centroid refinements from the 1.5 s windows that sit inside one label.
5. Local-speaker audit: every seg-3.0 (chunk, local speaker) group votes with the path. A cluster that does not own
   its groups (local purity < 0.55: laughter, mixtures, noise) is dropped and the path recomputed (max 8 rounds).
6. Overlap: frames with count 2 get the second speaker from the seg-3.0 local speaker mapped to a cluster, else the
   runner-up of the resegmentation score.

The output has the shape of the G3_DIA ``/v2/diarize`` answer (``diarization``, ``exclusive_diarization``,
``overlaps`` in integer milliseconds), so ``_process_diarization`` and the WXC pipeline work unchanged.
CPU part is numpy/scipy only; the GPU parts (segmentation, embeddings) are passed in or use the shared models.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


SAMPLE_RATE = 16000
GRID_S = 0.02  # output frame grid (s)
GRID_MS = 20
LOCAL_DIARIZATION_MODEL_ID = "g3-whisper/seg3-redimnet2-v4"
# Window scales: name -> (length s, hop s). The names are the research feature names.
BATCH_WINDOW_SCALES: Dict[str, tuple[float, float]] = {"s3.0": (3.0, 0.5), "s1.5": (1.5, 0.25)}
WINDOW_MIN_SPEECH = 0.3  # a window is embedded only with this share of speech frames

# Frozen V4 configuration (bench: batch/NOTES.md). Plateau-tested; change only with a new benchmark.
DEFAULT_PARAMS: Dict[str, Any] = dict(
    ahc_thr=0.55,        # AHC cosine distance threshold
    min_spk_s=6.0,       # clusters with less clean-window support are dissolved
    iters=2,             # centroid refinements after the first resegmentation
    iters_after=1,       # refinements after every audit drop
    w3=0.5,              # weight of the 3 s windows in the frame score (1.5 s windows weigh 1)
    core=0.5,            # a window scores only its central share
    temp=0.1, penalty=3.0, pause_free=0.5,
    max_overlap=2,
    map_min_s=0.3, map_purity=0.6, ov_min_act=0.05,
    min_cov=0.7, clean_max_ov=0.1,
    audit_rounds=8, dom_share=0.5, junk_pur=0.55,
    ov_fill="score",
)


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def grid_frames(duration_s: float) -> int:
    return int(duration_s / GRID_S) + 1


# ---------------------------------------------------------------------------
# seg-3.0 aggregation
# ---------------------------------------------------------------------------

@dataclass
class SegInfo:
    data: np.ndarray       # (chunks, frames, 3) 0/1
    gis: np.ndarray        # (chunks, frames) grid frame of every chunk frame
    frame_step: float
    window: np.ndarray     # Hamming window over the chunk frames


def seg_info(seg: Mapping[str, Any], n: int) -> SegInfo:
    data = np.asarray(seg["data"]).astype(np.float32)
    nc, nf, _ = data.shape
    cstep, fstep = float(seg["chunk_step"]), float(seg["frame_step"])
    t = np.arange(nc)[:, None] * cstep + (np.arange(nf) * fstep)[None, :] + fstep / 2
    gis = np.clip((t / GRID_S).astype(int), 0, n - 1)
    return SegInfo(data, gis, fstep, np.hamming(nf).astype(np.float32))


def seg_count(info: SegInfo, n: int) -> np.ndarray:
    """Aggregated local-speaker count per grid frame (pyannote-style Hamming overlap-add, rounded)."""

    nc = len(info.data)
    cover = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    if nc:
        win = np.broadcast_to(info.window, info.gis.shape)
        np.add.at(cover, info.gis.ravel(), win.ravel())
        np.add.at(cnt, info.gis.ravel(), (info.window[None, :] * info.data.sum(2)).ravel())
    count = np.zeros(n, np.int32)
    ok = cover > 0
    count[ok] = np.rint(cnt[ok] / cover[ok]).astype(np.int32)
    return count


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def ahc(X: np.ndarray, thr: float) -> np.ndarray:
    if len(X) == 1:
        return np.zeros(1, int)
    Z = linkage(X.astype(np.float64), method="average", metric="cosine")
    return fcluster(Z, t=thr, criterion="distance") - 1


def ahc_k(X: np.ndarray, k: int) -> np.ndarray:
    if len(X) <= k:
        return np.arange(len(X))
    Z = linkage(X.astype(np.float64), method="average", metric="cosine")
    return fcluster(Z, t=k, criterion="maxclust") - 1


def merge_small(labels: np.ndarray, X: np.ndarray, weights: np.ndarray, min_weight: float,
                keep_at_least: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dissolve clusters below ``min_weight`` into the nearest big one (the ``keep_at_least`` heaviest always stay)."""

    labs = np.unique(labels)
    w = {l: weights[labels == l].sum() for l in labs}
    big = [l for l in labs if w[l] >= min_weight] or [max(w, key=w.get)]
    if len(big) < keep_at_least:
        ranked = sorted(labs, key=lambda l: -w[l])[:keep_at_least]
        big = [l for l in labs if l in ranked]
    C = unit(np.stack([X[labels == l].mean(0) for l in big]))
    out = labels.copy()
    for i in range(len(labels)):
        if labels[i] not in big:
            out[i] = big[int(np.argmax(C @ X[i]))]
    remap = {l: k for k, l in enumerate(big)}
    return np.array([remap[l] for l in out]), C


def viterbi(score: np.ndarray, penalty: float, free: Optional[np.ndarray] = None) -> np.ndarray:
    """Best label path; ``score`` T x K (higher is better), ``penalty`` per switch, none into a ``free`` frame."""

    T, K = score.shape
    dp = score[0].copy()
    bp = np.zeros((T, K), np.int32)
    arange = np.arange(K)
    for t in range(1, T):
        pen = 0.0 if (free is not None and free[t]) else penalty
        best = dp.argmax()
        sw = dp[best] - pen
        choose_sw = sw > dp
        bp[t] = np.where(choose_sw, best, arange)
        dp = np.where(choose_sw, sw, dp) + score[t]
    path = np.zeros(T, np.int32)
    path[-1] = dp.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def free_switch_frames(sm: np.ndarray, pause_frames: int) -> np.ndarray:
    """True on a speech frame that follows at least ``pause_frames`` non-speech frames."""

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


def window_frames(start: np.ndarray, end: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.clip((start / GRID_S).astype(int), 0, n), np.clip((end / GRID_S).astype(int), 0, n)


def _run_mean(values: np.ndarray, a: np.ndarray, b: np.ndarray, empty: float) -> np.ndarray:
    """Mean of ``values[a:b]`` per pair (exactly the float64 mean of a 0/1 slice), ``empty`` when b <= a."""

    cs = np.concatenate([[0], np.cumsum(values.astype(np.int64))])
    length = b - a
    out = np.full(len(a), empty, np.float64)
    ok = length > 0
    out[ok] = (cs[b[ok]] - cs[a[ok]]) / length[ok]
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

Windows = tuple  # (start s, end s, vectors N x 192)


@dataclass
class Features:
    """Everything the CPU stage needs: seg-3.0 arrays, window embeddings per scale, the signal length."""

    seg: Mapping[str, Any]
    windows: Dict[str, Windows]
    duration_s: float


def plan_window_starts(num_samples: int, length_s: float, hop_s: float, speech: np.ndarray, frame_samples: int,
                       min_speech: float = WINDOW_MIN_SPEECH) -> np.ndarray:
    """Start samples of the windows worth embedding: a regular grid, kept when >= ``min_speech`` of the speech
    frames it covers (``speech``: one bool per ``frame_samples``) are speech. None when the signal is shorter."""

    n = int(round(length_s * SAMPLE_RATE))
    if num_samples < n:
        return np.zeros(0, np.int64)
    starts = np.arange(0, max(1, num_samples - n + 1), int(round(hop_s * SAMPLE_RATE)), dtype=np.int64)
    L = len(speech)
    f0 = np.minimum(starts // frame_samples, L)
    f1 = np.minimum((starts + n) // frame_samples, L)
    frac = _run_mean(np.asarray(speech, bool), f0, f1, empty=-1.0)
    return starts[frac >= min_speech]


def _voice_windows(audio: np.ndarray, starts: np.ndarray, n: int) -> Iterator[Any]:
    from .genesis_whisper_server_vid import VoiceWindow

    for s in starts:
        s = int(s)
        yield VoiceWindow(audio=audio[s: s + n], start_ms=round(s * 1000 / SAMPLE_RATE),
                          end_ms=round((s + n) * 1000 / SAMPLE_RATE), clean_duration_seconds=n / SAMPLE_RATE)


def embed_window_starts(audio: np.ndarray, starts: np.ndarray, length_s: float) -> np.ndarray:
    """ReDimNet2 vectors (N x 192, unit) of the windows at ``starts``, embedded at their native length."""

    from .genesis_whisper_server_vid import embed_voice_windows

    n = int(round(length_s * SAMPLE_RATE))
    if len(starts) == 0:
        return np.zeros((0, 192), np.float32)
    embedded = embed_voice_windows(_voice_windows(np.asarray(audio, np.float32), starts, n), window_samples=n)
    return np.stack([item.vector for item in embedded]).astype(np.float32)


EmbedFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def plan_scale_starts(num_samples: int, speech: np.ndarray, frame_samples: int,
                      scales: Mapping[str, tuple[float, float]] = BATCH_WINDOW_SCALES) -> Dict[str, np.ndarray]:
    """Start samples of the windows worth embedding, per scale (CPU part of ``compute_window_features``)."""

    return {name: plan_window_starts(num_samples, length_s, hop_s, speech, frame_samples)
            for name, (length_s, hop_s) in scales.items()}


def assemble_windows(starts: Mapping[str, np.ndarray], vectors: Mapping[str, np.ndarray],
                     scales: Mapping[str, tuple[float, float]] = BATCH_WINDOW_SCALES) -> Dict[str, Windows]:
    """(start s, end s, vectors) per scale from the planned starts and their embeddings (same order)."""

    out: Dict[str, Windows] = {}
    for name, (length_s, _) in scales.items():
        st = np.asarray(starts[name]) / SAMPLE_RATE
        X = np.asarray(vectors.get(name, np.zeros((0, 192), np.float32)), np.float32)
        if len(X) == 0:
            X = np.zeros((0, 192), np.float32)
        if len(X) != len(st):
            raise ValueError(f"{name}: {len(X)} Vektoren fuer {len(st)} Fenster.")
        out[name] = (st, st + length_s, X)
    return out


def compute_window_features(audio: np.ndarray, speech: np.ndarray, frame_samples: int,
                            scales: Mapping[str, tuple[float, float]] = BATCH_WINDOW_SCALES,
                            embed: Optional[EmbedFn] = None) -> Dict[str, Windows]:
    """Plan and embed the windows of every scale; ``embed(audio, starts, length_s)`` defaults to ReDimNet2."""

    embed = embed or embed_window_starts
    audio = np.asarray(audio, np.float32).reshape(-1)
    starts = plan_scale_starts(len(audio), speech, frame_samples, scales)
    vectors = {name: (embed(audio, starts[name], scales[name][0]) if len(starts[name])
                      else np.zeros((0, 192), np.float32)) for name in scales}
    return assemble_windows(starts, vectors, scales)


# ---------------------------------------------------------------------------
# The V4 pipeline on features
# ---------------------------------------------------------------------------

@dataclass
class _State:
    n: int
    sm: np.ndarray
    count: np.ndarray
    info: SegInfo
    windows: Dict[str, Windows]
    free: np.ndarray
    p: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


def _unit_windows(windows: Mapping[str, Windows]) -> Dict[str, Windows]:
    out = {}
    for name, (st, en, X) in windows.items():
        X = np.asarray(X, np.float32).reshape(-1, 192) if len(X) else np.zeros((0, 192), np.float32)
        out[name] = (np.asarray(st, np.float64), np.asarray(en, np.float64), unit(X))
    return out


def _reseg(s: _State, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, p = s.n, s.p
    K = len(C)
    score = np.zeros((n, K), np.float32)
    cnt = np.zeros(n, np.float32)
    for scale, wgt in (("s1.5", 1.0), ("s3.0", p["w3"])):
        st, en, X = s.windows[scale]
        if len(X) == 0:
            continue
        S = X @ C.T
        fa, fb = window_frames(st, en, n)
        mid = (fa + fb) // 2
        half = np.maximum(1, ((fb - fa) * p["core"]).astype(int) // 2)
        lo = np.maximum(0, mid - half)
        hi = np.minimum(n, mid + half)
        for i in range(len(X)):
            score[lo[i]:hi[i]] += wgt * S[i]
            cnt[lo[i]:hi[i]] += wgt
    has = cnt > 0
    score[has] /= cnt[has, None]
    if not has.all():
        idx = np.where(has)[0]
        if len(idx):
            near = idx[np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)]
            score[~has] = score[near[~has]]
    return viterbi(score / p["temp"], p["penalty"], s.free), score


def _refine(s: _State, C: np.ndarray, path: np.ndarray) -> np.ndarray:
    st, en, X = s.windows["s1.5"]
    if len(X) == 0:
        return C
    fa, fb = window_frames(st, en, s.n)
    speech = _run_mean(s.sm, fa, fb, empty=0.0)
    valid = fb > fa
    newC = []
    for kk in range(len(C)):
        own = _run_mean(path == kk, fa, fb, empty=0.0)
        sel = valid & (own > 0.9) & (speech > 0.6)
        newC.append(unit(X[sel].mean(0)) if int(sel.sum()) >= 3 else C[kk])
    return np.stack(newC)


def _local_votes(s: _State, path: np.ndarray, K: int) -> list[tuple[int, int, np.ndarray]]:
    seg, gis = s.info.data, s.info.gis
    nc, nf, nk = seg.shape
    out = []
    sole_all = seg.sum(2) == 1
    for c in range(nc):
        sole = sole_all[c]
        for k in range(nk):
            ck = (seg[c, :, k] > 0.5) & sole
            g = gis[c][ck]
            g = g[s.sm[g]]
            if len(g) == 0:
                continue
            out.append((c, k, np.bincount(path[g], minlength=K)))
    return out


def _audit(votes: Sequence[tuple[int, int, np.ndarray]], K: int, dom_share: float) -> tuple[np.ndarray, np.ndarray]:
    own = np.zeros(K)
    tot = np.zeros(K)
    for _, _, v in votes:
        total = v.sum()
        m = int(v.argmax())
        tot += v
        if v[m] >= dom_share * total:
            own[m] += v[m]
    return own / np.maximum(tot, 1), tot


def _overlap_activity(s: _State, path: np.ndarray, K: int) -> np.ndarray:
    """Per frame and cluster: seg-3.0 activity of the local speakers mapped to that cluster (Hamming-averaged)."""

    seg, gis, win = s.info.data, s.info.gis, s.info.window
    nc, nf, nk = seg.shape
    p = s.p
    act = np.zeros((s.n, K), np.float32)
    cover = np.zeros(s.n, np.float32)
    for c in range(nc):
        gi = gis[c]
        np.add.at(cover, gi, win)
        sole = seg[c].sum(1) == 1
        for k in range(nk):
            ck = (seg[c, :, k] > 0.5) & sole
            if ck.sum() * s.info.frame_step < p["map_min_s"]:
                continue
            votes = np.bincount(path[gi[ck]], minlength=K)
            g = int(votes.argmax())
            if votes[g] < p["map_purity"] * ck.sum():
                continue
            np.add.at(act[:, g], gi, win * seg[c, :, k])
    ok = cover > 0
    act[ok] /= cover[ok, None]
    return act


def _merge_closest(C: np.ndarray) -> np.ndarray:
    M = C @ C.T
    np.fill_diagonal(M, -2)
    i, j = np.unravel_index(int(np.argmax(M)), M.shape)
    merged = unit(C[i] + C[j])
    keep = [k for k in range(len(C)) if k not in (i, j)]
    return np.concatenate([C[keep], merged[None]], axis=0)


@dataclass
class DiarizationResult:
    """Frame-level result on the 20 ms grid: active[t, k] (incl. overlap), primary[t] (-1 = silence)."""

    active: np.ndarray
    primary: np.ndarray
    centroids: np.ndarray
    debug: Dict[str, Any]


def diarize_features(features: Features, *, num_speakers: Optional[int] = None, min_speakers: Optional[int] = None,
                     max_speakers: Optional[int] = None, params: Optional[Mapping[str, Any]] = None) -> DiarizationResult:
    """Run the V4 pipeline on precomputed features (CPU only)."""

    p = dict(DEFAULT_PARAMS, **(params or {}))
    if num_speakers is not None:
        min_speakers = max_speakers = int(num_speakers)
    min_k = max(1, int(min_speakers)) if min_speakers else 1
    max_k = int(max_speakers) if max_speakers else None
    n = grid_frames(features.duration_s)
    info = seg_info(features.seg, n)
    count = np.minimum(seg_count(info, n), p["max_overlap"])
    sm = count >= 1
    windows = _unit_windows(features.windows)
    s = _State(n=n, sm=sm, count=count, info=info, windows=windows,
               free=free_switch_frames(sm, int(p["pause_free"] / GRID_S)), p=p)
    empty = DiarizationResult(np.zeros((n, 0), bool), np.full(n, -1, np.int32), np.zeros((0, 192), np.float32), s.debug)
    if not sm.any():
        return empty

    # ---- clustering seeds: clean 3 s windows (1.5 s windows when there is no 3 s window at all)
    seed_scale = "s3.0" if len(windows["s3.0"][2]) else "s1.5"
    st, en, X3 = windows[seed_scale]
    if len(X3) == 0:
        # less than one window of speech: one speaker on every speech frame
        active = sm[:, None].copy()
        return DiarizationResult(active, np.where(sm, 0, -1).astype(np.int32), np.zeros((1, 192), np.float32), s.debug)
    a, b = window_frames(st, en, n)
    cov = _run_mean(sm, a, b, empty=0.0)
    ovf = _run_mean(count >= 2, a, b, empty=1.0)
    good = (cov >= p["min_cov"]) & (ovf <= p["clean_max_ov"])
    if good.sum() < 2:
        good = cov >= 0
    Xg = X3[good]
    lab = ahc(Xg, p["ahc_thr"])
    if max_k is not None and len(np.unique(lab)) > max_k:
        lab = ahc_k(Xg, max_k)
    lab, C = merge_small(lab, Xg, np.full(len(Xg), 0.5), p["min_spk_s"],
                         keep_at_least=min_k if min_speakers else 0)
    if len(C) < min_k and len(Xg) >= min_k:
        lab = ahc_k(Xg, min_k)
        C = unit(np.stack([Xg[lab == l].mean(0) for l in np.unique(lab)]))
    s.debug.update(K0=len(C), nwin=int(len(Xg)))

    for it in range(p["iters"] + 1):
        path, score = _reseg(s, C)
        if it < p["iters"]:
            C = _refine(s, C, path)

    # ---- local-speaker audit: dissolve clusters that do not own their seg-3.0 local speakers
    for _ in range(p["audit_rounds"]):
        K = len(C)
        if K < 2 or K <= min_k:
            break
        pur, tot = _audit(_local_votes(s, path, K), K, p["dom_share"])
        s.debug.setdefault("pur", []).append([round(float(x), 3) for x in pur])
        j = int(np.argmin(pur))
        if pur[j] >= p["junk_pur"]:
            break
        C = C[np.arange(K) != j]
        for it in range(p["iters_after"] + 1):
            path, score = _reseg(s, C)
            if it < p["iters_after"]:
                C = _refine(s, C, path)

    # ---- an explicit upper bound: merge the closest centroids
    while max_k is not None and len(C) > max(1, max_k):
        C = _merge_closest(C)
        for it in range(p["iters_after"] + 1):
            path, score = _reseg(s, C)
            if it < p["iters_after"]:
                C = _refine(s, C, path)

    K = len(C)
    s.debug["K"] = K
    act = _overlap_activity(s, path, K)
    active = np.zeros((n, K), bool)
    primary = np.full(n, -1, np.int32)
    speech_idx = np.where(count > 0)[0]
    primary[speech_idx] = path[speech_idx]
    active[speech_idx, path[speech_idx]] = True
    for t in np.where(count >= 2)[0]:
        prim = int(path[t])
        chosen = {prim}
        for g in np.argsort(-act[t]):
            if len(chosen) >= count[t] or act[t, g] <= p["ov_min_act"]:
                break
            chosen.add(int(g))
        if len(chosen) < count[t] and K > 1 and p["ov_fill"]:
            sc = score[t].copy()
            sc[prim] = -9
            chosen.add(int(sc.argmax()))
        active[t, list(chosen)] = True
    return DiarizationResult(active, primary, C.astype(np.float32), s.debug)


# ---------------------------------------------------------------------------
# DIA v2 payload
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    bounded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(bounded[1:] != bounded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def result_to_dia_payload(result: DiarizationResult, duration_ms: int, *, num_speakers: Optional[int] = None,
                          min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
                          started: Optional[float] = None) -> Dict[str, Any]:
    """The G3_DIA ``/v2/diarize`` answer shape (validated by ``dia_client._validate_dia_payload``)."""

    active = result.active
    n, K = active.shape
    # Public ids by first appearance of the speaker in the exclusive (primary) track.
    order: list[int] = []
    for k in result.primary[result.primary >= 0].tolist():
        if k not in order:
            order.append(k)
            if len(order) == K:
                break
    for k in range(K):
        if k not in order and active[:, k].any():
            order.append(k)
    names = {k: f"SPEAKER_{i:02d}" for i, k in enumerate(order)}

    def ms(frame: int) -> int:
        return min(duration_ms, frame * GRID_MS)

    standard, exclusive = [], []
    for k in order:
        for a, b in _runs(active[:, k]):
            if ms(b) > ms(a):
                standard.append({"start_ms": ms(a), "end_ms": ms(b), "speaker_id": names[k]})
        for a, b in _runs(result.primary == k):
            if ms(b) > ms(a):
                exclusive.append({"start_ms": ms(a), "end_ms": ms(b), "speaker_id": names[k]})
    standard.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["speaker_id"]))
    exclusive.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["speaker_id"]))

    overlaps: list[Dict[str, Any]] = []
    multi = active.sum(1) >= 2 if K else np.zeros(n, bool)
    for a, b in _runs(multi):
        t = a
        while t < b:
            ids = sorted(names[k] for k in np.flatnonzero(active[t]))
            u = t + 1
            while u < b and np.array_equal(active[u], active[t]):
                u += 1
            if ms(u) > ms(t):
                if overlaps and overlaps[-1]["end_ms"] == ms(t) and overlaps[-1]["speaker_ids"] == ids:
                    overlaps[-1]["end_ms"] = ms(u)
                else:
                    overlaps.append({"start_ms": ms(t), "end_ms": ms(u), "speaker_ids": ids})
            t = u

    total_ms = round((time.monotonic() - started) * 1000) if started is not None else 0
    return {
        "schema_version": "2.0",
        "request_id": uuid.uuid4().hex[:10],
        "status": "completed",
        "model": {"id": LOCAL_DIARIZATION_MODEL_ID},
        "input": {"duration_ms": int(duration_ms), "num_speakers": num_speakers, "min_speakers": min_speakers,
                  "max_speakers": max_speakers},
        "counts": {"speakers": len(order), "diarization_segments": len(standard),
                   "exclusive_segments": len(exclusive), "overlaps": len(overlaps)},
        "diarization": standard,
        "exclusive_diarization": exclusive,
        "overlaps": overlaps,
        "total_duration_ms": max(0, int(total_ms)),
    }


def speech_frames_from_seg(seg: Mapping[str, Any], duration_s: float) -> np.ndarray:
    """The seg-3.0 speech mask on the 20 ms grid (count >= 1), usable as the window speech filter."""

    n = grid_frames(duration_s)
    return seg_count(seg_info(seg, n), n) >= 1


def diarize_audio(audio: np.ndarray, *, num_speakers: Optional[int] = None, min_speakers: Optional[int] = None,
                  max_speakers: Optional[int] = None, segmentation: Optional[Mapping[str, Any]] = None,
                  speech_probs: Optional[np.ndarray] = None, embed: Optional[EmbedFn] = None,
                  params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Whole pipeline on a 16 kHz mono signal; returns the DIA v2 payload.

    ``segmentation``: precomputed seg-3.0 arrays (else the shared model runs). ``speech_probs``: Silero probabilities
    per 512 samples for the window speech filter (as in the bench); without them the seg-3.0 speech mask is used.
    """

    started = time.monotonic()
    samples = np.asarray(audio, np.float32).reshape(-1)
    duration_s = len(samples) / SAMPLE_RATE
    if segmentation is None:
        from .genesis_whisper_server_segmentation import segment_audio

        segmentation = segment_audio(samples)
    if speech_probs is not None:
        speech, frame_samples = np.asarray(speech_probs) > 0.5, 512
    else:
        speech, frame_samples = speech_frames_from_seg(segmentation, duration_s), int(GRID_S * SAMPLE_RATE)
    windows = compute_window_features(samples, speech, frame_samples, embed=embed)
    result = diarize_features(Features(segmentation, windows, duration_s), num_speakers=num_speakers,
                              min_speakers=min_speakers, max_speakers=max_speakers, params=params)
    return result_to_dia_payload(result, round(duration_s * 1000), num_speakers=num_speakers,
                                 min_speakers=min_speakers, max_speakers=max_speakers, started=started)


__all__ = [
    "BATCH_WINDOW_SCALES",
    "DEFAULT_PARAMS",
    "WINDOW_MIN_SPEECH",
    "assemble_windows",
    "DiarizationResult",
    "Features",
    "LOCAL_DIARIZATION_MODEL_ID",
    "compute_window_features",
    "diarize_audio",
    "diarize_features",
    "embed_window_starts",
    "plan_scale_starts",
    "plan_window_starts",
    "result_to_dia_payload",
    "seg_count",
    "seg_info",
    "speech_frames_from_seg",
    "viterbi",
]
