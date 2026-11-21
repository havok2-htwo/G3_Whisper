"""Stateless ``speaker_turns`` for one live section (Genesis3 docs/243 section 8.8).

The client (Genesis3 live text) keeps the running speaker centroids of a recording and sends them with every
section as ``known_speakers`` ({id, vector[192], weight}, heaviest first, at most 16). The server finds the voices of
the section, maps each to a known centroid or calls it ``new:1``, ``new:2`` ..., labels every 20 ms frame with a
Viterbi pass and returns the label runs as turns::

    {start_ms, end_ms, text, speaker_ref, cos, vector[192], support_s}

``speaker_ref``/``vector``/``cos`` are None when the section has too little speech for one embedding window; the
turn still carries its text. ``text`` is filled from the ASR words by ``assign_words_to_turns``.

Inputs per section: the speech mask (seg-3.0 on the section, ``section_speech_mask``) and ReDimNet2 windows of
1.5 s (hop 0.5 s) and 3 s (hop 1.0 s) computed on the section audio only. The frozen configuration is the one the
live bench measured (docs: live2/NOTES.md of the research): every known centroid is a Viterbi state, known at
cos >= 0.6, AHC distance 0.4 inside the section, a new voice needs 1 s of window support, switch penalty 3,
temperature 0.1, turns shorter than 0.5 s of speech are absorbed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .genesis_whisper_server_local_diarization import (
    GRID_S,
    SAMPLE_RATE,
    ahc,
    compute_window_features,
    unit,
    viterbi,
)


SPEAKER_TURNS_EMBEDDING_MODEL = "ReDimNet2-B6"
SPEAKER_TURNS_DIMENSION = 192
SPEAKER_TURNS_MAX_KNOWN = 16
SPEAKER_TURNS_MAX_TURNS = 64  # the Genesis3 reader drops an answer with more turns
SPEAKER_TURNS_MAX_ID_LENGTH = 64
# Section windows: name -> (length s, hop s). Embedded in server batches (<= 8, padded to 1/2/4/8).
SECTION_WINDOW_SCALES: Dict[str, tuple[float, float]] = {"w15": (1.5, 0.5), "w30": (3.0, 1.0)}

DEFAULT_PARAMS: Dict[str, Any] = dict(
    known_thr=0.6,      # a section voice is a known speaker at cos >= known_thr
    slice_thr=0.4,      # AHC cosine distance inside the section
    min_new_s=1.0,      # a new voice needs this much window support in the section
    min_state_s=0.6,    # after the first Viterbi pass a state with less speech is dropped
    penalty=3.0, temp=0.1, pause_free=0.5, w3=0.5, core=0.5,
    min_turn_s=0.5,     # turns with less speech are absorbed into a neighbour
    states="all",       # every known centroid is a Viterbi state ("matched": only the ones found in the section)
    min_cov3=0.5,       # 3 s windows need this much speech to seed the voice discovery
    # Support seconds one seed window stands for. Kept at the benchmarked values (fine-hop bench, also used for the
    # measured hop 0.5/1.0 run); they only gate min_new_s.
    seed_w3=0.5, seed_w15=0.25,
)


# ---------------------------------------------------------------------------
# Request / capability helpers
# ---------------------------------------------------------------------------

def speaker_turns_capability(available: bool) -> Dict[str, Any]:
    """The ``speaker_turns`` block of ``GET /v2/capabilities``."""

    if not available:
        return {"supported": False}
    return {
        "supported": True,
        "embedding_model": SPEAKER_TURNS_EMBEDDING_MODEL,
        "dimension": SPEAKER_TURNS_DIMENSION,
        "max_known_speakers": SPEAKER_TURNS_MAX_KNOWN,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_speaker_turns_request(value: Any) -> List[Dict[str, Any]]:
    """Validated ``known_speakers`` of the request field; ValueError (-> 422) on anything malformed."""

    if not isinstance(value, Mapping):
        raise ValueError("speaker_turns muss ein Objekt sein.")
    unknown = set(value) - {"known_speakers"}
    if unknown:
        raise ValueError(f"speaker_turns: unbekannte Felder {sorted(unknown)}.")
    raw = value.get("known_speakers", [])
    if not isinstance(raw, list):
        raise ValueError("speaker_turns.known_speakers muss eine Liste sein.")
    if len(raw) > SPEAKER_TURNS_MAX_KNOWN:
        raise ValueError(f"speaker_turns.known_speakers: hoechstens {SPEAKER_TURNS_MAX_KNOWN} Eintraege.")
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        where = f"speaker_turns.known_speakers[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} muss ein Objekt sein.")
        extra = set(entry) - {"id", "vector", "weight"}
        if extra:
            raise ValueError(f"{where}: unbekannte Felder {sorted(extra)}.")
        speaker_id = entry.get("id")
        if not isinstance(speaker_id, str) or not speaker_id.strip() or len(speaker_id) > SPEAKER_TURNS_MAX_ID_LENGTH:
            raise ValueError(f"{where}.id muss ein nicht leerer Text sein.")
        if speaker_id.startswith("new:"):
            raise ValueError(f"{where}.id darf nicht mit 'new:' beginnen.")
        if speaker_id in seen:
            raise ValueError(f"{where}.id ist doppelt.")
        vector = entry.get("vector")
        if (not isinstance(vector, list) or len(vector) != SPEAKER_TURNS_DIMENSION
                or not all(_finite_number(x) for x in vector)):
            raise ValueError(f"{where}.vector muss {SPEAKER_TURNS_DIMENSION} endliche Zahlen enthalten.")
        array = np.asarray(vector, np.float32)
        norm = float(np.linalg.norm(array))
        if not norm > 0 or not math.isfinite(norm):
            raise ValueError(f"{where}.vector darf kein Nullvektor sein.")
        weight = entry.get("weight", 1.0)
        if not _finite_number(weight) or float(weight) <= 0:
            raise ValueError(f"{where}.weight muss eine positive Zahl sein.")
        seen.add(speaker_id)
        out.append({"id": speaker_id, "vector": array / norm, "weight": float(weight)})
    return out


# ---------------------------------------------------------------------------
# Section features
# ---------------------------------------------------------------------------

def section_frames(num_samples: int) -> int:
    return int(np.ceil(num_samples / SAMPLE_RATE / GRID_S))


def section_speech_mask(seg: Mapping[str, Any], n: int) -> np.ndarray:
    """Speech mask (seg-3.0 count >= 1) on the 20 ms grid of a section; frames past the section are ignored."""

    data = np.asarray(seg["data"]).astype(np.float32)
    cstep, fstep = float(seg["chunk_step"]), float(seg["frame_step"])
    nc, nf, _ = data.shape
    cover = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    win = np.hamming(nf).astype(np.float32)
    for c in range(nc):
        t = c * cstep + np.arange(nf) * fstep + fstep / 2
        ok = t < n * GRID_S
        gi = (t[ok] / GRID_S).astype(int)
        np.add.at(cover, gi, win[ok])
        np.add.at(cnt, gi, win[ok] * data[c][ok].sum(1))
    mask = np.zeros(n, bool)
    valid = cover > 0
    mask[valid] = np.rint(cnt[valid] / cover[valid]) >= 1
    return mask


def section_features(audio: np.ndarray, seg: Mapping[str, Any], embed: Any = None) -> Dict[str, Any]:
    """Speech mask and windows of one section (``embed`` as in ``compute_window_features``)."""

    samples = np.asarray(audio, np.float32).reshape(-1)
    n = section_frames(len(samples))
    sm = section_speech_mask(seg, n)
    windows = compute_window_features(samples, sm, int(GRID_S * SAMPLE_RATE), scales=SECTION_WINDOW_SCALES,
                                      embed=embed)
    return {"dur": len(samples) / SAMPLE_RATE, "sm": sm, "w15": windows["w15"], "w30": windows["w30"]}


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------

def _frames(st: np.ndarray, en: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.clip((st / GRID_S).astype(int), 0, n), np.clip((en / GRID_S).astype(int), 0, n)


def _windows(section: Mapping[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st, en, X = section[key]
    X = np.asarray(X, np.float32).reshape(-1, SPEAKER_TURNS_DIMENSION) if len(X) else np.zeros(
        (0, SPEAKER_TURNS_DIMENSION), np.float32)
    return np.asarray(st, np.float64), np.asarray(en, np.float64), unit(X)


def _score(section: Mapping[str, Any], C: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    n = len(section["sm"])
    score = np.zeros((n, len(C)), np.float32)
    cnt = np.zeros(n, np.float32)
    for key, wgt in (("w15", 1.0), ("w30", p["w3"])):
        st, en, X = section[key]
        if len(X) == 0:
            continue
        S = X @ C.T
        fa, fb = _frames(st, en, n)
        mid = (fa + fb) // 2
        half = np.maximum(1, ((fb - fa) * p["core"]).astype(int) // 2)
        for i in range(len(X)):
            lo, hi = max(0, mid[i] - half[i]), min(n, mid[i] + half[i])
            score[lo:hi] += wgt * S[i]
            cnt[lo:hi] += wgt
    has = cnt > 0
    score[has] /= cnt[has, None]
    if has.any() and not has.all():
        idx = np.where(has)[0]
        pos = np.searchsorted(idx, np.arange(n))
        right = idx[np.clip(pos, 0, len(idx) - 1)]
        left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
        use = np.where(np.abs(left - np.arange(n)) < np.abs(right - np.arange(n)), left, right)
        score[~has] = score[use[~has]]
    return score


def _free(sm: np.ndarray, pause_free: float) -> np.ndarray:
    k = max(1, int(round(pause_free / GRID_S)))
    free = np.zeros(len(sm), bool)
    run = 0
    for t in range(len(sm)):
        if not sm[t]:
            run += 1
        else:
            if run >= k:
                free[t] = True
            run = 0
    return free


def _turn_vec(section: Mapping[str, Any], lo: int, hi: int, path: np.ndarray, lab: int) -> tuple[Optional[np.ndarray], int]:
    """Mean of the windows (1.5 s first, else 3 s) inside [lo, hi) whose speech frames are >= 80 % ``lab``."""

    n = len(path)
    sm = section["sm"]
    for key in ("w15", "w30"):
        st, en, X = section[key]
        if len(X) == 0:
            continue
        fa, fb = _frames(st, en, n)
        sel = []
        for i in range(len(X)):
            if not (fa[i] >= lo - 5 and fb[i] <= hi + 5 and fb[i] > fa[i]):
                continue
            m = sm[fa[i]:fb[i]]
            if m.any() and (path[fa[i]:fb[i]][m] == lab).mean() >= 0.8:
                sel.append(i)
        if sel:
            return unit(X[sel].mean(0)), len(sel)
    return None, 0


def _merge_equal(runs: List[List[int]]) -> List[List[int]]:
    merged: List[List[int]] = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return merged


def _absorb(runs: List[List[int]], path: np.ndarray, sm: np.ndarray, score: np.ndarray, limit_s: float,
            max_runs: Optional[int] = None) -> List[List[int]]:
    """Absorb the run with the least speech into its better-scoring neighbour while it is below ``limit_s``
    (or, with ``max_runs``, while there are too many runs)."""

    while len(runs) > 1:
        sup = [sm[a:b + 1].sum() * GRID_S for _, a, b in runs]
        i = int(np.argmin(sup))
        if not (sup[i] < limit_s or (max_runs is not None and len(runs) > max_runs)):
            break
        _, a, b = runs[i]
        cand = []
        if i > 0:
            cand.append((score[a:b + 1, runs[i - 1][0]].mean(), i - 1))
        if i + 1 < len(runs):
            cand.append((score[a:b + 1, runs[i + 1][0]].mean(), i + 1))
        j = max(cand)[1]
        runs[i][0] = runs[j][0]
        path[a:b + 1] = runs[j][0]
        runs = _merge_equal(runs)
    return runs


def _speech_turn(sm: np.ndarray) -> Dict[str, Any]:
    idx = np.where(sm)[0]
    return dict(start_ms=int(idx[0] * GRID_S * 1000), end_ms=int((idx[-1] + 1) * GRID_S * 1000), text="",
                speaker_ref=None, cos=None, vector=None, support_s=round(float(sm.sum() * GRID_S), 2))


def speaker_turns(section: Mapping[str, Any], known_speakers: Sequence[Mapping[str, Any]],
                  params: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Turns of one section without text, in time order.

    ``section``: {sm: bool mask on the 20 ms grid, w15/w30: (start s, end s, vectors)} with section-relative times.
    ``known_speakers``: [{id, vector, weight}] heaviest first (as ``parse_speaker_turns_request`` returns them).
    """

    p = dict(DEFAULT_PARAMS, **(params or {}))
    sm = np.asarray(section["sm"], bool)
    n = len(sm)
    if not sm.any():
        return []
    sec = {"sm": sm, "w15": _windows(section, "w15"), "w30": _windows(section, "w30")}
    if known_speakers:
        Kn = unit(np.array([np.asarray(k["vector"], np.float32) for k in known_speakers], np.float32))
    else:
        Kn = np.zeros((0, SPEAKER_TURNS_DIMENSION), np.float32)
    ids = [str(k["id"]) for k in known_speakers]
    st3, en3, X3 = sec["w30"]
    _, _, X1 = sec["w15"]

    # ---- voices of this section
    a3, b3 = _frames(st3, en3, n)
    cov3 = np.array([sm[i:j].mean() if j > i else 0 for i, j in zip(a3, b3)])
    seed, seed_w = (X3[cov3 >= p["min_cov3"]] if len(X3) else X3), p["seed_w3"]
    if len(seed) < 2:
        seed, seed_w = X1, p["seed_w15"]
    if len(seed) == 0:
        # less than one window of speech: one turn without speaker data, the client keeps the text
        return [_speech_turn(sm)]
    lab = ahc(seed, p["slice_thr"])
    states: List[np.ndarray] = []
    refs: List[str] = []
    matched: set[int] = set()
    new_k = 0
    labels = np.unique(lab)
    for c in labels:
        v = unit(seed[lab == c].mean(0))
        sup = (lab == c).sum() * seed_w
        if len(Kn):
            s = Kn @ v
            j = int(np.argmax(s))
            if s[j] >= p["known_thr"]:
                matched.add(j)
                continue
        if sup < p["min_new_s"] and (len(Kn) or len(labels) > 1):
            continue
        new_k += 1
        states.append(v)
        refs.append(f"new:{new_k}")
    use_known = range(len(Kn)) if p["states"] == "all" else sorted(matched)
    for j in use_known:
        states.append(Kn[j])
        refs.append(ids[j])
    if not states:
        # every seed cluster was too small: the biggest becomes the one new voice
        c = np.bincount(lab).argmax()
        states.append(unit(seed[lab == c].mean(0)))
        refs.append("new:1")
    C = np.stack(states)
    free = _free(sm, p["pause_free"])
    for it in range(2):
        score = _score(sec, C, p)
        path = viterbi(score / p["temp"], p["penalty"], free)
        if it == 1:
            break
        talk = np.array([((path == k) & sm).sum() * GRID_S for k in range(len(C))])
        keep = talk >= p["min_state_s"]
        if not keep.any():
            keep[np.argmax(talk)] = True
        # refine the new centroids from the fine windows that sit in their label
        newC = []
        for k in range(len(C)):
            if refs[k].startswith("new:") and keep[k]:
                v, m = _turn_vec(sec, 0, n, path, k)
                newC.append(v if v is not None and m >= 3 else C[k])
            else:
                newC.append(C[k])
        C = np.stack(newC)[keep]
        refs = [r for r, k in zip(refs, keep) if k]
        count = 0
        for i, r in enumerate(refs):
            if r.startswith("new:"):
                count += 1
                refs[i] = f"new:{count}"

    # ---- turns: runs of one label over the speech frames (silence inside a turn does not split it)
    runs: List[List[int]] = []
    for t in np.where(sm)[0]:
        lab_t = int(path[t])
        if runs and runs[-1][0] == lab_t:
            runs[-1][2] = int(t)
        else:
            runs.append([lab_t, int(t), int(t)])
    runs = _absorb(runs, path, sm, score, p["min_turn_s"])
    if len(runs) > SPEAKER_TURNS_MAX_TURNS:
        runs = _absorb(runs, path, sm, score, 0.0, max_runs=SPEAKER_TURNS_MAX_TURNS)
    turns = []
    for l, a, b in runs:
        v, _ = _turn_vec(sec, a, b + 1, path, l)
        if v is None:
            v = C[l]
        turns.append(dict(start_ms=int(round(a * GRID_S * 1000)), end_ms=int(round((b + 1) * GRID_S * 1000)),
                          text="", speaker_ref=refs[l], cos=round(float(v @ C[l]), 3),
                          vector=v.astype(np.float32), support_s=round(float(sm[a:b + 1].sum() * GRID_S), 2)))
    return turns


# ---------------------------------------------------------------------------
# Words -> turns, wire format
# ---------------------------------------------------------------------------

def assign_words_to_turns(turns: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of ``turns`` with ``text`` = the ASR words whose midpoint lies in the turn (else the nearest turn).

    ``words``: [{t0, t1 (section seconds), word}] in time order. Words in a section without any turn (no speech in
    the mask) become one turn without speaker data, so no text is lost.
    """

    out = [dict(turn) for turn in turns]
    pieces: List[List[str]] = [[] for _ in out]
    loose: List[Mapping[str, Any]] = []
    for word in words:
        text = str(word.get("word") or "").strip()
        if not text:
            continue
        if not out:
            loose.append(word)
            continue
        mid_ms = (float(word["t0"]) + float(word["t1"])) * 500.0
        best, best_gap = 0, None
        for index, turn in enumerate(out):
            a, b = float(turn["start_ms"]), float(turn["end_ms"])
            gap = 0.0 if a <= mid_ms <= b else (a - mid_ms if mid_ms < a else mid_ms - b)
            if best_gap is None or gap < best_gap:
                best, best_gap = index, gap
            if gap == 0.0:
                break
        pieces[best].append(text)
    for turn, piece in zip(out, pieces):
        turn["text"] = " ".join(piece)
    if loose:
        t0 = min(float(w["t0"]) for w in loose)
        t1 = max(float(w["t1"]) for w in loose)
        out.append(dict(start_ms=max(0, int(round(t0 * 1000))), end_ms=max(0, int(round(t1 * 1000))),
                        text=" ".join(str(w.get("word")).strip() for w in loose), speaker_ref=None, cos=None,
                        vector=None, support_s=0.0))
    return out


def turns_to_wire(turns: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """JSON-ready turns (vectors as float lists)."""

    wire = []
    for turn in turns:
        vector = turn.get("vector")
        wire.append({
            "start_ms": int(turn["start_ms"]),
            "end_ms": int(turn["end_ms"]),
            "text": str(turn.get("text") or ""),
            "speaker_ref": turn.get("speaker_ref"),
            "cos": None if turn.get("cos") is None else float(turn["cos"]),
            "vector": None if vector is None else [float(x) for x in np.asarray(vector, np.float32)],
            "support_s": float(turn.get("support_s") or 0.0),
        })
    return wire


__all__ = [
    "DEFAULT_PARAMS",
    "SECTION_WINDOW_SCALES",
    "SPEAKER_TURNS_DIMENSION",
    "SPEAKER_TURNS_EMBEDDING_MODEL",
    "SPEAKER_TURNS_MAX_KNOWN",
    "SPEAKER_TURNS_MAX_TURNS",
    "assign_words_to_turns",
    "parse_speaker_turns_request",
    "section_features",
    "section_frames",
    "section_speech_mask",
    "speaker_turns",
    "speaker_turns_capability",
    "turns_to_wire",
]
