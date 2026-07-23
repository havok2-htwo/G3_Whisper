"""Turn gating and ASR-output hygiene for the v2 diarization path.

Measured on real meeting audio, DIA's exclusive partition emits many sub-word
"ghost" turns (17-150 ms) that Cohere fills with subtitle-style hallucinations
("...", "Vielen Dank", "Musik", stray CJK). This module removes that noise with
signals that are safe (duration + non-speech text token), NOT with a raw VAD
speech-ratio (which was shown to also delete quiet real speech) and NOT with a
text blacklist (real "ja"/"vielen dank" instances exist).

Two stages:
  pre-ASR   prefilter_turns()  -> drop ghost turns, then re-merge same-speaker
                                  neighbours the ghosts used to split apart.
  post-ASR  clean_segment_text()/is_empty_content() -> strip Cohere artefacts
                                  inline; a long turn that still decodes to
                                  nothing is an ASR failure and is kept+flagged,
                                  never silently dropped.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# A real word does not fit below ~150 ms and Silero cannot resolve it either;
# turns shorter than this are structural ghosts from the exclusive partition.
TURN_GATE_MIN_MS = 150
# After the ghosts are gone the two halves of an A-[ghost B]-A sandwich are
# adjacent again; merge same-speaker neighbours within this gap.
TURN_REMERGE_GAP_MS = 400
# Junk text on a turn this long is Cohere failing on real speech, not silence:
# keep it and flag it for re-transcription instead of dropping real audio.
TURN_LONG_JUNK_MS = 1000

# Hiragana, Katakana, CJK unified, Hangul, CJK symbols, fullwidth forms.
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯　-〿＀-￯]")
# Cohere occasionally leaks its internal correction framing into the text.
_PREFIX_RE = re.compile(r"^\s*input transcript corrected\s*:\s*", re.IGNORECASE)
_ELLIPSIS_ONLY_RE = re.compile(r"^[\s.…]*$")
_WORD_RE = re.compile(r"\w", re.UNICODE)


def clean_segment_text(text: str) -> str:
    """Strip Cohere artefacts inline. Does NOT decide keep/drop (see is_empty_content)."""
    if not text:
        return ""
    cleaned = _PREFIX_RE.sub("", text)
    cleaned = _CJK_RE.sub("", cleaned)
    # Removing an inline token can leave ", ," / doubled spaces / a leading comma.
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"(^|[,;:])\s*,", r"\1", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    return cleaned.strip(" ,").strip()


def is_empty_content(text: str) -> bool:
    """True when text carries no lexical content (empty, only '...'/punctuation)."""
    if not text or not text.strip():
        return True
    if _ELLIPSIS_ONLY_RE.match(text):
        return True
    return _WORD_RE.search(text) is None


def _turn_speaker_key(turn: Mapping[str, Any]) -> tuple[str, str]:
    speaker = str(turn.get("speaker_id", ""))
    original = str(turn.get("original_speaker_id", speaker))
    return speaker, original


def prefilter_turns(
    turns: Sequence[Mapping[str, Any]],
    *,
    min_ms: int = TURN_GATE_MIN_MS,
    remerge_gap_ms: int = TURN_REMERGE_GAP_MS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop sub-``min_ms`` ghost turns, then merge same-speaker neighbours.

    Returns (kept_turns, diagnostics). ``kept_turns`` keeps the original dict
    fields; merged turns span from the first start_ms to the last end_ms and set
    ``merged_from`` to the count of source turns (>1 means the text must be
    re-transcribed, since concatenating two half-word ASR strings is worse than
    one ASR pass over the joined audio).
    """
    dropped_ghost = 0
    survivors: list[dict[str, Any]] = []
    for turn in turns:
        duration = int(turn["end_ms"]) - int(turn["start_ms"])
        if duration < min_ms:
            dropped_ghost += 1
            continue
        survivors.append(dict(turn))

    merged: list[dict[str, Any]] = []
    remerged = 0
    for turn in survivors:
        if (
            merged
            and _turn_speaker_key(merged[-1]) == _turn_speaker_key(turn)
            and int(turn["start_ms"]) <= int(merged[-1]["end_ms"]) + remerge_gap_ms
        ):
            merged[-1]["end_ms"] = max(int(merged[-1]["end_ms"]), int(turn["end_ms"]))
            merged[-1]["merged_from"] = int(merged[-1].get("merged_from", 1)) + 1
            remerged += 1
        else:
            entry = dict(turn)
            entry.setdefault("merged_from", 1)
            merged.append(entry)

    diagnostics = {
        "input_turns": len(turns),
        "dropped_ghost_turns": dropped_ghost,
        "remerged_turns": remerged,
        "output_turns": len(merged),
    }
    return merged, diagnostics


def finalize_segment_text(text: str, duration_ms: int) -> tuple[str, str | None]:
    """Post-ASR hygiene for one segment.

    Returns (text, disposition). disposition is:
      None            -> keep the returned (cleaned) text
      "drop"          -> discard: short turn with no lexical content
      "asr_failure"   -> keep, but flag: long turn that decoded to nothing
    """
    cleaned = clean_segment_text(text)
    if is_empty_content(cleaned):
        if duration_ms >= TURN_LONG_JUNK_MS:
            return text.strip(), "asr_failure"
        return "", "drop"
    return cleaned, None
