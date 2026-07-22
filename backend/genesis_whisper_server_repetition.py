"""Conservative cleanup for exact repetition loops in ASR transcripts.

The filter deliberately works on normalized Unicode word tokens while applying
all removals to the original text.  This lets it compare case, compatibility
forms, whitespace, and terminal punctuation independently without rewriting the
first occurrence that is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


REPETITION_FILTER_HEADER = "X-G3-Repetition-Filter"
MAX_PATTERN_TOKENS = 32
MIN_SINGLE_TOKEN_REPETITIONS = 5
MIN_MULTI_TOKEN_REPETITIONS = 3

_WORD_RE = re.compile(r"\w+(?:[-'\u2019]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class _Token:
    normalized: str
    start: int
    end: int


def repetition_filter_enabled(header_value: str | None) -> bool:
    """Return whether transcript repetition filtering should run.

    Only the explicit value ``off`` disables filtering.  A missing, empty, or
    unknown value keeps it enabled so adding the opt-out header cannot
    accidentally turn filtering off for existing clients.
    """

    return header_value is None or header_value.strip().casefold() != "off"


def filter_repeated_patterns(text: str) -> str:
    """Collapse exact consecutive token-pattern loops to one occurrence.

    Single-token runs are collapsed from five repetitions onward.  Patterns of
    two through 32 tokens are collapsed from three repetitions onward.  Matching
    uses NFKC and case-folded Unicode word tokens; whitespace and terminal
    punctuation therefore do not affect equality.  The original spelling and
    punctuation of the first occurrence are retained.
    """

    if not text:
        return text

    current = text
    while True:
        filtered = _filter_once(current)
        if filtered == current:
            return filtered
        current = filtered


def _filter_once(text: str) -> str:
    tokens = _tokenize(text)
    if len(tokens) < MIN_SINGLE_TOKEN_REPETITIONS:
        return text

    normalized = [token.normalized for token in tokens]
    removals: list[tuple[int, int]] = []
    index = 0

    while index < len(tokens):
        match = _find_repetition(normalized, index)
        if match is None:
            index += 1
            continue

        pattern_length, repetition_count = match
        first_end = tokens[index + pattern_length - 1].end
        run_end_index = index + pattern_length * repetition_count - 1
        removals.append((first_end, tokens[run_end_index].end))
        index = run_end_index + 1

    if not removals:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in removals:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _find_repetition(normalized: list[str], start: int) -> tuple[int, int] | None:
    remaining = len(normalized) - start
    maximum_pattern_length = min(MAX_PATTERN_TOKENS, remaining // MIN_MULTI_TOKEN_REPETITIONS)

    for pattern_length in range(1, maximum_pattern_length + 1):
        minimum_repetitions = (
            MIN_SINGLE_TOKEN_REPETITIONS
            if pattern_length == 1
            else MIN_MULTI_TOKEN_REPETITIONS
        )
        if pattern_length * minimum_repetitions > remaining:
            continue

        pattern = normalized[start : start + pattern_length]
        repetitions = 1
        next_start = start + pattern_length
        while (
            next_start + pattern_length <= len(normalized)
            and normalized[next_start : next_start + pattern_length] == pattern
        ):
            repetitions += 1
            next_start += pattern_length

        if repetitions >= minimum_repetitions:
            return pattern_length, repetitions

    return None


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _WORD_RE.finditer(text):
        normalized = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if not normalized:
            continue

        # Keep punctuation attached to the word in the retained first form.  It
        # is deliberately excluded from the normalized token used for matching.
        end = match.end()
        while end < len(text) and unicodedata.category(text[end]).startswith("P"):
            end += 1
        tokens.append(_Token(normalized=normalized, start=match.start(), end=end))
    return tokens


__all__ = [
    "REPETITION_FILTER_HEADER",
    "filter_repeated_patterns",
    "repetition_filter_enabled",
]
