"""ASR glossary ("keywords"): terms the model should spell as given (names, products, jargon).

The server glossary (setting ``asr_keywords``) comes first, a request's own terms after it; the merged list is
trimmed, deduplicated case-insensitively and limited. Only Granite Speech uses the terms (as a prompt suffix), every
other backend accepts and ignores them. A non-empty glossary rides in the processing key as a ``KeywordSet``, so the
scheduler never batches two jobs with different glossaries; without terms the key stays the plain 5-tuple.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .genesis_whisper_server_globals import current_settings, settings_lock, uses_granite_backend

KEYWORDS_MAX_ENTRIES = 1000
KEYWORDS_MAX_CHARS = 16000
KEYWORD_MAX_LENGTH = 100
KEYWORDS_TRUNCATED_WARNING = "KEYWORDS_TRUNCATED"
# Position of the KeywordSet in a processing key (after model, device, cache path, language, precision).
KEYWORDS_KEY_INDEX = 5

_FORM_SEPARATORS = re.compile(r"[\r\n,]+")


class KeywordValidationError(ValueError):
    """The request's keywords are not a list of strings."""


@dataclass(frozen=True)
class Glossary:
    """The final terms of one request and whether merge, length drop or limits cut anything."""

    terms: Tuple[str, ...] = ()
    truncated: bool = False

    @property
    def count(self) -> int:
        return len(self.terms)


@dataclass(frozen=True)
class KeywordSet:
    """Processing-key element: equal and hashed by digest only, the terms travel along for the engine."""

    digest: str
    terms: Tuple[str, ...] = field(compare=False, repr=False)


def normalize_keywords(values: Iterable[str]) -> Tuple[List[str], bool]:
    """Trimmed, non-empty, at most KEYWORD_MAX_LENGTH chars, first spelling of each case-insensitive duplicate."""

    seen = set()
    kept: List[str] = []
    dropped = False
    for value in values:
        term = str(value).strip()
        if not term:
            continue
        if len(term) > KEYWORD_MAX_LENGTH:
            dropped = True
            continue
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(term)
    return kept, dropped


def limit_keywords(terms: Sequence[str]) -> Tuple[List[str], bool]:
    """At most KEYWORDS_MAX_ENTRIES terms and KEYWORDS_MAX_CHARS chars in all; the excess is cut from the end."""

    kept: List[str] = []
    chars = 0
    for term in terms:
        if len(kept) >= KEYWORDS_MAX_ENTRIES or chars + len(term) > KEYWORDS_MAX_CHARS:
            return kept, True
        kept.append(term)
        chars += len(term)
    return kept, False


def sanitize_keyword_setting(value: Any) -> List[str]:
    """The saved ``asr_keywords``: a list (or newline/comma text) normalised and capped, never an error."""

    if isinstance(value, str):
        values: Iterable[Any] = _FORM_SEPARATORS.split(value)
    elif isinstance(value, (list, tuple)):
        values = [item for item in value if isinstance(item, str)]
    else:
        return []
    terms, _ = normalize_keywords(values)
    return limit_keywords(terms)[0]


def parse_request_keywords(value: Any) -> Optional[List[str]]:
    """The v2 request's ``keywords``: None when absent, else the raw list after the type check."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise KeywordValidationError("keywords muss eine Liste von Strings sein.")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise KeywordValidationError(f"keywords[{index}] muss ein String sein.")
    return list(value)


def parse_form_keywords(value: Optional[str]) -> List[str]:
    """The v1 form field: terms separated by newlines or commas."""

    if not value:
        return []
    return _FORM_SEPARATORS.split(str(value))


def server_glossary() -> List[str]:
    with settings_lock:
        configured = current_settings.get("asr_keywords", [])
    return sanitize_keyword_setting(configured)


def resolve_glossary(request_terms: Optional[Sequence[str]] = None) -> Glossary:
    """Server glossary first, the request's terms after it, then normalised and limited."""

    terms, dropped = normalize_keywords([*server_glossary(), *(request_terms or [])])
    terms, cut = limit_keywords(terms)
    return Glossary(terms=tuple(terms), truncated=dropped or cut)


def keywords_digest(terms: Sequence[str]) -> str:
    payload = json.dumps(list(terms), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def with_keywords(processing_key: tuple, glossary: Optional[Glossary]) -> tuple:
    """``processing_key`` with the glossary as its sixth element; no terms leave the 5-tuple unchanged."""

    base = tuple(processing_key[:KEYWORDS_KEY_INDEX])
    if glossary is None or not glossary.terms:
        return base
    return base + (KeywordSet(keywords_digest(glossary.terms), tuple(glossary.terms)),)


def key_keywords(processing_key: Sequence[Any]) -> Tuple[str, ...]:
    if len(processing_key) > KEYWORDS_KEY_INDEX and isinstance(processing_key[KEYWORDS_KEY_INDEX], KeywordSet):
        return processing_key[KEYWORDS_KEY_INDEX].terms
    return ()


def keywords_block(glossary: Glossary, model_id: Optional[str]) -> dict:
    """Response object: applied only when Granite served the request and there were terms to send."""

    return {
        "applied": bool(glossary.terms) and bool(model_id) and uses_granite_backend(str(model_id)),
        "count": glossary.count,
        "truncated": glossary.truncated,
    }


def truncation_warnings(glossary: Glossary) -> List[dict]:
    return [{"code": KEYWORDS_TRUNCATED_WARNING}] if glossary.truncated else []


def capabilities_block(model_id: str) -> dict:
    return {
        "supported": uses_granite_backend(model_id),
        "max_entries": KEYWORDS_MAX_ENTRIES,
        "max_chars": KEYWORDS_MAX_CHARS,
        "server_glossary_count": len(server_glossary()),
    }


__all__ = [
    "Glossary",
    "KEYWORDS_KEY_INDEX",
    "KEYWORDS_MAX_CHARS",
    "KEYWORDS_MAX_ENTRIES",
    "KEYWORDS_TRUNCATED_WARNING",
    "KEYWORD_MAX_LENGTH",
    "KeywordSet",
    "KeywordValidationError",
    "capabilities_block",
    "key_keywords",
    "keywords_block",
    "keywords_digest",
    "limit_keywords",
    "normalize_keywords",
    "parse_form_keywords",
    "parse_request_keywords",
    "resolve_glossary",
    "sanitize_keyword_setting",
    "server_glossary",
    "truncation_warnings",
    "with_keywords",
]
