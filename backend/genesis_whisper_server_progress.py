"""Progress and estimated time remaining of a running /v2/audio/process request (Genesis3 docs/254).

A client names its request with ``X-Progress-ID``; ``GET /v2/progress/{id}`` then reports the phase, done/total
units of that phase, an overall fraction and an ETA while the synchronous POST is still running. The table is
in memory only (one process), bounded and forgets an entry ``FINISHED_TTL_S`` after its request ended. It never
holds request contents.

How the numbers come about:
- every mode has a fixed phase order; ``enter_phase`` moves on (skipped phases count as done);
- a phase knows ``done``/``total`` units when the code declares them, from ``run_sliced_gpu_phase`` (one unit per
  item, counted automatically in phases that opt in), or from a probe (the ASR batch jobs of the request);
- the overall fraction weighs every phase with its expected duration (milliseconds per audio second: measured
  defaults, replaced by a running mean of this process's finished requests);
- the ETA is the rest of the current phase at its own measured rate plus the expected rest of the later phases,
  scaled by how fast this request ran so far. It is ``None`` while it cannot be estimated honestly: audio not yet
  decoded, the current phase without measurable progress and already past its expectation, or the request is
  waiting for the GPU lock (``state`` "waiting").
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

PROGRESS_HEADER = "x-progress-id"
PROGRESS_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
MAX_ENTRIES = 256
FINISHED_TTL_S = 600.0
# Below this much audio the fixed costs dominate; such requests do not teach the per-audio-second rates.
LEARN_MIN_AUDIO_S = 30.0
LEARN_WEIGHT = 0.3
# A phase's own rate is trusted once this much of it ran (fraction) or this long (seconds of active time).
RATE_MIN_FRACTION = 0.02
RATE_MIN_ACTIVE_S = 1.0

# Milliseconds per audio second. Seeded from Genesis3 diarization timings of the RTX 4090 server (meetings of 14 s
# to 96 min: decode ~0.6, diarization ~14-27 split over its three sub-phases, embedding ~2, ASR ~4.5, the rest
# alignment/verification); every finished request of at least LEARN_MIN_AUDIO_S updates them.
DEFAULT_RATES_MS_PER_AUDIO_S: Dict[str, float] = {
    "upload": 0.5,
    "decode": 0.6,
    "vad": 1.5,
    "segmentation": 4.0,
    "speaker_windows": 10.0,
    "clustering": 6.0,
    "embedding": 2.0,
    "language_detection": 1.0,
    "transcription": 4.5,
    "alignment": 4.0,
    "verification": 3.0,
    "finishing": 0.5,
    "processing": 5.0,
}

MODE_PHASES: Dict[str, Tuple[str, ...]] = {
    "diarization": (
        "upload",
        "decode",
        "vad",
        "segmentation",
        "speaker_windows",
        "clustering",
        "embedding",
        "language_detection",
        "transcription",
        "alignment",
        "verification",
        "finishing",
    ),
}
DEFAULT_PHASES: Tuple[str, ...] = ("upload", "decode", "processing")
# Phases whose run_sliced_gpu_phase calls count as units without an explicit total.
AUTO_COUNTED_PHASES = frozenset({"segmentation", "embedding", "language_detection", "alignment", "verification"})

Probe = Callable[[], Optional[Tuple[int, int, bool]]]


class ProgressEntry:
    """One tracked request. Mutated on the event loop (and from worker threads via the context), read under the
    registry lock."""

    def __init__(self, progress_id: str, owner: Optional[str], now: float) -> None:
        self.progress_id = progress_id
        self.owner = owner
        self.mode: Optional[str] = None
        self.phases: Tuple[str, ...] = DEFAULT_PHASES
        self.started = now
        self.audio_seconds: Optional[float] = None
        self.phase_index = 0
        self.phase_started = now
        self.phase_waited = 0.0
        self.units_done = 0
        self.units_total = 0
        self.declared_total = False
        self.probe: Optional[Probe] = None
        self.probe_seen = False
        self.waiting_since: Optional[float] = None
        self.waiting_depth = 0
        self.active_s: Dict[str, float] = {}
        self.max_progress = 0.0
        self.status = "running"
        self.finished_at: Optional[float] = None

    @property
    def phase(self) -> str:
        return self.phases[self.phase_index]


class ProgressRegistry:
    def __init__(self, *, max_entries: int = MAX_ENTRIES, ttl_s: float = FINISHED_TTL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: "OrderedDict[str, ProgressEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self.clock = clock
        self.rates: Dict[str, float] = dict(DEFAULT_RATES_MS_PER_AUDIO_S)
        self.learned: set[str] = set()

    # -- table -----------------------------------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        for key in [key for key, entry in self._entries.items()
                    if entry.finished_at is not None and now - entry.finished_at > self._ttl_s]:
            del self._entries[key]
        while len(self._entries) > self._max_entries:
            finished = next((key for key, entry in self._entries.items() if entry.finished_at is not None), None)
            del self._entries[finished if finished is not None else next(iter(self._entries))]

    def start(self, progress_id: Optional[str], owner: Optional[str]) -> Optional[ProgressEntry]:
        """A new entry, or None for a missing/invalid id or one still running / owned by another key."""

        if not progress_id or not PROGRESS_ID_PATTERN.match(progress_id):
            return None
        with self._lock:
            now = self.clock()
            self._prune(now)
            existing = self._entries.get(progress_id)
            if existing is not None and (existing.finished_at is None or existing.owner != owner):
                return None
            entry = ProgressEntry(progress_id, owner, now)
            self._entries.pop(progress_id, None)
            self._entries[progress_id] = entry
            self._prune(now)
            return entry

    def get(self, progress_id: str, owner: Optional[str]) -> Optional[ProgressEntry]:
        with self._lock:
            self._prune(self.clock())
            entry = self._entries.get(progress_id)
            return entry if entry is not None and entry.owner == owner else None

    # -- updates (all tolerate entry None) -------------------------------------------------------------------

    def set_mode(self, entry: ProgressEntry, mode: str) -> None:
        with self._lock:
            current = entry.phase
            entry.mode = mode
            entry.phases = MODE_PHASES.get(mode, DEFAULT_PHASES)
            entry.phase_index = entry.phases.index(current) if current in entry.phases else 0

    def set_audio_seconds(self, entry: ProgressEntry, seconds: float) -> None:
        with self._lock:
            entry.audio_seconds = max(0.0, float(seconds))

    def _close_phase(self, entry: ProgressEntry, now: float) -> None:
        if entry.waiting_since is not None:
            entry.phase_waited += now - entry.waiting_since
            entry.waiting_since = now
        active = max(0.0, now - entry.phase_started - entry.phase_waited)
        entry.active_s[entry.phase] = entry.active_s.get(entry.phase, 0.0) + active

    def enter_phase(self, entry: ProgressEntry, phase: str, *, total: Optional[int] = None,
                    probe: Optional[Probe] = None) -> None:
        with self._lock:
            if phase not in entry.phases:
                return
            target = entry.phases.index(phase)
            if target < entry.phase_index:
                return  # never back: a later fallback path re-running ASR keeps the phase it reached
            if target > entry.phase_index:
                now = self.clock()
                self._close_phase(entry, now)
                entry.phase_index = target
                entry.phase_started = now
                entry.phase_waited = 0.0
            entry.units_done = 0
            entry.units_total = max(0, int(total)) if total is not None else 0
            entry.declared_total = total is not None
            entry.probe = probe
            entry.probe_seen = False

    def add_units(self, entry: ProgressEntry, *, done: int = 0, total: int = 0) -> None:
        with self._lock:
            if entry.probe is not None:
                return
            if total and not entry.declared_total and entry.phase in AUTO_COUNTED_PHASES:
                entry.units_total += int(total)
            if done and (entry.declared_total or entry.phase in AUTO_COUNTED_PHASES):
                entry.units_done = min(entry.units_done + int(done), entry.units_total)

    def begin_wait(self, entry: ProgressEntry) -> None:
        with self._lock:
            entry.waiting_depth += 1
            if entry.waiting_since is None:
                entry.waiting_since = self.clock()

    def end_wait(self, entry: ProgressEntry) -> None:
        with self._lock:
            entry.waiting_depth = max(0, entry.waiting_depth - 1)
            if entry.waiting_depth == 0 and entry.waiting_since is not None:
                entry.phase_waited += self.clock() - entry.waiting_since
                entry.waiting_since = None

    def finish(self, entry: ProgressEntry, status: str) -> None:
        with self._lock:
            if entry.finished_at is not None:
                return
            now = self.clock()
            self._close_phase(entry, now)
            entry.waiting_since = None
            entry.waiting_depth = 0
            entry.status = status
            entry.finished_at = now
            if status == "completed":
                entry.max_progress = 1.0
                self._learn(entry)

    def _learn(self, entry: ProgressEntry) -> None:
        audio = entry.audio_seconds or 0.0
        if audio < LEARN_MIN_AUDIO_S:
            return
        for phase, seconds in entry.active_s.items():
            if phase == "upload" or seconds <= 0:
                continue  # upload speed is the client's network, not this server
            measured = seconds * 1000.0 / audio
            if phase in self.learned:
                self.rates[phase] = (1 - LEARN_WEIGHT) * self.rates[phase] + LEARN_WEIGHT * measured
            else:
                self.rates[phase] = measured
                self.learned.add(phase)

    # -- read ------------------------------------------------------------------------------------------------

    def _phase_units(self, entry: ProgressEntry) -> Tuple[Optional[int], Optional[int], bool]:
        """(done, total, waiting_in_queue) of the current phase; done/total None when unknown."""

        if entry.probe is not None:
            try:
                probed = entry.probe()
            except Exception:
                probed = None
            if probed is not None:
                done, total, queued = probed
                if total > 0:
                    entry.probe_seen = True
                    return max(0, min(done, total)), total, queued
            return None, None, False
        if entry.units_total > 0:
            return entry.units_done, entry.units_total, False
        return None, None, False

    def snapshot(self, entry: ProgressEntry) -> Dict[str, Any]:
        with self._lock:
            now = self.clock()
            end = entry.finished_at if entry.finished_at is not None else now
            done, total, queued = (None, None, False) if entry.finished_at is not None else self._phase_units(entry)
            waiting = entry.finished_at is None and (entry.waiting_since is not None or queued)
            phase = entry.phase
            phase_waited = entry.phase_waited + (now - entry.waiting_since if entry.waiting_since is not None else 0.0)
            phase_active = max(0.0, end - entry.phase_started - phase_waited) if entry.finished_at is None else 0.0
            audio = entry.audio_seconds
            expected = {name: self.rates.get(name, 1.0) for name in entry.phases}
            weight_sum = sum(expected.values()) or 1.0
            fraction: Optional[float] = (done / total) if (done is not None and total) else None

            progress = 1.0 if entry.status == "completed" else 0.0
            eta_ms: Optional[int] = 0 if entry.finished_at is not None else None
            if entry.finished_at is None:
                completed_weight = sum(expected[name] for name in entry.phases[: entry.phase_index])
                current_expected_s = expected[phase] * (audio or 0.0) / 1000.0
                if fraction is None and audio:
                    phase_part = min(0.95, phase_active / current_expected_s) if current_expected_s > 0 else 0.0
                else:
                    phase_part = fraction or 0.0
                progress = (completed_weight + expected[phase] * phase_part) / weight_sum
                eta_ms = self._eta_ms(entry, expected, fraction, phase_active, waiting)
            progress = max(entry.max_progress, min(1.0, progress))
            entry.max_progress = progress
            return {
                "progress_id": entry.progress_id,
                "state": entry.status if entry.finished_at is not None else ("waiting" if waiting else "running"),
                "mode": entry.mode,
                "phase": phase,
                "phase_index": entry.phase_index,
                "phase_count": len(entry.phases),
                "phase_done": done,
                "phase_total": total,
                "progress": round(progress, 4),
                "elapsed_ms": round((end - entry.started) * 1000),
                "eta_ms": eta_ms,
                "audio_duration_ms": round(audio * 1000) if audio is not None else None,
            }

    def _eta_ms(self, entry: ProgressEntry, expected: Dict[str, float], fraction: Optional[float],
                phase_active: float, waiting: bool) -> Optional[int]:
        audio = entry.audio_seconds
        if waiting or not audio:
            return None
        phase = entry.phase
        # How fast this request ran so far, against the expectation (machine busy, other GPU users).
        # Only phases that really ran (a skipped one has no active time) and never the client's upload.
        done_phases = [name for name in entry.phases[: entry.phase_index]
                       if name != "upload" and name in entry.active_s]
        measured = sum(entry.active_s.get(name, 0.0) for name in done_phases)
        planned = sum(expected[name] * audio / 1000.0 for name in done_phases)
        speed = min(3.0, max(0.5, measured / planned)) if planned > 1.0 and measured > 0 else 1.0
        later = sum(expected[name] for name in entry.phases[entry.phase_index + 1:]) * audio / 1000.0 * speed
        if fraction is not None and (fraction >= RATE_MIN_FRACTION or phase_active >= RATE_MIN_ACTIVE_S) \
                and fraction > 0:
            current = phase_active * (1.0 - fraction) / fraction
        else:
            current_expected = expected[phase] * audio / 1000.0 * speed
            if phase_active > current_expected:
                return None
            current = current_expected - phase_active
        return round((current + later) * 1000)


registry = ProgressRegistry()
_current: ContextVar[Optional[ProgressEntry]] = ContextVar("g3_progress_entry", default=None)


def current_entry() -> Optional[ProgressEntry]:
    return _current.get()


def bind(entry: Optional[ProgressEntry]) -> Any:
    """Make ``entry`` the current request's entry (tasks and threads started from here inherit it); returns the
    token for ``unbind``."""

    return _current.set(entry)


def unbind(token: Any) -> None:
    _current.reset(token)


@contextmanager
def tracking(entry: Optional[ProgressEntry]) -> Iterator[Optional[ProgressEntry]]:
    token = bind(entry)
    try:
        yield entry
    finally:
        unbind(token)


def enter_phase(phase: str, *, total: Optional[int] = None, probe: Optional[Probe] = None) -> None:
    entry = _current.get()
    if entry is not None:
        registry.enter_phase(entry, phase, total=total, probe=probe)


def add_units(*, done: int = 0, total: int = 0) -> None:
    entry = _current.get()
    if entry is not None:
        registry.add_units(entry, done=done, total=total)


def begin_gpu_wait() -> Optional[ProgressEntry]:
    """Mark the current request as waiting for a GPU lock someone else holds; pass the result to end_gpu_wait."""

    entry = _current.get()
    if entry is not None:
        registry.begin_wait(entry)
    return entry


def end_gpu_wait(entry: Optional[ProgressEntry]) -> None:
    if entry is not None:
        registry.end_wait(entry)


def capability() -> Dict[str, Any]:
    return {"supported": True, "header": "X-Progress-ID", "path": "/v2/progress/{progress_id}",
            "ttl_seconds": int(FINISHED_TTL_S)}


__all__ = [
    "PROGRESS_HEADER",
    "ProgressRegistry",
    "add_units",
    "bind",
    "capability",
    "current_entry",
    "enter_phase",
    "registry",
    "begin_gpu_wait",
    "end_gpu_wait",
    "tracking",
    "unbind",
]
