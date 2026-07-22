"""Conservative, vector-only correction of DIA speaker turns.

The refinement pass deliberately reuses the time-coded ReDimNet windows that
were already extracted for speaker matching.  It never embeds audio and it
never changes turn boundaries.  Frozen, well-supported speaker prototypes vote
on suspicious windows; a whole original exclusive turn is reassigned only when
the evidence is deliberately stronger than the ordinary identity thresholds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np

from .genesis_whisper_server_speaker_matching import SpeakerCloud, build_robust_cloud
from .genesis_whisper_server_vid import EmbeddedVoiceWindow, REDIMNET_EMBEDDING_DIMENSION


SpeakerRefinementMode = Literal["off", "shadow", "conservative"]

SEED_PURITY_MIN = 0.70
SEED_INLIER_COUNT_MIN = 3
SEED_CLEAN_SECONDS_MIN = 6.0
CANDIDATE_OWN_COSINE_MAX = 0.55
WINDOW_QUALITY_MIN = 0.60
WINDOW_CLEAN_SECONDS_MIN = 2.0
TARGET_COSINE_MIN = 0.65
TARGET_GAIN_MIN = 0.10
TARGET_RUNNER_UP_MARGIN_MIN = 0.06
TURN_SUPPORTING_WINDOWS_MIN = 2
TURN_EVIDENCE_SECONDS_MIN = 4.0
TURN_WEIGHTED_VOTE_SHARE_MIN = 0.75
TURN_EVIDENCE_COVERAGE_MIN = 0.60
SHORT_TURN_TARGET_COSINE_MIN = 0.75
SHORT_TURN_GAIN_MIN = 0.15
SHORT_TURN_RUNNER_UP_MARGIN_MIN = 0.10
SHORT_TURN_MAX_MS = 4000
BOUNDARY_MARGIN_MS = 200
MAX_REASSIGNED_SPEECH_SHARE = 0.10
MAX_PUBLIC_CHANGES = 100


@dataclass(frozen=True)
class SpeakerRefinementResult:
    """Refined turn copies, the effective clouds, and public diagnostics."""

    turns: list[dict[str, Any]]
    speaker_clouds: dict[str, SpeakerCloud]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _WindowEvidence:
    source_speaker_id: str
    sample_index: int
    sample: EmbeddedVoiceWindow
    turn_index: int
    vector: np.ndarray
    own_cosine: float
    weight: float


@dataclass(frozen=True)
class _WindowVote:
    evidence: _WindowEvidence
    target_speaker_id: str
    target_cosine: float
    similarity_gain: float
    runner_up_margin: float


def _normalized_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(vector) != REDIMNET_EMBEDDING_DIMENSION or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return (vector / norm).astype(np.float32, copy=False)


def _copy_original_turns(
    exclusive_segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for index, item in enumerate(exclusive_segments):
        speaker_id = str(item.get("speaker_id") or "").strip()
        try:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Ungueltiger Exclusive-Turn an Position {index}.") from exc
        if not speaker_id or start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"Ungueltiger Exclusive-Turn an Position {index}.")
        turn = dict(item)
        turn["start_ms"] = start_ms
        turn["end_ms"] = end_ms
        turn["speaker_id"] = speaker_id
        # Provenance is intentionally reset from the current raw DIA label.
        # Callers must pass the original, unmerged exclusive timeline.
        turn["original_speaker_id"] = speaker_id
        turns.append(turn)
    return turns


def _intervals_from_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    speaker_id: str | None = None,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for item in segments:
        if speaker_id is not None and str(item.get("speaker_id") or "") != speaker_id:
            continue
        try:
            start_ms = int(item.get("start_ms", 0))
            end_ms = int(item.get("end_ms", 0))
        except (TypeError, ValueError):
            continue
        if end_ms > start_ms:
            intervals.append((start_ms, end_ms))
    return _merge_intervals(intervals)


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in sorted(intervals):
        if end_ms <= start_ms:
            continue
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def _overlaps_any(start_ms: int, end_ms: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(other_end > start_ms and other_start < end_ms for other_start, other_end in intervals)


def _find_unique_containing_turn(
    turns: Sequence[Mapping[str, Any]],
    speaker_id: str,
    start_ms: int,
    end_ms: int,
) -> int | None:
    matches = [
        index
        for index, turn in enumerate(turns)
        if str(turn["speaker_id"]) == speaker_id
        and int(turn["start_ms"]) <= start_ms
        and int(turn["end_ms"]) >= end_ms
    ]
    return matches[0] if len(matches) == 1 else None


def _stable_seed_indices(cloud: SpeakerCloud) -> list[int]:
    if cloud.status != "ready" or cloud.prototype is None:
        return []
    if cloud.purity is None or not math.isfinite(cloud.purity) or cloud.purity < SEED_PURITY_MIN:
        return []
    indices = [
        index
        for index in cloud.inlier_indices
        if 0 <= index < len(cloud.samples)
        and not cloud.samples[index].stitched
        and math.isfinite(cloud.samples[index].quality)
        and math.isfinite(cloud.samples[index].clean_duration_seconds)
        and cloud.samples[index].clean_duration_seconds > 0.0
    ]
    clean_seconds = sum(cloud.samples[index].clean_duration_seconds for index in indices)
    if len(indices) < SEED_INLIER_COUNT_MIN or clean_seconds < SEED_CLEAN_SECONDS_MIN:
        return []
    return indices


def _stable_seed_prototype(
    cloud: SpeakerCloud,
    indices: Sequence[int],
) -> np.ndarray | None:
    """Build a frozen target only from the qualifying non-stitched seed core."""

    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for index in indices:
        vector = _normalized_vector(cloud.samples[index].vector)
        if vector is None:
            return None
        vectors.append(vector)
        sample = cloud.samples[index]
        weights.append(max(float(sample.quality) * float(sample.clean_duration_seconds), 1e-6))
    if not vectors:
        return None
    weighted_sum = np.sum(
        np.stack(vectors, axis=0) * np.asarray(weights, dtype=np.float32)[:, None],
        axis=0,
        dtype=np.float64,
    )
    return _normalized_vector(weighted_sum)


def _coverage_ms(samples: Sequence[EmbeddedVoiceWindow], start_ms: int, end_ms: int) -> int:
    intervals = [
        (max(start_ms, int(sample.start_ms)), min(end_ms, int(sample.end_ms)))
        for sample in samples
        if sample.start_ms is not None
        and sample.end_ms is not None
        and int(sample.end_ms) > start_ms
        and int(sample.start_ms) < end_ms
    ]
    return sum(end - start for start, end in _merge_intervals(intervals))


def _fully_covers_safe_short_turn(
    samples: Sequence[EmbeddedVoiceWindow],
    start_ms: int,
    end_ms: int,
) -> bool:
    safe_start = min(end_ms, start_ms + BOUNDARY_MARGIN_MS)
    safe_end = max(start_ms, end_ms - BOUNDARY_MARGIN_MS)
    if safe_end <= safe_start:
        safe_start, safe_end = start_ms, end_ms
    return _coverage_ms(samples, safe_start, safe_end) >= safe_end - safe_start


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _build_proposals(
    turns: Sequence[Mapping[str, Any]],
    standard_segments: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    speaker_clouds: Mapping[str, SpeakerCloud],
) -> tuple[list[dict[str, Any]], int, dict[tuple[str, int], int]]:
    overlap_intervals = _intervals_from_segments(overlaps)
    standard_by_speaker = {
        speaker_id: _intervals_from_segments(standard_segments, speaker_id=speaker_id)
        for speaker_id in sorted(speaker_clouds)
    }
    seed_prototypes: dict[str, np.ndarray] = {}
    own_prototypes: dict[str, np.ndarray] = {}
    for speaker_id, cloud in sorted(speaker_clouds.items()):
        prototype = _normalized_vector(cloud.prototype)
        if prototype is not None:
            own_prototypes[speaker_id] = prototype
        indices = _stable_seed_indices(cloud)
        if indices and prototype is not None:
            seed_prototype = _stable_seed_prototype(cloud, indices)
            if seed_prototype is not None:
                seed_prototypes[speaker_id] = seed_prototype

    mapped_turns: dict[tuple[str, int], int] = {}
    evidence_by_turn: dict[int, list[_WindowEvidence]] = {}
    votes_by_turn: dict[int, list[_WindowVote]] = {}
    eligible_windows = 0

    for source_speaker_id, cloud in sorted(speaker_clouds.items()):
        own_prototype = own_prototypes.get(source_speaker_id)
        if own_prototype is None:
            continue
        inlier_indices = set(cloud.inlier_indices)
        for sample_index, sample in enumerate(cloud.samples):
            if sample.stitched or sample.start_ms is None or sample.end_ms is None:
                continue
            start_ms = int(sample.start_ms)
            end_ms = int(sample.end_ms)
            if end_ms <= start_ms:
                continue
            turn_index = _find_unique_containing_turn(
                turns,
                source_speaker_id,
                start_ms,
                end_ms,
            )
            if turn_index is None:
                continue
            mapped_turns[(source_speaker_id, sample_index)] = turn_index
            if (
                not math.isfinite(sample.quality)
                or not math.isfinite(sample.clean_duration_seconds)
                or sample.quality < WINDOW_QUALITY_MIN
                or sample.clean_duration_seconds < WINDOW_CLEAN_SECONDS_MIN
                or _overlaps_any(start_ms, end_ms, overlap_intervals)
            ):
                continue
            vector = _normalized_vector(sample.vector)
            if vector is None:
                continue
            own_cosine = float(vector @ own_prototype)
            evidence = _WindowEvidence(
                source_speaker_id=source_speaker_id,
                sample_index=sample_index,
                sample=sample,
                turn_index=turn_index,
                vector=vector,
                own_cosine=own_cosine,
                weight=max(float(sample.quality) * float(sample.clean_duration_seconds), 1e-6),
            )
            evidence_by_turn.setdefault(turn_index, []).append(evidence)
            eligible_windows += 1

            if sample_index in inlier_indices and own_cosine >= CANDIDATE_OWN_COSINE_MAX:
                continue
            turn = turns[turn_index]
            target_scores: list[tuple[float, str]] = []
            for target_speaker_id, target_prototype in seed_prototypes.items():
                if target_speaker_id == source_speaker_id:
                    continue
                # A label that DIA says is active anywhere in this original
                # exclusive turn cannot safely replace the turn's owner.
                if _overlaps_any(
                    int(turn["start_ms"]),
                    int(turn["end_ms"]),
                    standard_by_speaker.get(target_speaker_id, []),
                ):
                    continue
                target_scores.append((float(vector @ target_prototype), target_speaker_id))
            if not target_scores:
                continue
            target_scores.sort(key=lambda item: (-item[0], item[1]))
            target_cosine, target_speaker_id = target_scores[0]
            runner_up_cosine = target_scores[1][0] if len(target_scores) > 1 else -1.0
            similarity_gain = target_cosine - own_cosine
            runner_up_margin = target_cosine - runner_up_cosine
            if (
                target_cosine < TARGET_COSINE_MIN
                or similarity_gain < TARGET_GAIN_MIN
                or runner_up_margin < TARGET_RUNNER_UP_MARGIN_MIN
            ):
                continue
            votes_by_turn.setdefault(turn_index, []).append(
                _WindowVote(
                    evidence=evidence,
                    target_speaker_id=target_speaker_id,
                    target_cosine=target_cosine,
                    similarity_gain=similarity_gain,
                    runner_up_margin=runner_up_margin,
                )
            )

    proposals: list[dict[str, Any]] = []
    for turn_index in sorted(votes_by_turn):
        turn = turns[turn_index]
        turn_votes = votes_by_turn[turn_index]
        evidence = evidence_by_turn.get(turn_index, [])
        if not evidence:
            continue
        vote_weights: dict[str, float] = {}
        for vote in turn_votes:
            vote_weights[vote.target_speaker_id] = (
                vote_weights.get(vote.target_speaker_id, 0.0) + vote.evidence.weight
            )
        target_speaker_id = min(
            vote_weights,
            key=lambda speaker_id: (-vote_weights[speaker_id], speaker_id),
        )
        supporting = [vote for vote in turn_votes if vote.target_speaker_id == target_speaker_id]
        if any(vote.target_speaker_id != target_speaker_id for vote in turn_votes):
            # A second independently valid target is strong contradictory
            # evidence even if its accumulated weight is below 25 percent.
            continue

        target_prototype = seed_prototypes[target_speaker_id]
        if any(
            item.own_cosine >= TARGET_COSINE_MIN
            and item.own_cosine - float(item.vector @ target_prototype)
            >= TARGET_RUNNER_UP_MARGIN_MIN
            for item in evidence
        ):
            continue

        total_weight = sum(item.weight for item in evidence)
        supporting_weight = sum(vote.evidence.weight for vote in supporting)
        weighted_vote_share = supporting_weight / total_weight if total_weight > 0.0 else 0.0
        support_samples = [vote.evidence.sample for vote in supporting]
        turn_start_ms = int(turn["start_ms"])
        turn_end_ms = int(turn["end_ms"])
        turn_duration_ms = turn_end_ms - turn_start_ms
        evidence_duration_seconds = sum(
            min(
                float(vote.evidence.sample.clean_duration_seconds),
                max(0, int(vote.evidence.sample.end_ms) - int(vote.evidence.sample.start_ms))
                / 1000.0,
            )
            for vote in supporting
        )
        evidence_duration_ms = round(evidence_duration_seconds * 1000)
        evidence_coverage = _coverage_ms(
            support_samples,
            turn_start_ms,
            turn_end_ms,
        ) / max(1, turn_duration_ms)

        regular = (
            len(supporting) >= TURN_SUPPORTING_WINDOWS_MIN
            and evidence_duration_seconds >= TURN_EVIDENCE_SECONDS_MIN
            and weighted_vote_share >= TURN_WEIGHTED_VOTE_SHARE_MIN
            and evidence_coverage >= TURN_EVIDENCE_COVERAGE_MIN
        )
        short_exception = (
            not regular
            and turn_duration_ms < SHORT_TURN_MAX_MS
            and len(supporting) == 1
            and weighted_vote_share >= TURN_WEIGHTED_VOTE_SHARE_MIN
            and _fully_covers_safe_short_turn(support_samples, turn_start_ms, turn_end_ms)
            and supporting[0].target_cosine >= SHORT_TURN_TARGET_COSINE_MIN
            and supporting[0].similarity_gain >= SHORT_TURN_GAIN_MIN
            and supporting[0].runner_up_margin >= SHORT_TURN_RUNNER_UP_MARGIN_MIN
        )
        if not regular and not short_exception:
            continue

        metric_weight = sum(vote.evidence.weight for vote in supporting)
        def weighted_metric(attribute: str) -> float:
            return sum(
                getattr(vote, attribute) * vote.evidence.weight for vote in supporting
            ) / metric_weight

        proposals.append(
            {
                "turn_index": turn_index,
                "start_ms": turn_start_ms,
                "end_ms": turn_end_ms,
                "from_speaker_id": str(turn["speaker_id"]),
                "to_speaker_id": target_speaker_id,
                "supporting_windows": len(supporting),
                "evidence_duration_ms": evidence_duration_ms,
                "evidence_coverage": _round_metric(evidence_coverage),
                "weighted_vote_share": _round_metric(weighted_vote_share),
                "target_cosine": _round_metric(weighted_metric("target_cosine")),
                "own_cosine": _round_metric(
                    sum(vote.evidence.own_cosine * vote.evidence.weight for vote in supporting)
                    / metric_weight
                ),
                "similarity_gain": _round_metric(weighted_metric("similarity_gain")),
                "runner_up_margin": _round_metric(weighted_metric("runner_up_margin")),
                "short_turn_exception": short_exception,
                "applied": False,
            }
        )

    proposals.sort(
        key=lambda change: (
            change["start_ms"],
            change["end_ms"],
            change["from_speaker_id"],
            change["to_speaker_id"],
            change["turn_index"],
        )
    )
    return proposals, eligible_windows, mapped_turns


def _rebuild_clouds(
    original_clouds: Mapping[str, SpeakerCloud],
    labels: Sequence[str],
    mapped_turns: Mapping[tuple[str, int], int],
) -> tuple[dict[str, SpeakerCloud], dict[tuple[str, int], str]]:
    speaker_ids = sorted(set(original_clouds) | set(labels))
    samples_by_speaker: dict[str, list[EmbeddedVoiceWindow]] = {
        speaker_id: [] for speaker_id in speaker_ids
    }
    effective_sample_labels: dict[tuple[str, int], str] = {}
    for source_speaker_id, cloud in sorted(original_clouds.items()):
        for sample_index, sample in enumerate(cloud.samples):
            turn_index = mapped_turns.get((source_speaker_id, sample_index))
            target_speaker_id = (
                labels[turn_index]
                if turn_index is not None and 0 <= turn_index < len(labels)
                else source_speaker_id
            )
            samples_by_speaker.setdefault(target_speaker_id, []).append(sample)
            effective_sample_labels[(source_speaker_id, sample_index)] = target_speaker_id
    for samples in samples_by_speaker.values():
        # ``build_robust_cloud`` consumes samples as a deterministic stream.
        # Reassignment can combine several source clouds, so restore the same
        # chronological ordering produced by normal extraction first.
        samples.sort(
            key=lambda sample: (
                sample.start_ms is None,
                int(sample.start_ms) if sample.start_ms is not None else 0,
                int(sample.end_ms) if sample.end_ms is not None else 0,
            )
        )
    rebuilt = {
        speaker_id: build_robust_cloud(speaker_id, samples_by_speaker.get(speaker_id, []))
        for speaker_id in speaker_ids
    }
    return rebuilt, effective_sample_labels


def _rollback_reason(
    original_turns: Sequence[Mapping[str, Any]],
    refined_turns: Sequence[Mapping[str, Any]],
    original_clouds: Mapping[str, SpeakerCloud],
    rebuilt_clouds: Mapping[str, SpeakerCloud],
    effective_sample_labels: Mapping[tuple[str, int], str],
    stable_seed_indices: Mapping[str, Sequence[int]],
    standard_segments: Sequence[Mapping[str, Any]],
) -> str | None:
    if len(original_turns) != len(refined_turns) or any(
        int(original["start_ms"]) != int(refined["start_ms"])
        or int(original["end_ms"]) != int(refined["end_ms"])
        for original, refined in zip(original_turns, refined_turns)
    ):
        return "timeline_invariant_violation"

    original_speakers = {str(turn["speaker_id"]) for turn in original_turns}
    refined_speakers = {str(turn["speaker_id"]) for turn in refined_turns}
    if original_speakers != refined_speakers:
        return "speaker_count_changed"

    total_duration_ms = sum(int(turn["end_ms"]) - int(turn["start_ms"]) for turn in original_turns)
    reassigned_duration_ms = sum(
        int(original["end_ms"]) - int(original["start_ms"])
        for original, refined in zip(original_turns, refined_turns)
        if str(original["speaker_id"]) != str(refined["speaker_id"])
    )
    if total_duration_ms <= 0 or reassigned_duration_ms > total_duration_ms * MAX_REASSIGNED_SPEECH_SHARE:
        return "reassigned_speech_exceeds_10_percent"

    # This repeats the proposal-time collision gate as a fail-closed invariant
    # after all synchronous changes have been assembled.
    standard_by_speaker = {
        speaker_id: _intervals_from_segments(standard_segments, speaker_id=speaker_id)
        for speaker_id in refined_speakers
    }
    for original, refined in zip(original_turns, refined_turns):
        if str(original["speaker_id"]) == str(refined["speaker_id"]):
            continue
        if _overlaps_any(
            int(refined["start_ms"]),
            int(refined["end_ms"]),
            standard_by_speaker.get(str(refined["speaker_id"]), []),
        ):
            return "overlap_invariant_violation"

    for speaker_id, indices in sorted(stable_seed_indices.items()):
        retained = [
            index
            for index in indices
            if effective_sample_labels.get((speaker_id, index), speaker_id) == speaker_id
        ]
        retained_seconds = sum(original_clouds[speaker_id].samples[index].clean_duration_seconds for index in retained)
        if len(retained) < SEED_INLIER_COUNT_MIN or retained_seconds < SEED_CLEAN_SECONDS_MIN:
            return "seed_core_lost"

    for speaker_id, original_cloud in sorted(original_clouds.items()):
        rebuilt = rebuilt_clouds.get(speaker_id)
        if original_cloud.status == "ready" and rebuilt is not None and rebuilt.status == "mixed_cluster":
            return "ready_cloud_became_mixed_cluster"
    return None


def _diagnostics(
    *,
    mode: SpeakerRefinementMode,
    status: str,
    eligible_windows: int,
    proposals: Sequence[Mapping[str, Any]],
    applied: bool,
    reassigned_duration_ms: int,
    started: float,
    rollback_reason: str | None,
) -> dict[str, Any]:
    changes = [dict(change, applied=applied) for change in proposals[:MAX_PUBLIC_CHANGES]]
    return {
        "mode": mode,
        "status": status,
        "eligible_windows": eligible_windows,
        "proposed_turns": len(proposals),
        "applied_turns": len(proposals) if applied else 0,
        "reassigned_duration_ms": reassigned_duration_ms if applied else 0,
        "processing_ms": round((time.perf_counter() - started) * 1000),
        "rollback_reason": rollback_reason,
        "changes": changes,
        "changes_truncated": len(proposals) > MAX_PUBLIC_CHANGES,
    }


def refine_speaker_turns(
    mode: str,
    exclusive_segments: Sequence[Mapping[str, Any]],
    standard_segments: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
    speaker_clouds: Mapping[str, SpeakerCloud],
) -> SpeakerRefinementResult:
    """Return a deterministic one-pass DIA label refinement.

    ``exclusive_segments`` must be DIA's original, unmerged exclusive turns.
    ``speaker_clouds`` must contain the already-computed ReDimNet windows from
    the standard diarization.  No model call occurs in this function.
    """

    started = time.perf_counter()
    if mode not in {"off", "shadow", "conservative"}:
        raise ValueError("speaker_refinement muss 'off', 'shadow' oder 'conservative' sein.")
    typed_mode = cast(SpeakerRefinementMode, mode)
    original_turns = _copy_original_turns(exclusive_segments)
    original_clouds = dict(speaker_clouds)

    if mode == "off":
        return SpeakerRefinementResult(
            turns=original_turns,
            speaker_clouds=original_clouds,
            diagnostics=_diagnostics(
                mode=typed_mode,
                status="disabled",
                eligible_windows=0,
                proposals=[],
                applied=False,
                reassigned_duration_ms=0,
                started=started,
                rollback_reason=None,
            ),
        )

    proposals, eligible_windows, mapped_turns = _build_proposals(
        original_turns,
        standard_segments,
        overlaps,
        original_clouds,
    )
    if not proposals:
        return SpeakerRefinementResult(
            turns=original_turns,
            speaker_clouds=original_clouds,
            diagnostics=_diagnostics(
                mode=typed_mode,
                status="not_needed",
                eligible_windows=eligible_windows,
                proposals=[],
                applied=False,
                reassigned_duration_ms=0,
                started=started,
                rollback_reason=None,
            ),
        )

    if mode == "shadow":
        return SpeakerRefinementResult(
            turns=original_turns,
            speaker_clouds=original_clouds,
            diagnostics=_diagnostics(
                mode=typed_mode,
                status="shadow",
                eligible_windows=eligible_windows,
                proposals=proposals,
                applied=False,
                reassigned_duration_ms=0,
                started=started,
                rollback_reason=None,
            ),
        )

    labels = [str(turn["speaker_id"]) for turn in original_turns]
    for proposal in proposals:
        labels[int(proposal["turn_index"])] = str(proposal["to_speaker_id"])
    refined_turns = [dict(turn, speaker_id=labels[index]) for index, turn in enumerate(original_turns)]
    rebuilt_clouds, effective_sample_labels = _rebuild_clouds(
        original_clouds,
        labels,
        mapped_turns,
    )
    stable_seed_indices = {
        speaker_id: indices
        for speaker_id, cloud in sorted(original_clouds.items())
        if (indices := _stable_seed_indices(cloud))
    }
    rollback_reason = _rollback_reason(
        original_turns,
        refined_turns,
        original_clouds,
        rebuilt_clouds,
        effective_sample_labels,
        stable_seed_indices,
        standard_segments,
    )
    if rollback_reason is not None:
        return SpeakerRefinementResult(
            turns=original_turns,
            speaker_clouds=original_clouds,
            diagnostics=_diagnostics(
                mode=typed_mode,
                status="rejected",
                eligible_windows=eligible_windows,
                proposals=proposals,
                applied=False,
                reassigned_duration_ms=0,
                started=started,
                rollback_reason=rollback_reason,
            ),
        )

    reassigned_duration_ms = sum(
        int(turn["end_ms"]) - int(turn["start_ms"])
        for turn, label in zip(original_turns, labels)
        if str(turn["speaker_id"]) != label
    )
    return SpeakerRefinementResult(
        turns=refined_turns,
        speaker_clouds=rebuilt_clouds,
        diagnostics=_diagnostics(
            mode=typed_mode,
            status="applied",
            eligible_windows=eligible_windows,
            proposals=proposals,
            applied=True,
            reassigned_duration_ms=reassigned_duration_ms,
            started=started,
            rollback_reason=None,
        ),
    )


__all__ = [
    "SpeakerRefinementMode",
    "SpeakerRefinementResult",
    "refine_speaker_turns",
]
