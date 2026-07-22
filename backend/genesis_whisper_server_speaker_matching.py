"""Robust ReDimNet2 clouds and global known-speaker assignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .genesis_whisper_server_vid import (
    EmbeddedVoiceWindow,
    REDIMNET_EMBEDDING_DIMENSION,
    REDIMNET_SAMPLE_RATE,
    REDIMNET_WINDOW_SAMPLES,
    VoiceWindow,
    embed_voice_windows,
    iter_windows_from_audio,
)


COMPONENT_COSINE_MIN = 0.45
DOMINANT_WEIGHT_SHARE_MIN = 0.60
DOMINANT_SECOND_RATIO_MIN = 1.50
MATCH_COSINE_MIN = 0.60
MATCH_SUPPORT_MIN = 0.60
MATCH_STABILITY_MARGIN_MIN = 0.04
SAMPLE_SUPPORT_COSINE_MIN = 0.50
MATCH_DUMMY_SCORE = 0.55
TENTATIVE_DUMMY_SCORE = COMPONENT_COSINE_MIN - 1e-6
INVALID_ASSIGNMENT_SCORE = -1e6
NEAR_DUPLICATE_COSINE = 0.995
MAX_RETURNED_EMBEDDINGS = 64
MAX_STREAMING_COMPONENTS = 256
MAX_REPRESENTATIVE_CANDIDATES = 512
MATCH_SIMILARITY_BLOCK_SIZE = 256
MIN_CLEAN_REGION_MS = 2000
BOUNDARY_MARGIN_MS = 200


class SpeakerProfileValidationError(ValueError):
    pass


@dataclass
class SpeakerCloud:
    diarization_speaker_id: str
    status: str
    prototype: np.ndarray | None
    samples: list[EmbeddedVoiceWindow]
    inlier_indices: list[int]
    candidate_count: int
    discarded_outliers: int
    purity: float | None

    @property
    def inliers(self) -> list[EmbeddedVoiceWindow]:
        return [self.samples[index] for index in self.inlier_indices]

    def public_embeddings(self) -> list[dict[str, Any]]:
        if self.prototype is None:
            return []
        response: list[dict[str, Any]] = [
            {
                "kind": "prototype",
                "vector": self.prototype.astype(float).tolist(),
            }
        ]
        representatives = _select_representatives(self.inliers, MAX_RETURNED_EMBEDDINGS - 1)
        for item in representatives:
            response.append(
                {
                    "kind": "representative",
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "clean_duration_seconds": round(item.clean_duration_seconds, 3),
                    "quality": round(item.quality, 4),
                    "stitched": item.stitched,
                    **(
                        {
                            "source_spans": [
                                {"start_ms": start_ms, "end_ms": end_ms}
                                for start_ms, end_ms in item.source_spans
                            ]
                        }
                        if item.source_spans
                        else {}
                    ),
                    "vector": item.vector.astype(float).tolist(),
                }
            )
        return response


def _l2_normalize(vector: Sequence[float]) -> np.ndarray:
    if not isinstance(vector, (list, tuple, np.ndarray)):
        raise SpeakerProfileValidationError("Embedding muss ein flaches Array numerischer Werte sein.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
        raise SpeakerProfileValidationError("Embedding enthaelt nicht-numerische Werte.")
    try:
        array = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpeakerProfileValidationError("Embedding enthaelt nicht-numerische Werte.") from exc
    if array.ndim != 1:
        raise SpeakerProfileValidationError("Embedding muss ein flaches Array sein.")
    array = array.reshape(-1)
    if len(array) != REDIMNET_EMBEDDING_DIMENSION:
        raise SpeakerProfileValidationError(
            f"Embedding muss genau {REDIMNET_EMBEDDING_DIMENSION} Werte enthalten."
        )
    if not np.all(np.isfinite(array)):
        raise SpeakerProfileValidationError("Embedding enthaelt nicht-endliche Werte.")
    scale = float(np.max(np.abs(array)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise SpeakerProfileValidationError("Embedding darf kein Nullvektor sein.")
    scaled = array / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if not np.isfinite(scaled_norm) or scaled_norm <= 1e-12:
        raise SpeakerProfileValidationError("Embedding darf kein Nullvektor sein.")
    normalized = (scaled / scaled_norm).astype(np.float32)
    if not np.all(np.isfinite(normalized)) or float(np.linalg.norm(normalized.astype(np.float64))) <= 1e-12:
        raise SpeakerProfileValidationError("Embedding konnte nicht stabil normalisiert werden.")
    return normalized


def validate_known_speakers(known_speakers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for profile in known_speakers:
        speaker_id = str(profile.get("id") or "").strip()
        if not speaker_id or len(speaker_id.encode("utf-8")) > 128:
            raise SpeakerProfileValidationError("Bekannte Sprecher-ID ist leer oder laenger als 128 Bytes.")
        if speaker_id in seen_ids:
            raise SpeakerProfileValidationError(f"Sprecher-ID '{speaker_id}' wurde mehrfach geliefert.")
        seen_ids.add(speaker_id)
        raw_embeddings = profile.get("embeddings")
        if not isinstance(raw_embeddings, list) or not raw_embeddings:
            raise SpeakerProfileValidationError(
                f"Sprecher '{speaker_id}' benoetigt mindestens ein Embedding."
            )
        vectors = [_l2_normalize(vector) for vector in raw_embeddings]
        validated.append({"id": speaker_id, "vectors": vectors})
    return validated


def _weighted_mean(samples: Sequence[EmbeddedVoiceWindow], indices: Sequence[int]) -> np.ndarray:
    matrix = np.stack([samples[index].vector for index in indices], axis=0)
    weights = np.asarray(
        [max(samples[index].quality * samples[index].clean_duration_seconds, 1e-6) for index in indices],
        dtype=np.float32,
    )
    mean = np.average(matrix, axis=0, weights=weights)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("Embedding-Prototype konnte nicht gebildet werden.")
    return (mean / norm).astype(np.float32, copy=False)


def _clean_component_inliers(
    samples: Sequence[EmbeddedVoiceWindow],
    indices: Sequence[int],
) -> tuple[list[int], np.ndarray]:
    """Apply the iterative median/MAD filter to one cosine component."""

    retained = list(indices)
    for _ in range(2):
        if len(retained) <= 2:
            break
        prototype = _weighted_mean(samples, retained)
        similarities = np.asarray([float(samples[index].vector @ prototype) for index in retained])
        median = float(np.median(similarities))
        mad = float(np.median(np.abs(similarities - median)))
        threshold = max(COMPONENT_COSINE_MIN, median - 3.0 * 1.4826 * mad)
        next_retained = [index for index, similarity in zip(retained, similarities) if similarity >= threshold]
        if not next_retained or next_retained == retained:
            break
        retained = next_retained
    return retained, _weighted_mean(samples, retained)


def _streaming_components(
    samples: Sequence[EmbeddedVoiceWindow],
) -> tuple[list[list[int]], bool]:
    """Group every sample with bounded work and memory per input sample.

    Components are represented by their quality/duration-weighted online
    prototypes.  If more unrelated components are required than the bounded
    tracker can represent, ``capacity_exceeded`` is returned.  Callers must
    then fail closed instead of deriving an identity from a lossy partition.
    """

    components: list[list[int]] = []
    # Fixed matrices avoid rebuilding ``np.stack(prototypes)`` for every
    # candidate.  Long recordings normally have thousands of windows, while
    # the number of plausible components is deliberately bounded.
    component_capacity = min(MAX_STREAMING_COMPONENTS, max(1, len(samples)))
    weighted_sums = np.zeros(
        (component_capacity, REDIMNET_EMBEDDING_DIMENSION),
        dtype=np.float64,
    )
    prototypes = np.zeros_like(weighted_sums)
    capacity_exceeded = False

    for index, sample in enumerate(samples):
        weight = max(sample.quality * sample.clean_duration_seconds, 1e-6)
        component_count = len(components)
        if component_count:
            similarities = prototypes[:component_count] @ sample.vector
            component_index = int(np.argmax(similarities))
            best_similarity = float(similarities[component_index])
        else:
            component_index = -1
            best_similarity = -1.0

        if best_similarity < COMPONENT_COSINE_MIN:
            if len(components) >= component_capacity:
                # The sample was still inspected, but assigning it to an
                # unrelated component would fabricate evidence.  Mark the
                # whole cloud unusable and continue scanning deterministically.
                capacity_exceeded = True
                continue
            components.append([index])
            weighted_sum = sample.vector.astype(np.float64) * weight
            new_index = len(components) - 1
            weighted_sums[new_index] = weighted_sum
            prototypes[new_index] = sample.vector
            continue

        components[component_index].append(index)
        weighted_sums[component_index] += sample.vector.astype(np.float64) * weight
        norm = float(np.linalg.norm(weighted_sums[component_index]))
        if norm <= 1e-12 or not np.isfinite(norm):
            capacity_exceeded = True
            continue
        prototypes[component_index] = weighted_sums[component_index] / norm

    return components, capacity_exceeded


def build_robust_cloud(
    speaker_id: str,
    samples: Sequence[EmbeddedVoiceWindow],
) -> SpeakerCloud:
    samples = list(samples)
    candidate_count = len(samples)
    if not samples:
        return SpeakerCloud(
            diarization_speaker_id=speaker_id,
            status="insufficient_clean_speech",
            prototype=None,
            samples=[],
            inlier_indices=[],
            candidate_count=candidate_count,
            discarded_outliers=0,
            purity=None,
        )

    if len(samples) == 1:
        return SpeakerCloud(
            diarization_speaker_id=speaker_id,
            status="low_support",
            prototype=samples[0].vector.copy(),
            samples=samples,
            inlier_indices=[0],
            candidate_count=candidate_count,
            discarded_outliers=0,
            purity=1.0,
        )

    components, capacity_exceeded = _streaming_components(samples)
    sample_weights = np.asarray(
        [max(item.quality * item.clean_duration_seconds, 1e-6) for item in samples],
        dtype=np.float32,
    )
    weighted_components = sorted(
        ((float(np.sum(sample_weights[indices])), indices) for indices in components),
        key=lambda item: (-item[0], item[1][0]),
    )
    total_weight = float(np.sum(sample_weights))
    main_weight, main_indices = weighted_components[0]
    second_weight = weighted_components[1][0] if len(weighted_components) > 1 else 0.0
    purity = main_weight / total_weight if total_weight > 0 else 0.0
    dominant = (
        purity >= DOMINANT_WEIGHT_SHARE_MIN
        and (second_weight <= 0.0 or main_weight >= DOMINANT_SECOND_RATIO_MIN * second_weight)
    )
    retained, prototype = _clean_component_inliers(samples, main_indices)
    total_clean_seconds = sum(samples[index].clean_duration_seconds for index in retained)
    has_safe_support = len(retained) >= 2 and total_clean_seconds >= 4.0
    if capacity_exceeded:
        # Overflow means the component distribution is incomplete, so it can
        # never drive identification. A well-supported tracked component is
        # nevertheless safe to return for later enrollment/inspection.
        status = "mixed_cluster"
        if not has_safe_support:
            retained = []
            prototype = None
    elif not dominant:
        # Preserve the cleaned dominant component for future resolution while
        # refusing to claim that it represents the DIA cluster unambiguously.
        status = "mixed_cluster"
    else:
        status = "ready" if has_safe_support else "low_support"
    return SpeakerCloud(
        diarization_speaker_id=speaker_id,
        status=status,
        prototype=prototype,
        samples=samples,
        inlier_indices=retained,
        candidate_count=candidate_count,
        discarded_outliers=len(samples) - len(retained),
        purity=purity,
    )


def _select_representatives(
    samples: Sequence[EmbeddedVoiceWindow],
    limit: int,
) -> list[EmbeddedVoiceWindow]:
    if limit <= 0 or not samples:
        return []
    ordered = sorted(
        samples,
        key=lambda item: (
            item.start_ms is None,
            item.start_ms if item.start_ms is not None else 0,
            item.end_ms if item.end_ms is not None else 0,
        ),
    )
    # Deterministic reservoir sampling keeps work bounded even when a client
    # supplies thousands of windows.  Every candidate participates in the
    # sampling decision and the highest-quality candidate is always retained.
    reservoir: list[EmbeddedVoiceWindow] = []
    best_item = ordered[0]
    best_weight = max(best_item.quality * best_item.clean_duration_seconds, 1e-6)
    for index, item in enumerate(ordered):
        item_weight = max(item.quality * item.clean_duration_seconds, 1e-6)
        if item_weight > best_weight:
            best_item = item
            best_weight = item_weight
        if len(reservoir) < MAX_REPRESENTATIVE_CANDIDATES:
            reservoir.append(item)
            continue
        # SplitMix64 gives a stable, well-distributed slot without mutable RNG
        # state. This is classic reservoir sampling with deterministic entropy.
        mixed = (index + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        mixed ^= mixed >> 31
        slot = mixed % (index + 1)
        if slot < MAX_REPRESENTATIVE_CANDIDATES:
            reservoir[int(slot)] = item
    if all(item is not best_item for item in reservoir):
        reservoir[-1] = best_item

    # Near-duplicate removal is deliberately presentation-only.  The bounded
    # reservoir makes this quadratic-in-512 step constant with respect to the
    # number of supplied embeddings, while support and duration above retain
    # all original evidence.
    unique: list[EmbeddedVoiceWindow] = []
    for item in sorted(
        reservoir,
        key=lambda candidate: (
            -max(candidate.quality * candidate.clean_duration_seconds, 1e-6),
            candidate.start_ms is None,
            candidate.start_ms if candidate.start_ms is not None else 0,
        ),
    ):
        if unique and max(float(item.vector @ previous.vector) for previous in unique) >= NEAR_DUPLICATE_COSINE:
            continue
        unique.append(item)
    if len(unique) <= limit:
        return sorted(
            unique,
            key=lambda item: item.start_ms if item.start_ms is not None else -1,
        )

    weights = np.asarray(
        [max(item.quality * item.clean_duration_seconds, 1e-6) for item in unique],
        dtype=np.float32,
    )
    first = int(np.argmax(weights))
    selected = [first]
    min_distances = 1.0 - np.asarray([float(item.vector @ unique[first].vector) for item in unique])
    while len(selected) < limit:
        scores = min_distances * weights
        scores[selected] = -1.0
        next_index = int(np.argmax(scores))
        selected.append(next_index)
        distances = 1.0 - np.asarray(
            [float(item.vector @ unique[next_index].vector) for item in unique],
            dtype=np.float32,
        )
        min_distances = np.minimum(min_distances, distances)
    selected_items = [unique[index] for index in selected]
    return sorted(
        selected_items,
        key=lambda item: item.start_ms if item.start_ms is not None else -1,
    )


def _subtract_intervals(
    start_ms: int,
    end_ms: int,
    excluded: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    # ``excluded`` is sorted and unioned by the caller.  A cursor is both
    # simpler and avoids repeatedly rebuilding a growing pieces list for every
    # overlap in a long recording.
    pieces: list[tuple[int, int]] = []
    cursor = start_ms
    for excluded_start, excluded_end in excluded:
        if excluded_end <= cursor:
            continue
        if excluded_start >= end_ms:
            break
        if excluded_start > cursor:
            pieces.append((cursor, min(excluded_start, end_ms)))
        cursor = max(cursor, excluded_end)
        if cursor >= end_ms:
            break
    if cursor < end_ms:
        pieces.append((cursor, end_ms))
    return pieces


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


def _iter_stitched_windows(
    short_parts: Sequence[tuple[np.ndarray, int, int]],
) -> Iterable[VoiceWindow]:
    """Yield logical stitched windows without one speaker-sized concatenate."""

    minimum_samples = round(MIN_CLEAN_REGION_MS * REDIMNET_SAMPLE_RATE / 1000)
    part_index = 0
    part_position = 0
    while part_index < len(short_parts):
        pieces: list[np.ndarray] = []
        source_spans: list[tuple[int, int]] = []
        collected = 0
        while collected < REDIMNET_WINDOW_SAMPLES and part_index < len(short_parts):
            part, original_start_ms, _ = short_parts[part_index]
            available = len(part) - part_position
            if available <= 0:
                part_index += 1
                part_position = 0
                continue
            take = min(REDIMNET_WINDOW_SAMPLES - collected, available)
            pieces.append(part[part_position : part_position + take])
            source_start_ms = original_start_ms + round(
                part_position * 1000 / REDIMNET_SAMPLE_RATE
            )
            source_end_ms = original_start_ms + round(
                (part_position + take) * 1000 / REDIMNET_SAMPLE_RATE
            )
            source_spans.append((source_start_ms, source_end_ms))
            part_position += take
            collected += take
            if part_position >= len(part):
                part_index += 1
                part_position = 0

        if collected < minimum_samples:
            break
        chunk = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        for window in iter_windows_from_audio(
            chunk,
            start_ms=None,
            stitched=True,
            minimum_samples=minimum_samples,
        ):
            yield replace(window, source_spans=tuple(source_spans))


def _iter_speaker_windows(
    audio: np.ndarray,
    regions: Sequence[tuple[int, int]],
) -> Iterable[VoiceWindow]:
    minimum_samples = round(MIN_CLEAN_REGION_MS * REDIMNET_SAMPLE_RATE / 1000)
    short_parts: list[tuple[np.ndarray, int, int]] = []
    for start_ms, end_ms in regions:
        start_sample = round(start_ms * REDIMNET_SAMPLE_RATE / 1000)
        end_sample = round(end_ms * REDIMNET_SAMPLE_RATE / 1000)
        region_audio = audio[start_sample:end_sample]
        if end_ms - start_ms >= MIN_CLEAN_REGION_MS:
            yield from iter_windows_from_audio(
                region_audio,
                start_ms=start_ms,
                minimum_samples=minimum_samples,
            )
        elif len(region_audio) > 0:
            short_parts.append((region_audio, start_ms, end_ms))
    if short_parts:
        yield from _iter_stitched_windows(short_parts)


def extract_speaker_clouds(
    audio: np.ndarray,
    exclusive_segments: Sequence[Mapping[str, Any]],
    overlaps: Sequence[Mapping[str, Any]],
) -> dict[str, SpeakerCloud]:
    audio_samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_ms = round(len(audio_samples) * 1000 / REDIMNET_SAMPLE_RATE)
    overlap_intervals = _merge_intervals(
        [
            (max(0, int(item["start_ms"])), min(duration_ms, int(item["end_ms"])))
            for item in overlaps
            if int(item.get("end_ms", 0)) > int(item.get("start_ms", 0))
        ]
    )
    regions_by_speaker: dict[str, list[tuple[int, int]]] = {}
    for segment in exclusive_segments:
        speaker_id = str(segment.get("speaker_id") or "").strip()
        if not speaker_id:
            continue
        start_ms = max(0, int(segment.get("start_ms", 0)) + BOUNDARY_MARGIN_MS)
        end_ms = min(duration_ms, int(segment.get("end_ms", 0)) - BOUNDARY_MARGIN_MS)
        if end_ms <= start_ms:
            continue
        for clean_start, clean_end in _subtract_intervals(start_ms, end_ms, overlap_intervals):
            if clean_end > clean_start:
                regions_by_speaker.setdefault(speaker_id, []).append((clean_start, clean_end))

    all_speaker_ids = sorted(
        {str(item.get("speaker_id") or "") for item in exclusive_segments if item.get("speaker_id")}
    )
    owners: list[str] = []

    def all_windows() -> Iterable[VoiceWindow]:
        for speaker_id in all_speaker_ids:
            regions = sorted(regions_by_speaker.get(speaker_id, []))
            for window in _iter_speaker_windows(audio_samples, regions):
                owners.append(speaker_id)
                yield window

    # One global call fills batches across speaker boundaries.  The previous
    # per-speaker calls produced many tiny forwards in conversational audio.
    embedded = embed_voice_windows(all_windows())
    if len(embedded) != len(owners):
        raise RuntimeError("ReDimNet2 lieferte nicht fuer jedes Sprecherfenster einen Vektor.")
    by_speaker: dict[str, list[EmbeddedVoiceWindow]] = {
        speaker_id: [] for speaker_id in all_speaker_ids
    }
    for speaker_id, item in zip(owners, embedded):
        by_speaker[speaker_id].append(item)
    return {
        speaker_id: build_robust_cloud(speaker_id, by_speaker[speaker_id])
        for speaker_id in all_speaker_ids
    }


def _profile_cloud(profile: Mapping[str, Any]) -> SpeakerCloud:
    samples = [
        EmbeddedVoiceWindow(
            vector=vector,
            start_ms=None,
            end_ms=None,
            clean_duration_seconds=3.0,
            quality=1.0,
            stitched=False,
            source_spans=None,
        )
        for vector in profile["vectors"]
    ]
    return build_robust_cloud(str(profile["id"]), samples)


def _profile_support(
    cluster_matrix: np.ndarray,
    cluster_weights: np.ndarray,
    profile_matrix: np.ndarray,
) -> float:
    """Compute exact max-cosine support with bounded temporary matrices."""

    if len(cluster_matrix) == 0 or len(profile_matrix) == 0:
        return 0.0
    total_weight = float(np.sum(cluster_weights, dtype=np.float64))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return 0.0
    supported_weight = 0.0
    block_size = max(1, int(MATCH_SIMILARITY_BLOCK_SIZE))
    for cluster_start in range(0, len(cluster_matrix), block_size):
        cluster_block = cluster_matrix[cluster_start : cluster_start + block_size]
        maximum = np.full(len(cluster_block), -np.inf, dtype=np.float32)
        for profile_start in range(0, len(profile_matrix), block_size):
            profile_block = profile_matrix[profile_start : profile_start + block_size]
            similarities = np.matmul(cluster_block, profile_block.T)
            np.maximum(maximum, np.max(similarities, axis=1), out=maximum)
        block_weights = cluster_weights[
            cluster_start : cluster_start + len(cluster_block)
        ]
        supported_weight += float(
            np.sum(
                block_weights[maximum >= SAMPLE_SUPPORT_COSINE_MIN],
                dtype=np.float64,
            )
        )
    return supported_weight / total_weight


def match_known_speakers(
    known_speakers: Sequence[Mapping[str, Any]],
    speaker_clouds: Mapping[str, SpeakerCloud],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    """Return accepted assignments, unresolved profiles and tentative clusters.

    A tentative cluster is the globally selected profile/cluster pair whose
    evidence failed one of the acceptance gates.  Exposing it separately lets
    the API retain the DIA label and vectors while marking turns as
    ``unresolved`` instead of pretending the cluster is certainly unknown.
    """

    validated = validate_known_speakers(known_speakers)
    if not validated:
        return {}, [], {}
    profile_clouds = [_profile_cloud(profile) for profile in validated]
    unavailable_profile_indices = {
        index
        for index, cloud in enumerate(profile_clouds)
        if cloud.prototype is None or cloud.status == "mixed_cluster"
    }
    eligible_profiles = [
        (index, profile, profile_clouds[index])
        for index, profile in enumerate(validated)
        if profile_clouds[index].prototype is not None
        and profile_clouds[index].status != "mixed_cluster"
    ]
    cluster_items = [
        (speaker_id, cloud)
        for speaker_id, cloud in sorted(speaker_clouds.items())
        if cloud.prototype is not None
    ]
    if not eligible_profiles or not cluster_items:
        reason = (
            "profile_evidence_unavailable"
            if not eligible_profiles
            else "embedding_evidence_unavailable"
        )
        unavailable_clusters = {
            cluster_id: {
                "candidate_speaker_id": None,
                "cosine_similarity": None,
                "support": None,
                "stability_margin": None,
                "reason": reason,
            }
            for cluster_id in sorted(speaker_clouds)
        }
        return {}, [str(profile["id"]) for profile in validated], unavailable_clusters

    profile_count = len(eligible_profiles)
    cluster_count = len(cluster_items)
    cosine_scores = np.zeros((profile_count, cluster_count), dtype=np.float64)
    supports = np.zeros((profile_count, cluster_count), dtype=np.float64)
    valid_edges = np.zeros((profile_count, cluster_count), dtype=bool)
    cluster_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    for _, cluster_cloud in cluster_items:
        inliers = cluster_cloud.inliers
        if inliers:
            matrix = np.stack([item.vector for item in inliers], axis=0)
            weights = np.asarray(
                [max(item.quality * item.clean_duration_seconds, 1e-6) for item in inliers],
                dtype=np.float64,
            )
        else:
            matrix = np.empty((0, REDIMNET_EMBEDDING_DIMENSION), dtype=np.float32)
            weights = np.empty(0, dtype=np.float64)
        cluster_evidence.append((matrix, weights))
    for profile_index, (_, _, profile_cloud) in enumerate(eligible_profiles):
        profile_matrix = np.stack([item.vector for item in profile_cloud.inliers], axis=0)
        for cluster_index, (_, cluster_cloud) in enumerate(cluster_items):
            score = float(profile_cloud.prototype @ cluster_cloud.prototype)
            cosine_scores[profile_index, cluster_index] = score
            inlier_matrix, weights = cluster_evidence[cluster_index]
            supports[profile_index, cluster_index] = _profile_support(
                inlier_matrix,
                weights,
                profile_matrix,
            )
            valid_edges[profile_index, cluster_index] = (
                cluster_cloud.status == "ready"
                and score >= MATCH_COSINE_MIN
                and supports[profile_index, cluster_index] >= MATCH_SUPPORT_MIN
            )

    assignment_scores = np.full(
        (profile_count, cluster_count + profile_count),
        MATCH_DUMMY_SCORE,
        dtype=np.float64,
    )
    assignment_scores[:, :cluster_count] = np.where(
        valid_edges,
        cosine_scores,
        INVALID_ASSIGNMENT_SCORE,
    )

    # Stability is a global property. Remove unstable selected edges and solve
    # again so a rejected edge can never block another valid profile/cluster
    # pairing from being considered.
    final_pairs: list[tuple[int, int, float]] = []
    while True:
        row_indices, column_indices = linear_sum_assignment(-assignment_scores)
        base_total = float(np.sum(assignment_scores[row_indices, column_indices]))
        unstable: list[tuple[int, int]] = []
        selected: list[tuple[int, int, float]] = []
        for profile_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
            if column_index >= cluster_count:
                continue
            alternative_scores = assignment_scores.copy()
            alternative_scores[profile_index, column_index] = INVALID_ASSIGNMENT_SCORE
            alternative_rows, alternative_columns = linear_sum_assignment(-alternative_scores)
            alternative_total = float(np.sum(alternative_scores[alternative_rows, alternative_columns]))
            stability_margin = base_total - alternative_total
            if stability_margin < MATCH_STABILITY_MARGIN_MIN:
                unstable.append((profile_index, column_index))
            else:
                selected.append((profile_index, column_index, stability_margin))
        if not unstable:
            final_pairs = selected
            break
        for profile_index, column_index in unstable:
            assignment_scores[profile_index, column_index] = INVALID_ASSIGNMENT_SCORE

    assignments: dict[str, dict[str, Any]] = {}
    assigned_profile_indices: set[int] = set()
    assigned_cluster_indices: set[int] = set()
    for profile_index, column_index, stability_margin in final_pairs:
        original_profile_index, profile, _ = eligible_profiles[profile_index]
        profile_id = str(profile["id"])
        cluster_id = cluster_items[column_index][0]
        score = float(cosine_scores[profile_index, column_index])
        support = float(supports[profile_index, column_index])
        assignments[cluster_id] = {
            "speaker_id": profile_id,
            "cosine_similarity": round(score, 6),
            "support": round(support, 6),
            "stability_margin": round(stability_margin, 6),
        }
        assigned_profile_indices.add(original_profile_index)
        assigned_cluster_indices.add(column_index)

    unresolved = [
        str(profile["id"])
        for index, profile in enumerate(validated)
        if index not in assigned_profile_indices
    ]

    # Pair remaining profiles/clusters only for diagnostic ``unresolved``
    # output. These tentative pairs never become identities and cannot affect
    # the accepted Hungarian solution above.
    unresolved_clusters: dict[str, dict[str, Any]] = {}
    tentative_profile_rows = [
        row
        for row, (original_index, _, _) in enumerate(eligible_profiles)
        if original_index not in assigned_profile_indices
    ]
    tentative_cluster_columns = [
        column for column in range(cluster_count) if column not in assigned_cluster_indices
    ]
    if tentative_profile_rows and tentative_cluster_columns:
        tentative_real = cosine_scores[np.ix_(tentative_profile_rows, tentative_cluster_columns)]
        tentative_scores = np.full(
            (len(tentative_profile_rows), len(tentative_cluster_columns) + len(tentative_profile_rows)),
            TENTATIVE_DUMMY_SCORE,
            dtype=np.float64,
        )
        tentative_scores[:, : len(tentative_cluster_columns)] = np.where(
            tentative_real >= COMPONENT_COSINE_MIN,
            tentative_real,
            INVALID_ASSIGNMENT_SCORE,
        )
        tentative_rows, tentative_columns = linear_sum_assignment(-tentative_scores)
        tentative_total = float(np.sum(tentative_scores[tentative_rows, tentative_columns]))
        for local_profile_row, local_cluster_column in zip(
            tentative_rows.tolist(),
            tentative_columns.tolist(),
        ):
            if local_cluster_column >= len(tentative_cluster_columns):
                continue
            profile_row = tentative_profile_rows[local_profile_row]
            cluster_column = tentative_cluster_columns[local_cluster_column]
            profile_id = str(eligible_profiles[profile_row][1]["id"])
            cluster_id = cluster_items[cluster_column][0]
            alternative = tentative_scores.copy()
            alternative[local_profile_row, local_cluster_column] = INVALID_ASSIGNMENT_SCORE
            alt_rows, alt_columns = linear_sum_assignment(-alternative)
            stability_margin = tentative_total - float(np.sum(alternative[alt_rows, alt_columns]))
            unresolved_clusters[cluster_id] = {
                "candidate_speaker_id": profile_id,
                "cosine_similarity": round(float(cosine_scores[profile_row, cluster_column]), 6),
                "support": round(float(supports[profile_row, cluster_column]), 6),
                "stability_margin": round(stability_margin, 6),
            }

    # A cloud that could not produce even a conservative prototype is still
    # unresolved (rather than confidently unknown) when known profiles remain.
    # This is most relevant for bounded component overflow and insufficient
    # clean speech. No candidate identity is invented without cosine evidence.
    if unresolved:
        for cluster_id, cloud in sorted(speaker_clouds.items()):
            if cluster_id in assignments or cluster_id in unresolved_clusters:
                continue
            if cloud.status == "ready" or cloud.prototype is not None:
                continue
            unresolved_clusters[cluster_id] = {
                "candidate_speaker_id": None,
                "cosine_similarity": None,
                "support": None,
                "stability_margin": None,
                "reason": "embedding_evidence_unavailable",
            }
    if unavailable_profile_indices:
        for cluster_id in sorted(speaker_clouds):
            if cluster_id in assignments or cluster_id in unresolved_clusters:
                continue
            unresolved_clusters[cluster_id] = {
                "candidate_speaker_id": None,
                "cosine_similarity": None,
                "support": None,
                "stability_margin": None,
                "reason": "profile_evidence_unavailable",
            }
    return assignments, unresolved, unresolved_clusters


__all__ = [
    "SpeakerCloud",
    "SpeakerProfileValidationError",
    "build_robust_cloud",
    "extract_speaker_clouds",
    "match_known_speakers",
    "validate_known_speakers",
]
