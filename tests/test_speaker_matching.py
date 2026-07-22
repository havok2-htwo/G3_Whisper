import unittest
from unittest.mock import patch

import numpy as np

from backend.genesis_whisper_server_speaker_matching import (
    SpeakerCloud,
    SpeakerProfileValidationError,
    build_robust_cloud,
    match_known_speakers,
    validate_known_speakers,
)
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow, REDIMNET_EMBEDDING_DIMENSION


def _normalized(*components: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(REDIMNET_EMBEDDING_DIMENSION, dtype=np.float32)
    for index, value in components:
        vector[index] = value
    return vector / np.linalg.norm(vector)


def _sample(
    vector: np.ndarray,
    index: int,
    *,
    duration: float = 3.0,
    quality: float = 1.0,
) -> EmbeddedVoiceWindow:
    return EmbeddedVoiceWindow(
        vector=np.asarray(vector, dtype=np.float32),
        start_ms=index * 3000,
        end_ms=(index + 1) * 3000,
        clean_duration_seconds=duration,
        quality=quality,
        stitched=False,
    )


def _ready_cloud(speaker_id: str, vector: np.ndarray):
    base_index = int(np.argmax(np.abs(vector)))
    variants = []
    for index in range(3):
        variant = vector.copy()
        variant[(base_index + index + 1) % len(vector)] += 0.12
        variants.append(variant / np.linalg.norm(variant))
    samples = [_sample(variant, index) for index, variant in enumerate(variants)]
    cloud = build_robust_cloud(speaker_id, samples)
    if cloud.status != "ready":
        raise AssertionError(f"test fixture did not create a ready cloud: {cloud.status}")
    return cloud


class SpeakerProfileValidationTests(unittest.TestCase):
    def test_rejects_512_and_256_dimensions(self) -> None:
        for dimension in (512, 256):
            with self.subTest(dimension=dimension):
                with self.assertRaises(SpeakerProfileValidationError):
                    validate_known_speakers([{"id": "person", "embeddings": [[1.0] * dimension]}])

    def test_rejects_non_finite_and_zero_vectors(self) -> None:
        invalid_vectors = [
            [0.0] * 192,
            [float("nan")] + [0.0] * 191,
            [float("inf")] + [0.0] * 191,
        ]
        for vector in invalid_vectors:
            with self.subTest(first_value=vector[0]):
                with self.assertRaises(SpeakerProfileValidationError):
                    validate_known_speakers([{"id": "person", "embeddings": [vector]}])

    def test_normalizes_valid_profiles_and_rejects_duplicate_ids(self) -> None:
        validated = validate_known_speakers(
            [{"id": "person-17", "embeddings": [[2.0] + [0.0] * 191]}]
        )
        self.assertEqual(validated[0]["vectors"][0].shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(validated[0]["vectors"][0])), 1.0, places=6)

        with self.assertRaises(SpeakerProfileValidationError):
            validate_known_speakers(
                [
                    {"id": "duplicate", "embeddings": [[1.0] + [0.0] * 191]},
                    {"id": "duplicate", "embeddings": [[0.0, 1.0] + [0.0] * 190]},
                ]
            )


class SpeakerMatchingTests(unittest.TestCase):
    def test_identical_time_separated_windows_retain_support_and_duration(self) -> None:
        vector = _normalized((0, 1.0))
        samples = [_sample(vector, index) for index in range(3)]

        cloud = build_robust_cloud("SPEAKER_00", samples)

        self.assertEqual(cloud.status, "ready")
        self.assertEqual(cloud.candidate_count, 3)
        self.assertEqual(len(cloud.inliers), 3)
        self.assertEqual(cloud.discarded_outliers, 0)
        self.assertAlmostEqual(
            sum(item.clean_duration_seconds for item in cloud.inliers),
            9.0,
        )
        # Near-identical vectors are collapsed only in the public payload.
        self.assertEqual(len(cloud.public_embeddings()), 2)

    def test_component_capacity_fails_closed_after_considering_large_cloud(self) -> None:
        samples = [_sample(_normalized((index, 1.0)), index) for index in range(8)]

        with patch(
            "backend.genesis_whisper_server_speaker_matching.MAX_STREAMING_COMPONENTS",
            4,
        ):
            cloud = build_robust_cloud("SPEAKER_00", samples)

        self.assertEqual(cloud.status, "mixed_cluster")
        self.assertIsNone(cloud.prototype)
        self.assertEqual(cloud.candidate_count, 8)
        self.assertEqual(len(cloud.samples), 8)
        self.assertEqual(cloud.discarded_outliers, 8)

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [{"id": "person", "embeddings": [_normalized((0, 1.0)).tolist()]}],
            {"SPEAKER_00": cloud},
        )
        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["person"])
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["reason"],
            "embedding_evidence_unavailable",
        )

    def test_capacity_overflow_returns_only_a_supported_tracked_component(self) -> None:
        primary = _normalized((0, 1.0))
        samples = [_sample(primary, index) for index in range(3)]
        samples.extend(
            _sample(_normalized((index, 1.0)), index + 3)
            for index in range(1, 7)
        )

        with patch(
            "backend.genesis_whisper_server_speaker_matching.MAX_STREAMING_COMPONENTS",
            3,
        ):
            cloud = build_robust_cloud("SPEAKER_00", samples)

        self.assertEqual(cloud.status, "mixed_cluster")
        self.assertIsNotNone(cloud.prototype)
        self.assertEqual(cloud.inlier_indices, [0, 1, 2])
        self.assertEqual(len(cloud.public_embeddings()), 2)
        self.assertGreater(float(cloud.prototype @ primary), 0.99)

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [{"id": "person", "embeddings": [primary.tolist()]}],
            {"SPEAKER_00": cloud},
        )
        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["person"])
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["candidate_speaker_id"],
            "person",
        )

    def test_large_near_duplicate_cloud_is_ready_with_bounded_public_output(self) -> None:
        vector = _normalized((0, 1.0))
        samples = [_sample(vector, index) for index in range(5000)]

        cloud = build_robust_cloud("SPEAKER_00", samples)
        public = cloud.public_embeddings()

        self.assertEqual(cloud.status, "ready")
        self.assertEqual(cloud.candidate_count, 5000)
        self.assertEqual(len(cloud.inliers), 5000)
        self.assertEqual(cloud.discarded_outliers, 0)
        self.assertEqual(len(public), 2)

    def test_five_detected_four_known_leaves_exactly_one_unknown(self) -> None:
        vectors = [_normalized((index, 1.0)) for index in range(5)]
        clouds = {
            f"SPEAKER_{index:02d}": _ready_cloud(f"SPEAKER_{index:02d}", vector)
            for index, vector in enumerate(vectors)
        }
        known = [
            {"id": f"person-{index}", "embeddings": [vectors[index].tolist()]}
            for index in range(4)
        ]

        assignments, unresolved, unresolved_clusters = match_known_speakers(known, clouds)

        self.assertEqual(len(assignments), 4)
        self.assertEqual(unresolved, [])
        self.assertEqual(unresolved_clusters, {})
        self.assertEqual(set(clouds) - set(assignments), {"SPEAKER_04"})
        self.assertEqual(
            {assignment["speaker_id"] for assignment in assignments.values()},
            {"person-0", "person-1", "person-2", "person-3"},
        )

    def test_weak_evidence_is_unresolved_instead_of_forced(self) -> None:
        known_vector = _normalized((0, 1.0))
        # Deliberately below the accepted-match dummy score (0.55), but still
        # above the plausible-component threshold (0.45). Diagnostic matching
        # must retain it as unresolved instead of silently calling it unknown.
        weak_match_vector = _normalized((0, 0.50), (1, float(np.sqrt(1.0 - 0.50**2))))
        clouds = {"SPEAKER_00": _ready_cloud("SPEAKER_00", weak_match_vector)}

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [{"id": "known-person", "embeddings": [known_vector.tolist()]}],
            clouds,
        )

        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["known-person"])
        self.assertEqual(set(unresolved_clusters), {"SPEAKER_00"})
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["candidate_speaker_id"],
            "known-person",
        )
        self.assertLess(unresolved_clusters["SPEAKER_00"]["cosine_similarity"], 0.60)

    def test_invalid_high_score_edge_cannot_steal_valid_assignment(self) -> None:
        profile_a = _normalized((0, 1.0))
        profile_b = _normalized((1, 1.0))
        cluster_a_prototype = _normalized((0, 0.65), (1, 0.76))
        cluster_b_prototype = _normalized((1, 0.61), (2, 0.79))
        # Cluster A's prototype is closer to B, but its actual windows support
        # only A. The high B/A edge must be gated before Hungarian matching.
        cluster_a_samples = [_sample(profile_a, index) for index in range(2)]
        cluster_b_samples = [_sample(cluster_b_prototype, index + 2) for index in range(2)]
        clouds = {
            "SPEAKER_A": SpeakerCloud(
                diarization_speaker_id="SPEAKER_A",
                status="ready",
                prototype=cluster_a_prototype,
                samples=cluster_a_samples,
                inlier_indices=[0, 1],
                candidate_count=2,
                discarded_outliers=0,
                purity=1.0,
            ),
            "SPEAKER_B": SpeakerCloud(
                diarization_speaker_id="SPEAKER_B",
                status="ready",
                prototype=cluster_b_prototype,
                samples=cluster_b_samples,
                inlier_indices=[0, 1],
                candidate_count=2,
                discarded_outliers=0,
                purity=1.0,
            ),
        }

        assignments, unresolved, _ = match_known_speakers(
            [
                {"id": "person-a", "embeddings": [profile_a.tolist()]},
                {"id": "person-b", "embeddings": [profile_b.tolist()]},
            ],
            clouds,
        )

        self.assertEqual(unresolved, [])
        self.assertEqual(assignments["SPEAKER_A"]["speaker_id"], "person-a")
        self.assertEqual(assignments["SPEAKER_B"]["speaker_id"], "person-b")

    def test_inconsistent_known_profile_remains_unresolved(self) -> None:
        profile_a = _normalized((0, 1.0))
        profile_b = _normalized((1, 1.0))
        target = _ready_cloud("SPEAKER_00", profile_a)

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [
                {
                    "id": "inconsistent",
                    "embeddings": [profile_a.tolist(), profile_b.tolist()],
                }
            ],
            {"SPEAKER_00": target},
        )

        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["inconsistent"])
        self.assertEqual(set(unresolved_clusters), {"SPEAKER_00"})
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["reason"],
            "profile_evidence_unavailable",
        )

    def test_non_ready_cluster_is_never_promoted_to_known(self) -> None:
        vector = _normalized((0, 1.0))
        cloud = build_robust_cloud("SPEAKER_00", [_sample(vector, 0)])
        self.assertEqual(cloud.status, "low_support")

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [{"id": "known-person", "embeddings": [vector.tolist()]}],
            {"SPEAKER_00": cloud},
        )

        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["known-person"])
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["candidate_speaker_id"],
            "known-person",
        )

    def test_mixed_cluster_returns_cleaned_main_component_as_unresolved(self) -> None:
        main = _normalized((0, 1.0))
        secondary = _normalized((1, 1.0))
        samples = [_sample(main, index) for index in range(3)]
        samples.extend(_sample(secondary, index + 3) for index in range(3))

        cloud = build_robust_cloud("SPEAKER_00", samples)
        self.assertEqual(cloud.status, "mixed_cluster")
        self.assertIsNotNone(cloud.prototype)
        self.assertEqual(cloud.inlier_indices, [0, 1, 2])
        self.assertEqual(cloud.discarded_outliers, 3)
        self.assertEqual(len(cloud.public_embeddings()), 2)

        assignments, unresolved, unresolved_clusters = match_known_speakers(
            [{"id": "known-person", "embeddings": [main.tolist()]}],
            {"SPEAKER_00": cloud},
        )
        self.assertEqual(assignments, {})
        self.assertEqual(unresolved, ["known-person"])
        self.assertEqual(
            unresolved_clusters["SPEAKER_00"]["candidate_speaker_id"],
            "known-person",
        )

    def test_outlier_component_is_discarded(self) -> None:
        main_vector = _normalized((0, 1.0))
        outlier = _normalized((1, 1.0))
        main_variants = []
        for index in range(5):
            variant = main_vector.copy()
            variant[index + 2] += 0.12
            main_variants.append(variant / np.linalg.norm(variant))
        samples = [_sample(vector, index) for index, vector in enumerate(main_variants)] + [_sample(outlier, 5)]

        cloud = build_robust_cloud("SPEAKER_00", samples)

        self.assertEqual(cloud.status, "ready")
        self.assertEqual(cloud.candidate_count, 6)
        self.assertEqual(cloud.discarded_outliers, 1)
        self.assertEqual(cloud.inlier_indices, [0, 1, 2, 3, 4])
        self.assertGreater(float(cloud.prototype @ main_vector), 0.99)

    def test_public_embedding_limit_is_64_and_deterministic(self) -> None:
        samples = []
        for index in range(80):
            # Every pair remains in one cosine component (0.64), while no two vectors
            # are near-duplicates. This exercises the deterministic diversity sampler.
            vector = _normalized((0, 0.8), (index + 1, 0.6))
            samples.append(_sample(vector, index))
        cloud = build_robust_cloud("SPEAKER_00", samples)

        first = cloud.public_embeddings()
        second = cloud.public_embeddings()

        self.assertEqual(cloud.status, "ready")
        self.assertEqual(len(first), 64)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["kind"], "prototype")
        self.assertTrue(all(len(item["vector"]) == 192 for item in first))
        self.assertEqual(len({item.get("start_ms") for item in first[1:]}), 63)


if __name__ == "__main__":
    unittest.main()
