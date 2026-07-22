import unittest

import numpy as np

from backend.genesis_whisper_server_speaker_matching import SpeakerCloud
from backend.genesis_whisper_server_speaker_refinement import refine_speaker_turns
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow, REDIMNET_EMBEDDING_DIMENSION


def _vector(*components: tuple[int, float]) -> np.ndarray:
    value = np.zeros(REDIMNET_EMBEDDING_DIMENSION, dtype=np.float32)
    for index, component in components:
        value[index] = component
    return value / np.linalg.norm(value)


def _sample(
    vector: np.ndarray,
    start_ms: int,
    end_ms: int,
    *,
    quality: float = 1.0,
    stitched: bool = False,
) -> EmbeddedVoiceWindow:
    return EmbeddedVoiceWindow(
        vector=vector,
        start_ms=start_ms,
        end_ms=end_ms,
        clean_duration_seconds=(end_ms - start_ms) / 1000.0,
        quality=quality,
        stitched=stitched,
    )


def _cloud(
    speaker_id: str,
    prototype: np.ndarray,
    samples: list[EmbeddedVoiceWindow],
    inlier_indices: list[int],
    *,
    status: str = "ready",
    purity: float = 0.9,
) -> SpeakerCloud:
    return SpeakerCloud(
        diarization_speaker_id=speaker_id,
        status=status,
        prototype=prototype,
        samples=samples,
        inlier_indices=inlier_indices,
        candidate_count=len(samples),
        discarded_outliers=len(samples) - len(inlier_indices),
        purity=purity,
    )


def _regular_fixture(*, compact: bool = False):
    source = _vector((0, 1.0))
    target = _vector((1, 1.0))
    suspicious = _vector((0, 0.30), (1, 0.80), (2, float(np.sqrt(0.27))))
    if compact:
        turns = [
            {"start_ms": 0, "end_ms": 9000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 10000, "end_ms": 16000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 17000, "end_ms": 26000, "speaker_id": "SPEAKER_B"},
        ]
        candidate_start = 10200
        target_start = 17200
    else:
        # Six corrected seconds are below the global ten-percent rollback gate.
        turns = [
            {"start_ms": 0, "end_ms": 30000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 32000, "end_ms": 38000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 40000, "end_ms": 70000, "speaker_id": "SPEAKER_B"},
        ]
        candidate_start = 32200
        target_start = 40200
    source_samples = [
        _sample(source, 200, 3200),
        _sample(source, 3200, 6200),
        _sample(source, 6200, 9200),
        _sample(suspicious, candidate_start, candidate_start + 3000),
        _sample(suspicious, candidate_start + 3000, candidate_start + 5600),
    ]
    target_samples = [
        _sample(target, target_start, target_start + 3000),
        _sample(target, target_start + 3000, target_start + 6000),
        _sample(target, target_start + 6000, target_start + 9000),
    ]
    clouds = {
        "SPEAKER_A": _cloud("SPEAKER_A", source, source_samples, [0, 1, 2]),
        "SPEAKER_B": _cloud("SPEAKER_B", target, target_samples, [0, 1, 2]),
    }
    return turns, [dict(turn) for turn in turns], [], clouds


class SpeakerRefinementTests(unittest.TestCase):
    def test_off_adds_provenance_without_inspecting_or_rebuilding_clouds(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()

        result = refine_speaker_turns("off", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "disabled")
        self.assertEqual(result.diagnostics["eligible_windows"], 0)
        self.assertTrue(all(turn["speaker_id"] == turn["original_speaker_id"] for turn in result.turns))
        self.assertIs(result.speaker_clouds["SPEAKER_A"], clouds["SPEAKER_A"])

    def test_shadow_reports_deterministic_proposal_without_applying_it(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()

        first = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)
        second = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(first.diagnostics["status"], "shadow")
        self.assertEqual(first.diagnostics["proposed_turns"], 1)
        self.assertEqual(first.diagnostics["applied_turns"], 0)
        self.assertEqual(first.diagnostics["reassigned_duration_ms"], 0)
        self.assertEqual(first.turns[1]["speaker_id"], "SPEAKER_A")
        self.assertIs(first.speaker_clouds["SPEAKER_A"], clouds["SPEAKER_A"])
        self.assertFalse(first.diagnostics["changes"][0]["applied"])
        comparable_keys = set(first.diagnostics) - {"processing_ms"}
        self.assertEqual(
            {key: first.diagnostics[key] for key in comparable_keys},
            {key: second.diagnostics[key] for key in comparable_keys},
        )

    def test_conservative_applies_whole_turn_and_rebuilds_from_same_vectors(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        moved_samples = clouds["SPEAKER_A"].samples[3:]

        result = refine_speaker_turns("conservative", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "applied")
        self.assertEqual(result.diagnostics["applied_turns"], 1)
        self.assertEqual(result.diagnostics["reassigned_duration_ms"], 6000)
        self.assertEqual(result.turns[1]["speaker_id"], "SPEAKER_B")
        self.assertEqual(result.turns[1]["original_speaker_id"], "SPEAKER_A")
        self.assertEqual(
            [(turn["start_ms"], turn["end_ms"]) for turn in result.turns],
            [(turn["start_ms"], turn["end_ms"]) for turn in turns],
        )
        self.assertIsNot(result.speaker_clouds["SPEAKER_B"], clouds["SPEAKER_B"])
        self.assertTrue(
            all(
                any(item is sample for item in result.speaker_clouds["SPEAKER_B"].samples)
                for sample in moved_samples
            )
        )
        self.assertTrue(all(not item.stitched for item in result.speaker_clouds["SPEAKER_B"].inliers))
        self.assertTrue(result.diagnostics["changes"][0]["applied"])

    def test_global_ten_percent_gate_rolls_back_turns_and_clouds(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture(compact=True)

        result = refine_speaker_turns("conservative", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "rejected")
        self.assertEqual(
            result.diagnostics["rollback_reason"],
            "reassigned_speech_exceeds_10_percent",
        )
        self.assertEqual(result.turns[1]["speaker_id"], "SPEAKER_A")
        self.assertEqual(result.diagnostics["applied_turns"], 0)
        self.assertIs(result.speaker_clouds["SPEAKER_A"], clouds["SPEAKER_A"])
        self.assertIs(result.speaker_clouds["SPEAKER_B"], clouds["SPEAKER_B"])

    def test_target_active_in_standard_timeline_blocks_vote(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        standard.append({"start_ms": 34000, "end_ms": 35000, "speaker_id": "SPEAKER_B"})

        result = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "not_needed")
        self.assertEqual(result.diagnostics["proposed_turns"], 0)

    def test_explicit_overlap_blocks_all_supporting_windows(self) -> None:
        turns, standard, _, clouds = _regular_fixture()
        overlaps = [{"start_ms": 32000, "end_ms": 38000, "speaker_ids": ["SPEAKER_A", "SPEAKER_B"]}]

        result = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "not_needed")

    def test_strong_window_for_original_speaker_vetoes_turn(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        clouds["SPEAKER_A"].samples.append(_sample(_vector((0, 1.0)), 35000, 37500))

        result = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "not_needed")

    def test_stitched_windows_never_drive_reassignment(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        source_cloud = clouds["SPEAKER_A"]
        clouds["SPEAKER_A"] = _cloud(
            "SPEAKER_A",
            source_cloud.prototype,
            source_cloud.samples[:3]
            + [
                EmbeddedVoiceWindow(
                    vector=source_cloud.samples[3].vector,
                    start_ms=32200,
                    end_ms=37800,
                    clean_duration_seconds=5.6,
                    quality=1.0,
                    stitched=True,
                    source_spans=((32200, 35000), (35200, 37800)),
                )
            ],
            [0, 1, 2],
        )

        result = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "not_needed")

    def test_target_seed_prototype_excludes_stitched_cloud_contamination(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        target_cloud = clouds["SPEAKER_B"]
        contaminated = EmbeddedVoiceWindow(
            vector=_vector((0, 1.0)),
            start_ms=None,
            end_ms=None,
            clean_duration_seconds=3.0,
            quality=0.6,
            stitched=True,
            source_spans=((71000, 72500), (73000, 74500)),
        )
        # Simulate a legacy aggregate prototype that was pulled toward a
        # stitched sample. The frozen refinement seed must be recomputed only
        # from the three qualifying, non-stitched inliers.
        clouds["SPEAKER_B"] = _cloud(
            "SPEAKER_B",
            _vector((0, 1.0)),
            target_cloud.samples + [contaminated],
            [0, 1, 2],
        )

        result = refine_speaker_turns("shadow", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "shadow")
        self.assertEqual(result.diagnostics["changes"][0]["to_speaker_id"], "SPEAKER_B")

    def test_moving_seed_inlier_below_minimum_rolls_back(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        source_cloud = clouds["SPEAKER_A"]
        clouds["SPEAKER_A"] = _cloud(
            "SPEAKER_A",
            source_cloud.prototype,
            source_cloud.samples,
            [0, 1, 3],
        )

        result = refine_speaker_turns("conservative", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "rejected")
        self.assertEqual(result.diagnostics["rollback_reason"], "seed_core_lost")
        self.assertIs(result.speaker_clouds["SPEAKER_A"], clouds["SPEAKER_A"])

    def test_previously_ready_cloud_becoming_mixed_rolls_back(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        source = _vector((0, 1.0))
        unrelated = _vector((3, 1.0))
        suspicious_samples = clouds["SPEAKER_A"].samples[3:]
        source_samples = [
            _sample(source, 200, 3200),
            _sample(source, 3200, 6200),
            _sample(source, 6200, 9200),
            _sample(unrelated, 10000, 13000),
            _sample(unrelated, 13000, 16000),
            _sample(unrelated, 16000, 19000),
            *suspicious_samples,
        ]
        # Purity below 0.70 deliberately prevents this manually supplied
        # ready cloud from serving as a frozen target seed. Once its two
        # suspicious windows move, rebuilding exposes two equally weighted
        # components and must trigger the ready->mixed rollback invariant.
        clouds["SPEAKER_A"] = _cloud(
            "SPEAKER_A",
            source,
            source_samples,
            [0, 1, 2],
            purity=0.65,
        )

        result = refine_speaker_turns("conservative", turns, standard, overlaps, clouds)

        self.assertEqual(result.diagnostics["status"], "rejected")
        self.assertEqual(
            result.diagnostics["rollback_reason"],
            "ready_cloud_became_mixed_cluster",
        )
        self.assertIs(result.speaker_clouds["SPEAKER_A"], clouds["SPEAKER_A"])

    def test_public_change_list_is_deterministically_capped(self) -> None:
        source = _vector((0, 1.0))
        target = _vector((1, 1.0))
        candidate = _vector((0, 0.20), (1, 0.80), (2, float(np.sqrt(0.32))))
        turns = [{"start_ms": 0, "end_ms": 30000, "speaker_id": "SPEAKER_A"}]
        source_samples = [
            _sample(source, 200, 3200),
            _sample(source, 3200, 6200),
            _sample(source, 6200, 9200),
        ]
        cursor = 32000
        for _ in range(101):
            turns.append(
                {"start_ms": cursor, "end_ms": cursor + 2600, "speaker_id": "SPEAKER_A"}
            )
            source_samples.append(_sample(candidate, cursor + 200, cursor + 2400))
            cursor += 3000
        target_start = cursor + 1000
        turns.append(
            {"start_ms": target_start, "end_ms": target_start + 30000, "speaker_id": "SPEAKER_B"}
        )
        target_samples = [
            _sample(target, target_start + 200, target_start + 3200),
            _sample(target, target_start + 3200, target_start + 6200),
            _sample(target, target_start + 6200, target_start + 9200),
        ]
        clouds = {
            "SPEAKER_A": _cloud("SPEAKER_A", source, source_samples, [0, 1, 2]),
            "SPEAKER_B": _cloud("SPEAKER_B", target, target_samples, [0, 1, 2]),
        }

        result = refine_speaker_turns("shadow", turns, turns, [], clouds)

        self.assertEqual(result.diagnostics["proposed_turns"], 101)
        self.assertEqual(len(result.diagnostics["changes"]), 100)
        self.assertTrue(result.diagnostics["changes_truncated"])
        self.assertEqual(
            [change["start_ms"] for change in result.diagnostics["changes"]],
            sorted(change["start_ms"] for change in result.diagnostics["changes"]),
        )

    def test_seeded_label_corruption_meets_precision_and_recall_gates(self) -> None:
        true_vectors = {
            "SPEAKER_A": _vector((0, 1.0)),
            "SPEAKER_B": _vector((1, 1.0)),
        }

        def run_fixture(seed: int, corruption_share: float):
            true_labels = ["SPEAKER_A" if index % 2 == 0 else "SPEAKER_B" for index in range(100)]
            rng = np.random.default_rng(seed)
            corrupted_indices = set(
                int(value)
                for value in rng.choice(
                    len(true_labels),
                    size=round(len(true_labels) * corruption_share),
                    replace=False,
                )
            )
            observed_labels = [
                (
                    "SPEAKER_B" if true_label == "SPEAKER_A" else "SPEAKER_A"
                )
                if index in corrupted_indices
                else true_label
                for index, true_label in enumerate(true_labels)
            ]
            turns = []
            samples_by_observed = {"SPEAKER_A": [], "SPEAKER_B": []}
            inliers_by_observed = {"SPEAKER_A": [], "SPEAKER_B": []}
            for index, (true_label, observed_label) in enumerate(
                zip(true_labels, observed_labels)
            ):
                start_ms = index * 4000
                turns.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": start_ms + 3000,
                        "speaker_id": observed_label,
                    }
                )
                observed_samples = samples_by_observed[observed_label]
                observed_samples.append(
                    _sample(true_vectors[true_label], start_ms + 200, start_ms + 2800)
                )
                if true_label == observed_label:
                    inliers_by_observed[observed_label].append(len(observed_samples) - 1)
            clouds = {
                speaker_id: _cloud(
                    speaker_id,
                    true_vectors[speaker_id],
                    samples_by_observed[speaker_id],
                    inliers_by_observed[speaker_id],
                    purity=1.0 - corruption_share,
                )
                for speaker_id in sorted(true_vectors)
            }
            result = refine_speaker_turns(
                "conservative",
                turns,
                turns,
                [],
                clouds,
            )
            changed = {
                index
                for index, (before, after) in enumerate(zip(observed_labels, result.turns))
                if before != after["speaker_id"]
            }
            correct = {
                index
                for index in changed
                if result.turns[index]["speaker_id"] == true_labels[index]
            }
            precision = len(correct) / len(changed) if changed else 1.0
            recall = len(correct) / len(corrupted_indices) if corrupted_indices else 1.0
            return result, precision, recall, changed, corrupted_indices

        pristine, precision, recall, changed, corrupted = run_fixture(0, 0.0)
        self.assertEqual(pristine.diagnostics["status"], "not_needed")
        self.assertEqual(changed, set())
        self.assertEqual(corrupted, set())
        self.assertEqual((precision, recall), (1.0, 1.0))

        for corruption_share, precision_minimum, recall_minimum in (
            (0.05, 0.98, 0.90),
            (0.10, 0.95, 0.80),
        ):
            for seed in range(10):
                with self.subTest(corruption_share=corruption_share, seed=seed):
                    result, precision, recall, changed, corrupted = run_fixture(
                        seed,
                        corruption_share,
                    )
                    self.assertEqual(result.diagnostics["status"], "applied")
                    self.assertGreaterEqual(precision, precision_minimum)
                    self.assertGreaterEqual(recall, recall_minimum)
                    self.assertEqual(changed, corrupted)

    def test_short_turn_exception_requires_one_strong_full_safe_window(self) -> None:
        source = _vector((0, 1.0))
        target = _vector((1, 1.0))
        candidate = _vector((0, 0.20), (1, 0.80), (2, float(np.sqrt(0.32))))
        turns = [
            {"start_ms": 0, "end_ms": 30000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 32000, "end_ms": 34600, "speaker_id": "SPEAKER_A"},
            {"start_ms": 40000, "end_ms": 70000, "speaker_id": "SPEAKER_B"},
        ]
        source_samples = [
            _sample(source, 200, 3200),
            _sample(source, 3200, 6200),
            _sample(source, 6200, 9200),
            _sample(candidate, 32200, 34400),
        ]
        target_samples = [
            _sample(target, 40200, 43200),
            _sample(target, 43200, 46200),
            _sample(target, 46200, 49200),
        ]
        clouds = {
            "SPEAKER_A": _cloud("SPEAKER_A", source, source_samples, [0, 1, 2]),
            "SPEAKER_B": _cloud("SPEAKER_B", target, target_samples, [0, 1, 2]),
        }

        result = refine_speaker_turns("conservative", turns, turns, [], clouds)

        self.assertEqual(result.diagnostics["status"], "applied")
        self.assertTrue(result.diagnostics["changes"][0]["short_turn_exception"])
        self.assertEqual(result.turns[1]["speaker_id"], "SPEAKER_B")

    def test_losing_last_speaker_turn_rolls_back(self) -> None:
        source = _vector((0, 1.0))
        target = _vector((1, 1.0))
        candidate = _vector((0, 0.20), (1, 0.80), (2, float(np.sqrt(0.32))))
        turns = [
            {"start_ms": 0, "end_ms": 2600, "speaker_id": "SPEAKER_A"},
            {"start_ms": 3000, "end_ms": 50000, "speaker_id": "SPEAKER_B"},
        ]
        clouds = {
            "SPEAKER_A": _cloud(
                "SPEAKER_A",
                source,
                [
                    _sample(candidate, 200, 2400),
                    _sample(source, 51000, 54000),
                    _sample(source, 54000, 57000),
                    _sample(source, 57000, 60000),
                ],
                [1, 2, 3],
            ),
            "SPEAKER_B": _cloud(
                "SPEAKER_B",
                target,
                [
                    _sample(target, 3200, 6200),
                    _sample(target, 6200, 9200),
                    _sample(target, 9200, 12200),
                ],
                [0, 1, 2],
            ),
        }

        result = refine_speaker_turns("conservative", turns, turns, [], clouds)

        self.assertEqual(result.diagnostics["status"], "rejected")
        self.assertEqual(result.diagnostics["rollback_reason"], "speaker_count_changed")
        self.assertEqual(result.turns[0]["speaker_id"], "SPEAKER_A")

    def test_invalid_mode_is_rejected(self) -> None:
        turns, standard, overlaps, clouds = _regular_fixture()
        with self.assertRaisesRegex(ValueError, "speaker_refinement"):
            refine_speaker_turns("aggressive", turns, standard, overlaps, clouds)


if __name__ == "__main__":
    unittest.main()
