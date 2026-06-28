import unittest
from itertools import combinations

import numpy as np

from moe_topk.ranking import (
    displacement_by_source_rank,
    empirical_null_ceiling_adjusted_overlap,
    hypergeometric_calibrated_overlap,
    hypergeometric_intersection_pmf,
    inverse_ranks_from_logits,
    kendall_tau_from_ranks,
    mean_absolute_rank_displacement,
    normalized_footrule_similarity,
    rank_biased_overlap,
    rank_transition_counts,
    selected_bitmasks_from_ranks,
    selected_expert_order_stats,
    summarize_selected_expert_order_stats,
    topm_overlap_curve,
)


class FullRankingTest(unittest.TestCase):
    def test_identical_rankings_are_diagonal(self):
        logits = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 4.0, 2.0]])
        ranks = inverse_ranks_from_logits(logits, chunk_size=1)
        counts = rank_transition_counts(ranks, ranks, chunk_size=1)
        self.assertEqual(int(counts.sum()), 8)
        np.testing.assert_array_equal(counts, np.diag([2, 2, 2, 2]))
        np.testing.assert_allclose(topm_overlap_curve(counts), np.ones(4))
        self.assertAlmostEqual(rank_biased_overlap(topm_overlap_curve(counts), 0.9), 1.0)
        self.assertEqual(mean_absolute_rank_displacement(counts), 0.0)
        self.assertEqual(normalized_footrule_similarity(counts), 1.0)
        self.assertEqual(kendall_tau_from_ranks(ranks, ranks), 1.0)

    def test_reversed_ranking_tracks_full_transition(self):
        left = inverse_ranks_from_logits(np.array([[4.0, 3.0, 2.0, 1.0]]))
        right = inverse_ranks_from_logits(np.array([[1.0, 2.0, 3.0, 4.0]]))
        counts = rank_transition_counts(left, right)
        expected = np.fliplr(np.eye(4, dtype=np.int64))
        np.testing.assert_array_equal(counts, expected)
        np.testing.assert_allclose(topm_overlap_curve(counts), [0.0, 0.0, 2 / 3, 1.0])
        self.assertEqual(mean_absolute_rank_displacement(counts), 2.0)
        self.assertEqual(normalized_footrule_similarity(counts), 0.0)
        self.assertEqual(kendall_tau_from_ranks(left, right), -1.0)
        mean_target, mean_absolute = displacement_by_source_rank(counts)
        np.testing.assert_allclose(mean_target, [4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(mean_absolute, [3.0, 1.0, 1.0, 3.0])

    def test_selected_metrics_separate_set_and_order_changes(self):
        left = inverse_ranks_from_logits(np.array([[4.0, 3.0, 2.0, 1.0]]))
        right = inverse_ranks_from_logits(np.array([[3.0, 4.0, 1.0, 2.0]]))
        stats = selected_expert_order_stats(left, right, cutoff_a=2, cutoff_b=3)
        summary = summarize_selected_expert_order_stats(stats, cutoff_a=2, cutoff_b=3)

        self.assertAlmostEqual(summary["selected_containment_a"], 1.0)
        self.assertAlmostEqual(summary["selected_containment_b"], 2 / 3)
        self.assertAlmostEqual(summary["selected_jaccard"], 2 / 3)
        self.assertAlmostEqual(summary["hypergeometric_calibrated_overlap"], 1.0)
        self.assertAlmostEqual(summary["selected_union_kendall_tau"], 1 / 3)
        self.assertAlmostEqual(summary["common_selected_kendall_tau"], -1.0)
        self.assertAlmostEqual(summary["common_pairwise_order_agreement"], 0.0)
        self.assertAlmostEqual(summary["common_selected_exact_order_fraction"], 0.0)
        self.assertAlmostEqual(summary["full_containment_record_fraction"], 1.0)
        self.assertAlmostEqual(summary["ordered_containment_record_fraction"], 0.0)
        self.assertAlmostEqual(summary["contained_exact_order_fraction"], 0.0)
        self.assertAlmostEqual(summary["common_selected_record_mean_tau"], -1.0)
        self.assertAlmostEqual(summary["common_selected_valid_record_fraction"], 1.0)
        self.assertAlmostEqual(summary["common_pair_coverage"], 1.0)

    def test_common_selected_order_is_undefined_for_cutoff_one(self):
        ranks = inverse_ranks_from_logits(np.array([[4.0, 3.0, 2.0, 1.0]]))
        stats = selected_expert_order_stats(ranks, ranks, cutoff_a=1, cutoff_b=2)
        summary = summarize_selected_expert_order_stats(stats, cutoff_a=1, cutoff_b=2)

        self.assertTrue(np.isnan(summary["common_selected_kendall_tau"]))
        self.assertTrue(np.isnan(summary["common_pairwise_order_agreement"]))
        self.assertTrue(np.isnan(summary["common_selected_exact_order_fraction"]))
        self.assertEqual(summary["full_containment_record_fraction"], 1.0)
        self.assertTrue(np.isnan(summary["ordered_containment_record_fraction"]))
        self.assertTrue(np.isnan(summary["contained_exact_order_fraction"]))
        self.assertEqual(summary["common_selected_valid_record_fraction"], 0.0)
        self.assertTrue(np.isnan(summary["common_pair_coverage"]))

    def test_selected_metrics_match_brute_force(self):
        rng = np.random.default_rng(17)
        left = inverse_ranks_from_logits(rng.normal(size=(19, 6)))
        right = inverse_ranks_from_logits(rng.normal(size=(19, 6)))
        cutoff_a, cutoff_b = 2, 4
        summary = summarize_selected_expert_order_stats(
            selected_expert_order_stats(left, right, cutoff_a, cutoff_b, chunk_size=3),
            cutoff_a,
            cutoff_b,
        )

        intersections = []
        jaccards = []
        union_taus = []
        common_taus = []
        common_signed_sum = 0
        common_pair_count = 0
        common_exact_count = 0
        full_containment_count = 0
        ordered_containment_count = 0
        for ranks_a, ranks_b in zip(left, right):
            selected_a = set(np.flatnonzero(ranks_a <= cutoff_a).tolist())
            selected_b = set(np.flatnonzero(ranks_b <= cutoff_b).tolist())
            common = sorted(selected_a & selected_b)
            union = sorted(selected_a | selected_b)
            intersections.append(len(common))
            jaccards.append(len(common) / len(union))

            def tau(experts):
                signs = [
                    1 if (ranks_a[i] < ranks_a[j]) == (ranks_b[i] < ranks_b[j]) else -1
                    for i, j in combinations(experts, 2)
                ]
                return float(np.mean(signs)) if signs else float("nan"), signs

            union_tau, _ = tau(union)
            common_tau, common_signs = tau(common)
            union_taus.append(union_tau)
            if np.isfinite(common_tau):
                common_taus.append(common_tau)
                common_exact_count += int(common_tau == 1.0)
            full_containment = len(common) == cutoff_a
            full_containment_count += int(full_containment)
            ordered_containment_count += int(full_containment and common_tau == 1.0)
            common_signed_sum += sum(common_signs)
            common_pair_count += len(common_signs)

        np.testing.assert_allclose(
            summary["selected_containment_a"], np.mean(intersections) / cutoff_a
        )
        np.testing.assert_allclose(
            summary["selected_containment_b"], np.mean(intersections) / cutoff_b
        )
        np.testing.assert_allclose(summary["selected_jaccard"], np.mean(jaccards))
        np.testing.assert_allclose(summary["selected_union_kendall_tau"], np.mean(union_taus))
        np.testing.assert_allclose(
            summary["common_selected_kendall_tau"], common_signed_sum / common_pair_count
        )
        np.testing.assert_allclose(
            summary["common_pairwise_order_agreement"],
            (common_signed_sum / common_pair_count + 1.0) / 2.0,
        )
        np.testing.assert_allclose(
            summary["common_selected_exact_order_fraction"],
            common_exact_count / len(common_taus),
        )
        np.testing.assert_allclose(
            summary["full_containment_record_fraction"], full_containment_count / len(left)
        )
        np.testing.assert_allclose(
            summary["ordered_containment_record_fraction"],
            ordered_containment_count / len(left),
        )
        np.testing.assert_allclose(
            summary["contained_exact_order_fraction"],
            ordered_containment_count / full_containment_count,
        )
        np.testing.assert_allclose(
            summary["common_selected_record_mean_tau"], np.mean(common_taus)
        )
        np.testing.assert_allclose(
            summary["common_selected_valid_record_fraction"], len(common_taus) / len(left)
        )
        np.testing.assert_allclose(
            summary["common_pair_coverage"],
            common_pair_count / (len(left) * (cutoff_a * (cutoff_a - 1) / 2)),
        )

    def test_hypergeometric_calibration_maps_null_and_ceiling(self):
        pmf = hypergeometric_intersection_pmf(4, 1, 2)
        np.testing.assert_allclose(pmf, [0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            hypergeometric_calibrated_overlap(np.array([50, 50, 0, 0, 0]), 4, 1, 2),
            0.0,
        )
        self.assertAlmostEqual(
            hypergeometric_calibrated_overlap(np.array([0, 100, 0, 0, 0]), 4, 1, 2),
            1.0,
        )
        self.assertAlmostEqual(
            hypergeometric_calibrated_overlap(np.array([100, 0, 0, 0, 0]), 4, 1, 2),
            -1.0,
        )

    def test_empirical_adjustment_uses_random_and_optimal_matchings(self):
        ranks_a = inverse_ranks_from_logits(
            np.array([[4.0, 2.0, 1.0], [1.0, 4.0, 2.0], [2.0, 1.0, 4.0]])
        )
        ranks_b = ranks_a.copy()
        masks_a = selected_bitmasks_from_ranks(ranks_a, 1)
        masks_b = selected_bitmasks_from_ranks(ranks_b, 1)
        metadata = {
            "sample_ids": np.array([0, 1, 2]),
            "task_ids": np.array([0, 0, 0]),
            "layer_ids": np.array([0, 0, 0]),
            "token_positions": np.array([0, 0, 0]),
        }
        aligned = empirical_null_ceiling_adjusted_overlap(
            masks_a, masks_b, cutoff_a=1, **metadata
        )
        self.assertAlmostEqual(aligned["empirical_observed_overlap"], 1.0)
        self.assertAlmostEqual(aligned["empirical_null_overlap"], 0.0)
        self.assertAlmostEqual(aligned["empirical_ceiling_overlap"], 1.0)
        self.assertAlmostEqual(aligned["frequency_adjusted_overlap"], 1.0)
        self.assertAlmostEqual(aligned["empirical_ceiling_utilization"], 1.0)

        reversed_alignment = empirical_null_ceiling_adjusted_overlap(
            masks_a, np.roll(masks_b, -1), cutoff_a=1, **metadata
        )
        self.assertAlmostEqual(reversed_alignment["empirical_observed_overlap"], 0.0)
        self.assertAlmostEqual(reversed_alignment["empirical_null_overlap"], 0.5)
        self.assertAlmostEqual(reversed_alignment["empirical_ceiling_overlap"], 1.0)
        self.assertAlmostEqual(reversed_alignment["frequency_adjusted_overlap"], -1.0)
        self.assertAlmostEqual(reversed_alignment["empirical_ceiling_utilization"], -1.0)


if __name__ == "__main__":
    unittest.main()
