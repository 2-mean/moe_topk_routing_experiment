import unittest

import numpy as np

from moe_topk.metrics import coactivation_matrix, nestedness, top1_agreement, topk_ids_from_logits


class MetricsTest(unittest.TestCase):
    def test_nestedness_cutoff(self):
        small = np.array([[1], [2], [3]])
        large = np.array([[1, 4], [2, 5], [3, 6]])
        self.assertEqual(nestedness(small, large), 1.0)

    def test_nestedness_partial_overlap(self):
        small = np.array([[1, 2], [3, 4]])
        large = np.array([[2, 5, 6], [4, 3, 7]])
        self.assertAlmostEqual(nestedness(small, large), 0.75)

    def test_top1_agreement(self):
        left = np.array([[1], [2], [3]])
        right = np.array([[1, 4], [0, 2], [3, 5]])
        self.assertAlmostEqual(top1_agreement(left, right), 2 / 3)

    def test_topk_ids_from_logits(self):
        logits = np.array([[0.1, 0.5, 0.3], [3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(topk_ids_from_logits(logits, 2), np.array([[1, 2], [0, 2]]))

    def test_coactivation_matrix(self):
        selected = np.array([[0, 2, 3], [2, 3, 1]])
        matrix = coactivation_matrix(selected, 4)
        self.assertGreater(matrix[2, 3], matrix[0, 1])
        self.assertAlmostEqual(matrix.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
