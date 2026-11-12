import unittest

import numpy as np

from mexalign.metrics import dice_score, iou_score


class MetricsTest(unittest.TestCase):
    def test_perfect_overlap(self):
        a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        b = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        self.assertEqual(dice_score(a, b), 1.0)
        self.assertEqual(iou_score(a, b), 1.0)

    def test_partial_overlap(self):
        a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        b = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        self.assertAlmostEqual(dice_score(a, b), 0.5)
        self.assertAlmostEqual(iou_score(a, b), 1.0 / 3.0)

    def test_empty_masks(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(dice_score(a, b), 1.0)
        self.assertEqual(iou_score(a, b), 1.0)

    def test_shape_mismatch_raises(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((5, 5), dtype=np.uint8)
        with self.assertRaises(ValueError):
            dice_score(a, b)
        with self.assertRaises(ValueError):
            iou_score(a, b)


if __name__ == "__main__":
    unittest.main()
