import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from mexalign.contracts import RunInput
from mexalign.pipeline import evaluate_registration


class EvaluateRegistrationTest(unittest.TestCase):
    @staticmethod
    def _write_mask(path: Path, arr: np.ndarray) -> None:
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))

    def test_returns_scores_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = root / "fixed.tif"
            moving = root / "moving.tif"
            registered = root / "registered.tif"

            base = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            self._write_mask(fixed, base)
            self._write_mask(moving, np.array([[1, 0], [0, 1]], dtype=np.uint8))
            self._write_mask(registered, base)

            result = evaluate_registration(
                RunInput(
                    fixed_mask_path=fixed,
                    moving_mask_path=moving,
                    registered_mask_path=registered,
                )
            )

            self.assertEqual(result.dice, 1.0)
            self.assertEqual(result.iou, 1.0)
            self.assertEqual(result.metadata["fixed_mask_path"], str(fixed))
            self.assertEqual(result.metadata["registered_mask_path"], str(registered))
            self.assertEqual(result.metadata["moving_mask_path"], str(moving))
            self.assertEqual(result.metadata["metrics"], "dice,iou")

    def test_shape_mismatch_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = root / "fixed.tif"
            moving = root / "moving.tif"
            registered = root / "registered.tif"

            self._write_mask(fixed, np.zeros((4, 4), dtype=np.uint8))
            self._write_mask(moving, np.zeros((4, 4), dtype=np.uint8))
            self._write_mask(registered, np.zeros((5, 5), dtype=np.uint8))

            with self.assertRaises(ValueError):
                evaluate_registration(
                    RunInput(
                        fixed_mask_path=fixed,
                        moving_mask_path=moving,
                        registered_mask_path=registered,
                    )
                )


if __name__ == "__main__":
    unittest.main()
