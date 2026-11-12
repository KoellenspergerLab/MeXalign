import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from mexalign.io_masks import read_binary_mask


class ReadBinaryMaskTest(unittest.TestCase):
    def test_thresholds_scalar_to_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.tif"
            arr = np.array([[0.0, -2.0, 3.0], [1.0, 0.0, -0.1]], dtype=np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))

            got = read_binary_mask(path)
            expected = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.uint8)

            np.testing.assert_array_equal(got, expected)
            self.assertEqual(got.dtype, np.uint8)

    def test_uses_first_channel_for_vector_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vector_mask.tif"
            ch0 = sitk.GetImageFromArray(np.zeros((4, 4), dtype=np.uint8))
            ch1 = sitk.GetImageFromArray(np.ones((4, 4), dtype=np.uint8))
            vec = sitk.Compose(ch0, ch1)
            sitk.WriteImage(vec, str(path))

            got = read_binary_mask(path)
            np.testing.assert_array_equal(got, np.zeros((4, 4), dtype=np.uint8))

    def test_uses_first_slice_for_3d_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stack_mask.tif"
            arr = np.zeros((2, 3, 5), dtype=np.uint8)
            arr[1, :, :] = 1
            sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))

            got = read_binary_mask(path)
            np.testing.assert_array_equal(got, np.zeros((3, 5), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
