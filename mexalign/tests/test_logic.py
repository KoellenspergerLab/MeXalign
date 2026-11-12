import unittest

import numpy as np
import SimpleITK as sitk

from mexalign.core.config import RegistrationStageConfig
from mexalign.core.logic import CoregistrationLogic, SCIPY_AVAILABLE


class CoregistrationLogicTest(unittest.TestCase):
    def test_image_center_respects_geometry(self):
        img = sitk.Image([4, 6], sitk.sitkUInt8)
        img.SetSpacing((2.0, 4.0))
        img.SetOrigin((1.0, 2.0))
        img.SetDirection((0.0, -1.0, 1.0, 0.0))

        cx, cy = CoregistrationLogic.image_center(img)
        self.assertAlmostEqual(cx, -9.0)
        self.assertAlmostEqual(cy, 5.0)

    def test_as_affine_copies_euler_parameters(self):
        tx = sitk.Euler2DTransform()
        tx.SetCenter((4.0, 5.0))
        tx.SetAngle(0.2)
        tx.SetTranslation((2.0, -3.0))

        aff = CoregistrationLogic.as_affine(tx)

        self.assertIsInstance(aff, sitk.AffineTransform)
        np.testing.assert_allclose(aff.GetMatrix(), tx.GetMatrix(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(aff.GetTranslation(), tx.GetTranslation(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(aff.GetFixedParameters(), tx.GetFixedParameters(), rtol=0, atol=1e-8)

    def test_mask_to_sdm_inside_positive_outside_negative(self):
        arr = np.zeros((7, 7), dtype=np.uint8)
        arr[2:5, 2:5] = 1
        img = sitk.GetImageFromArray(arr)

        sdm = CoregistrationLogic.mask_to_sdm(img)
        sdm_arr = sitk.GetArrayFromImage(sdm)

        self.assertEqual(sdm.GetPixelID(), sitk.sitkFloat32)
        self.assertGreater(float(sdm_arr[3, 3]), 0.0)
        self.assertLess(float(sdm_arr[0, 0]), 0.0)

    def test_normalize_per_label_scales_each_cell_independently(self):
        labels = np.array(
            [[0, 1, 1], [2, 2, 0]],
            dtype=np.int64,
        )
        dist = np.array(
            [[0.0, 1.0, 2.0], [2.0, 4.0, 0.0]],
            dtype=np.float32,
        )

        got = CoregistrationLogic._normalize_per_label(labels, dist)

        self.assertTrue(np.isnan(got[0, 0]))
        self.assertTrue(np.isnan(got[1, 2]))
        self.assertAlmostEqual(float(got[0, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(got[0, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(got[1, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(got[1, 1]), 1.0, places=6)

    def test_erode_with_gaps_creates_boundary_gap_between_touching_labels(self):
        labels = np.zeros((5, 6), dtype=np.uint16)
        labels[:, :3] = 1
        labels[:, 3:] = 2
        img = sitk.GetImageFromArray(labels)

        eroded = CoregistrationLogic.erode_with_gaps(img, radius=1)
        arr = sitk.GetArrayFromImage(eroded)

        self.assertEqual(int(arr[2, 2]), 0)
        self.assertEqual(int(arr[2, 3]), 0)
        self.assertEqual(int(arr[2, 1]), 1)
        self.assertEqual(int(arr[2, 4]), 2)

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy required for per_cell_gradient")
    def test_per_cell_gradient_preserves_geometry(self):
        labels = np.zeros((9, 9), dtype=np.uint16)
        labels[2:7, 2:7] = 1
        img = sitk.GetImageFromArray(labels)
        img.SetSpacing((0.7, 1.3))
        img.SetOrigin((3.0, -2.0))
        img.SetDirection((1.0, 0.0, 0.0, 1.0))

        grad = CoregistrationLogic.per_cell_gradient(img)
        arr = sitk.GetArrayFromImage(grad)

        self.assertEqual(grad.GetSpacing(), img.GetSpacing())
        self.assertEqual(grad.GetOrigin(), img.GetOrigin())
        self.assertEqual(grad.GetDirection(), img.GetDirection())
        self.assertGreater(float(arr[4, 4]), 0.0)
        self.assertEqual(float(arr[0, 0]), 0.0)

    def test_run_registration_rejects_unknown_metric(self):
        fixed = sitk.GetImageFromArray(np.zeros((8, 8), dtype=np.float32))
        moving = sitk.GetImageFromArray(np.zeros((8, 8), dtype=np.float32))
        tx = sitk.TranslationTransform(2)
        cfg = RegistrationStageConfig(
            iters=2,
            learning_rate=1.0,
            min_step=1e-4,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            sampling_percentage=1.0,
        )

        with self.assertRaises(ValueError):
            CoregistrationLogic.run_registration(
                fixed=fixed,
                moving=moving,
                initial_tx=tx,
                cfg=cfg,
                metric="bad_metric",
            )

    def test_normalize_optimizer_name_accepts_lbfgs2(self):
        self.assertEqual(CoregistrationLogic.normalize_optimizer_name("LBFGS2"), "lbfgs2")

    def test_normalize_optimizer_name_rejects_unknown(self):
        with self.assertRaises(ValueError):
            CoregistrationLogic.normalize_optimizer_name("adam")

    def test_normalize_metric_name_accepts_mattes(self):
        self.assertEqual(CoregistrationLogic.normalize_metric_name("Mattes"), "mattes")

    def test_resolve_bspline_scale_factors_defaults_to_powers_of_two(self):
        self.assertEqual(
            CoregistrationLogic.resolve_bspline_scale_factors(
                level_count=3,
                configured_factors=None,
            ),
            [1, 2, 4],
        )

    def test_resolve_bspline_scale_factors_uses_configured(self):
        self.assertEqual(
            CoregistrationLogic.resolve_bspline_scale_factors(
                level_count=3,
                configured_factors=[1, 3, 9],
            ),
            [1, 3, 9],
        )

    def test_resolve_bspline_scale_factors_rejects_level_mismatch(self):
        with self.assertRaises(ValueError):
            CoregistrationLogic.resolve_bspline_scale_factors(
                level_count=3,
                configured_factors=[1, 2],
            )

    def test_erode_with_gaps_removes_radius_layers_for_any_cell_id(self):
        # Every cell must lose `radius` pixels per side, whatever its label ID.
        # (LabelContour keeps cell IDs, and BinaryDilate only grows value 1, so an
        # unbinarised contour eroded only the cell with ID 1 beyond one layer.)
        for ids in ((1, 2), (3, 2), (5, 9)):
            arr = np.zeros((40, 80), np.uint16)
            arr[10:25, 10:25] = ids[0]
            arr[10:25, 50:65] = ids[1]
            for radius in (1, 2, 4):
                out = sitk.GetArrayFromImage(
                    CoregistrationLogic.erode_with_gaps(sitk.GetImageFromArray(arr), radius))
                for cell_id in ids:
                    self.assertEqual(int((out == cell_id).sum()), (15 - 2 * radius) ** 2,
                                     msg=f"ids={ids} radius={radius} cell={cell_id}")


if __name__ == "__main__":
    unittest.main()
