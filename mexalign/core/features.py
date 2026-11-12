from typing import Tuple

import numpy as np
import SimpleITK as sitk

try:
    from scipy.signal import fftconvolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class FeatureLogic:
    """FFT-based translation/rotation estimation on masks."""
    @staticmethod
    def solve_translation_fft(fixed_img: sitk.Image, moving_img: sitk.Image, coarse_spacing: float = 10.0, negate_output: bool = True) -> Tuple[float, float, float]:
        """Estimates pure translation (moving -> fixed) using FFT cross-correlation.

        Returns (dx, dy, peak_correlation_score). The score lets solve_rotation_fft
        compare candidate rotation angles; callers that only need the shift can
        ignore the third value.
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not found. FFT prealignment disabled.")
            return (0.0, 0.0, 0.0)



        def get_corners(img):
            sz = img.GetSize()
            idxs = [(0, 0), (sz[0]-1, 0), (sz[0]-1, sz[1]-1), (0, sz[1]-1)]
            return [img.TransformIndexToPhysicalPoint(i) for i in idxs]

        c_f = get_corners(fixed_img)
        c_m = get_corners(moving_img)
        all_c = np.array(c_f + c_m)

        min_x, min_y = np.min(all_c, axis=0)
        max_x, max_y = np.max(all_c, axis=0)

        grid_origin = [min_x, min_y]
        width = int((max_x - min_x) / coarse_spacing) + 1
        height = int((max_y - min_y) / coarse_spacing) + 1

        max_dim = 4000  # cap on the FFT coarse-search grid (px per side)
        if width > max_dim or height > max_dim:
            factor = max(width, height) / max_dim
            coarse_spacing *= factor
            width = int((max_x - min_x) / coarse_spacing) + 1
            height = int((max_y - min_y) / coarse_spacing) + 1

        grid_size = [width, height]
        grid_spacing = [coarse_spacing, coarse_spacing]
        grid_direction = [1.0, 0.0, 0.0, 1.0]

        def _resample(img):
            return sitk.Resample(
                img, grid_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                grid_origin, grid_spacing, grid_direction, 0.0, sitk.sitkFloat32
            )

        fix_c = _resample(fixed_img)
        mov_c = _resample(moving_img)

        def _to_bin(img):
            arr = sitk.GetArrayFromImage(img)
            if arr.min() < 0 and arr.max() > 0:
                return (arr < 0).astype(np.float32)
            return (arr > 0).astype(np.float32)

        f_arr = _to_bin(fix_c)
        m_arr = _to_bin(mov_c)

        corr = fftconvolve(f_arr, m_arr[::-1, ::-1], mode='full')
        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
        peak_val = corr[peak_idx]
        y, x = peak_idx

        m_shape = m_arr.shape
        dy_pix = y - (m_shape[0] - 1)
        dx_pix = x - (m_shape[1] - 1)

        shift_x = dx_pix * coarse_spacing
        shift_y = dy_pix * coarse_spacing
        if negate_output:
            shift_x, shift_y = -shift_x, -shift_y
        return (float(shift_x), float(shift_y), float(peak_val))

    @staticmethod
    def solve_rotation_fft(
        fixed: sitk.Image,
        moving: sitk.Image,
        center: Tuple[float, float],
        spacing: float,
        rot_range: float,
        rot_step: float,
        negate: bool,
    ) -> Tuple[float, float, float, float]:
        """Exhaustive grid search for the best rotation angle in ±rot_range degrees.

        For each candidate angle the moving image is rotated about ``center`` (its
        physical centre), then translation is solved with solve_translation_fft;
        the candidate with the highest cross-correlation peak wins.

        Returns (best_angle_deg, best_dx, best_dy, best_score).
        """
        best_ang = 0.0
        best_dx = 0.0
        best_dy = 0.0
        best_score = -1e9

        if rot_step <= 0:
            rot_step = 1.0

        # Integer-step grid to avoid float drift in the loop.
        steps = int(rot_range / rot_step)
        angles = [i * rot_step for i in range(-steps, steps + 1)]
        if 0.0 not in angles:
            angles.append(0.0)
        angles = sorted(set(angles))

        for ang in angles:
            tx = sitk.Euler2DTransform()
            tx.SetCenter(tuple(center))
            tx.SetAngle(np.radians(ang))

            # Resample at the same coarse FFT spacing so the candidate has the same
            # pixel pitch as solve_translation_fft's grid (avoids double-downsampling).
            scale = spacing / moving.GetSpacing()[0]
            out_sz = [int(sz / scale) for sz in moving.GetSize()]
            out_sp = [spacing, spacing]
            out_org = moving.GetOrigin()
            out_dir = moving.GetDirection()

            rotated_c = sitk.Resample(
                moving, out_sz, tx, sitk.sitkNearestNeighbor,
                out_org, out_sp, out_dir, 0, moving.GetPixelID(),
            )

            dx, dy, score = FeatureLogic.solve_translation_fft(
                fixed, rotated_c,
                coarse_spacing=spacing, negate_output=negate,
            )

            if score > best_score:
                best_score = score
                best_ang = ang
                best_dx = dx
                best_dy = dy

        return best_ang, best_dx, best_dy, best_score
