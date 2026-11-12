"""
Overlap metrics used to evaluate coregistration quality.

Both functions compare two binary masks (cell segmentations) on the same pixel
grid and return a number between 0 and 1:

    0 = no overlap at all (complete misalignment)
    1 = perfect overlap (identical masks)

The masks are expected to already be on the same pixel grid — i.e. the MOVING
mask must have been registered (warped) into the FIXED tissue's coordinate space
before calling these functions.
"""
import numpy as np


def dice_score(fixed: np.ndarray, moving: np.ndarray) -> float:
    """
    Sørensen–Dice coefficient — the standard overlap metric in medical imaging.

    Formula:
        Dice = 2 × |A ∩ B| / (|A| + |B|)

    Where:
        A = foreground pixels of the fixed (reference) segmentation mask
        B = foreground pixels of the registered (warped) moving segmentation mask
        |A ∩ B| = pixels that are foreground in BOTH masks (true overlap)
        |A| + |B| = total foreground pixels across both masks

    Interpretation:
        > 0.90  Excellent alignment
        0.80 – 0.90  Good — suitable for most downstream analyses
        0.65 – 0.80  Acceptable — inspect the visual overlay
        < 0.65  Poor — check inputs and registration settings

    Edge case:
        If both masks are completely empty (all background), returns 1.0
        (two empty masks are considered perfectly "aligned").

    Parameters
    ----------
    fixed   : 2-D numpy array, any dtype — treated as binary (> 0 = foreground)
    moving  : 2-D numpy array, same shape — treated as binary (> 0 = foreground)

    Returns
    -------
    float in [0, 1]
    """
    _validate_shapes(fixed, moving)
    intersection = np.logical_and(fixed > 0, moving > 0).sum(dtype=np.float64)
    fixed_sum    = (fixed  > 0).sum(dtype=np.float64)
    moving_sum   = (moving > 0).sum(dtype=np.float64)
    denom = fixed_sum + moving_sum
    if denom == 0.0:
        return 1.0
    return float(2.0 * intersection / denom)


def iou_score(fixed: np.ndarray, moving: np.ndarray) -> float:
    """
    Jaccard Index (Intersection over Union, IoU).

    Formula:
        IoU = |A ∩ B| / |A ∪ B|

    Where:
        |A ∩ B| = pixels foreground in BOTH masks
        |A ∪ B| = pixels foreground in EITHER mask (union)

    Relationship to Dice:
        IoU is always slightly lower than Dice for the same pair of masks.
        IoU = Dice / (2 - Dice)

    Interpretation:
        > 0.82  Excellent (equivalent to Dice > 0.90)
        0.67 – 0.82  Good
        0.50 – 0.67  Acceptable
        < 0.50  Poor

    Edge case:
        If both masks are completely empty, returns 1.0.

    Parameters
    ----------
    fixed   : 2-D numpy array — treated as binary (> 0 = foreground)
    moving  : 2-D numpy array, same shape — treated as binary (> 0 = foreground)

    Returns
    -------
    float in [0, 1]
    """
    _validate_shapes(fixed, moving)
    intersection = np.logical_and(fixed > 0, moving > 0).sum(dtype=np.float64)
    union        = np.logical_or (fixed > 0, moving > 0).sum(dtype=np.float64)
    if union == 0.0:
        return 1.0
    return float(intersection / union)


def _validate_shapes(fixed: np.ndarray, moving: np.ndarray) -> None:
    """Raise a clear error if the two masks are not on the same pixel grid."""
    if fixed.shape != moving.shape:
        raise ValueError(
            f"Mask shape mismatch: fixed={fixed.shape}, moving={moving.shape}. "
            "Both masks must be on the same pixel grid. "
            "Ensure the moving mask has been registered (warped) into fixed space "
            "before computing overlap metrics."
        )

