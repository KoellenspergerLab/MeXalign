from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_binary_mask(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    if img.GetDimension() == 3:
        img = img[:, :, 0]
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)
