import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

class ImageIO:
    """Helper for reading/writing images safely across platforms."""
    @staticmethod
    def _positive_count(img2d: sitk.Image) -> int:
        """Counts foreground pixels (>0) using SITK filters."""
        bin_img = sitk.Cast(img2d > 0, sitk.sitkUInt8)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(bin_img)
        if hasattr(stats, "GetSum"):
            return int(round(stats.GetSum()))
        return int(np.count_nonzero(sitk.GetArrayViewFromImage(bin_img)))

    @staticmethod
    def _plane_score(img2d: sitk.Image) -> Tuple[int, float]:
        """
        Scores a candidate 2D plane for mask use.
        Higher is better: more foreground pixels, then larger max intensity.
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img2d)
        max_val = float(stats.GetMaximum())
        if max_val <= 0.0:
            return (0, max_val)
        return (ImageIO._positive_count(img2d), max_val)

    @staticmethod
    def to_2d(img: sitk.Image) -> sitk.Image:
        """Ensures image is 2D scalar by selecting the best non-empty plane/channel."""
        dim = img.GetDimension()
        ncomp = img.GetNumberOfComponentsPerPixel()

        # Fast path: already 2D scalar.
        if dim == 2 and ncomp == 1:
            return img

        best_plane = None
        best_score = (-1, -1.0)

        def _consider(plane: sitk.Image):
            nonlocal best_plane, best_score
            score = ImageIO._plane_score(plane)
            if score > best_score:
                best_score = score
                best_plane = plane

        # 2D Vector: evaluate all components (not just channel 0).
        if dim == 2 and ncomp > 1:
            for c in range(ncomp):
                _consider(sitk.VectorIndexSelectionCast(img, c))

        # 3D Scalar: evaluate all slices.
        elif dim == 3 and ncomp == 1:
            size = img.GetSize()
            for z in range(size[2]):
                _consider(img[:, :, z])

        # 3D Vector: evaluate all (component, slice) planes.
        elif dim == 3 and ncomp > 1:
            size = img.GetSize()
            for c in range(ncomp):
                comp = sitk.VectorIndexSelectionCast(img, c)
                for z in range(size[2]):
                    _consider(comp[:, :, z])

        else:
            raise ValueError(f"Unsupported image format for 2D conversion: dim={dim}, components={ncomp}")

        if best_plane is None:
            raise ValueError("Failed to extract a 2D scalar plane from input image.")
        return best_plane

    @staticmethod
    def copy_geometry(dst: sitk.Image, ref: sitk.Image) -> sitk.Image:
        dst.SetSpacing(ref.GetSpacing())
        dst.SetOrigin(ref.GetOrigin())
        dst.SetDirection(ref.GetDirection())
        return dst

    @staticmethod
    def read_with_geometry(path: str, spacing: Optional[float] = None) -> sitk.Image:
        """Reads image, force 2D, optionally override spacing."""
        # 1. Read
        try:
            img = sitk.ReadImage(path)
        except Exception:
            if sys.platform.startswith("win"):
                img = ImageIO._read_with_pil_fallback(path)
            else:
                raise
        
        # 2. Normalize
        img = ImageIO.to_2d(img)
        
        # 3. Apply Spacing Override
        if spacing and img.GetDimension() == 2:
            img.SetSpacing([float(spacing)] * 2)
            # Reset Origin/Direction for safety in this specific pipeline context
            # (matches original logic of read_image_safe)
            img.SetOrigin([0.0] * 2)
            img.SetDirection([1.0, 0.0, 0.0, 1.0])
            
        return img

    @staticmethod
    def read_image_safe(path: str, spacing: float = 1.0) -> sitk.Image:
        """Legacy wrapper for compatibility."""
        return ImageIO.read_with_geometry(path, spacing)

    @staticmethod
    def _read_with_pil_fallback(path: str) -> sitk.Image:
        try:
            from PIL import Image as PILImage
            # Allow very large scientific masks.
            PILImage.MAX_IMAGE_PIXELS = None

            best_plane = None
            best_score = (-1, -1.0)  # (foreground_count, max_value)

            def _consider_plane(arr2d: np.ndarray):
                nonlocal best_plane, best_score
                if arr2d.size == 0:
                    return
                max_val = float(arr2d.max())
                if max_val <= 0.0:
                    score = (0, max_val)
                else:
                    score = (int(np.count_nonzero(arr2d > 0)), max_val)
                if score > best_score:
                    best_score = score
                    best_plane = arr2d

            with PILImage.open(path) as im:
                n_frames = getattr(im, "n_frames", 1)
                for i in range(n_frames):
                    if n_frames > 1:
                        im.seek(i)
                    frame = im
                    # Keep label values; only normalize 1-bit.
                    if frame.mode == "1":
                        frame = frame.convert("L")
                    arr = np.array(frame)

                    if arr.ndim == 2:
                        _consider_plane(arr)
                    elif arr.ndim == 3:
                        # Evaluate each channel and keep the best non-empty one.
                        for c in range(arr.shape[2]):
                            _consider_plane(arr[:, :, c])

            if best_plane is None:
                raise RuntimeError("PIL fallback did not yield a readable plane.")

            return sitk.GetImageFromArray(best_plane)
        except Exception as e:
            raise RuntimeError(f"Failed to read image {path}: {e}")

    @staticmethod
    def write_image(img: sitk.Image, path: str, pixel_type: Optional[int] = None):
        """Writes an image, creating directories if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Default to 32-bit float for TIFF outputs if not specified
        ext = os.path.splitext(path)[1].lower()
        target_type = pixel_type
        if target_type is None and ext in [".tif", ".tiff"]:
            target_type = sitk.sitkFloat32

        to_write = img if target_type is None else sitk.Cast(img, target_type)

        sitk.WriteImage(to_write, path)

    @staticmethod
    def load_as_planes(path: str, tiff_3d_mode: str = "reject") -> Tuple[List[sitk.Image], str]:
        """
        Reads image and splits into scalar 2D planes.
        Returns (planes, mode_string).
        Modes: 'scalar', 'rgb', 'rgba', 'stack'
        """
        # 1. Read raw
        try:
            img = sitk.ReadImage(path)
        except Exception:
            if sys.platform.startswith("win"):
                img = ImageIO._read_with_pil_fallback(path)
            else:
                raise

        dim = img.GetDimension()
        ncomp = img.GetNumberOfComponentsPerPixel()
        
        # Case A: 2D Scalar
        if dim == 2 and ncomp == 1:
            return [img], "scalar"
            
        # Case B: 2D Vector (RGB/RGBA)
        if dim == 2 and ncomp in [3, 4]:
            planes = [sitk.VectorIndexSelectionCast(img, i, sitk.sitkFloat32) for i in range(ncomp)]
            return planes, ("rgba" if ncomp == 4 else "rgb")
            
        # Case C: 3D Stack (if enabled)
        if dim == 3 and ncomp == 1:
            if tiff_3d_mode == "stack_as_channels":
                # Slice into list of 2D images
                size = img.GetSize()
                planes = []
                for z in range(size[2]):
                    # Slicing in SITK [:,:,z] returns 2D
                    plane = img[:, :, z]
                    # Ensure 2D geometry is clean (Reset origin Z? No, slicing handles it)
                    # But we might want to reset origin to 2D 0,0 for registration consistency
                    # if they are treated as channels associated with the 2D mask.
                    # Yes, typically 'channels' imply they align with the 2D mask.
                    plane.SetOrigin(img.GetOrigin()[:2])
                    plane.SetDirection(img.GetDirection()[:4]) # simplistic 2x2
                    # Actually standard slicing preserves appropriate physics. 
                    # But if we want to register to a 2D fixed mask, we assume 
                    # these slices *are* the channels.
                    planes.append(plane)
                return planes, "stack"
            else:
                raise ValueError(f"3D image detected ({path}) but 3D support is disabled.")

        raise ValueError(f"Unsupported image format: dim={dim}, components={ncomp}")

    @staticmethod
    def save_from_planes(planes: List[sitk.Image], mode: str, path: str, out_dtype: str = "uint8"):
        """Recombines planes and saves."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if mode == "scalar":
            # Just write the single plane
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".tif", ".tiff", ".mha", ".mhd", ".nii", ".nii.gz"]:
                # formats like PNG/JPG require integer types
                dest_type = sitk.sitkUInt8 if out_dtype == "uint8" else sitk.sitkUInt16
                casted = sitk.Cast(planes[0], dest_type)
                sitk.WriteImage(casted, path)
            else:
                # Preserve float32 for scientific formats
                ImageIO.write_image(planes[0], path) 
            return

        elif mode in ["rgb", "rgba"]:
            ext = os.path.splitext(path)[1].lower()
            max_val = max(float(sitk.GetArrayViewFromImage(p).max()) for p in planes)

            # TIFF can preserve high-range RGB data. This avoids uint8 wraparound
            # for inputs like the haut sample, which stores RGB as 16-bit values.
            use_uint16 = out_dtype == "uint16" or (ext in [".tif", ".tiff"] and max_val > 255.0)
            upper = 65535.0 if use_uint16 else 255.0
            scalar_type = sitk.sitkUInt16 if use_uint16 else sitk.sitkUInt8

            casted_planes = [
                sitk.Cast(sitk.Clamp(p, sitk.sitkFloat32, 0.0, upper), scalar_type)
                for p in planes
            ]
            vec = sitk.Compose(casted_planes)
            sitk.WriteImage(vec, path)
            
        elif mode == "stack":
            # Join Series (re-stack in Z)
            vol = sitk.JoinSeries(planes)
            # Write as TIFF (usually supports multi-page/3D)
            # Default to original type or Float32? 
            # Stacks often analysis data -> Float32
            sitk.WriteImage(vol, path)
        else:
            raise ValueError(f"Unknown save mode: {mode}")

