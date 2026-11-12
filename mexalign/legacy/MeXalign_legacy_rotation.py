"""
Professional 2D Coregistration Application (PySide6 + SimpleITK)
----------------------------------------------------------------
Modular refactor of the 2D coregistration pipeline.
"""

import sys
import os
import re
import time
import traceback
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import SimpleITK as sitk

# Try importing scipy for fast distance transform
try:
    from scipy.ndimage import distance_transform_cdt
    from scipy.signal import fftconvolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not found. Install it for faster mask processing.")

from PySide6 import QtCore, QtWidgets, QtGui
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except ImportError:
    # Fallback for older matplotlib versions if needed
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure

# ----------------------------------------------------------------------
# Configuration & Constants
# ----------------------------------------------------------------------

@dataclass
class RegistrationStageConfig:
    """Parameters for a single registration stage."""
    iters: int
    learning_rate: float
    min_step: float
    shrink_factors: List[int]
    smoothing_sigmas: List[float]
    sampling_percentage: float
    # Extended options
    bspline_mesh: Tuple[int, int] = (4, 4)
    metric_bins: int = 64
    max_translation: float = 0.0  # limit in physical units (0 = unlimited)
    # Optimizer Tuning
    relaxation_factor: float = 0.5
    gradient_tolerance: float = 1e-4

@dataclass
class PipelineConfig:
    """Master configuration for the coregistration pipeline."""
    # Input/Output
    directory: str = os.path.dirname(os.path.abspath(__file__))
    output_subdir: str = "Coregistration"
    
    # File Identification
    fixed_mask_filename: str = "ICP_mask.png"
    moving_mask_filename: str = "F_mask.png"
    # fixed_channel_pattern removed as unused
    moving_channel_pattern: str = "F_*"   # e.g. "F_DAPI"
    
    # Physical Properties
    fixed_pixel_size: float = 1.0
    moving_pixel_size: float = 0.2738
    
    # FFT pre-alignment
    use_fft_init: bool = True
    fft_spacing: float = 1.0
    fft_negate: bool = True
    fft_erosion_enabled: bool = False
    fft_erosion_radius: int = 1
    crop_padding_px: int = 100
    
    # Selection
    selected_moving_channels: Optional[List[str]] = None
    
    # Stage 1: Coarse/SDM

    coarse_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=100, learning_rate=1.0, min_step=1e-4, 
        shrink_factors=[8, 4, 2], smoothing_sigmas=[4.0, 2.0, 1.0], sampling_percentage=1.00,
        relaxation_factor=0.7, gradient_tolerance=1e-4
    ))
    sdm_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=800, learning_rate=0.5, min_step=5e-6,
        shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0], sampling_percentage=1.00,
        max_translation=0.0,
        relaxation_factor=0.6, gradient_tolerance=1e-4
    ))
    
    # Mask Erosion (SDM only)
    sdm_erosion_enabled: bool = False
    sdm_erosion_radius: int = 1  # pixels in fixed image
    
    # Stage 2: B-Spline
    bspline_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=200, learning_rate=0.5, min_step=5e-7,
        shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0], sampling_percentage=0.25,
        bspline_mesh=(4, 4), metric_bins=64,
        relaxation_factor=0.5, gradient_tolerance=1e-4
    ))

    transform_stem: str = "Coregistration"
    edge_band_um: float = 10.0
    
    # Advanced IO
    support_rgb: bool = True
    # 'reject' or 'stack_as_channels' (treat z-stack slices as separate channels)
    tiff_3d_mode: str = "reject" 
    # 'uint8' or 'uint16' for vector output (PNG/TIF)
    output_dtype: str = "uint8"

    # Rotation Search
    rotation_enabled: bool = False
    rotation_range: float = 5.0
    rotation_step: float = 1.0

# ----------------------------------------------------------------------
# Robust Image I/O
# ----------------------------------------------------------------------

class ImageIO:
    """Helper for reading/writing images safely across platforms."""
    @staticmethod
    def to_2d(img: sitk.Image) -> sitk.Image:
        """Ensures image is 2D scalar (flattening RGB if needed)."""
        # 1. Handle Channels (RGB -> Scalar)
        if img.GetNumberOfComponentsPerPixel() > 1:
            # Take 1st channel (compatible with grayscale-as-RGB)
            img = sitk.VectorIndexSelectionCast(img, 0)
            
        # 2. Handle 3D Dimensions
        if img.GetDimension() == 3:
            return img[:, :, 0]
            
        return img

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
            with PILImage.open(path) as im:
                # Do NOT convert 'P' to RGBA; we want the indices for masks.
                if im.mode == "1":
                    im = im.convert("L")
                arr = np.array(im)
            
            is_vector = (arr.ndim == 3)
            img = sitk.GetImageFromArray(arr, isVector=is_vector)
            return img
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

# ----------------------------------------------------------------------
# FFT-based coarse translation finder
# ----------------------------------------------------------------------

class FeatureLogic:
    """FFT-based translation estimation on masks."""
    @staticmethod
    def solve_translation_fft(fixed_img: sitk.Image, moving_img: sitk.Image, coarse_spacing: float = 10.0, negate_output: bool = True) -> Tuple[float, float, float]:
        """Estimates pure translation (moving -> fixed) using FFT cross-correlation. Returns (dx, dy, corr_score)."""
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

        max_dim = 4000
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
    def solve_rotation_fft(fixed: sitk.Image, moving: sitk.Image, spacing: float, 
                           rot_range: float, rot_step: float, negate: bool) -> Tuple[float, float, float, float]:
        """
        Exhaustive search for rotation.
        Returns: (best_angle_deg, best_dx, best_dy, best_score)
        """
        best_ang = 0.0
        best_dx = 0.0
        best_dy = 0.0
        best_score = -1e9
        
        # Grid for loop
        if rot_step <= 0: rot_step = 1.0
        
        # Generate angles: -range to +range (inclusive-ish)
        # Use integer steps to avoid float drift in loop
        steps = int(rot_range / rot_step)
        angles = [i * rot_step for i in range(-steps, steps + 1)]
        if 0.0 not in angles: angles.append(0.0)
        angles = sorted(list(set(angles)))
        
        # Center of rotation (Center of Image)
        # We need to rotate the moving image around its physical center
        center = CoregistrationLogic.image_center(moving)
        
        for ang in angles:
            # 1. Rotate Moving Mask (Resampling)
            # Center is physical center of moving image
            tx = sitk.Euler2DTransform()
            tx.SetCenter(center)
            tx.SetAngle(np.radians(ang))
            
            # Use 'spacing' (FFT coarse spacing) to define the resampling grid.
            # reducing resolution here avoids re-downsampling in solve_translation_fft
            scale = spacing / moving.GetSpacing()[0]
            out_sz = [int(sz / scale) for sz in moving.GetSize()]
            out_sp = [spacing, spacing]
            out_org = moving.GetOrigin()
            out_dir = moving.GetDirection()
            
            rotated_c = sitk.Resample(
                moving, out_sz, tx, sitk.sitkLinear,
                out_org, out_sp, out_dir, 0.0, sitk.sitkFloat32
            )
            
            # 2. Solve Translation on Rotated Candidate
            dx, dy, score = FeatureLogic.solve_translation_fft(fixed, rotated_c, coarse_spacing=spacing, negate_output=negate)
            
            if score > best_score:
                best_score = score
                best_ang = ang
                best_dx = dx
                best_dy = dy
        
        return best_ang, best_dx, best_dy, best_score


# ----------------------------------------------------------------------
# Logic & Algorithms
# ----------------------------------------------------------------------

class CoregistrationLogic:
    """Stateless collection of coregistration algorithms."""

    @staticmethod
    def image_center(img: sitk.Image) -> Tuple[float, float]:
        """Physical center based on geometry (not mass)."""
        size = np.array(img.GetSize(), dtype=np.float64)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(2, 2)
        center_index = (size - 1) * 0.5
        return tuple(origin + direction @ (center_index * spacing))
    
    @staticmethod
    def as_affine(tx: sitk.Transform) -> sitk.AffineTransform:
        """Converts a transform (e.g. Euler2D) to an AffineTransform by copying parameters."""
        if isinstance(tx, sitk.AffineTransform):
            return sitk.AffineTransform(tx)
        
        # Create fresh Affine (defaults to identity)
        # Dimension check
        dim = tx.GetDimension()
        aff = sitk.AffineTransform(dim)
        
        # Copy fixed parameters (Center)
        if len(tx.GetFixedParameters()) >= dim:
            aff.SetFixedParameters(tx.GetFixedParameters())
            
        try:
            mat = tx.GetMatrix()
            aff.SetMatrix(mat)
            aff.SetTranslation(tx.GetTranslation())
        except Exception:
            pass
            
        return aff

    @staticmethod
    def mask_to_sdm(binary_mask: sitk.Image) -> sitk.Image:
        """Converts binary mask to Signed Maurer Distance Map."""
        dist = sitk.SignedMaurerDistanceMap(binary_mask, True, False, True)
        return sitk.Cast(dist, sitk.sitkFloat32)

    @staticmethod
    def _boundary_seeds(labels: np.ndarray) -> np.ndarray:
        H, W = labels.shape
        P = np.pad(labels, 1, mode="constant", constant_values=-1)
        seeds = np.zeros((H, W), dtype=bool)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0: continue
                nbr = P[1+dy:1+dy+H, 1+dx:1+dx+W]
                seeds |= (nbr != labels)
        seeds &= (labels > 0)
        return seeds

    @staticmethod
    def _chessboard_distance_per_cell(labels: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy is required for SDM generation (Chessboard Distance Transform).")
            

        seeds = CoregistrationLogic._boundary_seeds(labels)
        inside = (labels > 0) & (~seeds)
        dist = distance_transform_cdt(inside, metric='chessboard').astype(np.float32)
        return dist

    @staticmethod
    def _normalize_per_label(labels: np.ndarray, dist: np.ndarray) -> np.ndarray:
        norm = np.full(labels.shape, np.nan, np.float32)
        if labels.max() == 0: return norm
        
        # Original flattened logic for speed
        L = labels.ravel()
        D = dist.ravel()
        m = L > 0
        L1, D1 = L[m], D[m]
        
        order = np.argsort(L1, kind="mergesort")
        Ls, Ds = L1[order], D1[order]
        
        starts = np.r_[0, np.flatnonzero(Ls[1:] != Ls[:-1]) + 1]
        
        # ReduceAt max
        maxs = np.maximum.reduceat(Ds, starts)
        uniq = Ls[starts]
        
        max_by_label = np.zeros(int(labels.max()) + 1, np.float32)
        max_by_label[uniq] = maxs
        
        denom = max_by_label[labels]
        inside = labels > 0
        
        # Divide
        # Avoid div/0 warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            val = dist[inside] / denom[inside]
        
        norm[inside] = val
        return norm

    @staticmethod
    def per_cell_gradient(label_img: sitk.Image) -> sitk.Image:
        """
        Computes the gradient-like field using the original logic:
        Chessboard distance from boundary + Per-label normalization.
        """
        arr = sitk.GetArrayFromImage(label_img)
        labels = arr.astype(np.int64)
        
        # 1. Distance
        dist = CoregistrationLogic._chessboard_distance_per_cell(labels)
        
        # 2. Normalize
        grad = CoregistrationLogic._normalize_per_label(labels, dist)
        
        # 3. Clean
        grad = np.nan_to_num(grad, nan=0.0).astype(np.float32)
        
        g = sitk.GetImageFromArray(grad)
        g.CopyInformation(label_img)
        return g

    @staticmethod
    def erode_with_gaps(lbl_img: sitk.Image, radius: int) -> sitk.Image:
        """
        Applies 'Boundary Peeling' erosion to a label map.
        Shrinks every cell individually (including touching ones) to create gaps.
        """
        # Ensure int types
        if lbl_img.GetPixelID() not in [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkUInt64]:
            lbl_img = sitk.Cast(lbl_img, sitk.sitkUInt32)
            
        # Detect contours as a plain 0/1 image. LabelContour keeps each cell's ID as the
        # contour value, and BinaryDilate only grows pixels equal to 1, so without the
        # binarisation only the cell with ID 1 would be eroded beyond one layer.
        edges = sitk.Cast(sitk.LabelContour(lbl_img, False) > 0, sitk.sitkUInt8)
        
        # Dilate the edges to create the "remove zone"
        # LabelContour gives the *inner* border. Removing it removes 1 layer.
        # To remove X layers: Dilate the contour by X-1.
        eff_rad = max(0, radius - 1)
        if eff_rad > 0:
            thick_edges = sitk.BinaryDilate(edges, [eff_rad]*2, sitk.sitkBall)
        else:
            thick_edges = edges
            
        # Invert thick edges to make a "keep mask"
        keep = (thick_edges == 0)
        
        # Apply
        return lbl_img * sitk.Cast(keep, lbl_img.GetPixelID())

    @staticmethod
    def run_registration(
        fixed: sitk.Image, 
        moving: sitk.Image, 
        initial_tx: sitk.Transform, 
        cfg: RegistrationStageConfig,
        metric: str = "msq", # 'msq' or 'mattes'
        fixed_mask: Optional[sitk.Image] = None,
        interpolator: int = sitk.sitkLinear,
        observer = None
    ) -> sitk.Transform:
        """Generic registration runner to reduce duplication."""
        reg = sitk.ImageRegistrationMethod()
        
        # Metric
        if metric == "msq":
            reg.SetMetricAsMeanSquares()
        elif metric == "mattes":
            reg.SetMetricAsMattesMutualInformation(cfg.metric_bins)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if fixed_mask:
            reg.SetMetricFixedMask(fixed_mask)
            
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(cfg.sampling_percentage)
        
        # Optimizer
        # Assuming RegularStepGradientDescent for now as it's used in all stages
        reg.SetOptimizerAsRegularStepGradientDescent(
            cfg.learning_rate,
            cfg.min_step,
            cfg.iters,
            cfg.relaxation_factor,
            cfg.gradient_tolerance,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        
        # Multi-resolution
        reg.SetShrinkFactorsPerLevel(cfg.shrink_factors)
        reg.SetSmoothingSigmasPerLevel(cfg.smoothing_sigmas)
        reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
        
        reg.SetInterpolator(interpolator)
        
        # Initialization
        # For BSpline, initial might be the BSpline transform itself.
        # For Rigid/Affine, it's the moving transform.
        # We trust the caller provided a sound initial_tx.
        reg.SetInitialTransform(initial_tx, inPlace=True)
        
        # Observer
        if observer:
             reg.AddCommand(sitk.sitkIterationEvent, lambda: observer(reg))
             
        reg.Execute(fixed, moving)
        return initial_tx # InPlace=True means initial_tx is updated

# ----------------------------------------------------------------------
# Pipeline Class
# ----------------------------------------------------------------------

class CoregistrationPipeline(QtCore.QObject):
    """
    Main pipeline controller. 
    Signals progress and status to the GUI.
    """
    progress = QtCore.Signal(str, int, int) # Stage, iter, total
    message = QtCore.Signal(str)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    overlay_update = QtCore.Signal(object) # Can pass data/image to GUI

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.cfg = config
        self._abort = False
        
        self.fixed_mask: Optional[sitk.Image] = None
        self.moving_mask: Optional[sitk.Image] = None
        self.final_transform = sitk.CompositeTransform(2)
        
        # Threads
        # Threads
        if hasattr(sitk, 'ProcessObject_SetGlobalDefaultNumberOfThreads'):
            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(os.cpu_count())
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(os.cpu_count())

    def abort(self):
        self._abort = True
        self.message.emit("Attempting to abort...")

    def _check_abort(self):
        if self._abort:
            raise InterruptedError("Operation aborted by user.")

    @property
    def output_dir(self):
        return os.path.join(self.cfg.directory, self.cfg.output_subdir)

    def run(self):
        try:
            t_start = time.perf_counter()
            self.message.emit("Initializing pipeline...")
            os.makedirs(self.output_dir, exist_ok=True)

            # 1. Load Masks
            self._check_abort()
            self._step_load_masks()
            
            # FFT pre-alignment (geometry-based center)
            self.message.emit("Running FFT prealignment...")
            initial_tx = self._build_initial_transform()
            self._emit_mask_overlay(initial_tx)
            
            # Restore viz masks if we changed them for FFT erosion
            if self.cfg.fft_erosion_enabled:
                self._update_viz_masks(self.fixed_mask, self.moving_mask)
            
            # --- Dynamic Cropping ---
            self.message.emit("Optimizing with Dynamic Cropping...")
            moving_work = self._crop_moving_to_fixed(initial_tx)
            
            # Pre-compute representations
            self.message.emit("Computing mask representations...")
            # 2. SDM Registration
            self.message.emit("Starting SDM Registration...")
            t_sdm = time.perf_counter()
            
            if self.cfg.sdm_erosion_enabled:
                # Calculate radii
                fix_rad = self.cfg.sdm_erosion_radius
                ratio = self.cfg.fixed_pixel_size / self.cfg.moving_pixel_size
                mov_rad = math.ceil(fix_rad * ratio)
                mov_rad = max(1, int(mov_rad))
                
                self.message.emit(f"Applying Per-Cell Erosion for SDM: Fixed={fix_rad}px, Moving={mov_rad}px")
                
                # Apply Erosion to LABELS using "Boundary Peeling" logic
                f_eroded = CoregistrationLogic.erode_with_gaps(self.fixed_mask, fix_rad)
                m_eroded = CoregistrationLogic.erode_with_gaps(moving_work, mov_rad)
                
                # Convert these eroded labels to binary for registration logic (metric)
                f_bin = (f_eroded > 0)
                m_bin = (m_eroded > 0)
                
                # VISUALIZATION UPDATE: Show these eroded masks
                # Since we used masking, f_eroded/m_eroded still have labels for colored viz
                self.message.emit("Updating visualization to show eroded masks (Gaps Created!)")
                self._update_viz_masks(f_eroded, m_eroded)

            else:
                # No erosion -> standard binary
                f_bin = (self.fixed_mask > 0)
                m_bin = (moving_work > 0)

            f_sdm = CoregistrationLogic.mask_to_sdm(f_bin)
            m_sdm = CoregistrationLogic.mask_to_sdm(m_bin)

            sdm_tx = self._run_sdm_stage(f_sdm, m_sdm, initial_tx)
            self._check_abort()
            self.message.emit(f"SDM Stage took {time.perf_counter() - t_sdm:.2f}s")
            
            # ERASE EROSION VIZ: Restore original masks for B-Spline / Final Viz
            if self.cfg.sdm_erosion_enabled:
                self.message.emit("Restoring full masks for B-Spline visualization...")
                self._update_viz_masks(self.fixed_mask, moving_work)
            
            # 3. B-Spline Stage
            self.message.emit("Preparing B-Spline inputs (Gradients)...")
            t_grads = time.perf_counter()
            
            # Compute gradients
            fixed_grad = CoregistrationLogic.per_cell_gradient(self.fixed_mask)
            moving_grad = CoregistrationLogic.per_cell_gradient(moving_work)
            
            self.message.emit(f"Gradient computation took {time.perf_counter() - t_grads:.2f}s")
            self._check_abort()
            
            self.message.emit("Saving gradient maps...")
            fix_base, _ = os.path.splitext(self.cfg.fixed_mask_filename)
            mov_base, _ = os.path.splitext(self.cfg.moving_mask_filename)
            # Convert to 8-bit for storage; rescale ensures full 0-255 range
            fixed_grad_8 = sitk.Cast(sitk.RescaleIntensity(fixed_grad, 0, 255), sitk.sitkUInt8)
            moving_grad_8 = sitk.Cast(sitk.RescaleIntensity(moving_grad, 0, 255), sitk.sitkUInt8)
            ImageIO.write_image(fixed_grad_8, os.path.join(self.output_dir, f"{fix_base}_gradient.tif"), sitk.sitkUInt8)
            ImageIO.write_image(moving_grad_8, os.path.join(self.output_dir, f"{mov_base}_gradient.tif"), sitk.sitkUInt8)
            
            self.message.emit("Starting B-Spline Registration...")
            t_bspline = time.perf_counter()
            bspline_tx = self._run_bspline_stage(fixed_grad, moving_grad, self.fixed_mask, initial_tx=sdm_tx)
            self._check_abort()
            
            # Clean composition at the absolute end of the pipeline lifecycle
            self.final_transform = sitk.CompositeTransform(2)
            self.final_transform.AddTransform(sdm_tx)
            self.final_transform.AddTransform(bspline_tx)
            self.message.emit(f"B-Spline Stage took {time.perf_counter() - t_bspline:.2f}s")
            
            # Save Final Registered Mask
            self.message.emit("Saving final registered mask...")
            final_mask = sitk.Resample(self.moving_mask, self.fixed_mask, self.final_transform, sitk.sitkNearestNeighbor, 0, self.moving_mask.GetPixelID())
            ImageIO.write_image(final_mask, os.path.join(self.output_dir, "registered_mask.tif"))
            
            # 4. Save Transforms
            self._save_transforms()
            
            # 5. Apply to Channels
            self._apply_to_channels()
            
            total_time = time.perf_counter() - t_start
            self.message.emit(f"Pipeline Finished Successfully in {total_time:.1f}s")

        except InterruptedError:
            self.message.emit("Process Aborted.")
            
        except Exception as e:
            # Capture full traceback for debugging via GUI log
            tb_str = traceback.format_exc()
            self.message.emit(f"CRITICAL ERROR:\n{tb_str}")
            
            # Emit short error for Popup
            self.error.emit(str(e))
        
        finally:
            self.finished.emit()

    def _step_load_masks(self):
        # Paths
        p_fix = os.path.join(self.cfg.directory, self.cfg.fixed_mask_filename)
        p_mov = os.path.join(self.cfg.directory, self.cfg.moving_mask_filename)
        
        if not os.path.exists(p_fix) or not os.path.exists(p_mov):
            raise FileNotFoundError(f"Masks not found: {p_fix} or {p_mov}")
            
        self.fixed_mask = ImageIO.read_image_safe(p_fix, self.cfg.fixed_pixel_size)
        self.moving_mask = ImageIO.read_image_safe(p_mov, self.cfg.moving_pixel_size)
        
        # Keep label IDs stable for large masks and enforce 0 background.
        self.fixed_mask = sitk.Cast(sitk.Mask(self.fixed_mask, self.fixed_mask > 0), sitk.sitkUInt32)
        self.moving_mask = sitk.Cast(sitk.Mask(self.moving_mask, self.moving_mask > 0), sitk.sitkUInt32)

        # Minimal TIFF fallback: only retry when SITK produced an empty moving mask.
        mov_view = sitk.GetArrayViewFromImage(self.moving_mask)
        mov_ext = os.path.splitext(p_mov)[1].lower()
        if np.count_nonzero(mov_view) == 0 and mov_ext in [".tif", ".tiff"]:
            alt_mov = ImageIO._read_with_pil_fallback(p_mov)
            alt_mov = ImageIO.to_2d(alt_mov)
            alt_mov.SetSpacing([float(self.cfg.moving_pixel_size)] * 2)
            alt_mov.SetOrigin([0.0] * 2)
            alt_mov.SetDirection([1.0, 0.0, 0.0, 1.0])
            alt_mov = sitk.Cast(sitk.Mask(alt_mov, alt_mov > 0), sitk.sitkUInt32)
            if np.count_nonzero(sitk.GetArrayViewFromImage(alt_mov)) > 0:
                self.moving_mask = alt_mov

        # Detect if we are dealing with a Label Map (many objects) or Binary Mask
        # If max > 1, assume Label Map
        # Use numpy for safe/fast min/max check avoiding SITK version differences
        arr_view = sitk.GetArrayViewFromImage(self.moving_mask)
        max_val = arr_view.max()
        
        self.is_label_map = (max_val > 1)
        if self.is_label_map:
            self.message.emit(f"Detected Label Map (Max={max_val}). Using distinct cell visualization.")


        # Create downsampled masks for visualization (performance optimization)
        self._update_viz_masks(self.fixed_mask, self.moving_mask)

    def _update_viz_masks(self, fix, mov):
        """Updates the internal visualization images (downsampled)."""
        # Target size ~3000px on long edge
        def _make_viz(img):
            size = img.GetSize()
            factor = max(1, max(size) // 3000)
            if factor > 1:
                return sitk.Shrink(img, [factor]*2)
            return img
            
        self.viz_fixed_mask = _make_viz(fix)
        self.viz_moving_mask = _make_viz(mov)

    def _crop_moving_to_fixed(self, tx: sitk.Transform) -> sitk.Image:
        """
        Crops the moving mask to the bounding box of the fixed mask 
        (projected into moving space via tx) + configurable pixel padding.
        Optimizes large image processing for subsequent stages.
        """
        pad = float(self.cfg.crop_padding_px)
        self.message.emit(f"Calculating ROI for dynamic cropping (padding={int(pad)} px)...")
        
        # 1. Get Fixed Image physical box corners
        f_size = self.fixed_mask.GetSize()
        corners_idx = [
            (0, 0), 
            (f_size[0], 0), 
            (0, f_size[1]), 
            (f_size[0], f_size[1])
        ]
        # Transform indices to physical points
        corners_phys = [self.fixed_mask.TransformIndexToPhysicalPoint(idx) for idx in corners_idx]
        
        corners_mov_phys = [tx.TransformPoint(p) for p in corners_phys]
        
        # 3. Convert mapped physical points to Moving Image Continuous Indices
        # We use continuous index to get float bounds
        corners_mov_idx = [self.moving_mask.TransformPhysicalPointToContinuousIndex(p) for p in corners_mov_phys]
        
        # 4. Find bounding box (min/max)
        xs = [c[0] for c in corners_mov_idx]
        ys = [c[1] for c in corners_mov_idx]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 5. Add padding and clamp to image bounds
        start_x = int(math.floor(min_x - pad))
        start_y = int(math.floor(min_y - pad))
        end_x = int(math.ceil(max_x + pad))
        end_y = int(math.ceil(max_y + pad))
        
        # Clamp to image bounds
        mov_w, mov_h = self.moving_mask.GetSize()
        
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(mov_w, end_x)
        end_y = min(mov_h, end_y)
        
        size_x = end_x - start_x
        size_y = end_y - start_y
        
        if size_x <= 0 or size_y <= 0:
            self.message.emit("Warning: Computed crop is empty/outside. Using full moving image.")
            return self.moving_mask
            
        self.message.emit(f"Cropping Moving Image: Origin({start_x},{start_y}) Size({size_x}x{size_y})")
        
        # 6. Extract ROI
        # RegionOfInterest keeps the correct physical origin/spacing relative to parent
        crop = sitk.RegionOfInterest(self.moving_mask, [size_x, size_y], [start_x, start_y])
        return crop

    def _build_initial_transform(self) -> sitk.Euler2DTransform:
        """Initial transform using FFT translation only (no center overlay)."""
        # If rotation is enabled, we MUST use the center of rotation used in the search (Moving Center).
        # Otherwise, use Fixed Center (legacy behavior, though technically arbitrary for translation only).
        if self.cfg.rotation_enabled:
             center_pt = np.array(CoregistrationLogic.image_center(self.moving_mask))
        else:
             center_pt = np.array(CoregistrationLogic.image_center(self.fixed_mask))
             
        tx = sitk.Euler2DTransform()
        tx.SetCenter(tuple(center_pt))

        if self.cfg.use_fft_init:
            if self.cfg.fft_erosion_enabled:
                # Erode masks for FFT
                fix_rad_fft = self.cfg.fft_erosion_radius
                ratio = self.cfg.fixed_pixel_size / self.cfg.moving_pixel_size
                mov_rad_fft = math.ceil(fix_rad_fft * ratio)
                mov_rad_fft = max(1, int(mov_rad_fft))
                
                self.message.emit(f"Applying Erosion for FFT: Fixed={fix_rad_fft}px, Moving={mov_rad_fft}px")
                fixed_for_fft = CoregistrationLogic.erode_with_gaps(self.fixed_mask, fix_rad_fft)
                moving_for_fft = CoregistrationLogic.erode_with_gaps(self.moving_mask, mov_rad_fft)
                
                # Update viz so the overlay shows what FFT actually saw
                self._update_viz_masks(fixed_for_fft, moving_for_fft)
            else:
                fixed_for_fft = self.fixed_mask
                moving_for_fft = self.moving_mask

            if self.cfg.rotation_enabled:
                self.message.emit(f"Scanning Rotation: +/-{self.cfg.rotation_range} deg, step {self.cfg.rotation_step}")
                ang, dx, dy, score = FeatureLogic.solve_rotation_fft(
                                        fixed_for_fft, moving_for_fft,
                                        spacing=self.cfg.fft_spacing,
                                        rot_range=self.cfg.rotation_range,
                                        rot_step=self.cfg.rotation_step,
                                        negate=self.cfg.fft_negate
                                    )
                # ROTATION COMPENSATION:
                # Rotate the (dx, dy) vector by the angle 'ang' before applying it.
                # This aligns the Visual Shift (Fixed Frame) into the Transform Shift (Rotated Frame).
                rad_ang = np.radians(ang)
                rot_dx = dx * math.cos(rad_ang) - dy * math.sin(rad_ang)
                rot_dy = dx * math.sin(rad_ang) + dy * math.cos(rad_ang)
                
                shift = (rot_dx, rot_dy)
                best_angle_deg = ang
                self.message.emit(f"Best Rotation: {ang} deg (Score={score:.2e}), RawShift=({dx:.1f}, {dy:.1f})")
            else:
                shift_x, shift_y, _ = FeatureLogic.solve_translation_fft(
                    fixed_for_fft, moving_for_fft,
                    coarse_spacing=self.cfg.fft_spacing,
                    negate_output=self.cfg.fft_negate
                )
                shift = (shift_x, shift_y)
                best_angle_deg = 0.0
            
            self.message.emit(f"FFT shift estimate: dx={shift[0]:.2f}, dy={shift[1]:.2f} (applied to moving)")
        else:
            shift = (0.0, 0.0)
            best_angle_deg = 0.0
            self.message.emit("FFT prealignment disabled; no initial shift applied.")

        tx.SetAngle(np.radians(best_angle_deg))
        tx.SetTranslation(tuple(shift))
        self.message.emit(f"Initial transform: Rotation={best_angle_deg} deg, Translation={shift}")
        return tx



    def _emit_mask_overlay(self, tx):
        """Helper to emit the current overlay state using original masks."""
        try:
            # Determine if we should treat this as a label map for visualization
            is_multi_label = getattr(self, 'is_label_map', False)
            if is_multi_label:
                max_val = sitk.GetArrayViewFromImage(self.viz_moving_mask).max()
                if max_val <= 1:
                    is_multi_label = False

            # Resample original moving mask (not SDM)
            if is_multi_label:
                # LABEL MAP: Use Nearest Neighbor to preserve IDs, then scramble for viz
                resampled = sitk.Resample(
                    self.viz_moving_mask, self.viz_fixed_mask, tx, 
                    sitk.sitkNearestNeighbor, 0.0, self.viz_moving_mask.GetPixelID()
                )
                mov_arr = sitk.GetArrayFromImage(resampled).astype(np.uint32)
                
                # Scramble labels to random colors for distinct visualization
                # Hash function: (val * large_prime) % 254 + 1 (if val > 0)
                # We use 0 as background.
                # A simple way to get "random" colors that are consistent
                # Using vectorized numpy operations
                pos_mask = mov_arr > 0
                # Simple linear congruential generator step
                scrambled = np.zeros_like(mov_arr, dtype=np.float32)
                # Map 1..N to 1..255 randomly
                scrambled[pos_mask] = ((mov_arr[pos_mask] * 13219 + 7) % 254) + 1
                
                # Normalize to 0..1
                mov_arr = scrambled / 255.0

            else:
                # BINARY MASK: Use Linear for smoothness/anti-aliasing
                resampled = sitk.Resample(
                    self.viz_moving_mask, self.viz_fixed_mask, tx, 
                    sitk.sitkLinear, 0.0, self.viz_moving_mask.GetPixelID()
                )
                mov_arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
                # Normalize binary (which might be 0..255 or 0..1)
                m_max = mov_arr.max()
                if m_max > 0: mov_arr /= m_max
            
            # Fixed mask viz logic
            ref_arr = sitk.GetArrayFromImage(self.viz_fixed_mask).astype(np.float32)
            r_max = ref_arr.max()
            if r_max > 0: ref_arr /= r_max
            
            self.overlay_update.emit((ref_arr, mov_arr))
        except Exception:
            pass

    def _run_sdm_stage(self, fixed, moving, initial_tx: sitk.Transform) -> sitk.Transform:
        # 1. Coarse SDM (Euler)
        c_cfg = self.cfg.coarse_config
        init = sitk.Euler2DTransform(initial_tx)
        
        # Determine fixed mask for metric if needed
        fix_mask_obj = None
        if self.cfg.edge_band_um > 0:
            fix_mask_obj = sitk.BinaryThreshold(sitk.Abs(fixed), 0, self.cfg.edge_band_um, 1, 0)

        def _obs(r):
            if self._abort:
                r.StopRegistration()
                return
            i = r.GetOptimizerIteration()
            if i == 1 or i % 50 == 0:
                self.progress.emit("SDM (Coarse)", i, c_cfg.iters)
                self._emit_mask_overlay(init)

        CoregistrationLogic.run_registration(
            fixed, moving, init, c_cfg, metric='msq', 
            fixed_mask=fix_mask_obj, observer=_obs
        )
        self._check_abort()
        self.progress.emit("SDM (Coarse)", c_cfg.iters, c_cfg.iters)
        
        # 2. Fine SDM (Affine)
        r_cfg = self.cfg.sdm_config
        
        # Initialize Affine from Coarse result
        init_fine = CoregistrationLogic.as_affine(init)
        
        def _obs_fine(r):
            if self._abort:
                r.StopRegistration()
                return
            i = r.GetOptimizerIteration()
            if i == 1 or i % 50 == 0:
                self.progress.emit("SDM (Fine)", i, r_cfg.iters)
                self._emit_mask_overlay(init_fine)

        CoregistrationLogic.run_registration(
            fixed, moving, init_fine, r_cfg, metric='msq', 
            fixed_mask=fix_mask_obj, observer=_obs_fine
        )
        self._check_abort()
        self.progress.emit("SDM (Fine)", r_cfg.iters, r_cfg.iters)
        
        return init_fine

    def _run_bspline_stage(self, fixed, moving, fixed_mask, initial_tx) -> sitk.Transform:
        b_cfg = self.cfg.bspline_config
        
        # Initial BSpline
        bspline_init = sitk.BSplineTransformInitializer(fixed, list(b_cfg.bspline_mesh))
        
        # IMPORTANT: When running BSpline on top of Rigid, we set MovingInitialTransform
        # But 'reg.SetInitialTransform' in our helper only takes one tx.
        # We need to manually compose or pass specialized logic?
        # Simpler: The helper sets InitialTransform(inPlace=True).
        # We can handle the MovingInitialTransform part by:
        # A) Modifying helper (overkill)
        # B) Composing the logic here manually (defeated purpose)
        # C) Resampling the moving image with initial_tx FIRST, then registering Identity->BSpline?
        #    - This is cleaner but computationally heavy? No, just one resample.
        #    - Actually, standard ITK practice: SetMovingInitialTransform(initial_tx).
        #    - Let's create a specialized 'run_registration_bspline' or adjust the helper?
        #    - Wait, the helper allows us to just run it if we handle the transforms.
        #    - Actually, let's keep it simple: We need to set MovingInitialTransform.
        #    - Let's modify the helper locally or just do it manually here? 
        #    - The User wanted Compactness.
        #    - I'll use the helper but pass a custom 'setup_func' hook? No complicate.
        #    - I Will just replicate the runner logic here concisely OR update the runner to accept 'moving_initial_tx'.
        #    - Let's update the runner to accept 'moving_initial_tx' is cleaner, but I can't change the helper in this tool call (it's already written).
        #    - Alternate: define a local observer and just run it. But I need to set MovingInitialTransform.
        #    - I will manually configure the few distinct lines for BSpline here but reuse the cleaner concepts.
        #    - OR effectively: The B-Spline stage is unique enough I'll just rewrite it cleanly without the helper to avoid backward-patching the helper now.
        #    - WAIT. I can't edit the helper I just wrote in THIS step.
        #    - Actually, I can use the duplicate code but cleaned up.
        #    - Better plan: I'll use `run_registration` for the pure optimization part, but `SetMovingInitialTransform` is critical.
        #    - I'll stick to a clean manual configuration here to respect the specific needs of B-Spline composition.
        
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricFixedMask(fixed_mask)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(b_cfg.sampling_percentage)
        
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetShrinkFactorsPerLevel(b_cfg.shrink_factors)
        reg.SetSmoothingSigmasPerLevel(b_cfg.smoothing_sigmas)
        reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
        
        reg.SetInitialTransform(bspline_init, inPlace=True)
        reg.SetMovingInitialTransform(initial_tx)
        
        # Optimizer match (GD)
        reg.SetOptimizerAsRegularStepGradientDescent(
            b_cfg.learning_rate,
            b_cfg.min_step,
            b_cfg.iters,
            b_cfg.relaxation_factor,
            b_cfg.gradient_tolerance,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        
        def _obs():
            if self._abort:
                reg.StopRegistration()
                return

            i = reg.GetOptimizerIteration()
            if i == 1 or i % 50 == 0:
                self.progress.emit("B-Spline", i, b_cfg.iters)
                comp = sitk.CompositeTransform(2)
                comp.AddTransform(initial_tx)
                comp.AddTransform(bspline_init)
                self._emit_mask_overlay(comp)

        reg.AddCommand(sitk.sitkIterationEvent, _obs)
        reg.Execute(fixed, moving)
        
        self.progress.emit("B-Spline", b_cfg.iters, b_cfg.iters)
        return bspline_init

    def _save_transforms(self):
        stem = self.cfg.transform_stem
        
        # Flatten composite
        # We know we have [Structure: Rigid, BSpline]
        # But Sitk composite is tricky.
        
        # Save composite
        comp_path = os.path.join(self.output_dir, f"{stem}_composite.h5")
        sitk.WriteTransform(self.final_transform, comp_path)
        self.message.emit(f"Saved transform: {comp_path}")
        
        # Save Displacement Field for QC
        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(self.fixed_mask)
        disp_field = disp_filter.Execute(self.final_transform)
        
        ImageIO.write_image(disp_field, os.path.join(self.output_dir, f"{stem}_displacement.mha"))

    def _apply_to_channels(self):
        # 1. Identify files
        inputs = self._scan_inputs()
        
        stems = sorted(inputs.keys())
        # Show full moving filenames for clarity
        display_names = []
        for s in stems:
            mp = inputs[s].get('moving_path')
            if mp:
                display_names.append(os.path.basename(mp))
            else:
                display_names.append(s) 
        
        self.message.emit(f"Found {len(inputs)} channel sets to process: {', '.join(display_names)}")
        
        # Helper inner function for clarity
        def _warp_and_save(stem_name, move_path):
            try:
                # 1. Load as planes (RGB/Stack/Scalar)
                planes, mode = ImageIO.load_as_planes(move_path, self.cfg.tiff_3d_mode)
                
                # 2. Warp each plane
                warped_planes = []
                # Use fixed mask as reference grid
                ref_grid = self.fixed_mask
                
                for p in planes:
                    # Ensure 2D (load_as_planes returns 2D, but just to be safe if Logic changes)
                    # p is already float32 from load_as_planes cast or implicit?
                    # VectorIndexSelectionCast(..., sitkFloat32) handles it for RGB.
                    # Scalar reading might be any type. cast to float for warp.
                    if p.GetPixelID() not in [sitk.sitkFloat32, sitk.sitkFloat64]:
                        p = sitk.Cast(p, sitk.sitkFloat32)
                        
                    # Override spacing if configured (legacy safety)
                    if self.cfg.moving_pixel_size:
                         p.SetSpacing([self.cfg.moving_pixel_size]*2)

                    w = sitk.Resample(
                        p, ref_grid, self.final_transform, 
                        sitk.sitkLinear, 0.0, sitk.sitkFloat32
                    )
                    warped_planes.append(w)
                
                # 3. Save
                original_base = os.path.splitext(os.path.basename(move_path))[0]
                out_name = f"{original_base}_coregistered{os.path.splitext(move_path)[1]}"
                # Force .tif if stack? No, user path rules.
                # If RGB output, original extension (.png/.tif) is fine.
                out_path = os.path.join(self.output_dir, out_name)
                
                ImageIO.save_from_planes(warped_planes, mode, out_path, self.cfg.output_dtype)
                
            except Exception as e:
                self.message.emit(f"Error warping {stem_name}: {e}\n{traceback.format_exc()}")
                print(traceback.format_exc())

        for stem in stems:
            info = inputs[stem]
            mov_path = info.get('moving_path')
            if mov_path:
                _warp_and_save(stem, mov_path)
            
            # Fixed channel copying removed per request

    def _scan_inputs(self) -> Dict[str, Dict]:
        """Scans directory for moving channels based on pattern."""
        files = os.listdir(self.cfg.directory)

        # Match semantics are aligned with GUI filtering:
        # - empty pattern -> all files
        # - pattern with '*' -> wildcard
        # - otherwise -> contains match
        mov_pat = (self.cfg.moving_channel_pattern or "").strip()
        if not mov_pat:
            mov_re = re.compile(".*")
        elif "*" in mov_pat:
            escaped = re.escape(mov_pat).replace(r"\*", ".*")
            mov_re = re.compile(f"^{escaped}$", re.IGNORECASE)
        else:
            escaped = re.escape(mov_pat)
            mov_re = re.compile(f".*{escaped}.*", re.IGNORECASE)
        
        pairs = {}
        
        # Helper to add to pairs
        def add(fname, type_key, regex):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in ['.tif', '.tiff', '.png']: return
            
            if regex.match(fname):
                stem = name
                if stem not in pairs: pairs[stem] = {'stem': stem}
                pairs[stem][type_key] = os.path.join(self.cfg.directory, fname)

        for f in files:
            add(f, 'moving_path', mov_re)
            
        # Filter selection
        if self.cfg.selected_moving_channels is not None:
            # Only keep pairs where the STEM is selected
            filtered = {k: v for k, v in pairs.items() if k in self.cfg.selected_moving_channels}
            return filtered
            
        return pairs


# ----------------------------------------------------------------------
# GUI Implementation (PySide6)
# ----------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity Coregistration")
        self.resize(1200, 850)
        
        self.cfg = PipelineConfig()
        
        # Visualization State
        self._current_ref_arr = None
        self._current_mov_arr = None
        
        # Plot Artists (for persistent zoom)
        self.ref_artist = None
        self.mov_artist = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # LEFT: Tabbed Control Panel
        # We use a tab widget to organize settings without overwhelming the user
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(450)
        
        # TAB 1: Inputs & Channels
        self.tab_inputs = QtWidgets.QWidget()
        self._setup_tab_inputs()
        self.tabs.addTab(self.tab_inputs, "Inputs")
        
        # TAB 2: FFT Settings
        self.tab_fft = QtWidgets.QWidget()
        self._setup_tab_fft()
        self.tabs.addTab(self.tab_fft, "FFT")
        
        # TAB 3: SDM Settings
        self.tab_sdm = QtWidgets.QWidget()
        self._setup_tab_sdm()
        self.tabs.addTab(self.tab_sdm, "SDM")
        
        # TAB 4: B-Spline Settings
        self.tab_bspline = QtWidgets.QWidget()
        self._setup_tab_bspline()
        self.tabs.addTab(self.tab_bspline, "B-Spline")
        
        main_layout.addWidget(self.tabs, 1)
        
        # RIGHT: Visualization
        self.viz_panel = QtWidgets.QWidget()
        self.viz_layout = QtWidgets.QVBoxLayout(self.viz_panel)
        
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("Overlay Preview (Zoom Persists)")
        self.ax.axis('off')
        
        self.toolbar = NavigationToolbar(self.canvas, self.viz_panel)
        
        self.viz_layout.addWidget(self.toolbar)
        self.viz_layout.addWidget(self.canvas)
        
        # Log Box at bottom of right panel
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(200)
        self.viz_layout.addWidget(self.log_box)
        
        main_layout.addWidget(self.viz_panel, 2)
        
        # Apply Logic
        self._apply_defaults()
        # Default directory to script location and scan
        if not self.dir_input.text():
            self.dir_input.setText(self.cfg.directory)
        self._scan_channels()

    def _setup_tab_inputs(self):
        layout = QtWidgets.QVBoxLayout(self.tab_inputs)
        form = QtWidgets.QFormLayout()
        
        # Directories
        self.dir_input = QtWidgets.QLineEdit()
        dir_btn = QtWidgets.QPushButton("Browse...")
        dir_btn.clicked.connect(self._browse_dir)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.dir_input)
        row.addWidget(dir_btn)
        form.addRow("Directory:", row)
        
        self.refresh_btn = QtWidgets.QPushButton("Scan Files")
        self.refresh_btn.clicked.connect(self._scan_channels)
        form.addRow(self.refresh_btn)
        
        # Masks
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("<b>Registration Masks</b>"))
        form_masks = QtWidgets.QFormLayout()
        self.fix_mask_combo = QtWidgets.QComboBox()
        self.mov_mask_combo = QtWidgets.QComboBox()
        self.fix_mask_combo.currentTextChanged.connect(self._update_bs_suggestion)
        form_masks.addRow("Mask of Fixed Image (Reference):", self.fix_mask_combo)
        
        self.fix_ps = QtWidgets.QDoubleSpinBox()
        self.fix_ps.setDecimals(4)
        self.fix_ps.setRange(0, 100)
        self.fix_ps.valueChanged.connect(self._update_bs_suggestion)
        form_masks.addRow("Fixed Image Pixel Size [µm/pixel]:", self.fix_ps)
        
        form_masks.addRow("Mask of Moving Image:", self.mov_mask_combo)
        
        self.mov_ps = QtWidgets.QDoubleSpinBox(); self.mov_ps.setDecimals(4); self.mov_ps.setRange(0, 100)
        form_masks.addRow("Moving Image Pixel Size [µm/pixel]:", self.mov_ps)
        layout.addLayout(form_masks)
        
        layout.addSpacing(10)
        layout.addWidget(QtWidgets.QLabel("<b>Channel Batch Processing</b>"))
        layout.addWidget(QtWidgets.QLabel("Pattern to identify channels associated with the masks above."))
        
        form_pats = QtWidgets.QFormLayout()
        self.mov_pat = QtWidgets.QLineEdit("F_*")
        self.mov_pat.textChanged.connect(self._filter_channels)
        
        form_pats.addRow("Moving Pattern:", self.mov_pat)
        
        layout.addLayout(form_pats)
        
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list)
        
        self.run_btn = QtWidgets.QPushButton("RUN REGISTRATION START")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #007AFF; color: white;")
        self.run_btn.clicked.connect(self._start_run)
        
        self.stop_btn = QtWidgets.QPushButton("STOP / INTERRUPT")
        self.stop_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #FF3B30; color: white;")
        self.stop_btn.clicked.connect(self._stop_run)
        self.stop_btn.setEnabled(False)
        
        
        # Vertical layout for buttons (Stop below Run)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)
        
        layout.addStretch()

    def _setup_tab_fft(self):
        layout = QtWidgets.QVBoxLayout(self.tab_fft)
        
        form = QtWidgets.QFormLayout()
        
        layout.addWidget(QtWidgets.QLabel("<b>FFT Prealignment Settings</b>"))
        
        self.fft_spacing = QtWidgets.QDoubleSpinBox()
        self.fft_spacing.setDecimals(2)
        self.fft_spacing.setRange(0.1, 200.0)
        self.fft_spacing.setSingleStep(0.5)
        
        form.addRow("FFT coarse spacing [µm]:", self.fft_spacing)
        
        self.fft_erode_chk = QtWidgets.QCheckBox("Apply Erosion to Masks")
        form.addRow(self.fft_erode_chk)

        self.fft_erode_rad = QtWidgets.QSpinBox()
        self.fft_erode_rad.setRange(1, 100)
        self.fft_erode_rad.setValue(1)
        
        # Add to form for consistent alignment
        form.addRow("Erosion Radius [Pixel]:", self.fft_erode_rad)

        self.crop_padding_px = QtWidgets.QSpinBox()
        self.crop_padding_px.setRange(0, 10000)
        self.crop_padding_px.setSingleStep(10)
        form.addRow("Padding in pixels:", self.crop_padding_px)
        
        # Rotation Search
        self.fft_rot_chk = QtWidgets.QCheckBox("Enable Rotation Search")
        form.addRow(self.fft_rot_chk)
        
        self.fft_rot_range = QtWidgets.QDoubleSpinBox()
        self.fft_rot_range.setRange(0.0, 180.0)
        self.fft_rot_range.setValue(5.0)
        form.addRow("Rotation Range (+/- deg):", self.fft_rot_range)
        
        self.fft_rot_step = QtWidgets.QDoubleSpinBox()
        self.fft_rot_step.setRange(0.1, 45.0)
        self.fft_rot_step.setValue(1.0)
        self.fft_rot_step.setSingleStep(0.5)
        form.addRow("Step Size (deg):", self.fft_rot_step)
        
        layout.addLayout(form)
        layout.addStretch()

    def _setup_tab_sdm(self):
        layout = QtWidgets.QVBoxLayout(self.tab_sdm)
        
        # Unified Form Layout for consistent alignment vs "Left" issue
        form = QtWidgets.QFormLayout()
        
        # Erosion
        self.chk_erode = QtWidgets.QCheckBox("Apply Erosion to Masks")
        form.addRow(self.chk_erode)
        
        self.erode_rad = QtWidgets.QSpinBox()
        self.erode_rad.setRange(1, 100)
        self.erode_rad.setValue(1)
        form.addRow("Erosion Radius [Pixel]:", self.erode_rad)
        
        # Spacer
        form.addRow(QtWidgets.QLabel(""))
        
        # Coarse
        form.addRow(QtWidgets.QLabel("<b>Coarse Settings</b>"))
        
        self.coarse_iter = QtWidgets.QSpinBox()
        self.coarse_iter.setRange(1, 10000)
        
        self.coarse_lr = QtWidgets.QDoubleSpinBox()
        self.coarse_lr.setRange(0, 10)
        self.coarse_lr.setSingleStep(0.1)
        
        self.coarse_samp = QtWidgets.QDoubleSpinBox()
        self.coarse_samp.setRange(0.01, 1.0)
        self.coarse_samp.setSingleStep(0.05)
        
        form.addRow("Number of Iterations:", self.coarse_iter)
        form.addRow("Learning Rate:", self.coarse_lr)
        form.addRow("Metric Sampling Rate (0-1):", self.coarse_samp)
        
        # Fine
        form.addRow(QtWidgets.QLabel(""))
        form.addRow(QtWidgets.QLabel("<b>Fine SDM (Affine) Settings</b>"))

        self.sdm_iter = QtWidgets.QSpinBox()
        self.sdm_iter.setRange(1, 10000)
        
        self.sdm_lr = QtWidgets.QDoubleSpinBox()
        self.sdm_lr.setRange(0, 10)
        self.sdm_lr.setSingleStep(0.1)
        
        self.sdm_samp = QtWidgets.QDoubleSpinBox()
        self.sdm_samp.setRange(0.01, 1.0)
        self.sdm_samp.setSingleStep(0.05)
        
        form.addRow("Number of Iterations:", self.sdm_iter)
        form.addRow("Learning Rate:", self.sdm_lr)
        form.addRow("Metric Sampling Rate (0-1):", self.sdm_samp)
        
        layout.addLayout(form)
        layout.addStretch()

    def _setup_tab_bspline(self):
        layout = QtWidgets.QVBoxLayout(self.tab_bspline)
        form = QtWidgets.QFormLayout()
        
        self.bs_grid_hint = QtWidgets.QLabel("Suggested Grid: (select Fixed mask)")
        self.bs_grid_hint.setStyleSheet("color: gray; font-style: italic;")
        form.addRow(self.bs_grid_hint)
        
        self.bs_grid = QtWidgets.QLineEdit()
        self.bs_iter = QtWidgets.QSpinBox()
        self.bs_iter.setRange(1, 10000)
        
        self.bs_lr = QtWidgets.QDoubleSpinBox()
        self.bs_lr.setRange(0, 10)
        self.bs_lr.setSingleStep(0.1)
        
        self.bs_samp = QtWidgets.QDoubleSpinBox()
        self.bs_samp.setRange(0.01, 1.0)
        self.bs_samp.setSingleStep(0.05)
        
        # Removed duplicate self.bs_samp definition here
        self.bs_shrink = QtWidgets.QLineEdit()
        self.bs_shrink.setPlaceholderText("e.g. 4,2,1")
        
        form.addRow("Grid (HxW):", self.bs_grid)
        form.addRow("Number of Iterations:", self.bs_iter)
        form.addRow("Learning Rate:", self.bs_lr)
        form.addRow("Metric Sampling Rate (0-1):", self.bs_samp)
        form.addRow("Shrink Factors:", self.bs_shrink)
        
        layout.addLayout(form)
        layout.addStretch()

    def _apply_defaults(self):
        cfg = self.cfg
        self.mov_ps.setValue(cfg.moving_pixel_size)
        self.fix_ps.setValue(cfg.fixed_pixel_size)
        # self.use_fft_chk.setChecked(cfg.use_fft_init) # Removed
        self.fft_spacing.setValue(cfg.fft_spacing)
        if not self.dir_input.text():
            self.dir_input.setText(cfg.directory)
        
        self.fft_erode_chk.setChecked(cfg.fft_erosion_enabled)
        self.fft_erode_rad.setValue(cfg.fft_erosion_radius)
        self.crop_padding_px.setValue(cfg.crop_padding_px)
        self.fft_rot_chk.setChecked(cfg.rotation_enabled)
        self.fft_rot_range.setValue(cfg.rotation_range)
        self.fft_rot_step.setValue(cfg.rotation_step)

        

        
        self.chk_erode.setChecked(cfg.sdm_erosion_enabled)
        self.erode_rad.setValue(cfg.sdm_erosion_radius)
        
        # Coarse
        self.coarse_iter.setValue(cfg.coarse_config.iters)
        self.coarse_lr.setValue(cfg.coarse_config.learning_rate)
        self.coarse_samp.setValue(cfg.coarse_config.sampling_percentage)
        
        # SDM
        self.sdm_iter.setValue(cfg.sdm_config.iters)
        self.sdm_lr.setValue(cfg.sdm_config.learning_rate)
        self.sdm_samp.setValue(cfg.sdm_config.sampling_percentage)
        
        # BSpline
        self.bs_iter.setValue(cfg.bspline_config.iters)
        self.bs_lr.setValue(cfg.bspline_config.learning_rate)
        self.bs_samp.setValue(cfg.bspline_config.sampling_percentage)
        self.bs_shrink.setText(",".join(map(str, cfg.bspline_config.shrink_factors)))
        self.bs_grid.setText(f"{cfg.bspline_config.bspline_mesh[0]},{cfg.bspline_config.bspline_mesh[1]}")
        self._update_bs_suggestion()

    def _browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            self.dir_input.setText(d)
            self._scan_channels()

    def _scan_channels(self):
        d = self.dir_input.text().strip()
        if not d:
            d = self.cfg.directory
            self.dir_input.setText(d)
        if not d or not os.path.exists(d): return
        
        files = sorted(os.listdir(d))
        images = [f for f in files if f.lower().endswith(('.png', '.tif', '.tiff'))]
        
        # Save current selection
        cur_fix = self.fix_mask_combo.currentText()
        cur_mov = self.mov_mask_combo.currentText()
        
        self.fix_mask_combo.clear()
        self.fix_mask_combo.addItems(images)
        self.mov_mask_combo.clear()
        self.mov_mask_combo.addItems(images)
        
        # Restore or Smart Defaults
        if cur_fix and cur_fix in images: self.fix_mask_combo.setCurrentText(cur_fix)
        elif "ICP_mask.png" in images: self.fix_mask_combo.setCurrentText("ICP_mask.png")
        
        if cur_mov and cur_mov in images: self.mov_mask_combo.setCurrentText(cur_mov)
        elif "F_mask.png" in images: self.mov_mask_combo.setCurrentText("F_mask.png")
            
        # Populate list with ALL images, then filter
        self.file_list.clear()
        for f in images:
            item = QtWidgets.QListWidgetItem(f)
            item.setCheckState(QtCore.Qt.Checked)
            self.file_list.addItem(item)
            
        self._filter_channels()

    def _filter_channels(self):
        pat = self.mov_pat.text().strip()
        
        # Construct Search Regex
        if not pat:
            regex = re.compile(".*")
        elif '*' in pat:
            # Wildcard mode
            escaped = re.escape(pat).replace(r"\*", ".*")
            regex = re.compile(f"^{escaped}$", re.IGNORECASE)
        else:
            # "Contains" mode (search behavior)
            escaped = re.escape(pat)
            regex = re.compile(f".*{escaped}.*", re.IGNORECASE)
            
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            # Match against filename
            visible = bool(regex.match(item.text()))
            item.setHidden(not visible)

        
        # Refresh B-spline grid suggestion based on new fixed selection
        self._update_bs_suggestion()
            
    def _log(self, msg):
        self.log_box.appendPlainText(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _update_bs_suggestion(self, text=None):
        """
        Auto-compute B-Spline grid based on the fixed mask size.
        Rule: ceil(H/200), ceil(W/200). Displays physical size using fixed pixel size.
        """
        d = self.dir_input.text().strip()
        fname = self.fix_mask_combo.currentText().strip()

        if not d or not fname:
            self.bs_grid_hint.setText("Suggested Grid: (select Fixed mask)")
            return

        full_path = os.path.join(d, fname)
        if not os.path.exists(full_path):
            self.bs_grid_hint.setText("Suggested Grid: (file not found)")
            return

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(full_path)
            reader.ReadImageInformation()
            w, h = reader.GetSize()  # width, height in pixels

            sugg_w = math.ceil(w / 200.0)
            sugg_h = math.ceil(h / 200.0)

            px_size = self.fix_ps.value()
            phys_w = w * px_size
            phys_h = h * px_size

            self.bs_grid_hint.setText(
                f"Fixed: {h}x{w} px (~{phys_h:.0f}x{phys_w:.0f} µm). "
                f"Suggested Grid: {sugg_h},{sugg_w} (ceil(H/200), ceil(W/200))"
            )
            self.bs_grid.setText(f"{sugg_h},{sugg_w}")
        except (RuntimeError, OSError) as e:
            self.bs_grid_hint.setText(f"Suggested Grid: error reading image ({e})")

    def _plot_overlay(self, overlay_data):
        if not overlay_data: return
        ref, mov = overlay_data
        
        self._current_ref_arr = ref
        self._current_mov_arr = mov
        
        h, w = ref.shape[:2] # handle 3d if color, but viz is likely mask which is 2d
        
        # Initialize if needed or if shape changed
        if (self.ref_artist is None or 
            self.mov_artist is None or 
            self.ref_artist.get_array().shape != ref.shape):
            
            self.ax.clear()
            self.ax.set_axis_off()
            self.ref_artist = self.ax.imshow(ref, cmap='gray', extent=[0, w, h, 0], interpolation='bilinear')
            self.mov_artist = self.ax.imshow(mov, cmap='jet', alpha=0.5, extent=[0, w, h, 0], interpolation='bilinear')
        else:
            # Persistent update (keeps zoom)
            self.ref_artist.set_data(ref)
            self.mov_artist.set_data(mov)
            # Extent is usually static for registration unless shape changes
        
        self.canvas.draw_idle()

    def _start_run(self):
        # Reset visualization state so the new run defines the "Home" zoom level
        self.ref_artist = None
        self.mov_artist = None
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw_idle()
        
        # Reset Toolbar History (Clear "Home" stack)
        # Accessing private _nav_stack is standard workaround for MPL resets
        if hasattr(self.toolbar, '_nav_stack'):
            self.toolbar._nav_stack.clear()

        try:
            cfg = PipelineConfig()
            
            # --- Inputs ---
            cfg.directory = self.dir_input.text().strip() or cfg.directory
            cfg.fixed_mask_filename = self.fix_mask_combo.currentText()
            cfg.moving_mask_filename = self.mov_mask_combo.currentText()
            cfg.moving_pixel_size = self.mov_ps.value()
            cfg.fixed_pixel_size = self.fix_ps.value()
            cfg.use_fft_init = True # Always True
            cfg.fft_spacing = self.fft_spacing.value()
            cfg.moving_channel_pattern = self.mov_pat.text().strip()
            # Fixed channel pattern removed

            cfg.fft_erosion_enabled = self.fft_erode_chk.isChecked()
            cfg.fft_erosion_radius = self.fft_erode_rad.value()
            cfg.crop_padding_px = self.crop_padding_px.value()
            cfg.rotation_enabled = self.fft_rot_chk.isChecked()
            cfg.rotation_range = self.fft_rot_range.value()
            cfg.rotation_step = self.fft_rot_step.value()
            
            # --- SDM ---
            cfg.sdm_erosion_enabled = self.chk_erode.isChecked()
            cfg.sdm_erosion_radius = self.erode_rad.value()
            
            # Propagate simple GUI values to Stage Configs
            # 1. Coarse
            cfg.coarse_config.iters = self.coarse_iter.value()
            cfg.coarse_config.learning_rate = self.coarse_lr.value()
            cfg.coarse_config.sampling_percentage = self.coarse_samp.value()
            # 2. SDM
            cfg.sdm_config.iters = self.sdm_iter.value()
            cfg.sdm_config.learning_rate = self.sdm_lr.value()
            cfg.sdm_config.sampling_percentage = self.sdm_samp.value()
            # --- BSpline ---
            gx, gy = map(int, self.bs_grid.text().split(","))
            cfg.bspline_config.bspline_mesh = (gx, gy)
            cfg.bspline_config.iters = self.bs_iter.value()
            cfg.bspline_config.learning_rate = self.bs_lr.value()
            cfg.bspline_config.sampling_percentage = self.bs_samp.value()
            # Parse Shrink Factors (comma separated)
            try:
                shrinks = [int(x.strip()) for x in self.bs_shrink.text().split(",") if x.strip()]
                if not shrinks: raise ValueError("Empty shrink factors")
                cfg.bspline_config.shrink_factors = shrinks
                
                # Auto-adjust smoothing sigmas to match length (simple logic: half of shrinking)
                # Or just keep default patterns repeated? Better to be safe:
                # Rule of thumb: sigma = shrink / 2.0
                cfg.bspline_config.smoothing_sigmas = [float(s)/2.0 for s in shrinks]
                # Fix: Level 1 usuall has sigma=0 or 0.5?
                # The original default was [4,2,1] -> [2.0, 1.0, 0.0]
                # Let's adjust the last level to 0.0 sigma if shrink is 1
                if shrinks[-1] == 1:
                    cfg.bspline_config.smoothing_sigmas[-1] = 0.0
                    
            except Exception:
                raise ValueError("Invalid Shrink Factors format (use e.g. 4,2,1)")
                
            # Metric Bins default is 64 (set in dataclass), widget removed per user request.
            
            # --- Selection ---
            sel = []
            
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                # Use only visible+checked entries so hidden files cannot affect selection.
                if it.isHidden() or it.checkState() != QtCore.Qt.Checked:
                    continue

                sel.append(os.path.splitext(it.text())[0])

            # Keep deterministic order while removing duplicates.
            sel = list(dict.fromkeys(sel))

            cfg.selected_moving_channels = sel
            
            # Validation
            if not os.path.exists(os.path.join(cfg.directory, cfg.fixed_mask_filename)):
                raise ValueError("Fixed mask not found")

            self.run_btn.setEnabled(False)
            self.thread = QtCore.QThread()
            self.worker = CoregistrationPipeline(cfg)
            self.worker.moveToThread(self.thread)
            
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self._on_finished)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            
            self.worker.message.connect(self._log)
            # Handle error specifically to unlock GUI
            self.worker.error.connect(self._on_error)
            self.worker.progress.connect(self._update_progress)
            self.worker.overlay_update.connect(self._plot_overlay)
            
            self.thread.start()
            
            # Update GUI State
            self.stop_btn.setEnabled(True)
            self.run_btn.setEnabled(False)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def _stop_run(self):
        if hasattr(self, 'worker') and self.worker:
            self.worker.abort()
            self._log("Stopping...")
            self.stop_btn.setEnabled(False) # Prevent multiple clicks

    def _on_error(self, err_msg):
        self._log(f"ERROR OCCURRED: {err_msg}")
        # Unlock buttons
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN REGISTRATION START")
        self.stop_btn.setEnabled(False)

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN REGISTRATION START")
        self.stop_btn.setEnabled(False)
        self._log("DONE.")

    def _update_progress(self, stage, i, total):
        self.run_btn.setText(f"Running: {stage} ({i}/{total})")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
