import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    # Optional second label mask (e.g. a different segmentation from the same moving
    # imaging session). Warped with NearestNeighbor so integer cell IDs are preserved.
    second_mask_filename: str = ""
    # Empty by default: an empty pattern matches every image in the folder, so the
    # user explicitly ticks the channels they want in the GUI list instead of
    # relying on a name prefix.
    moving_channel_pattern: str = ""   # e.g. "F_*"

    # Physical Properties
    fixed_pixel_size: float = 1.0
    moving_pixel_size: float = 0.2738

    # Stage 1: FFT pre-alignment (translation + rotation search)
    use_fft_init: bool = True
    fft_spacing: float = 1.0
    fft_negate: bool = True
    fft_erosion_enabled: bool = False
    fft_erosion_radius: int = 1
    crop_padding_px: int = 100

    # Rotation search during FFT pre-alignment.
    # Merged in from the legacy "with rotation" script so there is a single tool.
    # Enabled by default with a tight ±3.0° / 0.25° sweep — cheap enough to leave
    # on, and corrects the small mounting rotations typical of adjacent sections.
    rotation_enabled: bool = True
    rotation_range: float = 3.0   # ± degrees scanned
    rotation_step: float = 0.25   # step size of the angle sweep

    # Selection
    selected_moving_channels: Optional[List[str]] = None

    # Stage 2: Rigid/Affine (Coarse Euler -> Fine Affine)
    # Coarse Euler uses a coarse-to-fine pyramid [8,4,2]. This is far faster and
    # more robust than running at full resolution, and it hands the B-Spline stage
    # a strong starting alignment (so B-Spline converges quickly instead of grinding
    # through every level).
    coarse_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=100, learning_rate=1.0, min_step=1e-4,
        shrink_factors=[8, 4, 2], smoothing_sigmas=[4.0, 2.0, 1.0], sampling_percentage=1.00,
        relaxation_factor=0.7, gradient_tolerance=1e-4
    ))
    affine_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=200, learning_rate=0.5, min_step=5e-6,
        shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0], sampling_percentage=1.00,
        max_translation=0.0,
        relaxation_factor=0.6, gradient_tolerance=1e-4
    ))

    # Affine (fine) plateau early stopping — a conservative SECONDARY stop on top of
    # RegularStepGradientDescent's native minStep / gradient-tolerance / max-iters.
    # After each iteration of the FINAL (full-resolution) affine level, the best MSQ
    # of the two most recent `window`-iteration blocks is compared; if the recent
    # block did not beat the previous block's best MSQ by at least the relative
    # tolerance, the level stops. Applied ONLY at the final level (StopRegistration
    # halts the whole pyramid). window = 0 disables it. The coarse Euler (rigid)
    # sub-stage is native-only (no plateau detector), just per-level reporting.
    affine_early_stop_window: int = 10
    affine_early_stop_relative_tolerance: float = 0.001

    # Rigid/affine-stage metric signal:
    # 'gradient' -> per-cell normalized boundary-distance field (one signal per cell;
    #               far more robust on dense tissue with touching cells, matches the
    #               B-Spline signal). This is the default.
    # 'sdm'      -> classic signed Maurer distance map of the binary mask silhouette.
    rigid_affine_signal: str = "gradient"  # gradient | sdm
    # When the per-cell gradient is normalized to [0, 1] the MSQ metric magnitude is
    # tiny and RegularStepGradientDescent barely moves. Multiplying the signal by a
    # constant scales the metric gradient (not the landscape) back into the range the
    # learning rates were tuned for.
    gradient_signal_scale: float = 1000.0
    # gradient_edges_bright: False -> cell cores bright, edges dark (default).
    # True -> "gradient turned": edges bright, cores dark (legacy gradient_turned).
    gradient_edges_bright: bool = False

    # Correspondence metric (_evaluate_correspondence): an ICP(fixed) cell and an
    # IF(registered) cell are linked when their overlap is at least this fraction of
    # EITHER cell's own area. Linked cells form connected components which are scored
    # as one fused unit, so a 1-ICP-to-N-IF doublet is not punished for being correct.
    # Default 0.50 (a majority-overlap match); editable in the GUI ("Cell-match
    # overlap"). Note the fraction is quantised for small cells (at 15 px, 0.50 == 8 px).
    percell_min_overlap_frac: float = 0.50

    # Seed for the random metric sampler used by the rigid/affine stage (and the
    # B-Spline stage when deterministic sampling is off). A FIXED seed makes
    # repeated runs reproducible — the same pixels are sampled every time. Set to 0
    # to use a wall-clock seed instead (stochastic; a different result each run).
    metric_sampling_seed: int = 1234

    # Mask Erosion (FFT stage). Rigid/affine-stage erosion was dropped from the GUI; the
    # field stays for backward compatibility and defaults to off.
    rigid_affine_erosion_enabled: bool = False
    rigid_affine_erosion_radius: int = 1  # pixels in fixed image

    # Stage 3: B-Spline (iters is the HARD MAXIMUM; LBFGS2 early stopping usually
    # finishes each pyramid level sooner — see bspline_early_stop_* below).
    bspline_config: RegistrationStageConfig = field(default_factory=lambda: RegistrationStageConfig(
        iters=20, learning_rate=0.5, min_step=5e-7,
        shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0], sampling_percentage=0.25,
        bspline_mesh=(4, 4), metric_bins=64,
        relaxation_factor=0.5, gradient_tolerance=1e-4
    ))
    # B-Spline strategy:
    # - mesh refinement uses base mesh scaled per pyramid level
    # - metric defaults to Mattes MI (MSQ remains available)
    # - deterministic sampling evaluates all pixels in the fixed mask
    # bspline_mesh_scale_factors: mesh refinement per pyramid level.
    # [1, 2, 4] -> coarse mesh at low resolution, fine mesh at full resolution.
    bspline_mesh_scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4])

    # bspline_optimizer: 'lbfgs2' (quasi-Newton + line search, faster, no learning
    # rate tuning) or 'regular_step_gd' (classic, fires iteration events every step).
    # The pipeline keeps the GUI responsive for LBFGS2 via the pyramid-level event.
    bspline_optimizer: str = "lbfgs2"  # lbfgs2 | regular_step_gd

    # bspline_metric: 'mattes' (Mutual Information) is slightly more robust than
    # 'msq' (Mean Squares) for tissue-to-tissue alignment.
    bspline_metric: str = "mattes"  # mattes | msq

    # bspline_deterministic_sampling: True = use every pixel inside the fixed
    # mask each step (reproducible, slightly slower). False = random subset.
    bspline_deterministic_sampling: bool = True

    # B-Spline early stopping (LBFGS2 delta-convergence). The window is how many
    # iterations back the metric is compared over; the tolerance is the minimum
    # RELATIVE improvement required across that window. If improvement falls below
    # the tolerance, the current pyramid level stops early — 'iters' above stays the
    # hard cap. window = 0 DISABLES metric-improvement early stopping (LBFGS2 then
    # relies on its solution-accuracy criterion and the iteration cap).
    bspline_early_stop_window: int = 3
    bspline_early_stop_tolerance: float = 0.01   # 0.01 = 1% relative improvement

    edge_band_um: float = 50.0

    # Advanced IO
    support_rgb: bool = True
    # 'reject' or 'stack_as_channels' (treat z-stack slices as separate channels)
    tiff_3d_mode: str = "reject"
    # 'uint8' or 'uint16' for vector output (PNG/TIF)
    output_dtype: str = "uint8"
