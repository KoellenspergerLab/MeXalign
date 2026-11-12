from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from .config import RegistrationStageConfig

try:
    from scipy.ndimage import distance_transform_cdt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
    def _normalize_per_label(labels: np.ndarray, dist: np.ndarray, edges_bright: bool = False) -> np.ndarray:
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

        # Divide, guarding against zero denominators (thin cells with no interior).
        val = np.zeros(int(np.count_nonzero(inside)), dtype=np.float32)
        denom_inside = denom[inside]
        valid = denom_inside > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            val[valid] = dist[inside][valid] / denom_inside[valid]

        if edges_bright:
            # "Gradient turned": edges bright, cell cores dark. Thin cells with no
            # interior are all edge, so keep them fully bright.
            val = 1.0 - val
            val[~valid] = 1.0

        norm[inside] = np.clip(val, 0.0, 1.0)
        return norm

    @staticmethod
    def _chessboard_distance_per_cell(labels: np.ndarray) -> np.ndarray:
        """Chessboard (L-infinity) distance of each cell's interior pixels from that
        cell's own boundary. Each cell's boundary seeds enclose it, so distances
        cannot leak between touching cells."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy is required for the per-cell distance transform.")
        seeds = CoregistrationLogic._boundary_seeds(labels)
        inside = (labels > 0) & (~seeds)
        return distance_transform_cdt(inside, metric="chessboard").astype(np.float32)

    @staticmethod
    def per_cell_gradient(
        label_img: sitk.Image,
        edges_bright: bool = False,
    ) -> sitk.Image:
        """
        Computes the gradient-like field:
        per-cell chessboard distance from boundary + per-label normalization.

        edges_bright=False -> cell cores bright, edges dark (default).
        edges_bright=True  -> edges bright, cores dark ("gradient turned").
        """
        arr = sitk.GetArrayFromImage(label_img)
        labels = arr.astype(np.int64)

        # 1. Distance
        dist = CoregistrationLogic._chessboard_distance_per_cell(labels)

        # 2. Normalize
        grad = CoregistrationLogic._normalize_per_label(labels, dist, edges_bright=edges_bright)

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
        observer = None,
        level_observer = None,
        final_observer = None,
        seed: int = 1234,
    ) -> sitk.Transform:
        """Generic registration runner to reduce duplication.

        The optimizer is RegularStepGradientDescent. A quasi-Newton alternative
        (LBFGS2) was tried and removed: with the low-DOF Euler/Affine transforms
        used here its line search terminates at iteration 0 on this signal.
        """
        reg = sitk.ImageRegistrationMethod()

        # Metric
        metric_name = CoregistrationLogic.normalize_metric_name(metric)
        if metric_name == "msq":
            reg.SetMetricAsMeanSquares()
        elif metric_name == "mattes":
            reg.SetMetricAsMattesMutualInformation(cfg.metric_bins)

        if fixed_mask:
            reg.SetMetricFixedMask(fixed_mask)

        if cfg.sampling_percentage >= 1.0:
            # Use EVERY pixel (deterministic, most stable). At 100% sampling there
            # is no reason to draw a random subset — using all pixels removes the
            # "lucky/unlucky seed" effect entirely, so a good registration doesn't
            # depend on which subset happened to be drawn.
            reg.SetMetricSamplingStrategy(reg.NONE)
        else:
            # Sub-sampling (<100%, chosen for speed): keep it reproducible via a
            # fixed seed so the same pixels are drawn every run.
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(cfg.sampling_percentage, seed)

        # Optimizer
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=cfg.learning_rate,
            minStep=cfg.min_step,
            numberOfIterations=cfg.iters,
            relaxationFactor=cfg.relaxation_factor,
            gradientMagnitudeTolerance=cfg.gradient_tolerance,
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
        
        # Observers. sitkIterationEvent fires per iteration; sitkMultiResolution
        # IterationEvent fires at each pyramid-level change; final_observer runs once
        # after Execute (so a level's stats can be reported with reg still available).
        if observer:
             reg.AddCommand(sitk.sitkIterationEvent, lambda: observer(reg))
        if level_observer:
             reg.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: level_observer(reg))
              
        reg.Execute(fixed, moving)
        if final_observer:
             final_observer(reg)
        return initial_tx # InPlace=True means initial_tx is updated

    @staticmethod
    def bidirectional_best_match(mask_a: np.ndarray, mask_b: np.ndarray,
                                 min_overlap_frac: float = 0.5) -> dict:
        """Bidirectional best-match Dice/IoU — symmetric; every object scored once.

        Each object in BOTH masks independently picks the object in the other mask it
        overlaps MOST, and is scored using the full areas of that pair:
            Dice = 2*I / (a + b),   IoU = I / (a + b - I)
        The best-overlap partner only COUNTS AS A MATCH when the shared area covers at
        least ``min_overlap_frac`` of at least ONE of the two objects (the same
        criterion the correspondence scores use); an object with no qualifying partner
        scores 0. The n_A + n_B directional scores are averaged (Dice/IoU separately).

        Why bidirectional: a one-directional per-cell metric enumerates only the fixed
        mask, so an EXTRA object in the moving mask is invisible (recall without
        precision). Scoring both directions penalises missing AND spurious objects.
        Swapping the masks just swaps the two sums, so the result is identical.

        This is deliberately NOT a global one-to-one assignment (no Hungarian): several
        objects may pick the same partner, which is what exposes splits and merges.

        Implementation note: only pairs that ACTUALLY share pixels are evaluated (via a
        single combined-label pass), so the cost is O(P log P) in the number of
        overlapping pairs — never O(n_A x n_B), which would be billions of pairs on a
        dense sample. Ties on the largest intersection resolve to the lowest partner
        id, matching a plain ``max()`` over ascending ids.
        """
        a = np.asarray(mask_a).ravel().astype(np.int64)
        b = np.asarray(mask_b).ravel().astype(np.int64)
        ids_a = np.unique(a[a > 0])
        ids_b = np.unique(b[b > 0])
        n_a, n_b = int(ids_a.size), int(ids_b.size)
        if n_a == 0 and n_b == 0:      # two empty masks are trivially identical
            return {"dice": 1.0, "iou": 1.0, "n_fixed": 0, "n_registered": 0,
                    "n_unmatched_fixed": 0, "n_unmatched_registered": 0}
        if n_a == 0 or n_b == 0:       # one side empty -> every object scores 0
            return {"dice": 0.0, "iou": 0.0, "n_fixed": n_a, "n_registered": n_b,
                    "n_unmatched_fixed": n_a, "n_unmatched_registered": n_b}

        area_a = np.bincount(a)
        area_b = np.bincount(b)
        both = (a > 0) & (b > 0)
        if both.any():
            stride = int(ids_b.max()) + 1
            uk, cnt = np.unique(a[both] * stride + b[both], return_counts=True)
            pa, pb = uk // stride, uk % stride
        else:
            pa = pb = cnt = np.zeros(0, np.int64)

        def _direction(keys, others, counts, area_self, area_other, ids_self):
            dice = np.zeros(ids_self.size, dtype=np.float64)
            iou = np.zeros(ids_self.size, dtype=np.float64)
            if keys.size:
                # primary key asc, then count asc, then -other asc (= other desc), so the
                # LAST entry per key has the max overlap and, among ties, the lowest id.
                order = np.lexsort((-others, counts, keys))
                k, o, c = keys[order], others[order], counts[order]
                last = np.r_[np.flatnonzero(k[1:] != k[:-1]), k.size - 1]
                bk, bo, bc = k[last], o[last], c[last]
                sa = area_self[bk].astype(np.float64)
                sb = area_other[bo].astype(np.float64)
                inter = bc.astype(np.float64)
                # Same match criterion as the correspondence scores: the best-overlap
                # partner counts only if the overlap covers >= frac of at least ONE of
                # the two objects. Otherwise the object is unmatched and stays at 0.
                meets = (inter >= min_overlap_frac * sa) | (inter >= min_overlap_frac * sb)
                pos = np.searchsorted(ids_self, bk)[meets]
                sa, sb, inter = sa[meets], sb[meets], inter[meets]
                dice[pos] = 2.0 * inter / (sa + sb)
                iou[pos] = inter / (sa + sb - inter)
            return dice, iou

        d_a, i_a = _direction(pa, pb, cnt, area_a, area_b, ids_a)
        d_b, i_b = _direction(pb, pa, cnt, area_b, area_a, ids_b)
        return {
            "dice": float(np.r_[d_a, d_b].mean()),
            "iou": float(np.r_[i_a, i_b].mean()),
            "n_fixed": n_a,
            "n_registered": n_b,
            "n_unmatched_fixed": int((d_a == 0).sum()),
            "n_unmatched_registered": int((d_b == 0).sum()),
        }

    @staticmethod
    def normalize_metric_name(metric: str) -> str:
        metric_name = str(metric).strip().lower()
        if metric_name not in {"msq", "mattes"}:
            raise ValueError(f"Unknown metric: {metric}")
        return metric_name

    @staticmethod
    def normalize_optimizer_name(optimizer: str) -> str:
        optimizer_name = str(optimizer).strip().lower()
        if optimizer_name not in {"regular_step_gd", "lbfgs2"}:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return optimizer_name

    @staticmethod
    def resolve_bspline_scale_factors(
        level_count: int,
        configured_factors: Optional[Sequence[int]],
    ) -> list[int]:
        if level_count <= 0:
            raise ValueError(f"Level count must be positive, got {level_count}.")

        if configured_factors:
            factors = [int(v) for v in configured_factors]
            if len(factors) != level_count:
                raise ValueError(
                    "B-Spline mesh scale factors must match the number of pyramid levels "
                    f"({level_count}), got {len(factors)}."
                )
            if any(v < 1 for v in factors):
                raise ValueError(f"B-Spline mesh scale factors must be >= 1, got {factors}.")
            return factors

        return [2**i for i in range(level_count)]
