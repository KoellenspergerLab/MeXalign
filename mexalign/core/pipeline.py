import csv
import dataclasses
import datetime
import math
import os
import re
import time
import traceback
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk

from .config import PipelineConfig
from .features import FeatureLogic
from .image_io import ImageIO
from .logic import CoregistrationLogic
from .signals import LocalSignal

class CoregistrationPipeline:
    """
    Main pipeline controller. 
    Signals progress and status to the GUI.
    """
    def __init__(self, config: PipelineConfig):
        self.progress = LocalSignal()  # stage, iter, total
        self.message = LocalSignal()
        self.finished = LocalSignal()
        self.error = LocalSignal()
        self.overlay_update = LocalSignal()
        self.percell_result = LocalSignal()  # per-cell metric summary (dict)
        self.percell = None                  # last per-cell summary dict

        self.cfg = config
        self._abort = False

        # Wall-clock seconds per stage, filled in as run() proceeds. Read by the
        # sweep runner for its comparison report — more robust than parsing logs.
        # Keys: fft, gradient, rigid_affine, bspline, total.
        # NOTE: 'gradient' and 'rigid_affine' are DISJOINT (the gradient time is not
        # folded into the rigid/affine figure), unlike the "Rigid/affine stage took"
        # log line, which covers gradient + registration together.
        self.timings: Dict[str, float] = {}

        # Correspondence metric filled in by _evaluate_correspondence(): 1:1-only and
        # fused ("corrected") Dice/IoU + ICP-only / IF-only / doublet / merge counts.
        self.correspondence: Optional[dict] = None

        # Bidirectional best-match Dice/IoU filled in by _evaluate_bidirectional().
        self.bidirectional: Optional[dict] = None
        # Global silhouette Dice/IoU filled in by _evaluate_and_record().
        self.global_dice: Optional[float] = None
        self.global_iou: Optional[float] = None

        # Folding QC filled in by _evaluate_jacobian(): min/max det(J), folded-pixel
        # counts, and the affine determinant the field is centred on. None if skipped.
        self.jacobian: Optional[Dict[str, float]] = None

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
            t_fft = time.perf_counter()
            initial_tx = self._build_initial_transform()
            self.timings["fft"] = time.perf_counter() - t_fft
            self.message.emit(f"FFT prealignment took {self.timings['fft']:.2f}s")
            self._emit_mask_overlay(initial_tx)
            
            # Restore viz masks if we changed them for FFT erosion
            if self.cfg.fft_erosion_enabled:
                self._update_viz_masks(self.fixed_mask, self.moving_mask)
            
            # --- Dynamic Cropping ---
            self.message.emit("Optimizing with Dynamic Cropping...")
            moving_work = self._crop_moving_to_fixed(initial_tx)
            
            # Pre-compute representations
            edges_bright = bool(self.cfg.gradient_edges_bright)
            rigid_affine_signal_mode = str(getattr(self.cfg, "rigid_affine_signal", "gradient")).lower()
            scale = float(self.cfg.gradient_signal_scale)

            # 2. Rigid/affine registration (coarse Euler -> fine Affine)
            self.message.emit(f"Starting rigid/affine registration (signal={rigid_affine_signal_mode})...")
            t_stage2 = time.perf_counter()

            # Optional per-cell erosion (creates gaps between touching cells).
            if self.cfg.rigid_affine_erosion_enabled:
                fix_rad = self.cfg.rigid_affine_erosion_radius
                ratio = self.cfg.fixed_pixel_size / self.cfg.moving_pixel_size
                mov_rad = max(1, int(math.ceil(fix_rad * ratio)))
                self.message.emit(f"Applying per-cell erosion (rigid/affine stage): Fixed={fix_rad}px, Moving={mov_rad}px")
                f_labels = CoregistrationLogic.erode_with_gaps(self.fixed_mask, fix_rad)
                m_labels = CoregistrationLogic.erode_with_gaps(moving_work, mov_rad)
                self.message.emit("Updating visualization to show eroded masks (Gaps Created!)")
                self._update_viz_masks(f_labels, m_labels)
            else:
                f_labels = self.fixed_mask
                m_labels = moving_work

            stage2_fix_mask = None
            if rigid_affine_signal_mode == "gradient":
                # Per-cell gradient signal: one little signal per cell, scaled so the
                # MSQ metric has enough magnitude to drive the optimizer. Matches the
                # B-Spline stage's signal so both stages "see" the same features.
                self.message.emit("Computing per-cell gradients for the rigid/affine stage...")
                t_grads = time.perf_counter()
                f_signal = CoregistrationLogic.per_cell_gradient(f_labels, edges_bright=edges_bright) * scale
                m_signal = CoregistrationLogic.per_cell_gradient(m_labels, edges_bright=edges_bright) * scale
                # Restrict the MSQ metric to where the fixed image actually has cells.
                stage2_fix_mask = sitk.Cast(f_labels > 0, sitk.sitkUInt8)
                self.timings["gradient"] = time.perf_counter() - t_grads
                self.message.emit(
                    f"Per-cell gradient (edges_bright={edges_bright}, "
                    f"scaled x{scale:g}) took {self.timings['gradient']:.2f}s"
                )
            else:
                # Classic signed-distance-map silhouette signal.
                t_grads = time.perf_counter()
                f_signal = CoregistrationLogic.mask_to_sdm(f_labels > 0)
                m_signal = CoregistrationLogic.mask_to_sdm(m_labels > 0)
                self.timings["gradient"] = time.perf_counter() - t_grads

            # Time the registration itself, separate from the signal construction
            # above (t_stage2 spans both, so it is not usable as a "rigid/affine" figure).
            t_reg = time.perf_counter()
            affine_tx = self._run_rigid_affine_stage(f_signal, m_signal, initial_tx, fix_mask_override=stage2_fix_mask)
            self.timings["rigid_affine"] = time.perf_counter() - t_reg
            self._check_abort()
            self.final_transform.AddTransform(affine_tx)
            self.message.emit(
                f"Rigid/affine registration took {self.timings['rigid_affine']:.2f}s "
                f"(+ {self.timings.get('gradient', 0.0):.2f}s gradient "
                f"= {time.perf_counter() - t_stage2:.2f}s stage total)"
            )

            # ERASE EROSION VIZ: Restore original masks for B-Spline / Final Viz
            if self.cfg.rigid_affine_erosion_enabled:
                self.message.emit("Restoring full masks for B-Spline visualization...")
                self._update_viz_masks(self.fixed_mask, moving_work)

            # 3. B-Spline Stage — reuse the rigid/affine gradient signals when nothing changed.
            if rigid_affine_signal_mode == "gradient" and not self.cfg.rigid_affine_erosion_enabled:
                self.message.emit("Reusing per-cell gradients for B-Spline.")
                fixed_grad = f_signal
                moving_grad = m_signal
            else:
                self.message.emit("Preparing B-Spline inputs (per-cell gradients on full masks)...")
                t_grads_bs = time.perf_counter()
                fixed_grad = CoregistrationLogic.per_cell_gradient(self.fixed_mask, edges_bright=edges_bright) * scale
                moving_grad = CoregistrationLogic.per_cell_gradient(moving_work, edges_bright=edges_bright) * scale
                bs_grad_secs = time.perf_counter() - t_grads_bs
                # Recomputed rather than reused, so this is additional gradient cost.
                self.timings["gradient"] = self.timings.get("gradient", 0.0) + bs_grad_secs
                self.message.emit(f"B-Spline gradient computation took {bs_grad_secs:.2f}s")

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
            bspline_tx = self._run_bspline_stage(fixed_grad, moving_grad, self.fixed_mask, initial_tx=self.final_transform)
            self.timings["bspline"] = time.perf_counter() - t_bspline
            self._check_abort()
            self.final_transform.AddTransform(bspline_tx)
            self.message.emit(f"B-Spline Stage took {self.timings['bspline']:.2f}s")
            
            # Save Final Registered Mask
            self.message.emit("Saving final registered mask...")
            final_mask = sitk.Resample(self.moving_mask, self.fixed_mask, self.final_transform, sitk.sitkNearestNeighbor, 0, self.moving_mask.GetPixelID())
            ImageIO.write_image(final_mask, os.path.join(self.output_dir, "registered_mask.tif"))
            moving_mask_base = os.path.splitext(os.path.basename(self.cfg.moving_mask_filename))[0]
            final_mask_binary = sitk.Cast(final_mask > 0, sitk.sitkUInt8)
            final_mask_binary = sitk.Cast(
                sitk.RescaleIntensity(final_mask_binary, 0, 255),
                sitk.sitkUInt8,
            )
            bin_name = f"mask_moving_{moving_mask_base}_coregistered_binarized.png"
            ImageIO.write_image(
                final_mask_binary,
                os.path.join(self.output_dir, bin_name),
                sitk.sitkUInt8,
            )
            self.message.emit(f"Saved binarized coregistered moving mask: {bin_name}")

            # Evaluate overlap (fixed mask vs registered moving mask) and record it.
            self._evaluate_and_record(final_mask)
            self._evaluate_correspondence(final_mask)
            self._evaluate_bidirectional(final_mask)

            # Folding QC on the transform that was actually applied.
            self._evaluate_jacobian()

            # Write the parameters file LAST, so the scores/IoU are included.
            self._save_parameters()

            # 4. Warp optional second mask (NearestNeighbor — preserves label IDs)
            self._warp_second_mask()

            # 5. Apply to Channels
            self._apply_to_channels()
            
            total_time = time.perf_counter() - t_start
            self.timings["total"] = total_time
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

    def _save_parameters(self):
        """Write Coregistration_parameters.txt at the END of the run.

        A [Summary] of the scores + folding first (the scores only exist after the
        evaluations run), then the full config dump built generically from the config
        dataclass so new settings are picked up automatically.
        """
        path = os.path.join(self.output_dir, "Coregistration_parameters.txt")
        corr = self.correspondence or {}
        bd = self.bidirectional or {}

        lines = [
            "Coregistration parameters",
            f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            "=" * 50,
            "",
            "[Summary]",
        ]

        def score(label, blk):
            if blk:
                lines.append(f"{label}: Dice {blk['mean_dice']:.4f}; IoU {blk['mean_iou']:.4f}")

        score("All-cell score with fused-cell groups", corr.get("complete"))
        if bd:
            lines.append(f"All-cell score: Dice {bd['dice']:.4f}; IoU {bd['iou']:.4f}")
        score("1:1 matches only", corr.get("strict"))
        score("1:1 and fused-cell groups", corr.get("corrected"))
        if self.global_dice is not None:
            lines.append(f"Silhouette-based score: Dice {self.global_dice:.4f}; "
                         f"IoU {self.global_iou:.4f}")
        if self.jacobian is not None:
            lines.append("Tissue folding" if self.jacobian.get("folding_in_mask")
                         else "No tissue folding")

        lines += ["", "=" * 50, ""]
        for key, val in dataclasses.asdict(self.cfg).items():
            if isinstance(val, dict):
                lines.append(f"{key}:")
                for sub_key, sub_val in val.items():
                    lines.append(f"    {sub_key} = {sub_val}")
            else:
                lines.append(f"{key} = {val}")

        # ---- [Details]: the numbers behind the summary scores + folding ----
        if corr or bd or self.jacobian is not None:
            lines += ["", "=" * 50, "", "[Details]"]

        if corr:
            s = corr.get("strict", {}); c = corr.get("corrected", {}); k = corr.get("complete", {})
            lines += [
                "",
                "# Cell correspondence (fixed=ICP vs registered=IF)",
                f"match_overlap_criterion = {corr.get('min_overlap_frac', 0):.2f}"
                "  (>= this fraction of EITHER cell's area)",
                f"n_icp_cells = {corr.get('n_icp_cells')}",
                f"n_if_cells = {corr.get('n_if_cells')}",
                f"n_one_to_one = {corr.get('n_one_to_one')}",
                f"n_icp_doublets = {corr.get('n_icp_doublets')}  (1 ICP <- N IF)",
                f"n_if_merges = {corr.get('n_if_merges')}  (1 IF -> N ICP)",
                f"n_complex = {corr.get('n_complex')}",
                f"n_cells_in_multiplets = {corr.get('n_cells_in_multiplets')}",
                f"n_icp_only = {corr.get('n_icp_only')}  (unmatched)",
                f"n_if_only = {corr.get('n_if_only')}  (unmatched)",
                f"score_1to1_only         = Dice {s.get('mean_dice',0):.6f}; IoU {s.get('mean_iou',0):.6f}  (n={s.get('n')})",
                f"score_1to1_and_fused    = Dice {c.get('mean_dice',0):.6f}; IoU {c.get('mean_iou',0):.6f}  (n={c.get('n')})",
                f"score_all_cell_fused    = Dice {k.get('mean_dice',0):.6f}; IoU {k.get('mean_iou',0):.6f}"
                f"  ({k.get('n_scored')} groups + {k.get('n_zero')} unmatched = {k.get('n')} units)",
            ]
        if bd:
            lines += [
                "",
                "# All-cell (bidirectional best-match) score",
                f"all_cell_dice = {bd.get('dice',0):.6f}",
                f"all_cell_iou = {bd.get('iou',0):.6f}",
                f"n_unmatched_fixed = {bd.get('n_unmatched_fixed')}",
                f"n_unmatched_registered = {bd.get('n_unmatched_registered')}",
            ]
        if self.jacobian is not None:
            j = self.jacobian
            lines += [
                "",
                "# Folding QC (Jacobian determinant of the applied deformation field;",
                "# det(J) = local area scale: 1 unchanged, <=0 = folding / non-invertible)",
                f"folding_in_tissue = {j.get('folding_in_mask')}",
                f"jacobian_min_in_tissue = {j.get('min_in_mask', float('nan')):.6f}",
                f"jacobian_max_in_tissue = {j.get('max_in_mask', float('nan')):.6f}",
                f"affine_determinant_baseline = {j.get('affine_det', float('nan')):.6f}",
                f"folded_pixels_in_tissue = {j.get('n_folded_in_mask')}",
                f"border_excluded_px = {j.get('border_excluded_px')}",
            ]

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            self.message.emit(f"Saved parameters: {os.path.basename(path)}")
        except OSError as exc:
            self.message.emit(f"Could not save parameters file: {exc}")

    def _evaluate_correspondence(self, registered_mask: sitk.Image):
        """Correspondence metric: 1:1-only and fused ("corrected") Dice/IoU.

        The existing per-cell metric forces ONE partner per reference cell, which
        punishes the common 1-ICP-to-2-IF doublet for being correct (a 93%-covered
        cell scores IoU 0.47 because the second IF cell is discarded) and charges a
        shared IF cell's full area to each claimant. It also never enumerates IF
        cells, so spurious IF cells are invisible.

        This method instead builds a BIPARTITE GRAPH: fixed(ICP) cell i is linked to
        registered(IF) cell j when their overlap is >= percell_min_overlap_frac of
        EITHER cell's own area (either, not just the IF side: a large IF cell lying
        entirely over a small ICP cell must still link). Connected components of that
        graph are the true correspondence groups, and each is scored ONCE as
        union(ICP cells) vs union(IF cells).

        Components classify every cell exactly once:
            1 ICP + 1 IF   -> one_to_one
            1 ICP + N IF   -> icp_doublet   (one ICP cell split by N IF cells)
            M ICP + 1 IF   -> if_merge      (one IF cell spanning M ICP cells)
            M ICP + N IF   -> complex       (tangle; rare)
            1 ICP + 0 IF   -> icp_only      (excluded from scores)
            0 ICP + 1 IF   -> if_only       (excluded from scores)

        Two score sets are reported: over 1:1 components only (clean registration
        accuracy) and over ALL matched components with groups fused (the corrected
        number). The gap between them is the cost of segmentation mismatch.
        """
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            self.message.emit("Correspondence metric skipped: SciPy sparse unavailable.")
            return

        frac = float(getattr(self.cfg, "percell_min_overlap_frac", 0.20))
        fix = sitk.GetArrayViewFromImage(self.fixed_mask).astype(np.int64)
        reg = sitk.GetArrayViewFromImage(registered_mask).astype(np.int64)
        if fix.max() == 0 or reg.max() == 0:
            self.message.emit("Correspondence metric skipped: a mask has no cells.")
            return

        fix_area = np.bincount(fix.ravel())
        reg_area = np.bincount(reg.ravel())
        fix_ids = np.unique(fix[fix > 0])
        reg_ids = np.unique(reg[reg > 0])
        n_f, n_r = fix_ids.size, reg_ids.size

        # Compact label -> node index maps.
        f_idx = np.zeros(int(fix_ids.max()) + 1, np.int64); f_idx[fix_ids] = np.arange(n_f)
        r_idx = np.zeros(int(reg_ids.max()) + 1, np.int64); r_idx[reg_ids] = np.arange(n_r)

        # Overlap table over pixels where BOTH masks have a cell.
        both = (fix > 0) & (reg > 0)
        stride = int(reg_ids.max()) + 1
        key = fix[both] * stride + reg[both]
        uk, cnt = np.unique(key, return_counts=True)
        p_f, p_r = uk // stride, uk % stride

        # Link when the overlap is significant for EITHER cell.
        linked = (cnt >= frac * fix_area[p_f]) | (cnt >= frac * reg_area[p_r])

        # Bipartite graph: nodes 0..n_f-1 = ICP, n_f..n_f+n_r-1 = IF.
        N = n_f + n_r
        rows = f_idx[p_f[linked]]
        cols = n_f + r_idx[p_r[linked]]
        g = coo_matrix((np.ones(rows.size, np.uint8), (rows, cols)), shape=(N, N))
        n_comp, comp = connected_components(g, directed=False)
        comp_f, comp_r = comp[:n_f], comp[n_f:]

        # Per-component cell counts and union areas (labels are disjoint, so the
        # union area is just the sum of the member areas).
        n_icp_per = np.bincount(comp_f, minlength=n_comp)
        n_if_per = np.bincount(comp_r, minlength=n_comp)
        A = np.bincount(comp_f, weights=fix_area[fix_ids].astype(float), minlength=n_comp)
        B = np.bincount(comp_r, weights=reg_area[reg_ids].astype(float), minlength=n_comp)

        # Intersection per component: only overlaps whose BOTH endpoints landed in
        # that component (a sub-threshold overlap bridging two components is leakage
        # and must not be credited to either).
        cf, cr = comp_f[f_idx[p_f]], comp_r[r_idx[p_r]]
        same = cf == cr
        inter = np.bincount(cf[same], weights=cnt[same].astype(float), minlength=n_comp)

        denom = A + B
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom - inter > 0, inter / (denom - inter), 0.0)
            dice = np.where(denom > 0, 2.0 * inter / denom, 0.0)

        one_to_one = (n_icp_per == 1) & (n_if_per == 1)
        icp_doublet = (n_icp_per == 1) & (n_if_per >= 2)
        if_merge = (n_icp_per >= 2) & (n_if_per == 1)
        complex_ = (n_icp_per >= 2) & (n_if_per >= 2)
        matched = (n_icp_per > 0) & (n_if_per > 0)
        icp_only = (n_icp_per == 1) & (n_if_per == 0)
        if_only = (n_icp_per == 0) & (n_if_per == 1)

        # Cells that were fused into multiplet (non-1:1) groups = total member cells
        # (ICP + IF) across every doublet / merge / complex group.
        multiplet = icp_doublet | if_merge | complex_
        n_cells_in_multiplets = int((n_icp_per[multiplet] + n_if_per[multiplet]).sum())

        def stats(mask):
            if not mask.any():
                return dict(n=0, mean_dice=0.0, mean_iou=0.0)
            return dict(
                n=int(mask.sum()),
                mean_dice=float(dice[mask].mean()),
                mean_iou=float(iou[mask].mean()),
            )

        summary = {
            "min_overlap_frac": frac,
            "n_icp_cells": int(n_f),
            "n_if_cells": int(n_r),
            "n_icp_only": int(icp_only.sum()),
            "n_if_only": int(if_only.sum()),
            "n_one_to_one": int(one_to_one.sum()),
            "n_icp_doublets": int(icp_doublet.sum()),
            "n_if_merges": int(if_merge.sum()),
            "n_complex": int(complex_.sum()),
            "n_cells_in_multiplets": n_cells_in_multiplets,
            "strict": stats(one_to_one),
            "corrected": stats(matched),
        }

        # COMPLETE score: nothing is excluded. Every scoring unit is either a matched
        # correspondence GROUP (1:1 pairs and fused multiplets alike, each scored once
        # on its union so a split cell is not punished for being split) or an UNMATCHED
        # cell, which contributes 0. Denominator = matched groups + unmatched cells.
        # This is the strictest of the three: 'strict' keeps only clean pairs,
        # 'corrected' fuses multiplets but still ignores unmatched cells, 'complete'
        # additionally makes every missing/spurious cell cost something.
        n_zero = int(icp_only.sum()) + int(if_only.sum())
        n_units = int(matched.sum()) + n_zero
        summary["complete"] = dict(
            n=n_units,
            n_scored=int(matched.sum()),
            n_zero=n_zero,
            mean_dice=float(dice[matched].sum() / n_units) if n_units else 0.0,
            mean_iou=float(iou[matched].sum() / n_units) if n_units else 0.0,
        )
        self.correspondence = summary

        s, c, k = summary["strict"], summary["corrected"], summary["complete"]
        self.message.emit(
            f"Correspondence — 1:1 only: Dice {s['mean_dice']:.4f} / IoU {s['mean_iou']:.4f} "
            f"({s['n']} cells)  |  corrected (fused): Dice {c['mean_dice']:.4f} / "
            f"IoU {c['mean_iou']:.4f} ({c['n']} groups)  |  complete (fused + unmatched): "
            f"Dice {k['mean_dice']:.4f} / IoU {k['mean_iou']:.4f} "
            f"({k['n_scored']} groups + {k['n_zero']} unmatched = {k['n']} units)  |  "
            f"ICP-only {summary['n_icp_only']}, IF-only {summary['n_if_only']}, "
            f"ICP doublets {summary['n_icp_doublets']}, IF merges {summary['n_if_merges']}"
        )


    def _evaluate_bidirectional(self, registered_mask: sitk.Image):
        """Bidirectional best-match Dice/IoU (symmetric; see logic.bidirectional_best_match).

        Complements the one-directional per-cell metric, which enumerates only the fixed
        (ICP) mask and is therefore blind to EXTRA objects in the registered mask.
        """
        try:
            fix = sitk.GetArrayViewFromImage(self.fixed_mask)
            reg = sitk.GetArrayViewFromImage(registered_mask)
            frac = float(getattr(self.cfg, "percell_min_overlap_frac", 0.5))
            r = CoregistrationLogic.bidirectional_best_match(fix, reg, min_overlap_frac=frac)
            self.bidirectional = r
            self.message.emit(
                f"Bidirectional best-match — Dice: {r['dice']:.6f}  |  IoU: {r['iou']:.6f}   "
                f"({r['n_fixed']} fixed + {r['n_registered']} registered objects scored; "
                f"unmatched: {r['n_unmatched_fixed']} fixed, "
                f"{r['n_unmatched_registered']} registered)"
            )
        except Exception as exc:
            self.message.emit(f"Bidirectional metric failed (non-fatal): {exc}")

    def _evaluate_jacobian(self):
        """Folding QC: Jacobian determinant of the applied deformation field.

        det(J) is the LOCAL AREA SCALE at each pixel: 1 = unchanged, >1 expansion,
        <1 compression, 0 = collapsed, <0 = the map flipped orientation, i.e. the
        tissue folded over itself (non-injective). So "min(det J) > 0 everywhere"
        is exactly the statement "the deformation is locally invertible / no folding".

        Computed on ``self.final_transform`` — the composite that is actually applied
        to the data. Because an affine is linear its det(J) is CONSTANT, so the field
        is centred on the affine determinant rather than on 1.0; the affine value is
        reported alongside as the baseline. Only the B-Spline can fold.

        Both a global figure and one restricted to the fixed mask are reported:
        folding inside tissue matters, folding in empty background does not.

        BORDER: the Jacobian is computed by finite differences, so the outermost
        pixels have no neighbour on one side and fall back to a boundary condition.
        That fabricates spurious negative determinants in a 1-2 px rim (verified on
        real data: rows 0-1 read det(J) ~= -0.54 while row 2 onward reads ~= 1.04, a
        discontinuous step). Statistics therefore EXCLUDE a border margin; the saved
        map is still the full, uncropped field.
        """
        BORDER = 2  # px excluded from the statistics (finite-difference rim)
        try:
            t0 = time.perf_counter()
            self.message.emit("Computing Jacobian determinant (folding check)...")
            field = sitk.TransformToDisplacementField(
                self.final_transform,
                sitk.sitkVectorFloat32,          # half the memory of Float64
                self.fixed_mask.GetSize(),
                self.fixed_mask.GetOrigin(),
                self.fixed_mask.GetSpacing(),
                self.fixed_mask.GetDirection(),
            )
            jac_img = sitk.DisplacementFieldJacobianDeterminant(field)
            jac_full = sitk.GetArrayFromImage(jac_img)

            # Drop the finite-difference rim before measuring anything.
            b = BORDER
            jac = jac_full[b:-b, b:-b] if min(jac_full.shape) > 2 * b else jac_full
            fg_full = sitk.GetArrayViewFromImage(self.fixed_mask) > 0
            fg = fg_full[b:-b, b:-b] if min(fg_full.shape) > 2 * b else fg_full
            in_cells = jac[fg] if fg.any() else jac

            # Save the map so folding can be LOCATED, not just detected. Open in Fiji:
            # the float map shows where det(J) dips, the mask shows folded pixels only.
            try:
                folded = sitk.Cast(jac_img <= 0, sitk.sitkUInt8) * 255
                folded.CopyInformation(self.fixed_mask)
                ImageIO.write_image(
                    folded, os.path.join(self.output_dir, "Coregistration_folded_mask.png"),
                    sitk.sitkUInt8,
                )
                self.message.emit(
                    "Saved Coregistration_folded_mask.png (white = folded pixels)"
                )
            except Exception as e:
                self.message.emit(f"Could not save folded-pixel map: {e}")

            summary = {
                "min": float(jac.min()),
                "max": float(jac.max()),
                "min_in_mask": float(in_cells.min()),
                "max_in_mask": float(in_cells.max()),
                "n_folded": int((jac <= 0).sum()),
                "n_folded_in_mask": int((in_cells <= 0).sum()),
                "frac_folded": float((jac <= 0).mean()),
                "folding": bool((jac <= 0).any()),
                "folding_in_mask": bool((in_cells <= 0).any()),
                "border_excluded_px": b,
            }
            # Affine baseline: det of the linear part, the value det(J) centres on.
            try:
                aff = CoregistrationLogic.as_affine(self.final_transform.GetNthTransform(0))
                m = np.array(aff.GetMatrix(), dtype=float).reshape(2, 2)
                summary["affine_det"] = float(np.linalg.det(m))
            except Exception:
                summary["affine_det"] = float("nan")

            self.jacobian = summary
            self.message.emit(
                f"Jacobian det — min {summary['min']:.4f}, max {summary['max']:.4f} "
                f"(in mask: min {summary['min_in_mask']:.4f}); "
                + ("NO FOLDING" if not summary["folding"]
                   else f"FOLDING in {summary['n_folded']} px ({summary['frac_folded']*100:.3f}%)")
                + f"  [{time.perf_counter() - t0:.2f}s]"
            )


        except Exception as e:
            self.message.emit(f"Jacobian/folding check failed (non-fatal): {e}")

    def _evaluate_and_record(self, registered_mask: sitk.Image):
        """Compute Dice / IoU (fixed mask vs registered moving mask, both binarized
        at >0), log them, and append them to the parameters text file."""
        fix = sitk.GetArrayViewFromImage(self.fixed_mask) > 0
        reg = sitk.GetArrayViewFromImage(registered_mask) > 0
        inter = float(np.logical_and(fix, reg).sum())
        union = float(np.logical_or(fix, reg).sum())
        denom = float(fix.sum()) + float(reg.sum())
        dice = (2.0 * inter / denom) if denom > 0 else 1.0
        iou = (inter / union) if union > 0 else 1.0

        self.global_dice, self.global_iou = dice, iou
        self.message.emit(f"Evaluation — Dice: {dice:.6f}  |  IoU: {iou:.6f}")

    def _step_load_masks(self):
        # Paths
        p_fix = os.path.join(self.cfg.directory, self.cfg.fixed_mask_filename)
        p_mov = os.path.join(self.cfg.directory, self.cfg.moving_mask_filename)
        
        if not os.path.exists(p_fix) or not os.path.exists(p_mov):
            raise FileNotFoundError(f"Masks not found: {p_fix} or {p_mov}")

        def _log_source_info(tag: str, path: str):
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(path)
                r.ReadImageInformation()
                size = r.GetSize()
                ncomp = r.GetNumberOfComponents() if hasattr(r, "GetNumberOfComponents") else "unknown"
                if hasattr(r, "GetPixelIDTypeAsString"):
                    ptype = r.GetPixelIDTypeAsString()
                elif hasattr(r, "GetPixelID"):
                    ptype = str(r.GetPixelID())
                else:
                    ptype = "unknown"
                self.message.emit(
                    f"{tag} source: dim={len(size)}, size={size}, "
                    f"components={ncomp}, type={ptype}"
                )
            except Exception as e:
                self.message.emit(f"{tag} source info unavailable: {e}")

        _log_source_info("Fixed mask", p_fix)
        _log_source_info("Moving mask", p_mov)
            
        self.fixed_mask = ImageIO.read_image_safe(p_fix, self.cfg.fixed_pixel_size)
        self.moving_mask = ImageIO.read_image_safe(p_mov, self.cfg.moving_pixel_size)
        
        # Keep label IDs stable for large TIFF masks:
        # - remove non-positive labels (e.g., -1 background) to 0
        # - use UInt32 to avoid UInt16 truncation for >65535 labels
        self.fixed_mask = sitk.Cast(sitk.Mask(self.fixed_mask, self.fixed_mask > 0), sitk.sitkUInt32)
        self.moving_mask = sitk.Cast(sitk.Mask(self.moving_mask, self.moving_mask > 0), sitk.sitkUInt32)

        # Log mask stats to diagnose TIFF/PNG interpretation differences.
        fix_view = sitk.GetArrayViewFromImage(self.fixed_mask)
        mov_view = sitk.GetArrayViewFromImage(self.moving_mask)
        self.message.emit(
            "Fixed mask stats: "
            f"size={self.fixed_mask.GetSize()}, "
            f"type={self.fixed_mask.GetPixelIDTypeAsString()}, "
            f"min={int(fix_view.min())}, max={int(fix_view.max())}, "
            f"foreground={int(np.count_nonzero(fix_view))}"
        )
        self.message.emit(
            "Moving mask stats: "
            f"size={self.moving_mask.GetSize()}, "
            f"type={self.moving_mask.GetPixelIDTypeAsString()}, "
            f"min={int(mov_view.min())}, max={int(mov_view.max())}, "
            f"foreground={int(np.count_nonzero(mov_view))}"
        )

        # Some TIFFs load as all-zero with certain ITK readers although data exists.
        # If moving mask is empty, try PIL fallback once and keep it only if non-empty.
        mov_ext = os.path.splitext(p_mov)[1].lower()
        if np.count_nonzero(mov_view) == 0 and mov_ext in [".tif", ".tiff"]:
            self.message.emit("Moving mask appears empty via SimpleITK; trying PIL TIFF fallback...")
            try:
                alt_mov = ImageIO._read_with_pil_fallback(p_mov)
                alt_mov = ImageIO.to_2d(alt_mov)
                alt_mov.SetSpacing([float(self.cfg.moving_pixel_size)] * 2)
                alt_mov.SetOrigin([0.0] * 2)
                alt_mov.SetDirection([1.0, 0.0, 0.0, 1.0])
                alt_mov = sitk.Cast(sitk.Mask(alt_mov, alt_mov > 0), sitk.sitkUInt32)
                alt_view = sitk.GetArrayViewFromImage(alt_mov)
                self.message.emit(
                    "Moving mask fallback stats: "
                    f"size={alt_mov.GetSize()}, "
                    f"type={alt_mov.GetPixelIDTypeAsString()}, "
                    f"min={int(alt_view.min())}, max={int(alt_view.max())}, "
                    f"foreground={int(np.count_nonzero(alt_view))}"
                )
                if np.count_nonzero(alt_view) > 0:
                    self.moving_mask = alt_mov
                    mov_view = alt_view
                    self.message.emit("Using PIL TIFF fallback for moving mask.")
            except Exception as e:
                self.message.emit(f"PIL TIFF fallback failed: {e}")

        if np.count_nonzero(fix_view) == 0:
            raise ValueError("Fixed mask is empty after loading (no values > 0).")
        if np.count_nonzero(mov_view) == 0:
            raise ValueError(
                "Moving mask is empty after loading (no values > 0). "
                "Check TIFF channel/slice layout and pixel type."
            )

        # Detect if we are dealing with a Label Map (many objects) or Binary Mask
        # If max > 1, assume Label Map
        # Use numpy for safe/fast min/max check avoiding SITK version differences
        max_val = mov_view.max()
        
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
        """Initial transform from FFT (translation, and optionally rotation)."""
        # With rotation enabled the transform must rotate about the MOVING image's
        # centre — that's the centre the rotation search rotates around. For
        # translation-only we keep the legacy fixed-centre behaviour (arbitrary for
        # pure translation, but preserves reproducibility with old runs).
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
                self.message.emit(
                    f"Scanning Rotation: ±{self.cfg.rotation_range}°, "
                    f"step {self.cfg.rotation_step}°"
                )
                ang, dx, dy, score = FeatureLogic.solve_rotation_fft(
                    fixed_for_fft, moving_for_fft,
                    center=tuple(center_pt),
                    spacing=self.cfg.fft_spacing,
                    rot_range=self.cfg.rotation_range,
                    rot_step=self.cfg.rotation_step,
                    negate=self.cfg.fft_negate,
                )
                # Rotate the (dx, dy) translation by the chosen angle so the shift
                # expressed in the FIXED frame becomes the shift in the ROTATED frame
                # the transform applies.
                rad = np.radians(ang)
                rot_dx = dx * math.cos(rad) - dy * math.sin(rad)
                rot_dy = dx * math.sin(rad) + dy * math.cos(rad)
                shift = (rot_dx, rot_dy)
                best_angle_deg = ang
                self.message.emit(
                    f"Best Rotation: {ang}° (score={score:.2e}), raw shift=({dx:.1f}, {dy:.1f})"
                )
            else:
                sx, sy, _ = FeatureLogic.solve_translation_fft(
                    fixed_for_fft, moving_for_fft,
                    coarse_spacing=self.cfg.fft_spacing,
                    negate_output=self.cfg.fft_negate,
                )
                shift = (sx, sy)
                best_angle_deg = 0.0

            self.message.emit(
                f"FFT estimate: rotation={best_angle_deg}°, dx={shift[0]:.2f}, dy={shift[1]:.2f} (applied to moving)"
            )
        else:
            shift = (0.0, 0.0)
            best_angle_deg = 0.0
            self.message.emit("FFT prealignment disabled; no initial shift applied.")

        tx.SetAngle(np.radians(best_angle_deg))
        tx.SetTranslation(tuple(shift))
        self.message.emit(f"Initial transform: rotation={best_angle_deg}°, translation={shift}")
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

            # Resample original moving mask (not the gradient)
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

    @staticmethod
    def _rsgd_reason(desc, iters_run, max_iters, custom=None):
        """Human-readable RegularStepGradientDescent stop reason."""
        if custom:
            return custom
        s = (desc or "").strip().lower()
        if "step too small" in s or "less than minimum step" in s:
            return "optimizer step fell below the minimum (converged)"
        if "gradient magnitude tolerance" in s:
            return "gradient magnitude below tolerance (converged)"
        if "maximum number of iterations" in s or iters_run >= max_iters:
            return "reached the maximum iterations"
        return (desc or "converged").split(":")[-1].strip() or "converged"

    def _make_rsgd_observers(self, label, cfg, tx, plateau_window=0, plateau_tol=0.0):
        """Build (iter, level, final) observers for an RSGD run_registration call.

        Emits one report per pyramid level ("<label>: pyramid level L/N used X of MAX
        iterations; stopped: <reason>; metric=..") and, when plateau_window > 0,
        applies an MSQ-plateau early stop at the FINAL level ONLY: it records the MSQ
        after each final-level iteration and stops when the best MSQ of the most recent
        window did not beat the previous window's best by at least plateau_tol
        (relative). Best-of-window guards against a single noisy iteration; recording
        is confined to the final level so metrics from coarser levels (a different
        scale) are never compared. The stop must be final-level-only because
        StopRegistration() halts the entire multi-resolution run.
        """
        n_levels = len(cfg.shrink_factors)
        st = {"level": 0, "imax": {}, "metrics": [], "custom": {}}

        def _report(r, level):
            iters = min(st["imax"].get(level, 0) + 1, cfg.iters)
            reason = self._rsgd_reason(
                r.GetOptimizerStopConditionDescription(), iters, cfg.iters, st["custom"].get(level))
            self.message.emit(
                f"{label}: pyramid level {level}/{n_levels} used {iters} of {cfg.iters} "
                f"max iterations; stopped: {reason}; metric={float(r.GetMetricValue()):.6g}"
            )

        def iter_obs(r):
            if self._abort:
                r.StopRegistration(); return
            i = r.GetOptimizerIteration()
            lvl = st["level"]
            if lvl >= 1:
                st["imax"][lvl] = max(st["imax"].get(lvl, 0), i)
            if i == 1 or i % 50 == 0:
                self.progress.emit(label, i, cfg.iters)
                self._emit_mask_overlay(tx)
            if plateau_window > 0 and lvl == n_levels:
                st["metrics"].append(float(r.GetMetricValue()))
                m = st["metrics"]
                if len(m) >= 2 * plateau_window:
                    prev_best = min(m[-2 * plateau_window:-plateau_window])
                    recent_best = min(m[-plateau_window:])
                    rel = max(0.0, (prev_best - recent_best) / max(abs(prev_best), 1e-12))
                    if rel < plateau_tol:
                        st["custom"][lvl] = (
                            f"MSQ plateau (best improved {rel:.3g} < {plateau_tol:g} over "
                            f"two {plateau_window}-iteration windows)")
                        r.StopRegistration()

        def level_obs(r):
            if self._abort:
                r.StopRegistration(); return
            if st["level"] >= 1:
                _report(r, st["level"])
            st["level"] += 1
            self.progress.emit(label, st["level"], n_levels)
            self._emit_mask_overlay(tx)

        def final_obs(r):
            if st["level"] >= 1:
                _report(r, st["level"])

        return iter_obs, level_obs, final_obs

    def _run_rigid_affine_stage(self, fixed, moving, initial_tx: sitk.Transform,
                       fix_mask_override: Optional[sitk.Image] = None) -> sitk.Transform:
        """Rigid/affine stage: Coarse Euler (rigid) -> Fine Affine.

        Metric fixed mask:
        - If ``fix_mask_override`` is given (cell foreground for the per-cell gradient
          signal), use it directly.
        - Otherwise fall back to the edge-band threshold, which is only meaningful
          when ``fixed`` is a signed distance map.
        """
        if fix_mask_override is not None:
            fix_mask_obj = fix_mask_override
        elif self.cfg.edge_band_um > 0:
            fix_mask_obj = sitk.BinaryThreshold(sitk.Abs(fixed), 0, self.cfg.edge_band_um, 1, 0)
        else:
            fix_mask_obj = None

        # 1. Coarse (Euler / rigid) — native RSGD stopping only, per-level reporting.
        c_cfg = self.cfg.coarse_config
        init = sitk.Euler2DTransform(initial_tx)
        c_iter, c_level, c_final = self._make_rsgd_observers("Rigid (Euler)", c_cfg, init)
        CoregistrationLogic.run_registration(
            fixed, moving, init, c_cfg, metric='msq',
            fixed_mask=fix_mask_obj, observer=c_iter,
            level_observer=c_level, final_observer=c_final,
            seed=self.cfg.metric_sampling_seed,
        )
        self._check_abort()

        # 2. Fine (Affine) — native RSGD stopping + an MSQ-plateau early stop applied
        # ONLY at the final full-resolution level (StopRegistration halts the whole
        # pyramid, so it must never fire at a coarse level).
        r_cfg = self.cfg.affine_config
        init_fine = CoregistrationLogic.as_affine(init)   # initialize from the coarse result
        pw = int(getattr(self.cfg, "affine_early_stop_window", 10))
        pt = float(getattr(self.cfg, "affine_early_stop_relative_tolerance", 0.001))
        a_iter, a_level, a_final = self._make_rsgd_observers(
            "Affine (Fine)", r_cfg, init_fine, plateau_window=pw, plateau_tol=pt)
        CoregistrationLogic.run_registration(
            fixed, moving, init_fine, r_cfg, metric='msq',
            fixed_mask=fix_mask_obj, observer=a_iter,
            level_observer=a_level, final_observer=a_final,
            seed=self.cfg.metric_sampling_seed,
        )
        self._check_abort()
        return init_fine

    def _run_bspline_stage(self, fixed, moving, fixed_mask, initial_tx) -> sitk.Transform:
        b_cfg = self.cfg.bspline_config
        # B-Spline early stopping (LBFGS2 delta-convergence); see config.
        es_window = int(getattr(self.cfg, "bspline_early_stop_window", 3))
        es_tol = float(getattr(self.cfg, "bspline_early_stop_tolerance", 0.01))
        metric_name = CoregistrationLogic.normalize_metric_name(self.cfg.bspline_metric)
        optimizer_name = CoregistrationLogic.normalize_optimizer_name(self.cfg.bspline_optimizer)
        scale_factors = CoregistrationLogic.resolve_bspline_scale_factors(
            level_count=len(b_cfg.shrink_factors),
            configured_factors=self.cfg.bspline_mesh_scale_factors,
        )
        
        # Initial BSpline
        bspline_init = sitk.BSplineTransformInitializer(fixed, list(b_cfg.bspline_mesh))
        fixed_mask_bin = sitk.Cast(fixed_mask > 0, sitk.sitkUInt8)

        self.message.emit(
            "B-Spline strategy: "
            f"metric={metric_name}, optimizer={optimizer_name}, "
            f"deterministic_sampling={self.cfg.bspline_deterministic_sampling}, "
            f"mesh_scale_factors={scale_factors}"
        )
        
        reg = sitk.ImageRegistrationMethod()
        if metric_name == "mattes":
            reg.SetMetricAsMattesMutualInformation(b_cfg.metric_bins)
        else:
            reg.SetMetricAsMeanSquares()
        reg.SetMetricFixedMask(fixed_mask_bin)

        if self.cfg.bspline_deterministic_sampling:
            reg.SetMetricSamplingStrategy(reg.NONE)
        else:
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(b_cfg.sampling_percentage, self.cfg.metric_sampling_seed)
        
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetShrinkFactorsPerLevel(b_cfg.shrink_factors)
        reg.SetSmoothingSigmasPerLevel(b_cfg.smoothing_sigmas)
        reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
        
        reg.SetInitialTransformAsBSpline(
            bspline_init,
            inPlace=True,
            scaleFactors=scale_factors,
        )
        reg.SetMovingInitialTransform(initial_tx)
        
        if optimizer_name == "lbfgs2":
            # 'iters' is the hard maximum. LBFGS2 delta-convergence stops a pyramid
            # level early once the metric stops improving meaningfully:
            #   deltaConvergenceDistance  = early-stop WINDOW (iterations compared
            #                               over; 0 disables this test)
            #   deltaConvergenceTolerance = early-stop TOLERANCE (min relative
            #                               improvement, e.g. 0.01 = 1%)
            reg.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-5,
                numberOfIterations=b_cfg.iters,
                deltaConvergenceDistance=es_window,
                deltaConvergenceTolerance=es_tol,
            )
            # NOTE: SetOptimizerScalesFromPhysicalShift is not compatible with
            # LBFGS2 and can cause the optimizer to hang — omit it here.
        else:
            reg.SetOptimizerAsRegularStepGradientDescent(
                learningRate=b_cfg.learning_rate,
                minStep=b_cfg.min_step,
                numberOfIterations=b_cfg.iters,
                relaxationFactor=b_cfg.relaxation_factor,
                gradientMagnitudeTolerance=b_cfg.gradient_tolerance,
            )
            # Scale parameters by physical shift (RSGD only — not valid for LBFGS2)
            reg.SetOptimizerScalesFromPhysicalShift()

        # ---- Callbacks + per-level reporting ---------------------------------
        # sitkIterationEvent fires per iteration (both RSGD and LBFGS2), so we track
        # the max iteration reached at each pyramid level. sitkMultiResolutionIteration
        # Event fires at the START of each new level, at which point the optimizer's
        # stop-condition still reflects the level that just finished — so we report a
        # finished level from the NEXT level's event (and the last level after Execute).
        n_levels = len(b_cfg.shrink_factors)
        _level = [0]
        _iter_max = {}   # level -> max 0-based iteration observed

        def _reason(stop_desc, iters_run):
            s = (stop_desc or "").strip().lower()
            if iters_run >= b_cfg.iters:
                return "reached the maximum"
            if "already minimized" in s:
                return "already minimized (no improvement possible)"
            if "rounding error" in s or "line search" in s or "line-search" in s:
                return "converged (line search could not improve further)"
            if es_window > 0 and (not s or "unknown" in s):
                return (f"early stop: metric improved < {es_tol:g} over the last "
                        f"{es_window} iteration(s)")
            return stop_desc or "converged"

        def _report_level(level):
            iters_run = min(_iter_max.get(level, 0) + 1, b_cfg.iters)
            self.message.emit(
                f"B-Spline level {level}/{n_levels}: ran {iters_run} of {b_cfg.iters} "
                f"iterations — {_reason(reg.GetOptimizerStopConditionDescription(), iters_run)}"
            )

        def _obs_iter():
            if self._abort:
                reg.StopRegistration()
                return
            i = reg.GetOptimizerIteration()
            if _level[0] >= 1:
                _iter_max[_level[0]] = max(_iter_max.get(_level[0], 0), i)
            if i == 1 or i % 50 == 0:
                self.progress.emit("B-Spline", i, b_cfg.iters)
                comp = sitk.CompositeTransform(2)
                comp.AddTransform(initial_tx)
                comp.AddTransform(bspline_init)
                self._emit_mask_overlay(comp)

        def _obs_level():
            if self._abort:
                reg.StopRegistration()
                return
            if _level[0] >= 1:          # a level just finished — report it
                _report_level(_level[0])
            _level[0] += 1
            self.message.emit(f"B-Spline: starting pyramid level {_level[0]} / {n_levels}")
            self.progress.emit("B-Spline", _level[0], n_levels)
            comp = sitk.CompositeTransform(2)
            comp.AddTransform(initial_tx)
            comp.AddTransform(bspline_init)
            self._emit_mask_overlay(comp)

        reg.AddCommand(sitk.sitkIterationEvent, _obs_iter)
        reg.AddCommand(sitk.sitkMultiResolutionIterationEvent, _obs_level)

        reg.Execute(fixed, moving)

        # Report the final level (no level-change event fires after the last one).
        if _level[0] >= 1:
            _report_level(_level[0])
        return bspline_init

    def _warp_second_mask(self):
        """Warp the optional second label mask with NearestNeighbor interpolation.

        Linear interpolation (used for channels) blends adjacent integer label IDs
        into non-integer values, destroying cell identity. NearestNeighbor assigns
        each output pixel the label of the nearest input pixel, preserving all IDs.
        """
        filename = self.cfg.second_mask_filename
        if not filename:
            return
        p = os.path.join(self.cfg.directory, filename)
        if not os.path.exists(p):
            self.message.emit(f"Second mask not found, skipping: {filename}")
            return
        self.message.emit(f"Warping second mask (nearest-neighbour): {filename}")
        try:
            mask2 = ImageIO.read_image_safe(p, self.cfg.moving_pixel_size)
            mask2 = sitk.Cast(sitk.Mask(mask2, mask2 > 0), sitk.sitkUInt32)
            warped = sitk.Resample(
                mask2, self.fixed_mask, self.final_transform,
                sitk.sitkNearestNeighbor, 0, mask2.GetPixelID(),
            )
            base = os.path.splitext(os.path.basename(filename))[0]
            out_path = os.path.join(self.output_dir, f"{base}_coregistered.tif")
            ImageIO.write_image(warped, out_path)
            self.message.emit(f"Saved warped second mask: {os.path.basename(out_path)}")
        except Exception as e:
            self.message.emit(f"Error warping second mask: {e}\n{traceback.format_exc()}")

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
                         # We reset Origin/Direction to match the "ImageIO.read_with_geometry" assumptions
                         # if we want to stay strictly compatible with previous "read_with_geometry" logic.
                         # The new load_as_planes didn't include the spacing override logic.
                         # Let's add it here or update load_as_planes? 
                         # Actually load_as_planes raw read -> spacing might be wrong.
                         # We should apply overrides.
                         p.SetOrigin([0.0]*2)
                         p.SetDirection([1.0, 0.0, 0.0, 1.0])

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
