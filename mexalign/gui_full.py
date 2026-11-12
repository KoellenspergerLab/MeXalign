"""
MeXalign Modular GUI
=================
Graphical interface for the 2D coregistration pipeline.

This is the merged interface:
- Tabbed settings (Inputs / FFT / Rigid-Affine / B-Spline) so every parameter has its own
  labelled field — no free-text "parameter overrides".
- Mask selection via dropdowns, pixel sizes as inputs, channel batch selection.
- Quick-settings presets (MSQ fast / MSQ high-detail / Mattes) + a Gradient dropdown.
- Live overlay preview on the right, plus Dice / IoU scores and a pipeline log.

HOW TO USE (for non-programmers)
----------------------------------
1. Pick a Quick setting (Mattes is fine for most cases).
2. On the Inputs tab, Browse to the folder with your images, choose the fixed and
   moving masks from the dropdowns, and set the pixel sizes.
3. Tick the channel images you want warped along with the masks.
4. Click "Run Alignment" and watch the overlay update. Dice / IoU appear when done.
"""

import json
import math
import os
import traceback
from pathlib import Path

import SimpleITK as sitk
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .compat_runner import (
    build_mexalign_worker,
    build_output_paths,
    evaluate_coregistration_output,
)
from .contracts import MeXalignRunInput, MeXalignRunResult


def _monospace_font() -> QtGui.QFont:
    """A genuinely fixed-pitch font, so the results text lines up.

    Qt's QFontDatabase.systemFont(FixedFont) can resolve to the generic family
    "monospace" with fixedPitch=False, which renders proportionally. Naming real
    families (plus the Monospace style hint as a fallback) avoids that on every OS.
    """
    f = QtGui.QFont()
    f.setFamilies(["Menlo", "Monaco", "SF Mono", "Consolas",
                   "DejaVu Sans Mono", "Courier New"])
    f.setStyleHint(QtGui.QFont.Monospace)
    f.setFixedPitch(True)
    return f


# ---------------------------------------------------------------------------
# Quick-settings presets
# ---------------------------------------------------------------------------
# A preset bundles a few B-Spline choices. Everything else comes from the tabs, so
# the user can still fine-tune after picking a preset. Gradient orientation is NOT
# a preset — it's the "Gradient generation" dropdown on the Inputs tab.
#   variant          : pipeline backend (always "mexalign")
#   bspline_metric   : similarity metric for the B-Spline stage
#   bs_spacing       : control-point spacing (µm)
#   bs_iter          : MAXIMUM B-Spline iterations (hard cap)
#   bs_shrink        : shrink factors
#   bs_scale         : mesh scale factors
#   bs_early_window  : early-stop window (0 = early stopping off)
#   bs_early_tol     : early-stop tolerance (min relative improvement)
PRESETS = {
    "MSQ fast": {
        "variant": "mexalign", "bspline_metric": "msq",
        "bs_spacing": 150.0, "bs_iter": 20, "bs_shrink": "4,1", "bs_scale": "1,2",
        "bs_early_window": 3, "bs_early_tol": 0.01,
        "tip": "MSQ metric, 150 µm spacing, shrink 4,1, mesh scale 1,2, max 20 iters "
               "with early stopping (window 3, tol 0.01). Quickest good result.",
    },
    "MSQ high-detail": {
        "variant": "mexalign", "bspline_metric": "msq",
        "bs_spacing": 150.0, "bs_iter": 20, "bs_shrink": "4,1", "bs_scale": "1,3",
        "bs_early_window": 3, "bs_early_tol": 0.01,
        "tip": "Like MSQ fast, but a finer mesh at full resolution (mesh scale 1,3) "
               "for more local detail. Same early stopping.",
    },
    "Mattes": {
        "variant": "mexalign", "bspline_metric": "mattes",
        "bs_spacing": 200.0, "bs_iter": 40, "bs_shrink": "4,2,1", "bs_scale": "1,2,4",
        "bs_early_window": 0, "bs_early_tol": 0.01,
        "tip": "FFT(+rotation) → gradient rigid/affine → B-Spline with Mattes Mutual "
               "Information (40 iters, no early stop). The original default.",
    },
}


# ---------------------------------------------------------------------------
# Background worker – runs the pipeline on a Qt thread
# ---------------------------------------------------------------------------

class MeXalignRunWorker(QtCore.QObject):
    """
    Runs the coregistration pipeline in the background so the GUI stays
    responsive.  Emits signals that the MainWindow connects to.
    """

    log = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    overlay_update = QtCore.Signal(object)
    completed = QtCore.Signal(object)   # MeXalignRunResult
    failed = QtCore.Signal(str)         # traceback string
    aborted = QtCore.Signal()           # user pressed Stop — not an error
    percell = QtCore.Signal(object)     # per-cell metric summary (dict)
    jacobian = QtCore.Signal(object)    # folding QC summary (dict)
    correspondence = QtCore.Signal(object)  # 1:1 / fused Dice+IoU, class counts
    bidirectional = QtCore.Signal(object)   # bidirectional best-match Dice/IoU
    timings = QtCore.Signal(object)     # per-stage seconds (dict)
    done = QtCore.Signal()

    def __init__(self, run_input: MeXalignRunInput):
        super().__init__()
        self._run_input = run_input
        self._pipeline_worker = None
        self._abort_requested = False

    @QtCore.Slot()
    def run(self):
        """Run the coregistration once and emit the scores."""
        try:
            cfg, worker = build_mexalign_worker(self._run_input)
            self._pipeline_worker = worker
            self._connect_worker_signals(worker)

            worker.run()

            # User pressed Stop: the pipeline unwound cleanly without producing
            # outputs. That is not a failure — report it as an abort, no popup.
            if getattr(worker, "_abort", False):
                self.aborted.emit()
                return

            output_dir, registered_mask = build_output_paths(cfg)
            if not registered_mask.exists():
                raise RuntimeError(f"Missing expected output: {registered_mask}")

            evaluation = evaluate_coregistration_output(cfg, self._run_input, registered_mask)
            # Emit BEFORE completed so the handler can fold it into the results panel.
            self.jacobian.emit(getattr(worker, "jacobian", None))
            self.correspondence.emit(getattr(worker, "correspondence", None))
            self.bidirectional.emit(getattr(worker, "bidirectional", None))
            self.timings.emit(getattr(worker, "timings", None))
            self.completed.emit(MeXalignRunResult(
                output_dir=output_dir,
                registered_mask_path=registered_mask,
                evaluation=evaluation,
            ))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.done.emit()

    @QtCore.Slot()
    def abort(self):
        """Ask the running pipeline to stop at the next safe checkpoint."""
        self._abort_requested = True
        worker = self._pipeline_worker
        if worker is not None and hasattr(worker, "abort"):
            worker.abort()

    def _connect_worker_signals(self, worker):
        self._connect_signal(worker, "message", lambda msg: self.log.emit(str(msg)))
        self._connect_signal(worker, "error",   lambda msg: self.log.emit(f"ERROR: {msg}"))
        self._connect_signal(
            worker, "progress",
            lambda stage, i, total: self.progress.emit(f"{stage}: {i}/{total}"),
        )
        self._connect_signal(worker, "overlay_update", self.overlay_update.emit)
        self._connect_signal(worker, "percell_result", self.percell.emit)

    @staticmethod
    def _connect_signal(worker, name: str, callback):
        sig = getattr(worker, name, None)
        if sig is not None and hasattr(sig, "connect"):
            sig.connect(callback)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    """
    Left panel  – quick settings, tabbed inputs, run/stop, status, Dice/IoU, log
    Right panel – live overlay canvas (matplotlib) + zoom toolbar
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeXalign — Modular Pipeline")
        self.resize(1320, 860)

        # Defaults come straight from the pipeline config so the UI and backend
        # never drift apart.
        from .core import PipelineConfig
        self.cfg = PipelineConfig()

        self._thread: QtCore.QThread | None = None
        self._worker: MeXalignRunWorker | None = None

        # Preset-only state (no dedicated widget)
        self._preset_variant = PRESETS["Mattes"]["variant"]

        # Matplotlib artists kept so we update data without a full redraw (preserves zoom)
        self._ref_artist = None
        self._mov_artist = None

        # Channel-list state: filenames in the folder + their in-plane (W, H) sizes.
        self._images: list[str] = []
        self._file_sizes: dict[str, tuple] = {}
        self._updating_combos = False

        # Last folding-QC summary (single-run mode); folded into the results panel.
        self._last_jacobian: dict | None = None
        self._last_correspondence: dict | None = None
        self._last_bidirectional: dict | None = None
        self._last_timings: dict | None = None
        self._last_percell: dict | None = None

        # Derived B-Spline mesh in (W, H) = SimpleITK (x, y) order; filled from the
        # config default then recomputed from the control-point spacing. Shown in the
        # grid hint, no editable widget.
        self._bs_grid_wh: tuple = tuple(self.cfg.bspline_config.bspline_mesh)

        self._build_ui()
        self._apply_defaults()
        # Open on "MSQ fast" so the window shows the new defaults (MSQ, 150 µm, max 20
        # iters, early-stop window 3 / tol 0.01). setCurrentText fires _apply_preset.
        self.preset_combo.setCurrentText("MSQ fast")
        # "MSQ fast" is already index 0, so setCurrentText does not change the
        # text and its currentTextChanged signal will NOT fire — apply it directly
        # so the preset is actually applied on startup (not just displayed).
        self._apply_preset("MSQ fast")

        # Watch the chosen folder and auto-rescan its contents.
        self._dir_watcher = QtCore.QFileSystemWatcher(self)
        self._dir_watcher.directoryChanged.connect(lambda _p: self._scan_channels())
        self.directory_edit.editingFinished.connect(self._on_dir_input_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(8)

        # ---- LEFT: control panel ----------------------------------------
        # The panel's natural height (tabs 512 + results + log + labels ~= 920 px)
        # exceeds a laptop screen, which previously pushed the pipeline log off the
        # bottom. Wrapping it in a scroll area makes every control reachable at any
        # window/screen height instead of being clipped.
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(470)  # deterministic width; fits inside the 490 scroll
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        title = QtWidgets.QLabel("<b>MeXalign Settings</b>")
        title.setStyleSheet("font-size: 14px; margin-bottom: 2px;")
        left_layout.addWidget(title)

        # Quick settings preset
        preset_row = QtWidgets.QFormLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addRow("Quick settings:", self.preset_combo)
        left_layout.addLayout(preset_row)

        # Tabs — document mode + borderless pane so the tab content blends
        # seamlessly into the panel instead of looking like a boxed widget.
        self.tabs = QtWidgets.QTabWidget()
        # Pill / segmented-style tabs — pure Qt stylesheet (QSS), no extra library.
        # palette(...) keeps the colours adaptive to the OS light/dark theme and the
        # user's accent colour. Selected tab = filled accent pill; others = plain text.
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 0; margin: 0; padding: 0; }"
            "QTabBar { border: 0; background: transparent; }"
            "QTabBar::tab {"
            "  background: transparent;"
            "  color: palette(text);"
            "  padding: 6px 16px;"
            "  margin: 2px 6px 8px 0;"
            "  border: 0;"
            "  border-radius: 13px;"
            "}"
            "QTabBar::tab:hover:!selected { background: palette(midlight); }"
            "QTabBar::tab:selected {"
            "  background: palette(highlight);"
            "  color: palette(highlighted-text);"
            "}"
        )
        self.tabs.addTab(self._build_tab_inputs(), "Inputs")
        self.tabs.addTab(self._build_tab_fft(), "FFT")
        self.tabs.addTab(self._build_tab_rigid_affine(), "Rigid/Affine")
        self.tabs.addTab(self._build_tab_bspline(), "B-Spline")
        left_layout.addWidget(self.tabs)

        # Run / Stop
        btn_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("▶  Run Alignment")
        self.run_btn.setStyleSheet(
            "font-weight: bold; padding: 8px; background-color: #1a7f37; color: white; border-radius: 4px;"
        )
        self.stop_btn = QtWidgets.QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "font-weight: bold; padding: 8px; background-color: #b91c1c; color: white; border-radius: 4px;"
        )
        self.run_btn.clicked.connect(self._start_run)
        self.stop_btn.clicked.connect(self._stop_run)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        left_layout.addLayout(btn_row)

        # Status + metrics
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        left_layout.addWidget(self.status_label)

        self.metrics_label = QtWidgets.QLabel("Dice: —  |  IoU: —")
        self.metrics_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.metrics_label.setToolTip(
            "Dice score: 1.0 = perfect overlap, 0.0 = no overlap.\n"
            "IoU (Intersection over Union): related measure of alignment quality."
        )
        left_layout.addWidget(self.metrics_label)

# Only ONE score is shown outside the results box: the headline
        # "All-cell score with fused-cell groups" (set in _on_completed). The per-cell
        # label is kept as a hidden widget so existing handlers stay valid.
        self.percell_label = QtWidgets.QLabel("")
        self.percell_label.setVisible(False)

        left_layout.addWidget(QtWidgets.QLabel("<b>Results</b>"))
        # QTextEdit (not QPlainTextEdit) so one score line can be colour-coded via
        # HTML; setPlainText still works for the sweep table and tracebacks.
        self.result_box = QtWidgets.QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Dice / IoU / folding / per-stage timings appear here after the run.")
        # Monospace so the sweep comparison table lines up. NOTE: do NOT use
        # QFontDatabase.systemFont(FixedFont) — it returns the generic alias
        # "monospace" with fixedPitch=False, which renders proportionally and makes
        # the columns drift. Name real families and let the style hint pick a fallback.
        self.result_box.setFont(_monospace_font())
        # Never wrap: the panel is ~60 monospace chars wide, and a wrapped table or
        # a wrapped output path is unreadable. Scroll horizontally instead.
        self.result_box.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        # NOT setFixedHeight: a fixed height makes the whole left panel unshrinkable,
        # so the window is forced taller than the screen and the log falls off the
        # bottom. A minimum + stretch lets it fit any window and grow when there is room.
        self.result_box.setMinimumHeight(110)
        left_layout.addWidget(self.result_box, stretch=3)

        left_layout.addWidget(QtWidgets.QLabel("<b>Pipeline log</b>"))
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Step-by-step progress will appear here…")
        self.log_box.setMinimumHeight(72)
        left_layout.addWidget(self.log_box, stretch=2)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(490)  # 470 panel + room for a vertical scrollbar
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(left_scroll)

        # ---- RIGHT: live overlay canvas ---------------------------------
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 8, 8, 8)

        canvas_title = QtWidgets.QLabel(
            "<b>Live Alignment Preview</b>  "
            "<span style='color:#555; font-weight:normal; font-size:11px;'>"
            "Grey = fixed (reference) · Colour = moving (being aligned) · "
            "Use toolbar to zoom/pan</span>"
        )
        right_layout.addWidget(canvas_title)

        self._fig = Figure(facecolor="#1e1e1e")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e1e1e")
        self._ax.axis("off")
        self._ax.set_title("Waiting for run to start…", color="white", fontsize=10, pad=6)

        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, right_panel)
        self._toolbar.setStyleSheet("QToolBar { background: #2d2d2d; }")
        # The toolbar's Save button exports ONLY the overlay (the axes image), not the
        # whole window (dark background + margins + title). See _save_overlay.
        self._toolbar.save_figure = lambda *a: self._save_overlay()

        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self._canvas, stretch=1)

        root.addWidget(right_panel, stretch=1)

    # ---- Tab: Inputs --------------------------------------------------

    def _build_tab_inputs(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)  # flush with the rest of the panel

        form = QtWidgets.QFormLayout()
        self.directory_edit = QtWidgets.QLineEdit()
        self.directory_edit.setPlaceholderText("Select folder containing your image files…")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_directory)
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self.directory_edit)
        dir_row.addWidget(browse_btn)
        form.addRow("Data folder:", dir_row)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("<b>Registration Masks</b>"))
        form_masks = QtWidgets.QFormLayout()
        self.fix_mask_combo = QtWidgets.QComboBox()
        self.mov_mask_combo = QtWidgets.QComboBox()
        self.fix_mask_combo.currentTextChanged.connect(self._update_bs_suggestion)
        # Changing either mask re-filters the channel list (a channel must share the
        # moving mask's dimensions; the masks themselves are excluded).
        self.fix_mask_combo.currentTextChanged.connect(self._rebuild_channel_list)
        self.mov_mask_combo.currentTextChanged.connect(self._rebuild_channel_list)
        form_masks.addRow("Fixed mask (reference):", self.fix_mask_combo)

        self.fix_ps = QtWidgets.QDoubleSpinBox()
        self.fix_ps.setDecimals(4); self.fix_ps.setRange(0, 100)
        self.fix_ps.valueChanged.connect(self._update_bs_suggestion)
        form_masks.addRow("Fixed pixel size [µm/px]:", self.fix_ps)

        form_masks.addRow("Moving mask:", self.mov_mask_combo)
        self.mov_ps = QtWidgets.QDoubleSpinBox()
        self.mov_ps.setDecimals(4); self.mov_ps.setRange(0, 100)
        form_masks.addRow("Moving pixel size [µm/px]:", self.mov_ps)

        # Per-cell gradient orientation (used by both the rigid/affine and B-Spline
        # stages). Center-based = cell cores bright (default); Edge-based = edges
        # bright. Was previously a "Gradient turned" quick-settings preset.
        self.grad_mode = QtWidgets.QComboBox()
        self.grad_mode.addItems(["Center-based", "Edge-based"])
        self.grad_mode.setToolTip(
            "How the per-cell gradient is generated.\n"
            "Center-based: cell cores bright, edges dark (default).\n"
            "Edge-based:   edges bright, cell cores dark."
        )
        form_masks.addRow("Gradient generation:", self.grad_mode)

        self.second_mask_combo = QtWidgets.QComboBox()
        self.second_mask_combo.currentTextChanged.connect(self._rebuild_channel_list)
        form_masks.addRow("Second mask (optional):", self.second_mask_combo)
        layout.addLayout(form_masks)

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("<b>Channel Batch Processing</b>"))
        hint = QtWidgets.QLabel(
            "Tick the channel images to warp with the masks. Only images matching "
            "the moving mask's dimensions are listed."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Show ~5 rows; the list scrolls when there are more channels.
        row_h = self.file_list.fontMetrics().height() + 6
        self.file_list.setFixedHeight(row_h * 5 + 2 * self.file_list.frameWidth() + 4)
        layout.addWidget(self.file_list)
        layout.addStretch()
        return tab

    # ---- Tab: FFT -----------------------------------------------------

    def _build_tab_fft(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)  # flush with the rest of the panel
        layout.addWidget(QtWidgets.QLabel("<b>FFT Prealignment</b>"))
        note = QtWidgets.QLabel(
            "Coarse spacing is set automatically to the larger of the two pixel sizes "
            "(and coarsened further if the search grid would exceed 4000 px)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()

        self.fft_erode_chk = QtWidgets.QCheckBox("Apply erosion to masks (FFT)")
        form.addRow(self.fft_erode_chk)
        self.fft_erode_rad = QtWidgets.QSpinBox()
        self.fft_erode_rad.setRange(1, 100); self.fft_erode_rad.setValue(1)
        form.addRow("Erosion radius [px]:", self.fft_erode_rad)

        self.crop_padding_px = QtWidgets.QSpinBox()
        self.crop_padding_px.setRange(0, 10000); self.crop_padding_px.setSingleStep(10)
        form.addRow("Crop padding [px]:", self.crop_padding_px)

        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("<b>Rotation Search</b>"))
        rform = QtWidgets.QFormLayout()
        self.fft_rot_chk = QtWidgets.QCheckBox("Enable rotation search")
        rform.addRow(self.fft_rot_chk)
        self.fft_rot_range = QtWidgets.QDoubleSpinBox()
        self.fft_rot_range.setRange(0.0, 180.0); self.fft_rot_range.setDecimals(2); self.fft_rot_range.setSingleStep(0.5)
        rform.addRow("Rotation range (± deg):", self.fft_rot_range)
        self.fft_rot_step = QtWidgets.QDoubleSpinBox()
        self.fft_rot_step.setRange(0.05, 45.0); self.fft_rot_step.setDecimals(2); self.fft_rot_step.setSingleStep(0.05)
        rform.addRow("Step size (deg):", self.fft_rot_step)
        layout.addLayout(rform)
        layout.addStretch()
        return tab

    # ---- Tab: Rigid/Affine (no erosion here, per design) -----------------------

    def _build_tab_rigid_affine(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)  # flush with the rest of the panel
        form = QtWidgets.QFormLayout()

        form.addRow(QtWidgets.QLabel("<b>Coarse (Euler / rigid)</b>"))
        self.coarse_iter = QtWidgets.QSpinBox(); self.coarse_iter.setRange(1, 10000)
        self.coarse_lr = QtWidgets.QDoubleSpinBox(); self.coarse_lr.setRange(0, 10); self.coarse_lr.setSingleStep(0.1)
        self.coarse_samp = QtWidgets.QDoubleSpinBox(); self.coarse_samp.setRange(0.01, 1.0); self.coarse_samp.setSingleStep(0.05)
        form.addRow("Iterations:", self.coarse_iter)
        form.addRow("Learning rate:", self.coarse_lr)
        form.addRow("Metric sampling (0-1):", self.coarse_samp)
        # Shrink factors are fixed at [8, 4, 2] (coarse->fine pyramid) — no widget.

        form.addRow(QtWidgets.QLabel(""))
        form.addRow(QtWidgets.QLabel("<b>Fine (Affine)</b>"))
        self.affine_iter = QtWidgets.QSpinBox(); self.affine_iter.setRange(1, 10000)
        self.affine_lr = QtWidgets.QDoubleSpinBox(); self.affine_lr.setRange(0, 10); self.affine_lr.setSingleStep(0.1)
        self.affine_samp = QtWidgets.QDoubleSpinBox(); self.affine_samp.setRange(0.01, 1.0); self.affine_samp.setSingleStep(0.05)
        self.affine_shrink = QtWidgets.QLineEdit(); self.affine_shrink.setPlaceholderText("e.g. 4,2,1")
        form.addRow("Iterations:", self.affine_iter)
        form.addRow("Learning rate:", self.affine_lr)
        form.addRow("Metric sampling (0-1):", self.affine_samp)
        form.addRow("Shrink factors:", self.affine_shrink)

        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel(
            "<i>Stage signal: per-cell gradients (shared with B-Spline). "
            "Smoothing sigmas auto-derived from shrink factors.</i>"
        ))

        layout.addStretch()
        return tab

    # ---- Tab: B-Spline ------------------------------------------------

    def _build_tab_bspline(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)  # flush with the rest of the panel
        form = QtWidgets.QFormLayout()

        # Control-point spacing is set in PHYSICAL units (µm), so the grid stays at a
        # constant physical density regardless of pixel size. The grid (mesh cell
        # count) is derived from it: cells = ceil(physical_extent / spacing).
        self.bs_spacing = QtWidgets.QDoubleSpinBox()
        self.bs_spacing.setDecimals(1); self.bs_spacing.setRange(1.0, 100000.0)
        self.bs_spacing.setSingleStep(10.0); self.bs_spacing.setValue(200.0)
        self.bs_spacing.setSuffix(" µm")
        self.bs_spacing.setToolTip(
            "Target distance between B-Spline control points at the coarsest level, in\n"
            "micrometres. The grid below is computed from this and the fixed image's\n"
            "physical size, so it is independent of pixel size. ~200 µm ≈ 10-30 cells\n"
            "per control point for typical tissue."
        )
        form.addRow("Control-point spacing:", self.bs_spacing)

        # The derived grid (mesh size, W,H) is shown in this hint and stored in
        # self._bs_grid_wh; there is no separate editable field for it.
        self.bs_grid_hint = QtWidgets.QLabel("Grid: (select fixed mask)")
        self.bs_grid_hint.setStyleSheet("color: gray; font-style: italic;")
        self.bs_grid_hint.setWordWrap(True)
        form.addRow("Grid:", self.bs_grid_hint)
        self.bs_iter = QtWidgets.QSpinBox(); self.bs_iter.setRange(1, 10000)
        self.bs_iter.setToolTip("Hard maximum LBFGS2 iterations per pyramid level. "
                                "Early stopping usually finishes sooner.")
        form.addRow("Maximum iterations:", self.bs_iter)
        self.bs_shrink = QtWidgets.QLineEdit(); self.bs_shrink.setPlaceholderText("e.g. 4,2,1")
        form.addRow("Shrink factors:", self.bs_shrink)

        self.bs_metric = QtWidgets.QComboBox()
        # Display MSQ / MMI; internally still "msq" / "mattes".
        self.bs_metric.addItems(["MSQ", "MMI"])
        self.bs_metric.setToolTip("MSQ = Mean Squares; MMI = Mattes Mutual Information.")
        form.addRow("Metric:", self.bs_metric)

        self.bs_scale = QtWidgets.QLineEdit(); self.bs_scale.setPlaceholderText("e.g. 1,2,4  (one per shrink level)")
        form.addRow("Mesh scale factors:", self.bs_scale)

        # LBFGS2 early stopping: stop a pyramid level once the metric stops improving
        # meaningfully. Window = iterations compared over (0 disables); tolerance =
        # minimum relative improvement across that window.
        self.bs_early_window = QtWidgets.QSpinBox(); self.bs_early_window.setRange(0, 1000)
        self.bs_early_window.setToolTip(
            "Early-stop window: how many iterations back the metric is compared over.\n"
            "0 disables metric-improvement early stopping."
        )
        form.addRow("Early-stop window:", self.bs_early_window)
        self.bs_early_tol = QtWidgets.QDoubleSpinBox()
        self.bs_early_tol.setDecimals(4); self.bs_early_tol.setRange(0.0, 1.0)
        self.bs_early_tol.setSingleStep(0.01)
        self.bs_early_tol.setToolTip(
            "Early-stop tolerance: minimum relative metric improvement across the "
            "window.\nBelow this, the level stops early. 0.01 = 1%."
        )
        form.addRow("Early-stop tolerance:", self.bs_early_tol)

        # Re-derive the grid whenever the spacing (or the mesh scale factors, which
        # set the finest-level display) change.
        self.bs_spacing.valueChanged.connect(self._update_bs_suggestion)
        self.bs_scale.editingFinished.connect(self._update_bs_suggestion)

        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel(
            "<i>Optimizer: LBFGS2 (auto step size, full sampling inside mask). "
            "'Maximum iterations' is a hard cap; early stopping ends a level sooner "
            "when the metric plateaus.</i>"
        ))
        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Defaults + presets
    # ------------------------------------------------------------------

    def _apply_defaults(self):
        cfg = self.cfg
        self.fix_ps.setValue(cfg.fixed_pixel_size)
        self.mov_ps.setValue(cfg.moving_pixel_size)

        self.fft_erode_chk.setChecked(cfg.fft_erosion_enabled)
        self.fft_erode_rad.setValue(cfg.fft_erosion_radius)
        self.crop_padding_px.setValue(cfg.crop_padding_px)
        self.fft_rot_chk.setChecked(cfg.rotation_enabled)
        self.fft_rot_range.setValue(cfg.rotation_range)
        self.fft_rot_step.setValue(cfg.rotation_step)

        self.coarse_iter.setValue(cfg.coarse_config.iters)
        self.coarse_lr.setValue(cfg.coarse_config.learning_rate)
        self.coarse_samp.setValue(cfg.coarse_config.sampling_percentage)

        self.affine_iter.setValue(cfg.affine_config.iters)
        self.affine_lr.setValue(cfg.affine_config.learning_rate)
        self.affine_samp.setValue(cfg.affine_config.sampling_percentage)
        self.affine_shrink.setText(",".join(map(str, cfg.affine_config.shrink_factors)))

        self.bs_iter.setValue(cfg.bspline_config.iters)
        self.bs_shrink.setText(",".join(map(str, cfg.bspline_config.shrink_factors)))
        self._bs_grid_wh = tuple(cfg.bspline_config.bspline_mesh)  # derived mesh (W,H); no widget
        self.bs_scale.setText(",".join(map(str, cfg.bspline_mesh_scale_factors)))
        self.bs_metric.setCurrentText("MMI" if str(cfg.bspline_metric).lower() == "mattes" else "MSQ")
        self.bs_early_window.setValue(getattr(cfg, "bspline_early_stop_window", 3))
        self.bs_early_tol.setValue(getattr(cfg, "bspline_early_stop_tolerance", 0.01))
        self._set_combo(self.grad_mode, "Edge-based" if cfg.gradient_edges_bright else "Center-based")
        self._update_bs_suggestion()

    def _apply_preset(self, name: str):
        preset = PRESETS.get(name)
        if not preset:
            return
        self._preset_variant = preset["variant"]
        self.bs_metric.setCurrentText("MMI" if str(preset["bspline_metric"]).lower() == "mattes" else "MSQ")
        self.bs_spacing.setValue(preset["bs_spacing"])
        self.bs_iter.setValue(preset["bs_iter"])
        self.bs_shrink.setText(preset["bs_shrink"])
        self.bs_scale.setText(preset["bs_scale"])
        self.bs_early_window.setValue(preset["bs_early_window"])
        self.bs_early_tol.setValue(preset["bs_early_tol"])
        self._update_bs_suggestion()

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value: str):
        idx = combo.findText(str(value).lower())
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Directory / file scanning
    # ------------------------------------------------------------------

    def _browse_directory(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder containing your image files"
        )
        if selected:
            self.directory_edit.setText(selected)
            self._on_dir_input_changed()

    def _on_dir_input_changed(self):
        path = self.directory_edit.text().strip()
        watched = self._dir_watcher.directories()
        if watched:
            self._dir_watcher.removePaths(watched)
        if path and os.path.isdir(path):
            self._dir_watcher.addPath(path)
        self._scan_channels()

    def _read_inplane_size(self, fname: str):
        """In-plane (width, height) of an image header, or None if unreadable."""
        d = self.directory_edit.text().strip()
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(os.path.join(d, fname))
            reader.ReadImageInformation()
            sz = reader.GetSize()
            return (int(sz[0]), int(sz[1])) if len(sz) >= 2 else None
        except Exception:
            return None

    def _scan_channels(self):
        d = self.directory_edit.text().strip()
        if not d or not os.path.exists(d):
            return
        files = sorted(os.listdir(d))
        images = [f for f in files if f.lower().endswith((".png", ".tif", ".tiff"))]
        self._images = images
        # Header-only read (fast) so we can filter channels by dimension.
        self._file_sizes = {f: self._read_inplane_size(f) for f in images}

        # Bulk-update the mask dropdowns without triggering a rebuild per change.
        self._updating_combos = True
        cur_fix = self.fix_mask_combo.currentText()
        cur_mov = self.mov_mask_combo.currentText()
        cur_sec = self.second_mask_combo.currentText()
        self.fix_mask_combo.clear(); self.fix_mask_combo.addItems(images)
        self.mov_mask_combo.clear(); self.mov_mask_combo.addItems(images)
        if cur_fix and cur_fix in images:
            self.fix_mask_combo.setCurrentText(cur_fix)
        elif "ICP_mask.png" in images:
            self.fix_mask_combo.setCurrentText("ICP_mask.png")
        if cur_mov and cur_mov in images:
            self.mov_mask_combo.setCurrentText(cur_mov)
        elif "F_mask.png" in images:
            self.mov_mask_combo.setCurrentText("F_mask.png")
        # Second mask dropdown: first item is always "— none —" (no second mask).
        none_label = "— none —"
        self.second_mask_combo.clear()
        self.second_mask_combo.addItem(none_label)
        self.second_mask_combo.addItems(images)
        if cur_sec and cur_sec in images:
            self.second_mask_combo.setCurrentText(cur_sec)
        else:
            self.second_mask_combo.setCurrentIndex(0)  # "— none —"
        self._updating_combos = False

        self._rebuild_channel_list()
        self._update_bs_suggestion()

    def _rebuild_channel_list(self, *_):
        """Populate the channel list with only the files eligible for warping:
        same in-plane dimensions as the moving mask, and never the masks themselves
        (the moving mask is auto-transformed and saved; the fixed mask is the
        reference). Pattern filtering is applied on top.
        """
        if self._updating_combos:
            return
        # Preserve existing tick states across rebuilds.
        prev = {}
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            prev[it.text()] = it.checkState()

        mov = self.mov_mask_combo.currentText().strip()
        fix = self.fix_mask_combo.currentText().strip()
        sec_raw = self.second_mask_combo.currentText().strip()
        sec = "" if sec_raw == "— none —" else sec_raw
        ref_size = self._file_sizes.get(mov)

        self.file_list.clear()
        for f in self._images:
            if f in (mov, fix) or (sec and f == sec):
                continue  # masks are never channels
            size = self._file_sizes.get(f)
            # Only warp images sharing the moving mask's pixel grid. If the moving
            # size is unknown, fall back to showing everything (can't judge).
            if ref_size is not None and size is not None and size != ref_size:
                continue
            item = QtWidgets.QListWidgetItem(f)
            item.setCheckState(prev.get(f, QtCore.Qt.Checked))
            self.file_list.addItem(item)

    def _update_bs_suggestion(self, *_):
        d = self.directory_edit.text().strip()
        fname = self.fix_mask_combo.currentText().strip()
        if not d or not fname:
            self.bs_grid_hint.setText("Grid: (select fixed mask)")
            return
        full_path = os.path.join(d, fname)
        if not os.path.exists(full_path):
            self.bs_grid_hint.setText("Grid: (file not found)")
            return
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(full_path)
            reader.ReadImageInformation()
            w, h = reader.GetSize()[:2]
            px = self.fix_ps.value()
            spacing = self.bs_spacing.value()
            if px <= 0 or spacing <= 0:
                self.bs_grid_hint.setText("Grid: (set a positive pixel size and spacing)")
                return

            # Physical extent of the fixed image, then mesh cells from the target
            # PHYSICAL spacing — independent of pixel size.
            w_um, h_um = w * px, h * px
            sugg_w = max(1, math.ceil(w_um / spacing))
            sugg_h = max(1, math.ceil(h_um / spacing))

            # Realized spacing (ceil rounds the count up, so it is <= target), and the
            # finest-level spacing after the mesh scale factors refine the grid.
            real_w, real_h = w_um / sugg_w, h_um / sugg_h
            try:
                max_scale = max(int(x) for x in self.bs_scale.text().split(",") if x.strip())
            except ValueError:
                max_scale = 1
            fine_w, fine_h = real_w / max_scale, real_h / max_scale

            self.bs_grid_hint.setText(
                f"Fixed: {w}×{h} px @ {px:g} µm/px  →  {w_um:.0f}×{h_um:.0f} µm.  "
                f"Grid (W,H) = {sugg_w},{sugg_h}.  "
                f"Realized spacing ≈ {real_w:.0f}×{real_h:.0f} µm coarsest, "
                f"{fine_w:.0f}×{fine_h:.0f} µm finest (÷{max_scale})."
            )
            # (W, H) = SimpleITK (x, y) order, so the mesh size maps to the matching
            # image axes in BSplineTransformInitializer.
            self._bs_grid_wh = (sugg_w, sugg_h)
        except (RuntimeError, OSError) as e:
            self.bs_grid_hint.setText(f"Grid: error reading image ({e})")

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _start_run(self):
        directory_text = self.directory_edit.text().strip()
        if not directory_text:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please select a data folder first.")
            return

        try:
            overrides = self._collect_overrides()
            fixed_name = self.fix_mask_combo.currentText().strip()
            moving_name = self.mov_mask_combo.currentText().strip()
            if not fixed_name or not moving_name:
                raise ValueError("Select both a fixed and a moving mask.")
            if not os.path.exists(os.path.join(directory_text, fixed_name)):
                raise ValueError(f"Fixed mask not found: {fixed_name}")
            if not os.path.exists(os.path.join(directory_text, moving_name)):
                raise ValueError(f"Moving mask not found: {moving_name}")
            # Never warp the masks themselves as channels (the fixed mask is the
            # reference; the moving mask is already saved as registered_mask).
            mask_stems = {
                os.path.splitext(fixed_name)[0],
                os.path.splitext(moving_name)[0],
            }
            selected = [s for s in self._collect_selected_channels() if s not in mask_stems]
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        run_input = MeXalignRunInput(
            directory=Path(directory_text),
            fixed_mask_filename=fixed_name,
            moving_mask_filename=moving_name,
            script_variant=self._preset_variant,
            selected_moving_channels=selected,
            overrides=overrides,
        )

        # Reset UI
        self.log_box.clear()
        self.result_box.clear()
        self.metrics_label.setText("ALL-CELL SCORE WITH FUSED-CELL GROUPS<br>—")
        self.percell_label.setText("")
        self._last_jacobian = None
        self._last_timings = None
        self._last_correspondence = None
        self._last_bidirectional = None
        self._last_percell = None
        self._reset_canvas()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running…")

        # Launch worker thread
        self._thread = QtCore.QThread(self)
        self._worker = MeXalignRunWorker(run_input)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(lambda s: self.status_label.setText(f"Status: {s}"))
        self._worker.overlay_update.connect(self._plot_overlay)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.aborted.connect(self._on_aborted)
        self._worker.percell.connect(self._on_percell)
        self._worker.jacobian.connect(self._on_jacobian)
        self._worker.correspondence.connect(self._on_correspondence)
        self._worker.bidirectional.connect(self._on_bidirectional)
        self._worker.timings.connect(self._on_timings)
        self._worker.done.connect(self._cleanup_thread)
        self._thread.start()

    def _stop_run(self):
        if self._worker is not None:
            self._worker.abort()
            self._append_log("Stop requested — finishing current step…")
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Override / selection collection
    # ------------------------------------------------------------------

    def _collect_overrides(self) -> dict:
        """Build the full config override dict from every tab widget."""
        # Coarse (Euler) uses a fixed coarse-to-fine pyramid (no widget): [8,4,2].
        coarse_shrinks = [8, 4, 2]
        affine_shrinks = self._parse_shrinks(self.affine_shrink.text(), "Fine (Affine)", "4,2,1")
        bs_shrinks = self._parse_shrinks(self.bs_shrink.text(), "B-Spline", "4,2,1")
        gx, gy = (int(v) for v in self._bs_grid_wh)  # derived from spacing; no widget
        scale_factors = self._parse_shrinks(self.bs_scale.text(), "Mesh scale", "1,2,4")
        if len(scale_factors) != len(bs_shrinks):
            raise ValueError(
                f"Mesh scale factors ({len(scale_factors)}) must match the number of "
                f"B-Spline shrink levels ({len(bs_shrinks)})."
            )

        # Sigmas go to ITK in physical units, and the pyramid shrinks the FIXED
        # grid, so the fixed pixel size converts shrink-in-pixels to sigma-in-µm.
        fix_ps = self.fix_ps.value()

        return {
            "fixed_pixel_size": fix_ps,
            "moving_pixel_size": self.mov_ps.value(),
            # FFT — coarse spacing is derived (no widget): the larger pixel size.
            "fft_spacing": max(fix_ps, self.mov_ps.value()),
            "fft_erosion_enabled": self.fft_erode_chk.isChecked(),
            "fft_erosion_radius": self.fft_erode_rad.value(),
            "crop_padding_px": self.crop_padding_px.value(),
            "rotation_enabled": self.fft_rot_chk.isChecked(),
            "rotation_range": self.fft_rot_range.value(),
            "rotation_step": self.fft_rot_step.value(),
            # Rigid/affine coarse
            "coarse_config.iters": self.coarse_iter.value(),
            "coarse_config.learning_rate": self.coarse_lr.value(),
            "coarse_config.sampling_percentage": self.coarse_samp.value(),
            "coarse_config.shrink_factors": coarse_shrinks,
            "coarse_config.smoothing_sigmas": self._auto_sigmas(coarse_shrinks, fix_ps),
            # Rigid/affine fine
            "affine_config.iters": self.affine_iter.value(),
            "affine_config.learning_rate": self.affine_lr.value(),
            "affine_config.sampling_percentage": self.affine_samp.value(),
            "affine_config.shrink_factors": affine_shrinks,
            "affine_config.smoothing_sigmas": self._auto_sigmas(affine_shrinks, fix_ps),
            # B-Spline
            "bspline_config.bspline_mesh": (gx, gy),
            "bspline_config.iters": self.bs_iter.value(),
            "bspline_config.shrink_factors": bs_shrinks,
            "bspline_config.smoothing_sigmas": self._auto_sigmas(bs_shrinks, fix_ps),
            "bspline_metric": "mattes" if self.bs_metric.currentText() == "MMI" else "msq",
            "bspline_mesh_scale_factors": scale_factors,
            "bspline_early_stop_window": self.bs_early_window.value(),
            "bspline_early_stop_tolerance": self.bs_early_tol.value(),
            # Gradient generation (Inputs tab dropdown): Edge-based -> edges bright.
            "gradient_edges_bright": self.grad_mode.currentText() == "Edge-based",
            # Second mask
            "second_mask_filename": (
                "" if self.second_mask_combo.currentText() == "— none —"
                else self.second_mask_combo.currentText()
            ),
        }

    def _collect_selected_channels(self) -> list:
        sel = []
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.isHidden() or it.checkState() != QtCore.Qt.Checked:
                continue
            sel.append(os.path.splitext(it.text())[0])
        return list(dict.fromkeys(sel))

    @staticmethod
    def _parse_shrinks(text: str, stage: str, fallback: str) -> list:
        try:
            sf = [int(x.strip()) for x in text.split(",") if x.strip()]
            if not sf:
                raise ValueError("empty")
            return sf
        except Exception:
            raise ValueError(f"Invalid {stage} shrink factors (use e.g. {fallback}).")

    @staticmethod
    def _auto_sigmas(shrinks: list, pixel_size: float = 1.0) -> list:
        """Anti-aliasing sigma per pyramid level, in PHYSICAL units (µm).

        A shrink factor N samples the fixed grid every N pixels, i.e. every
        N * pixel_size µm, so the blur that matches that sampling step is
        sigma = (N / 2) * pixel_size. The full-resolution level (shrink 1) is
        not downsampled, so it needs no filtering at all.

        The pixel_size factor matters because the sigmas are handed to ITK in
        physical units while the shrink factor counts pixels. At 1.0 µm/px the
        two coincide and this reproduces the historical sigma = N / 2.
        """
        sigmas = [(float(s) / 2.0) * float(pixel_size) for s in shrinks]
        if shrinks and shrinks[-1] == 1:
            sigmas[-1] = 0.0
        return sigmas

    # ------------------------------------------------------------------
    # Live overlay drawing
    # ------------------------------------------------------------------

    def _reset_canvas(self):
        self._ref_artist = None
        self._mov_artist = None
        self._ax.clear()
        self._ax.set_facecolor("#1e1e1e")
        self._ax.axis("off")
        self._ax.set_title("Alignment in progress…", color="white", fontsize=10, pad=6)
        self._canvas.draw_idle()
        if hasattr(self._toolbar, "_nav_stack"):
            self._toolbar._nav_stack.clear()

    def _save_overlay(self):
        """Save only the overlay image (the axes region) — no surrounding window
        background, margins or title. Respects the current zoom/pan."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save overlay image", "overlay.png",
            "PNG image (*.png);;TIFF image (*.tif);;JPEG image (*.jpg)")
        if not path:
            return
        title = self._ax.get_title()
        self._ax.set_title("")                 # keep the title out of the crop
        self._canvas.draw()
        extent = self._ax.get_window_extent().transformed(
            self._fig.dpi_scale_trans.inverted())
        try:
            self._fig.savefig(path, dpi=200, bbox_inches=extent, transparent=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
        finally:
            self._ax.set_title(title)
            self._canvas.draw_idle()

    def _plot_overlay(self, overlay_data):
        if not overlay_data:
            return
        ref_arr, mov_arr = overlay_data
        h, w = ref_arr.shape[:2]
        if (
            self._ref_artist is None
            or self._mov_artist is None
            or self._ref_artist.get_array().shape != ref_arr.shape
        ):
            self._ax.clear()
            self._ax.set_facecolor("#1e1e1e")
            self._ax.axis("off")
            self._ax.set_title(
                "Live preview  —  grey = fixed · colour = moving",
                color="white", fontsize=9, pad=4
            )
            self._ref_artist = self._ax.imshow(
                ref_arr, cmap="gray", extent=[0, w, h, 0],
                interpolation="bilinear", vmin=0, vmax=1
            )
            self._mov_artist = self._ax.imshow(
                mov_arr, cmap="jet", extent=[0, w, h, 0],
                interpolation="bilinear", alpha=0.5, vmin=0, vmax=1
            )
        else:
            self._ref_artist.set_data(ref_arr)
            self._mov_artist.set_data(mov_arr)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Worker result handlers
    # ------------------------------------------------------------------

    def _on_completed(self, result: MeXalignRunResult):
        dice = result.evaluation.dice
        iou = result.evaluation.iou

        # The ONLY score shown outside the results box: the headline "All-cell score
        # with fused-cell groups" (nothing excluded), colour-coded green/red. Every
        # other score lives in the results box below.
        kk = self._last_correspondence or {}
        cpl = kk.get("complete")
        jj = self._last_jacobian or {}
        # Tissue-folding verdict, green/red (folding INSIDE the fixed-tissue mask).
        if jj:
            fold_line = (
                "<span style='color:#b91c1c'>Tissue folding</span>"
                if jj.get("folding_in_mask")
                else "<span style='color:#1a7f37'>No tissue folding</span>")
        else:
            fold_line = ""
        if cpl:
            n_1to1 = kk.get("n_one_to_one", 0)
            n_multi = (kk.get("n_icp_doublets", 0) + kk.get("n_if_merges", 0)
                       + kk.get("n_complex", 0))          # number of fused groups
            n_cells_fused = kk.get("n_cells_in_multiplets", 0)  # cells inside those groups
            n_unmatched = kk.get("n_icp_only", 0) + kk.get("n_if_only", 0)
            colour = "#1a7f37" if cpl["mean_dice"] >= 0.6 else "#b91c1c"
            self.metrics_label.setText(
                f"<span style='color:{colour}'>ALL-CELL SCORE WITH FUSED-CELL GROUPS<br>"
                f"Dice: {cpl['mean_dice']:.4f}  |  IoU: {cpl['mean_iou']:.4f}<br>"
                f"{n_1to1} 1:1 matches; {n_cells_fused} cells fused to {n_multi} multiplets, "
                f"{n_unmatched} unmatched cells</span>"
                + (f"<br>{fold_line}" if fold_line else "")
            )
        else:
            self.metrics_label.setText(
                "ALL-CELL SCORE WITH FUSED-CELL GROUPS<br>—"
                + (f"<br>{fold_line}" if fold_line else ""))
        self.metrics_label.setTextFormat(QtCore.Qt.RichText)

        # Same table as the sweep, with one row — so a single run and a sweep read
        # identically. The output folder goes on a trailing line.
        p = self._last_percell or {}
        # Sectioned, NARROW output. The left panel is ~470 px (~60 monospace chars),
        # so a single wide table wraps and becomes unreadable — keep every line short.
        j = self._last_jacobian or {}
        t = self._last_timings or {}
        k = self._last_correspondence
        f = lambda v: "—" if v is None else f"{v:.1f}"
        # The global Dice/IoU is already on the big label directly above, so it is
        # not repeated here — this box carries the breakdown the label cannot show.
        # Score names follow the manuscript. Only the headline score — "All-cell score
        # with fused-cell groups" — is colour-coded (green good / red poor); the others
        # are plain so the eye goes to the one that represents every object fairly.
        import html as _html
        rows: list[str] = []

        def add(title, dice, iou, detail, explain, colour=None):
            head = f"{title}"
            if colour:
                head = f"<span style='color:{colour};font-weight:bold'>{_html.escape(title)}</span>"
                val = (f"<span style='color:{colour};font-weight:bold'>"
                       f"  Dice {dice:.4f}   IoU {iou:.4f}</span>")
            else:
                head = _html.escape(title)
                val = _html.escape(f"  Dice {dice:.4f}   IoU {iou:.4f}")
            rows.append(head)
            rows.append(val)
            if detail:
                rows.append(_html.escape(f"  {detail}"))
            for line in explain:
                rows.append(f"<span style='color:#888'>{_html.escape('  ' + line)}</span>")
            rows.append("")

        # Total objects in each mask, above the silhouette score.
        bd0 = self._last_bidirectional or {}
        n_icp_tot = kk.get("n_icp_cells", bd0.get("n_fixed"))
        n_if_tot = kk.get("n_if_cells", bd0.get("n_registered"))
        if n_icp_tot is not None and n_if_tot is not None:
            rows.append(_html.escape(
                f"MASKS: {n_icp_tot} fixed (ICP) cells | {n_if_tot} registered (IF) cells"))
            rows.append("")

        # Headline first: All-cell score WITH fused-cell groups (the representative one).
        cp0 = (k or {}).get("complete") if k else None
        if cp0:
            add("ALL-CELL SCORE WITH FUSED-CELL GROUPS",
                cp0["mean_dice"], cp0["mean_iou"],
                f"{cp0['n_scored']} groups + {cp0['n_zero']} unmatched = {cp0['n']} units",
                ["as above, but unmatched cells also contribute Dice/IoU = 0",
                 "-> nothing excluded: fusion corrected, missing/extra cells cost"])

        # (1) Silhouette-based score — pooled foreground, labels ignored.
        add("SILHOUETTE-BASED SCORE", dice, iou, None,
            ["pooled foreground pixels of both masks, cell labels ignored",
             "-> macroscopic / global tissue alignment"])

        # (2) All-cell score — every cell vs its best counterpart, unmatched = 0.
        bd = self._last_bidirectional
        if bd:
            add("ALL-CELL SCORE", bd["dice"], bd["iou"],
                f"{bd['n_fixed']} + {bd['n_registered']} cells scored | unmatched "
                f"{bd['n_unmatched_fixed']} fixed, {bd['n_unmatched_registered']} registered",
                ["every cell in BOTH masks vs its best-overlapping counterpart",
                 "cells without a counterpart are included as Dice/IoU = 0",
                 "-> fusion or splitting lowers this score"])

        if k:
            s, cr = k["strict"], k["corrected"]
            # (3) 1:1 matches only.
            add("1:1 MATCHES ONLY", s["mean_dice"], s["mean_iou"],
                f"{s['n']} unambiguous pairs | excluded: ICP-only {k['n_icp_only']}, "
                f"IF-only {k['n_if_only']}",
                ["only cells with a clear one-to-one counterpart",
                 "unmatched and fused/split groups are excluded",
                 "-> high value need not reflect all objects in the masks"])

            # (4) 1:1 and fused-cell groups.
            add("1:1 AND FUSED-CELL GROUPS", cr["mean_dice"], cr["mean_iou"],
                f"{cr['n']} groups | doublets {k['n_icp_doublets']}, merges "
                f"{k['n_if_merges']}, complex {k['n_complex']}",
                ["1:1 pairs plus matched fused groups; a group's cells are merged",
                 "before scoring, so a fused mask is not counted repeatedly",
                 "-> merging removes internal boundaries, may slightly overestimate"])

            # (5) All-cell score with fused-cell groups is shown FIRST (top of the box).
            rows.append(_html.escape(
                f"  cells: {k['n_icp_cells']} ICP | {k['n_if_cells']} IF   "
                f"(linked at >= {k['min_overlap_frac']:.0%} of either cell's area)"))
            rows.append("")

        if j:
            verdict = ("FOLDING IN TISSUE" if j.get("folding_in_mask")
                       else "folding (background only)" if j.get("folding") else "none")
            rows += [_html.escape(x) for x in [
                "FOLDING  (Jacobian det of the applied transform)",
                f"  min det(J) {j['min_in_mask']:.3f} in tissue -> {verdict}",
                f"  affine baseline {j['affine_det']:.3f} | {j['n_folded']} folded px",
                ""]]

        if t:
            rows += [_html.escape(x) for x in [
                "TIMING (s)",
                f"  gradient {f(t.get('gradient'))} | rigid/affine {f(t.get('rigid_affine'))}"
                f" | b-spline {f(t.get('bspline'))}",
                f"  total {f(t.get('total'))}",
                ""]]

        rows.append(_html.escape(f"Output: {result.output_dir}"))
        fam = self.result_box.font().family()
        self.result_box.setHtml(
            f"<div style='font-family:\"{fam}\",monospace;white-space:pre'>"
            + "<br>".join(rows) + "</div>"
        )
        self.status_label.setText("Status: Completed ✓")

    def _on_percell(self, summary):
        if not summary:
            return
        self._last_percell = summary
        self.percell_label.setText(
            f"Per-cell — Dice: {summary['mean_dice']:.4f}  |  IoU: {summary['mean_iou']:.4f}"
            f"   ({summary['n_evaluated']}/{summary['n_reference_cells']} cells"
            f", {summary['n_no_overlap']} no-overlap excluded, {summary['n_merges']} merges)"
        )

    def _on_jacobian(self, summary):
        """Folding QC arrived (single-run mode) — stash it for the results panel."""
        self._last_jacobian = summary or None

    def _on_timings(self, summary):
        """Per-stage seconds arrived (single-run mode) — stash for the results panel."""
        self._last_timings = summary or None

    def _on_correspondence(self, summary):
        """Correspondence metric arrived — stash it for the results panel."""
        self._last_correspondence = summary or None

    def _on_bidirectional(self, summary):
        """Bidirectional best-match metric arrived — stash for the results panel."""
        self._last_bidirectional = summary or None

    def _on_failed(self, tb: str):
        self.result_box.setPlainText(tb)
        self.status_label.setText("Status: Failed ✗")
        QtWidgets.QMessageBox.critical(
            self, "Registration failed",
            "The alignment encountered an error. Check the log box for details."
        )

    def _on_aborted(self):
        # User pressed Stop — quiet, no popup.
        self.status_label.setText("Status: Stopped")
        self._append_log("Run stopped by user.")

    def _cleanup_thread(self):
        self.stop_btn.setEnabled(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        self.run_btn.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _append_log(self, message: str):
        self.log_box.appendPlainText(message)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    """Force one consistent dark theme on every OS (macOS, Windows, Linux).

    Without this, Qt follows each platform's native style/palette — dark on a
    macOS in dark mode, light/white on Windows — so the app looks different per
    machine. Fusion is the cross-platform style; the explicit palette pins the
    colours so it renders identically everywhere.
    """
    app.setStyle("Fusion")

    window = QtGui.QColor(45, 45, 45)
    base = QtGui.QColor(30, 30, 30)
    text = QtGui.QColor(220, 220, 220)
    disabled = QtGui.QColor(127, 127, 127)
    highlight = QtGui.QColor(38, 110, 183)

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, window)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, base)
    pal.setColor(QtGui.QPalette.AlternateBase, window)
    pal.setColor(QtGui.QPalette.ToolTipBase, window)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, window)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))
    pal.setColor(QtGui.QPalette.Link, highlight)
    pal.setColor(QtGui.QPalette.Highlight, highlight)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    # Frame shades used by the pill tab stylesheet (hover background etc.).
    pal.setColor(QtGui.QPalette.Midlight, QtGui.QColor(60, 60, 60))
    pal.setColor(QtGui.QPalette.Mid, QtGui.QColor(80, 80, 80))
    pal.setColor(QtGui.QPalette.PlaceholderText, disabled)
    for group in (QtGui.QPalette.Disabled,):
        pal.setColor(group, QtGui.QPalette.WindowText, disabled)
        pal.setColor(group, QtGui.QPalette.Text, disabled)
        pal.setColor(group, QtGui.QPalette.ButtonText, disabled)
    app.setPalette(pal)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
