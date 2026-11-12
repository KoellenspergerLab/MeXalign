# MeXalign

## What is MeXalign?

MeXalign lines up two images of the same tissue (for example an imaging-mass-cytometry image and an immunofluorescence image) so that every cell sits at the same place in both.

It works on **cell masks** (label images: background = 0, every cell has its own number > 0), not on the raw signal images. Once the masks are aligned, the same transformation is applied to all the channel images you tick, so every channel ends up registered.

- **Fixed image** = the reference. It never moves.
- **Moving image** = the image that is shifted and bent to fit the fixed one.

The alignment happens in three automatic steps:

| Step | Name | What it does |
|---|---|---|
| 1 | **FFT** | Quick coarse guess of the shift (and a small rotation). |
| 2 | **Rigid / Affine** | Moves, turns and stretches the whole image (no local bending). |
| 3 | **B-Spline** | Bends the image locally so that individual cells line up. |

---

## What is in this folder?

| Item | Description |
|---|---|
| `mexalign` | The program (main version). |
| `mexalign_NP` | Variant for single-cell tables. Same program, plus it can convert a CSV of cell positions (see [NP version](#np-version-mexalign_np) below). |
| `mexalign.yml` | List of packages the program needs. |
| `run_mexalign.bat` | Starts the program on Windows. |
| `run_mexalign.command` | Starts the program on Mac. |
| `test_sample` | Example data you can try. |

> **Keep `mexalign`, `mexalign.yml`, `run_mexalign.bat` and `run_mexalign.command` together in ONE folder.**

The **first** start downloads the packages and takes a few minutes. After that it starts quickly.

---

## Install and start — Windows

1. **Install Miniconda** (one time only):
   <https://www.anaconda.com/download/success>
   → download **"Miniconda" for Windows** and run the installer.

2. **Double-click `run_mexalign.bat`**

   > If Windows shows a blue *"Windows protected your PC"* box, click **More info → Run anyway**.

---

## Install and start — Mac

1. **Install Miniconda** (one time only):
   <https://www.anaconda.com/download/success>
   → download **"Miniconda" for macOS**:
   - New Macs (M1/M2/M3/M4): choose **Apple Silicon**
   - Older Macs (Intel): choose **Intel**

   Run the downloaded `.pkg` installer.

2. **Allow the launcher to run** (one time only).
   Open Terminal (press <kbd>Cmd</kbd> + <kbd>Space</kbd>, type `Terminal`, press <kbd>Enter</kbd>), then run these **two** lines. For each: type the text, add **one** space, drag the item from Finder into Terminal, then press <kbd>Enter</kbd>.

   - **a. Remove the "downloaded from the internet" block**
     (*this* is what causes the "Apple could not verify" warning).
     Type `xattr -cr`, then drag the **whole folder** in and press Enter, e.g.:

     ```bash
     xattr -cr /path/to/MeXalign
     ```

   - **b. Make the launcher runnable.**
     Type `chmod +x`, then drag `run_mexalign.command` in and press Enter, e.g.:

     ```bash
     chmod +x /path/to/MeXalign/run_mexalign.command
     ```

3. **Double-click `run_mexalign.command`**

   > If macOS still says *"Apple could not verify … run_mexalign.command"*: click **Done** (**never** "Move to Trash"!), then open **System Settings → Privacy & Security**, scroll to the bottom, and click **Open Anyway**. After that, double-click works normally.

---

## How to use it

1. **Put all files of ONE experiment in a single folder:**
   - the fixed mask (e.g. `ICP_mask.png`)
   - the moving mask (e.g. `F_mask.png`)
   - the channel images you want to warp (`.tif` / `.tiff` / `.png`)

   Masks can be PNG or TIFF.

2. **Start MeXalign** (see above). The window has settings on the left and a live picture of the alignment on the right.

3. **Tab "Inputs":**
   - **Data folder:** click **Browse…** and choose your folder.
   - **Fixed mask (reference)** and **Moving mask:** pick the files.
   - **Fixed / Moving pixel size [µm/px]:** type the real pixel size of each image.
     > **Important:** this is how the program knows the two images have different resolutions.
   - **Gradient generation:** "Center-based" is the default. Try "Edge-based" only if the default does not work.
   - **Second mask (optional):** another label mask of the moving image that should be warped along (cell numbers are kept).
   - Tick the **channel images** that should be warped too.

4. **Quick settings** (top of the left panel) — pick one:

   | Preset | Description |
   |---|---|
   | **MSQ fast** | Fastest. Good for most data. *(Starts selected.)* |
   | **MSQ high-detail** | Same, with a finer mesh for more local detail. |
   | **Mattes** | Slowest, uses mutual information. |

   You normally do not need to touch the other tabs (FFT, Rigid/Affine, B-Spline); they only show the details behind these presets.

5. **Click "Run Alignment"** and watch the overlay on the right.
   **Stop** cancels the run. Do not close the window while it runs.

6. When it finishes, the **Results** box shows the scores (see [Results](#results)).

---

## Things you can change if needed

| Setting | Tab | What it does |
|---|---|---|
| **Rotation search** | FFT | On by default, tries turns of ±3°. If the images are turned more than that, rotate one of them by hand **before** you start. Larger turns may cut off image edges. |
| **Crop padding** | FFT | Extra border (in moving-image pixels) around the area that is used. The best value depends on your pixel size. |
| **Erosion** | FFT | Off by default. Shrinks cells a little to separate cells that touch. The radius is adapted automatically to the difference in resolution between the two images. |
| **B-Spline spacing** | B-Spline | Distance between the bending points in µm. Presets use 150 µm (MSQ) or 200 µm (Mattes). Smaller = more local detail, but slower and less stable. |

---

## Results

Scores range from **1.0 = perfect** to **0.0 = no overlap**. The program shows:

| Score | Meaning |
|---|---|
| **Silhouette-based score** | Overall overlap of the two masks. |
| **All-cell score** | Cell by cell; cells without a partner count as 0. |
| **1:1 matches only** | Only cells that have exactly one partner. |
| **1:1 and fused-cell groups** | **Headline score.** If several cells of one image sit on one cell of the other, they are merged and compared as one unit. |

Each is given as **Dice** and **IoU**. Higher is better.

**Folding check:** the program also checks whether the tissue was folded or torn by the bending. **"No tissue folding"** is what you want.

### Output files

All output is written to a subfolder **`Coregistration`** inside your data folder:

| File | Description |
|---|---|
| `registered_mask.tif` | The moving mask after alignment. |
| `mask_moving_..._coregistered_binarized.png` | Black/white version of it. |
| `<channel>_coregistered.tif` | Every ticked channel, aligned. |
| `Coregistration_parameters.txt` | All settings, scores, folding check and timings of the run. |
| `Coregistration_folded_mask.png` | Only if folding was found (white = folded pixels). |
| `..._gradient.tif` | The helper images used for the alignment. |

---

## NP version (`mexalign_NP`)

Works exactly like the main version, with one extra field on the **Inputs** tab: **"IF single-cell CSV (optional)"**.

Pick a CSV with one row per cell and the columns `Center_X` and `Center_Y` (cell centres in IF pixels). After the run you also get:

```
Coregistration_single_cell_registered.csv
```

in which the centres are moved into the coordinates of the fixed (ICP) image.

- The original positions stay in extra columns (`..._IF_px`).
- All other columns are kept.
- Cells that cannot be converted are left empty.
- Only the coordinates move; the images are not shrunk, so touching cells stay separate.

Start it with the files inside the `mexalign_NP` folder (same steps as above).

---

## Advanced: command line (no window)

Open a Terminal / Anaconda Prompt in this folder and activate the environment:

```bash
conda activate mexalign
```

Then run:

**Windows**

```bat
python -m mexalign.cli_full --directory "path/to/data" ^
       --fixed-mask-filename ICP_mask.png ^
       --moving-mask-filename F_mask.png
```

**Mac / Linux** (use `\` instead of `^` at the end of a line)

```bash
python -m mexalign.cli_full --directory "path/to/data" \
       --fixed-mask-filename ICP_mask.png \
       --moving-mask-filename F_mask.png
```

Settings can be changed with `--override name=value`.

The window version can also be started by hand with:

```bash
python -m mexalign.gui_full
```

---

## If something goes wrong

- **"Conda/Miniconda was not found"**
  Install Miniconda (step 1).

- **Windows: "DLL load failed while importing QtCore"**
  Install the Microsoft Visual C++ Redistributable (x64): <https://aka.ms/vs/17/release/vc_redist.x64.exe> and start again.

- **Mac: "Apple could not verify…" warning**
  That is the internet quarantine flag. Do step 2a (`xattr -cr` on the whole folder). A fresh download / AirDrop / email re-adds it every time, so each new copy needs step 2 once.

- **Mac: file opens in TextEdit instead of running**
  Do step 2b (`chmod +x` on `run_mexalign.command`) — the run permission gets lost when the file is copied.

- **The overlay does not fit at all**
  Check the two pixel sizes, and rotate one image by hand if the turn is bigger than a few degrees.

- **Old environment called "coreg"**
  The environment is now called `mexalign`. Rename it with:

  ```bash
  conda rename -n coreg mexalign
  ```

  or just let the launcher create a new one.

- **Large images** need plenty of memory (~16 GB RAM recommended).
