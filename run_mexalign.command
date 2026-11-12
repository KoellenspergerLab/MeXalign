#!/bin/bash
# MeXalign GUI launcher (macOS) — double-click to run.
#
# On first run it auto-creates the conda env "mexalign" from mexalign.yml if no ready
# Python is found (Miniconda must be installed once), then launches the app.
# Subsequent runs just launch. To override detection, set PYTHON below, e.g.
#   PYTHON="$HOME/miniconda3/envs/mexalign/bin/python3"

PYTHON=""

cd "$(dirname "$0")" || exit 1

# 1) Use an existing Python that already has the packages (mexalign env first).
if [ -z "$PYTHON" ]; then
    for PY in \
        "$HOME/.conda/envs/mexalign/bin/python3" \
        "$HOME/miniconda3/envs/mexalign/bin/python3" \
        "$HOME/anaconda3/envs/mexalign/bin/python3" \
        "$HOME/miniforge3/envs/mexalign/bin/python3" \
        "$HOME/mambaforge/envs/mexalign/bin/python3" \
        "/opt/miniconda3/envs/mexalign/bin/python3" \
        "/opt/anaconda3/envs/mexalign/bin/python3" \
        "$HOME/miniconda3/bin/python3" \
        "$HOME/anaconda3/bin/python3" \
        "$HOME/miniforge3/bin/python3" \
        "$HOME/mambaforge/bin/python3" \
        "/opt/miniconda3/bin/python3" \
        "/opt/anaconda3/bin/python3" \
        python3 python; do
        if command -v "$PY" >/dev/null 2>&1 && \
           "$PY" -c "import PySide6, SimpleITK" >/dev/null 2>&1; then
            PYTHON="$PY"
            break
        fi
    done
fi

# 2) Nothing ready — create the "mexalign" env from mexalign.yml using conda.
if [ -z "$PYTHON" ]; then
    CONDA=""
    CONDA_BASE=""
    for BASE in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -x "$BASE/bin/conda" ]; then
            CONDA="$BASE/bin/conda"; CONDA_BASE="$BASE"; break
        fi
    done
    if [ -z "$CONDA" ] && command -v conda >/dev/null 2>&1; then
        CONDA="$(command -v conda)"
        CONDA_BASE="$("$CONDA" info --base 2>/dev/null)"
    fi

    if [ -z "$CONDA" ]; then
        echo "Conda/Miniconda was not found."
        echo "Install Miniconda once from:"
        echo "    https://www.anaconda.com/download/success"
        echo "then double-click this file again."
        echo
        read -r -p "Press Enter to close..."
        exit 1
    fi

    ENV_PY="$CONDA_BASE/envs/mexalign/bin/python3"
    # conda may put the env in the per-user store instead of the base.
    [ -x "$ENV_PY" ] || [ ! -x "$HOME/.conda/envs/mexalign/bin/python3" ] || ENV_PY="$HOME/.conda/envs/mexalign/bin/python3"
    if [ ! -x "$ENV_PY" ]; then
        echo "First-time setup: creating the 'mexalign' environment from mexalign.yml ..."
        echo "This downloads the packages and may take a few minutes."
        if ! "$CONDA" env create -f mexalign.yml; then
            echo "Environment creation failed — see the messages above."
            read -r -p "Press Enter to close..."
            exit 1
        fi
        # Re-resolve the interpreter: prefer the base env dir, fall back to
        # ~/.conda/envs, then ask conda directly (covers any envs_dirs).
        if [ -x "$CONDA_BASE/envs/mexalign/bin/python3" ]; then
            ENV_PY="$CONDA_BASE/envs/mexalign/bin/python3"
        elif [ -x "$HOME/.conda/envs/mexalign/bin/python3" ]; then
            ENV_PY="$HOME/.conda/envs/mexalign/bin/python3"
        else
            ENV_PY="$("$CONDA" run -n mexalign python -c 'import sys;print(sys.executable)' 2>/dev/null)"
        fi
    fi
    PYTHON="$ENV_PY"
fi

if [ -z "$PYTHON" ] || ! "$PYTHON" -c "import PySide6, SimpleITK" >/dev/null 2>&1; then
    echo "Could not find or set up a working Python. See messages above."
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "Launching with: $PYTHON"
exec "$PYTHON" -m mexalign.gui_full
