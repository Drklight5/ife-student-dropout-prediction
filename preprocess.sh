#!/usr/bin/env bash
# Run all preprocessing notebooks sequentially.
# Usage: bash preprocess.sh  (from any directory — script resolves its own path)
set -euo pipefail

# ── FIX 5: cd to project root so relative paths always work ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Use the project venv's Python directly ───────────────────────────────────
PYTHON=".venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    echo "       Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Ensure nbconvert is installed (installs silently if missing)
if ! "$PYTHON" -c "import nbconvert" 2>/dev/null; then
    echo "nbconvert not found — installing..."
    "$PYTHON" -m pip install --quiet nbconvert nbformat
fi

PATH_TO_NOTEBOOKS="data/preprocessing"

NOTEBOOKS=(
    "01_missingness_diagnosis.ipynb"
    "02_imputation_pipeline.ipynb"
    "03_data_engineer.ipynb"
)

echo "Starting preprocessing pipeline..."
echo "Project root : $SCRIPT_DIR"
echo "Python       : $PYTHON"
echo

for nb in "${NOTEBOOKS[@]}"; do
    FULL_PATH="${PATH_TO_NOTEBOOKS}/${nb}"

    # ── FIX 1 & 2: correct if/fi structure + exit on missing notebook ────────
    if [[ ! -f "$FULL_PATH" ]]; then
        echo "ERROR: Notebook not found -> $FULL_PATH"
        exit 1
    fi

    echo "▶  Running : $nb"

    # python -m nbconvert bypasses jupyter dispatch — works without jupyter-nbconvert entry point
    # 10-min timeout covers MICE + RandomForest in 02_imputation_pipeline.ipynb
    "$PYTHON" -m nbconvert \
        --to notebook \
        --execute "$FULL_PATH" \
        --inplace \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3

    echo "✓  Finished: $nb"
    echo
done

echo "All notebooks executed successfully."