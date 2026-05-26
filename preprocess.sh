# Run all preprocessing notebooks sequentially
set -euo pipefail

PATH_TO_NOTEBOOKS="data/preprocessing"

NOTEBOOKS=(
    "01_missingness_diagnosis.ipynb"
    "02_imputation_pipeline.ipynb"
    "03_data_engineer.ipynb"
)

echo "Starting preprocessing pipeline..."
echo

for nb in "${NOTEBOOKS[@]}"; do
    FULL_PATH="${PATH_TO_NOTEBOOKS}/${nb}"

    # Verify notebook exists
    if [[ ! -f "$FULL_PATH" ]]; then
        echo "ERROR: Notebook not found -> $FULL_PATH"
        
    fi
    else
        echo "Found notebook: $FULL_PATH"

        echo "Running: $nb"

        jupyter nbconvert \
            --to notebook \
            --execute "$FULL_PATH" \
            --inplace

        echo "Finished: $nb"
        echo
    fi
done

echo "All notebooks executed successfully."