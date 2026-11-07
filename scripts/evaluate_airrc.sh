#!/usr/bin/env bash
set -euo pipefail

run_root="$(realpath "$1")"
code_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${UMAMBA_ROOT:?Set UMAMBA_ROOT to the U-Mamba umamba directory}"
python_bin="${PYTHON:-python}"
raw="$run_root/full/raw/Dataset926_AirRC"
holdout="$run_root/evaluation/holdout"
mkdir -p "$holdout"
for kind in imagesTs labelsTs; do
    test -d "$raw/$kind"
    if [ ! -e "$holdout/$kind" ]; then
        ln -s "$raw/$kind" "$holdout/$kind"
    fi
done
cp "$raw/patient_splits.json" "$holdout/patient_splits.json"
export nnUNet_raw="$run_root/full/raw" nnUNet_preprocessed="$run_root/full/preprocessed" nnUNet_results="$run_root/full/results"
export PYTHONPATH="$UMAMBA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_compile=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_bin" -u "$code_root/evaluate_airrc.py" "$run_root"

