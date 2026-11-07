#!/usr/bin/env bash
set -euo pipefail

run_root="$(realpath "$1")"
shift
code_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${UMAMBA_ROOT:?Set UMAMBA_ROOT to the U-Mamba umamba directory}"
python_bin="${PYTHON:-python}"
gpus=("$@")
if [ "${#gpus[@]}" -eq 0 ]; then gpus=(0); fi
export nnUNet_raw="$run_root/full/raw" nnUNet_preprocessed="$run_root/full/preprocessed" nnUNet_results="$run_root/full/results"
export PYTHONPATH="$code_root:$UMAMBA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_compile=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_bin" -u -m evaluation.export_airrc_predictions \
    --cohort "$code_root/configs/airrc_av_cohort.json" \
    --split-file "$run_root/evaluation/holdout/patient_splits.json" \
    --source-val "$run_root/full/raw/Dataset926_AirRC" --source-test "$run_root/evaluation/holdout" \
    --model "$run_root/full/results/Dataset926_AirRC/AirRCTrainer__nnUNetPlans__3d_fullres" \
    --evaluation-summary "$run_root/evaluation/results/summary.json" \
    --previous-per-case "$run_root/evaluation/results/per_case.csv" \
    --output "$run_root/evaluation/results/predictions" \
    --records "$run_root/evaluation/results/prediction_export/cases" --gpus "${gpus[@]}"
