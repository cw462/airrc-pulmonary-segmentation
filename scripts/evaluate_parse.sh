#!/usr/bin/env bash
set -euo pipefail

manifest="$(realpath "$1")"
model="$(realpath "$2")"
output="$(realpath -m "$3")"
shift 3
gpus=("$@")
if [ "${#gpus[@]}" -eq 0 ]; then
    gpus=(0)
fi
code_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${UMAMBA_ROOT:?Set UMAMBA_ROOT to the U-Mamba umamba directory}"
python_bin="${PYTHON:-python}"
mkdir -p "$output"
export nnUNet_raw="$output" nnUNet_preprocessed="$output" nnUNet_results="$output"
export PYTHONPATH="$UMAMBA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_compile=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_bin" -u "$code_root/evaluate_parse.py" --manifest "$manifest" --model "$model" --output "$output" --gpus "${gpus[@]}"

