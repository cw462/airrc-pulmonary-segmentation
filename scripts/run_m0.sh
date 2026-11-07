#!/usr/bin/env bash
set -euo pipefail

run_root="$(realpath -m "$1")"
shift
code_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${UMAMBA_ROOT:?Set UMAMBA_ROOT to the U-Mamba umamba directory}"
python_bin="${PYTHON:-python}"
work="$run_root/full"
mkdir -p "$work/results" "$work/tmp"
status() { printf '%s\n' "$1" > "$work/state.txt.part"; mv "$work/state.txt.part" "$work/state.txt"; }
trap 'if [ "$?" -ne 0 ]; then status failed; fi' EXIT
export nnUNet_raw="$work/raw" nnUNet_preprocessed="$work/preprocessed" nnUNet_results="$work/results"
export PYTHONPATH="$UMAMBA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_n_proc_DA=4 nnUNet_compile=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$work/tmp" MPLCONFIGDIR="$work/tmp/matplotlib"
status running
"$python_bin" -u "$code_root/train_airrc.py" full "$@"
status done

