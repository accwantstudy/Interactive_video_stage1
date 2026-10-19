#!/usr/bin/env bash
# Run the A14B streaming example on one Slurm-allocated GPU.
# Usage: bash scripts/wan2.2_fun/test_streaming_srun.sh [inference arguments]
# Default resources: srun --gres=gpu:1 -c 32 --time 1-00:00
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/data/pdai/anaconda3/envs/VideoX_Fun/bin/python"
NUM_CPUS="${NUM_CPUS:-32}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00}"

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "VideoX_Fun Python not found: $PYTHON_BIN" >&2
    exit 1
fi

# Use the selected conda environment for Python and its subprocesses.
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
# srun sets CUDA_VISIBLE_DEVICES for the allocated GPU; do not override it.
exec srun --gres=gpu:1 -c "$NUM_CPUS" --time "$TIME_LIMIT" \
    "$PYTHON_BIN" -u examples/wan2.2_fun/predict_v2v_control_ref_streaming.py \
    --chunk-frames 32 \
    --overlap-frames 4 \
    --sample-size 1280 704\
    --num-inference-steps 50 \
    --fps 24 \
    --control-video asset/dance.mp4 \
    --ref-image asset/dance.png \
    --save-path samples/wan-videos-fun-control-a14b-streaming \
    "$@"
