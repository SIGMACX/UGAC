#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/path/to/MMWHS}
OUTPUT_DIR=${2:-outputs/mmwhs}
GPUS=${GPUS:-1}

if [ "$GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$GPUS" -m mmwhs_fns.train_mmwhs \
    --config configs/mmwhs.yaml \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"
else
  python -m mmwhs_fns.train_mmwhs \
    --config configs/mmwhs.yaml \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"
fi
