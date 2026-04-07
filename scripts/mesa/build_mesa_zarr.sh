#!/bin/bash

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/3d_flowmatch_actor"
cd "$REPO_DIR"

source scripts/mesa/horizon_env.sh

INPUT_DIR="${INPUT_DIR:-/storage/project/r-agarg35-0/shared/vla_benchmark_data/mesa_bimanual/overfit}"
MODE="${MODE:-both}"
MOTION_THRESHOLD="${MOTION_THRESHOLD:-0.02}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/storage/project/r-agarg35-0/fchang40/3dfa_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(mesa_data_path_for_horizon "$BASE_OUTPUT_DIR")}"
PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/fchang40/venvs/imitation-venv/bin/python}"

cmd=(
    "$PYTHON_BIN"
    data_processing/mesa_to_zarr.py
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --mode "$MODE"
    --motion_threshold "$MOTION_THRESHOLD"
    --action_horizon "$ACTION_HORIZON"
)

echo "Building Mesa zarr with ACTION_HORIZON=$ACTION_HORIZON"
echo "  INPUT_DIR=$INPUT_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODE=$MODE"

"${cmd[@]}"
