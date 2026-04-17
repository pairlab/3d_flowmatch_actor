#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH -J 3dfa_build_single
#SBATCH -o 3dfa_build_single_%j.out
#SBATCH -e 3dfa_build_single_%j.err
#
# Single-task wrapper around sbatch_build_multitask.sh. Restricts TASKS to one
# task (default: apple_tray_on) and inherits the same MODE/SKIP_POSTPROCESS
# interface. Use sbatch_build_multitask.sh directly when building more than
# one task.
#
# Env-var overrides:
#   TASKS                 single task name                      (default: apple_tray_on)
#   MODE                  dense|keypose|both                    (default: keypose)
#   SKIP_POSTPROCESS      1 to skip fk postprocess              (default: 0)
#   SOURCE_INPUT_DIR      root of <task>/demo/demo.hdf5 sources (default: overfit tree)
#   DEST_DATA_ROOT        output root                           (default: 3dfa_data_fk_fix)
#   ACTION_HORIZON        future-action chunk size              (default: 1)

set -euo pipefail

export TASKS="${TASKS:-apple_tray_on}"
export MODE="${MODE:-keypose}"
export DEST_DATA_ROOT="${DEST_DATA_ROOT:-/storage/project/r-agarg35-0/fchang40/3dfa_data_fk_fix}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-/storage/home/hcoda1/5/fchang40/3d_flowmatch_actor/artifacts/mesa_build_single}"

source "$(dirname "${BASH_SOURCE[0]}")/sbatch_build_multitask.sh"
