#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem-per-gpu=120G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH -J 3dfa_mesa_eval_keypose
#SBATCH -o 3dfa_mesa_eval_keypose_%j.out
#SBATCH -e 3dfa_mesa_eval_keypose_%j.err
#
# Eval for a mesa-bimanual keypose checkpoint. Variant is selected via env
# vars; defaults target the 17-task FK-fix multitask keypose run.
#
# Usage:
#   TASK_FILTER=candle_tray_on sbatch scripts/mesa/sbatch_eval_keypose.sh
#
# Env-var overrides:
#   CHECKPOINT_NAME   "best" (default) or "last"
#   CHECKPOINT_ROOT   checkpoint directory (default: multitask v2 FK-fix)
#   DATASET           dataset class (default: MesaBimanualMultiTask)
#   TASK_FILTER       task to roll out (REQUIRED)
#   VARIANT_NAME      output variant subdir (default derived from TASK_FILTER + CHECKPOINT_NAME)
#   NUM_ROLLOUTS      rollouts per task (default: 4)
#   EVAL_SET_NAME     vla-benchmark suite (default: mesa_bimanual)
#   EVAL_SPLIT        suite split (default: overfit)
#   ACTION_HORIZON    convenience override for both CHUNK_SIZE and REPLAN_STEPS
#   CHUNK_SIZE        actions returned per inference call (default: 1)
#   REPLAN_STEPS      actions executed before replanning (default: $CHUNK_SIZE)
#   MAX_STEPS         per-episode step cap (default: eval client default of 500)

set -euo pipefail

if [ -z "${TASK_FILTER:-}" ]; then
    echo "TASK_FILTER is required. Set it to one of MESA_TASKS (e.g. apple_tray_on)." >&2
    exit 2
fi

DATASET="${DATASET:-MesaBimanualMultiTask}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/storage/project/r-agarg35-0/fchang40/3dfa_checkpoints/MesaBimanualMultiTask/denoise3d-MesaBimanualMultiTask-C120-B16-lr1e-4-constant-H3-rectified_flow-keypose-multitask-v2-fk-fixed}"
CHECKPOINT="$CHECKPOINT_ROOT/${CHECKPOINT_NAME}.pth"
VARIANT_NAME="${VARIANT_NAME:-keypose-${TASK_FILTER}-${CHECKPOINT_NAME}}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
EVAL_SET_NAME="${EVAL_SET_NAME:-mesa_bimanual}"
EVAL_SPLIT="${EVAL_SPLIT:-overfit}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-$ACTION_HORIZON}"
REPLAN_STEPS="${REPLAN_STEPS:-$CHUNK_SIZE}"
MAX_STEPS="${MAX_STEPS:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Train it first with scripts/mesa/sbatch_keypose.sh." >&2
    exit 2
fi

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch/"
mkdir -p "$SCRATCH_BASE/hf" "$SCRATCH_BASE/tmp"

export HF_HOME="$SCRATCH_BASE/hf"
export HF_DATASETS_CACHE="$SCRATCH_BASE/hf/datasets"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$SCRATCH_BASE/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH_BASE/torch"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$SCRATCH_BASE/tmp"
export TMP="$SCRATCH_BASE/tmp"
export UV_CACHE_DIR="$SCRATCH_BASE/uv-cache"
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$TORCH_HOME/hub/checkpoints"

REPO_DIR="/storage/home/hcoda1/5/fchang40/3d_flowmatch_actor"
cd "$REPO_DIR"

source /storage/project/r-agarg35-0/fchang40/venvs/imitation-venv/bin/activate

EXTRA_ARGS=()
if [ -n "$MAX_STEPS" ]; then
    EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
fi

python scripts/mesa/launch_eval.py \
    --checkpoint "$CHECKPOINT" \
    --variant-name "$VARIANT_NAME" \
    --eval-set-name "$EVAL_SET_NAME" \
    --eval-split "$EVAL_SPLIT" \
    --task-filter "$TASK_FILTER" \
    --num-rollouts-per-task "$NUM_ROLLOUTS" \
    --num-env-workers 2 \
    --dataset "$DATASET" \
    "${EXTRA_ARGS[@]}" \
    --bimanual True \
    --num-history 3 \
    --embedding-dim 120 \
    --num-attn-heads 8 \
    --num-vis-instr-attn-layers 3 \
    --num-shared-attn-layers 4 \
    --rotation-format quat_xyzw \
    --denoise-timesteps 5 \
    --denoise-model rectified_flow \
    --backbone clip \
    --custom-img-size 128 \
    --chunk-size "$CHUNK_SIZE" \
    --replan-steps "$REPLAN_STEPS" \
    --controller-type osc_pose \
    --depth-transport raw \
    --camera-names egocentric robot0_eye_in_hand robot1_eye_in_hand \
    --robots ReverseMountedYam ReverseMountedYam \
    --camera-height 128 --camera-width 128
