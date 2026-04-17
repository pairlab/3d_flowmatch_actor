#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH -J 3dfa_build_multitask
#SBATCH -o 3dfa_build_multitask_%j.out
#SBATCH -e 3dfa_build_multitask_%j.err
#
# Build dense/keypose/both zarrs for a multi-task mesa-bimanual subset, with
# optional FK-action-fix postprocessing.
#
# Pipeline:
#   1. (Optional) Batch-postprocess every <task>/demo/demo.hdf5 under the
#      source tree with --action-fix-mode fk (SKIP_POSTPROCESS=1 to skip when
#      the source tree is already fk-repaired in place, e.g. the 2-task
#      depth-fix build).
#   2. Stage a task-filtered view (via symlinks or the postprocess output).
#   3. Run data_processing/mesa_to_zarr.py against the staged tree, writing
#      zarrs to /tmp (avoids the multi-million-file NFS metadata cost called
#      out in markdowns/mesa-to-zarr-howto.md).
#   4. Parallel cp the merged store to the final NFS destination, then
#      symlink val.zarr -> train.zarr (overfit layout).
#
# Env-var overrides:
#   MODE                  dense|keypose|both   (default: keypose)
#   TASKS                 space-separated task names to include
#                         (default: every task under SOURCE_INPUT_DIR)
#   SKIP_POSTPROCESS      1 to skip fk postprocess                (default: 0)
#   SOURCE_INPUT_DIR      root of <task>/demo/demo.hdf5 sources   (default: overfit tree)
#   DEST_DATA_ROOT        output root (default: 3dfa_data_fk_fix/multitask_v2)
#   ARTIFACT_DIR          fix report destination                  (default: artifacts/...)
#   ACTION_HORIZON        future-action chunk size                (default: 1)
#   POSTPROCESS_WORKERS   parallelism for postprocess             (default: SLURM cpu count)

set -euo pipefail

REPO_DIR="/storage/home/hcoda1/5/fchang40/3d_flowmatch_actor"
cd "$REPO_DIR"

source /storage/project/r-agarg35-0/fchang40/venvs/imitation-venv/bin/activate
source scripts/mesa/horizon_env.sh

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch/"
mkdir -p "$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export TEMP="$SCRATCH_BASE/tmp"
export TMP="$SCRATCH_BASE/tmp"
export HDF5_USE_FILE_LOCKING=FALSE
export MUJOCO_GL=egl

MODE="${MODE:-keypose}"
case "$MODE" in
    dense|keypose|both) ;;
    *) echo "Unsupported MODE='$MODE'. Must be dense, keypose, or both." >&2; exit 2 ;;
esac

SKIP_POSTPROCESS="${SKIP_POSTPROCESS:-0}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/storage/project/r-agarg35-0/shared/vla_benchmark_data/mesa_bimanual/overfit}"
DEST_DATA_ROOT_BASE="${DEST_DATA_ROOT:-/storage/project/r-agarg35-0/fchang40/3dfa_data_fk_fix/multitask_v2}"
DEST_DATA_ROOT="$(mesa_data_path_for_horizon "$DEST_DATA_ROOT_BASE")"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_DIR/artifacts/mesa_build_multitask}"

STAGING_ROOT="/tmp/3dfa_build_multitask_${SLURM_JOB_ID:-local}"
STAGING_INPUT_DIR="$STAGING_ROOT/input"
STAGING_ZARR_OUT_DIR="$STAGING_ROOT/zarr_out"

if [ ! -d "$SOURCE_INPUT_DIR" ]; then
    echo "Source input dir not found: $SOURCE_INPUT_DIR" >&2
    exit 2
fi

rm -rf "$STAGING_ROOT"
mkdir -p "$STAGING_INPUT_DIR" "$STAGING_ZARR_OUT_DIR" "$DEST_DATA_ROOT" "$ARTIFACT_DIR"

# Resolve task list.
if [ -n "${TASKS:-}" ]; then
    read -r -a TASK_ARRAY <<< "$TASKS"
else
    TASK_ARRAY=()
    for task_dir in "$SOURCE_INPUT_DIR"/*/demo/demo.hdf5; do
        task_name="$(basename "$(dirname "$(dirname "$task_dir")")")"
        TASK_ARRAY+=("$task_name")
    done
fi
if [ "${#TASK_ARRAY[@]}" -eq 0 ]; then
    echo "No tasks found under $SOURCE_INPUT_DIR" >&2
    exit 2
fi
echo "Tasks: ${TASK_ARRAY[*]}"

if [ "$SKIP_POSTPROCESS" = "1" ]; then
    echo "[1/3] Postprocess skipped; symlinking sources into staging input"
    for task in "${TASK_ARRAY[@]}"; do
        src="$SOURCE_INPUT_DIR/$task"
        if [ ! -f "$src/demo/demo.hdf5" ]; then
            echo "Missing source HDF5: $src/demo/demo.hdf5" >&2
            exit 2
        fi
        ln -s "$src" "$STAGING_INPUT_DIR/$task"
    done
else
    POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
    FILTERED_INPUT="$STAGING_ROOT/postprocess_input"
    mkdir -p "$FILTERED_INPUT"
    for task in "${TASK_ARRAY[@]}"; do
        src="$SOURCE_INPUT_DIR/$task"
        if [ ! -f "$src/demo/demo.hdf5" ]; then
            echo "Missing source HDF5: $src/demo/demo.hdf5" >&2
            exit 2
        fi
        ln -s "$src" "$FILTERED_INPUT/$task"
    done

    echo "[1/3] Batch-postprocessing $FILTERED_INPUT with --action-fix-mode fk (workers=$POSTPROCESS_WORKERS)"
    python -u scripts/postprocess_mesa_overfit_mirror_data.py \
        --input-dir "$FILTERED_INPUT" \
        --output-dir "$STAGING_INPUT_DIR" \
        --action-fix-mode fk \
        --max-workers "$POSTPROCESS_WORKERS" \
        --no-render-video \
        --overwrite

    echo "[1/3] Copying fix reports to $ARTIFACT_DIR"
    if [ -f "$STAGING_INPUT_DIR/aggregate_report.json" ]; then
        cp "$STAGING_INPUT_DIR/aggregate_report.json" "$ARTIFACT_DIR/aggregate_report.json"
    fi
    mkdir -p "$ARTIFACT_DIR/per_task"
    for task_dir in "$STAGING_INPUT_DIR"/*/demo; do
        task_name="$(basename "$(dirname "$task_dir")")"
        if [ -f "$task_dir/fix_report.json" ]; then
            cp "$task_dir/fix_report.json" "$ARTIFACT_DIR/per_task/${task_name}.json"
        fi
    done
fi

echo "[2/3] Building $MODE zarr(s) at $STAGING_ZARR_OUT_DIR (ACTION_HORIZON=$ACTION_HORIZON)"
python data_processing/mesa_to_zarr.py \
    --input_dir "$STAGING_INPUT_DIR" \
    --output_dir "$STAGING_ZARR_OUT_DIR" \
    --mode "$MODE" \
    --motion_threshold 0.02 \
    --action_horizon "$ACTION_HORIZON"

if [ "$MODE" = "both" ]; then
    COPY_MODES=(dense keypose)
else
    COPY_MODES=("$MODE")
fi

echo "[3/3] Parallel cp staged zarr(s) -> $DEST_DATA_ROOT"
for mode in "${COPY_MODES[@]}"; do
    staged="$STAGING_ZARR_OUT_DIR/$mode/train.zarr"
    dest_mode_dir="$DEST_DATA_ROOT/$mode"
    dest_train="$dest_mode_dir/train.zarr"
    if [ ! -d "$staged" ]; then
        echo "Expected staged zarr not found: $staged" >&2
        exit 3
    fi
    mkdir -p "$dest_mode_dir"
    if [ -e "$dest_train" ]; then
        rm -rf "$dest_train"
    fi
    mkdir -p "$dest_train"
    cp "$staged/.zgroup" "$dest_train/"
    ls "$staged" | grep -v '^\.zgroup$' \
        | xargs -I{} -P 8 cp -r "$staged/{}" "$dest_train/{}"

    cd "$dest_mode_dir"
    if [ -e val.zarr ] && [ ! -L val.zarr ]; then
        rm -rf val.zarr
    fi
    ln -sfn train.zarr val.zarr
    cd "$REPO_DIR"
done

rm -rf "$STAGING_ROOT"

echo
for mode in "${COPY_MODES[@]}"; do
    echo "=== $mode zarr summary ==="
    python -c "
import zarr, numpy as np
z = zarr.open_group('$DEST_DATA_ROOT/$mode/train.zarr', 'r')
for k in sorted(z.keys()):
    print(f'  {k}: shape={z[k].shape} dtype={z[k].dtype}')
print('  total samples:', z['action'].shape[0])
tids = z['task_id'][:]
ids, counts = np.unique(tids, return_counts=True)
print('  per-task counts:', dict(zip(ids.tolist(), counts.tolist())))
"
    echo
done

echo "Done. Output root: $DEST_DATA_ROOT"
