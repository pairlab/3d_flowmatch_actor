#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem-per-gpu=120G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH -J 3dfa_mesa_keypose
#SBATCH -o 3dfa_mesa_keypose_%j.out
#SBATCH -e 3dfa_mesa_keypose_%j.err

export SCRATCH_BASE="/storage/home/hcoda1/5/fchang40/scratch/"

mkdir -p $SCRATCH_BASE/hf $SCRATCH_BASE/tmp

export HF_HOME=$SCRATCH_BASE/hf
export HF_DATASETS_CACHE=$SCRATCH_BASE/hf/datasets
export TRANSFORMERS_CACHE=$SCRATCH_BASE/hf/transformers
export HUGGINGFACE_HUB_CACHE=$SCRATCH_BASE/hf/hub
# Force offline mode so from_pretrained() reads cache only and never makes
# network calls. The compute nodes have flaky outbound HTTPS to huggingface.co
# (10s read timeout killed job 6324540 mid-CLIP-load). Cache is fully populated
# at $TRANSFORMERS_CACHE for openai/clip-vit-base-patch32.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=$SCRATCH_BASE/torch

mkdir -p $TORCH_HOME/hub/checkpoints

export TMPDIR=$SCRATCH_BASE/tmp
export TEMP=$SCRATCH_BASE/tmp
export TMP=$SCRATCH_BASE/tmp
export UV_CACHE_DIR=$SCRATCH_BASE/uv-cache
export DGLBACKEND=pytorch
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_DIR=$SCRATCH_BASE/tmp/wandb/3dfa_mesa_keypose
# Redirect wandb's per-user data/cache off $HOME. Default
# get_staging_dir() lives under ~/.local/share/wandb/artifacts/staging/
# (WANDB_DATA_DIR), and $HOME quota filling killed job 6343733: the wandb
# SenderThread crashed with ENOSPC, then the trainer hung on the next
# wandb.log() until SLURM hit the 8h time limit.
export WANDB_DATA_DIR=$SCRATCH_BASE/wandb-data
export WANDB_CACHE_DIR=$SCRATCH_BASE/wandb-cache

mkdir -p $WANDB_DIR $WANDB_DATA_DIR/artifacts/staging $WANDB_CACHE_DIR

cd /storage/home/hcoda1/5/fchang40/3d_flowmatch_actor

source /storage/project/r-agarg35-0/fchang40/venvs/imitation-venv/bin/activate

source scripts/mesa/train_mesa_keypose.sh
