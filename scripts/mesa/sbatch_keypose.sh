#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --mem-per-gpu=120G
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:rtx_6000:1
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
export TORCH_HOME=$SCRATCH_BASE/torch

mkdir -p $TORCH_HOME/hub/checkpoints

export TMPDIR=$SCRATCH_BASE/tmp
export TEMP=$SCRATCH_BASE/tmp
export TMP=$SCRATCH_BASE/tmp
export UV_CACHE_DIR=$SCRATCH_BASE/uv-cache
export DGLBACKEND=pytorch
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_DIR=$SCRATCH_BASE/tmp/wandb/3dfa_mesa_keypose

mkdir -p $WANDB_DIR

cd /storage/home/hcoda1/5/fchang40/3d_flowmatch_actor

source /storage/project/r-agarg35-0/fchang40/venvs/imitation-venv/bin/activate

source scripts/mesa/train_mesa_keypose.sh
