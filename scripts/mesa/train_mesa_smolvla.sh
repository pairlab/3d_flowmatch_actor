#!/bin/bash
# Train SmolVLA prefix-KV actor on Mesa bimanual dense data.
# Usage: bash scripts/mesa/train_mesa_smolvla.sh

main_dir=MesaBimanual

DATA_PATH=""

train_data_dir=$DATA_PATH/dense/train.zarr
eval_data_dir=$DATA_PATH/dense/val.zarr
train_instructions=instructions/mesa/instructions.json
val_instructions=instructions/mesa/instructions.json

dataset=MesaBimanual
num_workers=2
B=8          # smaller than clip run — SmolVLM uses more VRAM
B_val=8
chunk_size=1
memory_limit=2

# Training
val_freq=500
eval_only=false
lr=1e-4
backbone_lr=1e-6   # unused for frozen SmolVLM, kept for compat
lr_scheduler=constant
wd=1e-10
train_iters=50000
use_compile=false
use_ema=false
lv2_batch_size=1

# ---- Model ----
model_type=smolvla_prefix_kv
bimanual=true
keypose_only=false
pre_tokenize=false   # SmolVLA tokenises internally, but check this to make sure - test
custom_img_size=128
workspace_normalizer_buffer=0.04

# SmolVLA-specific
smolvlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
smolvlm_local_files_only=true
smolvlm_freeze_vision_tower=true
smolvlm_freeze_connector=true
smolvlm_freeze_text_model=true
smolvlm_freeze_text_embeddings=true
smolvlm_tokenizer_max_length=48
smolvlm_append_state_tokens=false
smolvlm_state_dim=0
smolvlm_num_state_tokens=1

# Shared architecture (embedding_dim must be divisible by num_attn_heads)
C=120
num_attn_heads=8
num_vis_instr_attn_layers=3   # unused by SmolVLA, kept for compat
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

# Logging
base_log_dir=""
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-dense
checkpoint=${base_log_dir}/${main_dir}/${run_log_dir}/last.pth

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --base_log_dir $base_log_dir \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --custom_img_size $custom_img_size \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --fps_subsampling_factor 5 \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --smolvlm_model_name $smolvlm_model_name \
    --smolvlm_local_files_only $smolvlm_local_files_only \
    --smolvlm_freeze_vision_tower $smolvlm_freeze_vision_tower \
    --smolvlm_freeze_connector $smolvlm_freeze_connector \
    --smolvlm_freeze_text_model $smolvlm_freeze_text_model \
    --smolvlm_freeze_text_embeddings $smolvlm_freeze_text_embeddings \
    --smolvlm_tokenizer_max_length $smolvlm_tokenizer_max_length \
    --smolvlm_append_state_tokens $smolvlm_append_state_tokens \
    --smolvlm_state_dim $smolvlm_state_dim \
    --smolvlm_num_state_tokens $smolvlm_num_state_tokens \
    --use_wandb true \
    --wandb_project 3dfa_bimanual \
    --pace_copy true \
    --pace_tmp_dir /tmp
