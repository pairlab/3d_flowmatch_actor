"""
SmolVLAPrefixKVActor — SmolVLM-backed denoising policy with prefix-KV caching.

Prefix pass (once per env step):
    [visual (3D RoPE) | language (1D RoPE) | state (3D RoPE)]
        ↓ SmolVLMEncoder.forward()
    prefix_kv : DynamicCache
    proprio_feats

Denoising loop (N steps):
    noisy_action_chunk → PrefixKVTransformerHead.encode_action_tokens
        ↓ SmolVLMPrefixKVBackbone.decode_action()  (only action tokens recompute)
        ↓ output head → (pos, rot, openess)

Full bidirectional attention throughout.
"""

import torch
from torch import nn

from ..encoder.multimodal.smolvlm_encoder import SmolVLMEncoder
from ..utils.position_encodings import SinusoidalPosEmb
from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


class PrefixKVTransformerHead(nn.Module):

    def __init__(self, hidden_size: int, embedding_dim: int, nhist: int, rot_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rot_dim = rot_dim

        action_in_dim = 3 + rot_dim
        self.action_proj = nn.Linear(action_in_dim, hidden_size)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.gripper_proj = nn.Linear(embedding_dim * nhist, hidden_size)

        self.step_pos_emb = SinusoidalPosEmb(hidden_size)
        self.step_pos_proj = nn.Linear(hidden_size, hidden_size)

        action_out_dim = 3 + rot_dim + 1  # pos + rot + openess
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_out_dim),
        )

    def encode_action_tokens(self, trajectory, timestep, proprio_feats):
        """
        trajectory    : (B, T, 3 + rot_dim) noisy pos+rot
        timestep      : (B,) integer
        proprio_feats : (B, nhist, embedding_dim)
        returns       : (B, T, hidden_size)
        """
        B, T, _ = trajectory.shape
        device = trajectory.device

        action_emb = self.action_proj(trajectory)
        action_emb = action_emb + self.time_emb(timestep.float()).unsqueeze(1)
        action_emb = action_emb + self.gripper_proj(proprio_feats.reshape(B, -1)).unsqueeze(1)

        step_ids = torch.arange(T, device=device, dtype=torch.float32)
        step_emb = self.step_pos_proj(self.step_pos_emb(step_ids))
        return action_emb + step_emb.unsqueeze(0)

    def predict(self, action_hidden):
        return self.output_head(action_hidden)


class SmolVLAPrefixKVActor(BaseDenoiseActor):
    """
    Inherits ``compute_loss`` / ``compute_trajectory`` / ``conditional_sample``
    from ``BaseDenoiseActor``. Overrides ``encode_inputs`` and
    ``policy_forward_pass`` to route through SmolVLM with a prefix-KV cache.
    """

    def __init__(
        self,
        # SmolVLM / encoder kwargs
        smolvlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        smolvlm_local_files_only: bool = False,
        smolvlm_freeze_vision_tower: bool = True,
        smolvlm_freeze_connector: bool = True,
        smolvlm_freeze_text_model: bool = True,
        smolvlm_freeze_text_embeddings: bool = True,
        smolvlm_tokenizer_max_length: int = 48,
        smolvlm_append_state_tokens: bool = False,
        smolvlm_state_dim: int = 0,
        smolvlm_num_state_tokens: int = 1,
        smolvlm_lora_rank: int = 0,
        smolvlm_lora_alpha: float = 0.0,
        smolvlm_lora_last_n_layers: int = 0,
        smolvlm_xyz_rope_scale: float = 1.0,
        # Shared architecture kwargs (mirrored from BaseDenoiseActor)
        embedding_dim: int = 60,
        num_attn_heads: int = 9,
        nhist: int = 3,
        nhand: int = 1,
        rotation_format: str = "quat_xyzw",
        # Denoising kwargs
        denoise_timesteps: int = 100,
        denoise_model: str = "rectified_flow",
        lv2_batch_size: int = 1,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size,
            build_default_action_head=False,
        )

        rot_dim = 3 if rotation_format == "euler" else 6

        self.encoder = SmolVLMEncoder(
            model_name=smolvlm_model_name,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            tokenizer_max_length=smolvlm_tokenizer_max_length,
            freeze_vision_tower=smolvlm_freeze_vision_tower,
            freeze_connector=smolvlm_freeze_connector,
            freeze_text_model=smolvlm_freeze_text_model,
            freeze_text_embeddings=smolvlm_freeze_text_embeddings,
            local_files_only=smolvlm_local_files_only,
            append_state_tokens=smolvlm_append_state_tokens,
            state_dim=smolvlm_state_dim,
            num_state_tokens=smolvlm_num_state_tokens,
            lora_rank=smolvlm_lora_rank,
            lora_alpha=smolvlm_lora_alpha,
            lora_last_n_layers=smolvlm_lora_last_n_layers,
            xyz_rope_scale=smolvlm_xyz_rope_scale,
        )

        self.prediction_head = PrefixKVTransformerHead(
            hidden_size=self.encoder.smolvlm_hidden_size,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            rot_dim=rot_dim,
        )

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        proprio_feats, prefix_kv, n_prefix, prefix_pad_mask = self.encoder(
            rgb3d, pcd, instruction, proprio
        )
        return proprio_feats, prefix_kv, n_prefix, prefix_pad_mask

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        proprio_feats, prefix_kv, n_prefix, prefix_pad_mask = fixed_inputs

        B, traj_len, nhand, _ = trajectory.shape
        traj_flat = trajectory.flatten(1, 2)  # (B, T*nhand, 3+rot_dim)

        action_emb = self.prediction_head.encode_action_tokens(
            trajectory=traj_flat,
            timestep=timestep,
            proprio_feats=proprio_feats,
        )
        action_hidden = self.encoder.prefix_kv_backbone.decode_action(
            action_embeds=action_emb,
            prefix_kv=prefix_kv,
            n_prefix=n_prefix,
            prefix_pad_mask=prefix_pad_mask,
        )
        pred = self.prediction_head.predict(action_hidden)
        pred = pred.unflatten(1, (traj_len, nhand))
        return [pred]
