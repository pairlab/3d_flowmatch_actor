"""
SmolVLAPrefixKVActor
=====================
3D-aware SmolVLA variant that lives in the standalone 3d_flowmatch_actor repo.

Architecture
------------
Prefix pass (once per env step)
  [visual (3D RoPE) | language (1D RoPE) | state (3D RoPE)]
      ↓ SmolVLMEncoder.forward()
  prefix_kv : DynamicCache  (layer-wise K/V, frozen after this)
  visual_proj, cond_proj, proprio_feats, fps_feats  (plain features)

Denoising loop (N steps)
  noisy_action_chunk  →  PrefixKVTransformerHead
      ↓  project to hidden_size + timestep + gripper conditioning
      ↓  SmolVLMPrefixKVBackbone.decode_action()  [only action tokens computed]
      ↓  project → (pos=3, rot=6, openess=1)
  return [pred]                (list, matching base-class convention)

Full (bidirectional) attention throughout — no causal masking.
"""

import einops
import torch
from torch import nn

from ..utils.position_encodings import SinusoidalPosEmb
from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from ..encoder.multimodal.smolvlm_encoder import SmolVLMEncoder


# ---------------------------------------------------------------------------
# Action head
# ---------------------------------------------------------------------------

class PrefixKVTransformerHead(nn.Module):
    """
    Encodes the noisy trajectory + conditioning, decodes with prefix K/V,
    and predicts (pos, rot, openess).

    Parameters
    ----------
    hidden_size   : SmolVLM hidden dimension
    embedding_dim : encoder embedding dim (used only for gripper feature shape)
    nhist         : gripper/proprio history length
    rot_dim       : rotation output dim (6 for 6D, 3 for Euler)
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int,
        nhist: int = 3,
        rot_dim: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rot_dim = rot_dim
        action_in_dim = 3 + rot_dim  # pos + rot

        # 1. Project noisy action → hidden_size
        self.action_proj = nn.Linear(action_in_dim, hidden_size)

        # 2. Diffusion timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 3. Gripper/proprio conditioning  (nhist × embedding_dim → hidden_size)
        self.gripper_proj = nn.Linear(embedding_dim * nhist, hidden_size)

        # 4. Learnable per-step position offset
        # We don't know chunk_size at init; use a large enough default and slice.
        self.max_chunk = 128
        self.traj_pos_emb = nn.Embedding(self.max_chunk, hidden_size)

        # 5. Output head: LayerNorm → hidden → action_dim
        action_out_dim = 3 + rot_dim + 1  # pos + rot + openess
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_out_dim),
        )

    def encode_action_tokens(self, trajectory, timestep, proprio_feats):
        """
        Build action token embeddings.

        Args:
            trajectory   : (B, T, action_in_dim)  noisy pos+rot
            timestep     : (B,)                    continuous t in [0,1] or integer
            proprio_feats: (B, nhist, embedding_dim) or None

        Returns:
            (B, T, hidden_size)
        """
        B, T, _ = trajectory.shape
        device = trajectory.device

        action_emb = self.action_proj(trajectory)  # (B, T, H)

        # Timestep conditioning — broadcast to all steps
        t_cond = self.time_emb(timestep.float())   # (B, H)
        action_emb = action_emb + t_cond.unsqueeze(1)

        # Gripper conditioning
        if proprio_feats is not None:
            g_flat = proprio_feats.reshape(B, -1)  # (B, nhist*emb_dim)
            g_cond = self.gripper_proj(g_flat)     # (B, H)
            action_emb = action_emb + g_cond.unsqueeze(1)

        # Learnable positional offsets
        step_ids = torch.arange(T, device=device).clamp(max=self.max_chunk - 1)
        action_emb = action_emb + self.traj_pos_emb(step_ids).unsqueeze(0)

        return action_emb

    def predict(self, action_hidden):
        """(B, T, hidden_size) → (B, T, pos+rot+openess)"""
        return self.output_head(action_hidden)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class SmolVLAPrefixKVActor(BaseDenoiseActor):
    """
    Inherits train/inference scaffolding from BaseDenoiseActor (normalize_pos,
    unnormalize_pos, convert_rot, conditional_sample, compute_loss, forward).

    Overrides:
      encode_inputs()       — builds prefix K/V cache once per observation
      policy_forward_pass() — decodes only action tokens at each denoising step
    """

    def __init__(
        self,
        # SmolVLM / encoder kwargs
        smolvlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        smolvlm_local_files_only: bool = True,
        smolvlm_freeze_vision_tower: bool = True,
        smolvlm_freeze_connector: bool = True,
        smolvlm_freeze_text_model: bool = True,
        smolvlm_freeze_text_embeddings: bool = True,
        smolvlm_tokenizer_max_length: int = 48,
        smolvlm_append_state_tokens: bool = False,
        smolvlm_state_dim: int = 0,
        smolvlm_num_state_tokens: int = 1,
        # Shared architecture kwargs
        embedding_dim: int = 60,
        num_attn_heads: int = 9,
        nhist: int = 3,
        nhand: int = 1,
        fps_subsampling_factor: int = 5,
        # Decoder kwargs (passed to base)
        num_shared_attn_layers: int = 4,
        relative: bool = False,
        rotation_format: str = "quat_xyzw",
        # Denoising kwargs
        denoise_timesteps: int = 100,
        denoise_model: str = "rectified_flow",
        lv2_batch_size: int = 1,
        # Absorb kwargs passed by BaseTrainTester.get_model() that don't apply here
        **ignored_kwargs,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size,
        )

        rot_dim = 3 if rotation_format == "euler" else 6

        # --- SmolVLM encoder (replaces self.encoder from base) ---
        self.encoder = SmolVLMEncoder(
            model_name=smolvlm_model_name,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            tokenizer_max_length=smolvlm_tokenizer_max_length,
            freeze_vision_tower=smolvlm_freeze_vision_tower,
            freeze_connector=smolvlm_freeze_connector,
            freeze_text_model=smolvlm_freeze_text_model,
            freeze_text_embeddings=smolvlm_freeze_text_embeddings,
            local_files_only=smolvlm_local_files_only,
            append_state_tokens=smolvlm_append_state_tokens,
            state_dim=smolvlm_state_dim,
            num_state_tokens=smolvlm_num_state_tokens,
        )

        # --- Prefix-KV action head (replaces self.prediction_head from base) ---
        hidden_size = self.encoder.smolvlm_hidden_size
        self.prediction_head = PrefixKVTransformerHead(
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            rot_dim=rot_dim,
        )

        # Remove base-class traj_encoder (not used by prefix-KV head)
        if hasattr(self, "traj_encoder"):
            del self.traj_encoder

    # ------------------------------------------------------------------
    # encode_inputs  — once per observation step
    # ------------------------------------------------------------------

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """
        Run SmolVLMEncoder to build the prefix K/V cache and supporting features.

        Returns fixed_inputs tuple (passed to policy_forward_pass).
        """
        (
            visual_proj,
            pcd_flat,
            cond_proj,
            cond_mask,
            proprio_feats,
            fps_feats,
            fps_pos,
            prefix_kv,
            n_prefix,
            prefix_pad_mask,
        ) = self.encoder(rgb3d, rgb2d, pcd, instruction, proprio)

        # Query trajectory: last frame of proprio (used in relative mode)
        query_trajectory = proprio[:, -1:]  # (B, 1, C)

        return (
            query_trajectory,   # [0]  (B, 1, C)
            visual_proj,        # [1]  (B, n_vis, emb_dim)
            pcd_flat,           # [2]  (B, n_vis, 3)
            cond_proj,          # [3]  (B, n_lang+n_state, emb_dim)
            cond_mask,          # [4]  (B, n_lang+n_state) bool
            proprio_feats,      # [5]  (B, nhist, emb_dim)
            fps_feats,          # [6]  (B, M, emb_dim)
            fps_pos,            # [7]  (B, M, 3)
            prefix_kv,          # [8]  DynamicCache
            n_prefix,           # [9]  int
            prefix_pad_mask,    # [10] (B, n_prefix) bool
        )

    # ------------------------------------------------------------------
    # policy_forward_pass  — called at each denoising step
    # ------------------------------------------------------------------

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        """
        Args:
            trajectory   : (B, T, nhand, 3+rot_dim) noisy action chunk
            timestep     : (B,) denoising timestep
            fixed_inputs : tuple from encode_inputs()

        Returns:
            list of (B, T, nhand, 3+rot_dim+1)  — list for base-class compat
        """
        (
            query_trajectory,
            _visual_proj,
            pcd_flat,
            _cond_proj,
            _cond_mask,
            proprio_feats,
            _fps_feats,
            _fps_pos,
            prefix_kv,
            n_prefix,
            prefix_pad_mask,
        ) = fixed_inputs

        # Flatten nhand into sequence dimension
        _, traj_len, nhand, _ = trajectory.shape
        traj_flat = trajectory.flatten(1, 2)  # (B, T*nhand, 3+rot_dim)

        # Absolute xyz for relative mode (not used by prefix-KV head directly,
        # but preserved for consistency)
        if self._relative:
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(self.unnormalize_pos(traj_flat)[..., :3], dim=1)
            )

        # --- Encode action tokens ---
        action_emb = self.prediction_head.encode_action_tokens(
            trajectory=traj_flat,
            timestep=timestep,
            proprio_feats=proprio_feats,
        )  # (B, T*nhand, hidden_size)

        # --- Decode with prefix K/V (only action tokens at each step) ---
        action_hidden = self.encoder.prefix_kv_backbone.decode_action(
            action_embeds=action_emb,
            prefix_kv=prefix_kv,
            n_prefix=n_prefix,
            prefix_pad_mask=prefix_pad_mask,
        )  # (B, T*nhand, hidden_size)

        # --- Predict actions ---
        pred = self.prediction_head.predict(action_hidden)  # (B, T*nhand, pos+rot+1)
        pred = pred.unflatten(1, (traj_len, nhand))         # (B, T, nhand, pos+rot+1)

        return [pred]   # list — base class does `out = out[-1]`
