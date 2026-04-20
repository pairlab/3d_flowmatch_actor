"""
SmolVLMEncoder — adapts SmolVLM's vision + text towers to the
3d_flowmatch_actor encoder interface via prefix-KV fusion.

Input convention (same as the base Encoder):
    rgb3d       : (B, ncam, 3, H, W) in [0, 1]
    pcd         : (B, ncam, 3, H, W) world-frame point cloud
    instruction : list[str] of length B
    proprio     : (B, nhist, nhand, C) — last frame (index -1) is used for state

Output from forward() is a 4-tuple consumed by SmolVLAPrefixKVActor.encode_inputs():

    proprio_feats   : (B, nhist*nhand, embedding_dim)
    prefix_kv       : DynamicCache
    n_prefix        : int
    prefix_pad_mask : (B, n_prefix) bool  (True = padding)

Fusion (visual + language + optional state) happens inside SmolVLM's text model
during ``prefix_kv_backbone.encode_prefix``; the resulting layer-wise K/V is
reused by the action decoder, so nothing in this encoder is consumed by the
denoising loop besides ``proprio_feats`` and the cache triple.
"""

import math
from contextlib import nullcontext

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.layers import AttentionModule
from ...utils.position_encodings import RotaryPositionEncoding3D
from .smolvlm_backbone import SmolVLMPrefixKVBackbone, apply_lora_to_text_model


class SmolVLMEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        embedding_dim: int = 60,
        nhist: int = 3,
        num_attn_heads: int = 9,
        tokenizer_max_length: int = 48,
        freeze_vision_tower: bool = True,
        freeze_connector: bool = True,
        freeze_text_model: bool = True,
        freeze_text_embeddings: bool = True,
        local_files_only: bool = False,
        append_state_tokens: bool = False,
        state_dim: int = 0,
        num_state_tokens: int = 1,
        lora_rank: int = 0,
        lora_alpha: float = 0.0,
        lora_last_n_layers: int = 0,
        xyz_rope_scale: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_proprio_slots = int(nhist)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.freeze_vision_tower = freeze_vision_tower
        self.freeze_connector = freeze_connector
        self.freeze_text_model = freeze_text_model
        self.append_state_tokens = append_state_tokens
        self.state_dim = int(state_dim) if append_state_tokens else 0
        self.num_state_tokens = int(num_state_tokens) if append_state_tokens else 0

        vlm, tokenizer = self._load_smolvlm(model_name, local_files_only)
        self.tokenizer = tokenizer
        backbone = getattr(vlm, "model", vlm)
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.connector = backbone.connector
        self.smolvlm_hidden_size = self.text_model.config.hidden_size

        self.prefix_kv_backbone = SmolVLMPrefixKVBackbone(
            text_model=self.text_model,
            output_dim=embedding_dim,
            state_dim=self.state_dim,
            num_state_tokens=self.num_state_tokens,
            xyz_rope_scale=xyz_rope_scale,
        )

        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.curr_gripper_embed = nn.Embedding(self.num_proprio_slots, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False,
        )

        self._apply_trainability(
            freeze_vision_tower, freeze_connector, freeze_text_model, freeze_text_embeddings,
        )

        apply_lora_to_text_model(
            self.text_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha) if lora_alpha else float(lora_rank),
            last_n_layers=int(lora_last_n_layers),
        )

    @staticmethod
    def _load_smolvlm(model_name: str, local_files_only: bool):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vlm = AutoModelForImageTextToText.from_pretrained(model_name, local_files_only=local_files_only)
        return vlm, tokenizer

    def _apply_trainability(self, freeze_vis, freeze_conn, freeze_txt, freeze_emb):
        def _set(module, trainable):
            for p in module.parameters():
                p.requires_grad = trainable

        _set(self.vision_model, not freeze_vis)
        _set(self.connector, not freeze_conn)
        _set(self.text_model, not freeze_txt)
        if freeze_emb:
            _set(self.text_model.get_input_embeddings(), False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_tower:
            self.vision_model.eval()
        if self.freeze_connector:
            self.connector.eval()
        if self.freeze_text_model:
            self.text_model.eval()
        return self

    def _vision_forward_ctx(self):
        return torch.no_grad() if self.freeze_vision_tower else nullcontext()

    def _encode_vision_hidden(self, rgb: torch.Tensor):
        """(B*ncam, 3, H, W) in [0, 1] → (B*ncam, n_tok, hidden_size) in SigLIP-normalized space."""
        rgb = torch.clamp(rgb, 0.0, 1.0) * 2.0 - 1.0
        model_dtype = next(self.vision_model.parameters()).dtype
        with self._vision_forward_ctx():
            hidden = self.vision_model(pixel_values=rgb.to(dtype=model_dtype)).last_hidden_state
        hidden = self.connector(hidden).to(torch.float32)
        seq_len = hidden.shape[1]
        side = math.isqrt(seq_len)
        if side * side != seq_len:
            raise RuntimeError(
                f"SmolVLM connector produced {seq_len} tokens which is not a perfect square. "
                "The encoder's pcd↔feature alignment requires a square grid; adjust the "
                "connector output or the expected grid shape."
            )
        return hidden, side, side

    def encode_visual_tokens(self, rgb3d: torch.Tensor):
        """Returns ``(B, ncam*feat_h*feat_w, hidden_size)``, ``feat_h``, ``feat_w``, ``ncam``."""
        B, ncam, _, _, _ = rgb3d.shape
        rgb_flat = einops.rearrange(rgb3d, "b ncam c h w -> (b ncam) c h w")
        hidden, feat_h, feat_w = self._encode_vision_hidden(rgb_flat)
        visual_tokens = einops.rearrange(
            hidden, "(b ncam) (h w) c -> b (ncam h w) c",
            b=B, ncam=ncam, h=feat_h, w=feat_w,
        )
        return visual_tokens, feat_h, feat_w, ncam

    @staticmethod
    def _flatten_pcd(pcd: torch.Tensor, ncam: int, feat_h: int, feat_w: int):
        B = pcd.shape[0]
        pcd_flat = einops.rearrange(pcd, "b ncam c h w -> (b ncam) c h w")
        if pcd_flat.shape[-2:] != (feat_h, feat_w):
            pcd_flat = F.interpolate(pcd_flat, (feat_h, feat_w), mode="bilinear", align_corners=False)
        return einops.rearrange(
            pcd_flat, "(b ncam) c h w -> b (ncam h w) c",
            b=B, ncam=ncam, h=feat_h, w=feat_w,
        )

    def encode_proprio(self, proprio_flat, visual_feats, pcd_flat):
        """
        proprio_flat : (B, num_proprio_slots, C)
        visual_feats : (B, n_vis, embedding_dim)
        pcd_flat     : (B, n_vis, 3)
        """
        B = proprio_flat.shape[0]
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).expand(B, -1, -1)
        proprio_pos = self.relative_pe_layer(proprio_flat[..., :3])
        context_pos = self.relative_pe_layer(pcd_flat)
        return self.gripper_context_head(
            proprio_feats, visual_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos,
        )[-1]

    def tokenize(self, instruction, device):
        if isinstance(instruction, str):
            instruction = [instruction]
        tokens = self.tokenizer(
            list(instruction),
            padding="longest",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)

    def forward(self, rgb3d, pcd, instruction, proprio):
        """
        Run vision + language + state fusion, return the handles the action
        decoder needs.
        """
        device = rgb3d.device
        B = rgb3d.shape[0]

        visual_tokens, feat_h, feat_w, ncam = self.encode_visual_tokens(rgb3d)
        pcd_flat = self._flatten_pcd(pcd, ncam, feat_h, feat_w)
        n_vis = visual_tokens.shape[1]

        input_ids, lang_attn_mask = self.tokenize(instruction, device)

        if self.append_state_tokens:
            state = proprio[:, -1, 0, : self.state_dim].to(device)
            state_xyz = proprio[:, -1, 0, :3].to(device)
            n_state = self.num_state_tokens
        else:
            state = None
            state_xyz = None
            n_state = 0

        visual_proj, _cond_proj, _cond_mask, prefix_kv, n_prefix = (
            self.prefix_kv_backbone.encode_prefix(
                input_ids=input_ids,
                lang_attention_mask=lang_attn_mask,
                visual_tokens=visual_tokens,
                visual_xyz=pcd_flat,
                state=state,
                state_xyz=state_xyz,
            )
        )

        vis_pad = torch.zeros((B, n_vis), dtype=torch.bool, device=device)
        lang_pad = lang_attn_mask == 0
        mask_parts = [vis_pad, lang_pad]
        if n_state > 0:
            mask_parts.append(torch.zeros((B, n_state), dtype=torch.bool, device=device))
        prefix_pad_mask = torch.cat(mask_parts, dim=1)

        proprio_flat = proprio.reshape(B, self.num_proprio_slots, -1)
        proprio_feats = self.encode_proprio(proprio_flat, visual_proj, pcd_flat)

        return proprio_feats, prefix_kv, n_prefix, prefix_pad_mask
