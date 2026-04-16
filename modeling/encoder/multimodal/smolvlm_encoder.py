"""
SmolVLMEncoder: adapts SmolVLMPrefixKVBackbone to the 3d_flowmatch_actor
Encoder interface.

Input convention (same as base Encoder):
    rgb3d   : (B, ncam, 3, H, W) in [0, 1]
    pcd     : (B, ncam, 3, H, W) world-frame point cloud
    instruction : list[str] of length B
    proprio : (B, nhist, 3+6+X)   — last dim used for state tokens if enabled

Output from forward():
    visual_proj     : (B, n_vis, embedding_dim)
    pcd_flat        : (B, n_vis, 3)
    cond_proj       : (B, n_lang+n_state, embedding_dim)
    cond_mask       : (B, n_lang+n_state) bool, True=padding
    proprio_feats   : (B, nhist, embedding_dim)
    fps_scene_feats : (B, M, embedding_dim)
    fps_scene_pos   : (B, M, 3)
    prefix_kv       : DynamicCache
    n_prefix        : int
    prefix_pad_mask : (B, n_prefix) bool
"""

import math
from contextlib import nullcontext

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.layers import AttentionModule
from ...utils.position_encodings import RotaryPositionEncoding3D
from .base_encoder import Encoder as BaseEncoder
from .smolvlm_backbone import SmolVLMPrefixKVBackbone


MIN_TRANSFORMERS_VERSION = (4, 46, 0)


def _parse_version(version: str):
    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


class SmolVLMEncoder(nn.Module):
    """
    Vision-language encoder backed by SmolVLM.

    Runs the SmolVLM vision tower to extract visual tokens, fuses them with
    tokenized language (and optional state) through the shared Llama decoder
    using mixed 3D/1D rotary embeddings, and caches the resulting layer-wise
    K/V for use by the action decoder.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        embedding_dim: int = 60,
        nhist: int = 3,
        num_attn_heads: int = 9,
        fps_subsampling_factor: int = 5,
        tokenizer_max_length: int = 48,
        pad_language_to: str = "longest",
        freeze_vision_tower: bool = True,
        freeze_connector: bool = True,
        freeze_text_model: bool = True,
        freeze_text_embeddings: bool = True,
        local_files_only: bool = True,
        # state token options
        append_state_tokens: bool = False,
        state_keys: tuple = ("proprio",),
        state_dim: int = 0,
        num_state_tokens: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhist = nhist
        self.fps_subsampling_factor = fps_subsampling_factor
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.pad_language_to = pad_language_to
        self.freeze_vision_tower = freeze_vision_tower
        self.freeze_connector = freeze_connector
        self.freeze_text_model = freeze_text_model
        self.append_state_tokens = append_state_tokens
        self.state_keys = tuple(state_keys)
        self.state_dim = int(state_dim)
        self.num_state_tokens = int(num_state_tokens)

        # --- Load SmolVLM ---
        vlm, tokenizer = self._load_smolvlm(model_name, local_files_only)
        self.tokenizer = tokenizer
        backbone = getattr(vlm, "model", vlm)
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.connector = backbone.connector

        hidden_size = self.text_model.config.hidden_size
        self.smolvlm_hidden_size = hidden_size

        # --- Feature projections ---
        self.visual_proj = nn.Linear(hidden_size, embedding_dim)

        # --- Prefix-KV backbone ---
        self.prefix_kv_backbone = SmolVLMPrefixKVBackbone(
            text_model=self.text_model,
            output_dim=embedding_dim,
            state_dim=self.state_dim if self.append_state_tokens else 0,
            num_state_tokens=self.num_state_tokens,
        )

        # --- 3D positional encoding ---
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # --- Gripper / proprio encoding ---
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )

        # --- Apply trainability ---
        self._apply_trainability(
            freeze_vision_tower, freeze_connector, freeze_text_model, freeze_text_embeddings
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_smolvlm(model_name: str, local_files_only: bool):
        import transformers
        from transformers import AutoTokenizer

        if _parse_version(transformers.__version__) < MIN_TRANSFORMERS_VERSION:
            min_str = ".".join(str(x) for x in MIN_TRANSFORMERS_VERSION)
            raise RuntimeError(
                f"SmolVLM requires transformers >= {min_str}, "
                f"found {transformers.__version__}."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        loader = getattr(transformers, "AutoModelForImageTextToText", None)
        if loader is None:
            loader = transformers.AutoModelForVision2Seq
        vlm = loader.from_pretrained(model_name, local_files_only=local_files_only)
        return vlm, tokenizer

    def _apply_trainability(self, freeze_vis, freeze_conn, freeze_txt, freeze_emb):
        def _set(module, trainable):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = trainable
        _set(self.vision_model, not freeze_vis)
        _set(self.connector, not freeze_conn)
        _set(self.text_model, not freeze_txt)
        if freeze_emb and self.text_model is not None:
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

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------

    def _vision_ctx(self):
        return torch.no_grad if self.freeze_vision_tower else nullcontext

    def _encode_vision_hidden(self, rgb: torch.Tensor):
        """(B*ncam, 3, H, W) → (B*ncam, n_tok, hidden_size), grid_h, grid_w"""
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = torch.clamp(rgb, 0.0, 1.0) * 2.0 - 1.0
        model_dtype = next(self.vision_model.parameters()).dtype
        with self._vision_ctx()():
            hidden = self.vision_model(
                pixel_values=rgb.to(dtype=model_dtype)
            ).last_hidden_state
        hidden = self.connector(hidden).to(torch.float32)
        seq_len = hidden.shape[1]
        side = math.isqrt(seq_len)
        if side * side != seq_len:
            hidden = hidden[:, 1:]
            seq_len = hidden.shape[1]
            side = math.isqrt(seq_len)
        return hidden, side, side

    def encode_visual_tokens(self, rgb3d: torch.Tensor):
        """
        Args:
            rgb3d: (B, ncam, 3, H, W) in [0, 1]
        Returns:
            visual_tokens: (B, ncam*n_tok, hidden_size) — NOT projected
            feat_h, feat_w: grid resolution per camera
            ncam: number of cameras
        """
        B, ncam, C, H, W = rgb3d.shape
        rgb_flat = einops.rearrange(rgb3d, "b ncam c h w -> (b ncam) c h w")
        hidden, feat_h, feat_w = self._encode_vision_hidden(rgb_flat)
        # Reshape: (B, ncam*n_tok, hidden_size)
        visual_tokens = einops.rearrange(
            hidden, "(b ncam) (h w) c -> b (ncam h w) c",
            b=B, ncam=ncam, h=feat_h, w=feat_w
        )
        return visual_tokens, feat_h, feat_w, ncam

    def _flatten_pcd(self, pcd: torch.Tensor, ncam: int, feat_h: int, feat_w: int):
        """
        (B, ncam, 3, H, W) point cloud → (B, ncam*feat_h*feat_w, 3) matching visual tokens.
        """
        B = pcd.shape[0]
        orig_h, orig_w = pcd.shape[-2], pcd.shape[-1]
        pcd_flat = einops.rearrange(pcd, "b ncam c h w -> (b ncam) c h w")
        if orig_h != feat_h or orig_w != feat_w:
            pcd_flat = F.interpolate(pcd_flat, (feat_h, feat_w), mode="bilinear", align_corners=False)
        return einops.rearrange(
            pcd_flat, "(b ncam) c h w -> b (ncam h w) c",
            b=B, ncam=ncam, h=feat_h, w=feat_w
        )

    # ------------------------------------------------------------------
    # Proprio encoding
    # ------------------------------------------------------------------

    def encode_proprio(self, proprio, visual_feats, pcd_flat):
        """
        Args:
            proprio    : (B, nhist, C)
            visual_feats: (B, n_vis, embedding_dim)
            pcd_flat   : (B, n_vis, 3)
        Returns:
            (B, nhist, embedding_dim)
        """
        B = proprio.shape[0]
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).expand(B, -1, -1)
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        context_pos = self.relative_pe_layer(pcd_flat)
        proprio_feats = self.gripper_context_head(
            proprio_feats, visual_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos
        )[-1]
        return proprio_feats

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, instruction, device):
        if isinstance(instruction, str):
            instruction = [instruction]
        batch = list(instruction)
        batch = [t if t.endswith("\n") else f"{t}\n" for t in batch]
        tokens = self.tokenizer(
            batch,
            padding=self.pad_language_to,
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """
        Encode all modalities and build prefix K/V cache.

        Returns a tuple consumed by SmolVLAPrefixKVActor.encode_inputs().
        """
        device = rgb3d.device if torch.is_tensor(rgb3d) else next(self.parameters()).device
        B = rgb3d.shape[0]

        # --- Visual tokens + point cloud ---
        visual_tokens, feat_h, feat_w, ncam = self.encode_visual_tokens(rgb3d)
        pcd_flat = self._flatten_pcd(pcd, ncam, feat_h, feat_w)
        n_vis = visual_tokens.shape[1]

        # --- Language tokens ---
        input_ids, lang_attn_mask = self.tokenize(instruction, device)
        n_lang = input_ids.shape[1]

        # --- Optional state tokens from proprio ---
        state = None
        state_xyz = None
        if self.append_state_tokens and self.state_dim > 0:
            # Last frame of proprio as state vector
            state = proprio[:, -1, : self.state_dim].to(device)
            state_xyz = proprio[:, -1, :3].to(device)  # EEF position
        n_state = self.num_state_tokens if (self.append_state_tokens and self.state_dim > 0) else 0

        # --- Prefix K/V ---
        visual_proj, cond_proj, cond_mask, prefix_kv, n_prefix = (
            self.prefix_kv_backbone.encode_prefix(
                input_ids=input_ids,
                lang_attention_mask=lang_attn_mask,
                visual_tokens=visual_tokens,
                visual_xyz=pcd_flat,
                state=state,
                state_xyz=state_xyz,
            )
        )

        # --- Prefix padding mask for action-to-prefix attention ---
        vis_pad = torch.zeros((B, n_vis), dtype=torch.bool, device=device)
        lang_pad = lang_attn_mask == 0
        if n_state > 0:
            state_pad = torch.zeros((B, n_state), dtype=torch.bool, device=device)
            prefix_pad_mask = torch.cat([vis_pad, lang_pad, state_pad], dim=1)
        else:
            prefix_pad_mask = torch.cat([vis_pad, lang_pad], dim=1)

        # --- Proprio features ---
        if proprio is not None:
            # proprio: (B, nhist, C) — use projected visual features for context
            proprio_flat = proprio.reshape(B, self.nhist, -1) if proprio.ndim == 4 else proprio
            proprio_feats = self.encode_proprio(proprio_flat, visual_proj, pcd_flat)
        else:
            proprio_feats = None

        # --- FPS subsampling ---
        fps_feats, fps_pos = self._run_fps_safe(visual_proj, pcd_flat)

        return (
            visual_proj,      # (B, n_vis, emb_dim)
            pcd_flat,         # (B, n_vis, 3)
            cond_proj,        # (B, n_lang+n_state, emb_dim)
            cond_mask,        # (B, n_lang+n_state) bool
            proprio_feats,    # (B, nhist, emb_dim) or None
            fps_feats,        # (B, M, emb_dim)
            fps_pos,          # (B, M, 3)
            prefix_kv,        # DynamicCache
            n_prefix,         # int
            prefix_pad_mask,  # (B, n_prefix) bool
        )

    def _run_fps_safe(self, visual_feats, pcd_flat):
        if self.fps_subsampling_factor <= 1:
            return visual_feats, pcd_flat
        B, N, _ = visual_feats.shape
        M = max(N // self.fps_subsampling_factor, 1)
        # Use density-based sampler from base_encoder
        try:
            from .base_encoder import density_based_sampler
            inds = density_based_sampler(visual_feats, self.fps_subsampling_factor)
            fps_feats = torch.gather(
                visual_feats, 1,
                inds.unsqueeze(-1).expand(-1, -1, visual_feats.shape[-1])
            )
            fps_pos = torch.gather(
                pcd_flat, 1,
                inds.unsqueeze(-1).expand(-1, -1, 3)
            )
            return fps_feats, fps_pos
        except Exception:
            return visual_feats[:, :M], pcd_flat[:, :M]
