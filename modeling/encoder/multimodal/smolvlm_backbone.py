"""
SmolVLM backbone with mixed 3D/1D rotary position encoding and prefix KV caching.

Token ordering for the prefix: [visual (3D RoPE), language (1D RoPE), state (3D RoPE)].
Action tokens (decode phase): 1D RoPE at positions [n_prefix, n_prefix+n_action).
Full (bidirectional) attention throughout — no causal masking.

The prefix K/V tensors are cached in a DynamicCache after a single backbone
forward pass.  Each denoising step then seeds a fresh DynamicCache by
shallow-copying the list references (no tensor copies), runs only the action
tokens through the decoder layers, and discards the extended cache.
"""

import torch
from torch import nn

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
except ImportError:
    DynamicCache = None  # will raise at runtime if KV cache path is used
    DynamicLayer = None


# ---------------------------------------------------------------------------
# Mixed-modality rotary position embedding
# ---------------------------------------------------------------------------

class MixedModalityRotaryEmbedding(nn.Module):
    """
    Builds per-token RoPE (cos, sin) for a mixed-modality sequence:
      visual tokens  → 3D RoPE  (point-cloud xyz)
      language tokens → 1D RoPE  (sequential positions)
      state tokens   → 3D RoPE  (end-effector xyz)
      action tokens  → 1D RoPE  (sequential, starting after prefix)
    """

    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float = 1.0):
        super().__init__()
        self.register_buffer(
            "inv_freq", inv_freq.detach().clone().to(torch.float32), persistent=False
        )
        self.attention_scaling = attention_scaling

    def _build_1d(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def _build_3d(self, xyz: torch.Tensor, dtype: torch.dtype):
        half_dim = self.inv_freq.shape[0]
        base_split = half_dim // 3
        split_sizes = [base_split, base_split, half_dim - 2 * base_split]
        freq_chunks = torch.split(self.inv_freq, split_sizes)
        coords = [xyz[..., 0:1], xyz[..., 1:2], xyz[..., 2:3]]
        parts = []
        for coord, freq_chunk in zip(coords, freq_chunks):
            if freq_chunk.numel() == 0:
                continue
            parts.append(coord.to(torch.float32) * freq_chunk.view(1, 1, -1))
        freqs = torch.cat(parts, dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def forward_prefix(self, visual_xyz, lang_positions, state_xyz, dtype):
        vis_cos, vis_sin = self._build_3d(visual_xyz, dtype=dtype)
        lang_cos, lang_sin = self._build_1d(lang_positions, dtype=dtype)
        cos_parts, sin_parts = [vis_cos, lang_cos], [vis_sin, lang_sin]
        if state_xyz is not None:
            sc, ss = self._build_3d(state_xyz, dtype=dtype)
            cos_parts.append(sc)
            sin_parts.append(ss)
        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    def forward_action(self, n_prefix, n_action, batch_size, device, dtype):
        positions = (
            torch.arange(n_prefix, n_prefix + n_action, device=device)
            .unsqueeze(0).expand(batch_size, -1)
        )
        return self._build_1d(positions, dtype=dtype)


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class MixedPositionTextBackbone(nn.Module):
    """
    Wraps the SmolVLM/Llama text model.  Always full (bidirectional) attention.

    forward()                     — prefix pass, optionally fills DynamicCache
    forward_action_with_prefix()  — decode-only pass on action tokens
    """

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model
        self.config = text_model.config

        rotary_emb = getattr(text_model, "rotary_emb", None)
        if rotary_emb is None or not hasattr(rotary_emb, "inv_freq"):
            raise RuntimeError(
                "Expected a Llama-style text model with `rotary_emb.inv_freq`, "
                f"got {type(text_model)!r}."
            )
        self.mixed_rope = MixedModalityRotaryEmbedding(
            rotary_emb.inv_freq,
            attention_scaling=getattr(rotary_emb, "attention_scaling", 1.0),
        )

    def _full_attn_mask(self, inputs_embeds, attention_mask):
        if attention_mask is None:
            return None
        valid = attention_mask.to(torch.bool)
        if torch.all(valid):
            return None
        neg_inf = torch.finfo(inputs_embeds.dtype).min
        pad_mask = (~valid).to(inputs_embeds.dtype) * neg_inf
        return pad_mask[:, None, None, :]

    def _run_layers(self, hidden_states, attention_mask_4d, position_ids,
                    cache_position, position_embeddings, past_key_values, use_cache):
        for layer in self.text_model.layers[: self.config.num_hidden_layers]:
            out = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        return self.text_model.norm(hidden_states), past_key_values

    def forward(self, inputs_embeds, visual_xyz, n_visual, n_lang,
                state_xyz=None, attention_mask=None, position_ids=None,
                use_cache=False):
        """Run prefix [visual, lang, state]. Returns (hidden_states, cache_or_None)."""
        B, S, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        cache_position = torch.arange(S, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(B, -1)
        attention_mask_4d = self._full_attn_mask(inputs_embeds, attention_mask)
        lang_positions = position_ids[:, n_visual : n_visual + n_lang]
        position_embeddings = self.mixed_rope.forward_prefix(
            visual_xyz=visual_xyz, lang_positions=lang_positions,
            state_xyz=state_xyz, dtype=dtype,
        )
        if use_cache:
            if DynamicCache is None:
                raise RuntimeError("transformers.cache_utils.DynamicCache not available.")
            past_key_values = DynamicCache()
        else:
            past_key_values = None
        return self._run_layers(
            inputs_embeds, attention_mask_4d, position_ids,
            cache_position, position_embeddings, past_key_values, use_cache,
        )

    def forward_action_with_prefix(self, action_embeds, prefix_kv, n_prefix,
                                   prefix_pad_mask=None):
        """
        Decode action tokens using prefix K/V as read-only context.
        Seeds a fresh DynamicCache via shallow list copy — no tensor copies.
        """
        B, n_action, _ = action_embeds.shape
        device = action_embeds.device
        dtype = action_embeds.dtype

        # Seed fresh cache from prefix K/V (no tensor copies, prefix_kv stays frozen)
        action_cache = DynamicCache()
        for layer in prefix_kv.layers:
            new_layer = DynamicLayer()
            new_layer.keys = layer.keys
            new_layer.values = layer.values
            new_layer.dtype = layer.dtype
            new_layer.device = layer.device
            new_layer.is_initialized = True
            action_cache.layers.append(new_layer)

        cache_position = torch.arange(n_prefix, n_prefix + n_action, device=device)
        position_ids = cache_position.unsqueeze(0).expand(B, -1)

        if prefix_pad_mask is not None:
            action_pad = torch.zeros((B, n_action), dtype=torch.bool, device=device)
            full_pad = torch.cat([prefix_pad_mask, action_pad], dim=1)
            neg_inf = torch.finfo(dtype).min
            attn_bias = (full_pad.to(dtype) * neg_inf)[:, None, None, :]
        else:
            attn_bias = None

        position_embeddings = self.mixed_rope.forward_action(
            n_prefix=n_prefix, n_action=n_action,
            batch_size=B, device=device, dtype=dtype,
        )
        hidden_states, _ = self._run_layers(
            action_embeds, attn_bias, position_ids,
            cache_position, position_embeddings, action_cache, use_cache=True,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# High-level backbone: encode_prefix + decode_action
# ---------------------------------------------------------------------------

class SmolVLMPrefixKVBackbone(nn.Module):
    """
    Fuses [visual, lang, state] through the SmolVLM decoder once (encode_prefix),
    then decodes noisy action tokens using the cached K/V (decode_action).

    Parameters
    ----------
    text_model       : SmolVLM/Llama text model (shared decoder)
    output_dim       : projected dimension for visual/cond feature outputs
    state_dim        : low-dim robot state size (0 = disabled)
    num_state_tokens : tokens produced per state vector
    """

    def __init__(self, text_model: nn.Module, output_dim: int,
                 state_dim: int = 0, num_state_tokens: int = 1):
        super().__init__()
        self.text_model = text_model
        self.hidden_size = text_model.config.hidden_size
        self.output_proj = nn.Linear(self.hidden_size, output_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.state_dim = int(state_dim)
        self.state_proj: nn.Linear | None = None
        if self.state_dim > 0 and self.num_state_tokens > 0:
            self.state_proj = nn.Linear(
                self.state_dim, self.hidden_size * self.num_state_tokens
            )
        self.text_backbone = MixedPositionTextBackbone(text_model=text_model)

    def _encode_state(self, state, dtype):
        if self.state_proj is None or state is None:
            return None
        tokens = self.state_proj(state.to(dtype=torch.float32))
        return tokens.view(state.shape[0], self.num_state_tokens, self.hidden_size).to(dtype)

    def encode_prefix(
        self, *, input_ids, lang_attention_mask, visual_tokens, visual_xyz,
        visual_attention_mask=None, state=None, state_xyz=None,
    ):
        """
        Run prefix [visual, lang, state] through backbone with KV caching.

        Returns
        -------
        visual_proj   : (B, n_visual, output_dim)
        cond_proj     : (B, n_lang + n_state, output_dim)
        cond_mask     : (B, n_lang + n_state) bool, True = padding
        prefix_kv     : DynamicCache — layer-wise K/V (never mutated)
        n_prefix      : int — total prefix length
        """
        device = visual_tokens.device
        model_dtype = next(self.text_model.parameters()).dtype

        visual_tokens = visual_tokens.to(device=device, dtype=model_dtype)
        n_visual = visual_tokens.shape[1]
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(
                (visual_tokens.shape[0], n_visual), dtype=torch.long, device=device
            )
        else:
            visual_attention_mask = visual_attention_mask.to(dtype=torch.long, device=device)

        lang_embeds = self.text_model.get_input_embeddings()(
            input_ids.to(device)
        ).to(dtype=model_dtype)
        lang_mask = lang_attention_mask.to(dtype=torch.long, device=device)
        n_lang = lang_embeds.shape[1]

        state_tokens = self._encode_state(state, dtype=model_dtype)
        if state_tokens is not None:
            state_tokens = state_tokens.to(device=device)
            n_state = state_tokens.shape[1]
            state_mask = torch.ones(
                (state_tokens.shape[0], n_state), dtype=torch.long, device=device
            )
        else:
            n_state = 0

        parts = [visual_tokens, lang_embeds]
        mask_parts = [visual_attention_mask, lang_mask]
        if state_tokens is not None:
            parts.append(state_tokens)
            mask_parts.append(state_mask)
        full_inputs = torch.cat(parts, dim=1)
        full_mask = torch.cat(mask_parts, dim=1)

        # state xyz for 3D RoPE
        if state_xyz is not None and n_state > 0:
            state_xyz_rope = state_xyz.to(device=device, dtype=torch.float32)
            if state_xyz_rope.ndim == 2:
                state_xyz_rope = state_xyz_rope.unsqueeze(1)
            if state_xyz_rope.shape[1] == 1 and n_state > 1:
                state_xyz_rope = state_xyz_rope.expand(-1, n_state, -1)
        elif n_state > 0:
            state_xyz_rope = torch.zeros(
                (full_inputs.shape[0], n_state, 3), dtype=torch.float32, device=device
            )
        else:
            state_xyz_rope = None

        hidden_states, prefix_kv = self.text_backbone.forward(
            inputs_embeds=full_inputs,
            visual_xyz=visual_xyz.to(device=device, dtype=torch.float32),
            n_visual=n_visual, n_lang=n_lang,
            state_xyz=state_xyz_rope,
            attention_mask=full_mask,
            use_cache=True,
        )
        n_prefix = n_visual + n_lang + n_state

        visual_hidden = hidden_states[:, :n_visual]
        lang_hidden = hidden_states[:, n_visual : n_visual + n_lang]
        if n_state > 0:
            state_hidden = hidden_states[:, n_visual + n_lang:]
            cond_hidden = torch.cat([lang_hidden, state_hidden], dim=1)
            cond_attn_mask = torch.cat([lang_mask, state_mask], dim=1)
        else:
            cond_hidden = lang_hidden
            cond_attn_mask = lang_mask

        cond_mask = cond_attn_mask == 0
        vis_mask_bool = visual_attention_mask == 0
        cond_hidden = cond_hidden.masked_fill(cond_mask.unsqueeze(-1), 0.0)
        visual_hidden = visual_hidden.masked_fill(vis_mask_bool.unsqueeze(-1), 0.0)

        return (
            self.output_proj(visual_hidden.to(torch.float32)),
            self.output_proj(cond_hidden.to(torch.float32)),
            cond_mask,
            prefix_kv,
            n_prefix,
        )

    def decode_action(self, action_embeds, prefix_kv, n_prefix, prefix_pad_mask=None):
        """Run action tokens through decoder using prefix K/V cache."""
        return self.text_backbone.forward_action_with_prefix(
            action_embeds=action_embeds,
            prefix_kv=prefix_kv,
            n_prefix=n_prefix,
            prefix_pad_mask=prefix_pad_mask,
        )
