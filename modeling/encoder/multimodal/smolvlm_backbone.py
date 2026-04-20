"""
SmolVLM backbone with mixed 3D/1D rotary position encoding and prefix KV caching.

Prefix token ordering: [visual (3D RoPE), language (1D RoPE), state (3D RoPE)].
Action tokens (decode phase): 1D RoPE at positions [n_prefix, n_prefix + n_action).
Full bidirectional attention throughout — no causal masking.

The prefix K/V tensors are cached in a DynamicCache after a single backbone
forward pass. Each denoising step seeds a fresh DynamicCache whose per-layer
``DynamicLayer`` objects carry references to the prefix's key/value tensors —
no tensor copies. The action pass extends those per-layer ``keys``/``values``
via ``torch.cat``, which reassigns to new tensors on the action cache's
layers and therefore does not mutate the prefix cache.

Requires transformers >= 4.52 — the seeding path expects ``DynamicCache``
to expose ``layers: list[DynamicLayer]`` with each layer carrying ``keys``
and ``values`` tensors. Earlier versions (< 4.50) exposed parallel
``key_cache`` / ``value_cache`` lists with a ``_seen_tokens`` counter and
need the pre-refactor seeding (see git history).
"""

import math

import torch
from torch import nn

from transformers.cache_utils import DynamicCache, DynamicLayer


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen ``nn.Linear``.

    ``y = base(x) + (alpha / rank) * lora_B(lora_A(x))``

    The base linear's parameters are frozen; ``lora_A`` and ``lora_B`` are the
    only trainable tensors. ``lora_B`` is zero-initialized so the wrapped
    module's output matches the base module at step 0.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora_to_text_model(
    text_model: nn.Module,
    rank: int,
    alpha: float,
    last_n_layers: int,
    targets=("q_proj", "v_proj"),
) -> int:
    """Wrap attention projections on the last ``last_n_layers`` Llama layers with LoRA.

    Returns the number of wrapped projections. No-ops (returns 0) when
    ``rank <= 0`` or ``last_n_layers <= 0`` so callers can always invoke this
    unconditionally.
    """
    if rank <= 0 or last_n_layers <= 0:
        return 0
    layers = text_model.layers
    start = max(0, len(layers) - int(last_n_layers))
    wrapped = 0
    for layer in layers[start:]:
        attn = layer.self_attn
        for name in targets:
            base = getattr(attn, name, None)
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear at self_attn.{name}, got {type(base).__name__}"
                )
            setattr(attn, name, LoRALinear(base, rank=rank, alpha=alpha))
            wrapped += 1
    return wrapped


class MixedModalityRotaryEmbedding(nn.Module):
    """
    Builds per-token rotary (cos, sin) embeddings for a mixed-modality sequence.

      visual tokens  → 3D RoPE from point-cloud xyz
      language tokens → 1D RoPE from sequential positions
      state tokens   → 3D RoPE from end-effector xyz
      action tokens  → 1D RoPE from sequential positions offset by n_prefix
    """

    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float = 1.0,
                 xyz_rope_scale: float = 1.0):
        super().__init__()
        self.register_buffer(
            "inv_freq", inv_freq.detach().clone().to(torch.float32), persistent=False
        )
        self.attention_scaling = attention_scaling
        # Multiplied into xyz before 3D RoPE. Default 1.0 preserves original
        # behavior. Llama's inv_freq was tuned for integer positions (~0-48
        # for language, ~0-2k for long contexts) — meter-scale xyz (~0-2 m)
        # leaves most frequency slots at near-zero rotation, so scaling xyz
        # up brings visual/state positional resolution closer to language.
        self.xyz_rope_scale = float(xyz_rope_scale)

    def _build_1d(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(dtype), (emb.sin() * self.attention_scaling).to(dtype)

    def _build_3d(self, xyz: torch.Tensor, dtype: torch.dtype):
        half_dim = self.inv_freq.shape[0]
        base = half_dim // 3
        splits = [base, base, half_dim - 2 * base]
        chunks = torch.split(self.inv_freq, splits)
        xyz_scaled = xyz.to(torch.float32) * self.xyz_rope_scale
        parts = []
        for coord_idx, chunk in enumerate(chunks):
            if chunk.numel() == 0:
                continue
            parts.append(xyz_scaled[..., coord_idx : coord_idx + 1] * chunk.view(1, 1, -1))
        freqs = torch.cat(parts, dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(dtype), (emb.sin() * self.attention_scaling).to(dtype)

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


class MixedPositionTextBackbone(nn.Module):
    """Wraps a Llama-style text model with mixed-modality RoPE and KV-cache plumbing."""

    def __init__(self, text_model: nn.Module, xyz_rope_scale: float = 1.0):
        super().__init__()
        self.text_model = text_model
        self.config = text_model.config

        rotary_emb = getattr(text_model, "rotary_emb", None)
        if rotary_emb is None or not hasattr(rotary_emb, "inv_freq"):
            raise RuntimeError(
                f"MixedPositionTextBackbone requires a Llama-style text model with "
                f"`rotary_emb.inv_freq`, got {type(text_model).__name__}."
            )
        self.mixed_rope = MixedModalityRotaryEmbedding(
            rotary_emb.inv_freq,
            attention_scaling=getattr(rotary_emb, "attention_scaling", 1.0),
            xyz_rope_scale=xyz_rope_scale,
        )

    @staticmethod
    def _seed_action_cache(prefix_kv: DynamicCache) -> DynamicCache:
        """Clone a DynamicCache at layer granularity, sharing K/V tensor storage.

        Each new ``DynamicLayer`` references the prefix's ``keys`` / ``values``
        tensors directly. When a transformer layer later calls ``update`` on
        the action cache, ``DynamicLayer.update`` reassigns its ``keys`` /
        ``values`` attributes to the ``torch.cat`` result — so the prefix
        cache's tensors are never mutated in place. This avoids the tensor
        copies that ``from_legacy_cache`` round-tripping would incur on every
        denoising step.
        """
        action_cache = DynamicCache()
        for prefix_layer in prefix_kv.layers:
            new_layer = DynamicLayer()
            new_layer.keys = prefix_layer.keys
            new_layer.values = prefix_layer.values
            new_layer.dtype = prefix_layer.keys.dtype
            new_layer.device = prefix_layer.keys.device
            action_cache.layers.append(new_layer)
        return action_cache

    @staticmethod
    def _full_attn_mask(inputs_embeds, attention_mask):
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

    def forward_prefix(self, inputs_embeds, visual_xyz, n_visual, n_lang,
                       state_xyz=None, attention_mask=None, use_cache=True):
        B, S, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        cache_position = torch.arange(S, device=device)
        position_ids = cache_position.unsqueeze(0).expand(B, -1)
        attention_mask_4d = self._full_attn_mask(inputs_embeds, attention_mask)

        lang_positions = position_ids[:, n_visual : n_visual + n_lang]
        position_embeddings = self.mixed_rope.forward_prefix(
            visual_xyz=visual_xyz,
            lang_positions=lang_positions,
            state_xyz=state_xyz,
            dtype=dtype,
        )
        past_key_values = DynamicCache() if use_cache else None
        return self._run_layers(
            inputs_embeds, attention_mask_4d, position_ids,
            cache_position, position_embeddings, past_key_values, use_cache,
        )

    def forward_action_with_prefix(self, action_embeds, prefix_kv, n_prefix,
                                   prefix_pad_mask=None):
        """Decode action tokens using prefix K/V as read-only context."""
        B, n_action, _ = action_embeds.shape
        device = action_embeds.device
        dtype = action_embeds.dtype

        action_cache = self._seed_action_cache(prefix_kv)

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


class SmolVLMPrefixKVBackbone(nn.Module):
    """
    Fuses [visual, lang, state] through the SmolVLM decoder once (encode_prefix),
    then decodes action tokens using the cached K/V (decode_action).
    """

    def __init__(self, text_model: nn.Module, output_dim: int,
                 state_dim: int = 0, num_state_tokens: int = 1,
                 xyz_rope_scale: float = 1.0):
        super().__init__()
        self.text_model = text_model
        self.hidden_size = text_model.config.hidden_size
        self.output_proj = nn.Linear(self.hidden_size, output_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.state_dim = int(state_dim)
        if self.state_dim > 0 and self.num_state_tokens > 0:
            self.state_proj = nn.Linear(self.state_dim, self.hidden_size * self.num_state_tokens)
        else:
            self.state_proj = None
        self.text_backbone = MixedPositionTextBackbone(
            text_model=text_model, xyz_rope_scale=xyz_rope_scale,
        )

    def _encode_state(self, state, dtype):
        if self.state_proj is None or state is None:
            return None
        tokens = self.state_proj(state.to(dtype=torch.float32))
        return tokens.view(state.shape[0], self.num_state_tokens, self.hidden_size).to(dtype)

    def encode_prefix(
        self, *, input_ids, lang_attention_mask, visual_tokens, visual_xyz,
        visual_attention_mask=None, state=None, state_xyz=None,
    ):
        """Run prefix [visual, lang, state] through backbone with KV caching."""
        device = visual_tokens.device
        model_dtype = next(self.text_model.parameters()).dtype

        visual_tokens = visual_tokens.to(device=device, dtype=model_dtype)
        B, n_visual, _ = visual_tokens.shape
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones((B, n_visual), dtype=torch.long, device=device)
        else:
            visual_attention_mask = visual_attention_mask.to(dtype=torch.long, device=device)

        lang_embeds = self.text_model.get_input_embeddings()(input_ids.to(device)).to(dtype=model_dtype)
        lang_mask = lang_attention_mask.to(dtype=torch.long, device=device)
        n_lang = lang_embeds.shape[1]

        state_tokens = self._encode_state(state, dtype=model_dtype)
        if state_tokens is not None:
            n_state = state_tokens.shape[1]
            state_mask = torch.ones((B, n_state), dtype=torch.long, device=device)
        else:
            n_state = 0
            state_mask = None

        parts = [visual_tokens, lang_embeds]
        mask_parts = [visual_attention_mask, lang_mask]
        if state_tokens is not None:
            parts.append(state_tokens)
            mask_parts.append(state_mask)
        full_inputs = torch.cat(parts, dim=1)
        full_mask = torch.cat(mask_parts, dim=1)

        if state_tokens is not None:
            if state_xyz is None:
                state_xyz_rope = torch.zeros((B, n_state, 3), dtype=torch.float32, device=device)
            else:
                state_xyz_rope = state_xyz.to(device=device, dtype=torch.float32)
                if state_xyz_rope.ndim == 2:
                    state_xyz_rope = state_xyz_rope.unsqueeze(1)
                if state_xyz_rope.shape[1] == 1 and n_state > 1:
                    state_xyz_rope = state_xyz_rope.expand(-1, n_state, -1)
        else:
            state_xyz_rope = None

        hidden_states, prefix_kv = self.text_backbone.forward_prefix(
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
        return self.text_backbone.forward_action_with_prefix(
            action_embeds=action_embeds,
            prefix_kv=prefix_kv,
            n_prefix=n_prefix,
            prefix_pad_mask=prefix_pad_mask,
        )
