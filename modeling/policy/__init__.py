from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
from .denoise_actor_2d import DenoiseActor as DenoiseActor2D


def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    if model_type == 'denoise2d':  # standard 2DFA
        return DenoiseActor2D
    if model_type == 'smolvla_prefix_kv':  # SmolVLM backbone, prefix-KV cache
        # Lazy so non-smolvla runs (imitation-venv, transformers<4.52) don't
        # hit the DynamicLayer import in smolvlm_backbone.py.
        from .smolvla_prefix_kv_actor import SmolVLAPrefixKVActor
        return SmolVLAPrefixKVActor
    return None
