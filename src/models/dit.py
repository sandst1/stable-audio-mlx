import math
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DiTConfig:
    io_channels: int = 64
    embed_dim: int = 1024  # stable-audio-open-small uses 1024 (not 1536)
    depth: int = 16  # stable-audio-open-small has 16 blocks (not 24)
    num_heads: int = 8  # stable-audio-open-small uses 8 heads (not 24)
    cond_token_dim: int = 768  # T5 output dim (cross-attention conditioning)
    global_cond_dim: int = 768  # Global conditioning (seconds_start + seconds_total)
    project_cond_tokens: bool = True  # Project T5 tokens (768) to embed_dim (1024)
    transformer_type: str = "continuous_transformer"
    global_cond_type: str = "prepend"  # Prepend global cond as token, not adaLN
    patch_size: int = 1
    timestep_features_dim: int = 256


class FourierFeatures(nn.Module):
    """Learnable Fourier features for timestep embedding."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features % 2 == 0
        # Weight shape: (out_features // 2, in_features) = (128, 1)
        self.weight = mx.random.normal((out_features // 2, in_features))

    def __call__(self, x):
        # x: (B, 1) or (B,)
        if x.ndim == 1:
            x = x[:, None]
        
        f = 2 * math.pi * (x @ self.weight.T)
        return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # inv_freq: (dim,) - one freq per dimension
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        t = mx.arange(seq_len, dtype=mx.float32)
        # freqs: (seq_len, dim/2)
        freqs = t[:, None] * self.inv_freq[None, :]
        # Concatenate to get (seq_len, dim)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    # x: (..., dim)
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, freqs):
    """Apply rotary position embeddings to x (partial rotary).
    
    Only applies to the first rot_dim dimensions, leaving the rest unchanged.
    """
    # x: (B, H, L, D)
    # freqs: (L, rot_dim)
    rot_dim = freqs.shape[-1]
    head_dim = x.shape[-1]
    
    # Split x into rotated and unrotated parts
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    
    # Compute cos/sin
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    
    # Add batch and head dims: (L, rot_dim) -> (1, 1, L, rot_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
    # Apply rotation to the rotated part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    
    # Concatenate back
    if x_pass.shape[-1] > 0:
        return mx.concatenate([x_rot, x_pass], axis=-1)
    return x_rot


class GLU(nn.Module):
    """Gated Linear Unit with SiLU activation."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x):
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GLU."""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        # Using nested structure to match PyTorch: ff.ff.0.proj and ff.ff.2
        # Actually, we need to match the weight keys exactly
        # Weight keys: ff.ff.0.proj.weight, ff.ff.2.weight
        # So we need: self.ff = Sequential(GLU, ..., Linear)
        self.ff = nn.Sequential(
            GLU(dim, inner_dim),      # ff.ff.0 - the GLU has ff.ff.0.proj
            nn.Identity(),             # ff.ff.1 placeholder
            nn.Linear(inner_dim, dim)  # ff.ff.2
        )

    def __call__(self, x):
        return self.ff(x)


class SelfAttention(nn.Module):
    """Self-attention layer."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        
        # QK normalization for stability
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def __call__(self, x, rotary_freqs=None):
        B, L, D = x.shape
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        # Reshape: (B, L, D) -> (B, H, L, HeadDim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        if rotary_freqs is not None:
            q = apply_rotary_pos_emb(q, rotary_freqs)
            k = apply_rotary_pos_emb(k, rotary_freqs)
             
        # Attention
        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v
        
        # Reshape back: (B, H, L, HeadDim) -> (B, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention with grouped query attention support."""
    def __init__(self, dim: int, cond_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_heads = cond_dim // self.head_dim  # For GQA
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        # to_kv produces cond_dim * 2 (not dim * 2)
        self.to_kv = nn.Linear(cond_dim, cond_dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        
        # QK normalization for stability
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def __call__(self, x, cond):
        B, L, D = x.shape
        _, S, _ = cond.shape
        
        q = self.to_q(x)
        kv = self.to_kv(cond)
        k, v = mx.split(kv, 2, axis=-1)
        
        # q: (B, L, dim) -> (B, num_heads, L, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # k, v: (B, S, cond_dim) -> (B, kv_heads, S, head_dim)
        k = k.reshape(B, S, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # GQA: repeat k, v heads to match q heads
        if self.num_heads != self.kv_heads:
            repeats = self.num_heads // self.kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        
        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out(out)


class DiTBlock(nn.Module):
    """Transformer block for DiT (standard pre-norm transformer, no adaLN)."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.self_attn = SelfAttention(config.embed_dim, config.num_heads)
        
        self.cross_attend = config.cond_token_dim > 0
        if self.cross_attend:
            self.cross_attend_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
            # Use projected cond dimension if project_cond_tokens is True
            cond_dim = config.embed_dim if config.project_cond_tokens else config.cond_token_dim
            self.cross_attn = CrossAttention(config.embed_dim, cond_dim, config.num_heads)
        
        self.ff_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ff = FeedForward(config.embed_dim)

    def __call__(self, x, cond=None, rotary_freqs=None):
        # Standard pre-norm transformer block
        # Self Attention
        x = x + self.self_attn(self.pre_norm(x), rotary_freqs)
        
        # Cross Attention (if conditioning provided)
        if self.cross_attend and cond is not None:
            x = x + self.cross_attn(self.cross_attend_norm(x), cond)
        
        # Feed Forward
        x = x + self.ff(self.ff_norm(x))
        
        return x


class StableAudioDiT(nn.Module):
    """Stable Audio DiT model."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Timestep embedding
        self.timestep_features = FourierFeatures(1, config.timestep_features_dim)
        
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(config.timestep_features_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Global conditioning
        if config.global_cond_dim > 0:
            self.to_global_embed = nn.Sequential(
                nn.Linear(config.global_cond_dim, config.embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            )
        
        # Cross-attention conditioning
        if config.cond_token_dim > 0:
            cond_embed_dim = config.cond_token_dim if not config.project_cond_tokens else config.embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(config.cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        
        # Pre/post processing convolutions (residual)
        self.preprocess_conv = nn.Conv1d(config.io_channels, config.io_channels, kernel_size=1, bias=False)
        self.postprocess_conv = nn.Conv1d(config.io_channels, config.io_channels, kernel_size=1, bias=False)
        
        # Transformer input/output projections
        dim_in = config.io_channels * config.patch_size
        dim_out = config.io_channels * config.patch_size
        
        # These are called transformer.project_in/out in weights
        self.transformer = TransformerCore(dim_in, config.embed_dim, dim_out, config.num_heads)
        
        # Transformer blocks - must be explicit attributes for MLX to discover them
        # stable-audio-open-small has 16 blocks (0-15)
        self.block_0 = DiTBlock(config)
        self.block_1 = DiTBlock(config)
        self.block_2 = DiTBlock(config)
        self.block_3 = DiTBlock(config)
        self.block_4 = DiTBlock(config)
        self.block_5 = DiTBlock(config)
        self.block_6 = DiTBlock(config)
        self.block_7 = DiTBlock(config)
        self.block_8 = DiTBlock(config)
        self.block_9 = DiTBlock(config)
        self.block_10 = DiTBlock(config)
        self.block_11 = DiTBlock(config)
        self.block_12 = DiTBlock(config)
        self.block_13 = DiTBlock(config)
        self.block_14 = DiTBlock(config)
        self.block_15 = DiTBlock(config)
        self.num_blocks = config.depth
            
    def load_weights(self, weights):
        """Load weights with key mapping."""
        if isinstance(weights, dict):
            weights = list(weights.items())
        
        new_weights = {}
        
        for k, v in weights:
            
            nk = k  # new key
            
            # === Block weights ===
            if k.startswith("blocks."):
                # Convert blocks.0.xxx to block_0.xxx for MLX attribute discovery
                import re
                nk = re.sub(r'blocks\.(\d+)\.', r'block_\1.', k)
                
                # LayerNorm: gamma -> weight, beta -> bias
                if nk.endswith(".gamma"):
                    nk = nk[:-6] + ".weight"
                elif nk.endswith(".beta"):
                    nk = nk[:-5] + ".bias"
                # FeedForward: ff.ff.0.proj -> ff.ff.layers.0.proj, ff.ff.2 -> ff.ff.layers.2
                elif ".ff.ff." in nk:
                    nk = nk.replace(".ff.ff.", ".ff.ff.layers.")
            
            # === Sequential layers ===
            # to_timestep_embed.0 -> to_timestep_embed.layers.0
            elif k.startswith("to_timestep_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_timestep_embed.layers.{'.'.join(parts[1:])}"
            
            # to_global_embed.0 -> to_global_embed.layers.0
            elif k.startswith("to_global_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_global_embed.layers.{'.'.join(parts[1:])}"
            
            # to_cond_embed.0 -> to_cond_embed.layers.0
            elif k.startswith("to_cond_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_cond_embed.layers.{'.'.join(parts[1:])}"
            
            # transformer.project_in -> transformer.project_in
            # transformer.project_out -> transformer.project_out
            # (these should already match)
            
            new_weights[nk] = v
        
        # Debug: Print how many block weights we're loading
        block_weight_count = len([k for k in new_weights.keys() if 'block_' in k])
        print(f"  DiT: Loading {len(new_weights)} weights ({block_weight_count} block weights)")
        
        super().load_weights(list(new_weights.items()))

    def __call__(self, x, t, cross_attn_cond=None, global_embed=None, **kwargs):
        """Forward pass.
        
        Args:
            x: Input latents (B, C, T)
            t: Timesteps (B,)
            cross_attn_cond: Cross-attention conditioning (B, S, cond_dim)
            global_embed: Global conditioning (B, global_cond_dim)
        """
        B, C, T = x.shape
        
        # Preprocess conv (residual)
        # MLX Conv1d expects (B, T, C)
        x_t = x.transpose(0, 2, 1)  # (B, T, C)
        x_t = x_t + self.preprocess_conv(x_t)
        
        # Timestep embedding
        t_embed = self.to_timestep_embed(self.timestep_features(t))  # (B, embed_dim)
        
        # Global conditioning
        if global_embed is not None and hasattr(self, 'to_global_embed'):
            global_embed = self.to_global_embed(global_embed)  # (B, embed_dim)
            global_cond = global_embed + t_embed
        else:
            global_cond = t_embed
        
        # Cross-attention conditioning (project through to_cond_embed)
        if cross_attn_cond is not None and hasattr(self, 'to_cond_embed'):
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        # Project input
        x_t = self.transformer.project_in(x_t)  # (B, T, embed_dim)
        
        # PREPEND the global conditioning as a token
        # global_cond: (B, embed_dim) -> (B, 1, embed_dim)
        prepend_embed = global_cond[:, None, :]  # (B, 1, embed_dim)
        x_t = mx.concatenate([prepend_embed, x_t], axis=1)  # (B, 1+T, embed_dim)
        
        # Rotary embeddings for the full sequence (prepend + original)
        rotary_freqs = self.transformer.rotary_pos_emb(T + 1)
        
        # Transformer blocks
        for i in range(self.num_blocks):
            block = getattr(self, f'block_{i}')
            x_t = block(x_t, cond=cross_attn_cond, rotary_freqs=rotary_freqs)
        
        # Remove the prepended token before projection
        x_t = x_t[:, 1:, :]  # (B, T, embed_dim)
        
        # Project output
        x_t = self.transformer.project_out(x_t)  # (B, T, C)
        
        # Postprocess conv (residual)
        x_t = x_t + self.postprocess_conv(x_t)
        
        # Transpose back to (B, C, T)
        x = x_t.transpose(0, 2, 1)
        
        return x


class TransformerCore(nn.Module):
    """Contains transformer projections and rotary embeddings."""
    def __init__(self, dim_in: int, dim: int, dim_out: int, num_heads: int = 24):
        super().__init__()
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        # Rotary dim = max(head_dim // 2, 32), head_dim = dim // num_heads
        head_dim = dim // num_heads
        rotary_dim = max(head_dim // 2, 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim)
