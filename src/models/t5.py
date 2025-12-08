import math
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class T5Config:
    d_model: int = 768
    num_heads: int = 12
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    vocab_size: int = 32128
    dropout_rate: float = 0.0  # Disabled for inference (was 0.1)
    layer_norm_epsilon: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x):
        # T5 LayerNorm: x * weight * rsqrt(mean(x^2) + eps)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight

class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # Activation is typically ReLU for t5-base
        
    def __call__(self, x):
        return self.wo(nn.relu(self.wi(x)))

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = False
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.d_kv * self.num_heads # usually equals d_model but T5 allows difference
        
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = mx.arange(query_length, dtype=mx.int32)[:, None]
        memory_position = mx.arange(key_length, dtype=mx.int32)[None, :]
        relative_position = memory_position - context_position  # shape (q_len, k_len)
        
        rp_bucket = self._relative_position_bucket(
            relative_position, 
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        values = self.relative_attention_bias(rp_bucket) # (q_len, k_len, num_heads)
        values = mx.transpose(values, (2, 0, 1)) # (num_heads, q_len, k_len)
        return values.astype(mx.float32) # T5 attention bias is added to logits

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # Implementation adapted from HuggingFace Transformers T5
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(mx.int32) * num_buckets
            n = mx.abs(n)
        else:
            n = mx.maximum(n, 0)
            
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            mx.log(n.astype(mx.float32) / max_exact) / 
            math.log(max_distance / max_exact) * 
            (num_buckets - max_exact)
        ).astype(mx.int32)
        val_if_large = mx.minimum(val_if_large, num_buckets - 1)
        
        return mx.where(is_small, n, val_if_large) + ret

    def __call__(self, hidden_states, mask=None, position_bias=None):
        B, L, _ = hidden_states.shape
        
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        
        # Reshape to (B, L, H, D) -> (B, H, L, D)
        q = q.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) # (B, H, L, L)
        
        if self.has_relative_attention_bias:
            if position_bias is None:
                position_bias = self.compute_bias(L, L) # (H, L, L)
                position_bias = mx.expand_dims(position_bias, 0) # (1, H, L, L)
            scores += position_bias
        elif position_bias is not None:
             scores += position_bias

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        out = attn_weights @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.inner_dim)
        return self.o(out), position_bias

class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        # Self Attention sublayer
        self.self_attn_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # FF sublayer
        self.ff_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.ff = T5DenseActDense(config)

    def __call__(self, hidden_states, mask=None, position_bias=None):
        # Attention
        normed_hidden_states = self.self_attn_norm(hidden_states)
        attn_output, position_bias = self.self_attn(normed_hidden_states, mask, position_bias)
        hidden_states = hidden_states + attn_output
        
        # FF
        normed_hidden_states = self.ff_norm(hidden_states)
        ff_output = self.ff(normed_hidden_states)
        hidden_states = hidden_states + ff_output
        
        return hidden_states, position_bias

class T5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens=None):
        super().__init__()
        # Accept shared embedding or create own
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Create blocks as Python list (like DiT does)
        # MLX will load weights into the list via load_weights()
        self.block = [
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ]
        
        self.final_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)

    def __call__(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        mask = None
        if attention_mask is not None:
            # attention_mask: (B, L) where 1 = keep, 0 = pad
            # Convert to additive mask for attention logits
            if attention_mask.ndim == 2:
                mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
            else:
                mask = attention_mask
        
        position_bias = None
        for layer in self.block:
            hidden_states, position_bias = layer(hidden_states, mask, position_bias)
            
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Zero out padding tokens in the output (important for matching TFLite behavior)
        if attention_mask is not None:
            # Expand attention_mask to (B, L, 1) and multiply
            mask_expanded = attention_mask[:, :, None].astype(mx.float32)
            hidden_states = hidden_states * mask_expanded
        
        return hidden_states

class T5EncoderModel(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # Create shared embedding first
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # Pass it to encoder so they share the same embedding layer
        self.encoder = T5Stack(config, embed_tokens=self.shared)

    def __call__(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)
