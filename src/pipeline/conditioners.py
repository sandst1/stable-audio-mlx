import mlx.core as mx
import mlx.nn as nn
import numpy as np

class NumberEmbedder(nn.Module):
    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        # Structure inferred from weight shapes:
        # embedding.0.weights: (128,) - Fourier frequencies
        # embedding.1.weight: (768, 257) - Linear layer (257 = 256 + 1 for sin/cos + original)
        # embedding.1.bias: (768,)
        
        self.freqs = None
        self.linear_w = None 
        self.linear_b = None

    def load_weights(self, weights):
        # weights: dict of {key: array}
        # keys: "embedding.0.weights", "embedding.1.bias", "embedding.1.weight"
        
        self.freqs = mx.array(weights["embedding.0.weights"])  # (128,)
        self.linear_w = mx.array(weights["embedding.1.weight"])  # (768, 257)
        self.linear_b = mx.array(weights["embedding.1.bias"])  # (768,)
        
    def __call__(self, x):
        # x: (B,) or scalar
        if x.ndim == 0:
            x = x[None]
        if x.ndim == 1:
            x = x[:, None]  # (B, 1)
            
        # Fourier features
        # x: (B, 1), freqs: (128,)
        # proj = x * freqs -> broadcast -> (B, 128)
        # Apply 2*pi scaling as in original FourierFeatures
        proj = 2 * np.pi * x * self.freqs[None, :]  # (B, 128)
        
        # Sin/cos features: (B, 256)
        # Original uses [cos, sin]
        fourier = mx.concatenate([mx.cos(proj), mx.sin(proj)], axis=-1)
        
        # Concat original input: (B, 257)
        h = mx.concatenate([fourier, x], axis=-1)
        
        # Linear: (B, 257) @ (257, 768) + (768,) -> (B, 768)
        out = h @ self.linear_w.T + self.linear_b
        return out

class Conditioners(nn.Module):
    def __init__(self):
        super().__init__()
        self.seconds_start = NumberEmbedder(1, 768) # Dim guesses
        self.seconds_total = NumberEmbedder(1, 768)
        
    def load_weights(self, cond_state):
        # cond_state: dict from convert.py
        # Keys: conditioner.conditioners.seconds_total.embedder.embedding.0.weights
        
        # Split into start/total groups
        start_weights = {}
        total_weights = {}
        
        for k, v in cond_state.items():
            k_short = k.split("embedder.")[1] # embedding.0.weights...
            if "seconds_start" in k:
                start_weights[k_short] = v
            elif "seconds_total" in k:
                total_weights[k_short] = v
        
        # stable-audio-open-small only has seconds_total, not seconds_start
        if start_weights:
            self.seconds_start.load_weights(start_weights)
        if total_weights:
            self.seconds_total.load_weights(total_weights)
            # If no seconds_start, use seconds_total for both
            if not start_weights:
                self.seconds_start = self.seconds_total

    def __call__(self, seconds_start, seconds_total):
        # inputs are arrays or floats
        if isinstance(seconds_start, (float, int)):
            seconds_start = mx.array([seconds_start])
        if isinstance(seconds_total, (float, int)):
            seconds_total = mx.array([seconds_total])
            
        emb_start = self.seconds_start(seconds_start)  # (B, 768)
        emb_total = self.seconds_total(seconds_total)  # (B, 768)
        
        # stable-audio-open-small uses only seconds_total for global conditioning
        # (Not concatenated like the larger models)
        global_cond = emb_total  # (B, 768)
        
        # Cross-attention tokens: each as (B, 1, 768) to be concatenated with T5 output
        cross_attn_start = emb_start[:, None, :]  # (B, 1, 768)
        cross_attn_total = emb_total[:, None, :]  # (B, 1, 768)
        
        return global_cond, cross_attn_start, cross_attn_total
