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
    """MLX conditioners that match TFLite interface.
    
    Takes prompt and seconds_total, returns cross_attn and global_cond.
    Internally handles T5 text encoding and time embedding.
    """
    def __init__(self, t5_model, tokenizer):
        super().__init__()
        self.t5 = t5_model
        self.tokenizer = tokenizer
        self.seconds_total = NumberEmbedder(1, 768)
        
    def load_weights(self, cond_state):
        # cond_state: dict from convert.py
        # Keys: conditioner.conditioners.seconds_total.embedder.embedding.0.weights
        total_weights = {}
        
        for k, v in cond_state.items():
            k_short = k.split("embedder.")[1] # embedding.0.weights...
            if "seconds_total" in k:
                total_weights[k_short] = v
        
        # stable-audio-open-small only has seconds_total, not seconds_start
        if total_weights:
            self.seconds_total.load_weights(total_weights)

    def __call__(self, prompt, seconds_total):
        """Generate conditioning for the given prompt and duration.
        
        Args:
            prompt: Text prompt (str)
            seconds_total: Duration in seconds (float)
            
        Returns:
            cross_attn: (1, 65, 768) - T5 tokens (64) + time embedding (1)
            global_cond: (1, 768) - Time embedding for global conditioning
        """
        # 1. Tokenize and encode text with T5
        tokens = self.tokenizer(prompt, return_tensors="np", padding="max_length", 
                               max_length=128, truncation=True)
        input_ids = mx.array(tokens["input_ids"])
        attn_mask = mx.array(tokens["attention_mask"])
        
        t5_output = self.t5(input_ids, attn_mask)  # (1, 128, 768)
        t5_tokens = t5_output[:, :64, :]  # Extract first 64 tokens (1, 64, 768)
        
        # 2. Encode time parameter
        if isinstance(seconds_total, (float, int)):
            seconds_total = mx.array([seconds_total])
            
        time_emb = self.seconds_total(seconds_total)  # (1, 768)
        
        # 3. Build outputs (matching TFLite interface)
        # Global conditioning: time embedding only
        global_cond = time_emb  # (1, 768)
        
        # Cross-attention: concatenate T5 tokens + time embedding
        time_token = time_emb[:, None, :]  # (1, 1, 768)
        cross_attn = mx.concatenate([t5_tokens, time_token], axis=1)  # (1, 65, 768)
        
        return cross_attn, global_cond
