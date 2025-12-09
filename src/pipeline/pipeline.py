import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
import os
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.vae import AutoencoderOobleck, OobleckConfig
from src.models.dit import StableAudioDiT, DiTConfig
from src.models.t5 import T5EncoderModel, T5Config
from src.pipeline.conditioners import Conditioners

def get_rf_schedule(steps, sigma_max=1.0):
    """Get the timestep schedule for rectified flow sampling.
    
    Uses logSNR-based schedule as in the reference implementation.
    Reference: stable_audio_tools/inference/sampling.py sample_rf()
    """
    # Compute logsnr_max based on sigma_max (matches reference line 441)
    if sigma_max < 1.0:
        logsnr_max = math.log((1 - sigma_max) / sigma_max + 1e-6)
    else:
        logsnr_max = -6.0
    
    # Linear schedule in logSNR space from logsnr_max to 2 (matches reference line 443)
    logsnr = np.linspace(logsnr_max, 2, steps + 1)
    
    # Convert to timesteps via sigmoid (matches reference line 445)
    # torch.sigmoid(-logsnr) is equivalent to 1 / (1 + exp(logsnr))
    t = 1.0 / (1.0 + np.exp(logsnr))
    
    # Clamp endpoints (matches reference lines 447-448)
    t[0] = sigma_max
    t[-1] = 0.0
    
    return mx.array(t.astype(np.float32))


def sample_rk4(model_fn, x, timesteps, cond_tokens, uncond_tokens, global_cond, cfg_scale, steps):
    """RK4 (Runge-Kutta 4th order) sampler for rectified flow.
    
    This is a much more accurate ODE solver than Euler, requiring 4 model 
    evaluations per step but providing 4th order accuracy.
    """
    
    def get_velocity(latents, t_value):
        """Get velocity prediction with optional CFG."""
        t_in = mx.full((1,), t_value)
        
        if cfg_scale > 1.0 and uncond_tokens is not None:
            # CFG: Run model twice (batched)
            latents_batch = mx.concatenate([latents, latents], axis=0)
            t_batch = mx.concatenate([t_in, t_in], axis=0)
            cond_batch = mx.concatenate([cond_tokens, uncond_tokens], axis=0)
            global_batch = mx.concatenate([global_cond, global_cond], axis=0)
            
            v_batch = model_fn(latents_batch, t_batch, cond_batch, global_batch)
            v_cond, v_uncond = mx.split(v_batch, 2, axis=0)
            
            # CFG formula
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model_fn(latents, t_in, cond_tokens, global_cond)
        
        return v
    
    for i in tqdm(range(steps), desc="RK4 Sampling"):
        t_curr = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt = t_next - t_curr  # Negative (going from 1 to 0)
        
        # RK4 stages
        k1 = get_velocity(x, t_curr)
        mx.eval(k1)
        
        k2 = get_velocity(x + 0.5 * dt * k1, t_curr + 0.5 * dt)
        mx.eval(k2)
        
        k3 = get_velocity(x + 0.5 * dt * k2, t_curr + 0.5 * dt)
        mx.eval(k3)
        
        k4 = get_velocity(x + dt * k3, t_next)
        mx.eval(k4)
        
        # RK4 update
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        mx.eval(x)
    
    return x


def sample_euler(model_fn, x, timesteps, cond_tokens, uncond_tokens, global_cond, cfg_scale, steps):
    """Euler sampler for rectified flow (simple 1st order).
    
    Fast but less accurate than RK4, especially with few steps.
    """
    
    for i in tqdm(range(steps), desc="Euler Sampling"):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr
        
        t_in = mx.full((1,), t_curr)
        
        if cfg_scale > 1.0 and uncond_tokens is not None:
            latents_batch = mx.concatenate([x, x], axis=0)
            t_batch = mx.concatenate([t_in, t_in], axis=0)
            cond_batch = mx.concatenate([cond_tokens, uncond_tokens], axis=0)
            global_batch = mx.concatenate([global_cond, global_cond], axis=0)
            
            v_batch = model_fn(latents_batch, t_batch, cond_batch, global_batch)
            v_cond, v_uncond = mx.split(v_batch, 2, axis=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model_fn(x, t_in, cond_tokens, global_cond)
        
        x = x + v * dt
        mx.eval(x)
    
    return x

class StableAudioPipeline:
    def __init__(self, vae, dit, t5, conditioners, tokenizer):
        self.vae = vae
        self.dit = dit
        self.t5 = t5
        self.conditioners = conditioners
        self.tokenizer = tokenizer
        
    @classmethod
    def from_pretrained(cls, weights_path):
        """Load pipeline from weights.
        
        Args:
            weights_path: Path to MLX weights NPZ file
        """
        print("Loading MLX weights...")
        data = np.load(weights_path, allow_pickle=True)
        
        # VAE
        print("Initializing VAE...")
        vae_config = OobleckConfig()
        vae = AutoencoderOobleck(vae_config)
        vae_weights = {k: mx.array(v) for k, v in data['vae'].item().items()}
        vae.load_weights(list(vae_weights.items()))
        vae.eval()  # Disable dropout and other training-specific layers
        
        # DiT
        print("Initializing DiT...")
        dit_config = DiTConfig()
        dit = StableAudioDiT(dit_config)
        dit_weights = {k: mx.array(v) for k, v in data['dit'].item().items()}
        dit.load_weights(list(dit_weights.items()))
        dit.eval()  # Disable dropout and other training-specific layers
        
        # T5 - Load from local Stable Audio weights, not HuggingFace t5-base
        print("Loading T5 from Stable Audio weights...")
        t5_config = T5Config()
        t5 = T5EncoderModel(t5_config)
        
        # Load tokenizer from local weights directory or download from HuggingFace
        weights_dir = os.path.dirname(weights_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(weights_dir)
            print(f"Loaded tokenizer from {weights_dir}")
        except:
            print(f"Tokenizer not found in {weights_dir}, downloading from stable-audio-open-1.0...")
            # Both stable-audio-open-1.0 and small use the same T5-based tokenizer
            tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="tokenizer")
            # Save tokenizer to model directory for future use
            tokenizer.save_pretrained(weights_dir)
            print(f"Saved tokenizer to {weights_dir}")
        
        # Load T5 weights from local Stable Audio model
        from safetensors import safe_open
        
        t5_weights_path = os.path.join(weights_dir, "t5.safetensors")
        if not os.path.exists(t5_weights_path):
            print(f"Warning: T5 weights not found at {t5_weights_path}")
            print("Run src/conversion/convert.py first to extract T5 weights.")
        else:
            try:
                print(f"Loading T5 weights from {t5_weights_path}")
                
                t5_weights = {}
                with safe_open(t5_weights_path, framework="np", device="cpu") as f:
                    for k in f.keys():
                        t5_weights[k] = mx.array(f.get_tensor(k))
                
                # Map T5 keys to our MLX model keys
                mapped_weights = {}
                
                for k, v in t5_weights.items():
                    nk = k  # new key
                    
                    # Shared embedding
                    if k == "shared.weight":
                        mapped_weights["shared.weight"] = v
                        mapped_weights["encoder.embed_tokens.weight"] = v
                        continue
                    
                    # Final layer norm
                    if k == "encoder.final_layer_norm.weight":
                        mapped_weights["encoder.final_layer_norm.weight"] = v
                        continue
                    
                    # Block mappings  
                    if k.startswith("encoder.block."):
                        # encoder.block.X.layer.0.SelfAttention.Y -> encoder.block.X.self_attn.Y
                        # encoder.block.X.layer.0.layer_norm -> encoder.block.X.self_attn_norm
                        # encoder.block.X.layer.1.DenseReluDense.Y -> encoder.block.X.ff.Y
                        # encoder.block.X.layer.1.layer_norm -> encoder.block.X.ff_norm
                        
                        # Keep block.X format (Python list indexing in MLX)
                        nk = k.replace(".layer.0.SelfAttention.", ".self_attn.")
                        nk = nk.replace(".layer.0.layer_norm.", ".self_attn_norm.")
                        nk = nk.replace(".layer.1.DenseReluDense.", ".ff.")
                        nk = nk.replace(".layer.1.layer_norm.", ".ff_norm.")
                        
                        mapped_weights[nk] = v
                        continue
                
                # Load the mapped weights
                t5.load_weights(list(mapped_weights.items()))
                print(f"T5 weights loaded successfully! ({len(mapped_weights)} parameters)")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to load T5 weights: {e}")
                print("Using random weights for T5.")
        
        # Set T5 to eval mode to disable dropout
        t5.eval()
        
        # Conditioners - Initialize after T5 and tokenizer are loaded
        print("Initializing Conditioners...")
        conditioners = Conditioners(t5, tokenizer)
        cond_weights = {k: v for k, v in data['cond'].item().items()}
        conditioners.load_weights(cond_weights)
             
        return cls(vae, dit, t5, conditioners, tokenizer)

    def generate(self, prompt, negative_prompt="", steps=100, cfg_scale=7.0, sigma_max=1.0, seconds_total=30, seed=None, sampler="rk4"):
        """Generate audio using rectified flow sampling.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt for CFG
            steps: Number of sampling steps (default 100)
            cfg_scale: Classifier-free guidance scale (default 7.0)
            sigma_max: Maximum sigma/timestep (default 1.0 for rectified flow)
            seconds_total: Total duration in seconds
            seed: Random seed
            sampler: Sampling method - "rk4" (recommended, 4th order) or "euler" (1st order)
        """
        if seed is not None:
            mx.random.seed(seed)
        
        # 1. Get conditioning
        print("Encoding text and time conditioning...")
        cond_tokens, global_cond = self.conditioners(prompt, seconds_total)
        # cond_tokens: (1, 65, 768) - T5 tokens (64) + time embedding (1)
        # global_cond: (1, 768) - time embedding
        
        # Encode negative prompt for CFG
        if cfg_scale > 1.0:
            uncond_tokens, _ = self.conditioners(negative_prompt, seconds_total)
        else:
            uncond_tokens = None
        
        # 3. Initialize Noise (pure noise for rectified flow, no scaling)
        latent_rate = 44100 / 2048  # ~21.5 Hz
        latent_length = int(seconds_total * latent_rate)
        
        latents = mx.random.normal((1, 64, latent_length))
        
        # 4. Get timestep schedule for rectified flow
        timesteps = get_rf_schedule(steps, sigma_max)
        
        # 5. Sampling Loop
        print(f"Sampling with {sampler.upper()} sampler, CFG scale {cfg_scale}...")
        
        if sampler == "rk4":
            # RK4 - 4th order accurate, recommended for quality
            latents = sample_rk4(
                self.dit, latents, timesteps, 
                cond_tokens, uncond_tokens, global_cond, 
                cfg_scale, steps
            )
        elif sampler == "euler":
            # Euler - 1st order, faster but less accurate
            latents = sample_euler(
                self.dit, latents, timesteps,
                cond_tokens, uncond_tokens, global_cond,
                cfg_scale, steps
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Use 'rk4' or 'euler'.")
            
        # 6. Decode
        print("Decoding...")
        # Transpose latents from (B, C, T) to (B, T, C) for MLX Conv1d
        latents = latents.transpose(0, 2, 1)
        mx.eval(latents)
        
        audio = self.vae.decode(latents)
        mx.eval(audio)  # Force evaluation
        
        # VAE outputs (B, T, C) where C=2 for stereo
        # Transpose to (B, C, T) for standard audio format (channel-first)
        audio = audio.transpose(0, 2, 1)
        mx.eval(audio)
        
        # Note: Decoder has final_tanh=False, so output is not bounded
        # The model is trained to produce output in a reasonable range
        # Don't clip the output
        mx.eval(audio)
        
        return audio

