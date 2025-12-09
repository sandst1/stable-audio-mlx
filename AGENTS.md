# AGENTS.md - Project Overview for AI Assistants

## Project Description

**Stable Audio MLX** - Text-to-audio generation using Stability AI's stable-audio-open-small model, optimized for Apple Silicon (MLX framework).

⚠️ **Important**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

## Quick Start

1. **Convert model weights** (downloads automatically if needed):
   ```bash
   python src/conversion/convert.py
   ```

2. **Generate audio**:
   ```bash
   python generate.py --prompt "your description" --seconds 10 --steps 30 --cfg-scale 7.0 --seed 42 --output output.wav
   ```

## Project Structure

```
stable-audio-mlx/
├── generate.py                      # Main CLI for audio generation
├── model/                      # Model weights (auto-downloaded)
│   ├── model.safetensors      # Original PyTorch weights
│   ├── model_config.json      # Model configuration
│   ├── stable_audio_small.npz # Converted MLX weights (VAE, DiT, Conditioners)
│   └── t5.safetensors         # T5 text encoder weights
├── src/
│   ├── conversion/
│   │   └── convert.py         # PyTorch → MLX weight conversion
│   ├── models/
│   │   ├── vae.py             # VAE (Oobleck) encoder/decoder
│   │   ├── dit.py             # Diffusion Transformer (16 blocks)
│   │   └── t5.py              # T5 text encoder
│   └── pipeline/
│       ├── pipeline.py        # Main inference pipeline
│       └── conditioners.py    # Time/duration conditioning
└── requirements.txt           # Python dependencies
```

## Model Architecture (stable-audio-open-small)

### DiT (Diffusion Transformer)
- **Config**: `src/models/dit.py` - `DiTConfig` class
- **Key Parameters**:
  - `embed_dim = 1024` (not 1536 like stable-audio-open-1.0)
  - `depth = 16` blocks (not 24)
  - `num_heads = 8` (not 24)
  - `global_cond_dim = 768`
  - `cond_token_dim = 768` (T5 output)
  - `project_cond_tokens = True` (projects T5 to embed_dim)
- **Features**:
  - QK normalization in self/cross-attention (128 LayerNorm params)
  - Rotary position embeddings (partial, 32-64 dims)
  - Cross-attention with T5 conditioning
  - Prepended global conditioning token

### VAE (Autoencoder Oobleck)
- **Config**: `src/models/vae.py` - `OobleckConfig`
- **Compression**: 44.1kHz → ~21.5Hz latent rate (2048x downsampling)
- **Channels**: 64 latent channels
- **Architecture**: Snake activation, weight normalization

### T5 Text Encoder
- **Config**: `src/models/t5.py` - `T5Config`
- **Output**: 768-dim embeddings
- **Tokenizer**: Loaded from `stabilityai/stable-audio-open-1.0/tokenizer`

### Conditioners
- **Config**: `src/pipeline/conditioners.py`
- **Note**: stable-audio-open-small only has `seconds_total` conditioner
  - Uses same embedder for both `seconds_start` and `seconds_total`
  - Global conditioning: only `seconds_total` (768-dim), not concatenated

## Conversion Process

The `convert.py` script:
1. Downloads model from HuggingFace if not present
2. Splits weights into: VAE, DiT, Conditioners, T5
3. Handles key transformations:
   - `blocks.X.` → `block_X.` (MLX attribute discovery)
   - LayerNorm: `.gamma` → `.weight`, `.beta` → `.bias`
   - FeedForward: `.ff.ff.N` → `.ff.ff.layers.N`
   - Conv1d: transpose for MLX format `(Out, K, In)`
   - Weight normalization: fuse `weight_g` and `weight_v`
4. Saves to NPZ (VAE/DiT/Cond) and safetensors (T5)

## Sampling

### Samplers
- **RK4** (Runge-Kutta 4th order): Higher quality, 4 evals per step
- **Euler**: Faster, 1 eval per step

### Timestep Schedule
- Rectified Flow with logSNR-based schedule
- Goes from noise (t=1.0) to clean (t=0.0)

### Classifier-Free Guidance (CFG)
- Default scale: 7.0
- Unconditional = empty prompt
- Formula: `v = v_uncond + cfg_scale * (v_cond - v_uncond)`

## Key Differences: stable-audio-open-1.0 vs small

| Component | 1.0 | small |
|-----------|-----|-------|
| DiT embed_dim | 1536 | 1024 |
| DiT depth | 24 | 16 |
| DiT num_heads | 24 | 8 |
| Global cond | concat(start, total) = 1536 | only total = 768 |
| Conditioners | seconds_start + seconds_total | only seconds_total |
| Tokenizer | Same (T5-based) | Same (T5-based) |

## Common Issues & Solutions

### 1. Parameter Mismatch Errors
- **Cause**: DiTConfig doesn't match model architecture
- **Fix**: Use values from `DiTConfig` class (already configured for small)

### 2. Missing QK Normalization
- **Cause**: Model has `q_norm`/`k_norm` but architecture doesn't
- **Fix**: Already implemented in `SelfAttention` and `CrossAttention`

### 3. Shape Mismatches
- **Cause**: Wrong `embed_dim`, `num_heads`, or `global_cond_dim`
- **Fix**: Use small model config (1024/8/768, not 1536/24/1536)

### 4. Tokenizer Not Found
- **Cause**: stable-audio-open-small repo has no tokenizer
- **Fix**: Downloads from `stable-audio-open-1.0/tokenizer` (same for both)

### 5. Metal/GPU Initialization Errors
- **Symptom**: NSRangeException on MLX initialization
- **Fix**: Check macOS/Metal compatibility, try updating MLX

## Development Notes

- **MLX Requirements**: Blocks must be explicit attributes (`block_0`, `block_1`, ...) not lists
- **Weight Loading**: Use `load_weights()` override for key mapping
- **Evaluation**: Models need `.eval()` to disable dropout
- **Memory**: Full pipeline ~4-6GB RAM, ensure sufficient memory

## Testing

Generate test audio to verify setup:
```bash
python generate.py --prompt "warm arpeggios on house beats 120BPM with drums" \
  --seconds 10 --steps 30 --cfg-scale 7.0 --seed 42 \
  --output test.wav --sampler euler
```

Expected output: `test.wav` (44.1kHz, stereo, 10 seconds)

## File Formats

- **Input**: `.safetensors` (PyTorch weights)
- **Output (MLX)**: `.npz` (numpy archive), `.safetensors` (T5 only)
- **Audio**: `.wav` (32-bit float, 44.1kHz, stereo)

## Dependencies

Key packages:
- `mlx` (0.30.0) - Apple Silicon ML framework
- `transformers` (4.57.3) - T5 tokenizer
- `safetensors` (0.7.0) - Weight loading
- `soundfile` (0.13.1) - Audio I/O
- `numpy` (2.3.5) - Array operations
- `tqdm` (4.67.1) - Progress bars
