# Implementation Plan: Stable Audio Open 1.0 on MLX

This plan outlines the steps to convert the Stability AI "Stable Audio Open 1.0" model to run on Apple Silicon using the MLX framework and build a text-to-audio generation application.

## Phase 1: Preparation & Analysis

### 1.1 Environment Setup
- [ ] Verify MLX installation and version.
- [ ] Install utility libraries: `huggingface_hub`, `safetensors`, `transformers`, `torch` (CPU version for weight extraction), `numpy`, `soundfile`.
- [ ] Create a project structure:
    ```
    stable-audio-mlx/
    ├── src/
    │   ├── models/         # MLX model definitions (VAE, DiT, T5)
    │   ├── conversion/     # Conversion scripts
    │   └── pipeline/       # Inference pipeline
    ├── tests/
    ├── app.py              # Main application entry point
    └── requirements.txt
    ```

### 1.2 Model Discovery
- [ ] Locate the `stable-audio-open-1.0` model files.
    - **Found**: Model config is blob `3e7482...` and T5 config is `0db501...`.
    - Main weights (`pytorch_model.bin` equivalent) are likely `7b2045...` (~4.8GB).
    - Action: Use `huggingface-cli` or script to properly restore filenames from the cache blobs to a local `weights/` directory for easier access.

### 1.3 Architecture Analysis (Confirmed)
- **Text Encoder**: `t5-base` (768 dim).
- **Autoencoder (VAE)**:
    - Type: "Oobleck" 1D CNN.
    - Config: `channels=128`, `strides=[2, 4, 4, 8, 8]`, `latent_dim=64`.
    - Activation: Uses "Snake" activation (requires custom MLX implementation).
- **Diffusion Model (DiT)**:
    - Type: Continuous Transformer (`dit`).
    - Config: `embed_dim=1536`, `depth=24`, `num_heads=24`.
    - Conditioning: Cross-attention (Text), Global (Timing).

## Phase 2: Model Conversion

### 2.1 Text Encoder (T5)
- [ ] Use `mlx-lm` or `transformers` port for `t5-base`.
- [ ] Verify `t5-base` output matches PyTorch (embeddings).

### 2.2 Autoencoder (VAE)
- [ ] Implement `Snake` activation function in MLX (sinusoidal).
- [ ] Implement `OobleckEncoder` and `OobleckDecoder` (1D Convs, ResNets).
- [ ] **Conversion**: Map PyTorch weights. Note: PyTorch Conv1d `(C_out, C_in, K)` vs MLX `(N, L, C)` or `(Out, In, K)` conventions need checking.

### 2.3 Diffusion Model (DiT)
- [ ] Implement `DiffusionTransformer` class.
    - Support Rotary Embeddings (RoPE) if used (implied by "continuous_transformer").
    - Implement `TimestepEmbedding`.
    - Implement `GlobalConditioning` (concatenated or added to embeddings).
- [ ] **Conversion**: Map DiT weights.

### 2.4 Weight Conversion Script
- [ ] Create `convert_weights.py` to:
    1. Load the ~4.8GB blob as a PyTorch state dict.
    2. Extract T5 weights (if not loading separately).
    3. Extract VAE weights -> `vae.npz`.
    4. Extract DiT weights -> `dit.npz`.
    5. Save config files alongside weights.

## Phase 3: Inference Pipeline Implementation

### 3.1 Scheduler / Sampler
- [ ] Implement `RectifiedFlow` or `DPMSolver` (Stable Audio uses a specific Rectified Flow or diffusion scheduler).
- [ ] Implement `sigma` schedule (Linear/Cosine).

### 3.2 Generation Loop
- [ ] Create `StableAudioPipeline` class.
- [ ] Implement `__call__`:
    - `cond = text_encoder(prompt)`
    - `noise = random_normal(shape)`
    - `latents = sample(model, noise, cond, steps)`
    - `audio = vae.decode(latents)`

## Phase 4: Application Development

### 4.1 Core Application
- [ ] Create `generate.py` CLI.
    - Inputs: Prompt, Seconds (Duration), Steps, CFG Scale.
    - Output: WAV file.

### 4.2 Optimization (Optional)
- [ ] `mlx.core.compile` key functions.
- [ ] Quantization (4-bit/8-bit) for the DiT (1.5GB -> ~0.8GB).

## Phase 5: Verification & Refinement

- [ ] "Sanity Check" Generation: Generate a simple prompt.
- [ ] Performance Tuning.
