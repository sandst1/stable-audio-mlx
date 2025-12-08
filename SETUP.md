# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Convert the Model (Auto-Download)

Run the unified conversion script:

```bash
python src/conversion/convert.py
```

This will:
- ✓ Read your `model/model.safetensors` file
- ✓ Extract and convert VAE, DiT, and Conditioner weights
- ✓ Extract T5 text encoder weights
- ✓ Handle weight normalization and tensor transposition
- ✓ Save to `model/stable_audio_small.npz` (main model)
- ✓ Save to `model/t5.safetensors` (text encoder)

### 3. Run Inference

The tokenizer will be automatically downloaded on first run:

```bash
python app.py --prompt "A beautiful orchestral symphony" --seconds 10 --output music.wav
```

## What Changed?

### Before (Old Setup)
- Had TWO conversion scripts: `convert.py` and `convert_weights.py`
- Used `weights/` folder with hardcoded paths
- Required manual tokenizer setup

### After (New Setup)
- **Single unified conversion script**: `src/conversion/convert.py`
- **Everything in `model/` folder**: Clean, organized structure
- **Auto-downloads tokenizer**: No manual setup needed
- **Flexible input names**: Accepts `model.safetensors` or `stable_audio_small.safetensors`

## Folder Structure

```
model/                              # All model files here
├── model.safetensors              # [INPUT] Your weights
├── model_config.json              # [INPUT] Config (optional)
├── stable_audio_small.npz         # [OUTPUT] Converted weights
├── t5.safetensors                 # [OUTPUT] T5 encoder
└── tokenizer*.json                # [OUTPUT] Auto-downloaded

src/conversion/
└── convert.py                     # Unified conversion script
```

## Advanced Usage

### Custom Output Location

The conversion script is hardcoded to output to the `model/` folder for consistency. If you need different locations, edit the paths in `src/conversion/convert.py` lines 106-113.

### Using Different Model Sizes

The script works with any Stable Audio model variant. Just place the safetensors file in the `model/` folder and run the conversion.

### Troubleshooting

**"Weights not found"**: Make sure `model/model.safetensors` exists
**"T5 weights not found"**: Re-run `python src/conversion/convert.py`
**"No tokenizer"**: The first run will auto-download it from HuggingFace

## Files Summary

| File | Purpose | When Created |
|------|---------|--------------|
| `model/model.safetensors` | Your input weights | **You provide** |
| `model/model_config.json` | Model configuration | **You provide** (optional) |
| `model/stable_audio_small.npz` | Converted MLX weights | After running `convert.py` |
| `model/t5.safetensors` | T5 encoder weights | After running `convert.py` |
| `model/tokenizer*.json` | T5 tokenizer files | Auto-downloaded on first use |
