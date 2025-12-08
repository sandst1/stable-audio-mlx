# Stable Audio MLX

**MLX (Apple Silicon)** implementation of Stability AI's **stable-audio-open-small** model for text-to-audio generation.

**Status**: ✅ **WORKING!** Produces high-quality audio using hybrid TFLite+MLX approach.

## ⚠️ Important Requirements

1. **macOS** with Apple Silicon (M1/M2/M3)
2. **Python 3.10+** (Recommended: 3.11 or 3.12)  
3. **TensorFlow** (for TFLite conditioners): `pip install tensorflow-macos`
4. **TFLite models** in `related/tflite_model/` (included in repo)

## Why Hybrid TFLite+MLX?

Our pure MLX implementation has a bug in the time conditioning (NumberEmbedder). Using the TFLite conditioners model fixes this and produces perfect audio quality. The DiT and VAE run entirely in MLX (fast on Apple Silicon!).

See `BUG_FIX_SUMMARY.md` for full technical details.

## Installation

1. Clone or navigate to this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Preparation

The weights are converted from the PyTorch safetensors format to MLX format.

### Automatic Setup (Recommended)

Simply run the conversion script - it will automatically download the model if needed:

```bash
python src/conversion/convert.py
```

This script will:
- ✓ Auto-download model files from HuggingFace if not present
- ✓ Extract and convert the VAE, DiT, Conditioners, and T5 encoder weights
- ✓ Save converted weights to `model/stable_audio_small.npz` and `model/t5.safetensors`
- ✓ Handle weight normalization fusion and tensor transposition for MLX compatibility

The first run will download ~3GB of model files. Subsequent runs will use the cached files.

### Manual Setup (Optional)

If you have your own model files or want to use a different variant:

1. Place your files in the `model/` folder:
   - `model/model.safetensors` - the model weights
   - `model/model_config.json` - model configuration (optional)

2. Run the conversion script as above.

## Usage

Use the `app.py` script to generate audio.

```bash
python app.py --prompt "A beautiful orchestral symphony, classical music" --seconds 10 --output output.wav
```

### Arguments

- `--prompt`: The text description of the audio (required).
- `--seconds`: Duration in seconds (default: 10).
- `--steps`: Number of diffusion steps (default: 50).
- `--output`: Output filename (default: `output.wav`).
- `--seed`: Random seed for reproducibility.

## Project Structure

```
model/                          # Model files (place your weights here)
├── model.safetensors          # Input: PyTorch weights
├── model_config.json          # Input: Model config (optional)
├── stable_audio_small.npz     # Output: Converted MLX weights (VAE, DiT, Conditioners)
├── t5.safetensors            # Output: T5 encoder weights
└── tokenizer files...         # Output: Auto-downloaded tokenizer

src/
├── conversion/
│   └── convert.py             # Unified conversion script
├── models/                    # Model architectures (VAE, DiT, T5)
└── pipeline/                  # Inference pipeline
```

## Known Limitations

- **CFG**: Classifier-Free Guidance is supported but may require tuning for best results.
- **Performance**: The DiT model is large. Ensure you have sufficient RAM (16GB+ recommended).

## License

This project adapts architecture from Stability AI's open release. Please refer to their license for model usage.
