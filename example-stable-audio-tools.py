import torch
import torchaudio
import noisereduce as nr
import numpy as np
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Parameters
seconds_total = 3
seed = -1  # Set to a positive integer for reproducible results
volume = 0.75  # Output volume (0-1), leaves headroom to avoid harshness

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = seconds_total * sample_rate

if device == "mps":
    model = model.to(device, dtype=torch.float32)
else:
    model = model.to(device)

model.eval()  # Ensure model is in eval mode (disables dropout, etc.)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "80BPM vintage hiphop, mellow",
    "seconds_start": 0,
    "seconds_total": seconds_total
}]

# Generate stereo audio
# For rectified flow (small model): euler sampler, 50-100 steps
print(f"Generating {seconds_total}s of audio...")
with torch.inference_mode():
    output = generate_diffusion_cond(
        model,
        steps=8,             # 8 is way too low! Need 50+ for clean output
        cfg_scale=5,          # Lower CFG = less artifacts
        conditioning=conditioning,
        sample_size=sample_size,
        sampler_type="rk4",   # rk4, Best for small model
        seed=seed,
        sigma_max=1.0,
        device=device
    )
print("Generation complete!")

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Convert to float32 for processing
output = output.to(torch.float32).cpu()

# Debug: check raw output range
print(f"Raw output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Raw output std: {output.std():.3f}")
print(f"Sample rate: {sample_rate}")

# Save RAW output (no post-processing) to compare
raw_output = output.clone()
raw_output = raw_output.div(torch.max(torch.abs(raw_output))).clamp(-1, 1).mul(32767).to(torch.int16)
torchaudio.save("output_raw.wav", raw_output, sample_rate)
print("Saved raw (unprocessed) to output_raw.wav")

# Spectral noise reduction (noisereduce)
print("Applying noise reduction...")
output_np = output.numpy()
output_np = nr.reduce_noise(
    y=output_np,
    sr=sample_rate,
    stationary=True,       # Assume stationary noise (hiss, hum)
    prop_decrease=0.5,     # Reduced - less aggressive now that steps are higher
    n_fft=2048,            # FFT size
    n_std_thresh_stationary=2.0,  # Higher threshold = less aggressive
)
output = torch.from_numpy(output_np)

# Optional: Low-pass filter
lowpass_cutoff = 14000  # Hz - raised since output should be cleaner now
output = torchaudio.functional.lowpass_biquad(output, sample_rate, lowpass_cutoff)

# Peak normalize, apply volume, clip, convert to int16, and save to file
output = output.div(torch.max(torch.abs(output))).mul(volume).clamp(-1, 1).mul(32767).to(torch.int16)
torchaudio.save("output.wav", output, sample_rate)
print(f"Saved to output.wav")
