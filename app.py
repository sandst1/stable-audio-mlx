import argparse
import soundfile as sf
import mlx.core as mx
import numpy as np

from src.pipeline.pipeline import StableAudioPipeline

def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open MLX")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--seconds", type=float, default=10.0, help="Audio duration")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (8 steps recommended, matches TFLite reference)")
    parser.add_argument("--cfg-scale", type=float, default=1.0, 
                        help="Classifier-free guidance scale (1.0=no CFG, matches TFLite reference; >1.0 experimental)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.wav", help="Output filename")
    parser.add_argument("--sampler", type=str, default="rk4", choices=["rk4", "euler"], 
                        help="Sampler method: 'rk4' (recommended, 4th order accurate) or 'euler' (faster, 1st order)")
    parser.add_argument("--use-tflite-conditioners", action="store_true", default=True,
                        help="Use TFLite conditioners model (RECOMMENDED, fixes time conditioning)")
    parser.add_argument("--tflite-model-dir", type=str, default="related/tflite_model",
                        help="Directory containing TFLite models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    print("Loading pipeline...")
    # Ensure conversion is done
    weights_path = "model/stable_audio_small.npz"
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Please run src/conversion/convert.py first.")
        return

    pipe = StableAudioPipeline.from_pretrained(
        weights_path,
        use_tflite_conditioners=args.use_tflite_conditioners,
        tflite_model_dir=args.tflite_model_dir if args.use_tflite_conditioners else None
    )
    
    print(f"Generating audio for '{args.prompt}'...")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")
    print(f"CFG scale: {args.cfg_scale}, Sampler: {args.sampler}")
    
    audio = pipe.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seconds_total=args.seconds,
        seed=args.seed,
        sampler=args.sampler
    )
    
    # Save - match TFLite output format exactly
    print(f"Saving to {args.output}...")
    
    # Convert to numpy: audio shape is (1, 2, T) after generation
    # audio[0] is (2, T) = (channels, samples)
    audio_arr = np.array(audio[0], dtype=np.float32)  # Ensure float32
    
    # Debug info
    print(f"  Audio shape: {audio_arr.shape}")
    print(f"  Audio range: [{audio_arr.min():.3f}, {audio_arr.max():.3f}]")
    
    # Extract channels explicitly (matching TFLite format)
    # TFLite output: [L0, L1, ..., R0, R1, ...] (channel-first planar)
    num_samples = audio_arr.shape[1]
    left_ch = audio_arr[0]   # Shape: (T,)
    right_ch = audio_arr[1]  # Shape: (T,)
    
    # Create interleaved array for WAV: [L0, R0, L1, R1, ...]
    # This exactly matches what TFLite does in save_as_wav()
    interleaved = np.empty((num_samples, 2), dtype=np.float32)
    interleaved[:, 0] = left_ch
    interleaved[:, 1] = right_ch
    
    # Write as 32-bit float WAV (same as TFLite's IEEE float format)
    sf.write(args.output, interleaved, 44100, subtype='FLOAT')
    print("Done!")

if __name__ == "__main__":
    import os
    main()
