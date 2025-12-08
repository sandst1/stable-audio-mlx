import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import soundfile as sf
import mlx.core as mx
import numpy as np
import subprocess

from src.pipeline.pipeline import StableAudioPipeline

def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open MLX")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--seconds", type=float, default=5.0, help="Audio duration")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (8-30 recommended)")
    parser.add_argument("--cfg-scale", type=float, default=6.0, 
                        help="Classifier-free guidance scale (1.0=no CFG recommended; >1.0 experimental)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--sampler", type=str, default="euler", choices=["rk4", "euler"], 
                        help="Sampler method: 'rk4' (recommended, 4th order accurate) or 'euler' (faster, 1st order)")
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

    pipe = StableAudioPipeline.from_pretrained(weights_path)
    
    # Generate a random seed if none provided
    if args.seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    else:
        seed = args.seed
    
    print(f"Generating audio for '{args.prompt}'...")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")
    print(f"CFG scale: {args.cfg_scale}, Sampler: {args.sampler}, Seed: {seed}")
    
    audio = pipe.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seconds_total=args.seconds,
        seed=seed,
        sampler=args.sampler
    )
    
    # Generate output filename from prompt with seed
    output_filename = args.prompt.replace(" ", "_") + f"_seed_{seed}.wav"
    
    # Save audio
    print(f"Saving to {output_filename}...")
    
    # Convert to numpy: audio shape is (1, 2, T) after generation
    # audio[0] is (2, T) = (channels, samples)
    audio_arr = np.array(audio[0], dtype=np.float32)
    
    print(f"  Audio shape: {audio_arr.shape}")
    print(f"  Audio range: [{audio_arr.min():.3f}, {audio_arr.max():.3f}]")
    
    # Create interleaved stereo array for WAV: [L0, R0, L1, R1, ...]
    num_samples = audio_arr.shape[1]
    interleaved = np.empty((num_samples, 2), dtype=np.float32)
    interleaved[:, 0] = audio_arr[0]  # Left channel
    interleaved[:, 1] = audio_arr[1]  # Right channel
    
    # Write as 32-bit float WAV
    sf.write(output_filename, interleaved, 44100, subtype='FLOAT')
    print("Done!")
    
    # Play the audio
    print(f"\nPlaying audio...")
    try:
        subprocess.run(["afplay", output_filename], check=True)
    except FileNotFoundError:
        print("Note: 'afplay' not found. Skipping playback. (Audio file saved successfully)")
    except subprocess.CalledProcessError:
        print("Error playing audio, but file was saved successfully.")

if __name__ == "__main__":
    main()
