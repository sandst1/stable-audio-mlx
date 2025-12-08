"""Convert TFLite models to MLX format.

This extracts weights from the working TFLite models instead of PyTorch safetensors.
Requires: pip install tensorflow (or tensorflow-macos on Mac)
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import lite as tflite
except ImportError:
    print("ERROR: TensorFlow not installed!")
    print("Install with: pip install tensorflow")
    print("Or on Mac: pip install tensorflow-macos")
    exit(1)

def extract_tflite_weights(model_path):
    """Extract all weights from a TFLite model."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    weights = {}
    for detail in tensor_details:
        tensor_idx = detail['index']
        tensor_name = detail['name']
        tensor_shape = detail['shape']
        
        # Try to get tensor data
        try:
            tensor_data = interpreter.get_tensor(tensor_idx)
            weights[tensor_name] = tensor_data
            print(f"  {tensor_name:60s} shape={tuple(tensor_shape)} dtype={tensor_data.dtype}")
        except:
            # Some tensors might not be accessible (outputs, etc.)
            pass
    
    return weights

def main():
    print("Converting TFLite models to MLX format...")
    print("="*80)
    print()
    
    tflite_dir = "related/tflite_model"
    output_path = "model/stable_audio_tflite.npz"
    
    # Extract weights from each TFLite model
    print("1. Extracting VAE (autoencoder) weights...")
    vae_weights = extract_tflite_weights(f"{tflite_dir}/autoencoder_model.tflite")
    
    print("\n2. Extracting DiT weights...")
    dit_weights = extract_tflite_weights(f"{tflite_dir}/dit_model.tflite")
    
    print("\n3. Extracting Conditioner (T5 + time) weights...")
    cond_weights = extract_tflite_weights(f"{tflite_dir}/conditioners_float32.tflite")
    
    print("\n" + "="*80)
    print("Extracted weights:")
    print(f"  VAE: {len(vae_weights)} tensors")
    print(f"  DiT: {len(dit_weights)} tensors")
    print(f"  Conditioners: {len(cond_weights)} tensors")
    print()
    
    # Now we need to map TFLite tensor names to our MLX model structure
    # This is complex because TFLite uses numeric indices, not semantic names
    
    print("Note: TFLite uses numeric tensor indices, not semantic names.")
    print("We need to manually map these to our MLX model structure.")
    print()
    print("Next steps:")
    print("1. Inspect the tensor names/indices to understand the structure")
    print("2. Create a mapping from TFLite tensors to MLX parameter names")
    print("3. Transpose weights to MLX format")
    print("4. Save as NPZ for MLX")
    
    # Save raw extracted weights for inspection
    print(f"\nSaving raw extracted weights to {output_path}.raw...")
    np.savez(output_path + ".raw", 
             vae=vae_weights, 
             dit=dit_weights, 
             cond=cond_weights)
    print("✓ Saved raw weights for inspection")

if __name__ == "__main__":
    main()

