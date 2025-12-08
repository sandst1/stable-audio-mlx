"""Extract weights from TFLite models using TensorFlow Lite."""
import numpy as np
import tensorflow as tf

def extract_tflite_weights(model_path):
    """Extract all weights and their metadata from a TFLite model."""
    print(f"Reading: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    weights = {}
    tensor_info = []
    
    for detail in tensor_details:
        tensor_idx = detail['index']
        tensor_name = detail['name']
        tensor_shape = tuple(detail['shape'])
        tensor_dtype = detail['dtype']
        
        # Try to get tensor data (only works for constants/weights)
        try:
            tensor_data = interpreter.get_tensor(tensor_idx)
            weights[tensor_name] = tensor_data
            tensor_info.append((tensor_name, tensor_shape, tensor_dtype, tensor_data.dtype))
            
        except:
            # This is likely an input/output tensor, not a weight
            pass
    
    print(f"  Extracted {len(weights)} weight tensors (out of {len(tensor_details)} total)")
    
    return weights, tensor_info

def main():
    print("Extracting weights from TFLite models...")
    print("="*80)
    print()
    
    tflite_dir = "related/tflite_model"
    output_path = "model/weights_from_tflite_raw.npz"
    
    # Extract from all three models
    print("1. VAE (Autoencoder)")
    vae_weights, vae_info = extract_tflite_weights(f"{tflite_dir}/autoencoder_model.tflite")
    print()
    
    print("2. DiT (Diffusion Transformer)")
    dit_weights, dit_info = extract_tflite_weights(f"{tflite_dir}/dit_model.tflite")
    print()
    
    print("3. Conditioners (T5 + Time)")
    cond_weights, cond_info = extract_tflite_weights(f"{tflite_dir}/conditioners_float32.tflite")
    print()
    
    print("="*80)
    print("Summary:")
    print(f"  VAE: {len(vae_weights)} tensors")
    print(f"  DiT: {len(dit_weights)} tensors")
    print(f"  Conditioners: {len(cond_weights)} tensors")
    print()
    
    # Save
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, 
                        vae=vae_weights,
                        dit=dit_weights,
                        cond=cond_weights)
    print("✓ Saved!")
    print()
    
    # Show sample tensor names
    print("Sample tensor names:")
    print()
    print("VAE (first 15):")
    for name, shape, _, dtype in vae_info[:15]:
        print(f"  {name:70s} shape={shape} dtype={dtype}")
    print()
    
    print("DiT (first 15):")
    for name, shape, _, dtype in dit_info[:15]:
        print(f"  {name:70s} shape={shape} dtype={dtype}")
    print()
    
    print("Conditioners (first 15):")
    for name, shape, _, dtype in cond_info[:15]:
        print(f"  {name:70s} shape={shape} dtype={dtype}")
    print()
    
    print("="*80)
    print("Weights extracted! Next: Create mapping to MLX model structure.")

if __name__ == "__main__":
    main()

