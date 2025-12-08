import os
import json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from huggingface_hub import hf_hub_download

def convert_vae_key(k):
    """Convert VAE keys to MLX format."""
    k = k.replace("pretransform.model.", "")
    parts = k.split('.')
    new_parts = []
    for i, part in enumerate(parts):
        new_parts.append(part)
        if part == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
            new_parts.append("layers") 
    return ".".join(new_parts)

def fuse_weight_norm(g, v):
    """Fuse Weight Normalization parameters: w = g * (v / ||v||)"""
    norm = np.linalg.norm(v, axis=(1, 2), keepdims=True)
    w = g * (v / (norm + 1e-9))
    return w

def download_model_files():
    """Download model files from HuggingFace if not present locally."""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    repo_id = "stabilityai/stable-audio-open-small"
    files_to_download = {
        "model.safetensors": "model.safetensors",
        "model_config.json": "model_config.json"
    }
    
    downloaded_any = False
    for local_name, repo_filename in files_to_download.items():
        local_path = os.path.join(model_dir, local_name)
        if not os.path.exists(local_path):
            print(f"Downloading {repo_filename} from {repo_id}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_filename,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                # Move to expected location if needed
                if downloaded_path != local_path and os.path.exists(downloaded_path):
                    os.rename(downloaded_path, local_path)
                print(f"✓ Downloaded {local_name}")
                downloaded_any = True
            except Exception as e:
                print(f"Warning: Could not download {repo_filename}: {e}")
    
    if downloaded_any:
        print()
    
    return downloaded_any

def download_t5_weights():
    """Download T5 encoder weights from stable-audio-open-1.0 text_encoder."""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # T5 weights are in the 1.0 model, not the small model
    repo_id = "stabilityai/stable-audio-open-1.0"
    t5_filename = "text_encoder/model.safetensors"
    output_path = os.path.join(model_dir, "t5_encoder.safetensors")
    
    print(f"T5 weights not found in model file.")
    print(f"Downloading T5 encoder from {repo_id}/{t5_filename}...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=t5_filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        # Move/rename to expected location
        final_path = os.path.join(model_dir, "t5.safetensors")
        if os.path.exists(downloaded_path):
            os.rename(downloaded_path, final_path)
            print(f"✓ Downloaded T5 encoder to {final_path}")
            return final_path
        
    except Exception as e:
        print(f"Error downloading T5 weights: {e}")
        return None
    
    return None

def convert():
    """
    Convert Stable Audio model weights from safetensors to MLX format.
    
    Reads from:
        - model/model.safetensors (or model/stable_audio_small.safetensors)
        - model/model_config.json (optional, for validation)
    
    If files are not found, automatically downloads from HuggingFace.
    
    Outputs:
        - model/stable_audio_small.npz (VAE, DiT, Conditioners)
        - model/t5.safetensors (T5 text encoder)
    """
    
    # Check if model files exist, download if not
    weights_path = None
    for filename in ["model.safetensors", "stable_audio_small.safetensors"]:
        path = os.path.join("model", filename)
        if os.path.exists(path):
            weights_path = path
            break
    
    if weights_path is None:
        print("Model files not found locally. Attempting to download from HuggingFace...")
        download_model_files()
        
        # Check again after download
        for filename in ["model.safetensors", "stable_audio_small.safetensors"]:
            path = os.path.join("model", filename)
            if os.path.exists(path):
                weights_path = path
                break
        
        if weights_path is None:
            print("\nError: Could not find or download model weights.")
            print("Please manually download model.safetensors from:")
            print("  https://huggingface.co/stabilityai/stable-audio-open-small")
            print("And place it in the model/ folder.")
            return
    
    print(f"Loading weights from {weights_path}...")
    
    # Check for config file (optional)
    config_path = "model/model_config.json"
    if os.path.exists(config_path):
        print(f"Found config at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Model config: {json.dumps(config, indent=2)}")
    else:
        print("Note: model_config.json not found (optional)")
    
    vae_state = {}
    dit_state = {}
    cond_state = {}
    t5_state = {}
    wn_pairs = {}
    
    with safe_open(weights_path, framework="np", device="cpu") as f:
        keys = f.keys()
        total_keys = len(keys)
        print(f"Processing {total_keys} tensors...")
        
        for k in keys:
            val = f.get_tensor(k)
            
            # --- T5 Text Encoder ---
            if "encoder.block" in k or (k.startswith("shared") and "embed" in k) or "final_layer_norm" in k:
                t5_state[k] = val
                continue
            
            # --- VAE (Autoencoder) ---
            if k.startswith("pretransform."):
                # Handle Weight Normalization (g, v pairs)
                if k.endswith(".weight_g"):
                    prefix = k[:-9]
                    if prefix not in wn_pairs: 
                        wn_pairs[prefix] = {}
                    wn_pairs[prefix]['g'] = val
                    continue
                if k.endswith(".weight_v"):
                    prefix = k[:-9]
                    if prefix not in wn_pairs: 
                        wn_pairs[prefix] = {}
                    wn_pairs[prefix]['v'] = val
                    continue
                
                clean_k = convert_vae_key(k)
                
                # Detect ConvTranspose1d layers (upsampling layers in decoder)
                # Pattern: decoder.layers.layers.X.layers.layers.1.weight
                is_transpose = False
                parts = k.split('.')
                if (len(parts) == 8 and 
                    parts[0] == "decoder" and parts[1] == "layers" and parts[2] == "layers" and
                    parts[4] == "layers" and parts[5] == "layers" and parts[6] == "1" and parts[7] == "weight"):
                    is_transpose = True
                
                # Transpose conv weights for MLX format
                if "weight" in k and val.ndim == 3 and "conv" in k:
                    if is_transpose:
                        val = val.transpose(1, 2, 0)  # (In, Out, K) -> (Out, K, In)
                    else:
                        val = val.transpose(0, 2, 1)  # (Out, In, K) -> (Out, K, In)
                        
                vae_state[clean_k] = val
                
            # --- DiT (Diffusion Transformer) ---
            elif k.startswith("model.model."):
                clean_k = k.replace("model.model.transformer.layers.", "blocks.")
                clean_k = clean_k.replace("model.model.", "")
                
                # Transpose conv weights
                if "conv" in k and "weight" in k and val.ndim == 3:
                    val = val.transpose(0, 2, 1)
                     
                dit_state[clean_k] = val
            
            # --- Conditioners (time, global parameters) ---
            elif k.startswith("conditioner."):
                cond_state[k] = val

    # Process Weight Norm pairs for VAE
    for prefix, pair in wn_pairs.items():
        if 'g' in pair and 'v' in pair:
            # Detect ConvTranspose1d layers
            # Prefix format: pretransform.model.decoder.layers.X.layers.1 (ConvTranspose)
            #            or: pretransform.model.decoder.layers.X.layers.Y.layers.Z (regular Conv)
            is_transpose = False
            parts = prefix.split('.')
            # Pattern: exactly 7 parts ending with decoder.layers.X.layers.1
            if (len(parts) == 7 and 
                parts[0] == "pretransform" and parts[1] == "model" and
                parts[2] == "decoder" and parts[3] == "layers" and
                parts[5] == "layers" and parts[6] == "1"):
                is_transpose = True
            
            w = fuse_weight_norm(pair['g'], pair['v'])
            
            # Transpose based on layer type
            if is_transpose:
                w = w.transpose(1, 2, 0)
            else:
                w = w.transpose(0, 2, 1)
                
            clean_prefix = convert_vae_key(prefix)
            vae_state[clean_prefix + ".weight"] = w
        else:
            print(f"Warning: incomplete weight norm pair for {prefix}")

    # Save outputs
    print(f"\nConverted weights:")
    print(f"  VAE: {len(vae_state)} tensors")
    print(f"  DiT: {len(dit_state)} tensors")
    print(f"  Conditioners: {len(cond_state)} tensors")
    print(f"  T5: {len(t5_state)} tensors")
    
    # Save main model weights
    output_npz = "model/stable_audio_small.npz"
    np.savez(output_npz, vae=vae_state, dit=dit_state, cond=cond_state)
    print(f"\n✓ Saved main model to {output_npz}")
    
    # Save T5 weights
    output_t5 = "model/t5.safetensors"
    if t5_state:
        save_file(t5_state, output_t5)
        print(f"✓ Saved T5 encoder to {output_t5}")
    else:
        print("⚠ Warning: No T5 weights found in model file")
        # Download T5 weights from stable-audio-open-1.0 (has text encoder)
        downloaded_t5 = download_t5_weights()
        if downloaded_t5:
            print(f"✓ Using downloaded T5 encoder")
        else:
            print("⚠ Could not obtain T5 weights. Text encoding may not work properly.")
    
    print("\nConversion complete! Ready to use with app.py")

if __name__ == "__main__":
    convert()
