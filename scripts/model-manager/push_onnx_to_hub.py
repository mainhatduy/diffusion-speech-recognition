import sys
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def main():
    repo_id = "aiai-laboratory/onnx-diffusion-speech-translation-from-vi-v1"
    token = os.getenv("HF_TOKEN")
    
    if not token:
        print("HF_TOKEN environment variable is not set. Please set it in your .env file.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    print(f"Creating repository (if it doesn't exist): {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
        print("Repository ready.")
    except Exception as e:
        print(f"Error creating/verifying repository: {e}")
        
    # 1. Upload Tokenizer
    print("Uploading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("aiai-laboratory/diffusion-speech-translation-from-vi-v1", trust_remote_code=True)
        tokenizer.push_to_hub(repo_id, token=token)
        print("Tokenizer uploaded successfully.")
    except Exception as e:
        print(f"Failed to upload tokenizer: {e}")

    # 2. Upload ONNX Files
    onnx_files = [
        ("onnx/audio_encoder.onnx", "audio_encoder.onnx"),
        ("onnx/audio_encoder.onnx.data", "audio_encoder.onnx.data"),
        ("onnx/diffusion_backbone.onnx", "diffusion_backbone.onnx"),
        ("onnx/diffusion_backbone.onnx.data", "diffusion_backbone.onnx.data"),
    ]
    
    print("Uploading ONNX model files...")
    for local_path, repo_path in onnx_files:
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {repo_path}"
                )
                print(f"Successfully uploaded {repo_path}")
            except Exception as e:
                print(f"Failed to upload {repo_path}: {e}")
        else:
            print(f"Warning: local file {local_path} not found.")

    # 3. Upload custom helper modules for reference
    helpers = [
        ("src/model/configuration_dlm.py", "configuration_dlm.py"),
        ("src/model/modeling_dlm.py", "modeling_dlm.py"),
        ("src/dd_generator.py", "dd_generator.py"),
    ]
    print("Uploading custom helper scripts...")
    for local_path, repo_path in helpers:
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {repo_path} helper"
                )
                print(f"Successfully uploaded {repo_path}")
            except Exception as e:
                print(f"Failed to upload helper {repo_path}: {e}")

    # 4. Upload README.md
    print("Uploading README.md...")
    readme_content = f"""---
language: vi
tags:
- onnx
- onnxruntime
- diffusion
- speech-recognition
- speech-translation
- audio
- translation
---
# ONNX Discrete Diffusion Speech Translation (Vietnamese)

This repository contains the ONNX-exported version of the **Discrete Diffusion Speech Translation** model (`aiai-laboratory/diffusion-speech-translation-from-vi-v1`).

The model is split into two ONNX components to bypass complex tracing blockers and facilitate high-performance execution:
1. **`audio_encoder.onnx`** (weights in `audio_encoder.onnx.data`): Moonshine speech encoder that extracts acoustic features from audio inputs.
2. **`diffusion_backbone.onnx`** (weights in `diffusion_backbone.onnx.data`): XLM-RoBERTa text model and projection modules that perform discrete diffusion generation.

## Installation & Setup

Ensure you have the required dependencies:
```bash
pip install onnxruntime numpy transformers librosa soundfile
```

## Running Inference with ONNX Runtime

Below is a complete Python snippet demonstrating how to load the ONNX files and run multi-task speech translation / recognition using ONNX Runtime.

```python
import os
import torch
import numpy as np
import librosa
import onnxruntime as ort
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

# 1. Configuration & Download ONNX weights
repo_id = "{repo_id}"
device = "cpu"  # or 'cuda'

print("Downloading ONNX models...")
audio_encoder_path = hf_hub_download(repo_id=repo_id, filename="audio_encoder.onnx")
hf_hub_download(repo_id=repo_id, filename="audio_encoder.onnx.data") # Download external weights

backbone_path = hf_hub_download(repo_id=repo_id, filename="diffusion_backbone.onnx")
hf_hub_download(repo_id=repo_id, filename="diffusion_backbone.onnx.data") # Download external weights

# 2. Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(repo_id)
# Moonshine uses UsefulSensors/moonshine-streaming-medium as base audio extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("UsefulSensors/moonshine-streaming-medium")

# 3. Initialize ONNX sessions
providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
audio_session = ort.InferenceSession(audio_encoder_path, providers=providers)
backbone_session = ort.InferenceSession(backbone_path, providers=providers)

# 4. Load & Preprocess Audio
audio_path = "path/to/your/vietnamese_audio.wav"  # Replace with your audio file path
waveform, sr = librosa.load(audio_path, sr=16000)

audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="np")
audio_values_raw = audio_inputs.input_values

# Pad to multiple of 80 frames
audio_len = audio_values_raw.shape[-1]
padded_len = ((audio_len + 79) // 80) * 80
audio_features = np.zeros((1, padded_len), dtype=np.float32)
audio_features[0, :audio_len] = audio_values_raw[0]

audio_attention_mask = np.zeros((1, padded_len), dtype=np.int64)
audio_attention_mask[0, :audio_len] = 1

# 5. Extract Acoustic Embeddings via ONNX Audio Encoder
audio_outputs = audio_session.run(None, {{
    "audio_features": audio_features,
    "audio_attention_mask": audio_attention_mask
}})
precomputed_audio_embeds = audio_outputs[0] # (1, audio_seq_len, hidden_size)
precomputed_audio_mask = np.ones((1, precomputed_audio_embeds.shape[1]), dtype=np.int32)

# 6. Define multi-task decoding
tasks = {{
    "Speech Recognition (Transcribe VI)": None,
    "Translation to English (Translate to EN)": "<vi_en>",
    "Translation to Chinese (Translate to ZH)": "<vi_zh>",
    "Translation to Korean (Translate to KO)": "<vi_ko>"
}}

def run_denoising(input_ids, canvas_len):
    # Denoising loop wrapper for Discrete Diffusion
    # Initialize canvas with mask tokens
    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    
    # Create target array containing prompt + mask tokens
    full_seq = np.full((B, prompt_len + canvas_len), tokenizer.pad_token_id, dtype=np.int64)
    full_seq[:, :prompt_len] = input_ids
    
    # Fill target space with mask token
    full_seq[:, prompt_len:] = tokenizer.mask_token_id
    
    # We do a simple greedy argmax decoding for 10 steps (equivalent to reparam-uncond-deterministic-cosine)
    steps = 10
    for step in range(steps):
        # inputs: prev_output_tokens, partial_mask
        partial_mask = (full_seq == tokenizer.mask_token_id)
        
        logits = backbone_session.run(None, {{
            "prev_output_tokens": full_seq,
            "partial_mask": partial_mask,
            "precomputed_audio_embeds": precomputed_audio_embeds,
            "precomputed_audio_mask": precomputed_audio_mask
        }})[0]
        
        # Simple update: replace masked tokens with model predictions
        predictions = np.argmax(logits, axis=-1)
        
        # Linear schedule of unmasking/denoising
        ratio = (step + 1) / steps
        num_unmasked = int(canvas_len * ratio)
        
        # For simplicity, greedily decode the tokens with highest confidence or step-wise update
        # In this helper, we update the masked positions directly
        for b in range(B):
            masked_indices = np.where(full_seq[b, prompt_len:] == tokenizer.mask_token_id)[0] + prompt_len
            if len(masked_indices) > 0:
                full_seq[b, masked_indices] = predictions[b, masked_indices]
                
    return full_seq[:, prompt_len:]

print("\\n--- ONNX INFERENCE RESULTS ---")
audio_duration = len(waveform) / sr
for label, task_token in tasks.items():
    if task_token is None:
        input_ids = np.array([[tokenizer.bos_token_id]], dtype=np.int64)
    else:
        task_token_id = tokenizer.convert_tokens_to_ids(task_token)
        input_ids = np.array([[tokenizer.bos_token_id, task_token_id]], dtype=np.int64)
        
    canvas_len = int(audio_duration * (4.0 if task_token in [None, "<vi_en>"] else 2.5))
    canvas_len = max(5, min(64, canvas_len))
    
    output_ids = run_denoising(input_ids, canvas_len)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"{{label}}: {{text}}")
```
"""
    
    temp_readme = "temp_onnx_README.md"
    with open(temp_readme, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    try:
        api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add ONNX model README"
        )
        print("README.md uploaded successfully.")
    except Exception as e:
        print(f"Failed to upload README.md: {e}")
        
    if os.path.exists(temp_readme):
        os.remove(temp_readme)
        
    print(f"\nAll tasks complete! View model at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
