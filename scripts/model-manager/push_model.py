import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.abspath("src"))

from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/model-manager/push_model.py <repo_id> [experiment_dir] [checkpoint_dir]"
        )
        sys.exit(1)

    repo_id = sys.argv[1]
    experiment_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/vi_multitask"
    checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else None

    args_path = os.path.join(experiment_dir, "args.json")
    if not os.path.exists(args_path):
        print(f"Args file {args_path} not found.")
        sys.exit(1)

    # Find the highest checkpoint
    checkpoints = [d for d in os.listdir(experiment_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"No checkpoints found in {experiment_dir}")
        sys.exit(1)

    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints[-1]
    checkpoint_dir = checkpoint_dir or os.path.join(experiment_dir, highest_checkpoint)
    print(f"Using checkpoint: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} not found.")
        sys.exit(1)

    print(f"Loading args from {args_path}")
    with open(args_path, "r") as f:
        all_args = json.load(f)

    model_args = all_args["model"]
    data_args = all_args["data"]

    # Load tokenizer
    # If saved tokenizer exists in experiment_dir, load it from there to preserve added tokens
    if os.path.exists(os.path.join(experiment_dir, "tokenizer.json")):
        print(f"Loading tokenizer from {experiment_dir}")
        tokenizer = AutoTokenizer.from_pretrained(experiment_dir, use_fast=False)
    else:
        print(f"Loading tokenizer from {model_args['pretrained']}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args["pretrained"], use_fast=False
        )

    # Add AMR tokens if this is an AMR parsing model
    if data_args.get("dataset_type") == "amr_parsing":
        # Import and add AMR tokens
        sys.path.append(os.path.abspath("src"))
        from data.amr_process.additional_tokens import get_added_vocabulary

        new_tokens = get_added_vocabulary()
        num_added_toks = tokenizer.add_tokens(new_tokens)
        print(
            f"Added {num_added_toks} AMR tokens to tokenizer (vocab_size: {len(tokenizer)})"
        )

    # Load backbone config
    print(f"Loading backbone config from {model_args['pretrained']}")
    backbone_config = AutoConfig.from_pretrained(model_args["pretrained"])

    # Update vocab_size if tokenizer has additional tokens
    if len(tokenizer) != backbone_config.vocab_size:
        print(
            f"Updating backbone vocab_size from {backbone_config.vocab_size} to {len(tokenizer)}"
        )
        backbone_config.vocab_size = len(tokenizer)

    # Create DiscreteDiffusionConfig
    config = DiscreteDiffusionConfig(
        backbone_config=backbone_config,
        num_diffusion_timesteps=model_args["num_diffusion_timesteps"],
        diffusion_type=model_args["diffusion_type"],
        attention_strategy=model_args["attention_strategy"],
        vocab_pad_to_multiple=model_args["vocab_pad_to_multiple"],
        lora=model_args["lora"],
        lora_target_modules=model_args["lora_target_modules"],
        lora_alpha=model_args["lora_alpha"],
        lora_rank=model_args["lora_rank"],
        lora_bias=model_args["lora_bias"],
        lora_dropout=model_args["lora_dropout"],
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        argmax_decoding=True,  # Add this for deterministic inference
        dataset_type=data_args.get("dataset_type", "bilingual"),
        audio_encoder_name=data_args.get("audio_encoder_name", "facebook/mms-300m"),
    )

    # Update config auto_map
    config.auto_map = {
        "AutoConfig": "configuration_dlm.DiscreteDiffusionConfig",
        "AutoModel": "modeling_dlm.DiscreteDiffusionModel",
        "AutoModelForMaskedLM": "modeling_dlm.DiscreteDiffusionModel",
    }

    # Initialize model
    print("Initializing model...")
    model = DiscreteDiffusionModel(config)

    # Load state dict
    ckpt_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    print(f"Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Check for mismatch
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    # Push to hub
    print(f"Pushing to hub: {repo_id}")
    # This pushes model weights and config
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    # Upload custom code files
    print("Uploading custom code files...")
    api = HfApi(token=os.getenv("HF_TOKEN"))

    api.upload_file(
        path_or_fileobj="src/model/configuration_dlm.py",
        path_in_repo="configuration_dlm.py",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update configuration",
    )
    api.upload_file(
        path_or_fileobj="src/model/modeling_dlm.py",
        path_in_repo="modeling_dlm.py",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update modeling",
    )
    api.upload_file(
        path_or_fileobj="src/dd_generator.py",
        path_in_repo="dd_generator.py",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update dd_generator",
    )

    # Create and upload README.md
    print("Creating and uploading README.md...")
    if os.path.exists("MODEL_README.md"):
        with open("MODEL_README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
    else:
        readme_content = """---
language: vi
tags:
- diffusion
- speech-recognition
- speech-translation
- audio
- translation
---
# Discrete Diffusion Speech Translation (Vietnamese)

This is a **Discrete Diffusion** model for Vietnamese Speech Recognition (ASR) and Multi-task Speech Translation (AST) to English (EN), Chinese (ZH), and Korean (KO).

The model integrates a **Moonshine** audio encoder with an **XLM-RoBERTa** text model via absorbing discrete diffusion.

## Quick Start Guide

### 1. Install Dependencies
Install the required packages using `pip` or `uv`:

```bash
# Using uv (recommended):
uv pip install torch transformers librosa soundfile

# Or using pip:
pip install torch transformers librosa soundfile
```

### 2. Python Inference Code
The code snippet below automatically downloads the model from Hugging Face Hub, preprocesses the audio input, and runs speech recognition and multi-task translation tasks.

```python
import torch
import librosa
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor

# 1. Load model, tokenizer, and feature extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "{repo_id}"

print("Loading model from Hugging Face Hub...")
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model.config.audio_encoder_name)

# Use bfloat16 on GPU for faster decoding
if device == "cuda":
    model = model.to(torch.bfloat16)

# 2. Load and preprocess Vietnamese audio input (WAV, MP3, FLAC, etc.)
audio_path = "path/to/your/vietnamese_audio.wav"  # Replace with your Vietnamese speech audio file
waveform, sr = librosa.load(audio_path, sr=16000)

audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
audio_values_raw = audio_inputs.input_values.to(device)

# Pad audio features to a multiple of 80 frames (Moonshine requirement)
audio_len = audio_values_raw.size(-1)
padded_len = ((audio_len + 79) // 80) * 80
audio_features = torch.zeros(1, padded_len, device=device)
audio_features[0, :audio_len] = audio_values_raw[0]

audio_attention_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
audio_attention_mask[0, :audio_len] = 1

if device == "cuda":
    audio_features = audio_features.to(torch.bfloat16)

# 3. Define the ASR and translation tasks
tasks = {
    "Speech Recognition (Transcribe VI)": None,
    "Translation to English (Translate to EN)": "<vi_en>",
    "Translation to Chinese (Translate to ZH)": "<vi_zh>",
    "Translation to Korean (Translate to KO)": "<vi_ko>"
}

print("\\n--- INFERENCE RESULTS ---")
for label, task_token in tasks.items():
    # Initialize input_ids with the target task token
    if task_token is None:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
        task_token_id = tokenizer.convert_tokens_to_ids(task_token)
        input_ids = torch.tensor([[tokenizer.bos_token_id, task_token_id]], device=device)

    # Calculate dynamic canvas length based on audio duration to prevent repetition
    audio_duration = len(waveform) / sr
    if task_token == "<vi_en>" or task_token is None:
        canvas_len = int(audio_duration * 4.0)
    else:
        canvas_len = int(audio_duration * 2.5)
    canvas_len = max(5, min(64, canvas_len))

    # Run discrete diffusion generation (Denoising loop)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            audio_features=audio_features,
            audio_attention_mask=audio_attention_mask,
            max_iterations=10,  # Number of denoising steps
            max_length=canvas_len, # Dynamic sequence length
            strategy="reparam-uncond-deterministic-cosine"
        )

    # Decode target tokens to text
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(label + ": " + text)
```
"""

    # Replace @@@MODEL_ID and {repo_id} with actual repo_id
    readme_content = readme_content.replace("@@@MODEL_ID", repo_id)
    readme_content = readme_content.replace("{repo_id}", repo_id)

    # Save temporarily
    temp_readme_path = "temp_README.md"
    with open(temp_readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=temp_readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add README.md",
    )

    # Clean up temp file
    os.remove(temp_readme_path)

    print("Done! You can now load the model with:")
    print("from transformers import AutoModel")
    print(f"model = AutoModel.from_pretrained('{repo_id}', trust_remote_code=True)")


if __name__ == "__main__":
    main()
