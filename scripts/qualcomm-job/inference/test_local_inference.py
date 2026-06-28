"""
Local Inference Verification Script for Diffusion Speech Translation
===================================================================
This script runs end-to-end inference of the Vietnamese speech translation model
locally using either PyTorch or ONNX Runtime. Unlike the single-step benchmark,
this script executes the full multi-step diffusion denoising loop to produce
a high-quality translation.

Usage:
  uv run python scripts/qualcomm-job/inference/test_local_inference.py --mode both
  uv run python scripts/qualcomm-job/inference/test_local_inference.py --mode onnx --audio test/test_data/test_sample.mp3
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Add src to Python path
sys.path.insert(0, os.path.abspath("src"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (
    AUDIO_SAMPLE_RATE,
    MAX_SEQ_LEN,
    load_audio,
    prepare_audio_inputs,
    reparam_decoding,
    decode_tokens,
)

from model.modeling_dlm import DiscreteDiffusionModel
from model.configuration_dlm import DiscreteDiffusionConfig


# ────────────────────────────── Run Functions ──────────────────────────────

def run_pytorch(audio_path, repo_id, steps):
    print("\n--- Running PyTorch Local Inference ---")
    t0 = time.time()
    
    # Load config and model
    config = DiscreteDiffusionConfig.from_pretrained(repo_id)
    if isinstance(config.backbone_config, dict):
        config.backbone_config["attn_implementation"] = "eager"
    config.pretrained_audio_encoder = True
    
    print("    [+] Instantiating DiscreteDiffusionModel...")
    model = DiscreteDiffusionModel(config)
    
    print("    [+] Downloading and loading weights...")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Preprocess audio
    audio = load_audio(audio_path, target_sr=AUDIO_SAMPLE_RATE)
    audio_inputs = prepare_audio_inputs(audio)
    
    audio_features_t = torch.tensor(audio_inputs["audio_features"])
    audio_attention_mask_t = torch.tensor(audio_inputs["audio_attention_mask"])
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    task_token = "<vi_en>"
    task_token_id = tokenizer.convert_tokens_to_ids(task_token)
    
    # Format prefix: [bos_token_id, task_token_id]
    input_ids = torch.tensor([[config.bos_token_id, task_token_id]], dtype=torch.long)
    attention_mask = torch.tensor([[True, True]], dtype=torch.bool)
    
    # Max length of generated tokens = MAX_SEQ_LEN (32) - len(prefix) (2) - len(eos) (1) = 29
    canvas_len = MAX_SEQ_LEN - 3
    
    print(f"    [+] Executing {steps}-step generation (prefix: {input_ids.tolist()[0]})...")
    
    with torch.no_grad():
        final_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_iterations=steps,
            strategy="reparam-uncond-deterministic-cosine",
            temperature=1.0,
            max_length=canvas_len,
            audio_features=audio_features_t,
            audio_attention_mask=audio_attention_mask_t
        )
    
    output_text = tokenizer.decode(final_tokens[0], skip_special_tokens=True)
    
    duration = time.time() - t0
    print(f"    [+] PyTorch inference complete in {duration:.2f}s")
    print(f"    [+] Result: \"{output_text}\"")
    
    # Return full sequence of length 32 to compare with ONNX
    # model.generate returns final tokens. Let's make sure it is padded/truncated to MAX_SEQ_LEN (32)
    # The returned token length should be 32 (prefix + generated + eos).
    ret_tokens = final_tokens[0].tolist()
    if len(ret_tokens) < MAX_SEQ_LEN:
        ret_tokens += [config.pad_token_id] * (MAX_SEQ_LEN - len(ret_tokens))
    else:
        ret_tokens = ret_tokens[:MAX_SEQ_LEN]
        
    return output_text, ret_tokens


def run_onnx(audio_path, repo_id, steps):
    print("\n--- Running ONNX Runtime Local Inference ---")
    t0 = time.time()
    
    # 1. Paths
    encoder_path = None
    possible_enc_paths = [
        "onnx/audio_encoder_pkg.onnx/audio_encoder.onnx",
        "onnx/audio_encoder.onnx"
    ]
    for p in possible_enc_paths:
        if os.path.exists(p):
            encoder_path = p
            break
            
    backbone_path = None
    possible_bb_paths = [
        "onnx/diffusion_backbone_pkg.onnx/diffusion_backbone.onnx",
        "onnx/diffusion_backbone.onnx"
    ]
    for p in possible_bb_paths:
        if os.path.exists(p):
            backbone_path = p
            break
            
    if not encoder_path or not backbone_path:
        print("[!] Error: Could not find ONNX model files.")
        sys.exit(1)
        
    print(f"    [+] Loading Audio Encoder from: {encoder_path}")
    print(f"    [+] Loading Diffusion Backbone from: {backbone_path}")
    
    # 2. Start ORT Sessions
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    enc_session = ort.InferenceSession(encoder_path, session_options)
    bb_session = ort.InferenceSession(backbone_path, session_options)
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    config = DiscreteDiffusionConfig.from_pretrained(repo_id)
    
    task_token = "<vi_en>"
    task_token_id = tokenizer.convert_tokens_to_ids(task_token)
    
    # 3. Process Audio
    audio = load_audio(audio_path, target_sr=AUDIO_SAMPLE_RATE)
    audio_inputs = prepare_audio_inputs(audio)
    
    # Run Audio Encoder
    print("    [+] Running Audio Encoder...")
    enc_outputs = enc_session.run(
        ["last_hidden_state"],
        {
            "audio_features": audio_inputs["audio_features"],
            "audio_attention_mask": audio_inputs["audio_attention_mask"]
        }
    )
    audio_embeds = enc_outputs[0]
    audio_len = audio_embeds.shape[1]
    
    # 4. Initialize Backbone inputs
    precomputed_audio_mask = np.ones((1, audio_len), dtype=np.int32)
    
    # Initialize output_tokens: [bos, task, mask, mask, ..., mask, eos]
    output_tokens = torch.full((1, MAX_SEQ_LEN), config.mask_token_id, dtype=torch.long)
    output_tokens[0, 0] = config.bos_token_id
    output_tokens[0, 1] = task_token_id
    output_tokens[0, -1] = config.eos_token_id
    
    output_scores = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.float32)
    
    # partial_masks is True for the prompt prefix, False for the generated part
    partial_masks = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.bool)
    partial_masks[0, 0] = True
    partial_masks[0, 1] = True
    
    non_fixed_sym_masks = (
        output_tokens.ne(config.pad_token_id) &
        output_tokens.ne(config.bos_token_id) &
        output_tokens.ne(config.eos_token_id) &
        ~partial_masks
    )
    xt_neq_x0 = output_tokens.eq(config.mask_token_id)
    
    strategy = "reparam-uncond-deterministic-cosine"
    
    # 5. Denoising Loop
    print(f"    [+] Executing {steps}-step generation (prefix: {output_tokens[0, :2].tolist()})...")
    for step in range(steps):
        # Run Backbone Session
        ort_outputs = bb_session.run(
            ["logits"],
            {
                "prev_output_tokens": output_tokens.numpy(),
                "precomputed_audio_embeds": audio_embeds,
                "precomputed_audio_mask": precomputed_audio_mask
            }
        )
        logits = ort_outputs[0]
        
        logits_tensor = torch.tensor(logits)
        logits_tensor[..., config.mask_token_id] = -float('inf')
        scores = torch.log_softmax(logits_tensor, dim=-1)
        
        cur_scores, cur_tokens = scores.max(-1)
        cur_scores = cur_scores.to(output_scores)
        
        xt_neq_x0 = reparam_decoding(
            output_tokens=output_tokens,
            output_scores=output_scores,
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            decoding_strategy=strategy,
            xt_neq_x0=xt_neq_x0,
            non_special_sym_mask=non_fixed_sym_masks,
            t=step + 1,
            max_step=steps,
            noise_id=config.mask_token_id
        )
    
    # Finalize output text
    # Cut off prefix (first 2 tokens) and decode the rest
    generated_tokens = output_tokens[:, 2:]
    output_text = decode_tokens(generated_tokens, tokenizer, config.eos_token_id)
    
    duration = time.time() - t0
    print(f"    [+] ONNX Runtime inference complete in {duration:.2f}s")
    print(f"    [+] Result: \"{output_text}\"")
    return output_text, output_tokens[0].tolist()


# ─────────────────────────────── Main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify local PyTorch and ONNX model correctness")
    parser.add_argument(
        "--mode", choices=["pytorch", "onnx", "both"], default="both",
        help="Inference mode (default: both)"
    )
    parser.add_argument(
        "--audio", type=str, default="test/test_data/test_sample.mp3",
        help="Path to the test audio file (default: test/test_data/test_sample.mp3)"
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of diffusion steps (default: 10)"
    )
    args = parser.parse_args()

    repo_id = "aiai-laboratory/diffusion-speech-translation-from-vi-v1"
    
    # Print ground truth if exists
    gt_path = args.audio.replace(".mp3", ".json").replace(".wav", ".json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        print("=" * 80)
        print(f"Ground Truth (vi): {gt.get('text', 'N/A')}")
        print(f"Ground Truth (en): {gt.get('english', 'N/A')}")
        print("=" * 80)
        
    py_text, on_text = None, None
    py_tokens, on_tokens = None, None
    
    if args.mode in ["pytorch", "both"]:
        py_text, py_tokens = run_pytorch(args.audio, repo_id, args.steps)
        
    if args.mode in ["onnx", "both"]:
        on_text, on_tokens = run_onnx(args.audio, repo_id, args.steps)
        
    if args.mode == "both" and py_text is not None and on_text is not None:
        print("\n" + "=" * 80)
        print("                         NUMERICAL COMPARISON")
        print("=" * 80)
        print(f"PyTorch Output: \"{py_text}\"")
        print(f"ONNX Output:    \"{on_text}\"")
        
        # Check token level matching
        match = py_tokens == on_tokens
        if match:
            print("[+] MATCH SUCCESS: PyTorch and ONNX generated tokens match perfectly!")
        else:
            print("[!] WARNING: PyTorch and ONNX generated tokens differ.")
            # Show diff
            print(f"    PyTorch tokens: {py_tokens}")
            print(f"    ONNX tokens:    {on_tokens}")
        print("=" * 80)

if __name__ == "__main__":
    main()
