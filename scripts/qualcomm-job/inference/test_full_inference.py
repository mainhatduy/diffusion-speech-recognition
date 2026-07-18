"""
Full Iterative Inference on Qualcomm Chipset via AI Hub
======================================================
This script runs the full diffusion denoising loop on the Qualcomm S25 chipset
by submitting step-by-step backbone inference requests to Qualcomm AI Hub.
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import onnxruntime as ort

# Add src to path for model configuration
sys.path.insert(0, os.path.abspath("src"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (
    AUDIO_SAMPLE_RATE,
    MAX_SEQ_LEN,
    setup_qualcomm_token,
    load_audio,
    prepare_audio_inputs,
    prepare_backbone_inputs,
    reparam_decoding,
    decode_tokens,
)

# ─────────────────────────────── Constants ────────────────────────────────
DEFAULT_DEVICE = "Samsung Galaxy S25 (Family)"


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end full iterative speech translation inference on Qualcomm chipset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device name (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--runtime",
        choices=["qnn", "onnx"],
        default="onnx",
        help="Target runtime. 'onnx' recommended for broader compatibility (default: onnx)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="test/test_data/test_sample.mp3",
        help="Path to test audio file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of diffusion denoising steps (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  QUALCOMM AI HUB — FULL ITERATIVE INFERENCE")
    print(f"  Device: {args.device}")
    print(f"  Runtime: {args.runtime}")
    print(f"  Audio: {args.audio}")
    print(f"  Denoising steps: {args.steps}")
    print("=" * 80)

    # Load environment
    setup_qualcomm_token()

    import qai_hub as hub

    # Load tokenizer and config
    print("\n[*] Loading tokenizer and config from HuggingFace Hub...")
    from transformers import AutoTokenizer
    from model.configuration_dlm import DiscreteDiffusionConfig

    repo_id = "aiai-laboratory/diffusion-speech-translation-from-vi-v1"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    config = DiscreteDiffusionConfig.from_pretrained(repo_id)
    print(f"    Tokenizer vocab size: {len(tokenizer)}")
    print(
        f"    mask_token_id: {config.mask_token_id}, eos_token_id: {config.eos_token_id}"
    )

    # Load and preprocess audio
    print(f"\n[*] Loading audio: {args.audio}")
    if not os.path.exists(args.audio):
        print(f"[!] Audio file not found: {args.audio}")
        sys.exit(1)

    audio = load_audio(args.audio, target_sr=AUDIO_SAMPLE_RATE)
    audio_inputs = prepare_audio_inputs(audio)

    # Load ground truth if available
    gt_path = args.audio.replace(".mp3", ".json").replace(".wav", ".json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        print(f"    Ground truth (vi): {gt.get('text', 'N/A')}")
        print(f"    Ground truth (en): {gt.get('english', 'N/A')}")

    # Precompute audio embeddings locally using ONNX Runtime
    print("\n[*] Precomputing audio embeddings locally using ONNX Runtime...")
    encoder_path = None
    possible_paths = [
        "onnx/audio_encoder_pkg.onnx/audio_encoder.onnx",
        "onnx/audio_encoder.onnx",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            encoder_path = path
            break

    if not encoder_path:
        print("[!] Error: Could not find audio encoder ONNX model.")
        sys.exit(1)

    print(f"    Loading ONNX model from: {encoder_path}")
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    ort_session = ort.InferenceSession(encoder_path, session_options)

    ort_outputs = ort_session.run(
        ["last_hidden_state"],
        {
            "audio_features": audio_inputs["audio_features"],
            "audio_attention_mask": audio_inputs["audio_attention_mask"],
        },
    )
    audio_embeds = ort_outputs[0]
    audio_len = audio_embeds.shape[1]
    print(f"    [+] Precomputed audio embeddings shape: {audio_embeds.shape}")

    # Prepare backbone inputs
    backbone_inputs = prepare_backbone_inputs(audio_embeds, audio_len)

    # Compilation/Device options
    runtime = args.runtime
    if runtime == "qnn":
        target_runtime = "qnn_context_binary"
    else:
        target_runtime = "precompiled_qnn_onnx"
    compile_options = f"--target_runtime {target_runtime} --truncate_64bit_io"

    backbone_specs = {
        "prev_output_tokens": ((1, MAX_SEQ_LEN), "int64"),
        "precomputed_audio_embeds": tuple(
            backbone_inputs["precomputed_audio_embeds"].shape
        ),
        "precomputed_audio_mask": (
            tuple(backbone_inputs["precomputed_audio_mask"].shape),
            "int32",
        ),
    }

    # Locate device and compile job (reusing cached compilation if already done)
    print(
        f"\n[*] Submitting/monitoring compile job on AI Hub (device: {args.device})..."
    )
    device = hub.Device(args.device)

    try:
        backbone_job = hub.submit_compile_job(
            model="onnx/diffusion_backbone_pkg.onnx",
            device=device,
            input_specs=backbone_specs,
            options=compile_options,
            name=f"backbone_{args.device[:20]}_{runtime}",
        )
        print(f"    Job submitted: {backbone_job.url}")
        status = backbone_job.wait()
        if status.code != "SUCCESS":
            print(f"[!] Compilation failed: {status.message}")
            sys.exit(1)
        print("[+] Compilation check passed (SUCCESS)!")
        backbone_target_model = backbone_job.get_target_model()
    except Exception as e:
        print(f"[!] Compilation check failed: {e}")
        sys.exit(1)

    # Initialize diffusion loop variables
    # output_tokens shape (1, 32), filled with mask_token_id, last is eos_token_id
    output_tokens = torch.full((1, MAX_SEQ_LEN), config.mask_token_id, dtype=torch.long)
    output_tokens[0, -1] = config.eos_token_id
    output_scores = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.float32)
    output_masks = output_tokens.eq(config.mask_token_id)
    non_fixed_sym_masks = (
        output_tokens.ne(config.pad_token_id)
        & output_tokens.ne(config.bos_token_id)
        & output_tokens.ne(config.eos_token_id)
    )
    xt_neq_x0 = output_masks.clone()

    strategy = "reparam-uncond-deterministic-cosine"
    print(f"\n[*] Starting full iterative inference loop on {args.device}...")
    print(
        f"    Initial state: {decode_tokens(output_tokens, tokenizer, config.eos_token_id)}"
    )

    t_loop_start = time.time()

    for step in range(args.steps):
        t_step_start = time.time()
        print(f"\n--- [Step {step + 1}/{args.steps}] ---")

        # Prepare tokens input for this step
        prev_output_tokens_np = output_tokens.numpy()
        if "--truncate_64bit_io" in compile_options:
            prev_output_tokens_np = prev_output_tokens_np.astype(np.int32)

        inf_inputs = {
            "prev_output_tokens": [prev_output_tokens_np],
            "precomputed_audio_embeds": [backbone_inputs["precomputed_audio_embeds"]],
            "precomputed_audio_mask": [backbone_inputs["precomputed_audio_mask"]],
        }

        # Submit inference job
        inf_job = hub.submit_inference_job(
            model=backbone_target_model,
            device=device,
            inputs=inf_inputs,
            name=f"backbone_inf_{args.device[:15]}_step{step}",
        )
        print(f"    Submitting inference step... Job: {inf_job.url}")
        status = inf_job.wait()
        if status.code != "SUCCESS":
            print(f"    [!] Inference step failed: {status.message}")
            sys.exit(1)

        # Download and extract logits
        output_data = inf_job.download_output_data()
        if isinstance(output_data, dict):
            logits = list(output_data.values())[0]
            if isinstance(logits, list):
                logits = logits[0]
        else:
            logits = output_data

        # logits shape: (1, seq_len, vocab_size)
        logits_tensor = torch.tensor(logits)
        logits_tensor[..., config.mask_token_id] = -float("inf")
        scores = torch.log_softmax(logits_tensor, dim=-1)

        # Perform decoding selection
        cur_scores, cur_tokens = scores.max(-1)
        cur_scores = cur_scores.to(output_scores)

        # Denoise / reparameterization step
        xt_neq_x0 = reparam_decoding(
            output_tokens=output_tokens,
            output_scores=output_scores,
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            decoding_strategy=strategy,
            xt_neq_x0=xt_neq_x0,
            non_special_sym_mask=non_fixed_sym_masks,
            t=step + 1,
            max_step=args.steps,
            noise_id=config.mask_token_id,
        )

        decoded_str = decode_tokens(output_tokens, tokenizer, config.eos_token_id)
        print(f"    [+] Step duration: {time.time() - t_step_start:.2f}s")
        print(f'    [+] Current prediction: "{decoded_str}"')

    total_time = time.time() - t_loop_start
    final_output = decode_tokens(output_tokens, tokenizer, config.eos_token_id)

    print("\n" + "=" * 80)
    print("                       FULL INFERENCE RESULTS")
    print("=" * 80)
    print(f"  Target Device:  {args.device}")
    print(f"  Total Steps:    {args.steps}")
    print(
        f"  Total Duration: {total_time:.2f} seconds ({total_time/args.steps:.2f}s per step)"
    )
    if os.path.exists(gt_path):
        print(f"  Ground Truth:   \"{gt.get('english', 'N/A')}\"")
    print(f'  Model Output:   "{final_output}"')
    print("=" * 80)


if __name__ == "__main__":
    main()
