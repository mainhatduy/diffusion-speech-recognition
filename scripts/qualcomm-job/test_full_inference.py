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
import traceback
from datetime import datetime

import dotenv
import numpy as np
import torch
import onnxruntime as ort

# Add src to path for model configuration
sys.path.insert(0, os.path.abspath("src"))

# ─────────────────────────────── Constants ────────────────────────────────
DEFAULT_DEVICE = "Samsung Galaxy S25 (Family)"
AUDIO_SAMPLE_RATE = 16000
MAX_SEQ_LEN = 32

# ─────────────────────────────── Helpers ──────────────────────────────────

def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and return as float32 numpy array at target sample rate."""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio.astype(np.float32)
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if sr != target_sr:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        return audio.astype(np.float32)


def prepare_audio_inputs(audio: np.ndarray, stride: int = 80) -> dict:
    """Prepare audio inputs for the audio_encoder ONNX model."""
    original_len = len(audio)
    if original_len % stride != 0:
        padded_len = ((original_len // stride) + 1) * stride
        audio_padded = np.zeros(padded_len, dtype=np.float32)
        audio_padded[:original_len] = audio
        mask = np.zeros(padded_len, dtype=np.int64)
        mask[:original_len] = 1
        print(f"    [pad] Audio padded from {original_len} → {padded_len} samples (multiple of {stride})")
    else:
        audio_padded = audio
        mask = np.ones(original_len, dtype=np.int64)
    
    audio_features = audio_padded[np.newaxis, :]  # (1, audio_len)
    audio_attention_mask = mask[np.newaxis, :]     # (1, audio_len)
    return {
        "audio_features": audio_features,
        "audio_attention_mask": audio_attention_mask,
    }


def prepare_backbone_inputs(audio_embeds: np.ndarray, audio_len: int) -> dict:
    """Prepare precomputed audio embeddings and mask."""
    batch_size = 1
    precomputed_audio_mask = np.ones((batch_size, audio_len), dtype=np.int32)
    return {
        "precomputed_audio_embeds": audio_embeds.astype(np.float32),
        "precomputed_audio_mask": precomputed_audio_mask,
    }


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """Select top-k lowest scores and create mask."""
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def reparam_decoding(
    output_tokens, 
    output_scores, 
    cur_tokens,
    cur_scores,
    decoding_strategy,
    xt_neq_x0, 
    non_special_sym_mask, 
    t,
    max_step,
    noise_id
):
    """Reparameterized decoding step."""
    _, condition, topk_mode, schedule = decoding_strategy.split("-")

    if schedule == "linear":
        rate = 1 - t / max_step
    elif schedule == "cosine":
        rate = np.cos(t / max_step * np.pi * 0.5)
    else:
        raise NotImplementedError

    cutoff_len = (
        non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
    ).long()
    _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
    
    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError
    
    if condition == "cond":
        not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
    elif condition == "uncond":
        not_v1_t = lowest_k_mask
    else:
        raise NotImplementedError
    
    not_v2_t = lowest_k_mask

    masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
    output_tokens.masked_fill_(masked_to_noise, noise_id)
    output_scores.masked_fill_(masked_to_noise, -float('inf'))

    masked_to_x0 = xt_neq_x0 & ~not_v2_t
    output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
    output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
    
    new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
    return new_xt_neq_x0


def decode_tokens(tokens, tokenizer, eos_id):
    """Truncate sequence at first EOS and decode."""
    tokens_list = tokens.tolist()[0] if hasattr(tokens, "tolist") else list(tokens[0])
    if eos_id in tokens_list:
        eos_idx = tokens_list.index(eos_id)
        tokens_list = tokens_list[:eos_idx]
    return tokenizer.decode(tokens_list, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end full iterative speech translation inference on Qualcomm chipset"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help=f"Device name (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--runtime", choices=["qnn", "onnx"], default="onnx",
        help="Target runtime. 'onnx' recommended for broader compatibility (default: onnx)"
    )
    parser.add_argument(
        "--audio", type=str, default="test/test_data/test_sample.mp3",
        help="Path to test audio file"
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of diffusion denoising steps (default: 10)"
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
    dotenv.load_dotenv()
    token = os.getenv("QUALCOMM_TOKEN")
    if not token:
        print("[!] Error: QUALCOMM_TOKEN not found in .env")
        sys.exit(1)
    os.environ["QAI_HUB_API_TOKEN"] = token
    
    import qai_hub as hub

    # Load tokenizer and config
    print("\n[*] Loading tokenizer and config from HuggingFace Hub...")
    from transformers import AutoTokenizer
    from model.configuration_dlm import DiscreteDiffusionConfig
    
    repo_id = "aiai-laboratory/diffusion-speech-translation-from-vi-v1"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    config = DiscreteDiffusionConfig.from_pretrained(repo_id)
    print(f"    Tokenizer vocab size: {len(tokenizer)}")
    print(f"    mask_token_id: {config.mask_token_id}, eos_token_id: {config.eos_token_id}")

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
        "onnx/audio_encoder.onnx"
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
            "audio_attention_mask": audio_inputs["audio_attention_mask"]
        }
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
        "precomputed_audio_embeds": tuple(backbone_inputs["precomputed_audio_embeds"].shape),
        "precomputed_audio_mask": (tuple(backbone_inputs["precomputed_audio_mask"].shape), "int32"),
    }

    # Locate device and compile job (reusing cached compilation if already done)
    print(f"\n[*] Submitting/monitoring compile job on AI Hub (device: {args.device})...")
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
        output_tokens.ne(config.pad_token_id) &
        output_tokens.ne(config.bos_token_id) &
        output_tokens.ne(config.eos_token_id)
    )
    xt_neq_x0 = output_masks.clone()

    strategy = "reparam-uncond-deterministic-cosine"
    print(f"\n[*] Starting full iterative inference loop on {args.device}...")
    print(f"    Initial state: {decode_tokens(output_tokens, tokenizer, config.eos_token_id)}")

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
        logits_tensor[..., config.mask_token_id] = -float('inf')
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
            noise_id=config.mask_token_id
        )
        
        decoded_str = decode_tokens(output_tokens, tokenizer, config.eos_token_id)
        print(f"    [+] Step duration: {time.time() - t_step_start:.2f}s")
        print(f"    [+] Current prediction: \"{decoded_str}\"")

    total_time = time.time() - t_loop_start
    final_output = decode_tokens(output_tokens, tokenizer, config.eos_token_id)
    
    print("\n" + "=" * 80)
    print("                       FULL INFERENCE RESULTS")
    print("=" * 80)
    print(f"  Target Device:  {args.device}")
    print(f"  Total Steps:    {args.steps}")
    print(f"  Total Duration: {total_time:.2f} seconds ({total_time/args.steps:.2f}s per step)")
    if os.path.exists(gt_path):
        print(f"  Ground Truth:   \"{gt.get('english', 'N/A')}\"")
    print(f"  Model Output:   \"{final_output}\"")
    print("=" * 80)


if __name__ == "__main__":
    main()
