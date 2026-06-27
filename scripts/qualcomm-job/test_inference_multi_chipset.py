"""
Multi-Chipset Inference Test for Diffusion Speech Translation Model
====================================================================
This script tests end-to-end inference of the Vietnamese speech translation model
on multiple Qualcomm chipsets via the AI Hub Workbench.

Workflow per device:
  1. Compile both ONNX sub-models (audio_encoder, diffusion_backbone) for the target device
  2. Preprocess test audio → numpy tensors
  3. Run audio_encoder inference → get audio embeddings
  4. Run diffusion_backbone inference (iterative denoising loop) → get logits
  5. Decode logits → text output
  6. Profile both sub-models and collect latency metrics
  7. Produce a summary report comparing chipsets

Usage:
  python scripts/qualcomm-job/test_inference_multi_chipset.py
  python scripts/qualcomm-job/test_inference_multi_chipset.py --devices "Samsung Galaxy S24 (Family)" "Snapdragon X Elite CRD"
  python scripts/qualcomm-job/test_inference_multi_chipset.py --runtime onnx --audio test/test_data/test_sample.mp3
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

# Add src to path for model configuration
sys.path.insert(0, os.path.abspath("src"))

# ─────────────────────────────── Constants ────────────────────────────────
# Default chipsets to test, covering a range of tiers
DEFAULT_DEVICES = [
    # Flagship Mobile 2025-2026
    "Samsung Galaxy S25 (Family)",
    "Samsung Galaxy S26 (Family)",
    # Flagship Mobile 2024
    "Samsung Galaxy S24 (Family)",
    # Flagship Mobile 2023
    "Samsung Galaxy S23 (Family)",
    # Compute / Laptop
    "Snapdragon X Elite CRD",
    # QRD Reference Boards
    "Snapdragon 8 Elite QRD",
]

AUDIO_SAMPLE_RATE = 16000
MAX_SEQ_LEN = 32
DIFFUSION_STEPS = 10  # Number of denoising iterations

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
            # Simple resampling fallback
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        return audio.astype(np.float32)


def prepare_audio_inputs(audio: np.ndarray, stride: int = 80) -> dict:
    """Prepare audio inputs for the audio_encoder ONNX model.
    
    The Moonshine encoder expects:
      - audio_features: (batch, audio_len) float32  — audio_len must be divisible by stride
      - audio_attention_mask: (batch, audio_len) int64
    
    The QAIRT converter requires fixed reshape dimensions, so audio_len must be
    a multiple of the encoder's convolutional stride (default 80 for Moonshine).
    We pad with zeros and set the attention mask accordingly.
    """
    original_len = len(audio)
    # Pad to nearest multiple of stride
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


def prepare_backbone_inputs(
    audio_embeds: np.ndarray,
    tokenizer,
    mask_id: int,
    eos_id: int,
    pad_id: int,
    seq_len: int = MAX_SEQ_LEN,
) -> dict:
    """Prepare inputs for the diffusion_backbone ONNX model.
    
    Creates a fully-masked canvas for the decoder to denoise.
    Returns numpy arrays matching the ONNX input spec.
    """
    batch_size = 1
    # Build token canvas: [MASK]*seq_len + [EOS]
    # We use a simple approach: all tokens are MASK, partial_mask all False (all generated)
    tokens = np.full((batch_size, seq_len), mask_id, dtype=np.int64)
    tokens[0, -1] = eos_id  # last token is EOS

    # partial_mask: False = generated, True = fixed source
    # For inference: everything is generated (we have no source prefix in this simple test)
    partial_mask = np.zeros((batch_size, seq_len), dtype=bool)

    # Audio embeddings from encoder
    audio_len = audio_embeds.shape[1]
    precomputed_audio_mask = np.ones((batch_size, audio_len), dtype=np.int32)

    return {
        "prev_output_tokens": tokens,
        "partial_mask": partial_mask,
        "precomputed_audio_embeds": audio_embeds.astype(np.float32),
        "precomputed_audio_mask": precomputed_audio_mask,
    }


def format_duration(seconds: float) -> str:
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    return f"{mins}m {secs:.1f}s"


def print_banner():
    print("=" * 75)
    print("  QUALCOMM AI HUB — MULTI-CHIPSET INFERENCE BENCHMARK")
    print("  Model: aiai-laboratory/onnx-diffusion-speech-translation-from-vi-v1")
    print("=" * 75)


# ───────────────────────── Main Testing Logic ─────────────────────────────

def test_device(
    hub,
    device_name: str,
    audio_inputs: dict,
    tokenizer,
    config,
    runtime: str = "onnx",
    skip_repackage: bool = True,
) -> dict:
    """Run full compile → inference → profile pipeline on a single device.
    
    Returns a result dict with status, latency, output text, errors, etc.
    """
    result = {
        "device": device_name,
        "runtime": runtime,
        "status": "pending",
        "audio_encoder": {"compile": None, "inference": None, "profile": None},
        "diffusion_backbone": {"compile": None, "inference": None, "profile": None},
        "output_text": None,
        "errors": [],
        "total_time_s": 0,
    }
    
    t_start = time.time()
    
    try:
        device = hub.Device(device_name)
        print(f"\n{'─'*70}")
        print(f"  Device: {device.name}")
        print(f"  Runtime: {runtime}")
        print(f"{'─'*70}")
    except Exception as e:
        result["status"] = "device_not_found"
        result["errors"].append(f"Device not found: {e}")
        print(f"  [!] Device '{device_name}' not found, skipping.")
        return result
    
    # Compile options
    if runtime == "qnn":
        target_runtime = "qnn_context_binary"
    else:
        target_runtime = "precompiled_qnn_onnx"
    compile_options = f"--target_runtime {target_runtime} --truncate_64bit_io"
    
    # ──── 1. Compile Audio Encoder ────
    print(f"\n  [1/6] Compiling audio_encoder for {device_name}...")
    audio_encoder_specs = {
        "audio_features": tuple(audio_inputs["audio_features"].shape),
        "audio_attention_mask": (tuple(audio_inputs["audio_attention_mask"].shape), "int64"),
    }
    
    try:
        audio_compile_job = hub.submit_compile_job(
            model="onnx/audio_encoder_pkg.onnx",
            device=device,
            input_specs=audio_encoder_specs,
            options=compile_options,
            name=f"audio_enc_{device_name[:20]}_{runtime}",
        )
        print(f"        Job URL: {audio_compile_job.url}")
        
        # Wait for compilation
        audio_compile_job.wait()
        status = audio_compile_job.get_status()
        if status.code != "SUCCESS":
            result["audio_encoder"]["compile"] = "FAILED"
            result["errors"].append(f"audio_encoder compile failed: {status.message}")
            print(f"        [!] FAILED: {status.message}")
            result["status"] = "compile_failed"
            result["total_time_s"] = time.time() - t_start
            return result
        
        audio_target_model = audio_compile_job.get_target_model()
        result["audio_encoder"]["compile"] = "SUCCESS"
        print(f"        [+] Compilation successful!")
    except Exception as e:
        result["audio_encoder"]["compile"] = "ERROR"
        result["errors"].append(f"audio_encoder compile error: {e}")
        print(f"        [!] Error: {e}")
        result["status"] = "compile_error"
        result["total_time_s"] = time.time() - t_start
        return result
    
    # ──── 2. Run Audio Encoder Inference ────
    print(f"  [2/6] Running audio_encoder inference...")
    try:
        audio_inference_job = hub.submit_inference_job(
            model=audio_target_model,
            device=device,
            inputs={
                "audio_features": [audio_inputs["audio_features"]],
                "audio_attention_mask": [
                    audio_inputs["audio_attention_mask"].astype(np.int32)
                    if "--truncate_64bit_io" in compile_options
                    else audio_inputs["audio_attention_mask"]
                ],
            },
            name=f"audio_inf_{device_name[:20]}",
        )
        print(f"        Job URL: {audio_inference_job.url}")
        
        audio_inference_job.wait()
        status = audio_inference_job.get_status()
        if status.code != "SUCCESS":
            result["audio_encoder"]["inference"] = "FAILED"
            result["errors"].append(f"audio_encoder inference failed: {status.message}")
            print(f"        [!] FAILED: {status.message}")
            result["status"] = "inference_failed"
            result["total_time_s"] = time.time() - t_start
            return result
        
        audio_output = audio_inference_job.download_output_data()
        # The output key is typically "last_hidden_state"
        if isinstance(audio_output, dict):
            audio_embeds = list(audio_output.values())[0]
            if isinstance(audio_embeds, list):
                audio_embeds = audio_embeds[0]
        else:
            audio_embeds = audio_output
        
        result["audio_encoder"]["inference"] = "SUCCESS"
        print(f"        [+] Audio embeddings shape: {audio_embeds.shape}")
    except Exception as e:
        result["audio_encoder"]["inference"] = "ERROR"
        result["errors"].append(f"audio_encoder inference error: {e}")
        print(f"        [!] Error: {e}")
        traceback.print_exc()
        result["status"] = "inference_error"
        result["total_time_s"] = time.time() - t_start
        return result
    
    # ──── 3. Compile Diffusion Backbone ────
    print(f"  [3/6] Compiling diffusion_backbone for {device_name}...")
    
    backbone_inputs = prepare_backbone_inputs(
        audio_embeds=audio_embeds,
        tokenizer=tokenizer,
        mask_id=config.mask_token_id,
        eos_id=config.eos_token_id,
        pad_id=config.pad_token_id,
        seq_len=MAX_SEQ_LEN,
    )
    
    backbone_specs = {
        "prev_output_tokens": (tuple(backbone_inputs["prev_output_tokens"].shape), "int64"),
        "partial_mask": (tuple(backbone_inputs["partial_mask"].shape), "bool"),
        "precomputed_audio_embeds": tuple(backbone_inputs["precomputed_audio_embeds"].shape),
        "precomputed_audio_mask": (tuple(backbone_inputs["precomputed_audio_mask"].shape), "int32"),
    }
    
    try:
        backbone_compile_job = hub.submit_compile_job(
            model="onnx/diffusion_backbone_pkg.onnx",
            device=device,
            input_specs=backbone_specs,
            options=compile_options,
            name=f"backbone_{device_name[:20]}_{runtime}",
        )
        print(f"        Job URL: {backbone_compile_job.url}")
        
        backbone_compile_job.wait()
        status = backbone_compile_job.get_status()
        if status.code != "SUCCESS":
            result["diffusion_backbone"]["compile"] = "FAILED"
            result["errors"].append(f"diffusion_backbone compile failed: {status.message}")
            print(f"        [!] FAILED: {status.message}")
            result["status"] = "compile_failed"
            result["total_time_s"] = time.time() - t_start
            return result
        
        backbone_target_model = backbone_compile_job.get_target_model()
        result["diffusion_backbone"]["compile"] = "SUCCESS"
        print(f"        [+] Compilation successful!")
    except Exception as e:
        result["diffusion_backbone"]["compile"] = "ERROR"
        result["errors"].append(f"diffusion_backbone compile error: {e}")
        print(f"        [!] Error: {e}")
        result["status"] = "compile_error"
        result["total_time_s"] = time.time() - t_start
        return result
    
    # ──── 4. Run Diffusion Backbone Inference (single step for now) ────
    print(f"  [4/6] Running diffusion_backbone inference (1 denoising step)...")
    try:
        backbone_inference_job = hub.submit_inference_job(
            model=backbone_target_model,
            device=device,
            inputs={
                "prev_output_tokens": [
                    backbone_inputs["prev_output_tokens"].astype(np.int32)
                    if "--truncate_64bit_io" in compile_options
                    else backbone_inputs["prev_output_tokens"]
                ],
                "partial_mask": [backbone_inputs["partial_mask"]],
                "precomputed_audio_embeds": [backbone_inputs["precomputed_audio_embeds"]],
                "precomputed_audio_mask": [backbone_inputs["precomputed_audio_mask"]],
            },
            name=f"backbone_inf_{device_name[:20]}",
        )
        print(f"        Job URL: {backbone_inference_job.url}")
        
        backbone_inference_job.wait()
        status = backbone_inference_job.get_status()
        if status.code != "SUCCESS":
            result["diffusion_backbone"]["inference"] = "FAILED"
            result["errors"].append(f"diffusion_backbone inference failed: {status.message}")
            print(f"        [!] FAILED: {status.message}")
            result["status"] = "inference_failed"
            result["total_time_s"] = time.time() - t_start
            return result
        
        backbone_output = backbone_inference_job.download_output_data()
        if isinstance(backbone_output, dict):
            logits = list(backbone_output.values())[0]
            if isinstance(logits, list):
                logits = logits[0]
        else:
            logits = backbone_output
        
        result["diffusion_backbone"]["inference"] = "SUCCESS"
        print(f"        [+] Logits shape: {logits.shape}")
        
        # Decode output tokens
        predicted_ids = np.argmax(logits, axis=-1)  # (batch, seq_len)
        output_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        result["output_text"] = output_text
        print(f"        [+] Decoded output: '{output_text}'")
    except Exception as e:
        result["diffusion_backbone"]["inference"] = "ERROR"
        result["errors"].append(f"diffusion_backbone inference error: {e}")
        print(f"        [!] Error: {e}")
        traceback.print_exc()
        result["status"] = "inference_error"
        result["total_time_s"] = time.time() - t_start
        return result
    
    # ──── 5. Profile Audio Encoder ────
    print(f"  [5/6] Profiling audio_encoder...")
    try:
        audio_profile_job = hub.submit_profile_job(
            model=audio_target_model,
            device=device,
            name=f"audio_prof_{device_name[:20]}",
        )
        audio_profile_job.wait()
        status = audio_profile_job.get_status()
        if status.code == "SUCCESS":
            profile_data = audio_profile_job.download_profile()
            result["audio_encoder"]["profile"] = profile_data
            # Try to extract latency
            if isinstance(profile_data, dict):
                summary = profile_data.get("execution_summary", {})
                latency_us = summary.get("estimated_inference_time", 0)
                print(f"        [+] Estimated latency: {latency_us/1000:.2f} ms")
            else:
                print(f"        [+] Profile data type: {type(profile_data)}")
        else:
            result["audio_encoder"]["profile"] = f"FAILED: {status.message}"
            print(f"        [!] Profile failed: {status.message}")
    except Exception as e:
        result["audio_encoder"]["profile"] = f"ERROR: {e}"
        print(f"        [!] Profile error: {e}")
    
    # ──── 6. Profile Diffusion Backbone ────
    print(f"  [6/6] Profiling diffusion_backbone...")
    try:
        backbone_profile_job = hub.submit_profile_job(
            model=backbone_target_model,
            device=device,
            name=f"backbone_prof_{device_name[:20]}",
        )
        backbone_profile_job.wait()
        status = backbone_profile_job.get_status()
        if status.code == "SUCCESS":
            profile_data = backbone_profile_job.download_profile()
            result["diffusion_backbone"]["profile"] = profile_data
            if isinstance(profile_data, dict):
                summary = profile_data.get("execution_summary", {})
                latency_us = summary.get("estimated_inference_time", 0)
                print(f"        [+] Estimated latency: {latency_us/1000:.2f} ms")
            else:
                print(f"        [+] Profile data type: {type(profile_data)}")
        else:
            result["diffusion_backbone"]["profile"] = f"FAILED: {status.message}"
            print(f"        [!] Profile failed: {status.message}")
    except Exception as e:
        result["diffusion_backbone"]["profile"] = f"ERROR: {e}"
        print(f"        [!] Profile error: {e}")
    
    result["status"] = "success"
    result["total_time_s"] = time.time() - t_start
    print(f"\n  ✓ Device {device_name} completed in {format_duration(result['total_time_s'])}")
    return result


def extract_latency(profile_data) -> str:
    """Extract latency from profile data, return human-readable string."""
    if profile_data is None:
        return "N/A"
    if isinstance(profile_data, str):
        return profile_data
    if isinstance(profile_data, dict):
        summary = profile_data.get("execution_summary", {})
        latency_us = summary.get("estimated_inference_time", 0)
        if latency_us > 0:
            return f"{latency_us/1000:.2f} ms"
    return "N/A"


def print_results_table(results: list):
    """Print a formatted comparison table of all device results."""
    print("\n" + "=" * 95)
    print("                        MULTI-CHIPSET BENCHMARK RESULTS")
    print("=" * 95)
    
    # Header
    print(f"{'Device':<35} {'Status':<15} {'Encoder':<12} {'Backbone':<12} {'Output':<20}")
    print(f"{'':─<35} {'':─<15} {'':─<12} {'':─<12} {'':─<20}")
    
    for r in results:
        device = r["device"][:33]
        status = r["status"]
        enc_lat = extract_latency(r["audio_encoder"].get("profile"))
        bb_lat = extract_latency(r["diffusion_backbone"].get("profile"))
        output = (r.get("output_text") or "")[:18]
        
        # Status emoji
        if status == "success":
            status_str = "✅ Success"
        elif "compile" in status:
            status_str = "❌ Compile"
        elif "inference" in status:
            status_str = "⚠️  Inference"
        elif "not_found" in status:
            status_str = "🔍 Not Found"
        else:
            status_str = f"❓ {status}"
        
        print(f"  {device:<33} {status_str:<15} {enc_lat:<12} {bb_lat:<12} {output:<20}")
    
    # Errors section
    errors_found = False
    for r in results:
        if r["errors"]:
            if not errors_found:
                print(f"\n{'─'*95}")
                print("ERRORS:")
                errors_found = True
            print(f"\n  {r['device']}:")
            for err in r["errors"]:
                print(f"    • {err[:120]}")
    
    print("\n" + "=" * 95)


# ───────────────────────────── Main ───────────────────────────────────────

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Test speech translation inference on multiple Qualcomm chipsets"
    )
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help="List of device names to test. If not specified, uses default selection."
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
        "--list-devices", action="store_true",
        help="Just list all available devices and exit"
    )
    parser.add_argument(
        "--skip-repackage", action="store_true", default=True,
        help="Skip ONNX model repackaging (assumes already done)"
    )
    parser.add_argument(
        "--output", type=str, default="onnx/benchmark_results.json",
        help="Path to save JSON results"
    )
    args = parser.parse_args()
    
    # Load environment
    dotenv.load_dotenv()
    token = os.getenv("QUALCOMM_TOKEN")
    if not token:
        print("[!] Error: QUALCOMM_TOKEN not found in .env")
        sys.exit(1)
    os.environ["QAI_HUB_API_TOKEN"] = token
    
    import qai_hub as hub
    
    # List devices mode
    if args.list_devices:
        print("\n[*] Available Qualcomm AI Hub Devices:")
        print("─" * 50)
        devices = hub.get_devices()
        for i, d in enumerate(devices, 1):
            print(f"  {i:3d}. {d.name}")
        print(f"\nTotal: {len(devices)} devices")
        return
    
    # Load tokenizer and config for decoding
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
    print(f"    Audio duration: {len(audio)/AUDIO_SAMPLE_RATE:.2f}s")
    print(f"    Audio shape: {audio_inputs['audio_features'].shape}")
    
    # Load ground truth if available
    gt_path = args.audio.replace(".mp3", ".json").replace(".wav", ".json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        print(f"    Ground truth (vi): {gt.get('text', 'N/A')}")
        print(f"    Ground truth (en): {gt.get('english', 'N/A')}")
    
    # Check ONNX packages exist
    for pkg in ["onnx/audio_encoder_pkg.onnx", "onnx/diffusion_backbone_pkg.onnx"]:
        if not os.path.exists(pkg):
            print(f"[!] Missing ONNX package: {pkg}")
            print("    Run the repackaging step first: python scripts/qualcomm-job/submit_qualcomm_job.py")
            sys.exit(1)
    
    # Determine devices
    device_names = args.devices if args.devices else DEFAULT_DEVICES
    print(f"\n[*] Testing {len(device_names)} devices:")
    for d in device_names:
        print(f"    • {d}")
    
    # Run tests
    results = []
    for i, device_name in enumerate(device_names, 1):
        print(f"\n{'═'*75}")
        print(f"  [{i}/{len(device_names)}] Testing: {device_name}")
        print(f"{'═'*75}")
        
        result = test_device(
            hub=hub,
            device_name=device_name,
            audio_inputs=audio_inputs,
            tokenizer=tokenizer,
            config=config,
            runtime=args.runtime,
            skip_repackage=args.skip_repackage,
        )
        results.append(result)
    
    # Print summary table
    print_results_table(results)
    
    # Save results to JSON (profile data might not be serializable, sanitize)
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, bytes):
            return "<binary>"
        return obj
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "audio_file": args.audio,
            "runtime": args.runtime,
            "results": sanitize_for_json(results),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[+] Results saved to: {args.output}")
    
    # Summary stats
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n[*] Summary: {success_count}/{len(results)} devices completed successfully")
    print("=" * 75)


if __name__ == "__main__":
    main()
