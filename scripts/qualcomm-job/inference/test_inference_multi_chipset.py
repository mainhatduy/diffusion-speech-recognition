"""
Multi-Chipset Inference Test for Diffusion Speech Translation Model (Asynchronous)
==================================================================================
This script tests end-to-end inference of the Vietnamese speech translation model
on multiple Qualcomm chipsets via the AI Hub Workbench.

To speed up the benchmarking process, it runs all operations asynchronously:
  1. Precomputes audio embeddings locally using ONNX Runtime to decouple the sub-models.
  2. Submits compile jobs for all target devices concurrently.
  3. Periodically polls the status of compilation jobs.
  4. Submits inference and profiling jobs for all devices concurrently.
  5. Periodically polls the evaluation jobs.
  6. Collects, decodes, and summarizes the results.

Usage:
  python scripts/qualcomm-job/inference/test_inference_multi_chipset.py
  python scripts/qualcomm-job/inference/test_inference_multi_chipset.py --devices "Samsung Galaxy S24 (Family)" "Snapdragon X Elite CRD"
  python scripts/qualcomm-job/inference/test_inference_multi_chipset.py --runtime onnx --audio test/test_data/test_sample.mp3
"""

import os
import sys
import time
import json
import argparse
import traceback
from datetime import datetime

import numpy as np

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
)

# ─────────────────────────────── Constants ────────────────────────────────
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

DIFFUSION_STEPS = 10  # Number of denoising iterations


def format_duration(seconds: float) -> str:
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    return f"{mins}m {secs:.1f}s"


def print_banner():
    print("=" * 75)
    print("  QUALCOMM AI HUB — ASYNC MULTI-CHIPSET INFERENCE BENCHMARK")
    print("  Model: aiai-laboratory/onnx-diffusion-speech-translation-from-vi-v1")
    print("=" * 75)


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
        "--output", type=str, default="onnx/benchmark_results.json",
        help="Path to save JSON results"
    )
    args = parser.parse_args()
    
    # Load environment
    setup_qualcomm_token()
    
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
            print("    Run the repackaging step first: python scripts/qualcomm-job/patches/repackage_models.py")
            sys.exit(1)
            
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
    import onnxruntime as ort
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
    print(f"    [+] Precomputed audio embeddings shape: {audio_embeds.shape}")
    
    # Prepare backbone inputs
    backbone_inputs = prepare_backbone_inputs(
        audio_embeds=audio_embeds,
        audio_len=audio_embeds.shape[1],
        mask_id=config.mask_token_id,
        eos_id=config.eos_token_id,
        seq_len=MAX_SEQ_LEN,
    )
    
    # Determine devices
    device_names = args.devices if args.devices else DEFAULT_DEVICES
    print(f"\n[*] Testing {len(device_names)} devices:")
    for d in device_names:
        print(f"    • {d}")
        
    t_start = time.time()
    
    # Compilation options
    runtime = args.runtime
    if runtime == "qnn":
        target_runtime = "qnn_context_binary"
    else:
        target_runtime = "precompiled_qnn_onnx"
    compile_options = f"--target_runtime {target_runtime} --truncate_64bit_io"
    
    audio_encoder_specs = {
        "audio_features": tuple(audio_inputs["audio_features"].shape),
        "audio_attention_mask": (tuple(audio_inputs["audio_attention_mask"].shape), "int64"),
    }
    
    backbone_specs = {
        "prev_output_tokens": (tuple(backbone_inputs["prev_output_tokens"].shape), "int64"),
        "precomputed_audio_embeds": tuple(backbone_inputs["precomputed_audio_embeds"].shape),
        "precomputed_audio_mask": (tuple(backbone_inputs["precomputed_audio_mask"].shape), "int32"),
    }
    
    # ──── 1. Submit Compilation Jobs ────
    print(f"\n[*] Submitting compile jobs to Qualcomm AI Hub (runtime: {runtime})...")
    compile_jobs = {}
    
    for device_name in device_names:
        try:
            device = hub.Device(device_name)
        except Exception as e:
            print(f"    [!] Device '{device_name}' not found, skipping.")
            continue
            
        print(f"    • Submitting compilation for {device_name}...")
        
        # Audio Encoder Compile Job
        audio_job = None
        try:
            audio_job = hub.submit_compile_job(
                model="onnx/audio_encoder_pkg.onnx",
                device=device,
                input_specs=audio_encoder_specs,
                options=compile_options,
                name=f"audio_enc_{device_name[:20]}_{runtime}",
            )
            print(f"      - audio_encoder job: {audio_job.url}")
        except Exception as e:
            print(f"      [!] audio_encoder submission failed: {e}")
            
        # Backbone Compile Job
        backbone_job = None
        try:
            backbone_job = hub.submit_compile_job(
                model="onnx/diffusion_backbone_pkg.onnx",
                device=device,
                input_specs=backbone_specs,
                options=compile_options,
                name=f"backbone_{device_name[:20]}_{runtime}",
            )
            print(f"      - diffusion_backbone job: {backbone_job.url}")
        except Exception as e:
            print(f"      [!] diffusion_backbone submission failed: {e}")
            
        compile_jobs[device_name] = {
            "device_obj": device,
            "audio": audio_job,
            "backbone": backbone_job
        }
        
    # ──── 2. Monitor Compilation Jobs ────
    print("\n[*] Monitoring compilation jobs...")
    pending_compiles = {}
    for dev_name, jobs in compile_jobs.items():
        if jobs["audio"]:
            pending_compiles[f"{dev_name} (audio)"] = jobs["audio"]
        if jobs["backbone"]:
            pending_compiles[f"{dev_name} (backbone)"] = jobs["backbone"]
            
    while pending_compiles:
        done_keys = []
        for name, job in list(pending_compiles.items()):
            status = job.get_status()
            code = status.code
            if code in ["SUCCESS", "FAILED"]:
                done_keys.append(name)
                if code == "SUCCESS":
                    print(f"    [+] {name} compile completed successfully!")
                else:
                    print(f"    [!] {name} compile failed: {status.message}")
            else:
                pass
        for k in done_keys:
            del pending_compiles[k]
        if pending_compiles:
            time.sleep(15)
            
    # ──── 3. Submit Inference and Profiling Jobs ────
    print("\n[*] Submitting inference and profiling jobs...")
    eval_jobs = {}
    
    for device_name, jobs in compile_jobs.items():
        device = jobs["device_obj"]
        audio_job = jobs["audio"]
        backbone_job = jobs["backbone"]
        
        audio_success = False
        audio_target_model = None
        if audio_job:
            try:
                status = audio_job.get_status()
                if status.code == "SUCCESS":
                    audio_success = True
                    audio_target_model = audio_job.get_target_model()
            except Exception:
                pass
                
        backbone_success = False
        backbone_target_model = None
        if backbone_job:
            try:
                status = backbone_job.get_status()
                if status.code == "SUCCESS":
                    backbone_success = True
                    backbone_target_model = backbone_job.get_target_model()
            except Exception:
                pass
                
        eval_jobs[device_name] = {
            "audio_success": audio_success,
            "backbone_success": backbone_success,
            "audio_inf": None,
            "backbone_inf": None,
            "audio_prof": None,
            "backbone_prof": None,
            "errors": []
        }
        
        if not audio_success and not backbone_success:
            print(f"    [!] Skipping evaluation for {device_name} (both compilation jobs failed).")
            continue
            
        print(f"    • Submitting evaluation for {device_name}...")
        
        if audio_success:
            try:
                # Audio Encoder Inference
                inf_inputs = {
                    "audio_features": [audio_inputs["audio_features"]],
                    "audio_attention_mask": [
                        audio_inputs["audio_attention_mask"].astype(np.int32)
                        if "--truncate_64bit_io" in compile_options
                        else audio_inputs["audio_attention_mask"]
                    ],
                }
                eval_jobs[device_name]["audio_inf"] = hub.submit_inference_job(
                    model=audio_target_model,
                    device=device,
                    inputs=inf_inputs,
                    name=f"audio_inf_{device_name[:20]}",
                )
            except Exception as e:
                eval_jobs[device_name]["errors"].append(f"Audio inference submission failed: {e}")
                print(f"      [!] Audio inference submission failed: {e}")
                
            try:
                # Audio Encoder Profiling
                eval_jobs[device_name]["audio_prof"] = hub.submit_profile_job(
                    model=audio_target_model,
                    device=device,
                    name=f"audio_prof_{device_name[:20]}",
                )
            except Exception as e:
                eval_jobs[device_name]["errors"].append(f"Audio profiling submission failed: {e}")
                print(f"      [!] Audio profiling submission failed: {e}")
                
        if backbone_success:
            try:
                # Backbone Inference
                backbone_inf_inputs = {
                    "prev_output_tokens": [
                        backbone_inputs["prev_output_tokens"].astype(np.int32)
                        if "--truncate_64bit_io" in compile_options
                        else backbone_inputs["prev_output_tokens"]
                    ],
                    "precomputed_audio_embeds": [backbone_inputs["precomputed_audio_embeds"]],
                    "precomputed_audio_mask": [backbone_inputs["precomputed_audio_mask"]],
                }
                eval_jobs[device_name]["backbone_inf"] = hub.submit_inference_job(
                    model=backbone_target_model,
                    device=device,
                    inputs=backbone_inf_inputs,
                    name=f"backbone_inf_{device_name[:20]}",
                )
            except Exception as e:
                eval_jobs[device_name]["errors"].append(f"Backbone inference submission failed: {e}")
                print(f"      [!] Backbone inference submission failed: {e}")
                
            try:
                # Backbone Profiling
                eval_jobs[device_name]["backbone_prof"] = hub.submit_profile_job(
                    model=backbone_target_model,
                    device=device,
                    name=f"backbone_prof_{device_name[:20]}",
                )
            except Exception as e:
                eval_jobs[device_name]["errors"].append(f"Backbone profiling submission failed: {e}")
                print(f"      [!] Backbone profiling submission failed: {e}")
                
    # ──── 4. Monitor Inference and Profiling Jobs ────
    print("\n[*] Monitoring inference and profiling jobs...")
    pending_evals = {}
    for dev_name, jobs in eval_jobs.items():
        for job_type in ["audio_inf", "backbone_inf", "audio_prof", "backbone_prof"]:
            job = jobs[job_type]
            if job:
                pending_evals[f"{dev_name} ({job_type})"] = job
                
    while pending_evals:
        done_keys = []
        for name, job in list(pending_evals.items()):
            status = job.get_status()
            code = status.code
            if code in ["SUCCESS", "FAILED"]:
                done_keys.append(name)
                if code == "SUCCESS":
                    print(f"    [+] {name} completed successfully!")
                else:
                    print(f"    [!] {name} failed: {status.message}")
            else:
                pass
        for k in done_keys:
            del pending_evals[k]
        if pending_evals:
            time.sleep(15)
            
    # ──── 5. Collect and Process Results ────
    print("\n[*] Processing results...")
    results = []
    
    for device_name, info in eval_jobs.items():
        result = {
            "device": device_name,
            "runtime": runtime,
            "status": "pending",
            "audio_encoder": {
                "compile": "SUCCESS" if info["audio_success"] else "FAILED",
                "inference": None,
                "profile": None
            },
            "diffusion_backbone": {
                "compile": "SUCCESS" if info["backbone_success"] else "FAILED",
                "inference": None,
                "profile": None
            },
            "output_text": None,
            "errors": list(info["errors"]),
            "total_time_s": 0,
        }
        
        # Audio Inference
        if info["audio_inf"]:
            try:
                status = info["audio_inf"].get_status()
                if status.code == "SUCCESS":
                    result["audio_encoder"]["inference"] = "SUCCESS"
                else:
                    result["audio_encoder"]["inference"] = "FAILED"
                    result["errors"].append(f"audio_encoder inference failed: {status.message}")
            except Exception as e:
                result["audio_encoder"]["inference"] = "ERROR"
                result["errors"].append(f"audio_encoder inference error: {e}")
        else:
            if info["audio_success"]:
                result["audio_encoder"]["inference"] = "SKIPPED"
                
        # Backbone Inference and decoding
        if info["backbone_inf"]:
            try:
                status = info["backbone_inf"].get_status()
                if status.code == "SUCCESS":
                    result["diffusion_backbone"]["inference"] = "SUCCESS"
                    backbone_output = info["backbone_inf"].download_output_data()
                    if isinstance(backbone_output, dict):
                        logits = list(backbone_output.values())[0]
                        if isinstance(logits, list):
                            logits = logits[0]
                    else:
                        logits = backbone_output
                        
                    predicted_ids = np.argmax(logits, axis=-1)
                    output_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                    result["output_text"] = output_text
                else:
                    result["diffusion_backbone"]["inference"] = "FAILED"
                    result["errors"].append(f"diffusion_backbone inference failed: {status.message}")
            except Exception as e:
                result["diffusion_backbone"]["inference"] = "ERROR"
                result["errors"].append(f"diffusion_backbone inference error: {e}")
        else:
            if info["backbone_success"]:
                result["diffusion_backbone"]["inference"] = "SKIPPED"
                
        # Audio Profile
        if info["audio_prof"]:
            try:
                status = info["audio_prof"].get_status()
                if status.code == "SUCCESS":
                    profile_data = info["audio_prof"].download_profile()
                    result["audio_encoder"]["profile"] = profile_data
                else:
                    result["audio_encoder"]["profile"] = f"FAILED: {status.message}"
            except Exception as e:
                result["audio_encoder"]["profile"] = f"ERROR: {e}"
                
        # Backbone Profile
        if info["backbone_prof"]:
            try:
                status = info["backbone_prof"].get_status()
                if status.code == "SUCCESS":
                    profile_data = info["backbone_prof"].download_profile()
                    result["diffusion_backbone"]["profile"] = profile_data
                else:
                    result["diffusion_backbone"]["profile"] = f"FAILED: {status.message}"
            except Exception as e:
                result["diffusion_backbone"]["profile"] = f"ERROR: {e}"
                
        # Final Status
        if not info["audio_success"] or not info["backbone_success"]:
            result["status"] = "compile_failed"
        elif (result["audio_encoder"]["inference"] == "SUCCESS" and 
              result["diffusion_backbone"]["inference"] == "SUCCESS"):
            result["status"] = "success"
        else:
            result["status"] = "inference_failed"
            
        results.append(result)
        
    total_duration = time.time() - t_start
    print(f"\n✓ All devices completed in {format_duration(total_duration)}")
    
    # Print comparison table
    print_results_table(results)
    
    # Save results to JSON
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
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n[*] Summary: {success_count}/{len(results)} devices completed successfully")
    print("=" * 75)


if __name__ == "__main__":
    main()
