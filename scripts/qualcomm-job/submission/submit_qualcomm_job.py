import os
import sys
import time
import argparse
import qai_hub as hub

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import setup_qualcomm_token, repackage_model, monitor_jobs

def print_banner():
    print("=" * 70)
    print("      QUALCOMM AI HUB - EDGE COMPILATION & BENCHMARKING SYSTEM")
    print("=" * 70)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Submit Speech Translation models to Qualcomm AI Hub.")
    parser.add_argument("--device", type=str, default="Samsung Galaxy S25 (Family)", help="Qualcomm target device/family name")
    parser.add_argument("--runtime", type=str, choices=["qnn", "onnx"], default="qnn", help="Target runtime (qnn or onnx)")
    parser.add_argument("--skip-repackage", action="store_true", help="Skip repackaging ONNX models")
    args = parser.parse_args()
    
    # Load environment variables
    setup_qualcomm_token()

    
    # Initialize client
    print("[*] Initializing Qualcomm AI Hub client...")
    client = hub.Client()
    
    try:
        device = hub.Device(args.device)
        print(f"[+] Target device selected: {device.name}")
    except Exception as e:
        print(f"[!] Error selecting device '{args.device}': {e}")
        sys.exit(1)
        
    # Check and Repackage
    if not args.skip_repackage:
        print("\n--- Repackaging ONNX Models with Compliant External Weight Offloading ---")
        try:
            repackage_model(
                model_path="onnx/audio_encoder.onnx",
                output_dir="onnx/audio_encoder_pkg.onnx",
                model_name="audio_encoder.onnx",
                data_name="audio_encoder.data"
            )
            repackage_model(
                model_path="onnx/diffusion_backbone.onnx",
                output_dir="onnx/diffusion_backbone_pkg.onnx",
                model_name="diffusion_backbone.onnx",
                data_name="diffusion_backbone.data"
            )
        except Exception as e:
            print(f"[!] Repackaging failed: {e}")
            sys.exit(1)
            
    # Set compilation options
    if args.runtime == "qnn":
        target_runtime = "qnn_context_binary"
    else:
        target_runtime = "precompiled_qnn_onnx"
        
    compile_options = f"--target_runtime {target_runtime} --truncate_64bit_io"
    print(f"\n[*] Target Runtime: {target_runtime}")
    print(f"[*] Compile Options: {compile_options}")
    
    # Submit compile jobs
    compile_jobs = {}
    
    # 1. Audio Encoder
    print("\n--- Submitting Audio Encoder Compile Job ---")
    audio_encoder_specs = {
        "audio_features": (1, 2400),
        "audio_attention_mask": ((1, 2400), "int64")
    }
    try:
        audio_compile_job = hub.submit_compile_job(
            model="onnx/audio_encoder_pkg.onnx",
            device=device,
            input_specs=audio_encoder_specs,
            options=compile_options,
            name=f"audio_encoder_{args.runtime}"
        )
        compile_jobs["audio_encoder"] = audio_compile_job
        print(f"[+] Audio Encoder job submitted: {audio_compile_job.url}")
    except Exception as e:
        print(f"[!] Failed to submit Audio Encoder compilation: {e}")
        sys.exit(1)
        
    # 2. Diffusion Backbone
    print("\n--- Submitting Diffusion Backbone Compile Job ---")
    backbone_specs = {
        "prev_output_tokens": ((1, 32), "int64"),
        "precomputed_audio_embeds": (1, 96, 768),
        "precomputed_audio_mask": ((1, 96), "int32")
    }
    try:
        backbone_compile_job = hub.submit_compile_job(
            model="onnx/diffusion_backbone_pkg.onnx",
            device=device,
            input_specs=backbone_specs,
            options=compile_options,
            name=f"diffusion_backbone_{args.runtime}"
        )
        compile_jobs["diffusion_backbone"] = backbone_compile_job
        print(f"[+] Diffusion Backbone job submitted: {backbone_compile_job.url}")
    except Exception as e:
        print(f"[!] Failed to submit Diffusion Backbone compilation: {e}")
        sys.exit(1)
        
    # Monitor compilation
    failures = monitor_jobs(compile_jobs, label="Compilation Jobs")
    if any(failures.values()):
        print("\n[!] Compilation failed for one or more models. Details:")
        for name, err in failures.items():
            if err:
                print(f"  - {name}: {err}")
        sys.exit(1)
        
    # Download compiled models and submit profiling
    print("\n--- Downloading Compiled Assets & Submitting Profile Jobs ---")
    os.makedirs("onnx/compiled", exist_ok=True)
    
    profile_jobs = {}
    for name, job in compile_jobs.items():
        target_model = job.get_target_model()
        ext = "dlc" if args.runtime == "qnn" else "onnx"
        local_output_path = f"onnx/compiled/{name}_compiled.{ext}"
        
        print(f"[*] Downloading optimized {name} model...")
        target_model.download(local_output_path)
        print(f"    -> Saved compiled model to {local_output_path}")
        
        print(f"[*] Submitting profile job for {name} on {device.name}...")
        try:
            profile_job = hub.submit_profile_job(
                model=target_model,
                device=device,
                name=f"{name}_profile"
            )
            profile_jobs[name] = profile_job
            print(f"    -> Profile job submitted: {profile_job.url}")
        except Exception as e:
            print(f"    [!] Failed to submit profile job: {e}")
            
    # Monitor profiling
    failures_profile = monitor_jobs(profile_jobs, label="Profile Jobs")
    
    # Print profile summaries
    print("\n" + "="*70)
    print("                       PROFILING PERFORMANCE REPORT")
    print("="*70)
    
    for name, job in profile_jobs.items():
        if failures_profile[name]:
            print(f"\n[!] Profiling for {name} failed: {failures_profile[name]}")
            continue
            
        try:
            profile = job.download_profile()
            summary = profile.get("execution_summary", {})
            estimated_latency_us = summary.get("estimated_inference_time", 0)
            latency_ms = estimated_latency_us / 1000.0
            
            compute_units = profile.get("device_info", {}).get("compute_units", {})
            peak_mem = summary.get("peak_memory_bytes", 0) / (1024 * 1024)
            
            print(f"\nModel: {name.upper()}")
            print(f"  - Estimated Inference Latency: {latency_ms:.3f} ms")
            print(f"  - Peak Memory Consumption: {peak_mem:.2f} MB")
            print(f"  - Compute Units Active: {list(compute_units.keys())}")
            
            # Print breakdown if available
            layers = profile.get("layer_info", [])
            print(f"  - Total layers profiled: {len(layers)}")
        except Exception as e:
            print(f"\n[!] Failed to extract profiling metrics for {name}: {e}")
            
    print("\n" + "="*70)
    print("[+] All tasks completed successfully! Compiled artifacts are ready.")
    print("    Check the onnx/compiled/ directory.")
    print("="*70)

if __name__ == "__main__":
    main()
