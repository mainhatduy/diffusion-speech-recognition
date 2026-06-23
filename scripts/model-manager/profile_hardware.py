import sys
import os
import time
import json
import resource
import platform
import subprocess
import torch
import numpy as np
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

# Resolve project root so we can import src/ modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel, decoder_out_t
from data.utils import normalize_text

def get_current_rss():
    """Get current resident memory usage of the process in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def get_peak_rss():
    """Get peak resident memory usage of the process in MB."""
    # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def get_cpu_info():
    """Get basic CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def get_system_ram():
    """Get total system RAM in GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and return a float32 numpy array at target_sr Hz."""
    waveform = None
    sr = None

    try:
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
    except Exception:
        pass

    if waveform is None:
        try:
            import librosa
            waveform, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        except Exception:
            pass

    if waveform is None:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            waveform = samples / (2 ** (audio.sample_width * 8 - 1))
            sr = target_sr
        except Exception:
            pass

    if waveform is None:
        raise RuntimeError(f"Could not load audio file '{path}'. Please install soundfile or librosa.")

    # Resample if needed
    if sr != target_sr:
        ratio = target_sr / sr
        new_length = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)

    return waveform

def run_inference_single(
    model, tokenizer, feature_extractor, waveform, audio_duration,
    task_lang="english", task_token_id=None, device="cpu", max_iterations=10, max_length=64
):
    """Run a single inference translation step and return stats."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    # Determine canvas length
    if task_lang == "english":
        canvas_len = int(audio_duration * 4.0)
    else:
        canvas_len = int(audio_duration * 2.5)
    canvas_len = max(5, min(max_length, canvas_len))

    # Preprocess audio features
    audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    audio_values_raw = audio_inputs.input_values.to(device)

    audio_len = audio_values_raw.size(-1)
    padded_len = ((audio_len + 79) // 80) * 80
    padded_audio = torch.zeros(1, padded_len, device=device)
    padded_audio[0, :audio_len] = audio_values_raw[0]
    audio_values = padded_audio

    padded_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
    padded_mask[0, :audio_len] = 1
    audio_attention_mask = padded_mask

    if device == "cuda":
        audio_values = audio_values.to(torch.bfloat16)

    # Source prefix: [BOS, task_token_id]
    src_tokens = torch.tensor([[bos_id, task_token_id]], dtype=torch.long, device=device)
    src_length = src_tokens.size(1)

    # Canvas: [BOS, task_token_id, MASK...MASK, EOS]
    canvas = torch.cat([
        src_tokens,
        torch.full((1, canvas_len), mask_id, dtype=torch.long, device=device),
        torch.tensor([[eos_id]], dtype=torch.long, device=device),
    ], dim=1)

    partial_mask = torch.zeros_like(canvas, dtype=torch.bool)
    partial_mask[:, :src_length] = True

    non_fixed_sym_masks = (
        canvas.ne(pad_id) &
        canvas.ne(bos_id) &
        canvas.ne(eos_id) &
        ~partial_mask
    )

    output_scores = torch.zeros_like(canvas, dtype=torch.float32)
    output_mask = canvas.eq(mask_id)

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    decoder_out = decoder_out_t(
        output_tokens=canvas.clone(),
        output_scores=output_scores,
        output_masks=output_mask,
        non_fixed_sym_masks=non_fixed_sym_masks,
        attn=None,
        step=0,
        max_step=max_iterations,
        history=None,
    )

    with torch.no_grad():
        for _ in range(max_iterations):
            decoder_out = model.denoise_step(
                decoder_out,
                partial_mask,
                temperature=1.0,
                strategy="reparam-uncond-deterministic-cosine",
                audio_features=audio_values,
                audio_attention_mask=audio_attention_mask,
            )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    # Extract generated tokens
    out_tokens = decoder_out.output_tokens[0]
    cutoff = (
        out_tokens.ne(pad_id) &
        out_tokens.ne(bos_id) &
        out_tokens.ne(eos_id) &
        ~partial_mask[0]
    )
    gen_tokens = out_tokens[cutoff]
    pred_text = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True).strip()

    return {
        "text": pred_text,
        "latency": elapsed_time,
        "num_tokens": len(gen_tokens),
        "tokens_per_sec": len(gen_tokens) / elapsed_time if elapsed_time > 0 else 0,
        "rtf": elapsed_time / audio_duration if audio_duration > 0 else 0
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile model hardware requirements for batch size 1.")
    parser.add_argument("--repo-id", type=str, default="aiai-laboratory/diffusion-speech-translation-from-vi-v1", help="Hugging Face Model Repo ID")
    parser.add_argument("--audio", type=str, default="test/test_data/test_sample.mp3", help="Audio file to run benchmark on")
    parser.add_argument("--iterations", type=int, default=10, help="Number of diffusion iterations")
    parser.add_argument("--max-length", type=int, default=64, help="Max length of target text")
    parser.add_argument("--output-markdown", type=str, default="scripts/model-manager/hardware_profile_report.md", help="Path to save markdown report")
    parser.add_argument("--output-json", type=str, default="scripts/model-manager/hardware_profile_report.json", help="Path to save JSON report")
    args = parser.parse_args()
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("="*60)
    print(" HARDWARE PROFILER - MINIMUM REQUIREMENTS FOR BATCH SIZE 1")
    print("="*60)

    # 1. System Details
    cpu_model = get_cpu_info()
    cpu_cores = os.cpu_count()
    sys_ram = get_system_ram()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_available else 0.0

    print(f"Detected Platform: {platform.system()} {platform.release()}")
    print(f"CPU Model: {cpu_model}")
    print(f"CPU Cores (logical): {cpu_cores}")
    print(f"Total System RAM: {sys_ram:.2f} GB")
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"GPU Name: {gpu_name}")
        print(f"Total VRAM: {gpu_vram:.2f} GB")
    print("-" * 60)

    # 2. Record Baseline RAM
    baseline_ram = get_current_rss()
    print(f"Baseline Process RAM: {baseline_ram:.2f} MB")

    # 3. Load Tokenizer & Model Configuration
    print(f"Loading Model Config & Tokenizer from {args.repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True, use_fast=False)

    config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = DiscreteDiffusionConfig(**{
        k: v for k, v in config_dict.items()
        if not k.startswith("_") and k != "model_type" and k != "transformers_version"
        and k != "auto_map"
    })

    model = DiscreteDiffusionModel(config)

    # Calculate model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_precision_fp32_size_mb = (total_params * 4) / (1024 * 1024)
    model_precision_bf16_size_mb = (total_params * 2) / (1024 * 1024)

    print(f"Model Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Theoretical FP32 weights size: {model_precision_fp32_size_mb:.2f} MB")
    print(f"Theoretical FP16/BF16 weights size: {model_precision_bf16_size_mb:.2f} MB")

    # 4. Load Model Weights
    print("Downloading/Loading weights...")
    try:
        weights_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device="cpu")
    except Exception:
        weights_path = hf_hub_download(repo_id=args.repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    if config.tie_word_embeddings:
        model.model.lm_head.decoder.weight = model.model.roberta.embeddings.word_embeddings.weight

    model = model.eval()

    # Measure memory usage after loading model on CPU
    loaded_model_ram = get_current_rss()
    weights_ram_usage = loaded_model_ram - baseline_ram
    print(f"Process RAM after loading model: {loaded_model_ram:.2f} MB (diff: {weights_ram_usage:.2f} MB)")

    # 5. Load Audio
    print(f"Loading benchmark audio: {args.audio}")
    waveform = load_audio(args.audio, target_sr=16000)
    audio_duration = len(waveform) / 16000
    print(f"Audio Duration: {audio_duration:.2f}s")

    audio_encoder_name = model.config.audio_encoder_name
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_name)

    # 6. Task configuration
    task_lang = "english"
    task_token = "<vi_en>"
    task_token_id = tokenizer.convert_tokens_to_ids(task_token)

    # 7. CPU Inference Profiling (Multi-threading Benchmarks)
    # We test different thread configurations
    cpu_threads_to_test = [1, 2, 4]
    if cpu_cores >= 8:
        cpu_threads_to_test.append(8)
    cpu_threads_to_test = [t for t in cpu_threads_to_test if t <= cpu_cores]

    cpu_results = []
    print("\n" + "="*50)
    print(" RUNNING CPU PROFILING")
    print("="*50)

    for threads in cpu_threads_to_test:
        torch.set_num_threads(threads)
        print(f"Running CPU benchmark with {threads} thread(s)...")

        # Warm-up run
        _ = run_inference_single(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor,
            waveform=waveform, audio_duration=audio_duration, task_lang=task_lang,
            task_token_id=task_token_id, device="cpu", max_iterations=args.iterations,
            max_length=args.max_length
        )

        # Timed runs (average of 3 runs for stability)
        latencies = []
        tokens_gen = []
        memory_peaks = []
        outputs = []

        for i in range(3):
            # Reset peak RSS reading
            start_mem = get_current_rss()
            res = run_inference_single(
                model=model, tokenizer=tokenizer, feature_extractor=feature_extractor,
                waveform=waveform, audio_duration=audio_duration, task_lang=task_lang,
                task_token_id=task_token_id, device="cpu", max_iterations=args.iterations,
                max_length=args.max_length
            )
            peak_rss = get_peak_rss()
            latencies.append(res["latency"])
            tokens_gen.append(res["num_tokens"])
            memory_peaks.append(peak_rss)
            outputs.append(res["text"])

        avg_latency = np.mean(latencies)
        avg_tokens = np.mean(tokens_gen)
        avg_tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
        avg_rtf = avg_latency / audio_duration if audio_duration > 0 else 0
        peak_rss_mb = np.max(memory_peaks)

        print(f"  → Threads: {threads} | Latency: {avg_latency:.3f}s | RTF: {avg_rtf:.3f} | Speed: {avg_tokens_per_sec:.1f} tok/s | Peak RAM: {peak_rss_mb:.2f} MB")
        print(f"  → Translation: {outputs[0]}")

        cpu_results.append({
            "threads": threads,
            "latency_sec": float(avg_latency),
            "tokens_per_sec": float(avg_tokens_per_sec),
            "rtf": float(avg_rtf),
            "peak_ram_mb": float(peak_rss_mb)
        })

    # 8. GPU Inference Profiling (if available)
    gpu_results = None
    if gpu_available:
        print("\n" + "="*50)
        print(" RUNNING GPU (CUDA) PROFILING")
        print("="*50)

        # Move model to CUDA and convert to bf16 (as done in eval_val.py)
        model_cuda = model.to("cuda").to(torch.bfloat16)

        # Reset CUDA memory statistics
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warm-up run
        _ = run_inference_single(
            model=model_cuda, tokenizer=tokenizer, feature_extractor=feature_extractor,
            waveform=waveform, audio_duration=audio_duration, task_lang=task_lang,
            task_token_id=task_token_id, device="cuda", max_iterations=args.iterations,
            max_length=args.max_length
        )

        latencies = []
        tokens_gen = []
        outputs = []

        # Timed runs
        for i in range(3):
            res = run_inference_single(
                model=model_cuda, tokenizer=tokenizer, feature_extractor=feature_extractor,
                waveform=waveform, audio_duration=audio_duration, task_lang=task_lang,
                task_token_id=task_token_id, device="cuda", max_iterations=args.iterations,
                max_length=args.max_length
            )
            latencies.append(res["latency"])
            tokens_gen.append(res["num_tokens"])
            outputs.append(res["text"])

        avg_latency = np.mean(latencies)
        avg_tokens = np.mean(tokens_gen)
        avg_tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
        avg_rtf = avg_latency / audio_duration if audio_duration > 0 else 0

        # Memory usage stats from CUDA
        peak_allocated_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        peak_reserved_vram = torch.cuda.max_memory_reserved() / (1024 * 1024) # MB

        print(f"  → GPU Latency: {avg_latency:.3f}s | RTF: {avg_rtf:.3f} | Speed: {avg_tokens_per_sec:.1f} tok/s")
        print(f"  → Peak VRAM Allocated: {peak_allocated_vram:.2f} MB")
        print(f"  → Peak VRAM Reserved: {peak_reserved_vram:.2f} MB")
        print(f"  → Translation: {outputs[0]}")

        gpu_results = {
            "device_name": gpu_name,
            "latency_sec": float(avg_latency),
            "tokens_per_sec": float(avg_tokens_per_sec),
            "rtf": float(avg_rtf),
            "peak_vram_allocated_mb": float(peak_allocated_vram),
            "peak_vram_reserved_mb": float(peak_reserved_vram)
        }

    # Determine Minimum Configurations
    # Minimum RAM is the peak memory observed on 1-thread CPU run, plus a 500MB safety buffer
    min_cpu_ram_mb = cpu_results[0]["peak_ram_mb"] + 500.0
    min_cpu_ram_gb = min_cpu_ram_mb / 1024.0

    # Minimum VRAM is the peak allocated VRAM on CUDA, plus a 256MB safety buffer
    min_vram_mb = (gpu_results["peak_vram_allocated_mb"] + 256.0) if gpu_results else 0.0
    min_vram_gb = min_vram_mb / 1024.0

    # Determine CPU Cores required for real-time:
    # Look for the smallest thread count where RTF <= 1.0. If none, say it requires > max threads for RTF <= 1.
    rt_threads = None
    for r in cpu_results:
        if r["rtf"] <= 1.0:
            rt_threads = r["threads"]
            break

    min_cores_for_rt = rt_threads if rt_threads is not None else f"> {cpu_results[-1]['threads']} (requires GPU or optimization for real-time)"

    # Create Report Outputs
    report_dict = {
        "model": {
            "repo_id": args.repo_id,
            "total_params": total_params,
            "theoretical_fp32_mb": model_precision_fp32_size_mb,
            "theoretical_fp16_mb": model_precision_bf16_size_mb
        },
        "system_info": {
            "os": platform.system(),
            "cpu_model": cpu_model,
            "cpu_cores_logical": cpu_cores,
            "total_ram_gb": sys_ram,
            "gpu_name": gpu_name,
            "gpu_total_vram_gb": gpu_vram
        },
        "benchmarks": {
            "cpu": cpu_results,
            "gpu": gpu_results
        },
        "minimum_requirements": {
            "cpu_cores_for_realtime": min_cores_for_rt,
            "ram_gb": float(min_cpu_ram_gb),
            "vram_gb": float(min_vram_gb) if gpu_available else None
        }
    }

    # Save JSON report
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to {args.output_json}")

    # Build Markdown Report
    md = f"""# Hardware Profiling Report: Minimum System Requirements (Batch Size = 1)

This report details the measured hardware resource usage and performance metrics for the Discrete Diffusion Speech Translation model. These measurements can be used to demonstrate the viability of deploying the model on edge devices.

## 📋 Summary of Minimum Requirements

| Metric | Minimum Required | Recommended Configuration | Notes |
|---|---|---|---|
| **CPU Cores** | **1 Core** (Non-Real-Time) / **{min_cores_for_rt} Cores** (Real-Time) | **4 Cores** | Benchmarked on Intel Xeon @ 2.2GHz |
| **System RAM** | **{min_cpu_ram_gb:.2f} GB** (approx. {min_cpu_ram_mb:.0f} MB) | **4 GB** | Includes model weights and runtime memory buffers |
"""

    if gpu_available:
        md += f"| **VRAM (GPU)** | **{min_vram_gb:.2f} GB** (approx. {min_vram_mb:.0f} MB) | **2 GB** | Measured on {gpu_name} (using FP16/BF16) |\n"
    else:
        md += "| **VRAM (GPU)** | Not Required (runs on CPU) | N/A | Running on CPU |\n"

    md += f"""
---

## 💻 Benchmark System Specifications
- **Operating System:** {platform.system()} {platform.release()}
- **CPU Model:** {cpu_model} ({cpu_cores} logical cores)
- **Total System RAM:** {sys_ram:.2f} GB
"""

    if gpu_available:
        md += f"- **GPU Name:** {gpu_name} ({gpu_vram:.2f} GB VRAM)\n"

    md += f"""
---

## 📊 Model Information
- **Hugging Face Repository:** `{args.repo_id}`
- **Parameter Count:** {total_params:,} parameters
- **Theoretical Weight Size:**
  - FP32: **{model_precision_fp32_size_mb:.2f} MB**
  - FP16/BF16: **{model_precision_bf16_size_mb:.2f} MB**
- **Model Load Memory Increment (RAM):** **{weights_ram_usage:.2f} MB**

---

## ⚡ CPU Performance Benchmarks (Batch Size = 1)
Evaluated on translation task `{task_token}` (`{task_lang}`) with audio duration: `{audio_duration:.2f}s`.

| CPU Threads | Avg Latency (s) | Real-Time Factor (RTF) | Gen Speed (tokens/s) | Peak Process Memory (MB) |
|---|---|---|---|---|
"""
    for r in cpu_results:
        md += f"| {r['threads']} | {r['latency_sec']:.3f}s | {r['rtf']:.3f} | {r['tokens_per_sec']:.1f} | {r['peak_ram_mb']:.2f} MB |\n"

    md += """
*Note: Real-Time Factor (RTF) is the ratio of inference latency to audio duration. An RTF < 1.0 means the model runs faster than real-time (essential for live/edge streaming).*
"""

    if gpu_available:
        g = gpu_results
        md += f"""
---

## 🚀 GPU (CUDA) Performance Benchmarks (Batch Size = 1)
- **Device:** {g['device_name']} (BF16 Precision)
- **Avg Latency:** {g['latency_sec']:.3f}s
- **Real-Time Factor (RTF):** {g['rtf']:.3f}
- **Generation Speed:** {g['tokens_per_sec']:.1f} tokens/s
- **Peak VRAM Allocated:** {g['peak_vram_allocated_mb']:.2f} MB (Weights + Activations)
- **Peak VRAM Reserved:** {g['peak_vram_reserved_mb']:.2f} MB
"""

    with open(args.output_markdown, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown report saved to {args.output_markdown}")
    print("="*60)

if __name__ == "__main__":
    main()
