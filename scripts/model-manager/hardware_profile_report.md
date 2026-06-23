# Hardware Profiling Report: Minimum System Requirements (Batch Size = 1)

This report details the measured hardware resource usage and performance metrics for the Discrete Diffusion Speech Translation model. These measurements can be used to demonstrate the viability of deploying the model on edge devices.

## 📋 Summary of Minimum Requirements

| Metric | Minimum Required | Recommended Configuration | Notes |
|---|---|---|---|
| **CPU Cores** | **1 Core** (Non-Real-Time) / **> 8 (requires GPU or optimization for real-time) Cores** (Real-Time) | **4 Cores** | Benchmarked on Intel Xeon @ 2.2GHz |
| **System RAM** | **4.53 GB** (approx. 4639 MB) | **4 GB** | Includes model weights and runtime memory buffers |
| **VRAM (GPU)** | **1.03 GB** (approx. 1054 MB) | **2 GB** | Measured on NVIDIA L4 (using FP16/BF16) |

---

## 💻 Benchmark System Specifications
- **Operating System:** Linux 6.8.0-1007-gcp
- **CPU Model:** Intel(R) Xeon(R) CPU @ 2.20GHz (8 logical cores)
- **Total System RAM:** 31.34 GB
- **GPU Name:** NVIDIA L4 (22.03 GB VRAM)

---

## 📊 Model Information
- **Hugging Face Repository:** `aiai-laboratory/diffusion-speech-translation-from-vi-v1`
- **Parameter Count:** 397,803,416 parameters
- **Theoretical Weight Size:**
  - FP32: **1517.50 MB**
  - FP16/BF16: **758.75 MB**
- **Model Load Memory Increment (RAM):** **3390.11 MB**

---

## ⚡ CPU Performance Benchmarks (Batch Size = 1)
Evaluated on translation task `<vi_en>` (`english`) with audio duration: `3.37s`.

| CPU Threads | Avg Latency (s) | Real-Time Factor (RTF) | Gen Speed (tokens/s) | Peak Process Memory (MB) |
|---|---|---|---|---|
| 1 | 10.679s | 3.169 | 1.2 | 4139.30 MB |
| 2 | 6.394s | 1.897 | 2.0 | 4186.55 MB |
| 4 | 5.503s | 1.633 | 2.4 | 4189.43 MB |
| 8 | 9.851s | 2.923 | 1.3 | 4193.05 MB |

*Note: Real-Time Factor (RTF) is the ratio of inference latency to audio duration. An RTF < 1.0 means the model runs faster than real-time (essential for live/edge streaming).*

---

## 🚀 GPU (CUDA) Performance Benchmarks (Batch Size = 1)
- **Device:** NVIDIA L4 (BF16 Precision)
- **Avg Latency:** 0.896s
- **Real-Time Factor (RTF):** 0.266
- **Generation Speed:** 14.5 tokens/s
- **Peak VRAM Allocated:** 797.55 MB (Weights + Activations)
- **Peak VRAM Reserved:** 960.00 MB
