#!/bin/bash
# Activate environment and run the Qualcomm AI Hub multi-chipset benchmark.

set -e

# Navigate to repository root
cd "/teamspace/studios/this_studio/diffusion-speech-recognition"

# Activate virtual environment using uv
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[!] Virtual environment (.venv) not found. Please make sure uv env is set up."
    exit 1
fi

# Run the benchmark
echo "[*] Launching Qualcomm AI Hub Benchmark on a single chipset..."
python scripts/qualcomm-job/inference/test_inference_multi_chipset.py \
  --runtime onnx \
  --audio test/test_data/test_sample.mp3 \
  --devices "Samsung Galaxy S25 (Family)" \
  --output onnx/benchmark_results.json \
  "$@"

echo "[+] Benchmark script finished. Results saved to onnx/benchmark_results.json."
