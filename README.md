# Diffusion Speech Recognition

A speech recognition and translation project using Discrete Diffusion Models to translate spoken Vietnamese into multiple target languages (English, Chinese, and Korean).

Technical Report:  [DiffusPeak Technical Teport](https://drive.google.com/file/d/1n-Iu1oxtDdeW_mhTzdTuLbpnmkkiVNK6)
---

## Methodology

### Data

The training data is based on the following resources:
* **Base Dataset**: [VietSpeech Dataset](https://huggingface.co/datasets/NhutP/VietSpeech)
* **Translation Process**: We use a translation model to translate from Vietnamese into other target languages.
* **Translation Model**: [tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B)
* **Translated Text Dataset**: [vietspeech-train-translated](https://huggingface.co/datasets/aiai-laboratory/vietspeech-train-translated)
* **Target Languages**: Translated into 3 languages: English (`en`), Chinese (`cn`), and Korean (`ko`).

### Model Architecture

The architecture consists of two primary components:
* **Backbone Architecture**: Diffusion Language Model (for details, see: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992), [arXiv:2508.15487](https://arxiv.org/abs/2508.15487), and the [OpenReview Forum](https://openreview.net/forum?id=6WnBITpnzD)).
* **Audio Encoder**: [Moonshine Streaming Medium](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)

---

## Result

### Benchmark

Model performance is benchmarked on the following validation set:
* **Validation Set**: [vietspeech-validation-translated](https://huggingface.co/datasets/aiai-laboratory/vietspeech-validation-translated) (10k samples)

---

## Environment Setup

We recommend using [uv](https://github.com/astral-sh/uv) to manage the Python virtual environment and project dependencies.

### ⚡ Method 1: Using `uv` (Recommended)

1. **Install `uv`**:
   * **Linux / macOS**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   * **Windows**:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

2. **Sync the Environment**:
   Navigate to the project directory and run:
   ```bash
   uv sync
   ```
   `uv` will automatically download the correct Python version (if missing), create the `.venv` directory, and install all required libraries from `pyproject.toml`.

3. **Run Scripts**:
   Run commands directly prefixing with `uv run` to auto-activate the environment:
   ```bash
   uv run python main.py
   ```

### 📦 Method 2: Using Default Python `venv`

1. **Install Prerequisites (Linux)**:
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-pip -y
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the Environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running Training

The repository provides pre-configured shell scripts under the `scripts/training/` directory to manage datasets and execute model training.

### 1. Run End-to-End Pipeline Training
This script handles downloading the precomputed dataset shards automatically before training.
```bash
# Run full end-to-end training pipeline
bash scripts/training/run_pipeline_end2end.sh
```

To run a fast validation test/dry-run (downloads a tiny test subset and runs training for 10 steps):
```bash
bash scripts/training/run_pipeline_end2end.sh --test
```

Specify target GPU device with `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/training/run_pipeline_end2end.sh
```

### 2. Run Standard Training with Custom Config
To launch training directly with a specific configuration JSON file:
```bash
bash scripts/training/speech_recognition_train.sh configs/speech_recognition_config.json
```

## Qualcomm Hardware Optimization and Deployment

For compilation, ONNX graph optimization/compatibility patching, packaging, and benchmark/inference job submission on **Qualcomm Snapdragon NPUs** using the Qualcomm AI Hub API, please refer to the dedicated guide:

👉 **[Qualcomm AI Hub Setup & Execution Guide](scripts/qualcomm-job/README.md)**

> [!IMPORTANT]
> **Hardware Validation:**
> Our model has been successfully validated and profiled on the **[Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com)**:
> * **Target Device**: `Samsung Galaxy S25 (Family)`
> * **Runtime**: `onnx`

---

## 📂 Git Guidelines

When pushing source code to Git, ensure the following configuration files are committed:
* `requirements.txt`
* `pyproject.toml`
* `uv.lock`
* `.python-version`

> [!IMPORTANT]
> **Never** commit the `.venv/` directory. It is already configured to be ignored in the project's `.gitignore` file.
