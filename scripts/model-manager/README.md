# Model Manager Scripts

This directory contains scripts used to manage models and checkpoints on Hugging Face Hub or locally. The functions are clearly separated between **Standard Models** (which only include weights, configs, and tokenizers for inference/deployment) and **Checkpoints** (which include the full training state, optimizer, scheduler, and rng_state for resuming training).

---

## List of Scripts

### 1. `push_model.py`
Pushes a standard model to Hugging Face Hub (includes only weights `pytorch_model.bin`, config files, tokenizer, custom code, and README.md). It does not include large optimizer or training state files.

**Usage:**
```bash
uv run python scripts/model-manager/push_model.py <repo_id> [experiment_dir] [checkpoint_dir]
```
* `<repo_id>`: Target repository name on Hugging Face (e.g., `your-username/your-model-name`).
* `[experiment_dir]` (Default: `outputs/vi_multitask`): Directory containing the configuration file `args.json` and tokenizer files.
* `[checkpoint_dir]` (Default: the highest checkpoint in `experiment_dir`): Specific directory containing the checkpoint weights.

**Example:**
```bash
uv run python scripts/model-manager/push_model.py aiai-laboratory/discrete-diffusion-vi-multitask
```

---

### 2. `push_checkpoint.py`
Pushes the entire training state (a checkpoint folder containing `pytorch_model.bin`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, and `training_args.bin`, along with the config, tokenizer, and custom code) to Hugging Face Hub for storage or for resuming training on another machine.

**Usage:**
```bash
uv run python scripts/model-manager/push_checkpoint.py <repo_id> [checkpoint_dir] [repo_type]
```
* `<repo_id>`: Target repository name on Hugging Face.
* `[checkpoint_dir]` (Default: the highest checkpoint in `outputs/vi_multitask`): The checkpoint directory to upload.
* `[repo_type]` (Default: `model`): Hugging Face repository type (`model` or `dataset`).

**Example:**
```bash
uv run python scripts/model-manager/push_checkpoint.py aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint outputs/vi_multitask/checkpoint-60000
```

---

### 3. `load_model.py`
Loads a model from a local checkpoint or from the Hugging Face Hub (loads weights and tokenizer only) to run inference on an input audio file.

**Usage:**
```bash
uv run python scripts/model-manager/load_model.py <model_path_or_repo_id> [audio_path] [json_path]
```
* `<model_path_or_repo_id>`: Path to the local checkpoint (e.g., `outputs/vi_multitask/checkpoint-60000`) or Hugging Face repository ID (e.g., `aiai-laboratory/discrete-diffusion-vi-multitask`).
* `[audio_path]` (Default: `test/test_data/test_sample.mp3`).
* `[json_path]` (Default: `test/test_data/test_sample.json`).

**Example:**
```bash
# Load local model
uv run python scripts/model-manager/load_model.py outputs/vi_multitask/checkpoint-60000

# Load model from Hugging Face
uv run python scripts/model-manager/load_model.py aiai-laboratory/discrete-diffusion-vi-multitask
```

---

### 4. `load_checkpoint.py`
Downloads the entire checkpoint directory (including optimizer, scheduler, rng_state, and training_args) from the Hugging Face Hub to your local machine so you can resume training.

**Usage:**
```bash
uv run python scripts/model-manager/load_checkpoint.py --repo_id <repo_id> [--target_dir outputs/vi_multitask_resumed]
```
* `--repo_id`: The Hugging Face repository ID containing the checkpoint.
* `--target_dir` (Default: `outputs/vi_multitask_resumed`): Directory where the downloaded checkpoint will be saved.

**Example:**
```bash
uv run python scripts/model-manager/load_checkpoint.py --repo_id aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint
```
*After a successful download, you can resume training by passing the `--resume_from_checkpoint` parameter pointing to the downloaded checkpoint directory.*

---

### 5. `eval_val.py`
Runs validation dataset evaluation to measure the performance and accuracy of the Discrete Diffusion Speech Translation model. The script evaluates speech translation from Vietnamese audio to three target languages (English, Chinese, Korean) using BLEU-n gram metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4) and measures inference performance (inference time per sample, Real-Time Factor).

To speed up execution on large datasets, the script supports **Batching**, **Task Selection**, and GPU compilation optimization via `torch.compile`.

**Usage:**
```bash
uv run python scripts/model-manager/eval_val.py [options]
```

**Key Parameters:**
* `--limit <int>` (Default: `100`): The number of samples to evaluate (`-1` to run on the entire validation set).
* `--batch-size <int>` (Default: `16`): The batch size to run in parallel on the GPU.
* `--tasks <str>` (Default: `english,chinese,korean`): Comma-separated list of tasks to evaluate. For example, `--tasks english` evaluates only English speech translation.
* `--compile`: Flag to activate graph compilation using `torch.compile` to optimize GPU computation speed.
* `--iterations <int>` (Default: `10`): Number of denoising steps for the Diffusion model.
* `--output-json <path>` (Default: `evaluation_results.json`): Path to save the results file.

**Example:**
```bash
# Run a quick evaluation on 100 samples with Batch Size = 16
uv run python scripts/model-manager/eval_val.py --limit 100 --batch-size 16

# Run evaluation on the entire validation set for English translation only (in tmux/background)
nohup uv run python scripts/model-manager/eval_val.py --limit -1 --batch-size 16 --tasks english > eval_en.log 2>&1 &
```

---

> [!IMPORTANT]
> Make sure you have set up the `HF_TOKEN` environment variable in your `.env` file or system environment before running scripts that involve downloading or pushing private Hugging Face data.
