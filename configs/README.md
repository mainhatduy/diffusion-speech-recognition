# Diffusion Speech Recognition - Complete Configuration Reference

This document provides a comprehensive reference of all configurable parameters, environment variables, command-line arguments, and JSON configuration options available in the **Diffusion Speech Recognition** project.

---

## Table of Contents

1. [Configuration Overview](#1-configuration-overview)
2. [Environment Variables (`.env`)](#2-environment-variables-env)
3. [Data Arguments (`DiscreteDiffusionDataArguments`)](#3-data-arguments-discretediffusiondataarguments)
4. [Model Architecture Arguments (`DiscreteDiffusionModelArguments`)](#4-model-architecture-arguments-discretediffusionmodelarguments)
5. [Training & Optimizer Arguments (`DiscreteDiffusionTrainingArguments`)](#5-training--optimizer-arguments-discretediffusiontrainingarguments)
6. [Inference & Generator Arguments (`DiscreteDiffusionGeneratorArguments`)](#6-inference--generator-arguments-discretediffusiongeneratorarguments)
7. [Script CLI Arguments](#7-script-cli-arguments)
   - [Data Preprocessing Scripts](#data-preprocessing-scripts)
   - [Qualcomm Hardware Job Submission](#qualcomm-hardware-job-submission)
   - [Pipeline Shell Orchestration](#pipeline-shell-orchestration)
8. [JSON Config Example](#8-json-config-example)

---

## 1. Configuration Overview

The project uses a unified configuration system built on Hugging Face's [`HfArgumentParser`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.HfArgumentParser). Parameters can be configured in three ways:

1. **JSON Configuration Files**: Passed directly as the first argument to `src/train.py` (e.g., `uv run python src/train.py configs/streaming_vi_multitask.json`).
2. **Command Line Flags**: Passed directly to scripts (e.g., `--per_device_train_batch_size 32`).
3. **Environment Variables**: Managed via `.env` files and read using `python-dotenv`.

---

## 2. Environment Variables (`.env`)

Environment variables configure external API credentials, runtime hardware selection, and tracking tools.

| Variable Name | Type | Description | Default / Example |
| :--- | :--- | :--- | :--- |
| `HF_TOKEN` | String | Hugging Face Hub API token used for downloading private datasets/models and pushing streaming shards or checkpoints. | `hf_xxx` |
| `WANDB_TOKEN` | String | Weights & Biases API key for experiment tracking. Automatically assigned to `WANDB_API_KEY`. | `wandb_xxx` |
| `WANDB_PROJECT` | String | Overrides the default Weights & Biases project name. | `diffusion-speech-recognition` |
| `WANDB_RUN_NAME` | String | Overrides the Weights & Biases run name for the active training session. | `streaming_vi_multitask_v1` |
| `QUALCOMM_TOKEN` | String | Qualcomm AI Hub API key required for hardware profiling, ONNX compilation, and job submission. | `qai_hub_xxx` |
| `CUDA_VISIBLE_DEVICES` | String | Specifies which GPU indices to expose to PyTorch. | `0` or `0,1` |

---

## 3. Data Arguments (`DiscreteDiffusionDataArguments`)

Data arguments control dataset selection, pre-processing, streaming settings, RAM caching, and multi-task token definitions.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dataset_type` | `str` | `"bilingual"` | Dataset type dispatch mode. Supported options: `"speech_translation_multitask"`, `"speech_recognition"`, `"speech_translation"`, `"bilingual"`, `"pair"`. |
| `audio_encoder_name` | `str` | `"UsefulSensors/moonshine-streaming-medium"` | Pretrained audio encoder model name/path from Hugging Face Hub (e.g., `"UsefulSensors/moonshine-streaming-medium"`, `"facebook/mms-300m"`). |
| `data_path` | `str` | `""` | Dataset repository name on Hugging Face Hub (e.g., `"NhutP/VietSpeech"`) or local dataset path. |
| `src_lang` | `str` | `""` | Source language code (e.g., `"vi"`). |
| `tgt_lang` | `str` | `""` | Target language code (e.g., `"en"`). |
| `src_column` | `str` | `"en"` | Column name for source text in raw datasets. |
| `tgt_column` | `str` | `"vi"` | Column name for target text in raw datasets. |
| `max_length` | `int` | `2048` | Maximum sequence length for tokenization padding/truncation (set to `64` for text targets in streaming setups). |
| `packing` | `bool` | `false` | Whether to pack multiple short sequences into a single max-length input batch item. |
| `task_tokens` | `list[str]` | `["<vi_en>", "<vi_zh>", "<vi_ko>"]` | Special task control tokens prepended to text prompts for multi-task speech translation. |
| `precomputed_data_dir` | `str` | `""` | Path to local directory containing pre-computed audio embeddings and token IDs. Activates `PrecomputedMultiTaskDataset` when present. |
| `streaming` | `bool` | `false` | Enable streaming dataset mode directly from Hugging Face Hub without local dataset downloads. |
| `streaming_repo_id` | `str` | `"aiai-laboratory/vietspeech-train-streaming"` | Hugging Face Hub repository ID containing the streaming Parquet dataset shards. |
| `streaming_buffer_size` | `int` | `1000` | Shuffling buffer size for HF streaming dataset. |
| `val_streaming_size` | `int` | `500` | Fixed number of validation samples drawn when evaluating in streaming mode. |
| `use_ram_cache` | `bool` | `false` | Enable caching raw audio bytes in system RAM during local dataset loading. |
| `ram_free_threshold_ratio` | `float` | `0.30` | Minimum free RAM ratio required before allowing audio preloading into RAM. |
| `hf_token` | `str` | `None` | Hugging Face token passed explicitly to dataset loading calls. |
| `dedupe` | `bool` | `false` | Deduplicate identical source/target text pairs during data loading. |
| `fix_ftfy` | `bool` | `false` | Apply `ftfy` text fixing to resolve Mojibake and formatting bugs in raw text. |
| `normalize_punct` | `bool` | `false` | Normalize unicode punctuation symbols. |
| `detokenize` | `bool` | `false` | Apply detokenization pre-processing. |
| `remove_wiki` | `bool` | `false` | Remove Wikipedia metadata comments from raw inputs. |
| `remove_bracketed` | `bool` | `false` | Remove target sentences that start and end with bracketed punctuation. |
| `dereify` | `bool` | `false` | Dereify AMR graphs (legacy). |

---

## 4. Model Architecture Arguments (`DiscreteDiffusionModelArguments`)

Configures discrete diffusion dynamics, backbone language model selection, audio fusion strategies, and LoRA adapters.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pretrained` | `str` | `None` | Pretrained language model backbone identifier (e.g., `"FacebookAI/xlm-roberta-base"`). |
| `num_diffusion_timesteps` | `int` | `50` | Total number of discrete diffusion timesteps $T$ for forward corruption and reverse denoising. |
| `diffusion_type` | `str` | `"absorbing"` | Type of discrete diffusion process. Currently supports `"absorbing"` (masking state). |
| `attention_strategy` | `str` | `"full"` | Attention masking pattern (`"full"` or `"prefix_lm"`). |
| `audio_fusion_strategy` | `str` | `"prefix"` | Mechanism to combine audio representations with text: `"deep_cross_attn"` (cross-attention across layers) or `"prefix"` (prepending audio embeddings). |
| `pretrained_audio_encoder` | `bool` | `true` | Load pretrained weights for the audio encoder backbone. |
| `cache_dir` | `str` | `"/mnt/bn/research/cache"` | Local directory used to store downloaded model weights and tokenizers. |
| `vocab_pad_to_multiple` | `int` | `1` | Pad vocabulary size to a multiple of this value for GPU hardware alignment. |
| `prefix_lm` | `bool` | `false` | *(Deprecated)* Pre-LM attention flag. Use `attention_strategy="prefix_lm"` instead. |

### LoRA (Low-Rank Adaptation) Arguments

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lora` | `bool` | `false` | Enable Low-Rank Adaptation (LoRA) parameter-efficient fine-tuning. |
| `lora_target_modules` | `list[str]` | `["query", "value"]` | Submodules to inject LoRA adaptors into. |
| `lora_rank` | `int` | `16` | LoRA intrinsic rank dimension ($r$). |
| `lora_alpha` | `float` | `16.0` | LoRA alpha scaling factor ($\alpha$). |
| `lora_dropout` | `float` | `0.0` | Dropout probability for LoRA layers. |
| `lora_bias` | `str` | `"none"` | LoRA bias parameter training configuration (`"none"`, `"all"`, or `"lora_only"`). |

### Advanced & Streaming Architecture Extensions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_ergodic_layers` | `int` | `6` | Number of ergodic attention layers in streaming architectures. |
| `ergodic_window_left` | `int` | `64` | Left window boundary (tokens) for ergodic attention. |
| `ergodic_window_right` | `int` | `16` | Right window boundary (tokens) for ergodic attention. |
| `num_position_layers` | `int` | `6` | Number of position-aware attention layers. |
| `position_window_left` | `int` | `128` | Left window size for position-aware attention. |
| `position_window_right` | `int` | `32` | Right window size for position-aware attention. |
| `rope_theta` | `float` | `10000.0` | Base frequency ($\theta$) for Rotary Position Embeddings (RoPE). |
| `active_window_size` | `int` | `64` | Active token context window size during streaming inference. |
| `frozen_cache_size` | `int` | `128` | Size of frozen past-context KV cache in streaming mode. |
| `streaming_denoise_steps` | `int` | `3` | Number of in-flight denoising passes per streaming chunk. |
| `freeze_confidence_threshold` | `float` | `0.92` | Logit probability threshold required to freeze predicted tokens during streaming. |
| `remask_confidence_threshold` | `float` | `0.30` | Threshold below which tokens are re-masked for refinement. |
| `loss_weight_supported` | `float` | `2.0` | Loss multiplier for supported token positions. |
| `loss_weight_unsupported` | `float` | `0.3` | Loss multiplier for unsupported token positions. |
| `use_confidence_calibration` | `bool` | `true` | Enable probability calibration for confidence scoring. |
| `audio_chunk_duration` | `float` | `2.0` | Audio stream chunk duration in seconds. |
| `audio_overlap_duration` | `float` | `0.5` | Overlap duration between consecutive audio chunks in seconds. |

---

## 5. Training & Optimizer Arguments (`DiscreteDiffusionTrainingArguments`)

Inherits from Hugging Face's [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) and adds custom discrete diffusion training controls.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_dir` | `str` | *(Required)* | Output directory for saving model checkpoints, training logs, and configuration dumps. |
| `finetune_from_model` | `str` | `None` | Path or model ID to initialize model weights from for multi-stage fine-tuning. |
| `resume_from_checkpoint` | `str` / `bool` | `None` | Path to a specific checkpoint folder to resume training from, or `true` to auto-resume from the last checkpoint in `output_dir`. |
| `mask_ratio_sampler` | `str` | `"diffusion"` | Mask ratio sampling mode during training: `"diffusion"` (random timestep $t \in [1, T]$) or `"fixed[ratio]"` (e.g. `"fixed0.15"` for standard MLM at 15%). |
| `weighting` | `str` | `"linear"` | Loss weighting strategy across diffusion timesteps (`"linear"` or `"constant"`). |
| `mask_on_source` | `bool` | `false` | Whether to apply masking to source prompt tokens in addition to target tokens. |
| `mask_on_paddings` | `bool` | `false` | Whether to compute loss and apply masking on padding tokens. |
| `train_length` | `bool` | `false` | Train an explicit target sequence length predictor module alongside the main diffusion model. |
| `batch_by_tokens` | `bool` | `false` | Dynamic token-budget batching using `TokenSizeDistributedLengthGroupSampler`. |
| `eval_metric` | `str` | `"none"` | Single evaluation metric selection (`"bleu"`, `"rouge"`, `"wer"`, or `"none"`). |
| `eval_metrics` | `list[str]` | `[]` | List of multiple evaluation metrics computed simultaneously (e.g. `["bleu", "wer"]`). Overrides `eval_metric`. |
| `wandb_project` | `str` | `"mlm-to-dlm"` | Weights & Biases project name. |
| `push_to_hub` | `bool` | `false` | Asynchronously push evaluated checkpoints to Hugging Face Hub using `HuggingFacePushCallback`. |
| `hub_model_id` | `str` | `None` | Target repository ID on Hugging Face Hub when `push_to_hub=true`. |
| `hub_model_repo_type` | `str` | `"model"` | Repository type on Hugging Face Hub (`"model"` or `"dataset"`). |

### Core Hugging Face Training Parameters

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `per_device_train_batch_size` | `int` | `8` | Batch size per GPU/CPU device for training. |
| `per_device_eval_batch_size` | `int` | `8` | Batch size per GPU/CPU device for evaluation. |
| `gradient_accumulation_steps` | `int` | `1` | Number of update steps to accumulate gradients before performing a backward pass. |
| `learning_rate` | `float` | `5e-5` | Initial learning rate for the optimizer. |
| `weight_decay` | `float` | `0.0` | Weight decay coefficient (L2 penalty). |
| `adam_beta1` | `float` | `0.9` | Beta1 parameter for AdamW optimizer. |
| `adam_beta2` | `float` | `0.999` | Beta2 parameter for AdamW optimizer. |
| `lr_scheduler_type` | `str` | `"linear"` | Learning rate decay schedule (`"linear"`, `"cosine"`, `"polynomial"`, `"constant"`). |
| `warmup_ratio` | `float` | `0.0` | Ratio of total training steps reserved for linear warmup. |
| `warmup_steps` | `int` | `0` | Absolute number of warmup steps (overrides `warmup_ratio` if $> 0$). |
| `max_grad_norm` | `float` | `1.0` | Maximum gradient norm for gradient clipping. |
| `max_steps` | `int` | `-1` | Total number of training steps. Overrides `num_train_epochs` if $> 0$. |
| `num_train_epochs` | `float` | `3.0` | Total number of training epochs to execute. |
| `bf16` | `bool` | `false` | Use bfloat16 mixed precision training. |
| `fp16` | `bool` | `false` | Use float16 mixed precision training. |
| `bf16_full_eval` | `bool` | `false` | Use bfloat16 precision during evaluation loops. |
| `label_smoothing_factor` | `float` | `0.0` | Label smoothing factor $\epsilon \in [0, 1]$. |
| `do_eval` | `bool` | `false` | Enable running evaluation on the validation set. |
| `eval_strategy` | `str` | `"no"` | Evaluation strategy (`"no"`, `"steps"`, or `"epoch"`). |
| `eval_steps` | `int` | `500` | Number of update steps between evaluations when `eval_strategy="steps"`. |
| `save_steps` | `int` | `500` | Number of update steps between saving checkpoints. |
| `save_total_limit` | `int` | `None` | Maximum number of checkponts to keep in `output_dir`. |
| `load_best_model_at_end` | `bool` | `false` | Load the best checkpoint found during training at the end of execution. |
| `metric_for_best_model` | `str` | `None` | Metric name used to compare models (e.g. `"bleu"`, `"wer"`). |
| `greater_is_better` | `bool` | `None` | Set `true` for BLEU/ROUGE (higher is better) or `false` for WER/Loss (lower is better). |
| `logging_steps` | `int` | `500` | Number of update steps between logging metrics. |
| `report_to` | `str` / `list` | `"all"` | Frameworks to report metrics to (`"wandb"`, `"tensorboard"`, `"none"`). |
| `dataloader_num_workers` | `int` | `0` | Number of subprocesses for PyTorch data loading. |
| `dataloader_pin_memory` | `bool` | `true` | Pin memory in PyTorch dataloaders for faster GPU transfer. |

---

## 6. Inference & Generator Arguments (`DiscreteDiffusionGeneratorArguments`)

Controls the iterative reverse diffusion generation process during inference and evaluation.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_iterations` | `int` | `10` | Total number of reverse diffusion denoising iterations $N_{steps}$ during generation. |
| `strategy` | `str` | `"reparam-uncond-deterministic-cosine"` | Denoising update rule strategy. Formatted as `reparam-<condition>-<topk_mode>-<schedule>`.<br>- **Condition**: `"cond"` or `"uncond"`.<br>- **TopK Mode**: `"deterministic"` or `"stochastic[temp]"` (e.g., `"stochastic0.1"`).<br>- **Schedule**: `"cosine"` or `"linear"`.<br>- **Alternative Modes**: `"cmlm"`, `"ar"`. |
| `argmax_decoding` | `bool` | `true` | If `true`, selects tokens with highest probability (`argmax`). If `false`, samples from categorical logit distributions. |
| `temperature` | `float` | `0.8` | Softmax temperature scaling applied when `argmax_decoding=false`. |
| `length_beam` | `int` | `1` | Beam size for target length candidate predictions. |
| `mbr` | `int` | `1` | Minimum Bayes Risk (MBR) candidate sampling count per sequence. |
| `oracle_length` | `bool` | `false` | If `true`, initializes generation target length using ground-truth sequence length (oracle mode). |
| `bpe` | `str` | `"sentencepiece"` | BPE tokenizer type used during text decoding (`"sentencepiece"` or `"fastbpe"`). |
| `bleu_tokenize` | `str` | `"13a"` | SacreBLEU tokenization standard for evaluation (`"13a"`, `"zh"`, `"intl"`). |
| `return_history` | `bool` | `false` | If `true`, returns intermediate token sequences generated at every diffusion step. |

---

## 7. Script CLI Arguments

### Data Preprocessing Scripts

#### 1. `scripts/data-preprocess/merge_to_streaming.py`
Merges pre-computed audio embeddings and token IDs into a streaming-ready Parquet dataset and pushes shards to Hugging Face Hub.

```bash
uv run python scripts/data-preprocess/merge_to_streaming.py [FLAGS]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source_repo` | `str` | `"aiai-laboratory/vietspeech-train-precompute"` | Source HF dataset repo ID containing unmerged shards. |
| `--target_repo` | `str` | `"aiai-laboratory/vietspeech-train-streaming"` | Target HF dataset repo ID to store streaming Parquet dataset. |
| `--shards_per_commit` | `int` | `5` | Number of shards bundled per commit to respect HF commit rate limits. |
| `--test` | Flag | `false` | Test mode: processes only the first shard. |
| `--dry_run` | Flag | `false` | Dry-run mode: generates Parquet files locally without pushing to HF Hub. |
| `--private` | Flag | `false` | Creates target HF repository as private if it does not already exist. |

#### 2. `scripts/data-preprocess/precompute_embeddings.py`
Extracts audio embeddings from raw audio files using Moonshine/Wav2Vec2 backbones and tokenizes text target sequences into disk cache.

```bash
uv run python scripts/data-preprocess/precompute_embeddings.py [FLAGS]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--output_dir` | `str` | *(Required)* | Output directory path to save precomputed dataset shards. |
| `--audio_encoder_name` | `str` | `"UsefulSensors/moonshine-streaming-medium"` | Pretrained audio encoder model name/path. |
| `--pretrained` | `str` | `"FacebookAI/xlm-roberta-base"` | Pretrained tokenizer and text backbone name. |
| `--cache_dir` | `str` | `"cache"` | Local directory for cached model downloads. |
| `--hf_token` | `str` | `None` | Hugging Face authentication token. |
| `--batch_size` | `int` | `32` | Batch size for audio embedding extraction. |
| `--max_length` | `int` | `128` | Maximum target text sequence length. |
| `--num_workers` | `int` | `4` | Number of dataloader worker processes. |
| `--device` | `str` | `"cuda:0"` | Computing device (`"cuda:0"`, `"cpu"`). |
| `--dtype` | `str` | `"float16"` | Storage data type for embeddings (`"float16"` or `"float32"`). |
| `--resume` | Flag | `false` | Resume precomputation from the last written shard index. |
| `--task_tokens` | `list` | `["<vi_en>", "<vi_zh>", "<vi_ko>"]` | Task token list for multi-task target generation. |

---

### Qualcomm Hardware Job Submission

#### `scripts/qualcomm-job/submission/submit_qualcomm_job.py`
Submits compiled ONNX models to Qualcomm AI Hub for profiling and execution on Snapdragon NPU devices.

```bash
uv run python scripts/qualcomm-job/submission/submit_qualcomm_job.py [FLAGS]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--device` | `str` | `"Samsung Galaxy S25 (Family)"` | Qualcomm target hardware device or device family string. |
| `--runtime` | `str` | `"qnn"` | Target NPU execution runtime (`"qnn"` or `"onnx"`). |
| `--skip-repackage` | Flag | `false` | Skip ONNX external weight repackaging step before job submission. |

---

### Pipeline Shell Orchestration

#### `scripts/training/run_pipeline_end2end.sh`
Main orchestration bash script for launching end-to-end training pipelines.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/training/run_pipeline_end2end.sh [FLAGS]
```

| Flag | Mode Triggered | Config Selected | Description |
| :--- | :--- | :--- | :--- |
| `--streaming` | Streaming Mode | `configs/vi_multitask_streaming_config.json` | Runs full training directly streaming shards from Hugging Face Hub. |
| `--streaming --test` | Streaming Test | `configs/test_vi_multitask_streaming_config.json` | Runs a quick 10-step streaming validation test. |
| *(None)* | Local Precomputed | `configs/vi_multitask_precomputed_config.json` | Downloads precomputed dataset if missing and runs local training. |
| `--test` | Local Test Mode | `configs/test_vi_multitask_precomputed_config.json` | Downloads lightweight 1-shard subset and runs 10-step validation. |

---

## 8. JSON Config Example

Below is a complete, production-ready configuration file (`configs/streaming_vi_multitask.json`) demonstrating how arguments across data, model, trainer, and generator classes are combined:

```json
{
    "dataset_type": "speech_translation_multitask",
    "data_path": "NhutP/VietSpeech",
    "audio_encoder_name": "UsefulSensors/moonshine-streaming-medium",
    "task_tokens": [
        "<vi_en>"
    ],
    "max_length": 64,
    "packing": false,
    "streaming": true,
    "streaming_repo_id": "aiai-laboratory/vietspeech-train-streaming",
    "streaming_buffer_size": 1000,
    "val_streaming_size": 500,

    "num_diffusion_timesteps": 50,
    "use_ram_cache": true,
    "ram_free_threshold_ratio": 0.30,
    "diffusion_type": "absorbing",
    "pretrained": "FacebookAI/xlm-roberta-base",
    "cache_dir": "cache",
    "attention_strategy": "full",
    "vocab_pad_to_multiple": 1,
    "lora": false,
    "audio_fusion_strategy": "deep_cross_attn",

    "output_dir": "outputs/streaming_vi_multitask",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 16,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": false,
    "gradient_accumulation_steps": 4,
    "max_steps": 100000,
    "do_eval": true,
    "eval_strategy": "steps",
    "eval_steps": 5000,
    "eval_metric": "wer",
    "eval_metrics": [
        "bleu",
        "wer"
    ],
    "load_best_model_at_end": true,
    "metric_for_best_model": "bleu",
    "greater_is_better": false,
    "logging_steps": 200,
    "save_steps": 10000,
    "save_total_limit": 3,
    "report_to": "wandb",
    "wandb_project": "diffusion-speech-recognition",
    "run_name": "streaming_vi_multitask_v1",

    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "bf16": true,
    "bf16_full_eval": true,
    "mask_ratio_sampler": "diffusion",
    "weighting": "linear",

    "max_iterations": 10,
    "mbr": 1,
    "length_beam": 1,
    "oracle_length": true,
    "strategy": "reparam-uncond-deterministic-cosine",
    "argmax_decoding": true,
    "bpe": "sentencepiece",
    "bleu_tokenize": "13a",
    "temperature": 1.0,

    "push_to_hub": false,
    "hub_model_id": "aiai-laboratory/streaming-diffusion-vi-multitask"
}
```
