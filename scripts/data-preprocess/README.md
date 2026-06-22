# Data Preprocess Scripts

This directory contains scripts for data preparation (downloading, extracting audio features, tokenizing text, converting formats, and uploading).

## List of Scripts

### 1. `download_precomputed_data.py`
Used to download precomputed datasets (precomputed audio embeddings & tokenized text) from Hugging Face Hub to your local machine for fast training.

**Usage:**
```bash
uv run python scripts/data-preprocess/download_precomputed_data.py \
    --target_dir precomputed_data \
    [--repo_id aiai-laboratory/vietspeech-train-precompute] \
    [--test] [--force]
```
* `--target_dir`: Directory to save the downloaded data (default: `precomputed_data`).
* `--repo_id`: Hugging Face dataset repository ID (default: `aiai-laboratory/vietspeech-train-precompute`).
* `--test`: Only download metadata and the first shard of data for testing.
* `--force`: Force redownloading and overwrite existing data.

---

### 2. `precompute_embeddings.py`
Used to manually extract audio embeddings and pre-process tokenized text from raw datasets.

**Usage:**
```bash
uv run python scripts/data-preprocess/precompute_embeddings.py \
    --output_dir precomputed_data \
    --audio_encoder_name UsefulSensors/moonshine-streaming-medium \
    --pretrained FacebookAI/xlm-roberta-base \
    --batch_size 32 \
    --max_length 128 \
    [--resume]
```
* `--output_dir`: Directory to save precomputed results.
* `--audio_encoder_name`: Audio encoder model (default: `UsefulSensors/moonshine-streaming-medium`).
* `--pretrained`: Backbone tokenizer/language model (default: `FacebookAI/xlm-roberta-base`).
* `--resume`: Resume processing from where it was interrupted.

---

### 3. `convert_npy_to_parquet.py`
Used to compress individual numpy `.npy` files into sharded `.parquet` files. This format facilitates faster data loading and saves memory during training via memory mapping.

**Usage:**
```bash
uv run python scripts/data-preprocess/convert_npy_to_parquet.py
```
*The script automatically looks for the `precomputed_data` directory at the project root, reads `index.json`, groups the `.npy` files, and converts them to Parquet format.*

---

### 5. `extract_validation.py`
Used to extract the validation split from the original `aiai-laboratory/vietspeech-train-translated` dataset using the same split configuration as in the training phase (shuffle seed=42, test_size=0.01). This validation set contains full labels for all 4 languages (Vietnamese, English, Chinese, Korean) with matching IDs. The result is saved as a Parquet file and can optionally be uploaded directly to Hugging Face.

**Usage:**
```bash
uv run python scripts/data-preprocess/extract_validation.py \
    --output_path outputs/validation.parquet \
    [--upload] \
    [--repo_id aiai-laboratory/vietspeech-validation-translated]
```
* `--output_path`: Path to save the validation parquet file.
* `--upload`: Enable this flag to upload the file to Hugging Face Hub after creation.
* `--repo_id`: Destination Hugging Face dataset repository (default: `aiai-laboratory/vietspeech-validation-translated`).

> [!IMPORTANT]
> You must configure `HF_TOKEN` in your `.env` file or environment variables to download the raw dataset and upload the resulting file to the Hugging Face Hub.
