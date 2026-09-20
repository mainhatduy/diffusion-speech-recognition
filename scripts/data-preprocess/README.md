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

### 6. `prepare_feasibility_cn_en.py`

Downloads the pinned CoVoST2 Chinese → English mirror, checks all filenames,
translations, and split assignments against the official CoVoST2 TSV, and converts
audio to mono 16 kHz PCM_16 WAV. The output columns are `id`, `chinese`, `english`,
`audio`, and `client_id`; `id` equals `audio.path`. Every audio sample is checked
with the project's WAV decoder. Official splits are preserved: 7,085 train,
4,843 validation, and 4,898 test examples.

```bash
uv run python scripts/data-preprocess/prepare_feasibility_cn_en.py --upload
```

The default destination is `aiai-laboratory/feasibility-cn-en`. The script reads
`HF_TOKEN` from `.env`. Omit `--upload` for local preparation only, or use
`--skip-download` to rebuild from already downloaded source files.

Local files are stored under `data/feasibility-cn-en/`: `source/` contains the
original mirror and official TSV, `clips/` contains WAV files, and `hub/` contains
self-contained Parquet shards, the dataset card, and a validation report with
SHA-256 hashes. Existing training loaders still hardcode VietSpeech repositories
and Vietnamese task tokens and need adaptation for Chinese → English training.
