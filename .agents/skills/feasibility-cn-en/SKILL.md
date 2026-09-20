---
name: feasibility-cn-en
description: Describe and use aiai-laboratory/feasibility-cn-en to evaluate speech translation architecture feasibility in the diffusion-speech-recognition project. Use for questions about its schema, provenance, loading, or designing and running Chinese audio → English experiments on this dataset.
---

# Feasibility Chinese → English

This dataset supports feasibility experiments for architectures that take Chinese audio and generate English text in the `diffusion-speech-recognition` project. Successful loading and decoding alone do not demonstrate that a model can learn or that an architecture is feasible.

All local paths below are relative to the **project root**, not the skill directory. Locate the checkout containing `src/model/dd_model.py` and `scripts/data-preprocess/prepare_feasibility_cn_en.py` before using local paths. For requests limited to dataset information or schema, answer from this skill and the dataset report without starting training.

## Dataset information

| Property | Value |
| --- | --- |
| Hugging Face | [aiai-laboratory/feasibility-cn-en](https://huggingface.co/datasets/aiai-laboratory/feasibility-cn-en) |
| Uploaded and verified revision | `e271559d073562656b5061fffd1acc2db4f1ba3c` |
| Config | `default` |
| Primary task | Speech translation: Chinese audio → English text |
| Source | CoVoST 2 `zh-CN → en`, based on Common Voice **v4** |
| Total examples | **16,826**, approximately **26.59 hours** of audio |
| Distribution format | 32 Parquet shards with embedded audio bytes; approximately 2.45 GB |
| Standardized audio | WAV signed 16-bit PCM, mono, 16,000 Hz |
| Data license | CC0-1.0, according to [CoVoST upstream](https://github.com/facebookresearch/covost#license) |
| Preparation status | Raw audio + text; **no precomputed** encoder embeddings or token IDs |

| Split | Examples | Audio hours | Original CoVoST2 labels |
| --- | ---: | ---: | --- |
| `train` | 7,085 | 10.4438 | `train` (7,079) + `train_covost` (6) |
| `validation` | 4,843 | 7.9019 | `dev` |
| `test` | 4,898 | 8.2442 | `test` |

Preserve the official splits instead of merging and randomly splitting them again. Auxiliary labels `dev_covost`, `test_covost`, `dev_dup`, and `test_dup` are excluded following the official CoVoST2 split procedure. Audio IDs are disjoint across splits; speaker and sentence disjointness has not been established.

These statistics describe the verified revision, not a guarantee that `main` remains unchanged. Pin this revision for reproducibility. If intentionally using another revision, inspect its corresponding dataset card and report.

## Data schema

Each row represents one utterance and contains exactly five columns:

| Column | Type | Meaning and constraints |
| --- | --- | --- |
| `id` | string | Unique WAV filename, such as `common_voice_zh-CN_18536372.wav`; equals `audio.path`. This is not a numeric index. |
| `chinese` | string | Original Chinese transcript with text and punctuation preserved. Use for inspection or as an auxiliary ASR label when required by the experiment. |
| `english` | string | Official English translation; the target for speech translation. |
| `audio` | Hugging Face `Audio(sampling_rate=16000)` | Stored in Parquet as `{bytes: binary, path: string}`. The bytes contain a complete WAV file, not a float array or headerless PCM. |
| `client_id` | string | Original Common Voice speaker ID; useful for checking speaker distribution and overlap. |

`audio.path` is a relative filename used as an identifier, not an absolute path that can be opened immediately after loading from the Hub. Read `audio.bytes`, or resolve the filename under the local `clips/` directory. Loading with `Audio(decode=False)` returns a `bytes/path` dictionary; do not assume it contains `array` or `sampling_rate` fields.

The raw dataset has no `source`, `target`, `transcription`, `translation`, `vietnamese`, `korean`, `audio_embeds`, `embedding_bytes`, or token ID columns. Training tensors such as `source`, `target`, `src_length`, and `audio_values` are created by the dataset adapter after decoding and tokenization; distinguish them from stored columns.

## Provenance, local files, and completed validation

MP3 audio and transcripts came from the [fixie-ai/covost2 mirror](https://huggingface.co/datasets/fixie-ai/covost2/tree/17c8c81e331e7a6929118121771a58c7ef7331d8/zh-CN_en), revision `17c8c81e331e7a6929118121771a58c7ef7331d8`. The original Common Voice v4 S3 URL returned HTTP 403 during preparation, so this mirror was used rather than substituting a newer Common Voice release.

Every filename, English translation, and split was checked against the [official CoVoST2 TSV](https://dl.fbaipublicfiles.com/covost/covost_v2.zh-CN_en.tsv.tar.gz). Audio was decoded from MP3, converted to mono where necessary, resampled with a polyphase filter, and written as WAV PCM16. Transcripts and translations were stored without text normalization.

| Project path | Contents |
| --- | --- |
| `data/feasibility-cn-en/source/` | Official TSV and 32 source Parquet shards under `mirror/zh-CN_en/` |
| `data/feasibility-cn-en/clips/` | 16,826 individual WAV files |
| `data/feasibility-cn-en/hub/data/` | Standardized Parquet shards ready to load |
| `data/feasibility-cn-en/hub/README.md` | Dataset card |
| `data/feasibility-cn-en/hub/validation_report.json` | Example counts, audio hours, source/output SHA-256 hashes, and validation results |
| `scripts/data-preprocess/prepare_feasibility_cn_en.py` | Reproduction script with optional upload |

Validation covered decoding every audio sample with `src/data/utils.py::_decode_wav_bytes`, split completeness, nonempty text/audio, and unique IDs. After upload, SHA-256 hashes for all 32 shards matched the local files, and one example per split was read directly from HF. These checks establish integrity and format compatibility; they do not constitute manual verification of every transcript or evaluation of model quality.

Prefer reusing local data. Rebuild only when needed, using the project's uv environment:

```bash
uv run python scripts/data-preprocess/prepare_feasibility_cn_en.py
```

`--skip-download` reuses downloaded source files. `--upload` is a separate publishing operation and is unnecessary for schema inspection or feasibility evaluation.

## Loading the dataset

From the project root, load local files without downloading them again:

```python
from pathlib import Path
from datasets import Audio, load_dataset

root = Path("data/feasibility-cn-en/hub/data")
dataset = load_dataset(
    "parquet",
    data_files={
        split: sorted(str(p) for p in root.glob(f"{split}-*.parquet"))
        for split in ("train", "validation", "test")
    },
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
row = dataset["train"][0]
assert row["id"] == row["audio"]["path"]
wav_bytes = row["audio"]["bytes"]
target_text = row["english"]
```

If local files are unavailable, replace the `load_dataset` call with:

```python
dataset = load_dataset(
    "aiai-laboratory/feasibility-cn-en",
    revision="e271559d073562656b5061fffd1acc2db4f1ba3c",
)
```

Keep the `Audio(decode=False)` cast when using the project's WAV decoder. For quick inspection of a few HF examples, `streaming=True` is an option; this streams **raw audio** and uses a different schema from the project's precomputed streaming datasets.

## Evaluating architecture feasibility

Inspect the current code and configuration before editing or running experiments. The following points describe the project at the time this skill was created:

- `src/data/translated_speech.py` and `src/data/multitask.py` hardcode VietSpeech loading and join two sources by ID. This dataset already contains audio and text in each row, so the adapter should read them directly and preserve the official splits. Do not join against `NhutP/VietSpeech` or create another 1% validation split.
- `TASK_TO_FIELD` currently contains only `<vi_en>`, `<vi_zh>`, and `<vi_ko>`. For a Chinese → English task token, define an explicit mapping, such as `<zh_en>` → `english`, register the token, and resize embeddings through the model's existing mechanism. Changing the token string alone does not establish the correct mapping.
- `src/data/precomputed_multitask.py` and `src/data/streaming_precomputed_multitask.py` expect embeddings and token IDs. Pointing their `streaming_repo_id` at this raw dataset does not make the schemas compatible.
- Inspect `src/model/dd_model.py`, `src/model/cross_attn_roberta.py`, and the experiment configuration to determine the actual encoder, tokenizer, fusion (`prefix`/`deep_cross_attn`), masking, and decoding setup. Schema compatibility does not establish that an encoder is suitable for Chinese speech.
- `configs/vi_en_deep_fusion.json` is a Vietnamese configuration with `oracle_length=true`, a long training schedule, and `push_to_hub=true`. Create a feasibility configuration with its own budget, output, and task settings based on the request instead of running that configuration unchanged. Results using ground-truth target lengths must be labeled as oracle results, not inference without reference labels.

When asked to run experiments, choose a scope appropriate to the user's time and resource constraints. A useful progression is to check batches and forward/backward passes, attempt to overfit a small subset of `train`, and then evaluate on `validation`. Reserve `test` for final reporting after configuration selection. Record the seed and subset IDs for reproducibility.

Evaluate direct speech translation with **audio as input and `english` as the label**. Do not feed `chinese` or English ground truth into inference conditioning unless the user requests a separate baseline, cascade, or oracle experiment and the distinction is reported clearly. To investigate whether the model uses audio, consider comparing correctly paired and shuffled audio on the same validation examples with identical decoding settings.

A feasibility report should distinguish correct pipeline execution, the ability to learn or overfit, and quality on held-out data. For translation evaluation, report BLEU with its tokenizer/signature and normalization rules; use WER as a supplementary metric. Record the dataset revision, subset, training steps, configuration, oracle/non-oracle settings, losses/metrics, and measured resource usage. Do not infer feasibility solely from decreasing loss or invent unspecified pass/fail thresholds.
