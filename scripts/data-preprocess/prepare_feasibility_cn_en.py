"""Download, validate, and package CoVoST2 zh-CN -> en for this project."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import yaml
from datasets import Audio, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parents[2]
MIRROR = "fixie-ai/covost2"
REVISION = "17c8c81e331e7a6929118121771a58c7ef7331d8"
TRANSLATIONS = "https://dl.fbaipublicfiles.com/covost/covost_v2.zh-CN_en.tsv.tar.gz"
SPLITS = {"train": {"train", "train_covost"}, "validation": {"dev"}, "test": {"test"}}
FEATURES = Features(
    {
        "id": Value("string"),
        "chinese": Value("string"),
        "english": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "client_id": Value("string"),
    }
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "covost_v2.zh-CN_en.tsv.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(TRANSLATIONS, partial)
        partial.replace(archive)
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".tsv"))
        (root / "covost_v2.zh-CN_en.tsv").write_bytes(tar.extractfile(member).read())
    snapshot_download(
        MIRROR,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["zh-CN_en/*.parquet", "README.md"],
        local_dir=root / "mirror",
        max_workers=8,
    )


def prepare(root):
    source = root / "source"
    output = root / "hub"
    (output / "data").mkdir(parents=True, exist_ok=True)
    clips = root / "clips"
    clips.mkdir(exist_ok=True)
    with (source / "covost_v2.zh-CN_en.tsv").open() as stream:
        official = list(
            csv.DictReader(
                stream,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )
        )
    by_path = {row["path"]: row for row in official}
    assert len(by_path) == len(official), "Duplicate paths in official TSV"
    spec = importlib.util.spec_from_file_location(
        "project_audio_utils", PROJECT / "src/data/utils.py"
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    seen = set()
    stats = {}
    source_hashes = {}
    output_hashes = {}
    for split, labels in SPLITS.items():
        expected = {r["path"] for r in official if r["split"] in labels}
        found = set()
        total_frames = 0
        shards = sorted((source / "mirror/zh-CN_en").glob(f"{split}-*.parquet"))
        assert shards, f"Missing source shards for {split}"
        for shard in shards:
            source_hashes[shard.name] = sha256(shard)
            records = []
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    name = Path(row["file"]).name
                    assert name in expected, (split, name, "unexpected split/path")
                    assert name not in seen, (name, "duplicate or split leakage")
                    assert row["translation"] == by_path[name]["translation"], (
                        name,
                        "translation mismatch",
                    )
                    assert row["sentence"].strip() and row["translation"].strip()
                    assert (
                        row["audio"]["bytes"]
                        and Path(row["audio"]["path"]).name == name
                    )
                    samples, rate = sf.read(
                        io.BytesIO(row["audio"]["bytes"]),
                        dtype="float32",
                        always_2d=True,
                    )
                    samples = samples.mean(axis=1)
                    assert samples.size and np.isfinite(samples).all(), name
                    if rate != 16000:
                        divisor = math.gcd(rate, 16000)
                        samples = resample_poly(
                            samples, 16000 // divisor, rate // divisor
                        )
                    wav = io.BytesIO()
                    sf.write(wav, samples, 16000, format="WAV", subtype="PCM_16")
                    audio_bytes = wav.getvalue()
                    decoded, decoded_rate = utils._decode_wav_bytes(audio_bytes)
                    assert decoded_rate == 16000 and len(decoded) == len(samples)
                    wav_name = Path(name).with_suffix(".wav").name
                    (clips / wav_name).write_bytes(audio_bytes)
                    records.append(
                        {
                            "id": wav_name,
                            "chinese": row["sentence"],
                            "english": row["translation"],
                            "audio": {"bytes": audio_bytes, "path": wav_name},
                            "client_id": row["client_id"],
                        }
                    )
                    found.add(name)
                    seen.add(name)
                    total_frames += len(samples)
            destination = output / "data" / shard.name
            pq.write_table(
                pa.Table.from_pylist(records, schema=FEATURES.arrow_schema),
                destination,
                compression="zstd",
            )
            output_hashes[f"data/{shard.name}"] = sha256(destination)
            print(
                f"{split}: {len(found)}/{len(expected)} checked and written", flush=True
            )
        assert found == expected, (
            split,
            "missing official records",
            len(expected - found),
        )
        stats[split] = {
            "num_examples": len(found),
            "hours": total_frames / 16000 / 3600,
        }
    report = {
        "source_repo": MIRROR,
        "source_revision": REVISION,
        "translations_url": TRANSLATIONS,
        "translations_sha256": sha256(source / "covost_v2.zh-CN_en.tsv"),
        "source_sha256": source_hashes,
        "parquet_sha256": output_hashes,
        "official_split_counts": dict(Counter(r["split"] for r in official)),
        "splits": stats,
        "total_examples": len(seen),
        "validation": {
            "exact_official_paths_and_translations": True,
            "disjoint_audio_ids_across_splits": True,
            "all_audio_decoded_by_project_wav_decoder": True,
            "all_text_and_audio_nonempty": True,
        },
        "audio": {"format": "WAV PCM_16", "sample_rate": 16000, "channels": 1},
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = {
        "language": ["zh", "en"],
        "license": "cc0-1.0",
        "task_categories": ["automatic-speech-recognition", "translation"],
        "tags": ["covost2", "common-voice", "speech-translation"],
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {"split": split, "path": f"data/{split}-*.parquet"}
                    for split in SPLITS
                ],
            }
        ],
    }
    split_table = "\n".join(
        f"| {s} | {v['num_examples']:,} | {v['hours']:.2f} |" for s, v in stats.items()
    )
    card = f"""---
{yaml.safe_dump(metadata, sort_keys=False).strip()}
---

# Feasibility Chinese → English

Full CoVoST 2 `zh-CN → en` subset, packaged for diffusion-speech-recognition.

| Split | Examples | Audio hours |
| --- | ---: | ---: |
{split_table}

## Schema

- `id`: string, unique WAV filename; equals `audio.path` for the project's ID-to-audio mapping.
- `chinese`: original Chinese transcript, preserved without normalization.
- `english`: official English translation, preserved without normalization.
- `audio`: Hugging Face Audio feature with embedded WAV PCM signed 16-bit bytes, mono, 16 kHz.
- `client_id`: original Common Voice speaker identifier.

No Vietnamese or Korean translations are fabricated. Text normalization/tokenization
and encoder embeddings are left to training/preprocessing.

## Provenance and validation

Audio and Chinese transcripts come from the Common Voice v4-based
[CoVoST2 mirror](https://huggingface.co/datasets/{MIRROR}/tree/{REVISION}/zh-CN_en),
pinned to `{REVISION}`. The historical Common Voice v4 S3 URL returned HTTP 403
during preparation, so this mirror supplies the original MP3 bytes.
Every filename, English translation, and split was checked against the
[official CoVoST2 TSV]({TRANSLATIONS}). Following the
[official split script](https://github.com/facebookresearch/covost/blob/main/get_covost_splits.py),
training includes `train` and `train_covost`; validation uses `dev`; test uses `test`.
Other auxiliary/duplicate split labels are excluded. No random re-splitting is applied.
Audio IDs are disjoint across splits; this does not imply speaker or sentence disjointness.

MP3 audio is decoded, averaged to mono if needed, resampled with a polyphase filter,
and written as WAV PCM_16. All records were decoded again with the project's
`src/data/utils.py::_decode_wav_bytes`. See `validation_report.json` for counts and SHA-256 hashes.

[CoVoST upstream](https://github.com/facebookresearch/covost) lists Common Voice
audio/transcripts and CoVoST translations under CC0.

## Usage

```python
from datasets import Audio, load_dataset

dataset = load_dataset("aiai-laboratory/feasibility-cn-en")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
sample = dataset["train"][0]
assert sample["id"] == sample["audio"]["path"]
# sample["audio"]["bytes"] is compatible with the project's WAV decoder.
# Use chinese for the transcript and english for the translation target.
```

The data follows the project's raw translated-speech column naming and WAV format.
Existing training loaders still hardcode VietSpeech repositories and Vietnamese task
tokens; those loaders must be configured/adapted before Chinese → English training.
This is raw audio/text, not precomputed encoder embeddings.
"""
    (output / "README.md").write_text(card)
    print(json.dumps(stats, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "data/feasibility-cn-en"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="aiai-laboratory/feasibility-cn-en")
    args = parser.parse_args()
    load_dotenv(PROJECT / ".env")
    if not args.skip_download:
        download(args.output_dir / "source")
    output = prepare(args.output_dir)
    if args.upload:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=output,
            allow_patterns=["data/*.parquet", "README.md", "validation_report.json"],
            commit_message="Add validated CoVoST2 Chinese-English in project WAV schema",
        )
        print(f"Uploaded: {commit}", flush=True)


if __name__ == "__main__":
    main()
