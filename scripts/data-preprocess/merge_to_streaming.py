#!/usr/bin/env python3
"""Merge precomputed dataset (metadata, token IDs, audio embeddings parquet)
into a unified streaming-ready Parquet dataset and push to Hugging Face Hub.

Usage:
    python scripts/data-preprocess/merge_to_streaming.py \
        --source_repo aiai-laboratory/vietspeech-train-precompute \
        --target_repo aiai-laboratory/vietspeech-train-streaming \
        [--test] [--dry_run]
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, upload_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge precomputed dataset into a streaming-ready dataset on HF Hub."
    )
    parser.add_argument(
        "--source_repo",
        type=str,
        default="aiai-laboratory/vietspeech-train-precompute",
        help="Source Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--target_repo",
        type=str,
        default="aiai-laboratory/vietspeech-train-streaming",
        help="Target Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only the first shard.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode: process locally without uploading to HF Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create target repository as private if it does not exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if not token and not args.dry_run:
        print(
            "Error: HF_TOKEN not found in environment or .env file. "
            "Please set HF_TOKEN to upload merged shards."
        )
        sys.exit(1)

    api = HfApi(token=token)

    print("=============================================================")
    print("  Merging Dataset to Streaming Format")
    print(f"  Source Repo : {args.source_repo}")
    print(f"  Target Repo : {args.target_repo}")
    print(f"  Test Mode   : {args.test}")
    print(f"  Dry Run     : {args.dry_run}")
    print("=============================================================")

    # Ensure target repo exists on HF Hub if not dry run
    if not args.dry_run:
        try:
            api.create_repo(
                repo_id=args.target_repo,
                repo_type="dataset",
                private=args.private,
                exist_ok=True,
            )
            print(f"Target repo '{args.target_repo}' verified/created.")
        except Exception as e:
            print(f"Warning/Error creating target repo: {e}")

    # 1. Download metadata, index, and token_ids JSONs (lightweight)
    print("\nDownloading metadata and token ID mappings from source repo...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_path = hf_hub_download(
            repo_id=args.source_repo,
            filename="metadata.json",
            repo_type="dataset",
            token=token,
            local_dir=tmp_dir,
        )
        index_path = hf_hub_download(
            repo_id=args.source_repo,
            filename="index.json",
            repo_type="dataset",
            token=token,
            local_dir=tmp_dir,
        )

        with open(meta_path) as f:
            metadata = json.load(f)

        with open(index_path) as f:
            index_data = json.load(f)

        # Download token ID json files
        token_ids_dir = os.path.join(tmp_dir, "token_ids")
        os.makedirs(token_ids_dir, exist_ok=True)
        task_tokens_map = {}

        tasks = ["english", "chinese", "korean"]
        for task in tasks:
            try:
                task_path = hf_hub_download(
                    repo_id=args.source_repo,
                    filename=f"token_ids/{task}.json",
                    repo_type="dataset",
                    token=token,
                    local_dir=tmp_dir,
                )
                with open(task_path) as f:
                    task_tokens_map[task] = json.load(f)
                print(f"  Loaded {len(task_tokens_map[task])} token entries for '{task}'")
            except Exception as e:
                print(f"  Warning: token_ids/{task}.json not loaded: {e}")

        # Upload metadata.json to target repo if not dry run
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=meta_path,
                path_in_repo="metadata.json",
                repo_id=args.target_repo,
                repo_type="dataset",
                token=token,
            )
            print("Uploaded metadata.json to target repo.")

    # 2. Build index lookups mapping embed_file -> token_ids & original index
    print("\nBuilding embed_file lookup map...")
    embed_file_to_tokens = {}
    for entry in index_data:
        data_idx = entry["idx"]
        embed_file = entry["embed_file"]
        tokens_entry = {
            "idx": data_idx,
            "wav_id": entry.get("wav_id", ""),
        }
        for task in tasks:
            if task in task_tokens_map and data_idx < len(task_tokens_map[task]):
                tokens_entry[task] = task_tokens_map[task][data_idx]
            else:
                tokens_entry[task] = []
        embed_file_to_tokens[embed_file] = tokens_entry

    print(f"Lookup map built for {len(embed_file_to_tokens)} unique embedding files.")

    # 3. List all parquet shards in source repo
    repo_files = list_repo_files(repo_id=args.source_repo, repo_type="dataset", token=token)
    parquet_shards = sorted(
        [f for f in repo_files if f.startswith("audio_embeds/") and f.endswith(".parquet")]
    )
    print(f"\nFound {len(parquet_shards)} parquet shards in source repo.")

    if args.test:
        parquet_shards = parquet_shards[:1]
        print(f"[Test Mode] Processing only 1 shard: {parquet_shards[0]}")

    # 4. Process shard by shard
    for idx, shard_file_path in enumerate(parquet_shards):
        shard_name = os.path.basename(shard_file_path)
        print(f"\n[{idx+1}/{len(parquet_shards)}] Processing shard: {shard_name}...")

        with tempfile.TemporaryDirectory() as shard_tmp_dir:
            # Download single parquet shard
            local_shard = hf_hub_download(
                repo_id=args.source_repo,
                filename=shard_file_path,
                repo_type="dataset",
                token=token,
                local_dir=shard_tmp_dir,
            )

            # Read PyArrow table
            table = pq.read_table(local_shard)
            pylist = table.to_pylist()
            print(f"  Read {len(pylist)} rows from {shard_name}")

            merged_rows = []
            for row in pylist:
                embed_file = row["embed_file"]
                tok_info = embed_file_to_tokens.get(embed_file, {})

                merged_row = {
                    "embed_file": embed_file,
                    "embedding_bytes": row["embedding_bytes"],
                    "shape": row.get("shape", []),
                    "english": tok_info.get("english", []),
                    "chinese": tok_info.get("chinese", []),
                    "korean": tok_info.get("korean", []),
                }
                merged_rows.append(merged_row)

            # Write to new merged Parquet shard
            out_shard_path = os.path.join(shard_tmp_dir, shard_name)
            merged_table = pa.Table.from_pylist(merged_rows)
            pq.write_table(merged_table, out_shard_path, compression="snappy")
            out_size_mb = os.path.getsize(out_shard_path) / (1024 * 1024)
            print(f"  Created merged shard {shard_name} ({out_size_mb:.2f} MB)")

            # Upload shard to target repo
            if not args.dry_run:
                target_path_in_repo = f"data/{shard_name}"
                print(f"  Uploading {shard_name} to {args.target_repo}:{target_path_in_repo}...")
                api.upload_file(
                    path_or_fileobj=out_shard_path,
                    path_in_repo=target_path_in_repo,
                    repo_id=args.target_repo,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  Uploaded {shard_name} successfully!")

    print("\n=============================================================")
    print("  Dataset Merge Completed Successfully!")
    print("=============================================================")


if __name__ == "__main__":
    main()
