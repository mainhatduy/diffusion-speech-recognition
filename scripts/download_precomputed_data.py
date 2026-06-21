#!/usr/bin/env python3
"""Download precomputed audio embeddings and tokenized text from Hugging Face.

Usage:
    python scripts/download_precomputed_data.py \
        --target_dir precomputed_data_1 \
        [--test] [--force]
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download precomputed speech dataset from Hugging Face.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="aiai-laboratory/vietspeech-train-precompute",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="precomputed_data_1",
        help="Local directory to save the downloaded dataset.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test/dry-run mode: only download metadata, token IDs, and the first parquet shard (shard_00000.parquet).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment or .env file. "
              "If the dataset is private, the download might fail.")
    
    print("=============================================================")
    print("  Downloading Precomputed Dataset from Hugging Face Hub")
    print(f"  Repository  : {args.repo_id}")
    print(f"  Destination : {args.target_dir}")
    print(f"  Test Mode   : {args.test}")
    print("=============================================================")

    # Check if target directory already exists and has content
    if os.path.exists(args.target_dir) and not args.force:
        # Check if we have files in target_dir
        files = os.listdir(args.target_dir)
        if len(files) > 0:
            print(f"Target directory '{args.target_dir}' already exists and is not empty.")
            print("Use --force to overwrite, or delete the directory if you want a clean download.")
            return

    # Define allowed patterns if in test mode
    allow_patterns = None
    if args.test:
        allow_patterns = [
            "metadata.json",
            "index.json",
            "token_ids/*.json",
            "audio_embeds/shard_00000.parquet",
        ]
        print("Filtering download to metadata and the first data shard (shard_00000.parquet) only...")

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.target_dir,
            token=token,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError during download: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
