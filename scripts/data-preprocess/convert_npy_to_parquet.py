#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    precomputed_dir = project_root / "precomputed_data"
    npy_dir = precomputed_dir / "audio_embeds"
    parquet_dir = precomputed_dir / "audio_embeds_parquet"

    print("=== Audio Embeddings to Parquet Conversion ===")

    # 1. Load index.json
    index_path = precomputed_dir / "index.json"
    if not index_path.exists():
        print(
            f"Error: {index_path} not found. Please run speech pre-computation first."
        )
        sys.exit(1)

    print(f"Loading index from {index_path}...")
    with open(index_path) as f:
        index = json.load(f)

    # 2. Extract unique embed files
    print("Extracting unique embed files...")
    unique_files = sorted(list(set(entry["embed_file"] for entry in index)))
    num_files = len(unique_files)
    print(f"Total unique embedding files: {num_files}")

    if num_files == 0:
        print("No embedding files found in index.json.")
        sys.exit(0)

    # Load save_dtype from metadata if possible
    save_dtype_str = "float16"
    metadata_path = precomputed_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            save_dtype_str = metadata.get("save_dtype", "float16")

    save_dtype = np.float16 if save_dtype_str == "float16" else np.float32
    print(f"Detected save dtype: {save_dtype_str}")

    # 3. Define shard parameters
    shard_size = 5000
    num_shards = (num_files + shard_size - 1) // shard_size
    print(f"Configured shard size: {shard_size} files per shard.")
    print(f"Total shards to create: {num_shards}")

    os.makedirs(parquet_dir, exist_ok=True)

    # 4. Conversion loop
    from concurrent.futures import ThreadPoolExecutor

    def load_single_npy(name):
        file_path = npy_dir / name
        if not file_path.exists():
            return None
        try:
            arr = np.load(file_path)
            return {
                "embed_file": name,
                "embedding_bytes": arr.tobytes(),
                "shape": list(arr.shape),
            }
        except Exception as e:
            print(f"\nError reading {file_path}: {e}")
            return None

    start_time = time.time()
    for shard_idx in range(num_shards):
        shard_file = parquet_dir / f"shard_{shard_idx:05d}.parquet"

        # Check if shard already exists (for resume capability)
        if shard_file.exists():
            print(
                f"Shard {shard_idx + 1}/{num_shards} ({shard_file.name}) already exists. Skipping."
            )
            continue

        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, num_files)
        shard_files = unique_files[start_idx:end_idx]

        print(
            f"Processing shard {shard_idx + 1}/{num_shards} ({len(shard_files)} files)..."
        )

        pylist = []
        missing_count = 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(
                tqdm(
                    executor.map(load_single_npy, shard_files),
                    total=len(shard_files),
                    desc=f"Shard {shard_idx:05d}",
                )
            )

        for res in results:
            if res is None:
                missing_count += 1
            else:
                pylist.append(res)

        if missing_count > 0:
            print(f"  Warning: {missing_count} files were missing in this shard.")

        if not pylist:
            print(f"  No data found for shard {shard_idx}. Skipping writing.")
            continue

        # Convert to PyArrow Table and write to Parquet
        table = pa.Table.from_pylist(pylist)
        pq.write_table(table, shard_file, compression="snappy")

        shard_file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        print(f"  Saved {shard_file.name} ({shard_file_size_mb:.2f} MB)")

    elapsed = time.time() - start_time
    print(f"\nAll shards processed in {elapsed:.1f}s.")

    # 5. Swap directories safely
    print("\nSwapping directories to activate Parquet embeddings...")
    backup_dir = precomputed_dir / "audio_embeds_npy"

    if backup_dir.exists():
        print(
            f"Warning: Backup directory {backup_dir} already exists. Appending timestamp."
        )
        backup_dir = precomputed_dir / f"audio_embeds_npy_{int(time.time())}"

    # Rename original to backup
    print(f"Renaming {npy_dir} -> {backup_dir}...")
    shutil.move(npy_dir, backup_dir)

    # Rename parquet to original
    print(f"Renaming {parquet_dir} -> {npy_dir}...")
    shutil.move(parquet_dir, npy_dir)

    print("\nConversion successfully completed!")
    print(f"Original .npy directory backed up to: {backup_dir}")
    print(f"Parquet files now active in: {npy_dir}")
    print(
        "\nYou can now safely push the entire precomputed_data folder to Hugging Face."
    )
    print("Or, if disk space is tight, you can delete the backup directory:")
    print(f"  rm -rf {backup_dir}")


if __name__ == "__main__":
    main()
