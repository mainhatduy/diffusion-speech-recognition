import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset, Audio
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Extract validation split from vietspeech-train-translated, map raw audio bytes from VietSpeech, save as parquet, and optionally upload to Hugging Face.")
    parser.add_argument("--repo_id", type=str, default="aiai-laboratory/vietspeech-validation-translated", help="Hugging Face repo ID to upload to.")
    parser.add_argument("--output_path", type=str, default="outputs/validation.parquet", help="Path to save the validation parquet file.")
    parser.add_argument("--upload", action="store_true", help="Whether to upload to Hugging Face.")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in .env file.")
        return

    print("Loading translated speech dataset from aiai-laboratory/vietspeech-train-translated...")
    dataset = load_dataset(
        "aiai-laboratory/vietspeech-train-translated",
        token=token,
        split="train[:100%]"
    )
    print("Original dataset size:", len(dataset))

    print("Loading audio dataset from NhutP/VietSpeech...")
    vietspeech_dataset = load_dataset(
        "NhutP/VietSpeech",
        token=token,
        split="train[:100%]"
    )
    # Cast to avoid decoding audio waveform locally (avoids torchcodec dependency)
    vietspeech_dataset = vietspeech_dataset.cast_column("audio", Audio(decode=False))

    # Build mapping from path to index in vietspeech_dataset
    print("Mapping audio paths in VietSpeech...")
    audio_column = vietspeech_dataset.data.column("audio")
    vs_paths = []
    for chunk in audio_column.chunks:
        vs_paths.extend(chunk.field("path").to_pylist())
    path_to_vs_idx = {path: idx for idx, path in enumerate(vs_paths)}
    print(f"VietSpeech paths mapped: {len(path_to_vs_idx)}")

    print("Shuffling and splitting dataset with seed 42, test_size 0.01...")
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.01, seed=42)
    val_dataset = split["test"]
    print(f"Validation split size: {len(val_dataset)}")
    print("Columns in validation text dataset:", val_dataset.column_names)

    # Add audio column
    print("Mapping audio bytes to validation split...")
    def add_audio(batch):
        audio_list = []
        for wav_id in batch["id"]:
            vs_idx = path_to_vs_idx.get(wav_id)
            if vs_idx is not None:
                audio_item = vietspeech_dataset[vs_idx]["audio"]
                audio_list.append({
                    "bytes": audio_item["bytes"],
                    "path": audio_item["path"]
                })
            else:
                audio_list.append(None)
        batch["audio"] = audio_list
        return batch

    val_dataset = val_dataset.map(add_audio, batched=True, batch_size=1000)
    val_dataset = val_dataset.cast_column("audio", Audio(decode=False))
    print("Final columns:", val_dataset.column_names)

    # Ensure output directory exists
    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Saving validation set to parquet at: {args.output_path}")
    val_dataset.to_parquet(args.output_path)
    print("Saved successfully!")

    if args.upload:
        print(f"Uploading to Hugging Face dataset repository: {args.repo_id}")
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
            print("Dataset repository created/verified.")
        except Exception as e:
            print(f"Warning during repository creation: {e}")
            
        print("Uploading parquet file...")
        try:
            api.upload_file(
                path_or_fileobj=args.output_path,
                path_in_repo="validation.parquet",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Add validation split parquet file with audio"
            )
            print("Upload completed successfully!")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
