import argparse
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a full training checkpoint (including training state, optimizer, etc.) from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID containing the checkpoint.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="outputs/vi_multitask_resumed",
        help="Local directory to download the checkpoint and experiment files to.",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        help="Hugging Face repository type ('model' or 'dataset').",
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
        print(
            "Warning: HF_TOKEN not found in environment or .env file. "
            "If the repository is private, the download might fail."
        )

    print("=============================================================")
    print("  Downloading Training Checkpoint from Hugging Face Hub")
    print(f"  Repository  : {args.repo_id}")
    print(f"  Destination : {args.target_dir}")
    print("=============================================================")

    if os.path.exists(args.target_dir) and not args.force:
        files = os.listdir(args.target_dir)
        if len(files) > 0:
            print(
                f"Target directory '{args.target_dir}' already exists and is not empty."
            )
            print("Use --force to overwrite, or delete/rename the directory.")
            sys.exit(1)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.target_dir,
            token=token,
            max_workers=8,
        )
        print("\nCheckpoint download completed successfully!")
        print("You can now resume training from the downloaded checkpoint. E.g.:")

        # Find checkpoint subdirectories in target_dir
        checkpoints = [
            d for d in os.listdir(args.target_dir) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            highest_ckpt = checkpoints[-1]
            ckpt_path = os.path.join(args.target_dir, highest_ckpt)
            print(
                f"  python src/train.py <config_path> --resume_from_checkpoint {ckpt_path}"
            )
        else:
            print(
                f"  python src/train.py <config_path> --resume_from_checkpoint {args.target_dir}"
            )

    except Exception as e:
        print(f"\nError during download: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
