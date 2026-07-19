import sys
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()


def push_checkpoint_to_hub(repo_id: str, checkpoint_dir: str, repo_type: str = "model", token: str = None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment or .env file.")

    print(f"Target checkpoint directory to push: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found.")

    # Get parent experiment directory
    experiment_dir = os.path.dirname(checkpoint_dir)

    api = HfApi(token=token)

    # Create repo if not exists
    print(f"Verifying/Creating repository {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # 1. Upload files from the experiment directory (args.json, tokenizer files)
    print("Uploading experiment metadata and tokenizer files...")
    meta_files = [
        "args.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for filename in meta_files:
        local_file_path = os.path.join(experiment_dir, filename)
        if os.path.exists(local_file_path):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload {filename}",
            )

    # 2. Upload custom code files
    print("Uploading custom code files...")
    # Try to find custom code files by checking relative to current directory and workspace root
    base_dir = os.getcwd()
    for code_file, path_in_repo in [
        ("src/model/configuration_dlm.py", "configuration_dlm.py"),
        ("src/model/modeling_dlm.py", "modeling_dlm.py"),
        ("src/dd_generator.py", "dd_generator.py"),
    ]:
        full_path = os.path.join(base_dir, code_file)
        if not os.path.exists(full_path):
            # Fallback to direct relative path
            full_path = code_file

        if os.path.exists(full_path):
            print(f"Uploading {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload {path_in_repo}",
            )

    # 3. Upload the checkpoint folder (optimizer.pt, pytorch_model.bin, rng_state.pth, scheduler.pt, trainer_state.json, etc.)
    folder_name = os.path.basename(checkpoint_dir)
    print(f"Uploading checkpoint folder '{folder_name}' to repository...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        path_in_repo=folder_name,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload {folder_name} training checkpoint states",
    )

    print("\nCheckpoint successfully pushed to Hub!")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/model-manager/push_checkpoint.py <repo_id> [checkpoint_dir] [repo_type]"
        )
        print(
            "Example: python scripts/model-manager/push_checkpoint.py aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint outputs/vi_multitask/checkpoint-60000"
        )
        sys.exit(1)

    repo_id = sys.argv[1]
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else None
    repo_type = sys.argv[3] if len(sys.argv) > 3 else "model"

    # If checkpoint_dir is not specified, find the highest one in outputs/vi_multitask
    if checkpoint_dir is None:
        experiment_dir = "outputs/vi_multitask"
        if not os.path.exists(experiment_dir):
            print(f"Default experiment directory {experiment_dir} not found.")
            sys.exit(1)
        checkpoints = [
            d for d in os.listdir(experiment_dir) if d.startswith("checkpoint-")
        ]
        if not checkpoints:
            print(f"No checkpoints found in {experiment_dir}")
            sys.exit(1)
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        checkpoint_dir = os.path.join(experiment_dir, checkpoints[-1])

    try:
        push_checkpoint_to_hub(repo_id, checkpoint_dir, repo_type)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

