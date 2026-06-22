import os
import sys
import time
from dotenv import load_dotenv
from huggingface_hub import HfApi

def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in .env file.")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = "aiai-laboratory/vietspeech-train-precompute"

    print("=== Robust Uploading Precomputed Data to Hugging Face ===")
    print(f"Repository: {repo_id} (dataset)")
    
    # 1. Ensure repository exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print("Dataset repository verified/created.")
    except Exception as e:
        print(f"Warning during repository creation: {e}")
        print("Attempting to proceed...")

    # 2. Get list of files already in the remote repo
    print("Fetching list of existing files in Hugging Face repository...")
    try:
        existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        print(f"Found {len(existing_files)} files already in the repository.")
    except Exception as e:
        print(f"Error fetching repo files: {e}")
        existing_files = set()

    # 3. Walk through precomputed_data to find files to upload
    local_dir = "precomputed_data"
    files_to_upload = []

    for root, dirs, files in os.walk(local_dir):
        # Exclude audio_embeds_npy directory
        if "audio_embeds_npy" in root:
            continue
        
        for file in files:
            local_path = os.path.join(root, file)
            # Get relative path inside precomputed_data folder
            rel_path = os.path.relpath(local_path, local_dir)
            
            # Skip if already uploaded
            if rel_path in existing_files:
                print(f"[Skip] {rel_path} already exists on Hub.")
            else:
                files_to_upload.append((local_path, rel_path))

    total_files = len(files_to_upload)
    print(f"\nTotal files remaining to upload: {total_files}")
    if total_files == 0:
        print("All files are already uploaded!")
        return

    # Sort files to upload metadata and index first
    files_to_upload.sort(key=lambda x: (not x[1].endswith('.json'), x[1]))

    # 4. Upload files one by one
    for idx, (local_path, rel_path) in enumerate(files_to_upload, 1):
        file_size_gb = os.path.getsize(local_path) / (1024 * 1024 * 1024)
        print(f"\n[{idx}/{total_files}] Uploading {rel_path} ({file_size_gb:.3f} GB)...")
        
        # Retry logic
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {rel_path}"
                )
                elapsed = time.time() - start_time
                print(f"Successfully uploaded {rel_path} in {elapsed:.1f}s.")
                break
            except Exception as e:
                print(f"Attempt {attempt}/{max_retries} failed to upload {rel_path}: {e}")
                if attempt < max_retries:
                    time.sleep(10)
                else:
                    print("Max retries reached. Exiting.")
                    sys.exit(1)

    print("\nAll remaining files uploaded successfully!")

if __name__ == "__main__":
    main()
