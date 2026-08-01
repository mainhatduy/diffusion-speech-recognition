import sys
import os
from datasets import load_dataset
from huggingface_hub import login

def load_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def main():
    try:
        # Tải biến môi trường từ .env
        load_env()
        hf_token = os.environ.get("HF_TOKEN")
        
        if hf_token:
            print("Đã tìm thấy HF_TOKEN, tiến hành đăng nhập...")
            login(token=hf_token)

        # Lần này ta bỏ qua metadata của dataset gốc bằng cách tải file parquet như một dataset độc lập!
        print("Đang tải file parquet thông qua hf:// ...")
        dataset_shard = load_dataset(
            "parquet", 
            data_files="hf://datasets/NhutP/VietSpeech/data/train-00000-of-00027.parquet",
            split="train"
        )
        print(f"Đã tải thành công {len(dataset_shard)} mẫu.")

        # Push to the target repository
        print("Đang đẩy (push) dữ liệu lên aiai-laboratory/VietSpeech-test...")
        dataset_shard.push_to_hub("aiai-laboratory/VietSpeech-test", private=True, token=hf_token)
        print("Hoàn tất! Dữ liệu đã được đẩy thành công.")

    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
