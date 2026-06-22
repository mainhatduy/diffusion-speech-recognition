# Data Preprocess Scripts

Thư mục này chứa các scripts phục vụ cho việc chuẩn bị dữ liệu (download, trích xuất đặc trưng âm thanh, tokenize text, chuyển đổi định dạng và upload).

## Danh sách các scripts

### 1. `download_precomputed_data.py`
Dùng để tải bộ dữ liệu đã được tính toán trước (precomputed audio embeddings & tokenized text) từ Hugging Face Hub về máy local để phục vụ training nhanh.

**Cách dùng:**
```bash
uv run python scripts/data-preprocess/download_precomputed_data.py \
    --target_dir precomputed_data \
    [--repo_id aiai-laboratory/vietspeech-train-precompute] \
    [--test] [--force]
```
* `--target_dir`: Thư mục lưu dữ liệu tải về (mặc định: `precomputed_data`).
* `--repo_id`: Repo dataset trên Hugging Face (mặc định: `aiai-laboratory/vietspeech-train-precompute`).
* `--test`: Chỉ tải file metadata và 1 shard dữ liệu đầu tiên để test thử.
* `--force`: Ép buộc tải lại và ghi đè dữ liệu cũ.

---

### 2. `precompute_embeddings.py`
Dùng để tự trích xuất đặc trưng âm thanh (audio embeddings) và tiền xử lý token text từ bộ dataset thô.

**Cách dùng:**
```bash
uv run python scripts/data-preprocess/precompute_embeddings.py \
    --output_dir precomputed_data \
    --audio_encoder_name UsefulSensors/moonshine-streaming-medium \
    --pretrained FacebookAI/xlm-roberta-base \
    --batch_size 32 \
    --max_length 128 \
    [--resume]
```
* `--output_dir`: Thư mục lưu kết quả precomputed.
* `--audio_encoder_name`: Model encoder âm thanh (mặc định: `UsefulSensors/moonshine-streaming-medium`).
* `--pretrained`: Tokenizer/Language Model xương sống (mặc định: `FacebookAI/xlm-roberta-base`).
* `--resume`: Bật chế độ chạy tiếp nếu bị gián đoạn.

---

### 3. `convert_npy_to_parquet.py`
Dùng để nén các file numpy `.npy` đơn lẻ thành các file `.parquet` sharded. Định dạng này giúp huấn luyện tải dữ liệu nhanh và tiết kiệm bộ nhớ nhờ memory mapping.

**Cách dùng:**
```bash
uv run python scripts/data-preprocess/convert_npy_to_parquet.py
```
*Script sẽ tự động tìm thư mục `precomputed_data` ở root của project, đọc file `index.json`, gom nhóm các file `.npy` và chuyển đổi sang định dạng Parquet.*

---

### 4. `upload_precomputed_data_robust.py`
Dùng để tải bộ dữ liệu precomputed sau khi xử lý (đã qua chuyển đổi sang Parquet) lên Hugging Face Hub. Script hỗ trợ tính năng retry và tự động bỏ qua các file đã tải lên trước đó.

**Cách dùng:**
```bash
uv run python scripts/data-preprocess/upload_precomputed_data_robust.py
```

> [!IMPORTANT]
> Cần cấu hình `HF_TOKEN` trong môi trường để thực hiện download (đối với repo private) và upload lên Hugging Face Hub.
