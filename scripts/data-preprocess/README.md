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

---

### 5. `extract_validation.py`
Dùng để tách tập validation (validation split) từ dataset gốc `aiai-laboratory/vietspeech-train-translated` theo cấu hình chia tập tương tự như trong quá trình train (shuffle seed=42, test_size=0.01). Tập validation này chứa đầy đủ label cho cả 4 ngôn ngữ (Vietnamese, English, Chinese, Korean) cùng ID. Kết quả được lưu dưới dạng file Parquet và có thể tùy chọn upload trực tiếp lên Hugging Face.

**Cách dùng:**
```bash
uv run python scripts/data-preprocess/extract_validation.py \
    --output_path outputs/validation.parquet \
    [--upload] \
    [--repo_id aiai-laboratory/vietspeech-validation-translated]
```
* `--output_path`: Đường dẫn lưu file validation parquet.
* `--upload`: Bật cờ này để upload lên Hugging Face Hub sau khi tạo xong.
* `--repo_id`: Repo dataset đích trên Hugging Face (mặc định: `aiai-laboratory/vietspeech-validation-translated`).

> [!IMPORTANT]
> Cần cấu hình `HF_TOKEN` trong file `.env` hoặc biến môi trường để thực hiện tải dataset gốc và upload file kết quả lên Hugging Face Hub.

