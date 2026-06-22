# Model Manager Scripts

Thư mục này chứa các scripts dùng để quản lý model và checkpoint trên Hugging Face Hub hoặc local. Các chức năng được tách biệt rõ ràng giữa **Model thông thường** (chỉ có weights, config, tokenizer dùng cho inference/deployment) và **Checkpoint** (gồm toàn bộ training state, optimizer, scheduler, rng_state dùng để resume training).

---

## Danh sách các scripts

### 1. `push_model.py`
Đẩy model thông thường lên Hugging Face Hub (chỉ bao gồm weights `pytorch_model.bin`, file config, tokenizer, custom code và README.md). Không chứa thông tin optimizer hay training state cồng kềnh.

**Cách dùng:**
```bash
uv run python scripts/model-manager/push_model.py <repo_id> [experiment_dir] [checkpoint_dir]
```
* `<repo_id>`: Tên repository đích trên Hugging Face (ví dụ: `your-username/your-model-name`).
* `[experiment_dir]` (Mặc định: `outputs/vi_multitask`): Thư mục chứa config `args.json` và files tokenizer.
* `[checkpoint_dir]` (Mặc định: checkpoint cao nhất trong `experiment_dir`): Thư mục cụ thể chứa checkpoint weights.

**Ví dụ:**
```bash
uv run python scripts/model-manager/push_model.py aiai-laboratory/discrete-diffusion-vi-multitask
```

---

### 2. `push_checkpoint.py`
Đẩy toàn bộ trạng thái huấn luyện (checkpoint folder gồm `pytorch_model.bin`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin` cùng với config, tokenizer và custom code) lên Hugging Face Hub để lưu trữ hoặc chuyển đổi máy tập huấn luyện.

**Cách dùng:**
```bash
uv run python scripts/model-manager/push_checkpoint.py <repo_id> [checkpoint_dir] [repo_type]
```
* `<repo_id>`: Tên repository đích trên Hugging Face.
* `[checkpoint_dir]` (Mặc định: checkpoint cao nhất trong `outputs/vi_multitask`): Thư mục checkpoint cần đẩy lên.
* `[repo_type]` (Mặc định: `model`): Loại repository trên Hugging Face (`model` hoặc `dataset`).

**Ví dụ:**
```bash
uv run python scripts/model-manager/push_checkpoint.py aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint outputs/vi_multitask/checkpoint-60000
```

---

### 3. `load_model.py`
Load model từ local checkpoint hoặc từ Hugging Face Hub (chỉ load weights và tokenizer) để chạy thử nghiệm inference trên file âm thanh đầu vào.

**Cách dùng:**
```bash
uv run python scripts/model-manager/load_model.py <model_path_or_repo_id> [audio_path] [json_path]
```
* `<model_path_or_repo_id>`: Đường dẫn tới checkpoint local (ví dụ: `outputs/vi_multitask/checkpoint-60000`) hoặc Repo ID trên Hugging Face (ví dụ: `aiai-laboratory/discrete-diffusion-vi-multitask`).
* `[audio_path]` (Mặc định: `test/test_data/test_sample.mp3`).
* `[json_path]` (Mặc định: `test/test_data/test_sample.json`).

**Ví dụ:**
```bash
# Load local model
uv run python scripts/model-manager/load_model.py outputs/vi_multitask/checkpoint-60000

# Load model từ Hugging Face
uv run python scripts/model-manager/load_model.py aiai-laboratory/discrete-diffusion-vi-multitask
```

---

### 4. `load_checkpoint.py`
Tải toàn bộ thư mục checkpoint (gồm đầy đủ optimizer, scheduler, rng_state, và training_args) từ Hugging Face Hub về máy local để có thể tiếp tục quá trình huấn luyện (resume training).

**Cách dùng:**
```bash
uv run python scripts/model-manager/load_checkpoint.py --repo_id <repo_id> [--target_dir outputs/vi_multitask_resumed]
```
* `--repo_id`: Repo ID chứa checkpoint trên Hugging Face.
* `--target_dir` (Mặc định: `outputs/vi_multitask_resumed`): Thư mục lưu checkpoint tải về.

**Ví dụ:**
```bash
uv run python scripts/model-manager/load_checkpoint.py --repo_id aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint
```
*Sau khi tải về thành công, bạn có thể chạy tiếp training bằng cách truyền tham số `--resume_from_checkpoint` trỏ tới thư mục checkpoint vừa tải.*

---

> [!IMPORTANT]
> Hãy chắc chắn rằng bạn đã thiết lập biến môi trường `HF_TOKEN` trong file `.env` hoặc hệ thống trước khi chạy các script liên quan đến tải/đẩy dữ liệu riêng tư trên Hugging Face.
