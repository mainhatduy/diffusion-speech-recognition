# Diffusion Speech Recognition

Dự án nhận dạng giọng nói sử dụng mô hình khuếch tán (Diffusion Model). Để chạy dự án một cách nhanh chóng và hiệu quả, bạn cần cài đặt môi trường ảo (virtual environment) phù hợp.

---

## 🚀 Hướng dẫn cài đặt môi trường ảo (Virtual Environment) trên Linux

Dưới đây là hướng dẫn chi tiết cách tạo và kích hoạt môi trường ảo cục bộ trực tiếp trên hệ điều hành Linux của bạn mà không cần phụ thuộc vào bất kỳ máy tính hay thiết bị nào khác.

### 📦 Cách 1: Sử dụng Python `venv` mặc định (Độc lập & Khuyên dùng cho Linux)

Cách này sử dụng thư viện có sẵn của Python, đảm bảo sự đơn giản, độc lập và tương thích tối đa trên mọi máy Linux.

#### **Bước 1: Cài đặt gói bổ trợ cho Linux**
Trên các bản phân phối Linux như Ubuntu/Debian, gói `venv` thường không đi kèm mặc định với Python. Hãy cài đặt bằng lệnh sau:
```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

#### **Bước 2: Clone dự án và truy cập thư mục**
```bash
git clone <url-cua-repo-github>
cd diffusion-speech-recognition
```

#### **Bước 3: Tạo môi trường ảo**
Khởi tạo thư mục môi trường ảo tên là `.venv` trong thư mục dự án:
```bash
python3 -m venv .venv
```

#### **Bước 4: Kích hoạt môi trường ảo**
Hãy chạy lệnh sau để kích hoạt môi trường ảo:
```bash
source .venv/bin/activate
```
> [!TIP]
> Khi kích hoạt thành công, terminal của bạn sẽ hiển thị tiền tố `(.venv)` ở đầu dòng.
> Để thoát khỏi môi trường ảo khi đã làm việc xong, chỉ cần gõ lệnh: `deactivate`.

#### **Bước 5: Cập nhật pip và cài đặt thư viện**
Chạy lệnh sau để nâng cấp công cụ cài đặt `pip` và tải toàn bộ thư viện cần thiết từ file `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### ⚡ Cách 2: Sử dụng công cụ `uv` (Tốc độ cao)

Nếu bạn muốn quá trình cài đặt diễn ra cực nhanh bằng công cụ hiện đại `uv` của Astral:

#### **Bước 1: Cài đặt `uv`**
* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### **Bước 2: Đồng bộ hóa môi trường**
Di chuyển vào thư mục dự án và chạy:
```bash
uv sync
```
`uv` sẽ tự động tải phiên bản Python phù hợp (nếu thiếu), tạo `.venv` và cài đặt toàn bộ thư viện cần thiết.

---

## 💻 Hướng dẫn chạy dự án

Sau khi đã hoàn tất các bước cài đặt trên:

* **Nếu cài đặt bằng Python `venv` (Cách 1):**
  Kích hoạt môi trường ảo và chạy file chính:
  ```bash
  source .venv/bin/activate
  python main.py
  ```

* **Nếu cài đặt bằng `uv` (Cách 2):**
  Chạy trực tiếp mà không cần kích hoạt môi trường ảo theo cách thủ công:
  ```bash
  uv run python main.py
  ```

---

## 📂 Lưu ý khi Git Push

Khi đẩy mã nguồn lên GitHub, hãy đảm bảo các file cấu hình và định nghĩa thư viện sau đã được commit:
* `requirements.txt`
* `pyproject.toml`
* `uv.lock`
* `.python-version`

> [!IMPORTANT]
> **Tuyệt đối không** commit thư mục `.venv/`. Thư mục này chứa hàng GB mã nguồn thư viện cục bộ đã tải về máy của bạn và đã được cấu hình loại trừ trong file `.gitignore` của dự án.
