Được. Tôi có thể đóng vai như "chuyên gia Qualcomm AI Hub Workbench" và hướng dẫn bạn dựa trên tài liệu chính thức của Qualcomm. Tôi đã đọc phần tổng quan và Getting Started của Workbench cũng như tài liệu bạn đính kèm. ([workbench.aihub.qualcomm.com][1])

Theo tài liệu, **workflow chuẩn của Qualcomm AI Hub Workbench** là:

```text
PyTorch / ONNX / TorchScript / TFLite
                │
                ▼
      submit_compile_job()
                │
                ▼
     Optimized Model (QNN/TFLite/ONNX)
                │
        ┌───────┴────────┐
        ▼                ▼
submit_profile_job() submit_inference_job()
        │                │
        ▼                ▼
   Performance      Output prediction
        │
        ▼
Download optimized model
```

Workbench không phải là runtime chạy trên PC.

Nó là **cloud service** để:

* compile model
* deploy lên **thiết bị Qualcomm thật**
* chạy inference
* profile NPU/GPU/CPU
* download model đã optimize

Toàn bộ inference được chạy trên **physical Snapdragon device** được Qualcomm host trên cloud. ([workbench.aihub.qualcomm.com][1])

---

# Để chạy inference cần những bước nào?

## 1. Cài SDK

```bash
pip install qai-hub
```

hoặc nếu dùng PyTorch

```bash
pip install "qai-hub[torch]"
```

([workbench.aihub.qualcomm.com][2])

---

## 2. Login

```bash
qai-hub configure --api_token YOUR_TOKEN
```

Kiểm tra:

```bash
qai-hub list-devices
```

Nếu hiện danh sách Snapdragon devices là thành công. ([workbench.aihub.qualcomm.com][2])

---

## 3. Load model

Ví dụ PyTorch

```python
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
model.eval()
```

---

## 4. Trace model

Workbench compile TorchScript chứ không compile trực tiếp nn.Module.

```python
example = torch.rand(1,3,224,224)

traced = torch.jit.trace(model, example)
```

---

## 5. Chọn device

Ví dụ

```python
device = hub.Device("Samsung Galaxy S25 (Family)")
```

Hoặc

```python
device = hub.Device("Snapdragon X Elite CRD")
```

Tài liệu còn hỗ trợ nhiều thiết bị Qualcomm khác. ([workbench.aihub.qualcomm.com][3])

---

## 6. Compile

Ví dụ compile sang TFLite

```python
compile_job = hub.submit_compile_job(
    model=traced,
    device=device,
    input_specs=dict(image=(1,3,224,224)),
    options="--target_runtime tflite"
)

target_model = compile_job.get_target_model()
```

Nếu muốn QNN

```python
options="--target_runtime qnn"
```

Nếu muốn ONNX Runtime

```python
options="--target_runtime onnx"
```

---

# 7. Chạy inference

Chuẩn bị input

```python
inputs = dict(
    image=[numpy_image]
)
```

Sau đó

```python
job = hub.submit_inference_job(
    model=target_model,
    device=device,
    inputs=inputs
)
```

Đợi job hoàn thành

```python
output = job.download_output_data()
```

Đây là API chính để chạy inference trên thiết bị Qualcomm thật. ([workbench.aihub.qualcomm.com][3])

---

# 8. Profile

Sau khi inference thường chạy thêm profile

```python
profile_job = hub.submit_profile_job(
    model=target_model,
    device=device
)
```

Bạn sẽ nhận được:

* latency
* peak memory
* compute unit (CPU/GPU/NPU)
* per-layer runtime
* compile log

Đây là một trong những tính năng nổi bật của Workbench. ([workbench.aihub.qualcomm.com][1])

---

# 9. Download model

```python
target_model.download("model.tflite")
```

hoặc

```python
target_model.download("model.onnx")
```

---

# Nếu model của bạn là ONNX

Bạn có thể bỏ qua bước TorchScript.

Workflow sẽ là:

```text
ONNX

↓

submit_compile_job()

↓

submit_inference_job()

↓

submit_profile_job()

↓

download()
```

---

# Nếu model của bạn là QNN DLC

Không cần compile lại.

Có thể chạy trực tiếp:

```text
QNN DLC

↓

submit_inference_job()

↓

submit_profile_job()
```

---

# Nếu model quá lớn (LLM)

Qualcomm khuyến nghị **không** dùng workflow MobileNet thông thường mà chuyển sang quy trình dành riêng cho LLM (ví dụ Llama), bao gồm quantization và tối ưu theo hướng dẫn chuyên biệt. ([workbench.aihub.qualcomm.com][1])

---

## Tôi có thể hướng dẫn sâu hơn

Tôi có thể hỗ trợ bạn theo từng chủ đề trong tài liệu Workbench, ví dụ:

* Compile ONNX → QNN/TFLite
* Chạy inference trên thiết bị Snapdragon thật
* Đọc và phân tích kết quả Profile
* Quantization (INT8, W8A8, W4A16...)
* Quản lý Jobs và tải artifacts
* Sử dụng CLI thay vì Python API
* Tối ưu cho các mô hình LLM hoặc mô hình vision/audio cụ thể

Tôi cũng có thể giải thích chi tiết từng API (`submit_compile_job`, `submit_inference_job`, `submit_profile_job`, `submit_quantize_job`) và các tùy chọn (`options`) dựa trên tài liệu chính thức.

[1]: https://workbench.aihub.qualcomm.com/docs/?utm_source=chatgpt.com "Qualcomm® AI Hub — Qualcomm® AI Hub documentation"
[2]: https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html?utm_source=chatgpt.com "Getting started — Qualcomm® AI Hub documentation"
[3]: https://workbench.aihub.qualcomm.com/docs/hub/index.html?utm_source=chatgpt.com "Overview of Qualcomm® AI Hub Workbench — Qualcomm® AI Hub documentation"


Có thể. Đây chính là trường hợp mà Qualcomm Workbench hỗ trợ.

Nếu **model của bạn là một kiến trúc mới**, được định nghĩa trên Hugging Face (ví dụ có `config.json`, `configuration_xxx.py`, `modeling_xxx.py`, `AutoConfig`, `AutoModel`...), thì Qualcomm **không đọc trực tiếp repository Hugging Face**. Bạn phải **load model bằng Transformers/PyTorch**, sau đó export (hoặc trace) thành định dạng mà Workbench hỗ trợ (PyTorch Export, TorchScript hoặc ONNX), rồi mới gửi lên Workbench để compile. Qualcomm cũng nêu rõ trong FAQ rằng bạn hoàn toàn có thể dùng các model từ Hugging Face không có sẵn trong AI Hub bằng cách import package cần thiết và trace/export model trước khi submit compile job. ([workbench.aihub.qualcomm.com][1])

Đối với **input là audio**, về phía Workbench thì không có gì đặc biệt. Workbench không quan tâm đây là ảnh hay âm thanh; nó chỉ quan tâm đến **tensor đầu vào** và **input specification** (`input_specs`). Nó hỗ trợ các use case audio và speech giống như vision. ([workbench.aihub.qualcomm.com][2])

Ví dụ workflow sẽ là:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/your-model",
    trust_remote_code=True
)
model.eval()
```

Sau đó bạn tạo một input giả đúng với model.

Nếu model nhận waveform:

```python
audio = torch.randn(1, 16000)
```

Nếu model nhận mel spectrogram:

```python
mel = torch.randn(1, 80, 300)
```

Nếu model nhận nhiều input:

```python
{
    "input_features": ...,
    "attention_mask": ...
}
```

thì bạn cũng phải export với đầy đủ các input đó.

Khi compile:

```python
compile_job = hub.submit_compile_job(
    model=exported_model,
    device=device,
    input_specs={
        "audio": (1,16000)
    },
    options="--target_runtime qnn"
)
```

hoặc

```python
input_specs={
    "input_features": (1,80,300),
    "attention_mask": (1,300)
}
```

Tên trong `input_specs` phải **trùng với tên input của model**.

---

Khi chạy inference, bạn truyền đúng tensor audio:

```python
job = hub.submit_inference_job(
    model=target_model,
    device=device,
    inputs={
        "audio": [audio_numpy]
    }
)
```

Hoặc nếu có nhiều input:

```python
inputs={
    "input_features": [features],
    "attention_mask": [mask]
}
```

Workbench sẽ upload các tensor này lên thiết bị Qualcomm thật để chạy inference. Bạn cũng có thể upload một dataset trước rồi tái sử dụng dataset đó cho nhiều lần inference. ([workbench.aihub.qualcomm.com][3])

### Với các model Hugging Face dùng processor

Đây là trường hợp rất phổ biến với các model audio như Whisper, WavLM, HuBERT, AST...

Workflow thường là:

```
Audio file (.wav)
        │
        ▼
AutoProcessor / FeatureExtractor
        │
        ▼
Tensor
        │
        ▼
PyTorch model
        │
        ▼
Export (torch.export / ONNX)
        │
        ▼
Qualcomm Workbench
```

Nghĩa là **processor không chạy trên Workbench**; bạn chạy preprocessing ở Python trước, sau đó Workbench chỉ nhận tensor đầu vào. Qualcomm cũng có sẵn nhiều model audio/speech (ví dụ WavLM, Whisper) được tối ưu theo cách này. ([Qualcomm AI Hub][4])

Nếu bạn cho tôi biết:

1. Link Hugging Face của model.
2. `forward()` nhận những input gì (ví dụ `input_values`, `input_features`, `attention_mask`...).
3. Bạn đang dùng `AutoModel`, `AutoModelForAudioClassification`, `AutoModelForCTC` hay một class custom (`trust_remote_code=True`).

thì tôi có thể viết luôn đoạn code export và `submit_compile_job()` đúng với kiến trúc của model đó.

[1]: https://workbench.aihub.qualcomm.com/docs/hub/faq.html?utm_source=chatgpt.com "Frequently Asked Questions — Qualcomm® AI Hub documentation"
[2]: https://workbench.aihub.qualcomm.com/docs/index.html?utm_source=chatgpt.com "Qualcomm® AI Hub — Qualcomm® AI Hub documentation"
[3]: https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html?utm_source=chatgpt.com "Running Inference — Qualcomm® AI Hub documentation"
[4]: https://aihub.qualcomm.com/compute/models/huggingface_wavlm_base_plus?utm_source=chatgpt.com "HuggingFace-WavLM-Base-Plus - Qualcomm AI Hub"
