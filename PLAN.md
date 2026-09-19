# PLAN.md — Hoàn thiện Streaming Training cho Discrete-Diffusion AST

## 1. Mục tiêu và tiêu chí hoàn thành

V1 chỉ hoàn thiện đường huấn luyện streaming; chưa triển khai microphone pipeline, stateful inference hoàn chỉnh, ONNX hoặc Qualcomm deployment.

Kết quả cuối phải bảo đảm:

- Streaming training sử dụng cùng một audio-conditioning path ở dataset, trainer và validation.
- Audio prefix không bỏ mất phần đuôi và giữ được thứ tự thời gian giữa các chunk.
- Target prefix được lấy từ alignment cache khi có; fallback về tỷ lệ coverage theo thời gian thật khi alignment lỗi hoặc thiếu.
- Training mô phỏng `frozen prefix + active suffix`, không đưa toàn bộ target tương lai vào canvas.
- Length predictor học số token mới của từng chunk thay vì trung bình token/utterance.
- Confidence loss có gradient và được đánh giá bằng calibration metric.
- Validation chạy trên audio prefix xác định trước; không dùng full-audio/offline path để chọn checkpoint streaming.
- Checkpoint streaming có thể save, resume và reload với output nhất quán.
- Toàn bộ đường offline hiện tại tiếp tục hoạt động.
- Acceptance đầu tiên áp dụng cho Vi→En; interface vẫn hỗ trợ Vi→Zh và Vi→Ko.

Không chạy full 50.000 steps trước khi hoàn tất toàn bộ correctness gate và smoke-training gate.

---

## 2. Kiến trúc training đích

### 2.1. Luồng dữ liệu

```text
Vietnamese audio + Vietnamese transcript + target translation
        │
        ├── Offline CTC forced alignment
        │       Vietnamese word → end timestamp
        │
        ├── Bilingual word alignment
        │       Vietnamese word ↔ target word
        │
        └── Streaming alignment cache
                target token → earliest supported audio timestamp

Audio utterance
        │
        ├── Exact chunking: chunk=2.0s, overlap=0.5s, hop=1.5s
        ├── Chọn current chunk k
        ├── Chọn previous cutoff p, 0 ≤ p < k
        └── Tạo training state
                frozen target prefix: supported tại p
                active target suffix: mới được support từ p+1..k
                unsupported future: không đưa vào model canvas
```

Model path:

```text
raw/precomputed chunks
    → frozen Moonshine Streaming encoder nếu là raw
    → StreamingAudioAdapter
    → AudioQueryResampler cho từng chunk
    → global audio-time encoding
    → concatenate theo thứ tự chunk
    → StreamingDiffusionBackbone
    → token logits + hidden states
    → diffusion CE + calibration loss + chunk-length loss
```

### 2.2. Nguyên tắc prefix-to-prefix

Mỗi training sample có hai mốc:

- `previous_cutoff`: phần target đã được coi là frozen.
- `current_cutoff`: phần target hiện đã có đủ audio evidence.

Canvas chỉ chứa target prefix tại `current_cutoff`:

```text
[TASK] [BOS] [frozen tokens] [active tokens]
```

Quy tắc:

- Task token và BOS luôn cố định.
- Frozen tokens luôn unmasked và không có token loss.
- Chỉ active tokens được áp dụng diffusion noise.
- CE chỉ tính trên active positions bị mask.
- Unsupported future tokens hoàn toàn không xuất hiện trong canvas.
- EOS chỉ xuất hiện khi cutoff đã phủ hết utterance.
- Full-audio sample vẫn được đưa vào với xác suất mặc định `0.20` để bảo toàn năng lực dịch hoàn chỉnh.

Cách này thay thế chiến lược “full transcript + unsupported loss 0.3”, tránh oracle target-length leak và tránh target tương lai trở thành bidirectional context.

---

## 3. Hợp đồng dữ liệu streaming

### 3.1. Streaming sample

Chuẩn hóa mọi raw, local-precomputed và Hub-streaming dataset về một sample contract:

```python
StreamingTrainingSample = {
    "id": str | int,
    "audio_kind": Literal["raw", "precomputed"],
    # Raw: [C, chunk_samples]
    # Precomputed: [C, chunk_frames, audio_hidden]
    "audio_chunks": Tensor,
    "audio_chunk_mask": BoolTensor[C],
    "chunk_start_samples": LongTensor[C],
    "chunk_valid_samples": LongTensor[C],
    # Target prefix at current cutoff; includes BOS and EOS only at final cutoff
    "text_ids": LongTensor[T],
    # Number of target tokens after BOS that are frozen
    "frozen_prefix_len": int,
    # Number of new target tokens trained in this state
    "active_target_len": int,
    # Exact number of new supported target tokens from previous to current cutoff
    "new_token_count": int,
    "task_token_id": int,
    "is_final_prefix": bool,
    "alignment_source": Literal[
        "ctc_bilingual",
        "coverage_ratio",
    ],
}
```

`frozen_prefix_len` và `active_target_len` không tính task token; BOS được giữ cố định và không đưa vào diffusion loss.

### 3.2. Exact chunking

Dùng công thức:

```python
num_chunks = max(1, ceil(max(total_samples - chunk_samples, 0) / hop_samples) + 1)
start_i = i * hop_samples
valid_i = min(chunk_samples, total_samples - start_i)
covered_end(k) = min(total_samples, (k - 1) * hop_samples + chunk_samples)
coverage(k) = covered_end(k) / total_samples
```

Yêu cầu:

- Utterance ngắn hơn 2 giây vẫn có một padded chunk.
- Audio dài hơn đúng một chunk dù chỉ một sample phải tạo thêm final chunk.
- Không tạo chunk chỉ chứa overlap đã được chunk trước phủ hoàn toàn.
- `coverage=1` chỉ khi chunk prefix thực sự phủ tới cuối audio.
- Luôn lưu `chunk_valid_samples` để padding không bị hiểu là speech.

### 3.3. Fallback không alignment

Fallback dùng thời gian audio thật:

```python
supported_content_tokens = floor(
    coverage(current_chunk) * number_of_target_content_tokens
)
```

Quy tắc đặc biệt:

- BOS luôn supported.
- EOS chỉ supported khi `coverage == 1`.
- Không tính task token vào target length.
- `previous_supported_len` được tính bằng cùng công thức tại previous cutoff.
- Kết quả phải monotonic theo chunk index.

### 3.4. Collator

Streaming collator phải:

- Hỗ trợ raw tensor `[C, samples]`.
- Hỗ trợ precomputed tensor `[C, frames, hidden]`.
- Pad riêng trục chunk và trục sample/frame.
- Giữ dtype của embedding; không ép mọi audio về `float32` nếu config dùng BF16/FP16.
- Trả `audio_chunk_mask`, `audio_frame_mask`, chunk starts và valid lengths.
- Assert batch không trộn `audio_kind`.
- Luôn trả `task_token_ids`; không suy ra task từ sample đầu tiên.
- Pad text bằng tokenizer pad ID và trả riêng:
  - `text_attention_mask`
  - `frozen_token_mask`
  - `active_token_mask`
  - `is_final_prefix`

---

## 4. Alignment pipeline hai tầng

### 4.1. Chạy offline, không chạy trong DataLoader

Thêm preprocessing command:

```bash
uv run python scripts/data-preprocess/build_streaming_alignments.py \
  --audio-dataset NhutP/VietSpeech \
  --translation-dataset aiai-laboratory/vietspeech-train-translated \
  --tasks vi_en \
  --ctc-model nguyenvulebinh/wav2vec2-base-vietnamese-250h \
  --target-aligner simalign \
  --output-dir data/streaming-alignments-v1 \
  --resume
```

Preprocessing phải:

- Join hai dataset bằng `id`.
- Đọc `transcription` từ VietSpeech.
- Normalize source và target bằng một versioned normalization pipeline.
- Chạy CTC forced alignment trên transcript đã biết, không dùng transcript ASR tự sinh.
- Chạy bilingual word alignment giữa transcript tiếng Việt và bản dịch.
- Chuyển word alignment thành target-token support timestamps.
- Ghi cache theo shard, hỗ trợ resume và atomic finalize.
- Ghi lỗi per-sample thay vì dừng toàn bộ job.
- Không tải aligner hoặc model CTC trong training worker.

### 4.2. CTC source alignment

Reference implementation dùng `AutoModelForCTC` và dynamic-programming forced alignment trên emission log-probabilities.

Các bước:

1. Chuyển transcript normalized sang CTC vocabulary.
2. Nếu có ký tự OOV, thử normalization bỏ punctuation/variant Unicode.
3. Chạy CTC trellis với blank transitions.
4. Backtrack best path.
5. Gộp character spans thành word spans.
6. Chuyển frame index sang millisecond từ emission stride thực tế.
7. Lưu confidence trung bình cho từng word.

Một source alignment hợp lệ khi:

- Ít nhất 90% source words có timestamp.
- Word end timestamps monotonic.
- Timestamp cuối không vượt audio duration quá một frame tolerance.
- Mean aligned-token confidence vượt configurable threshold.

Nếu không đạt, toàn sample/task dùng `coverage_ratio`; không trộn nửa aligned, nửa ratio trong cùng sample.

Model CTC mặc định trên chỉ là reference research và có license non-commercial. CLI phải cho phép thay model; cache manifest phải lưu model revision và license identifier. Không đóng gói CTC weights vào checkpoint AST.

### 4.3. Bilingual alignment

Dùng SimAlign với XLM-R/multilingual contextual embeddings và `itermax` matching. Đây là preprocessing dependency tùy chọn, đặt trong nhóm dependency `alignment`, không thêm vào runtime training core.

Chuyển alignment edge `source_word_i ↔ target_word_j` thành target support time:

```python
required_time(target_word_j) =
    max(end_time(source_word_i) for all aligned source_word_i)
```

Xử lý target word không được align:

- Nếu có aligned word kế tiếp: dùng required time của aligned word kế tiếp.
- Nếu chỉ có aligned word phía trước: dùng required time phía trước.
- Nếu không có edge nào trong câu: fallback toàn sample về ratio.
- Sau đó áp dụng cumulative maximum từ trái sang phải để tạo monotonic target-prefix boundary.
- Mọi subword thuộc cùng target word dùng cùng required time.
- BOS có required time `0`.
- EOS có required time bằng audio duration.

Điều này cho phép reordering nhưng vẫn bảo đảm output là một prefix liên tục.

### 4.4. Cache schema

Mỗi record cache:

```json
{
  "schema_version": 1,
  "sample_id": "...",
  "task": "vi_en",
  "sample_rate": 16000,
  "audio_num_samples": 123456,
  "source_normalized": "...",
  "target_normalized": "...",
  "source_words": ["..."],
  "source_word_end_ms": [320.0],
  "target_token_ids": [0, 123, 456, 2],
  "target_token_support_ms": [0.0, 640.0, 1280.0, 7716.0],
  "source_alignment_coverage": 0.97,
  "target_alignment_coverage": 0.84,
  "alignment_source": "ctc_bilingual",
  "ctc_model": "...",
  "ctc_revision": "...",
  "bilingual_aligner": "simalign-xlmr-itermax",
  "normalizer_version": 1
}
```

Manifest cấp dataset phải chứa:

- Source/translation dataset IDs và revisions.
- Tokenizer ID, revision và vocab hash.
- CTC/aligner identity.
- Chunk parameters.
- Record count, success count, fallback count và error histogram.

Training phải fail-fast nếu tokenizer hash hoặc normalization version không khớp cache. Thiếu record riêng lẻ được fallback nếu `alignment_required=false`; nếu `true` thì raise error.

---

## 5. Precomputed audio v2

### 5.1. Không hard-code frame rate

Bỏ giả định `100 frames/s`.

Moonshine Streaming có frontend khoảng 50 Hz; frame timing phải lấy từ preprocessing metadata hoặc đo từ model output/input length. Không được silently default frame rate khi chạy streaming trên precomputed embedding.

### 5.2. Precompute đúng theo chunk

Format v2 encode từng chunk bằng cùng chunker dùng ở raw training:

```text
raw chunk
    → Moonshine encoder
    → save [chunk_frames, audio_hidden]
```

Lưu kèm:

- `chunk_start_samples`
- `chunk_valid_samples`
- per-chunk output frame counts
- encoder model revision
- chunk duration/overlap
- sample rate
- schema version

Không encode toàn utterance rồi cắt embedding thành các đoạn giả định 100 Hz.

### 5.3. Tương thích format cũ

- Offline training tiếp tục đọc precomputed v1 như hiện tại.
- Streaming mode ưu tiên v2.
- Streaming mode có thể đọc v1 chỉ khi metadata có frame stride hợp lệ; log warning rằng boundary representation không hoàn toàn tương đương raw per-chunk encoding.
- Thiếu frame metadata trong v1 phải raise error rõ ràng, không dùng giá trị đoán.
- Fix collator để tensor precomputed `[B,C,F,D]` được pad đúng.

### 5.4. Hub IterableDataset

Không bọc `IterableDataset` bằng map-style `StreamingAugmentedDataset`.

Dùng shared `StreamingSampleBuilder` trong cả:

- raw map-style dataset,
- local precomputed map-style dataset,
- Hub precomputed iterable dataset.

Với Hub stream:

- Validation lấy deterministic first `N` records.
- Training dùng stream `.skip(N)` trước khi shuffle để tránh train/validation overlap.
- RAM fallback không được return sớm trước khi streaming augmentation/collator được chọn.
- Dataset phải khai báo capability `returns_streaming_samples=True` để `load_data` không wrap lần hai.

---

## 6. Model refactor

### 6.1. Một audio path duy nhất cho streaming

Thêm API nội bộ:

```python
StreamingAudioEncoding encode_streaming_chunks(
    audio_chunks,
    audio_chunk_mask,
    chunk_start_samples,
    chunk_valid_samples,
    *,
    audio_kind: Literal["raw", "precomputed"],
    audio_frame_mask=None,
)
```

Output:

```python
StreamingAudioEncoding(
    hidden_states: Tensor[B, A, D],
    attention_mask: BoolTensor[B, A],
    last_chunk_hidden: Tensor[B, Q, D],
    chunk_token_offsets: LongTensor[B, C],
)
```

API này là nguồn sự thật duy nhất cho:

- Streaming trainer.
- Streaming validation.
- Future streaming inference.

Trong streaming mode:

- Không dùng `audio_projector`.
- Raw và precomputed đều đi qua `audio_adapter → resampler → global time encoding`.
- `audio_projector` được giữ nguyên cho offline mode.
- Unit test phải chứng minh streaming trainer và model forward gọi cùng API.

### 6.2. Global audio-time encoding

Sau resampler, thêm deterministic sinusoidal time encoding cho từng query output.

Time coordinate:

```python
time_seconds =
    chunk_start_samples / sample_rate
    + query_fraction * chunk_valid_samples / sample_rate
```

Lý do đặt sau resampler:

- Position bên trong adapter có thể bị resampler nén.
- Query index chỉ biểu diễn vị trí tương đối trong chunk.
- Global time encoding gắn acoustic content với vị trí utterance, làm model phân biệt được chunk bị đảo.

Không dùng learned chunk-index embedding hữu hạn.

Acceptance:

- Đảo acoustic content giữa hai chunk trong khi giữ timestamp cố định phải thay đổi logits.
- Hoán vị padding-only chunks không làm đổi logits.
- Hai utterance có cùng chunk content nhưng timestamp khác phải có conditioning khác.

### 6.3. Text backbone

Chốt cho v1:

- Dùng RoPE ở tất cả self-attention layers.
- `num_ergodic_layers` chỉ còn điều khiển window profile và placement của cross-attention; không có nghĩa là text lower layers mất position.
- Giữ sliding-window sizes hiện tại làm defaults.
- Mọi cached forward phải nhận explicit `position_offset`.
- Cache state phải lưu absolute next-text-position, không suy từ cache tensor length vì cache có eviction.
- Cache KV của layer có cross-attention chỉ hợp lệ khi được tạo với cùng audio context; không tái tạo frozen KV bằng `audio_hidden=None`.

Khôi phục XLM-R MLM head đầy đủ:

```text
dense → GELU → LayerNorm → tied decoder + bias
```

Không chỉ dùng một tied linear layer, để giữ initialization semantics của pretrained XLM-R.

### 6.4. Model output

Streaming backbone trả structured output:

```python
StreamingBackboneOutput(
    logits,
    hidden_states,
    new_kv_caches,
)
```

`hidden_states` được dùng cho calibration/commit head sau này; offline output interface không thay đổi.

### 6.5. Vocabulary resize

`resize_token_embeddings` phải đồng bộ:

- Backbone word embeddings.
- Full MLM decoder và bias.
- Streaming length predictor context embeddings.
- `config.vocab_size`.
- Saved streaming config.

New task/rainbow-pad token rows dùng model initializer nhất quán. Test index toàn bộ tokenizer vocabulary qua mọi embedding table.

---

## 7. Training objective

### 7.1. Diffusion loss

Với active suffix:

```python
mask_indices = sampled_diffusion_mask & active_token_mask
ce = cross_entropy(logits, target_ids, reduction="none")
diffusion_loss = sum(ce * mask_indices * timestep_weight) / sum(
    mask_indices * timestep_weight
)
```

Không tính CE trên:

- task token,
- BOS,
- frozen prefix,
- padding,
- unsupported future.

Nếu một sample không có active token, sample đó chỉ đóng góp length loss; không tạo denominator giả bằng `clamp(1)` rồi silently cho zero-loss sample cùng trọng số.

### 7.2. Full-sample mixture

Sampling mặc định:

- 80% random streaming prefix state.
- 20% full utterance state.

Random prefix state:

- Chọn current chunk uniform trong `[1,N]`.
- Chọn previous chunk uniform trong `[0,current-1]`.
- Nếu current prefix không thêm target token, vẫn giữ sample để length predictor học class `0`; diffusion loss của sample bằng zero.

Full state:

- Previous cutoff vẫn có thể nhỏ hơn final cutoff để active suffix không luôn là toàn câu.
- EOS chỉ được train trong full state.

Không dùng curriculum callback trong v1. Static mixture tránh worker-state bug với persistent DataLoader workers và luôn giữ cả partial/full behavior trong suốt training. `curriculum_training` được đánh dấu deprecated trong streaming config và log warning nếu bật.

### 7.3. Confidence calibration

Thay loss hiện tại dùng `logits.detach()` bằng multiclass Brier loss trên masked active positions:

```python
calibration_loss = mean(sum((softmax(logits) - one_hot(target)) ** 2))
```

Defaults:

```json
"streaming_calibration_loss": "brier",
"streaming_calibration_weight": 0.05
```

Không dùng unsupported region làm negative calibration label vì unsupported positions không còn nằm trong canvas.

Validation báo:

- Expected Calibration Error.
- Brier score.
- Accuracy tại các confidence bins.
- Coverage/accuracy nếu áp threshold `0.80`, `0.90`, `0.92`, `0.95`.

Threshold freeze cuối cùng chưa được chọn trong v1 vì stateful inference ngoài scope.

### 7.4. Length predictor

Label chính xác:

```python
new_token_count =
    supported_target_len(current_cutoff)
    - supported_target_len(previous_cutoff)
```

Bao gồm class `0`; không clamp minimum thành `1`.

Input:

- `last_chunk_hidden`
- frozen target context tối đa 32 token
- explicit `context_attention_mask`

Quy tắc:

- Pad context bằng `pad_token_id`, không dùng MASK làm padding.
- Transformer context encoder phải nhận padding mask.
- Target clamp ở `max_chunk_tokens`; preprocessing thống kê overflow rate.
- Nếu overflow >0.1%, tăng config trước full training thay vì silently mất label.
- Báo exact accuracy, MAE, under-generation rate và over-generation rate.

### 7.5. Loss tổng

```python
loss = (
    diffusion_loss
    + streaming_calibration_weight * calibration_loss
    + streaming_length_loss_weight * length_loss
)
```

Defaults:

```json
{
  "streaming_calibration_weight": 0.05,
  "streaming_length_loss_weight": 0.20,
  "streaming_full_sample_probability": 0.20,
  "max_chunk_tokens": 32
}
```

Log từng component và gradient norm riêng cho:

- audio adapter,
- resampler,
- text backbone,
- length predictor.

Frozen Moonshine phải luôn có zero gradients.

---

## 8. Validation và checkpoint selection

### 8.1. Hai validation suite

Offline suite được giữ nguyên để phát hiện regression, nhưng không chọn best streaming checkpoint.

Streaming-prefix suite dùng deterministic cutoff:

- 25% audio coverage.
- 50% audio coverage.
- 75% audio coverage.
- 100% audio coverage.

Với utterance ít chunk, deduplicate cutoff trùng nhau.

Mỗi cutoff dùng target prefix từ cùng alignment cache như training nhưng deterministic và không random previous cutoff.

### 8.2. Metrics

Bắt buộc:

- `streaming_prefix_loss`
- `streaming_prefix_token_accuracy`
- `streaming_prefix_bleu`
- `streaming_length_accuracy`
- `streaming_length_mae`
- `streaming_brier`
- `streaming_ece`
- `alignment_fallback_rate`

Báo metric riêng theo coverage bucket và aggregate weighted theo số active target tokens.

Best checkpoint:

```json
{
  "metric_for_best_model": "streaming_prefix_bleu",
  "greater_is_better": true
}
```

`oracle_length=true` không được dùng cho metric chọn checkpoint. Có thể báo thêm oracle-prefix BLEU dưới tên riêng để tách chất lượng translation khỏi length predictor.

### 8.3. Checkpoint contract

Checkpoint streaming phải lưu:

- `streaming_architecture_version=2`
- toàn bộ streaming config,
- backbone,
- audio adapter,
- resampler,
- global audio-time encoder nếu có parameter,
- length predictor,
- tokenizer và task tokens,
- alignment/cache manifest hash dùng cho run.

Resume test phải kiểm tra:

- optimizer/scheduler/global step được phục hồi,
- tokenizer size khớp,
- cùng fixed batch cho logits/loss giống nhau trong eval mode,
- offline checkpoint loading không bị ảnh hưởng.

---

## 9. Cấu hình canonical

Cập nhật `configs/streaming_vi_multitask.json` thành Vi→En acceptance config:

```json
{
  "task_tokens": ["<vi_en>"],
  "enable_streaming_architecture": true,
  "streaming_architecture_version": 2,

  "audio_chunk_duration": 2.0,
  "audio_overlap_duration": 0.5,
  "streaming_full_sample_probability": 0.20,

  "alignment_cache_path": "data/streaming-alignments-v1",
  "alignment_required": false,
  "streaming_alignment_mode": "aligned_or_coverage",

  "text_rope_all_layers": true,
  "max_chunk_tokens": 32,

  "streaming_calibration_loss": "brier",
  "streaming_calibration_weight": 0.05,
  "streaming_length_loss_weight": 0.20,

  "metric_for_best_model": "streaming_prefix_bleu",
  "greater_is_better": true,
  "oracle_length": false,

  "curriculum_training": false
}
```

Loại khỏi canonical streaming config:

- `loss_weight_unsupported`
- phụ thuộc vào offline oracle length
- mô tả curriculum như yêu cầu bắt buộc

Các field cũ vẫn được parser nhận để đọc config cũ, nhưng streaming v2 log deprecation warning và không sử dụng.

---

## 10. Thứ tự triển khai và quality gates

### Phase 0 — Characterization tests

- Snapshot behavior của offline dataset/model/generator.
- Thêm failing tests tái hiện:
  - audio tail bị bỏ,
  - precomputed collator sai rank,
  - missing task token,
  - train/eval adapter mismatch,
  - chunk-order invariance,
  - calibration zero gradient,
  - BLEU direction sai.
- Không thay kiến trúc trước khi các regression tests offline được cố định.

Gate: test offline hiện tại pass; failing streaming tests phản ánh đúng lỗi đã audit.

### Phase 1 — Data contract và chunking

- Tạo shared exact chunker.
- Chuẩn hóa streaming sample/batch contract.
- Fix raw, local precomputed và Hub iterable paths.
- Bảo đảm task token được truyền explicit.
- Tách train/validation Hub stream không overlap.

Gate: toàn bộ boundary cases và collator shape tests pass.

### Phase 2 — Alignment cache

- Implement CTC forced alignment provider.
- Implement SimAlign bilingual provider.
- Implement target token timestamp projection.
- Implement sharded cache, manifest, resume và fallback.
- Chạy audit thủ công 200 Vi→En samples gồm câu ngắn, dài, số, punctuation và word reordering.

Gate:

- ≥90% source-word timestamp coverage trên accepted records.
- ≥70% target-word alignment coverage hoặc sample fallback.
- Không có target support timestamp giảm dần.
- Fallback rate được báo; không đặt hard quality claim nếu fallback >10%.

### Phase 3 — Unified audio/model path

- Implement `encode_streaming_chunks`.
- Thêm global audio-time encoding.
- Dùng RoPE all text layers.
- Restore XLM-R MLM head.
- Fix vocabulary resize và structured outputs.

Gate:

- Nonzero gradients ở adapter, resampler và backbone.
- Zero gradients ở Moonshine.
- Swapped acoustic chunks làm đổi logits.
- Raw/precomputed v2 cùng sample tạo representation gần nhau trong tolerance đã định.
- Save/reload giữ nguyên fixed-batch output.

### Phase 4 — Prefix-to-prefix objective

- Thay full-future canvas bằng frozen/active target prefix.
- Sửa length target thành per-chunk delta.
- Thêm context padding mask.
- Thay detached confidence penalty bằng Brier loss.
- Bỏ curriculum khỏi canonical path.

Gate:

- Tiny-set overfit 32 samples đạt ≥90% masked active-token accuracy.
- Length predictor overfit đạt MAE ≤0.25 token trên tiny set.
- Calibration loss có nonzero gradient.
- Không có NaN/Inf ở mọi timestep biên.

### Phase 5 — Streaming-prefix validation

- Deterministic coverage buckets.
- Prefix BLEU, length và calibration metrics.
- Best-checkpoint direction đúng.
- Offline validation chạy song song như regression signal.

Gate: cùng checkpoint, streaming validation deterministic qua hai lần chạy với cùng seed.

### Phase 6 — Pilot training và ablation

Chạy theo thứ tự:

1. 500-step smoke run trên subset.
2. 5.000-step Vi→En pilot.
3. Chỉ sau khi đạt gate mới cho phép 50.000-step run.

Ablations bắt buộc:

- Corrected ratio fallback vs CTC+bilingual alignment.
- Có/không global audio-time encoding.
- Audio đúng vs audio shuffled vs zero audio.
- Full-sample probability `0.20` vs `0`.
- Brier calibration on/off.
- Exact length labels vs average tokens/chunk cũ.

Full run chỉ được duyệt nếu:

- Loss ổn định, không NaN.
- Aligned-prefix pilot không kém corrected-ratio baseline trên aggregate prefix BLEU.
- Correct audio tốt hơn shuffled/zero audio có ý nghĩa thực nghiệm.
- Predicted-length prefix BLEU được báo riêng với oracle-prefix BLEU.
- Offline full-audio quality không sụt nghiêm trọng so với pilot baseline.
- Checkpoint resume đã được thử qua ít nhất một save boundary.

---

## 11. Test matrix

### Unit tests

- Chunk lengths: `0`, `<chunk`, `==chunk`, `chunk+1`, `chunk+hop-1`, `chunk+hop`, multiple chunks.
- Coverage monotonic và final coverage bằng `1`.
- EOS chỉ xuất hiện ở final prefix.
- Alignment projection cho one-to-one, many-to-one, one-to-many, unaligned words và reordered words.
- Tokenizer subword mapping giữ đúng target token IDs.
- Raw/precomputed collator shapes và masks.
- Task tokens cho `<vi_en>`, `<vi_zh>`, `<vi_ko>`.
- Length class `0` và max-token boundary.
- Brier loss gradient.
- Vocabulary resize trên mọi embedding/head.

### Model tests

- Sliding-window mask với padding.
- RoPE offset khi có cache.
- KV cache absolute-position bookkeeping sau eviction.
- Cross-attention cache được tạo với audio context.
- Audio permutation sensitivity.
- Padded chunk invariance.
- Streaming path không gọi `audio_projector`.
- Offline path vẫn gọi module offline đúng như trước.

### Integration tests

- Raw sample → collator → streaming loss → backward.
- Precomputed v2 sample → cùng flow.
- Hub iterable sample không bị map-style indexing.
- Alignment cache hit và fallback.
- Save → reload → same eval output.
- Resume trainer giữ global step.
- Deterministic streaming validation.
- Offline config smoke test không khởi tạo streaming modules.

### Training tests

- Overfit 32 samples.
- 500-step smoke không NaN.
- Gradient report xác nhận audio path được học.
- Shuffled/zero-audio ablation.
- Alignment fallback statistics xuất hiện trong logs.
- Best checkpoint thực sự có BLEU cao nhất, không phải thấp nhất.

---

## 12. Ngoài phạm vi v1

Không triển khai trong kế hoạch này:

- Tích hợp `StreamingDiffusionEngine` vào public inference pipeline.
- Confidence-based irreversible token freezing.
- Microphone demo.
- Latency/real-time-factor benchmark.
- Stateful ONNX export.
- Qualcomm AI Hub conversion.
- Galaxy S25 deployment.
- Chọn production freeze threshold.
- Vi→Zh và Vi→Ko full training.

Tuy nhiên model/data interfaces phải được thiết kế để phase inference sau dùng lại trực tiếp, đặc biệt là:

- `encode_streaming_chunks`
- global audio-time positions
- task token contract
- per-chunk length predictor
- absolute text position/cache metadata

---

## 13. Giả định và quyết định đã khóa

- Giữ nguyên toàn bộ offline mode và checkpoint path hiện có.
- Streaming v1 là mode riêng với `streaming_architecture_version=2`.
- Vi→En là acceptance language đầu tiên; code không hard-code tiếng Anh.
- Alignment là optional metadata với deterministic coverage fallback.
- CTC và bilingual alignment chỉ chạy offline.
- Bilingual mapping dùng SimAlign/XLM-R `itermax`.
- CTC reference model phải thay được qua CLI; model mặc định hiện có giới hạn non-commercial và không được coi là lựa chọn deployment.
- Chunk mặc định giữ 2 giây, overlap 0,5 giây, hop 1,5 giây.
- Moonshine tiếp tục frozen.
- RoPE dùng ở tất cả text layers.
- Unsupported target future không xuất hiện trong streaming training canvas.
- Không dùng curriculum trong v1.
- Không dùng oracle length để chọn checkpoint.
- Mọi command Python/dependency workflow dùng `uv`.
- Runtime CUDA verification thực hiện trên Linux/CUDA environment phù hợp; máy macOS ARM hiện tại chỉ dùng được cho static/unit tests không cần CUDA.

Nền tảng phương pháp phù hợp với prefix-to-prefix/wait-k đã được chứng minh trong [STACL](https://aclanthology.org/P19-1289/). Moonshine Streaming dùng frontend khoảng 50 Hz và bounded lookahead, vì vậy metadata frame timing phải dựa trên model/config thay vì hard-code 100 Hz ([model card](https://huggingface.co/UsefulSensors/moonshine-streaming-small)). SimAlign được chọn làm bilingual alignment provider vì chạy trực tiếp trên multilingual representations và không cần huấn luyện parallel aligner riêng ([project](https://github.com/cisnlp/simalign)).
