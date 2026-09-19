# TODO — Hoàn thiện Streaming Training cho Discrete-Diffusion AST

---

## Phase 0 — Characterization Tests

### Snapshot offline behavior
Tạo snapshot tests cho offline dataset, model forward và generator để phát hiện regression khi refactor streaming. Hiện tại test folder chỉ có `test_deep_fusion.py`, `test_hf_push_callback.py`, `test_metric_aggregation.py` — chưa có bất kỳ test nào cho streaming path.

### Test audio tail bị bỏ
Chunking hiện tại trong `streaming_augmented.py` dùng `num_chunks = max(1, (total_samples - overlap_samples) // hop_samples)` — đây là phép chia nguyên, bỏ phần dư. Audio dài hơn chunk cuối một sample vẫn không tạo chunk mới. Cần failing test tái hiện bug này.

### Test precomputed collator sai rank
`StreamingCollator` giả định `audio_chunks` luôn là 2D `[num_chunks, chunk_len]`. Với precomputed embeddings dạng `[num_chunks, frames, hidden]` (3D), collator sẽ pad sai vì chỉ đọc `shape[1]` làm `chunk_len`. Cần failing test.

### Test missing task token
`StreamingCollator` trả `task_token_ids = None` khi sample đầu tiên không có `task_token_id`. Trainer dựa vào `task_token_ids is not None` để prepend — nếu None thì task token bị mất hoàn toàn. Cần failing test bảo đảm task token luôn được truyền explicit.

### Test train/eval audio path mismatch
Streaming trainer `compute_loss` encode audio qua `audio_adapter → resampler`. Nhưng `get_eval_dataloader` dùng `DiscreteDiffusionDataCollator` cho evaluation, rồi delegate lên `DiscreteDiffusionTrainer.compute_loss` — eval path đi qua `audio_projector` thay vì `audio_adapter`. Hai path cho ra representation khác nhau. Cần failing test.

### Test chunk-order invariance
Model hiện tại chỉ dùng per-chunk relative position embedding trong `StreamingAudioAdapter`, không có global audio-time encoding. Đảo thứ tự chunk không thay đổi output vì mỗi chunk đều bắt đầu position từ 0. Cần failing test chứng minh model cần phân biệt thứ tự chunk.

### Test calibration zero gradient
Confidence calibration loss trong `StreamingDiffusionTrainer` (line 1379) dùng `logits.detach()` rồi tính `max_probs` — gradient bị cắt hoàn toàn, loss không đóng góp gradient cho model. Cần failing test.

### Test BLEU direction sai
Config `streaming_vi_multitask.json` set `"metric_for_best_model": "bleu"` và `"greater_is_better": false` — nghĩa là chọn checkpoint có BLEU **thấp nhất**. Cần failing test.

---

## Phase 1 — Data Contract và Chunking

### Tạo shared exact chunker
Implement exact chunking formula: `num_chunks = max(1, ceil(max(total_samples - chunk_samples, 0) / hop_samples) + 1)`. Lưu `chunk_start_samples` và `chunk_valid_samples` cho mỗi chunk. Hiện tại `StreamingAugmentedDataset` dùng floor division bỏ phần dư — audio tail bị mất.

### Chuẩn hóa streaming sample contract
Định nghĩa `StreamingTrainingSample` thống nhất cho raw, local precomputed và Hub streaming dataset. Hiện tại `StreamingAugmentedDataset` trả `support_ratio` + `supported_text_len` (full future target), còn PLAN yêu cầu `frozen_prefix_len` + `active_target_len` + `new_token_count` (prefix-only target). Hai format hoàn toàn khác nhau.

### Fix streaming collator cho precomputed data
`StreamingCollator` chỉ xử lý 2D audio chunks `[C, samples]`. Với precomputed `[C, frames, hidden]`, cần pad riêng trục chunk và trục frame. Hiện tại collator tạo `torch.zeros(B, max_chunks, chunk_len)` — sai dimensionality cho precomputed.

### Fix Hub IterableDataset
`StreamingPrecomputedMultiTaskDataset` là `IterableDataset` nhưng `load_data` trong `dd_data.py` có thể wrap nó bằng map-style `StreamingAugmentedDataset` — `IterableDataset` không hỗ trợ `__getitem__`. Cần khai báo `returns_streaming_samples=True` để `load_data` không wrap lần hai.

### Truyền task token explicit trong collator
Collator suy ra `task_token_ids` từ `batch[0].get("task_token_id")` — nếu None thì toàn batch mất task token. Cần luôn trả `task_token_ids` và assert batch không trộn task khác nhau mà không có token.

### Tách train/validation Hub stream
`StreamingPrecomputedMultiTaskDataset.load_data` tạo cả train và val từ cùng một stream không `.skip(N)`. Validation `max_samples=500` lấy 500 record đầu, nhưng train cũng bắt đầu từ đầu — overlap dữ liệu giữa train và validation.

### Giữ dtype audio embedding
`StreamingCollator` tạo `torch.zeros(B, max_chunks, chunk_len)` mặc định `float32`. Nếu config dùng BF16/FP16, embedding bị ép về `float32` rồi lại cast — lãng phí memory và có thể gây precision mismatch.

---

## Phase 2 — Alignment Cache

### Tạo script `build_streaming_alignments.py`
Script preprocessing chưa tồn tại. Cần implement CTC forced alignment + bilingual alignment + target token timestamp projection + sharded cache với resume.

### Implement CTC forced alignment provider
Dùng `AutoModelForCTC` + dynamic-programming forced alignment trên emission log-probabilities. Chuyển character spans thành word spans. Validate ≥90% source words có timestamp, monotonic end timestamps, confidence threshold.

### Implement SimAlign bilingual alignment provider
Dùng SimAlign + XLM-R multilingual embeddings + `itermax` matching. Chuyển alignment edge thành `required_time(target_word_j) = max(end_time(aligned_source_words))`. Xử lý unaligned words bằng forward/backward fill. Áp cumulative maximum cho monotonic boundary.

### Implement target token timestamp projection
Chuyển bilingual word alignment thành per-target-token `support_ms`. Mọi subword cùng word dùng cùng required time. BOS = 0ms, EOS = audio duration.

### Implement cache schema và manifest
Ghi cache JSON theo shard. Manifest lưu dataset IDs, tokenizer hash, CTC/aligner identity, chunk params, record counts. Training fail-fast nếu tokenizer hash không khớp.

### Implement fallback coverage ratio
Khi alignment lỗi hoặc thiếu, fallback dùng `supported_content_tokens = floor(coverage(current_chunk) * number_of_target_content_tokens)`. BOS luôn supported, EOS chỉ khi coverage=1. Kết quả monotonic theo chunk index.

---

## Phase 3 — Unified Audio/Model Path

### Implement `encode_streaming_chunks` API
API nội bộ duy nhất cho streaming audio encoding: `audio_chunks → audio_adapter → resampler → global time encoding → concatenate`. Hiện tại streaming trainer tự inline audio encoding logic (dd_trainer.py line 1249-1296), không dùng API chung với model forward.

### Thêm global audio-time encoding
Sau resampler, thêm deterministic sinusoidal time encoding dùng `time_seconds = chunk_start_samples/sample_rate + query_fraction * chunk_valid_samples/sample_rate`. Hiện tại chỉ có per-chunk relative position trong `StreamingAudioAdapter` — model không phân biệt được chunk thứ tự.

### Dùng RoPE ở tất cả text layers
PLAN yêu cầu `text_rope_all_layers=true`. Code hiện tại đã dùng RoPE ở cả ergodic và position layers (`ergodic_use_rope=True` default). Cần verify và config-ify — PLAN nói "num_ergodic_layers chỉ điều khiển window profile và cross-attention placement, không có nghĩa text lower layers mất position."

### Restore XLM-R MLM head đầy đủ
`StreamingDiffusionBackbone.lm_head` hiện tại chỉ là `nn.Linear(hidden, vocab, bias=False)` tied với word embeddings. XLM-R MLM head đầy đủ là `dense → GELU → LayerNorm → tied decoder + bias`. `from_pretrained_xlmr` chỉ copy decoder weights, bỏ dense và LayerNorm.

### Fix vocabulary resize đồng bộ
`resize_token_embeddings` trong `dd_model.py` (line 428-450) resize backbone `word_embeddings` và `lm_head` nhưng không resize: streaming length predictor `context_embed`, full MLM head (dense, LayerNorm nếu restore), `config.vocab_size`. Thiếu đồng bộ.

### Fix streaming path không dùng `audio_projector`
`dd_model.py` forward (line 550-570) khi có backbone vẫn gọi `audio_projector` cho precomputed/raw audio. PLAN yêu cầu streaming mode không dùng `audio_projector` — chỉ dùng `audio_adapter → resampler → time encoding`.

### Model trả structured output
Streaming backbone trả `(logits, new_kv_caches)` tuple. PLAN yêu cầu `StreamingBackboneOutput(logits, hidden_states, new_kv_caches)` — `hidden_states` cần cho calibration/commit head.

---

## Phase 4 — Prefix-to-Prefix Objective

### Thay full-future canvas bằng frozen/active target prefix
Hiện tại streaming training đưa **toàn bộ** text target vào canvas và dùng `loss_weight_unsupported=0.3` cho unsupported region. PLAN yêu cầu canvas chỉ chứa `[TASK] [BOS] [frozen_tokens] [active_tokens]` — unsupported future không xuất hiện trong canvas. Đây là thay đổi lớn nhất.

### CE chỉ tính trên active positions
Diffusion noise chỉ được áp dụng lên active tokens. CE chỉ tính trên masked active positions. Frozen tokens luôn unmasked, không có token loss. Hiện tại CE tính trên toàn bộ masked positions kể cả unsupported.

### Sửa length target thành per-chunk delta
Length predictor hiện tại dùng `gt_lengths = (N / total_chunks).round()` — trung bình token/chunk. PLAN yêu cầu `new_token_count = supported_target_len(current_cutoff) - supported_target_len(previous_cutoff)` — delta thực tế giữa hai cutoff. Bao gồm class 0.

### Thêm context padding mask cho length predictor
`StreamingLengthPredictor` (line 88-90) không nhận padding mask cho context. `context_pool` (TransformerEncoder) xử lý padded context không đúng — padding tokens tham gia attention. Trainer pad context bằng `mask_id` (line 1408) thay vì `pad_token_id`.

### Thay detached confidence penalty bằng Brier loss
Calibration loss hiện tại dùng `logits.detach()` → zero gradient. Thay bằng multiclass Brier loss: `mean(sum((softmax(logits) - one_hot(target))²))` trên masked active positions. Gradient phải chạy qua logits.

### Bỏ curriculum khỏi canonical path
Config có `curriculum_training: false` nhưng `StreamingAugmentedDataset` vẫn nhận `curriculum_step` param. Đánh dấu deprecated, log warning nếu bật. Dùng static 80/20 mixture (streaming prefix / full utterance).

### Implement full-sample mixture 80/20
Hiện tại `StreamingAugmentedDataset` random `num_visible` từ 1→N uniform. PLAN yêu cầu 80% streaming prefix state (random current chunk, random previous chunk) và 20% full utterance state. Full state vẫn có active suffix.

### Xử lý sample không có active token
Nếu current prefix không thêm target token, sample chỉ đóng góp length loss (class 0). Hiện tại code dùng `clamp(min=1)` cho denominator — zero-loss sample có cùng trọng số.

---

## Phase 5 — Streaming-prefix Validation

### Implement deterministic streaming validation suite
Validation streaming dùng deterministic cutoff: 25%, 50%, 75%, 100% audio coverage. Hiện tại `get_eval_dataloader` dùng `DiscreteDiffusionDataCollator` — chạy eval trên full-audio/offline path, không phải streaming prefix.

### Implement streaming validation metrics
Cần metric riêng: `streaming_prefix_loss`, `streaming_prefix_token_accuracy`, `streaming_prefix_bleu`, `streaming_length_accuracy`, `streaming_length_mae`, `streaming_brier`, `streaming_ece`, `alignment_fallback_rate`. Hiện tại không có metric streaming nào.

### Fix best checkpoint selection direction
Config hiện tại: `"metric_for_best_model": "bleu", "greater_is_better": false` → chọn checkpoint BLEU thấp nhất. PLAN yêu cầu `"metric_for_best_model": "streaming_prefix_bleu", "greater_is_better": true`.

### Giữ offline validation song song
Offline validation suite giữ nguyên để detect regression, nhưng không dùng để chọn best streaming checkpoint.

---

## Phase 6 — Pilot Training và Ablation

### 500-step smoke run
Chạy 500 steps trên subset, verify không NaN, gradient report xác nhận audio path được học.

### 5.000-step Vi→En pilot
Chạy pilot trên Vi→En, so sánh corrected ratio fallback vs CTC+bilingual alignment.

### Ablation studies bắt buộc
- Corrected ratio fallback vs CTC+bilingual alignment
- Có/không global audio-time encoding
- Audio đúng vs shuffled vs zero audio
- Full-sample probability 0.20 vs 0
- Brier calibration on/off
- Exact length labels vs average tokens/chunk cũ

### 50.000-step full run
Chỉ được duyệt sau khi tất cả pilot gates đạt.

---

## Precomputed Audio v2

### Bỏ hard-code frame rate 100 Hz
`StreamingAugmentedDataset` hard-code `self.frames_per_sec = 100`. Moonshine Streaming có frontend ~50 Hz. Frame timing phải lấy từ preprocessing metadata hoặc đo từ model output. Hiện tại silently dùng sai frame rate cho precomputed embeddings.

### Precompute đúng theo chunk
Format v2 phải encode từng chunk bằng cùng chunker dùng ở raw training. Lưu kèm `chunk_start_samples`, `chunk_valid_samples`, per-chunk frame counts, encoder model revision. Hiện tại precompute encode toàn utterance rồi cắt embedding — boundary representation khác raw per-chunk encoding.

### Tương thích format cũ
Offline training tiếp tục đọc precomputed v1. Streaming mode ưu tiên v2. V1 chỉ đọc được khi metadata có frame stride hợp lệ — thiếu metadata phải raise error rõ ràng, không dùng giá trị đoán.

---

## Cấu hình

### Cập nhật `streaming_vi_multitask.json`
Cần thêm: `streaming_architecture_version: 2`, `alignment_cache_path`, `alignment_required`, `streaming_alignment_mode`, `text_rope_all_layers`, `streaming_calibration_loss: "brier"`, `streaming_calibration_weight: 0.05`, `streaming_length_loss_weight: 0.20`, `streaming_full_sample_probability: 0.20`, `oracle_length: false`.

Cần bỏ/deprecate: `loss_weight_unsupported`, `oracle_length: true`, `greater_is_better: false` (sai direction).

### Deprecated fields backward compat
Các field cũ (`loss_weight_unsupported`, `curriculum_training`, `use_confidence_calibration`) vẫn được parser nhận nhưng streaming v2 log deprecation warning và không sử dụng.

---

## Checkpoint

### Checkpoint streaming contract
Checkpoint phải lưu: `streaming_architecture_version=2`, streaming config, backbone, audio adapter, resampler, global audio-time encoder, length predictor, tokenizer + task tokens, alignment/cache manifest hash.

### Resume test
Verify: optimizer/scheduler/global step phục hồi, tokenizer size khớp, cùng fixed batch cho logits/loss giống nhau trong eval mode, offline checkpoint loading không bị ảnh hưởng.
