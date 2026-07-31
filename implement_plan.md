# 📋 IMPLEMENTATION PLAN: Streaming Discrete Diffusion Speech Recognition & Translation

## Bản Thiết Kế Chi Tiết — Từ Kiến Trúc Đến Huấn Luyện

---

## MỤC LỤC

```
I.    Tổng Quan Kiến Trúc Mới
II.   Thành Phần 1: Audio Encoder (Giữ nguyên + Tinh chỉnh)
III.  Thành Phần 2: Audio Projector / Adapter (Sửa)
IV.   Thành Phần 3: Text Backbone — XLM-RoBERTa Streaming (Xây mới)
V.    Thành Phần 4: Streaming Diffusion Engine (Xây mới)
VI.   Thành Phần 5: Length Predictor (Sửa)
VII.  Training Pipeline (Xây mới)
VIII. Inference Pipeline (Xây mới)
IX.   Migration Plan từ Codebase Hiện Tại
X.    Checklist & Lưu Ý Quan Trọng
```

---

## I. TỔNG QUAN KIẾN TRÚC MỚI

### 1.1. Sơ Đồ Toàn Hệ Thống

```
🎤 INFINITE AUDIO STREAM
│
│  chunk_1 (2s) → chunk_2 (2s) → chunk_3 (2s) → ...
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: AUDIO ENCODER (Moonshine Streaming Medium)                │
│  ─────────────────────────────────────────────────────────────────  │
│  Trạng thái: GIỮ NGUYÊN (đã streaming sẵn)                         │
│  Sửa đổi:   Không cần sửa                                         │
│  Output:    audio_hidden [B, chunk_frames, D_audio=512]             │
│  Position:  ❌ KHÔNG có (ergodic, translation-invariant)            │
│  Attention: Sliding window (w_left=16, w_right=4) — có sẵn         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: AUDIO PROJECTOR / ADAPTER                                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Trạng thái: SỬA (thêm position embedding)                          │
│  Vai trò:   Cầu nối audio (không vị trí) → text (có vị trí)        │
│  Output:    audio_projected [B, chunk_frames, D_text=768/1024]      │
│  Position:  ✅ THÊM learned position embedding tại đây              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: TEXT BACKBONE — STREAMING XLM-RoBERTa                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Trạng thái: XÂY MỚI (thay thế XLM-RoBERTa gốc)                    │
│                                                                     │
│  Input:  [frozen_context (KV cache)] + [active_tokens] + [audio]    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Phase A: Ergodic Layers (Layer 1 → 6)                      │    │
│  │  • ❌ KHÔNG có position embedding                           │    │
│  │  • Sliding Window Attention (w_left=64, w_right=16)         │    │
│  │  • Xử lý: local syntax, n-gram patterns                    │    │
│  │  • Translation-invariant                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Phase B: Position-Aware Layers (Layer 7 → 12)              │    │
│  │  • ✅ RoPE (Rotary Position Embedding)                      │    │
│  │  • Wider Sliding Window (w_left=128, w_right=32)            │    │
│  │  • Cross-Attention với audio features                       │    │
│  │  • Xử lý: global semantics, word order, translation         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │                                          │
│  Output: logits [B, active_len, vocab_size]                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: STREAMING DIFFUSION ENGINE                                │
│  ─────────────────────────────────────────────────────────────────  │
│  Trạng thái: XÂY MỚI                                               │
│                                                                     │
│  Quản lý 3 vùng:                                                    │
│  ┌────────────┐  ┌──────────────────┐  ┌────────────────────┐      │
│  │ FROZEN     │  │ ACTIVE           │  │ FUTURE             │      │
│  │ (KV Cache) │  │ (Denoise Zone)   │  │ (MASK Placeholder) │      │
│  │ ≤128 tok   │  │ ≤64 tok          │  │ Không compute      │      │
│  └────────────┘  └──────────────────┘  └────────────────────┘      │
│                                                                     │
│  Denoise: 2-3 bước (không phải 10)                                 │
│  Freeze:  confidence > θ → FROZEN → yield output                   │
│  Remask:  confidence < φ → MASK lại                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2. So Sánh Kiến Trúc Cũ vs Mới

| Thành phần | Cũ (Offline) | Mới (Streaming) |
|---|---|---|
| Audio Encoder | Moonshine (full audio) | Moonshine (chunk-by-chunk) — **giữ nguyên** |
| Audio Position | Không rõ ràng | Learned pos embed tại Adapter |
| Text Backbone | XLM-R full attention, absolute pos | Hybrid: Ergodic + RoPE, Sliding Window |
| Text Position | Learned absolute (max 512) | RoPE (infinite) |
| Attention | Full O(N²) | Sliding Window O(N·w) |
| Diffusion Canvas | Fixed-length, toàn cục | 3-zone: Frozen/Active/Future |
| Denoise Steps | 10 bước, toàn bộ canvas | 2-3 bước, chỉ active zone |
| Output | Một lần sau khi xong | Từng token theo thời gian thực |
| Memory | O(N²) — tăng theo độ dài | O(w²) — **cố định** |
| Max Audio Length | Giới hạn bởi position embed | **Vô hạn** |

### 1.3. Các Tham Số Chính (Hyperparameters)

```python
@dataclass
class StreamingDiffusionConfig:
    # === Audio ===
    audio_encoder_name: str = "UsefulSensors/moonshine-streaming-medium"
    audio_chunk_duration: float = 2.0          # giây mỗi chunk
    audio_overlap_duration: float = 0.5        # giây overlap giữa chunks
    audio_hidden_size: int = 512               # Moonshine output dim
    
    # === Adapter ===
    adapter_hidden_size: int = 768             # = text hidden size
    adapter_add_position: bool = True          # thêm learned pos embed
    adapter_max_position: int = 256            # max frames per chunk
    
    # === Text Backbone ===
    backbone: str = "FacebookAI/xlm-roberta-base"
    num_hidden_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    
    # Phase A: Ergodic
    num_ergodic_layers: int = 6                # Layer 1-6
    ergodic_window_left: int = 64              # nhìn 64 token bên trái
    ergodic_window_right: int = 16             # nhìn 16 token bên phải
    ergodic_use_rope: bool = False             # KHÔNG dùng RoPE
    
    # Phase B: Position-Aware
    num_position_layers: int = 6               # Layer 7-12
    position_window_left: int = 128            # nhìn 128 token bên trái
    position_window_right: int = 32            # nhìn 32 token bên phải
    position_use_rope: bool = True             # DÙNG RoPE
    rope_theta: float = 10000.0                # RoPE base frequency
    rope_scaling: str = "none"                 # "none" | "linear" | "ntk"
    
    # === Diffusion ===
    num_diffusion_timesteps: int = 50          # T (cho training)
    streaming_denoise_steps: int = 3           # K (cho inference, ít hơn)
    diffusion_type: str = "absorbing"
    
    # === Streaming Zones ===
    active_window_size: int = 64               # max tokens đang denoise
    frozen_cache_size: int = 128               # max tokens trong KV cache
    freeze_confidence_threshold: float = 0.92  # θ: conf > θ → freeze
    remask_confidence_threshold: float = 0.30  # φ: bottom 30% → remask
    freeze_strategy: str = "monotonic_left"    # freeze từ trái sang phải
    
    # === Training ===
    streaming_augmentation: bool = True
    num_audio_chunks_per_sample: int = 5       # chia audio thành 5 chunks
    loss_weight_supported: float = 2.0         # weight cho token có audio
    loss_weight_unsupported: float = 0.3       # weight cho token chưa có audio
    curriculum_training: bool = True           # tăng dần số chunks
    
    # === Multi-task ===
    task_tokens: list = field(default_factory=lambda: ["<vi_en>", "<vi_zh>", "<vi_ko>"])
```

---

## II. THÀNH PHẦN 1: AUDIO ENCODER

### 2.1. Trạng Thái: GIỮ NGUYÊN

Moonshine Streaming Medium đã được thiết kế cho streaming. Không cần sửa.

```python
# Giữ nguyên như hiện tại
self.audio_encoder = AutoModel.from_pretrained(
    "UsefulSensors/moonshine-streaming-medium"
)
self.audio_encoder.eval()  # Frozen
for param in self.audio_encoder.parameters():
    param.requires_grad = False
```

### 2.2. Cách Gọi Trong Streaming Mode

```python
class StreamingAudioEncoder:
    """
    Wrapper cho Moonshine trong chế độ streaming.
    Xử lý từng chunk, giữ state giữa các chunks.
    """
    def __init__(self, encoder, chunk_duration=2.0, overlap_duration=0.5):
        self.encoder = encoder
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = 16000
        
        self.chunk_samples = int(chunk_duration * self.sample_rate)      # 32000
        self.overlap_samples = int(overlap_duration * self.sample_rate)  # 8000
        self.hop_samples = self.chunk_samples - self.overlap_samples     # 24000
        
        # Buffer cho audio chưa đủ 1 chunk
        self.audio_buffer = torch.tensor([])
        
    def feed(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Nhận audio raw (1D tensor).
        Trả về audio embeddings nếu đủ 1 chunk, ngược lại trả về None.
        """
        self.audio_buffer = torch.cat([self.audio_buffer, audio_chunk])
        
        if len(self.audio_buffer) < self.chunk_samples:
            return None  # Chưa đủ, đợi thêm
        
        # Lấy đúng 1 chunk
        chunk = self.audio_buffer[:self.chunk_samples]
        
        # Giữ lại phần overlap cho chunk tiếp theo
        self.audio_buffer = self.audio_buffer[self.hop_samples:]
        
        # Encode
        with torch.no_grad():
            embeddings = self.encoder(chunk.unsqueeze(0))  # [1, frames, D_audio]
        
        return embeddings.squeeze(0)  # [frames, D_audio]
    
    def flush(self) -> Optional[torch.Tensor]:
        """
        Gọi khi audio kết thúc.
        Xử lý phần còn lại trong buffer (pad nếu cần).
        """
        if len(self.audio_buffer) == 0:
            return None
        
        # Pad phần còn lại cho đủ chunk_size
        padded = F.pad(self.audio_buffer, (0, self.chunk_samples - len(self.audio_buffer)))
        
        with torch.no_grad():
            embeddings = self.encoder(padded.unsqueeze(0))
        
        self.audio_buffer = torch.tensor([])
        return embeddings.squeeze(0)
```

### 2.3. Lưu Ý

| # | Lưu ý | Chi tiết |
|---|---|---|
| 1 | **Overlap là bắt buộc** | Không có overlap → mất thông tin tại biên giới chunk → artifact |
| 2 | **Moonshine đã có sliding window nội bộ** | Window (16,4) frames → mỗi frame chỉ nhìn 20 frame lân cận → đã bounded |
| 3 | **Không cần position cho audio** | Moonshine v2 philosophy: encoder ergodic → không position → adapter thêm position |
| 4 | **Frame rate** | Moonshine output ~50 frames/giây → chunk 2s → ~100 frames → 100 audio tokens |
| 5 | **Frozen encoder** | Không train audio encoder → tiết kiệm memory, tránh catastrophic forgetting |

---

## III. THÀNH PHẦN 2: AUDIO PROJECTOR / ADAPTER

### 3.1. Vai Trò Mới

Trong kiến trúc cũ, Audio Projector chỉ là `nn.Linear(D_audio → D_text)`.

Trong kiến trúc mới, nó đóng vai trò **Adapter** theo triết lý Moonshine v2:
- Nhận audio embeddings **không có vị trí** (ergodic)
- **Thêm position embedding** (learned)
- Chiếu sang không gian text
- Đây là "cầu nối" giữa thế giới không vị trí (audio) và thế giới có vị trí (text)

### 3.2. Kiến Trúc Chi Tiết

```python
class StreamingAudioAdapter(nn.Module):
    """
    Adapter giữa Audio Encoder (ergodic) và Text Backbone (position-aware).
    
    Vai trò:
    1. Chiếu D_audio → D_text
    2. THÊM position embedding (audio tokens giờ có vị trí)
    3. Normalization
    4. (Optional) Downsample nếu audio frames quá nhiều
    """
    
    def __init__(self, config: StreamingDiffusionConfig):
        super().__init__()
        
        D_audio = config.audio_hidden_size       # 512
        D_text = config.adapter_hidden_size      # 768
        max_frames = config.adapter_max_position # 256
        
        # 1. Linear projection
        self.proj = nn.Linear(D_audio, D_text)
        
        # 2. Learned Position Embedding (THÊM MỚI)
        #    Mỗi frame trong chunk có vị trí tương đối (0, 1, 2, ..., max_frames-1)
        self.position_embeddings = nn.Embedding(max_frames, D_text)
        
        # 3. Layer Norm
        self.layer_norm = nn.LayerNorm(D_text)
        
        # 4. (Optional) Conv downsample nếu cần giảm số frames
        #    100 frames/chunk → 50 tokens/chunk (giảm 2x)
        self.downsample = nn.Conv1d(
            in_channels=D_text, 
            out_channels=D_text, 
            kernel_size=2, 
            stride=2
        )
        self.use_downsample = True  # Bật nếu muốn giảm compute
        
        # 5. Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_hidden: [B, num_frames, D_audio] — từ Moonshine, KHÔNG có position
            
        Returns:
            audio_projected: [B, num_tokens, D_text] — CÓ position, sẵn sàng fuse vào text
        """
        B, T, D = audio_hidden.shape
        
        # 1. Project
        x = self.proj(audio_hidden)  # [B, T, D_text]
        
        # 2. Thêm position embedding
        positions = torch.arange(T, device=x.device)  # [0, 1, 2, ..., T-1]
        pos_embed = self.position_embeddings(positions)  # [T, D_text]
        x = x + pos_embed.unsqueeze(0)  # broadcast [B, T, D_text]
        
        # 3. (Optional) Downsample
        if self.use_downsample and T > 1:
            x = x.transpose(1, 2)       # [B, D_text, T]
            x = self.downsample(x)       # [B, D_text, T//2]
            x = x.transpose(1, 2)       # [B, T//2, D_text]
        
        # 4. Norm + Dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x  # [B, num_tokens, D_text]
```

### 3.3. Lưu Ý

| # | Lưu ý | Chi tiết |
|---|---|---|
| 1 | **Position embedding là RELATIVE trong chunk** | Vị trí 0-99 cho mỗi chunk, không phải vị trí tuyệt đối toàn bài |
| 2 | **Mỗi chunk reset position về 0** | Chunk 1: pos [0..99], Chunk 2: pos [0..99] → không overflow |
| 3 | **Downsample giảm compute đáng kể** | 100 frames → 50 tokens → giảm 2x attention cost |
| 4 | **Adapter là phần TRAINABLE** | Audio encoder frozen, text backbone có thể frozen/LoRA → adapter học cách bridge |
| 5 | **Khởi tạo nhỏ** | `nn.init.xavier_uniform_(self.proj.weight, gain=0.01)` → ban đầu không phá vỡ pretrained text backbone |

---

## IV. THÀNH PHẦN 3: TEXT BACKBONE — STREAMING XLM-RoBERTa

### 4.1. Đây Là Phần Phức Tạp Nhất

Cần xây dựng lại XLM-RoBERTa với 3 thay đổi lớn:
1. Bỏ absolute position embedding
2. Chia thành 2 phase: Ergodic (không vị trí) + Position-Aware (RoPE)
3. Thêm Sliding Window Attention

### 4.2. RoPE Implementation

```python
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Áp dụng lên Query và Key trong attention.
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency bands
        # freq_i = 1 / (theta ^ (2i / dim))  với i = 0, 1, ..., dim/2 - 1
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cho tất cả vị trí
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos())   # [seq_len, dim]
        self.register_buffer("sin_cached", emb.sin())   # [seq_len, dim]
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """
        Args:
            x: [B, num_heads, seq_len, head_dim]
            positions: [seq_len] — vị trí của mỗi token (có thể không liên tục)
        Returns:
            cos, sin: [1, 1, seq_len, head_dim]
        """
        cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Áp dụng rotation lên Q và K.
    
    q, k: [B, H, S, D]
    cos, sin: [1, 1, S, D]
    """
    # Xoay nửa đầu và nửa sau
    q_rot = q[..., :cos.shape[-1]//2]
    q_pass = q[..., cos.shape[-1]//2:]
    k_rot = k[..., :cos.shape[-1]//2]
    k_pass = k[..., cos.shape[-1]//2:]
    
    # Apply rotation: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    q_rotated = torch.cat([
        q_rot * cos[..., :cos.shape[-1]//2] - q_pass * sin[..., :sin.shape[-1]//2],
        q_rot * sin[..., :sin.shape[-1]//2] + q_pass * cos[..., :cos.shape[-1]//2],
    ], dim=-1)
    
    k_rotated = torch.cat([
        k_rot * cos[..., :cos.shape[-1]//2] - k_pass * sin[..., :sin.shape[-1]//2],
        k_rot * sin[..., :sin.shape[-1]//2] + k_pass * cos[..., :cos.shape[-1]//2],
    ], dim=-1)
    
    return q_rotated, k_rotated
```

### 4.3. Sliding Window Attention

```python
class SlidingWindowAttention(nn.Module):
    """
    Multi-Head Attention với Sliding Window.
    Hỗ trợ: RoPE (optional), KV Cache (optional), Cross-Attention (optional).
    """
    
    def __init__(self, config, layer_idx: int, use_rope: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        
        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # RoPE (chỉ khởi tạo nếu cần)
        if use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.active_window_size + config.frozen_cache_size + 64,
                theta=config.rope_theta,
            )
        
        # Sliding window sizes
        if layer_idx < config.num_ergodic_layers:
            self.window_left = config.ergodic_window_left    # 64
            self.window_right = config.ergodic_window_right  # 16
        else:
            self.window_left = config.position_window_left    # 128
            self.window_right = config.position_window_right  # 32
    
    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Tạo attention mask dạng sliding window.
        
        Returns: [seq_len, seq_len] — True = được attend, False = bị mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            left = max(0, i - self.window_left)
            right = min(seq_len, i + self.window_right + 1)
            mask[i, left:right] = True
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, S, D] — active tokens
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[Tuple] = None,  # (K_cache, V_cache) từ frozen tokens
        audio_hidden: Optional[torch.Tensor] = None,  # [B, A, D] — cho cross-attn
    ) -> Tuple[torch.Tensor, Tuple]:
        
        B, S, D = hidden_states.shape
        
        # 1. Compute Q, K, V
        Q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, H, S, head_dim]
        
        # 2. Áp dụng RoPE (nếu có)
        if self.use_rope:
            positions = torch.arange(S, device=hidden_states.device)
            cos, sin = self.rotary_emb(Q, positions)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # 3. Ghép với KV cache (frozen context)
        if past_kv_cache is not None:
            K_cache, V_cache = past_kv_cache  # [B, H, cache_len, head_dim]
            K_full = torch.cat([K_cache, K], dim=2)  # [B, H, cache_len + S, head_dim]
            V_full = torch.cat([V_cache, V], dim=2)
        else:
            K_full = K
            V_full = V
        
        # 4. Sliding Window Mask
        total_len = K_full.shape[2]
        sw_mask = self._create_sliding_window_mask(S, hidden_states.device)
        # Mở rộng mask cho phần cache (cache luôn được attend)
        if past_kv_cache is not None:
            cache_len = K_cache.shape[2]
            cache_mask = torch.ones(S, cache_len, dtype=torch.bool, device=hidden_states.device)
            sw_mask = torch.cat([cache_mask, sw_mask], dim=1)  # [S, cache_len + S]
        
        # 5. Scaled Dot-Product Attention
        # Sử dụng PyTorch 2.0+ flash attention
        attn_output = F.scaled_dot_product_attention(
            Q, K_full, V_full,
            attn_mask=sw_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, S, total_len]
            dropout_p=0.1 if self.training else 0.0,
        )
        # attn_output: [B, H, S, head_dim]
        
        # 6. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.o_proj(attn_output)
        
        # 7. Trả về KV cache mới (chỉ của active tokens, để freeze sau này)
        new_kv_cache = (K, V)  # [B, H, S, head_dim]
        
        return output, new_kv_cache
```

### 4.4. Streaming Roberta Layer

```python
class StreamingRobertaLayer(nn.Module):
    """
    Một layer Roberta đã sửa cho streaming.
    Self-Attention (Sliding Window + RoPE) + Cross-Attention (Audio) + FFN
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        
        use_rope = (layer_idx >= config.num_ergodic_layers)
        
        # Self-Attention (Sliding Window)
        self.self_attn = SlidingWindowAttention(config, layer_idx, use_rope=use_rope)
        self.self_attn_norm = nn.LayerNorm(config.hidden_size)
        
        # Cross-Attention với Audio (chỉ ở Phase B)
        self.has_cross_attn = use_rope  # Phase B mới có cross-attn
        if self.has_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_size, config.num_attention_heads, 
                batch_first=True, dropout=0.1
            )
            self.cross_attn_norm = nn.LayerNorm(config.hidden_size)
            # Khởi tạo nhỏ (residual ban đầu ≈ 0)
            nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.01)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv_cache: Optional[Tuple] = None,
        audio_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        
        # 1. Self-Attention + Residual + Norm
        attn_out, new_kv = self.self_attn(hidden_states, past_kv_cache=past_kv_cache)
        hidden_states = self.self_attn_norm(hidden_states + attn_out)
        
        # 2. Cross-Attention với Audio (nếu có)
        if self.has_cross_attn and audio_hidden is not None:
            cross_out, _ = self.cross_attn(
                query=hidden_states,
                key=audio_hidden,
                value=audio_hidden,
            )
            hidden_states = self.cross_attn_norm(hidden_states + cross_out)
        
        # 3. FFN + Residual + Norm
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_out)
        
        return hidden_states, new_kv
```

### 4.5. Full Streaming Text Backbone

```python
class StreamingDiffusionBackbone(nn.Module):
    """
    Thay thế XLM-RoBERTa gốc.
    Hybrid: Ergodic layers (1-6) + Position-Aware layers (7-12).
    """
    
    def __init__(self, config: StreamingDiffusionConfig, tokenizer):
        super().__init__()
        self.config = config
        
        # Word Embedding (giữ từ XLM-R pretrained)
        self.word_embeddings = nn.Embedding(len(tokenizer), config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)  # segment A/B
        self.embed_norm = nn.LayerNorm(config.hidden_size)
        self.embed_dropout = nn.Dropout(0.1)
        
        # ❌ KHÔNG CÓ position_embeddings
        
        # Layers
        self.layers = nn.ModuleList([
            StreamingRobertaLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, len(tokenizer), bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,              # [B, active_len]
        past_kv_caches: Optional[List] = None, # List of (K, V) per layer
        audio_hidden: Optional[torch.Tensor] = None,  # [B, audio_len, D]
    ) -> Tuple[torch.Tensor, List]:
        
        B, S = input_ids.shape
        
        # 1. Embeddings (KHÔNG có position)
        h = self.word_embeddings(input_ids)
        h = self.embed_norm(h)
        h = self.embed_dropout(h)
        
        # 2. Forward qua các layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches else None
            h, new_kv = layer(h, past_kv_cache=layer_cache, audio_hidden=audio_hidden)
            new_kv_caches.append(new_kv)
        
        # 3. LM Head
        logits = self.lm_head(h)  # [B, active_len, vocab_size]
        
        return logits, new_kv_caches
    
    @classmethod
    def from_pretrained_xlmr(cls, config, tokenizer):
        """
        Khởi tạo từ XLM-R pretrained.
        Copy weights, bỏ position embeddings.
        """
        from transformers import XLMRobertaModel
        
        xlmr = XLMRobertaModel.from_pretrained(config.backbone)
        model = cls(config, tokenizer)
        
        # Copy word embeddings
        model.word_embeddings.weight.data = xlmr.embeddings.word_embeddings.weight.data.clone()
        
        # Copy layer weights (self-attn, FFN)
        for i, (src, dst) in enumerate(zip(xlmr.encoder.layer, model.layers)):
            # Self-attention
            dst.self_attn.q_proj.weight.data = src.attention.self.query.weight.data.clone()
            dst.self_attn.q_proj.bias.data = src.attention.self.query.bias.data.clone()
            dst.self_attn.k_proj.weight.data = src.attention.self.key.weight.data.clone()
            dst.self_attn.k_proj.bias.data = src.attention.self.key.bias.data.clone()
            dst.self_attn.v_proj.weight.data = src.attention.self.value.weight.data.clone()
            dst.self_attn.v_proj.bias.data = src.attention.self.value.bias.data.clone()
            dst.self_attn.o_proj.weight.data = src.attention.output.dense.weight.data.clone()
            dst.self_attn.o_proj.bias.data = src.attention.output.dense.bias.data.clone()
            
            # LayerNorm
            dst.self_attn_norm.weight.data = src.attention.output.LayerNorm.weight.data.clone()
            dst.self_attn_norm.bias.data = src.attention.output.LayerNorm.bias.data.clone()
            
            # FFN
            dst.ffn[0].weight.data = src.intermediate.dense.weight.data.clone()
            dst.ffn[0].bias.data = src.intermediate.dense.bias.data.clone()
            dst.ffn[3].weight.data = src.output.dense.weight.data.clone()
            dst.ffn[3].bias.data = src.output.dense.bias.data.clone()
            dst.ffn_norm.weight.data = src.output.LayerNorm.weight.data.clone()
            dst.ffn_norm.bias.data = src.output.LayerNorm.bias.data.clone()
        
        # ❌ BỎ: position_embeddings (không copy)
        # ✅ Cross-attention layers: khởi tạo ngẫu nhiên (mới)
        
        del xlmr  # Giải phóng bộ nhớ
        return model
```

### 4.6. Lưu Ý Quan Trọng Cho Text Backbone

| # | Lưu ý | Chi tiết | Hậu quả nếu bỏ qua |
|---|---|---|---|
| 1 | **Bỏ position_embeddings** | XLM-R gốc có `position_embeddings` → PHẢI bỏ | Nếu giữ → giới hạn 512 tokens, không infinite được |
| 2 | **RoPE chỉ ở Phase B** | Layer 1-6 KHÔNG dùng RoPE | Nếu dùng RoPE ở layer dưới → mất tính translation-invariant |
| 3 | **Cross-attn chỉ ở Phase B** | Layer 1-6 không cross-attend audio | Nếu cross-attn ở mọi layer → compute tăng, layer dưới chưa cần global info |
| 4 | **Khởi tạo cross-attn nhỏ** | `gain=0.01` cho output projection | Nếu khởi tạo lớn → phá vỡ pretrained weights → loss spike |
| 5 | **Sliding window mask** | Phải tạo đúng: token i chỉ attend [i-w_left, i+w_right] | Nếu mask sai → token nhìn thấy tương lai → leak thông tin |
| 6 | **KV cache shape** | `[B, num_heads, cache_len, head_dim]` | Sai shape → cat sai → crash |
| 7 | **RoPE positions** | Trong active zone: positions = [0, 1, 2, ..., S-1] | Nếu dùng position tuyệt đối → overflow khi text dài |
| 8 | **Flash Attention** | Dùng `F.scaled_dot_product_attention` (PyTorch ≥ 2.0) | Nếu dùng manual attention → chậm 3-5x, OOM |

---

## V. THÀNH PHẦN 4: STREAMING DIFFUSION ENGINE

### 5.1. Kiến Trúc 3 Vùng

```python
class StreamingDiffusionEngine:
    """
    Quản lý toàn bộ quá trình diffusion trong chế độ streaming.
    Quản lý 3 vùng: Frozen / Active / Future.
    """
    
    def __init__(self, config: StreamingDiffusionConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        self.MASK_ID = tokenizer.convert_tokens_to_ids("<mask>")
        self.EOS_ID = tokenizer.eos_token_id
        self.PAD_ID = tokenizer.pad_token_id
        
        # === 3 ZONES ===
        self.frozen_tokens: List[int] = []           # Token đã freeze
        self.frozen_kv_caches: List[Tuple] = []      # KV cache per layer
        self.active_tokens: List[int] = []           # Token đang denoise
        self.active_confidence: List[float] = []     # Confidence của active tokens
        
        # === AUDIO BUFFER ===
        self.audio_buffer: List[torch.Tensor] = []   # Audio embeddings đã nhận
        self.audio_kv_cache: Optional[torch.Tensor] = None
        
        # === STATE ===
        self.total_audio_chunks_received: int = 0
        self.total_tokens_yielded: int = 0
        
    def reset(self):
        """Reset toàn bộ state (cho sample mới)"""
        self.frozen_tokens = []
        self.frozen_kv_caches = []
        self.active_tokens = []
        self.active_confidence = []
        self.audio_buffer = []
        self.audio_kv_cache = None
        self.total_audio_chunks_received = 0
        self.total_tokens_yielded = 0
```

### 5.2. Quá Trình Nhận Audio Chunk Mới

```python
    def on_new_audio_chunk(
        self, 
        audio_embeds: torch.Tensor,  # [frames, D_audio] từ Moonshine
        model: StreamingDiffusionModel,
    ) -> List[int]:
        """
        Gọi mỗi khi nhận được 1 audio chunk mới.
        
        Returns: list of token IDs đã được freeze (yield ra output)
        """
        self.total_audio_chunks_received += 1
        
        # 1. Project audio qua adapter
        audio_projected = model.audio_adapter(audio_embeds.unsqueeze(0))  # [1, tokens, D_text]
        self.audio_buffer.append(audio_projected.squeeze(0))
        
        # 2. Dự đoán số text tokens mới cần sinh
        num_new_tokens = model.length_predictor.predict_chunk(
            audio_projected, 
            context_tokens=self.frozen_tokens[-32:]  # context gần nhất
        )
        num_new_tokens = max(1, min(num_new_tokens, 20))  # Clamp
        
        # 3. Thêm MASK tokens vào active zone
        self.active_tokens.extend([self.MASK_ID] * num_new_tokens)
        self.active_confidence.extend([0.0] * num_new_tokens)
        
        # 4. Nếu active zone quá dài → chỉ giữ window_size
        if len(self.active_tokens) > self.config.active_window_size:
            # Phần thừa giữ nguyên MASK, không xử lý (future zone)
            pass
        
        # 5. Denoise active zone
        self._denoise_active_zone(model)
        
        # 6. Freeze tokens tự tin
        newly_frozen = self._freeze_confident_tokens(model)
        
        return newly_frozen
```

### 5.3. Quá Trình Denoise

```python
    def _denoise_active_zone(self, model: StreamingDiffusionModel):
        """
        Chạy K bước denoise CHỈ trên active zone.
        Sử dụng KV cache từ frozen zone.
        """
        K = self.config.streaming_denoise_steps  # 2-3 bước
        
        for step in range(K):
            # Chuẩn bị input
            active_ids = torch.tensor([self.active_tokens], dtype=torch.long)  # [1, active_len]
            
            # Concat audio buffer (chỉ chunk hiện tại + 1 chunk trước)
            if len(self.audio_buffer) >= 2:
                audio_context = torch.cat(self.audio_buffer[-2:], dim=0)  # [frames*2, D]
            else:
                audio_context = self.audio_buffer[-1]  # [frames, D]
            
            # Forward qua backbone
            logits, new_kv = model.backbone(
                input_ids=active_ids,
                past_kv_caches=self.frozen_kv_caches if self.frozen_kv_caches else None,
                audio_hidden=audio_context.unsqueeze(0),
            )
            # logits: [1, active_len, vocab_size]
            
            # Không bao giờ sinh MASK
            logits[:, :, self.MASK_ID] = float('-inf')
            
            # Lấy token + confidence
            probs = F.softmax(logits, dim=-1)  # [1, active_len, vocab]
            confidence, predicted = probs.max(dim=-1)  # [1, active_len]
            
            predicted = predicted.squeeze(0).tolist()    # [active_len]
            confidence = confidence.squeeze(0).tolist()  # [active_len]
            
            # === REMASK: Token kém tự tin → đặt lại MASK ===
            if step < K - 1:  # Không remask ở bước cuối
                num_remask = int(len(predicted) * self.config.remask_confidence_threshold)
                if num_remask > 0:
                    # Tìm bottom-k kém tự tin nhất
                    conf_tensor = torch.tensor(confidence)
                    _, remask_indices = conf_tensor.topk(num_remask, largest=False)
                    
                    for idx in remask_indices.tolist():
                        predicted[idx] = self.MASK_ID
                        confidence[idx] = 0.0
            
            # Cập nhật active zone
            self.active_tokens = predicted
            self.active_confidence = confidence
```

### 5.4. Quá Trình Freeze

```python
    def _freeze_confident_tokens(self, model: StreamingDiffusionModel) -> List[int]:
        """
        Token có confidence > θ → chuyển sang FROZEN.
        Chiến lược: monotonic_left (chỉ freeze từ trái sang, không bỏ qua token chưa chắc).
        
        Returns: list of token IDs vừa được freeze
        """
        θ = self.config.freeze_confidence_threshold  # 0.92
        
        # Đếm số token liên tiếp từ trái có conf > θ
        freeze_count = 0
        for i, (token, conf) in enumerate(zip(self.active_tokens, self.active_confidence)):
            if conf >= θ and token != self.MASK_ID:
                freeze_count += 1
            else:
                break  # Gặp token chưa chắc → DỪNG (monotonic)
        
        if freeze_count == 0:
            return []
        
        # 1. Tách tokens
        newly_frozen_tokens = self.active_tokens[:freeze_count]
        self.active_tokens = self.active_tokens[freeze_count:]
        self.active_confidence = self.active_confidence[freeze_count:]
        
        # 2. Tính KV cache cho tokens mới freeze (chỉ tính 1 lần)
        frozen_ids = torch.tensor([newly_frozen_tokens], dtype=torch.long)
        with torch.no_grad():
            _, new_kv = model.backbone(
                input_ids=frozen_ids,
                past_kv_caches=self.frozen_kv_caches if self.frozen_kv_caches else None,
                audio_hidden=None,  # Frozen tokens không cần audio nữa
            )
        
        # 3. Cập nhật KV cache
        if not self.frozen_kv_caches:
            self.frozen_kv_caches = new_kv
        else:
            self.frozen_kv_caches = [
                (torch.cat([old_k, new_k], dim=2), torch.cat([old_v, new_v], dim=2))
                for (old_k, old_v), (new_k, new_v) in zip(self.frozen_kv_caches, new_kv)
            ]
        
        # 4. Cập nhật frozen tokens list
        self.frozen_tokens.extend(newly_frozen_tokens)
        
        # 5. Cắt KV cache nếu quá dài (sliding window trên cache)
        max_cache = self.config.frozen_cache_size  # 128
        if len(self.frozen_tokens) > max_cache:
            excess = len(self.frozen_tokens) - max_cache
            self.frozen_tokens = self.frozen_tokens[excess:]
            
            # Cắt KV cache: bỏ excess tokens đầu tiên
            self.frozen_kv_caches = [
                (k[:, :, excess:, :], v[:, :, excess:, :])
                for k, v in self.frozen_kv_caches
            ]
        
        self.total_tokens_yielded += freeze_count
        return newly_frozen
```

### 5.5. Flush (Khi Audio Kết Thúc)

```python
    def flush(self, model: StreamingDiffusionModel) -> List[int]:
        """
        Gọi khi audio stream kết thúc.
        Denoise phần active còn lại với nhiều bước hơn (không cần real-time nữa).
        """
        if not self.active_tokens:
            return []
        
        # Denoise với nhiều bước hơn (10 bước như offline)
        original_steps = self.config.streaming_denoise_steps
        self.config.streaming_denoise_steps = 10
        
        # Thêm EOS vào cuối
        self.active_tokens.append(self.EOS_ID)
        self.active_confidence.append(1.0)
        
        self._denoise_active_zone(model)
        
        # Freeze tất cả (không cần check confidence nữa)
        remaining = self.active_tokens.copy()
        self.active_tokens = []
        self.active_confidence = []
        self.frozen_tokens.extend(remaining)
        
        self.config.streaming_denoise_steps = original_steps
        return remaining
```

### 5.6. Lưu Ý Cho Diffusion Engine

| # | Lưu ý | Chi tiết |
|---|---|---|
| 1 | **Monotonic freeze** | Chỉ freeze từ trái sang, không bỏ qua token chưa chắc ở giữa. Nếu token 3 chưa chắc → token 4,5 dù chắc cũng KHÔNG freeze → tránh output lộn xộn |
| 2 | **Không remask ở bước cuối** | Bước denoise cuối cùng → giữ nguyên prediction → tránh token bị MASK khi yield |
| 3 | **KV cache chỉ tính 1 lần** | Token freeze → tính KV → cache → không bao giờ tính lại → tiết kiệm compute |
| 4 | **Audio buffer giới hạn** | Chỉ giữ 2 chunk audio gần nhất → không cần toàn bộ audio history |
| 5 | **Length predictor cho chunk** | Dự đoán số token cho mỗi chunk, không phải toàn bộ → cần sửa length predictor |
| 6 | **EOS handling** | Không thêm EOS vào active zone trong streaming → chỉ thêm khi flush |
| 7 | **Thread safety** | Nếu dùng async audio input → cần lock khi truy cập active_tokens |

---

## VI. THÀNH PHẦN 5: LENGTH PREDICTOR

### 6.1. Vấn Đề Với Length Predictor Hiện Tại

Length predictor hiện tại dự đoán **tổng độ dài output** cho toàn bộ audio. Trong streaming, cần dự đoán **số token mới cho mỗi chunk**.

### 6.2. Kiến Trúc Mới

```python
class StreamingLengthPredictor(nn.Module):
    """
    Dự đoán số text tokens mới cần sinh cho mỗi audio chunk.
    
    Input:  audio embeddings của chunk hiện tại + context tokens đã freeze
    Output: số token mới (integer)
    """
    
    def __init__(self, config):
        super().__init__()
        
        D = config.hidden_size  # 768
        max_output_tokens = 32  # Max tokens per chunk
        
        # Audio pooling
        self.audio_pool = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
        )
        
        # Context encoding (tokens đã freeze gần nhất)
        self.context_embed = nn.Embedding(config.vocab_size, D)
        self.context_pool = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, nhead=4, batch_first=True),
            num_layers=1,
        )
        
        # Fusion + Predict
        self.predictor = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D, max_output_tokens + 1),  # +1 cho class "0 tokens"
        )
        
        self.max_output_tokens = max_output_tokens
    
    def forward(self, audio_embeds, context_token_ids=None):
        """
        Args:
            audio_embeds: [B, frames, D]
            context_token_ids: [B, context_len] — frozen tokens gần nhất
            
        Returns:
            logits: [B, max_output_tokens + 1]
        """
        # Pool audio
        audio_feat = self.audio_pool(audio_embeds).mean(dim=1)  # [B, D]
        
        # Pool context
        if context_token_ids is not None and context_token_ids.shape[1] > 0:
            ctx_embed = self.context_embed(context_token_ids)
            ctx_feat = self.context_pool(ctx_embed).mean(dim=1)  # [B, D]
        else:
            ctx_feat = torch.zeros_like(audio_feat)
        
        # Predict
        combined = torch.cat([audio_feat, ctx_feat], dim=-1)  # [B, 2D]
        logits = self.predictor(combined)  # [B, max_tokens+1]
        
        return logits
    
    def predict_chunk(self, audio_embeds, context_tokens=None):
        """Inference: trả về int"""
        with torch.no_grad():
            if context_tokens:
                ctx = torch.tensor([context_tokens], dtype=torch.long)
            else:
                ctx = None
            logits = self.forward(audio_embeds.unsqueeze(0), ctx)
            predicted = logits.argmax(dim=-1).item()
        return predicted
```

### 6.3. Training Cho Length Predictor

```python
# Trong training, target = số token text tương ứng với chunk audio
# Ví dụ: audio chunk 1 (2s) tương ứng với 5 text tokens → target = 5

def length_predictor_loss(model, audio_chunk, context_tokens, target_num_tokens):
    logits = model.length_predictor(audio_chunk, context_tokens)
    loss = F.cross_entropy(logits, torch.tensor([target_num_tokens]))
    return loss
```

---

## VII. TRAINING PIPELINE

### 7.1. Tổng Quan Luồng Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                               │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Raw Dataset   │  (full_audio, full_text)                         │
│  │ VietSpeech    │  Không cần thu thập lại                          │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │ STREAMING AUGMENTATION (on-the-fly)                      │       │
│  │                                                          │       │
│  │ 1. Chia audio thành N chunks (2s mỗi chunk)             │       │
│  │ 2. Ngẫu nhiên chọn k ∈ {1, 2, ..., N}                  │       │
│  │ 3. Input audio = chunks[1..k] (che chunks[k+1..N])      │       │
│  │ 4. Target text = FULL text (không đổi)                   │       │
│  │ 5. Tính support_ratio = k / N                            │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │ DIFFUSION FORWARD                                        │       │
│  │                                                          │       │
│  │ 1. Sample timestep t ∈ {1, ..., T}                       │       │
│  │ 2. q_sample: mask target text theo t                     │       │
│  │ 3. Forward: model(masked_text, partial_audio) → logits   │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │ STREAMING LOSS                                           │       │
│  │                                                          │       │
│  │ 1. Position-weighted CE:                                 │       │
│  │    • Token trong vùng "supported" → weight 2.0           │       │
│  │    • Token ngoài vùng "supported" → weight 0.3           │       │
│  │                                                          │       │
│  │ 2. Timestep weighting (giữ từ cũ):                      │       │
│  │    • weight = T - (t - 1)                                │       │
│  │                                                          │       │
│  │ 3. (Optional) Confidence calibration loss:               │       │
│  │    • Penalize nếu model quá tự tin ở vùng unsupported    │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2. Dataset Class

```python
class StreamingAugmentedDataset(Dataset):
    """
    Wrapper dataset: từ (full_audio, full_text) tạo streaming training samples.
    
    Mỗi lần __getitem__ → ngẫu nhiên chọn "thời điểm streaming" khác nhau.
    → 1 sample gốc sinh ra N training samples khác nhau (data augmentation).
    """
    
    def __init__(
        self,
        base_dataset,                    # Dataset gốc (full audio + text)
        tokenizer,
        chunk_duration: float = 2.0,     # giây
        overlap_duration: float = 0.5,   # giây
        sample_rate: int = 16000,
        max_text_length: int = 256,
        curriculum_step: int = -1,       # -1 = không curriculum
        total_curriculum_steps: int = 10000,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.hop_samples = self.chunk_samples - self.overlap_samples
        
        # Curriculum
        self.curriculum_step = curriculum_step
        self.total_curriculum_steps = total_curriculum_steps
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        sample = self.base[idx]
        audio = sample["audio"]          # [total_samples]
        text = sample["text"]            # str
        task_token = sample.get("task_token", "<vi_en>")
        
        # 1. Tokenize text
        text_ids = self.tokenizer.encode(
            text, 
            max_length=self.max_text_length, 
            truncation=True,
            add_special_tokens=True,
        )
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        
        # 2. Chia audio thành chunks
        total_samples = len(audio)
        num_chunks = max(1, (total_samples - self.overlap_samples) // self.hop_samples)
        
        audio_chunks = []
        for i in range(num_chunks):
            start = i * self.hop_samples
            end = start + self.chunk_samples
            chunk = audio[start:end]
            if len(chunk) < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - len(chunk)))
            audio_chunks.append(chunk)
        
        # 3. Curriculum: giới hạn số chunks tối đa
        if self.curriculum_step >= 0:
            progress = self.curriculum_step / self.total_curriculum_steps
            max_visible = max(1, int(progress * num_chunks))
        else:
            max_visible = num_chunks
        
        # 4. Ngẫu nhiên chọn số chunks "nhìn thấy"
        num_visible = random.randint(1, max_visible)
        
        # 5. Tạo input audio (che tương lai)
        visible_chunks = torch.stack(audio_chunks[:num_visible])    # [num_visible, chunk_samples]
        # Không cần tạo masked chunks — chỉ đơn giản là không đưa vào model
        
        # 6. Tính support ratio
        support_ratio = num_visible / num_chunks
        
        # 7. Tính chunk boundaries cho text (ước lượng)
        #    Giả sử text phân bố đều theo audio
        tokens_per_chunk = len(text_ids) / num_chunks
        supported_text_len = int(support_ratio * len(text_ids))
        
        return {
            "audio_chunks": visible_chunks,           # [num_visible, chunk_samples]
            "text_ids": text_ids,                     # [text_len] — FULL text
            "support_ratio": support_ratio,           # float
            "supported_text_len": supported_text_len, # int
            "num_visible_chunks": num_visible,        # int
            "total_chunks": num_chunks,               # int
            "task_token": task_token,                 # str
        }
```

### 7.3. Data Collator

```python
class StreamingCollator:
    """
    Collate các samples thành batch.
    Xử lý padding cho audio và text có độ dài khác nhau.
    """
    
    def __init__(self, tokenizer, pad_token_id=1):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        # Audio: pad theo số chunks và chunk length
        max_chunks = max(s["audio_chunks"].shape[0] for s in batch)
        chunk_len = batch[0]["audio_chunks"].shape[1]
        
        audio_batch = torch.full((len(batch), max_chunks, chunk_len), 0.0)
        audio_mask = torch.zeros(len(batch), max_chunks, dtype=torch.bool)
        
        for i, s in enumerate(batch):
            n = s["audio_chunks"].shape[0]
            audio_batch[i, :n] = s["audio_chunks"]
            audio_mask[i, :n] = True
        
        # Text: pad theo max length
        max_text = max(len(s["text_ids"]) for s in batch)
        text_batch = torch.full((len(batch), max_text), self.pad_token_id, dtype=torch.long)
        text_mask = torch.zeros(len(batch), max_text, dtype=torch.bool)
        
        for i, s in enumerate(batch):
            n = len(s["text_ids"])
            text_batch[i, :n] = s["text_ids"]
            text_mask[i, :n] = True
        
        # Support info
        support_ratios = torch.tensor([s["support_ratio"] for s in batch])
        supported_lens = torch.tensor([s["supported_text_len"] for s in batch])
        
        # Task tokens
        task_tokens = [s["task_token"] for s in batch]
        
        return {
            "audio_chunks": audio_batch,       # [B, max_chunks, chunk_samples]
            "audio_mask": audio_mask,           # [B, max_chunks]
            "text_ids": text_batch,             # [B, max_text]
            "text_mask": text_mask,             # [B, max_text]
            "support_ratios": support_ratios,   # [B]
            "supported_lens": supported_lens,   # [B]
            "task_tokens": task_tokens,         # List[str]
        }
```

### 7.4. Trainer — Compute Loss

```python
class StreamingDiffusionTrainer(Trainer):
    """
    Trainer cho streaming diffusion.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        audio_chunks = inputs["audio_chunks"]       # [B, num_chunks, chunk_samples]
        audio_mask = inputs["audio_mask"]            # [B, num_chunks]
        text_ids = inputs["text_ids"]                # [B, text_len]
        text_mask = inputs["text_mask"]              # [B, text_len]
        support_ratios = inputs["support_ratios"]    # [B]
        supported_lens = inputs["supported_lens"]    # [B]
        
        B, T_len = text_ids.shape
        T = self.model.config.num_diffusion_timesteps  # 50
        
        # === 1. ENCODE AUDIO (chỉ chunks visible) ===
        # Moonshine encode từng chunk
        audio_embeds_list = []
        for b in range(B):
            visible = audio_chunks[b][audio_mask[b]]  # [num_visible, chunk_samples]
            with torch.no_grad():
                chunk_embeds = model.audio_encoder(visible)  # [num_visible, frames, D_audio]
            # Project qua adapter
            projected = model.audio_adapter(chunk_embeds)    # [num_visible, tokens, D_text]
            audio_embeds_list.append(projected.reshape(-1, projected.shape[-1]))
        
        # Pad audio embeds
        max_audio_len = max(a.shape[0] for a in audio_embeds_list)
        audio_embeds = torch.zeros(B, max_audio_len, model.config.hidden_size, device=text_ids.device)
        audio_embed_mask = torch.zeros(B, max_audio_len, dtype=torch.bool, device=text_ids.device)
        for b, a in enumerate(audio_embeds_list):
            audio_embeds[b, :a.shape[0]] = a
            audio_embed_mask[b, :a.shape[0]] = True
        
        # === 2. DIFFUSION FORWARD (q_sample) ===
        # Sample timestep
        t = torch.randint(1, T + 1, (B,), device=text_ids.device)
        
        # Maskable mask: chỉ mask text tokens (không mask special tokens)
        maskable_mask = text_mask.clone()
        # Bỏ BOS, EOS, task token khỏi maskable
        maskable_mask[:, 0] = False  # BOS
        
        # q_sample: thêm MASK theo timestep
        u = torch.rand_like(text_ids.float())
        mask_prob = t.float().unsqueeze(1) / T  # [B, 1]
        mask_indices = (u < mask_prob) & maskable_mask
        x_t = text_ids.masked_fill(mask_indices, self.model.config.mask_token_id)
        
        # === 3. FORWARD QUA MODEL ===
        # Thêm task token vào đầu
        task_ids = self.tokenizer.convert_tokens_to_ids(inputs["task_tokens"])
        task_tensor = torch.tensor(task_ids, device=text_ids.device).unsqueeze(1)
        model_input = torch.cat([task_tensor, x_t], dim=1)  # [B, 1 + text_len]
        
        logits, _ = model.backbone(
            input_ids=model_input,
            audio_hidden=audio_embeds,
        )
        logits = logits[:, 1:, :]  # Bỏ task token position → [B, text_len, vocab]
        
        # === 4. STREAMING LOSS ===
        # 4a. Position weighting
        position_weights = torch.ones(B, T_len, device=text_ids.device)
        for b in range(B):
            sup_len = supported_lens[b].item()
            position_weights[b, :sup_len] = self.config.loss_weight_supported    # 2.0
            position_weights[b, sup_len:] = self.config.loss_weight_unsupported  # 0.3
        
        # 4b. Timestep weighting
        timestep_weights = (T - (t.float() - 1)).unsqueeze(1)  # [B, 1]
        
        # 4c. Cross-entropy (chỉ trên masked positions)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            text_ids.reshape(-1),
            reduction='none',
        ).reshape(B, T_len)
        
        # 4d. Combined loss
        loss_mask = mask_indices.float()  # Chỉ tính loss trên positions bị mask
        weighted_loss = ce * position_weights * loss_mask
        weighted_loss = weighted_loss * timestep_weights
        
        # Normalize
        num_masked = loss_mask.sum(dim=1).clamp(min=1)  # [B]
        loss_per_sample = weighted_loss.sum(dim=1) / num_masked
        loss = loss_per_sample.mean()
        
        # === 5. (Optional) Confidence Calibration Loss ===
        # Penalize nếu model quá tự tin ở vùng UNSUPPORTED
        if self.config.use_confidence_calibration:
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values  # [B, text_len]
            
            # Vùng unsupported: model KHÔNG NÊN quá tự tin
            unsupported_mask = torch.zeros(B, T_len, device=text_ids.device)
            for b in range(B):
                sup_len = supported_lens[b].item()
                unsupported_mask[b, sup_len:] = 1.0
            
            # Penalize confidence > 0.5 ở vùng unsupported
            overconfident = (max_probs - 0.5).clamp(min=0) * unsupported_mask
            cal_loss = overconfident.sum() / unsupported_mask.sum().clamp(min=1)
            
            loss = loss + 0.1 * cal_loss  # λ_cal = 0.1
        
        if return_outputs:
            return loss, {"logits": logits}
        return loss
```

### 7.5. Curriculum Training Schedule

```python
class CurriculumCallback(TrainerCallback):
    """
    Curriculum: tăng dần độ khó theo training progress.
    
    Giai đoạn 1 (0-20%):   Model thấy 1-2 chunks → học basic mapping
    Giai đoạn 2 (20-50%):  Model thấy 1-5 chunks → học context
    Giai đoạn 3 (50-80%):  Model thấy 1-10 chunks → học long-range
    Giai đoạn 4 (80-100%): Model thấy full chunks → học full sequence
    """
    
    def on_step_begin(self, args, state, control, **kwargs):
        progress = state.global_step / state.max_steps
        
        if progress < 0.2:
            max_chunks_ratio = 0.2    # Thấy tối đa 20% audio
        elif progress < 0.5:
            max_chunks_ratio = 0.5
        elif progress < 0.8:
            max_chunks_ratio = 0.8
        else:
            max_chunks_ratio = 1.0    # Thấy toàn bộ
        
        # Cập nhật dataset
        if hasattr(state.train_dataloader.dataset, 'curriculum_step'):
            state.train_dataloader.dataset.curriculum_step = state.global_step
            state.train_dataloader.dataset.total_curriculum_steps = state.max_steps
```

### 7.6. Training Config

```json
{
    "output_dir": "./streaming-diffusion-v1",
    "backbone": "FacebookAI/xlm-roberta-base",
    "audio_encoder_name": "UsefulSensors/moonshine-streaming-medium",
    
    "num_hidden_layers": 12,
    "num_ergodic_layers": 6,
    "num_position_layers": 6,
    "hidden_size": 768,
    
    "ergodic_window_left": 64,
    "ergodic_window_right": 16,
    "position_window_left": 128,
    "position_window_right": 32,
    "rope_theta": 10000.0,
    
    "num_diffusion_timesteps": 50,
    "streaming_denoise_steps": 3,
    "active_window_size": 64,
    "frozen_cache_size": 128,
    "freeze_confidence_threshold": 0.92,
    "remask_confidence_threshold": 0.30,
    
    "audio_chunk_duration": 2.0,
    "audio_overlap_duration": 0.5,
    
    "loss_weight_supported": 2.0,
    "loss_weight_unsupported": 0.3,
    "use_confidence_calibration": true,
    
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "max_steps": 100000,
    "bf16": true,
    
    "eval_steps": 2000,
    "save_steps": 5000,
    "eval_metrics": ["bleu", "wer"],
    
    "task_tokens": ["<vi_en>", "<vi_zh>", "<vi_ko>"],
    "dataset_type": "speech_translation_multitask",
    "streaming_augmentation": true,
    "curriculum_training": true
}
```

### 7.7. Lưu Ý Training

| # | Lưu ý | Chi tiết | Hậu quả nếu bỏ qua |
|---|---|---|---|
| 1 | **Học LR nhỏ hơn bình thường** | 5e-5 thay vì 1e-4 | Vì backbone pretrained → LR lớn phá vỡ weights |
| 2 | **Warmup bắt buộc** | 5% tổng steps | Cross-attn layers mới khởi tạo → cần warmup |
| 3 | **Freeze audio encoder** | `requires_grad = False` | Nếu train → catastrophic forgetting, tốn memory |
| 4 | **LoRA cho backbone (optional)** | Rank 16, target Q/V | Nếu full fine-tune → tốn memory, overfit |
| 5 | **Curriculum training** | Tăng dần số chunks | Nếu không → model không học được streaming behavior |
| 6 | **Position weighting** | Supported 2.0, Unsupported 0.3 | Nếu đều 1.0 → model bị phạt nặng ở vùng chưa có audio → gradient noise |
| 7 | **Confidence calibration** | λ=0.1 | Nếu không → model quá tự tin ở vùng unsupported → yield token sai |
| 8 | **Gradient clipping** | max_norm=1.0 | Diffusion loss có thể spike → gradient explosion |
| 9 | **Eval trên streaming mode** | Eval phải dùng streaming inference | Nếu eval offline → metric không phản ánh thực tế |
| 10 | **Multi-GPU: DDP** | Dùng DeepSpeed ZeRO-2 | Model + KV cache tốn memory → ZeRO-2 chia optimizer states |

---

## VIII. INFERENCE PIPELINE

### 8.1. Toàn Bộ Luồng Inference

```python
class StreamingInferencePipeline:
    """
    Pipeline inference streaming hoàn chỉnh.
    Nhận audio từ microphone → yield text theo thời gian thực.
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        
        self.audio_encoder = StreamingAudioEncoder(
            model.audio_encoder,
            chunk_duration=config.audio_chunk_duration,
            overlap_duration=config.audio_overlap_duration,
        )
        self.diffusion_engine = StreamingDiffusionEngine(config, tokenizer)
    
    @torch.no_grad()
    def stream(self, audio_stream) -> Generator[str, None, None]:
        """
        Args:
            audio_stream: iterator yielding raw audio tensors (1D, 16kHz)
            
        Yields:
            str: text tokens đã được finalize (hiển thị lên UI)
        """
        self.diffusion_engine.reset()
        
        for audio_chunk in audio_stream:
            # 1. Feed audio vào encoder
            audio_embeds = self.audio_encoder.feed(audio_chunk)
            
            if audio_embeds is None:
                continue  # Chưa đủ 1 chunk, đợi thêm
            
            # 2. Xử lý chunk
            newly_frozen = self.diffusion_engine.on_new_audio_chunk(
                audio_embeds, self.model
            )
            
            # 3. Yield text đã freeze
            if newly_frozen:
                text = self.tokenizer.decode(newly_frozen, skip_special_tokens=True)
                yield text
        
        # 4. Flush phần còn lại
        remaining_audio = self.audio_encoder.flush()
        if remaining_audio is not None:
            newly_frozen = self.diffusion_engine.on_new_audio_chunk(
                remaining_audio, self.model
            )
            if newly_frozen:
                yield self.tokenizer.decode(newly_frozen, skip_special_tokens=True)
        
        # 5. Flush active zone
        final_tokens = self.diffusion_engine.flush(self.model)
        if final_tokens:
            yield self.tokenizer.decode(final_tokens, skip_special_tokens=True)


# === USAGE ===
def demo():
    model = load_streaming_model("path/to/checkpoint")
    tokenizer = load_tokenizer()
    config = model.config
    
    pipeline = StreamingInferencePipeline(model, tokenizer, config)
    
    # Từ microphone
    import sounddevice as sd
    
    def audio_generator():
        queue = []
        def callback(indata, frames, time_info, status):
            queue.append(torch.from_numpy(indata[:, 0]))
        
        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            while True:
                if queue:
                    yield queue.pop(0)
                else:
                    time.sleep(0.01)
    
    print("🎤 Bắt đầu nói...")
    full_text = ""
    for text_chunk in pipeline.stream(audio_generator()):
        full_text += text_chunk
        print(f"\r🗣️ {full_text}", end="", flush=True)
```

### 8.2. Độ Trễ (Latency Analysis)

```
Thành phần                    Thời gian ước tính
─────────────────────────────────────────────────
Audio chunk duration:         2000 ms
Audio overlap:                 500 ms
Moonshine encode (1 chunk):    ~20 ms  (NPU)
Audio adapter:                  ~2 ms
Length predictor:               ~1 ms
Denoise 3 bước:               ~45 ms  (3 × 15ms/backbone forward)
Freeze check:                   ~1 ms
─────────────────────────────────────────────────
Tổng processing:              ~69 ms
Tổng latency end-to-end:     ~2069 ms  (chủ yếu do đợi audio chunk)

→ Latency ≈ 2 giây (bằng chunk duration)
→ Có thể giảm xuống 1 giây nếu chunk_duration = 1.0
→ Trade-off: chunk nhỏ hơn → ít context hơn → chất lượng giảm
```

### 8.3. Memory Analysis

```
Thành phần                    Memory
──────────────────────────────────────
Audio encoder (frozen):       ~100 MB
Audio adapter:                  ~3 MB
Text backbone (12 layers):    ~500 MB  (xlm-roberta-base)
KV cache (128 tokens × 12 layers):
  = 128 × 12 × 2 × 768 × 4 bytes
  = ~9.4 MB
Active zone (64 tokens):       ~0.2 MB
Audio buffer (2 chunks):       ~0.4 MB
──────────────────────────────────────
Tổng inference memory:       ~613 MB

→ Fit trên Samsung Galaxy S25 (12GB RAM) ✅
→ Fit trên Snapdragon NPU (shared memory) ✅
```

---

## IX. MIGRATION PLAN TỪ CODEBASE HIỆN TẠI

### 9.1. File Nào Giữ, File Nào Sửa, File Nào Mới

```
diffusion-speech-recognition/
├── src/
│   ├── model/
│   │   ├── dd_model.py                    # ⚠️ SỬA NHIỀU
│   │   │   ├── Bỏ: DiscreteDiffusionXLMRModel (full attention)
│   │   │   ├── Thêm: StreamingDiffusionModel (mới)
│   │   │   ├── Thêm: StreamingAudioAdapter
│   │   │   ├── Thêm: StreamingLengthPredictor
│   │   │   └── Giữ: DiscreteDiffusionModelArguments (sửa thêm fields)
│   │   │
│   │   ├── streaming_backbone.py          # 🆕 MỚI HOÀN TOÀN
│   │   │   ├── RotaryEmbedding
│   │   │   ├── apply_rotary_pos_emb()
│   │   │   ├── SlidingWindowAttention
│   │   │   ├── StreamingRobertaLayer
│   │   │   └── StreamingDiffusionBackbone
│   │   │
│   │   ├── streaming_diffusion_engine.py  # 🆕 MỚI HOÀN TOÀN
│   │   │   └── StreamingDiffusionEngine (3-zone management)
│   │   │
│   │   ├── modeling_dlm.py               # ⚠️ SỬA
│   │   │   ├── Thêm: generate_stream() method
│   │   │   └── Giữ: generate() (cho offline eval)
│   │   │
│   │   ├── configuration_dlm.py          # ⚠️ SỬA
│   │   │   └── Thêm: streaming config fields
│   │   │
│   │   └── cross_attn_roberta.py         # ❌ XÓA (thay bằng StreamingRobertaLayer)
│   │
│   ├── data/
│   │   ├── streaming_augmented.py        # 🆕 MỚI
│   │   │   └── StreamingAugmentedDataset
│   │   │
│   │   ├── collator.py                   # ⚠️ SỬA
│   │   │   └── Thêm: StreamingCollator
│   │   │
│   │   └── (các file khác)               # ✅ GIỮ NGUYÊN
│   │
│   ├── trainer/
│   │   ├── dd_trainer.py                 # ⚠️ SỬA NHIỀU
│   │   │   ├── Thêm: StreamingDiffusionTrainer
│   │   │   ├── Thêm: streaming compute_loss()
│   │   │   └── Giữ: DiscreteDiffusionTrainer (cho offline)
│   │   │
│   │   └── curriculum_callback.py        # 🆕 MỚI
│   │
│   ├── dd_generator.py                   # ⚠️ SỬA
│   │   ├── Giữ: DiscreteDiffusionGenerator (offline)
│   │   └── Thêm: StreamingDiffusionGenerator
│   │
│   ├── streaming_demo.py                 # 🆕 MỚI
│   │   └── Demo với microphone
│   │
│   └── train.py                          # ⚠️ SỬA
│       └── Thêm: streaming training mode
│
├── configs/
│   ├── streaming_vi_multitask.json       # 🆕 MỚI
│   └── (configs cũ)                     # ✅ GIỮ
│
└── scripts/
    └── training/
        └── train_streaming.sh            # 🆕 MỚI
```

### 9.2. Thứ Tự Implement

```
Phase 1: Foundation (Tuần 1-2)
├── 1.1. Implement RoPE (streaming_backbone.py)
├── 1.2. Implement SlidingWindowAttention
├── 1.3. Implement StreamingRobertaLayer
├── 1.4. Implement StreamingDiffusionBackbone
├── 1.5. Test: forward pass chạy không crash
└── 1.6. Test: từ XLM-R pretrained → copy weights → loss ban đầu hợp lý

Phase 2: Adapter + Engine (Tuần 2-3)
├── 2.1. Implement StreamingAudioAdapter
├── 2.2. Implement StreamingLengthPredictor
├── 2.3. Implement StreamingDiffusionEngine (3 zones)
├── 2.4. Test: feed audio chunk → get frozen tokens
└── 2.5. Test: KV cache đúng shape, đúng giá trị

Phase 3: Training (Tuần 3-5)
├── 3.1. Implement StreamingAugmentedDataset
├── 3.2. Implement StreamingCollator
├── 3.3. Implement StreamingDiffusionTrainer.compute_loss()
├── 3.4. Implement CurriculumCallback
├── 3.5. Test: training 100 steps → loss giảm
├── 3.6. Test: eval streaming mode → BLEU/WER hợp lý
└── 3.7. Full training run

Phase 4: Inference + Demo (Tuần 5-6)
├── 4.1. Implement StreamingInferencePipeline
├── 4.2. Implement streaming_demo.py (microphone)
├── 4.3. Test: real-time demo
├── 4.4. Latency profiling
└── 4.5. Memory profiling

Phase 5: Deployment (Tuần 6-8)
├── 5.1. ONNX export (streaming mode)
├── 5.2. Qualcomm AI Hub conversion
├── 5.3. On-device testing (Galaxy S25)
└── 5.4. Optimization (quantization, pruning)
```

---

## X. CHECKLIST & LƯU Ý QUAN TRỌNG

### 10.1. Checklist Trước Khi Training

- [ ] XLM-R pretrained weights đã copy đúng (so sánh loss ban đầu)
- [ ] Position embeddings đã BỎ (không còn trong model)
- [ ] RoPE chỉ áp dụng ở Phase B (layer 7-12)
- [ ] Cross-attention chỉ ở Phase B, khởi tạo gain=0.01
- [ ] Sliding window mask đúng hướng (không leak tương lai)
- [ ] Audio encoder frozen (`requires_grad=False`)
- [ ] Audio adapter có position embedding
- [ ] Dataset augmentation: che audio, giữ full text
- [ ] Loss weighting: supported 2.0, unsupported 0.3
- [ ] Curriculum callback hoạt động
- [ ] Gradient clipping max_norm=1.0
- [ ] BF16 enabled
- [ ] Eval dùng streaming inference (không phải offline)

### 10.2. Các Lỗi Thường Gặp

| # | Lỗi | Triệu chứng | Nguyên nhân | Fix |
|---|---|---|---|---|
| 1 | Loss NaN sau vài steps | Loss = NaN | Gradient explosion từ cross-attn mới | Giảm LR, tăng warmup, kiểm tra init gain |
| 2 | Loss không giảm | Loss phẳng | Position embeddings cũ vẫn còn | Kiểm tra `model.parameters()`, đảm bảo không có pos_embed |
| 3 | Output toàn MASK | Model không sinh được text | Freeze threshold quá cao | Giảm θ từ 0.92 → 0.85 |
| 4 | Output lặp lại | "xin chào xin chào xin chào" | KV cache không cập nhật đúng | Debug `_freeze_confident_tokens`, kiểm tra cache shape |
| 5 | OOM khi eval | CUDA out of memory | Active window quá lớn | Giảm `active_window_size` từ 64 → 32 |
| 6 | Latency cao | > 5 giây | Denoise quá nhiều bước | Giảm `streaming_denoise_steps` từ 3 → 2 |
| 7 | Chất lượng giảm ở câu dài | BLEU giảm sau 30s | KV cache bị truncate mất context quan trọng | Tăng `frozen_cache_size` từ 128 → 256 |
| 8 | Token sai ở biên chunk | "mọingười" (thiếu space) | Overlap audio không đủ | Tăng `audio_overlap_duration` từ 0.5 → 1.0 |
| 9 | Model quá tự tin ở vùng chưa có audio | Yield token sai sớm | Thiếu confidence calibration loss | Bật `use_confidence_calibration=True` |
| 10 | Training chậm | < 1 step/s | Audio encode trong training loop | Dùng precomputed audio embeddings |

### 10.3. Hyperparameter Tuning Guide

```
Nếu chất lượng thấp (BLEU thấp, WER cao):
  → Tăng streaming_denoise_steps: 3 → 5
  → Tăng active_window_size: 64 → 96
  → Giảm freeze_confidence_threshold: 0.92 → 0.85
  → Tăng frozen_cache_size: 128 → 256

Nếu latency cao:
  → Giảm audio_chunk_duration: 2.0 → 1.0
  → Giảm streaming_denoise_steps: 3 → 2
  → Giảm active_window_size: 64 → 32
  → Bật downsample trong adapter

Nếu OOM:
  → Giảm per_device_train_batch_size: 32 → 16
  → Giảm active_window_size: 64 → 32
  → Giảm frozen_cache_size: 128 → 64
  → Bật gradient_checkpointing
  → Dùng LoRA thay vì full fine-tune

Nếu training không ổn định:
  → Giảm learning_rate: 5e-5 → 2e-5
  → Tăng warmup_ratio: 0.05 → 0.1
  → Tăng gradient_accumulation_steps: 4 → 8
  → Kiểm tra gradient norm (nếu > 10 → có vấn đề)
```

### 10.4. Metric Đánh Giá Streaming

Ngoài BLEU/WER thông thường, cần thêm:

```python
class StreamingMetrics:
    """Metrics đặc thù cho streaming evaluation"""
    
    def chunk_latency(self, audio_time, first_token_time):
        """Thời gian từ khi bắt đầu nói → token đầu tiên xuất hiện"""
        return first_token_time - audio_time  # ms
    
    def chunk_wer(self, partial_hypothesis, partial_reference):
        """WER tại mỗi thời điểm (partial evaluation)"""
        return wer(partial_reference, partial_hypothesis)
    
    def revision_rate(self, all_hypotheses):
        """Tỷ lệ token bị sửa lại (càng thấp càng tốt)"""
        # So sánh hypothesis tại thời điểm t và t+1
        revisions = 0
        total = 0
        for t in range(len(all_hypotheses) - 1):
            h1 = all_hypotheses[t]
            h2 = all_hypotheses[t + 1]
            # Đếm token trong h1 bị thay đổi trong h2
            revisions += count_changes(h1, h2)
            total += len(h1)
        return revisions / max(total, 1)
    
    def stability_score(self, all_hypotheses):
        """1 - revision_rate. Càng gần 1 càng ổn định."""
        return 1.0 - self.revision_rate(all_hypotheses)
```

---

## TÓM TẮT CUỐI CÙNG

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. Audio:    Moonshine (giữ nguyên) → chunk 2s → overlap 0.5s  │
│                                                                 │
│  2. Adapter:  Linear + Position Embedding + LayerNorm           │
│              (cầu nối ergodic → position-aware)                 │
│                                                                 │
│  3. Backbone: Layer 1-6: Ergodic (no pos, SWA w=64)            │
│              Layer 7-12: RoPE + SWA w=128 + Cross-Attn Audio   │
│                                                                 │
│  4. Diffusion: 3 zones (Frozen/Active/Future)                  │
│               Denoise 3 bước trên Active                       │
│               Confidence > 0.92 → Freeze → Yield               │
│               Confidence < 0.30 → Remask                       │
│                                                                 │
│  5. Training:  Che audio, giữ full text                        │
│               Position-weighted loss (2.0 / 0.3)               │
│               Curriculum: tăng dần số chunks                   │
│               Confidence calibration loss                      │
│                                                                 │
│  6. Inference: Audio stream → chunk → denoise → freeze → yield │
│               Latency ≈ 2s, Memory ≈ 600MB                     │
│               Infinite audio ✅                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```