"""
Streaming Backbone for Discrete Diffusion Speech Recognition.

Replaces the standard XLM-RoBERTa backbone with a hybrid architecture:
- Phase A (Layers 1–6): Ergodic — NO position embedding, sliding window attention
- Phase B (Layers 7–12): Position-Aware — RoPE, wider sliding window, cross-attention with audio

Key design decisions:
- Absolute position embeddings are REMOVED (enables infinite-length streaming)
- RoPE only in upper layers to preserve translation-invariance in lower layers
- Cross-attention with audio only in upper layers (lower layers handle local syntax)
- Uses F.scaled_dot_product_attention for flash attention (PyTorch 2.0+)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingBackboneConfig:
    """Configuration for the streaming hybrid backbone."""

    # Backbone identity
    backbone: str = "FacebookAI/xlm-roberta-base"
    num_hidden_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    # Phase A: Ergodic layers (no position, narrow sliding window)
    num_ergodic_layers: int = 6
    ergodic_window_left: int = 64
    ergodic_window_right: int = 16
    ergodic_use_rope: bool = True
    num_ergodic_cross_attn_layers: int = 0  # Number of last Phase A layers with cross-attention

    # Phase B: Position-Aware layers (RoPE, wider sliding window, cross-attention)
    num_position_layers: int = 6
    position_window_left: int = 128
    position_window_right: int = 32
    position_use_rope: bool = True
    rope_theta: float = 10000.0

    # Streaming zones (used by engine, exposed here for attention module)
    active_window_size: int = 64
    frozen_cache_size: int = 128

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applied to Query and Key in attention to inject relative position
    information without absolute position embeddings.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands: freq_i = 1 / (theta ^ (2i / dim))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # [seq_len, dim]
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # [seq_len, dim]

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [seq_len] — position indices (can be non-contiguous)
        Returns:
            cos, sin: [1, 1, seq_len, head_dim] — ready for broadcasting
        """
        # Extend cache if needed
        max_pos = positions.max().item() + 1 if positions.numel() > 0 else 0
        if max_pos > self.cos_cached.shape[0]:
            self._build_cache(max_pos)
            # Move to correct device
            self.cos_cached = self.cos_cached.to(positions.device)
            self.sin_cached = self.sin_cached.to(positions.device)

        cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, S, dim]
        sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, S, dim]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension: [x1, x2] → [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to Q and K.

    Args:
        q, k: [B, H, S, D]
        cos, sin: [1, 1, S, D]
    Returns:
        q_rotated, k_rotated: [B, H, S, D]
    """
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated, k_rotated


# =============================================================================
# Sliding Window Attention
# =============================================================================


class SlidingWindowAttention(nn.Module):
    """
    Multi-Head Attention with Sliding Window.

    Supports:
    - RoPE (optional, only for Phase B layers)
    - KV Cache (for frozen context in streaming)
    - Configurable per-layer window sizes
    """

    def __init__(self, config: StreamingBackboneConfig, layer_idx: int, use_rope: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        self.dropout_p = config.attention_probs_dropout_prob

        # Q, K, V, O projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # RoPE (only for Phase B)
        if use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.active_window_size + config.frozen_cache_size + 256,
                theta=config.rope_theta,
            )

        # Per-layer sliding window sizes
        if layer_idx < config.num_ergodic_layers:
            self.window_left = config.ergodic_window_left
            self.window_right = config.ergodic_window_right
        else:
            self.window_left = config.position_window_left
            self.window_right = config.position_window_right

    def _create_sliding_window_mask(
        self, query_len: int, key_len: int, cache_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window attention mask.

        For active tokens (query), the mask allows:
        - Full attention to ALL cached (frozen) tokens
        - Sliding window attention within active tokens

        Returns: [query_len, key_len] boolean mask (True = attend, False = ignore)
        """
        mask = torch.zeros(query_len, key_len, dtype=torch.bool, device=device)

        # 1. Cached tokens: always fully visible to all queries
        if cache_len > 0:
            mask[:, :cache_len] = True

        # 2. Active tokens: sliding window within active region
        # active_offset maps query position i → key position (cache_len + i)
        for i in range(query_len):
            left = max(0, i - self.window_left) + cache_len
            right = min(query_len, i + self.window_right + 1) + cache_len
            mask[i, left:right] = True

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, S, D] — active tokens
            past_kv_cache: (K_cache, V_cache) each [B, H, cache_len, head_dim]
            position_offset: starting position for RoPE (for streaming continuity)

        Returns:
            output: [B, S, D]
            new_kv_cache: (K, V) each [B, H, S, head_dim] — for this step's active tokens
        """
        B, S, D = hidden_states.shape

        # 1. Compute Q, K, V
        Q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, H, S, head_dim]

        # 2. Apply RoPE to Q, K (if enabled)
        if self.use_rope:
            positions = torch.arange(
                position_offset, position_offset + S, device=hidden_states.device
            )
            cos, sin = self.rotary_emb(positions)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Save KV for this step (before merging with cache)
        new_kv_cache = (K, V)

        # 3. Merge with KV cache (frozen context)
        cache_len = 0
        if past_kv_cache is not None:
            K_cache, V_cache = past_kv_cache
            cache_len = K_cache.shape[2]
            K = torch.cat([K_cache, K], dim=2)  # [B, H, cache_len + S, head_dim]
            V = torch.cat([V_cache, V], dim=2)

        # 4. Create sliding window mask
        total_key_len = K.shape[2]
        sw_mask = self._create_sliding_window_mask(S, total_key_len, cache_len, hidden_states.device)
        # Expand to [B, H, S, total_key_len] for SDPA
        attn_mask = sw_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)

        # 5. Scaled Dot-Product Attention (flash attention when possible)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        # attn_output: [B, H, S, head_dim]

        # 6. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.o_proj(attn_output)

        return output, new_kv_cache


# =============================================================================
# Streaming RoBERTa Layer
# =============================================================================


class StreamingRobertaLayer(nn.Module):
    """
    A single RoBERTa layer modified for streaming.

    Components:
    - Self-Attention with Sliding Window (+ optional RoPE)
    - Cross-Attention with Audio (only in Phase B layers)
    - Feed-Forward Network

    Uses Pre-LN (layer norm before attention/FFN) for stable training.
    """

    def __init__(self, config: StreamingBackboneConfig, layer_idx: int):
        super().__init__()

        use_rope = config.position_use_rope if layer_idx >= config.num_ergodic_layers else config.ergodic_use_rope

        # Self-Attention
        self.self_attn = SlidingWindowAttention(config, layer_idx, use_rope=use_rope)
        self.self_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Cross-Attention with Audio
        is_phase_b = layer_idx >= config.num_ergodic_layers
        is_late_ergodic = (
            not is_phase_b 
            and layer_idx >= config.num_ergodic_layers - config.num_ergodic_cross_attn_layers
        )
        self.has_cross_attn = is_phase_b or is_late_ergodic
        
        if self.has_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attn_dropout = nn.Dropout(config.hidden_dropout_prob)
            # Near-zero init so residual initially passes through unchanged
            nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.01)
            if self.cross_attn.out_proj.bias is not None:
                nn.init.zeros_(self.cross_attn.out_proj.bias)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        audio_hidden: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, S, D]
            past_kv_cache: (K, V) from frozen zone
            audio_hidden: [B, A, D] audio features for cross-attention
            position_offset: RoPE position offset

        Returns:
            hidden_states: [B, S, D]
            new_kv_cache: (K, V) for this layer
        """
        # 1. Self-Attention + Residual + Norm (Post-LN like original RoBERTa)
        attn_out, new_kv = self.self_attn(
            hidden_states, past_kv_cache=past_kv_cache, position_offset=position_offset
        )
        hidden_states = self.self_attn_norm(hidden_states + self.self_attn_dropout(attn_out))

        # 2. Cross-Attention with Audio (Phase B only)
        if self.has_cross_attn and audio_hidden is not None:
            cross_out, _ = self.cross_attn(
                query=hidden_states,
                key=audio_hidden,
                value=audio_hidden,
            )
            hidden_states = self.cross_attn_norm(
                hidden_states + self.cross_attn_dropout(cross_out)
            )

        # 3. FFN + Residual + Norm
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_out)

        return hidden_states, new_kv


# =============================================================================
# Full Streaming Text Backbone
# =============================================================================


class StreamingDiffusionBackbone(nn.Module):
    """
    Streaming Diffusion Backbone — replaces standard XLM-RoBERTa.

    Architecture:
    - Word Embeddings (from XLM-R pretrained, NO position embeddings)
    - Phase A (Layers 1–6): Ergodic — translation-invariant, local patterns
    - Phase B (Layers 7–12): Position-Aware — RoPE + cross-attention + global semantics
    - LM Head → vocabulary logits
    """

    def __init__(self, config: StreamingBackboneConfig, vocab_size: int):
        super().__init__()
        self.config = config

        # Word Embeddings (NO position_embeddings — this is intentional)
        self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)
        self.embed_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([
            StreamingRobertaLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # LM Head (predict vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        audio_hidden: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: [B, S] — token IDs for active zone
            past_kv_caches: list of (K, V) per layer — from frozen zone
            audio_hidden: [B, A, D] — projected audio features
            position_offset: starting RoPE position for streaming continuity

        Returns:
            logits: [B, S, vocab_size]
            new_kv_caches: list of (K, V) per layer — for this forward pass
        """
        B, S = input_ids.shape

        # 1. Embeddings (NO position embeddings)
        h = self.word_embeddings(input_ids)
        h = self.embed_norm(h)
        h = self.embed_dropout(h)

        # 2. Forward through all layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches else None
            h, new_kv = layer(
                h,
                past_kv_cache=layer_cache,
                audio_hidden=audio_hidden,
                position_offset=position_offset,
            )
            new_kv_caches.append(new_kv)

        # 3. LM Head
        logits = self.lm_head(h)  # [B, S, vocab_size]

        return logits, new_kv_caches

    @classmethod
    def from_pretrained_xlmr(
        cls,
        config: StreamingBackboneConfig,
        vocab_size: int,
        cache_dir: str | None = None,
    ) -> "StreamingDiffusionBackbone":
        """
        Initialize from XLM-RoBERTa pretrained weights.

        Copies: word embeddings, self-attention Q/K/V/O, FFN, LayerNorms.
        Discards: position_embeddings (replaced by RoPE in Phase B).
        Initializes: cross-attention layers (new, near-zero init).
        """
        from transformers import XLMRobertaForMaskedLM

        xlmr = XLMRobertaForMaskedLM.from_pretrained(config.backbone, cache_dir=cache_dir)
        xlmr_roberta = xlmr.roberta
        xlmr_lm_head = xlmr.lm_head

        model = cls(config, vocab_size)

        # --- Copy Word Embeddings ---
        src_embed = xlmr_roberta.embeddings.word_embeddings.weight.data
        tgt_embed = model.word_embeddings.weight.data
        copy_size = min(src_embed.shape[0], tgt_embed.shape[0])
        tgt_embed[:copy_size] = src_embed[:copy_size].clone()

        # --- Copy Embed LayerNorm ---
        model.embed_norm.weight.data = xlmr_roberta.embeddings.LayerNorm.weight.data.clone()
        model.embed_norm.bias.data = xlmr_roberta.embeddings.LayerNorm.bias.data.clone()

        # --- Copy Layer Weights ---
        for i, (src_layer, dst_layer) in enumerate(
            zip(xlmr_roberta.encoder.layer, model.layers)
        ):
            # Self-attention Q, K, V, O
            dst_layer.self_attn.q_proj.weight.data = src_layer.attention.self.query.weight.data.clone()
            dst_layer.self_attn.q_proj.bias.data = src_layer.attention.self.query.bias.data.clone()
            dst_layer.self_attn.k_proj.weight.data = src_layer.attention.self.key.weight.data.clone()
            dst_layer.self_attn.k_proj.bias.data = src_layer.attention.self.key.bias.data.clone()
            dst_layer.self_attn.v_proj.weight.data = src_layer.attention.self.value.weight.data.clone()
            dst_layer.self_attn.v_proj.bias.data = src_layer.attention.self.value.bias.data.clone()
            dst_layer.self_attn.o_proj.weight.data = src_layer.attention.output.dense.weight.data.clone()
            dst_layer.self_attn.o_proj.bias.data = src_layer.attention.output.dense.bias.data.clone()

            # Self-attention LayerNorm
            dst_layer.self_attn_norm.weight.data = src_layer.attention.output.LayerNorm.weight.data.clone()
            dst_layer.self_attn_norm.bias.data = src_layer.attention.output.LayerNorm.bias.data.clone()

            # FFN: intermediate dense → output dense
            dst_layer.ffn[0].weight.data = src_layer.intermediate.dense.weight.data.clone()
            dst_layer.ffn[0].bias.data = src_layer.intermediate.dense.bias.data.clone()
            dst_layer.ffn[3].weight.data = src_layer.output.dense.weight.data.clone()
            dst_layer.ffn[3].bias.data = src_layer.output.dense.bias.data.clone()

            # FFN LayerNorm
            dst_layer.ffn_norm.weight.data = src_layer.output.LayerNorm.weight.data.clone()
            dst_layer.ffn_norm.bias.data = src_layer.output.LayerNorm.bias.data.clone()

            # Cross-attention layers (Phase B): already initialized near-zero in __init__

        # --- Copy LM Head ---
        # XLM-R lm_head: dense → layer_norm → decoder (tied to embeddings)
        # Our lm_head is a simple linear, so we copy the decoder weights
        src_decoder_weight = xlmr_lm_head.decoder.weight.data
        tgt_lm_weight = model.lm_head.weight.data
        copy_size = min(src_decoder_weight.shape[0], tgt_lm_weight.shape[0])
        tgt_lm_weight[:copy_size] = src_decoder_weight[:copy_size].clone()

        # ❌ Deliberately NOT copying: position_embeddings
        # ✅ Cross-attention layers: fresh near-zero init (handled in StreamingRobertaLayer.__init__)

        del xlmr  # Free memory
        return model
