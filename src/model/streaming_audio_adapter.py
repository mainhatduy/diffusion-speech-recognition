"""
Streaming Audio Adapter.

Bridges the gap between:
- Audio Encoder (Moonshine, ergodic, NO position information)
- Text Backbone (position-aware in upper layers)

This module:
1. Projects audio features from D_audio (512) to D_text (768)
2. Adds LEARNED position embeddings (per-chunk relative, reset each chunk)
3. Optionally downsamples 2x via Conv1d (100 frames → 50 tokens/chunk)
4. Normalizes output

The position embeddings here are the ONLY positional signal injected into
audio tokens. Moonshine produces translation-invariant features; this adapter
"tags" them with within-chunk positions before they enter the text backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .streaming_backbone import StreamingBackboneConfig


class StreamingAudioAdapter(nn.Module):
    """
    Adapter between frozen Audio Encoder (ergodic) and streaming Text Backbone (position-aware).

    Input:  audio_hidden [B, num_frames, D_audio] — no position info
    Output: audio_projected [B, num_tokens, D_text] — WITH position info, ready for cross-attention
    """

    def __init__(
        self,
        audio_hidden_size: int = 512,
        text_hidden_size: int = 768,
        max_frames_per_chunk: int = 256,
        use_downsample: bool = True,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.audio_hidden_size = audio_hidden_size
        self.text_hidden_size = text_hidden_size
        self.use_downsample = use_downsample

        # 1. Linear projection: D_audio → D_text
        self.proj = nn.Linear(audio_hidden_size, text_hidden_size)
        # Small init to avoid disrupting pretrained text backbone at start
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # 2. Learned position embedding (per-chunk relative)
        #    Each chunk resets positions to [0, 1, ..., num_frames-1]
        self.position_embeddings = nn.Embedding(max_frames_per_chunk, text_hidden_size)

        # 3. Optional Conv1d downsample (2x reduction)
        #    ~100 frames/chunk → ~50 tokens/chunk
        if use_downsample:
            self.downsample = nn.Conv1d(
                in_channels=text_hidden_size,
                out_channels=text_hidden_size,
                kernel_size=2,
                stride=2,
            )

        # 4. LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(text_hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_hidden: [B, T, D_audio] — from Moonshine encoder (no position)

        Returns:
            audio_projected: [B, T', D_text] — with position, normalized
                             T' = T // 2 if downsample, else T
        """
        B, T, D = audio_hidden.shape

        # 1. Project to text hidden size
        x = self.proj(audio_hidden)  # [B, T, D_text]

        # 2. Add per-chunk position embeddings
        positions = torch.arange(T, device=x.device)  # [0, 1, ..., T-1]
        pos_embed = self.position_embeddings(positions)  # [T, D_text]
        x = x + pos_embed.unsqueeze(0)  # broadcast → [B, T, D_text]

        # 3. Optional 2x downsample
        if self.use_downsample and T > 1:
            x = x.transpose(1, 2)  # [B, D_text, T]
            x = self.downsample(x)  # [B, D_text, T//2]
            x = x.transpose(1, 2)  # [B, T//2, D_text]

        # 4. Normalize + dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x  # [B, T', D_text]

    @classmethod
    def from_config(cls, config: StreamingBackboneConfig, audio_hidden_size: int = 512) -> "StreamingAudioAdapter":
        """Create adapter from backbone config."""
        return cls(
            audio_hidden_size=audio_hidden_size,
            text_hidden_size=config.hidden_size,
            max_frames_per_chunk=256,
            use_downsample=True,
            dropout=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
        )
