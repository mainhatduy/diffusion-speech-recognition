"""
Streaming Length Predictor.

Predicts the number of new text tokens to generate for each audio chunk.

Unlike the existing length predictor (which predicts total output length for
the entire audio), this module predicts per-chunk token counts:
    audio_chunk → how many text tokens does this chunk correspond to?

Architecture:
    audio_pooling → context_encoding → fusion → classification (0..max_tokens)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingLengthPredictor(nn.Module):
    """
    Predicts number of new text tokens for each audio chunk.

    Input:
        - audio_embeds: [B, frames, D] — projected audio features for current chunk
        - context_token_ids: [B, context_len] — recently frozen token IDs (optional)

    Output:
        - logits: [B, max_output_tokens + 1] — classification over {0, 1, ..., max_tokens}
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 250064,
        max_output_tokens: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_output_tokens = max_output_tokens

        # Audio pooling
        self.audio_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Context encoding (1-layer Transformer over recent frozen tokens)
        self.context_embed = nn.Embedding(vocab_size, hidden_size)
        self.context_pool = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=1,
        )

        # Fusion → Prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max_output_tokens + 1),
        )

    def forward(
        self,
        audio_embeds: torch.Tensor,
        context_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_embeds: [B, frames, D]
            context_token_ids: [B, context_len] (optional)

        Returns:
            logits: [B, max_output_tokens + 1]
        """
        # Pool audio features
        audio_feat = self.audio_pool(audio_embeds).mean(dim=1)  # [B, D]

        # Pool context tokens
        if context_token_ids is not None and context_token_ids.shape[1] > 0:
            ctx_embed = self.context_embed(context_token_ids)  # [B, ctx_len, D]
            ctx_feat = self.context_pool(ctx_embed).mean(dim=1)  # [B, D]
        else:
            ctx_feat = torch.zeros_like(audio_feat)

        # Fuse and predict
        combined = torch.cat([audio_feat, ctx_feat], dim=-1)  # [B, 2D]
        logits = self.predictor(combined)  # [B, max_tokens + 1]

        return logits

    @torch.no_grad()
    def predict_chunk(
        self,
        audio_embeds: torch.Tensor,
        context_token_ids: torch.Tensor | None = None,
    ) -> int:
        """
        Inference: predict number of tokens for a single chunk.

        Args:
            audio_embeds: [1, frames, D] or [frames, D]
            context_token_ids: [1, context_len] (optional)

        Returns:
            int: predicted number of new tokens
        """
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)

        logits = self.forward(audio_embeds, context_token_ids)
        predicted = logits.argmax(dim=-1).item()
        return predicted
