import torch
import torch.nn as nn
from typing import Optional


class ResamplerCrossAttentionLayer(nn.Module):
    """
    Cross-attention layer where queries attend to key/values (audio features).
    """
    def __init__(
        self,
        hidden_size: int = 768,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Near-zero init so residual initially passes through unchanged / stable
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.01)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        audio_hidden: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        key_padding_mask = None
        if audio_mask is not None:
            key_padding_mask = (audio_mask == 0)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=audio_hidden,
            value=audio_hidden,
            key_padding_mask=key_padding_mask,
        )
        x = queries + self.cross_attn_dropout(attn_out)
        x = self.cross_attn_norm(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)
        return x


class AudioQueryResampler(nn.Module):
    """
    Q-Former style module to compress variable length audio features 
    into a fixed number of query tokens.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_queries: int = 32,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_size) * 0.02
        )
        self.layers = nn.ModuleList([
            ResamplerCrossAttentionLayer(
                hidden_size=hidden_size,
                nhead=nhead,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio_hidden: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_hidden: [B, T_audio, D]
            audio_mask: Optional [B, T_audio] (1 for valid, 0 for pad)
        Returns:
            compressed_audio: [B, num_queries, D]
        """
        B = audio_hidden.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, audio_hidden, audio_mask=audio_mask)

        return queries
