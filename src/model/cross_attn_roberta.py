import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEncoder
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class CrossAttnRobertaLayer(RobertaLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        
        # New Cross-Attention block
        hidden_size = getattr(config, "hidden_size", 768)
        num_heads = getattr(config, "num_attention_heads", 12)
        dropout = getattr(config, "attention_probs_dropout_prob", 0.1)
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-12)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Near-zero initialization on cross-attention output projection
        # to ensure the residual connection initially passes through unchanged
        if hasattr(self.cross_attention, "out_proj"):
            self.cross_attention.out_proj.weight.data.mul_(0.01)
            if self.cross_attention.out_proj.bias is not None:
                self.cross_attention.out_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Self-Attention
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        attention_output = self_attention_output

        # 2. Cross-Attention (NEW)
        # We check encoder_hidden_states which acts as our audio hidden states
        if encoder_hidden_states is not None:
            # Prepare key_padding_mask
            key_padding_mask = None
            if encoder_attention_mask is not None:
                # If float mask (HuggingFace converts masks to large negative values for padding)
                if encoder_attention_mask.dtype.is_floating_point:
                    key_padding_mask = (encoder_attention_mask < -1.0)
                else:
                    # Integer/Boolean mask: 1/True for active, 0/False for pad
                    # MultiheadAttention expects True for padding (ignored)
                    key_padding_mask = (encoder_attention_mask == 0)
            
            # Cross-Attention
            ca_out, _ = self.cross_attention(
                query=attention_output,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=key_padding_mask,
            )
            attention_output = self.cross_attn_layer_norm(attention_output + ca_out)

        # 3. FFN (chunked feed-forward)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

class CrossAttnRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Re-initialize the layers as CrossAttnRobertaLayer
        self.layer = nn.ModuleList([
            CrossAttnRobertaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
