import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, AutoModelForMaskedLM, AutoConfig, Wav2Vec2Model
try:
    from .configuration_dlm import DiscreteDiffusionConfig
except ImportError:
    from configuration_dlm import DiscreteDiffusionConfig

from collections import namedtuple
import math
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings

decoder_out_t = namedtuple(
    "decoder_out_t",
    ["output_tokens", "output_scores", "output_masks", "non_fixed_sym_masks", "attn", "step", "max_step", "history"],
)

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking

class DiscreteDiffusionModel(PreTrainedModel):
    config_class = DiscreteDiffusionConfig
    _keys_to_ignore_on_load_missing = ["fake_layer", "length_trm", "length_predictor", "model.lm_head.decoder.weight", "model.lm_head.decoder.bias"]
    _tied_weights_keys = {"model.lm_head.decoder.weight": "model.roberta.embeddings.word_embeddings.weight"}

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__(config)
        self.config = config
        self.args = config # Alias for compatibility with existing code
        self.all_tied_weights_keys = {
            "model.lm_head.decoder.weight": "model.roberta.embeddings.word_embeddings.weight"
        }

        # Initialize backbone
        if config.backbone_config:
            # We assume backbone_config is a dict
            backbone_config_obj = AutoConfig.for_model(**config.backbone_config)
            self.model = AutoModelForMaskedLM.from_config(backbone_config_obj)
        else:
             # Fallback or error
             raise ValueError("backbone_config must be provided in config")

        if config.tie_word_embeddings:
             self.model.lm_head.decoder.weight = self.model.roberta.embeddings.word_embeddings.weight

        self.mask_id = config.mask_token_id
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id
        self.pad_id = config.pad_token_id
        
        # Lora
        if config.lora:
            self.add_fake_layer()

        # Audio encoder (for speech_recognition, speech_translation and speech_translation_multitask dataset_type)
        self.has_audio_encoder = getattr(config, 'dataset_type', 'bilingual') in ['speech_recognition', 'speech_translation', 'speech_translation_multitask']
        if self.has_audio_encoder:
            audio_encoder_name = getattr(config, 'audio_encoder_name', 'facebook/mms-300m')
            pretrained_audio_encoder = getattr(config, 'pretrained_audio_encoder', False)
            
            # Check if we are inside accelerate's init_empty_weights context manager.
            # If so, temporarily restore the original register_parameter method so the audio encoder
            # is loaded on CPU/GPU with its actual weights instead of being an empty meta tensor.
            is_patched = hasattr(nn.Module.register_parameter, "__code__") and "register_empty_parameter" in nn.Module.register_parameter.__code__.co_name
            original_register = None
            if is_patched:
                closure = nn.Module.register_parameter.__closure__
                if closure is not None:
                    for cell in closure:
                        val = cell.cell_contents
                        if callable(val) and val.__name__ == "register_parameter":
                            original_register = val
                            break
            
            with torch.device("cpu"):
                def _init_audio_encoder():
                    if "moonshine" in audio_encoder_name:
                        from transformers import MoonshineStreamingModel
                        if pretrained_audio_encoder:
                            moonshine_model = MoonshineStreamingModel.from_pretrained(
                                audio_encoder_name, cache_dir=getattr(config, 'cache_dir', None)
                            )
                        else:
                            from transformers import AutoConfig
                            moonshine_config = AutoConfig.from_pretrained(
                                audio_encoder_name, cache_dir=getattr(config, 'cache_dir', None)
                            )
                            moonshine_model = MoonshineStreamingModel(moonshine_config)
                        return moonshine_model.encoder
                    else:
                        if pretrained_audio_encoder:
                            return Wav2Vec2Model.from_pretrained(
                                audio_encoder_name, cache_dir=getattr(config, 'cache_dir', None)
                            )
                        else:
                            from transformers import AutoConfig
                            wav2vec2_config = AutoConfig.from_pretrained(
                                audio_encoder_name, cache_dir=getattr(config, 'cache_dir', None)
                            )
                            return Wav2Vec2Model(wav2vec2_config)

                if original_register is not None:
                    old_patched = nn.Module.register_parameter
                    nn.Module.register_parameter = original_register
                    try:
                        self.audio_encoder = _init_audio_encoder()
                    finally:
                        nn.Module.register_parameter = old_patched
                else:
                    self.audio_encoder = _init_audio_encoder()
            
            # Freeze the audio encoder
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            
            audio_hidden_size = self.audio_encoder.config.hidden_size
            self.audio_projector = nn.Linear(audio_hidden_size, self.config.hidden_size)
            
            # Cross-attention layer
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=0.1,
                batch_first=True
            )
            self.cross_attn_ln = nn.LayerNorm(self.config.hidden_size)

        # Length predictor (optional, as in original code)
        self.length_trm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size, 
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                batch_first=True
            ),
            num_layers=1,
        )
        self.length_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_size , self.config.intermediate_size),
            nn.Tanh(),
            nn.Linear(self.config.intermediate_size, self.config.max_position_embeddings)
        )

    def add_fake_layer(self):
        self.fake_layer = nn.Parameter(torch.zeros((self.config.hidden_size, )))

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def _tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(
                self.model.lm_head.decoder,
                self.model.roberta.embeddings.word_embeddings
            )

    def _init_weights(self, module):
        """Initialize the weights - called after loading checkpoint."""
        # Call parent init_weights
        super()._init_weights(module)
        # Ensure weights are tied after initialization
        self._tie_weights()



    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # ... copy from DiscreteDiffusionBase ...
        assert self.config.diffusion_type == "absorbing", "we only support absorbing diffusion temporarily"
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()
        
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.config.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float) 
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.config.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)
        
        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }

    def initialize_decode_samples(self, tokens, partial_masks, prefix_masks, oracle_length=False, length_beam=1, mbr=1):
        # ... copy from DiscreteDiffusionBase ...
        if tokens is None:
            raise NotImplementedError
        else:
            if not oracle_length:
                inputs_tokens = tokens.masked_fill(~prefix_masks, self.pad_id)
                src_length = inputs_tokens.ne(self.pad_id).sum(dim=-1)
                inputs_tokens = inputs_tokens[:, :src_length.max()]
                length_logits = self.forward_length(inputs_tokens)
                # Giới hạn độ dài output tối đa: không quá 3x độ dài source và không quá 100 tokens
                max_allowed_length = torch.min(
                    torch.tensor([100]).to(src_length.device),
                    (src_length * 3)[:, None]
                )
                length = (
                    torch.min(
                        torch.min(
                            length_logits.topk(length_beam, dim=-1).indices + 1,
                            max_allowed_length
                        ),
                        self.config.max_position_embeddings - 2 - src_length[:, None] - 1
                    )
                )
                output_tokens = []
                new_partial_masks = []
                for i, token in enumerate(inputs_tokens):
                    for b in range(length_beam):
                        for m in range(mbr):
                            # Create output token sequence
                            seq = torch.cat([
                                token[:src_length[i]], 
                                torch.tensor([self.mask_id] * length[i][b] + [self.eos_id]).to(token)
                            ])
                            output_tokens.append(seq)
                            
                            # Create corresponding partial mask
                            # True for fixed (source), False for generated (mask/eos)
                            # partial_masks[i] corresponds to token[i]
                            # We assume partial_masks[i] has same length as token[i] (or at least src_length[i])
                            p_mask = torch.cat([
                                partial_masks[i][:src_length[i]],
                                torch.tensor([False] * (length[i][b] + 1)).to(partial_masks)
                            ])
                            new_partial_masks.append(p_mask)
                            
                output_tokens = pad_sequence(output_tokens, batch_first=True, padding_value=self.pad_id)
                # Pad partial masks to match output_tokens length
                # We need to pad with True (fixed) or False (maskable)?
                # Usually padding tokens should be ignored. 
                # In finalized_hypos: cutoff = tokens.ne(pad) & ... & (~partial_mask)
                # If we pad partial_mask with True, ~partial_mask is False, so it's filtered out.
                # If we pad with False, ~partial_mask is True, so it's kept (if not pad_id).
                # Since we check tokens.ne(pad_id), padding tokens are filtered anyway.
                # But for safety, let's pad with True (fixed) so they are treated as non-generated?
                # Actually, pad_sequence pads with 0. For bool tensor, 0 is False.
                # So if we use pad_sequence on bool tensor, it pads with False.
                partial_masks = pad_sequence(new_partial_masks, batch_first=True, padding_value=True) # Pad with True to be safe?
                # Wait, if we pad with True, then ~partial_mask is False.
                
                output_mask = output_tokens.eq(self.mask_id)
                # non_fixed_sym_masks should be all positions that can be modified (not source, not pad, not special tokens)
                # This is critical for _reparam_decoding to work correctly!
                non_fixed_sym_masks = (
                    output_tokens.ne(self.pad_id) &
                    output_tokens.ne(self.bos_id) &
                    ~partial_masks  # Not source tokens
                )
            else:
                output_tokens = torch.stack([token for token in tokens for m in range(mbr)])
                partial_masks = torch.stack([mask for mask in partial_masks for m in range(mbr)])
                prefix_masks = torch.stack([mask for mask in prefix_masks for m in range(mbr)])
                output_mask = (
                    output_tokens.ne(self.pad_id) &
                    output_tokens.ne(self.bos_id) &
                    output_tokens.ne(self.eos_id) &
                    ~prefix_masks
                )
                output_tokens = output_tokens.masked_fill(output_mask, self.mask_id)
                non_fixed_sym_masks = output_mask.clone()
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
            
            return partial_masks, decoder_out_t(
                output_tokens=output_tokens,
                output_scores=output_scores,
                output_masks=output_mask,
                non_fixed_sym_masks=non_fixed_sym_masks,
                attn=None,
                step=0,
                max_step=math.inf,
                history=None
            )

    def forward_length(self, input_ids):
        attention_mask = input_ids.ne(self.pad_id).int()
        with torch.no_grad():
            _feature = self.model.roberta(input_ids, attention_mask=attention_mask)[0]
        feature = self.length_trm(_feature, src_key_padding_mask=(1-attention_mask).bool())
        length = attention_mask.sum(dim=-1)
        pooled_feature = feature.masked_fill((attention_mask==0)[:, :, None], 0).float().sum(1) / length[:, None]
        length_logits = self.length_predictor(pooled_feature.to(feature))
        return length_logits

    def forward(self, prev_output_tokens, partial_mask, attention_mask=None, loss_mask=None, cache=None,
                audio_features=None, audio_attention_mask=None, precomputed_audio_embeds=None,
                precomputed_audio_mask=None):
        input_ids = prev_output_tokens
        if attention_mask is None:
            attention_mask = prev_output_tokens.ne(self.pad_id).int()        
        
        # Build full embeddings first (word + position + token type + LayerNorm + dropout)
        embeddings = self.model.roberta.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        
        if hasattr(self, "fake_layer") and self.training:
            self.fake_layer.requires_grad = True
            embeddings = embeddings + self.fake_layer * 0 
        
        # Audio fusion via Prefix Conditioning (Sequence Concatenation)
        if self.has_audio_encoder and precomputed_audio_embeds is not None:
            # Fast path: use pre-computed encoder output, only apply trainable projector
            audio_embeds = self.audio_projector(precomputed_audio_embeds)  # (B, T_audio, hidden_size)
            T_audio = audio_embeds.size(1)
            
            if precomputed_audio_mask is not None:
                audio_attn = precomputed_audio_mask.int()
            else:
                audio_attn = torch.ones(audio_embeds.size(0), T_audio, dtype=torch.int, device=audio_embeds.device)
            
            embeddings = torch.cat([audio_embeds, embeddings], dim=1)
            combined_attention_mask = torch.cat([audio_attn, attention_mask], dim=1)
        elif self.has_audio_encoder and audio_features is not None:
            with torch.no_grad():
                audio_outputs = self.audio_encoder(
                    audio_features,
                    attention_mask=audio_attention_mask
                )
                audio_embeds = audio_outputs.last_hidden_state  # (B, T_audio, D_audio)
            audio_embeds = self.audio_projector(audio_embeds)    # (B, T_audio, hidden_size)
            T_audio = audio_embeds.size(1)
            
            # Create audio attention mask
            if audio_attention_mask is not None:
                if hasattr(audio_outputs, "attention_mask") and audio_outputs.attention_mask is not None:
                    audio_attn = audio_outputs.attention_mask.int()
                else:
                    audio_attn = self.audio_encoder._get_feature_vector_attention_mask(
                        T_audio, audio_attention_mask
                    ).int()
            else:
                audio_attn = torch.ones(audio_embeds.size(0), T_audio, dtype=torch.int, device=audio_embeds.device)
            
            # Concatenate audio features and text embeddings along sequence dimension
            embeddings = torch.cat([audio_embeds, embeddings], dim=1)
            combined_attention_mask = torch.cat([audio_attn, attention_mask], dim=1)
        else:
            combined_attention_mask = attention_mask
            T_audio = 0
        
        if self.config.attention_strategy == "prefix_lm":
            # prefix_lm is only supported without audio or needs custom mapping.
            # If there is audio, warn and fall back to full attention.
            if T_audio > 0:
                if not hasattr(self, "_warned_prefix_lm"):
                    print("Warning: prefix_lm attention strategy is not fully supported with audio input. Falling back to full attention.")
                    self._warned_prefix_lm = True
                attention_mask_converted, _ = self.model.roberta._create_attention_masks(
                    attention_mask=combined_attention_mask,
                    encoder_attention_mask=None,
                    embedding_output=embeddings,
                    encoder_hidden_states=None,
                    past_key_values=None,
                )
            else:
                ext_partial_mask = partial_mask.float()
                ext_partial_mask = torch.bmm(ext_partial_mask[:, :, None], ext_partial_mask[:, None, :]).int()
                ext_mask = attention_mask[:, None, :].repeat(1, attention_mask.size(-1), 1)
                ext_mask[partial_mask] = ext_partial_mask[partial_mask]
                
                # Convert 3D mask using _create_attention_masks
                attention_mask_converted, _ = self.model.roberta._create_attention_masks(
                    attention_mask=ext_mask,
                    encoder_attention_mask=None,
                    embedding_output=embeddings,
                    encoder_hidden_states=None,
                    past_key_values=None,
                )
        else:
            # Convert 2D mask using _create_attention_masks
            attention_mask_converted, _ = self.model.roberta._create_attention_masks(
                attention_mask=combined_attention_mask,
                encoder_attention_mask=None,
                embedding_output=embeddings,
                encoder_hidden_states=None,
                past_key_values=None,
            )
            
        # Call the encoder directly, bypassing self.model.roberta's embedding layer (which would double-embed)
        encoder_outputs = self.model.roberta.encoder(
            embeddings,
            attention_mask=attention_mask_converted,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
        )
        outputs = encoder_outputs.last_hidden_state
        
        # Extract the text portion of the encoder outputs
        if T_audio > 0:
            outputs = outputs[:, T_audio:]
        
        if not (~torch.isnan(outputs)).all():
            outputs.masked_fill_(outputs.isnan(), 0)
        
        outputs = outputs[loss_mask] if loss_mask is not None else outputs
        return self.model.lm_head(outputs)

    def _reparam_decoding(
        self, 
        output_tokens, 
        output_scores, 
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0, 
        non_special_sym_mask, 
        t,
        max_step,
        noise
    ):
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
            ).long()
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError
        
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError
        
        not_v2_t = lowest_k_mask

        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt_neq_x0

    def denoise_step(self, decoder_out, partial_masks, temperature=1.0, strategy="reparam-uncond-deterministic-cosine",
                     audio_features=None, audio_attention_mask=None):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        prev_step, cur_step = decoder_out.step, decoder_out.step + 1 
        max_step = decoder_out.max_step
        
        logits = self.forward(output_tokens, partial_masks,
                              audio_features=audio_features, audio_attention_mask=audio_attention_mask)
        
        logits[..., self.mask_id] = -math.inf
        scores = torch.log_softmax(logits, dim=-1)
        
        if strategy == "cmlm":
            # get the mask
            # <bos>, <eos> are ignored in this case since
            # they are not equal to unk.
            output_masks = output_tokens.eq(self.mask_id)
            unmask_prob = 1 / (max_step - prev_step)
            # where to unmask
            changes = torch.rand(output_tokens.shape, device=output_tokens.device) < unmask_prob
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_and(changes, output_masks)

            if getattr(self.config, "argmax_decoding", False):
                output_scores, new_tokens = scores.max(-1)
            else:
                # Assuming dists is imported or available, otherwise use torch.multinomial or similar
                # But let's stick to what was in generator if possible, or implement simple sampling
                # The generator used: dists.Categorical(logits=scores / temperature).sample()
                # We need to import dists or use torch.distributions
                import torch.distributions as dists
                new_tokens = dists.Categorical(logits=scores / temperature).sample()
                output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            output_tokens[changes] = new_tokens[changes]
        elif strategy == "ar":
            output_masks = output_tokens.eq(self.mask_id)
            unmask_indices = (output_tokens.ne(self.mask_id) & output_tokens.ne(self.eos_id) & output_tokens.ne(self.pad_id)).sum(dim=-1)
            indices = torch.arange(output_tokens.size(-1)).expand(output_tokens.shape).to(output_masks.device)
            if getattr(self.config, "argmax_decoding", False):
                output_scores, new_tokens = scores.max(-1)
            else:
                import torch.distributions as dists
                new_tokens = dists.Categorical(logits=scores / temperature).sample()
                output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            output_tokens[unmask_indices[:, None]==indices] = new_tokens[unmask_indices[:, None]==indices]
        else:
            if getattr(self.config, "argmax_decoding", False):
                cur_scores, cur_tokens = scores.max(-1)
            else:
                import torch.distributions as dists
                cur_tokens = dists.Categorical(logits=scores / temperature).sample()
                cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
            cur_scores = cur_scores.to(output_scores)
            
            output_masks = self._reparam_decoding(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                decoding_strategy=strategy,
                xt_neq_x0=decoder_out.output_masks,
                non_special_sym_mask=decoder_out.non_fixed_sym_masks,
                t=cur_step,
                max_step=max_step,
                noise=self.mask_id
            )
        
        history = (
            ([] if decoder_out.history is None else decoder_out.history) + [output_tokens.clone()]
            if decoder_out.history is not None else None
        )
        
        return decoder_out._replace(
            step=cur_step,
            output_tokens=output_tokens,
            output_scores=output_scores,
            output_masks=output_masks,
            history=history,
        )

    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        attention_mask=None, 
        max_iterations=10, 
        strategy="reparam-uncond-deterministic-cosine",
        temperature=1.0,
        return_history=False,
        max_length=128,  # Fixed generation length hyperparameter (like LLaDA)
        **kwargs
    ):
        # Prepare inputs
        src_tokens = input_ids
        
        if attention_mask is None:
            partial_masks = torch.ones_like(src_tokens).bool()
        else:
            partial_masks = attention_mask.bool()
            
        prefix_masks = partial_masks 
        
        # Initialize canvas with fixed length (LLaDA approach)
        # Instead of predicting length, use max_length as hyperparameter
        batch_size = src_tokens.size(0)
        src_length = src_tokens.ne(self.pad_id).sum(dim=-1)
        
        # Create fully masked response of fixed length
        output_tokens = []
        new_partial_masks = []
        
        for i in range(batch_size):
            # Format: <source_without_eos> <mask>...<mask> <eos>
            # Remove EOS from source if it exists
            src_len = src_length[i].item()
            src_seq = src_tokens[i, :src_len]
            
            # Remove trailing EOS from source
            if src_seq[-1] == self.eos_id:
                src_seq = src_seq[:-1]
                src_len -= 1
            
            seq = torch.cat([
                src_seq,
                torch.full((max_length,), self.mask_id, dtype=src_tokens.dtype, device=src_tokens.device),
                torch.tensor([self.eos_id], dtype=src_tokens.dtype, device=src_tokens.device)
            ])
            output_tokens.append(seq)
            
            # Mask: True for source (fixed), False for generated part
            mask = torch.cat([
                torch.ones(src_len, dtype=torch.bool, device=src_tokens.device),
                torch.zeros(max_length + 1, dtype=torch.bool, device=src_tokens.device)  # +1 for eos
            ])
            new_partial_masks.append(mask)
        
        output_tokens = pad_sequence(output_tokens, batch_first=True, padding_value=self.pad_id)
        partial_masks = pad_sequence(new_partial_masks, batch_first=True, padding_value=True)
        
        # Create masks for decoding
        output_mask = output_tokens.eq(self.mask_id)
        non_fixed_sym_masks = (
            output_tokens.ne(self.pad_id) &
            output_tokens.ne(self.bos_id) &
            output_tokens.ne(self.eos_id) &
            ~partial_masks  # Not source tokens
        )
        
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
        
        prev_decoder_out = decoder_out_t(
            output_tokens=output_tokens,
            output_scores=output_scores,
            output_masks=output_mask,
            non_fixed_sym_masks=non_fixed_sym_masks,
            attn=None,
            step=0,
            max_step=max_iterations,
            history=None
        )
        
        if return_history:
            prev_decoder_out = prev_decoder_out._replace(history=[])
        
        for step in range(max_iterations):
            prev_decoder_out = self.denoise_step(
                prev_decoder_out, partial_masks, temperature=temperature, strategy=strategy,
                audio_features=kwargs.get('audio_features'), audio_attention_mask=kwargs.get('audio_attention_mask')
            )            
            
        # Finalize: discard tokens after EOS (LLaDA approach)
        def finalized_hypos(tokens, scores, partial_mask, history=None):
            # First, find EOS position and cut there
            eos_positions = (tokens == self.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # Cut everything after EOS
                tokens = tokens[:first_eos]  # Exclude EOS
                if scores is not None:
                    scores = scores[:first_eos]
                partial_mask = partial_mask[:first_eos]
            
            # Then apply cutoff logic: keep only generated tokens (not source, not special)
            cutoff = (
                tokens.ne(self.pad_id) & 
                tokens.ne(self.bos_id) & 
                tokens.ne(self.eos_id) & 
                (~partial_mask)  # Not source tokens (partial_mask=False for generated)
            )
            tokens = tokens[cutoff]
            if scores is None:
                score = None
            else:
                scores = scores[cutoff]
                score = scores.mean().item() if len(scores) > 0 else 0.0
            ret_dict = {
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "alignment": None
            }
            if history is not None:
                ret_dict["history"] = [
                    finalized_hypos(history_tokens, None, partial_mask, history=None)
                    for history_tokens in history
                ]
            return ret_dict
        
        def score_select(hyps):
            index = np.argmax([hyp["score"] for hyp in hyps])
            return hyps[index]
        
        output_tokens, output_scores = prev_decoder_out.output_tokens, prev_decoder_out.output_scores
        
        # Handle history if needed
        if return_history and prev_decoder_out.history is not None:
            full_history = prev_decoder_out.history 
            histories = [[full_history[j][i] for j in range(max_iterations)] for i in range(output_tokens.size(0))]
            hyps = []
            for tokens, scores, partial_mask, history in zip(output_tokens, output_scores, partial_masks, histories):
                hyps.append(finalized_hypos(tokens, scores, partial_mask, history))
        else:
            hyps = [
                finalized_hypos(tokens, scores, partial_mask, None) 
                for tokens, scores, partial_mask in zip(output_tokens, output_scores, partial_masks)
            ]
            
        repeatition = kwargs.get("mbr", 1) * kwargs.get("length_beam", 1)
        if repeatition > 1:
            hyps = [score_select(hyps[i:i+repeatition]) for i in range(0, len(hyps), repeatition)]
            
        finalized = pad_sequence([h["tokens"] for h in hyps ], batch_first=True, padding_value=self.pad_id)
        
        # If the user expects just tokens, we return finalized tokens.
        # The original model.generate returned just tokens.
        return finalized
