from transformers import AutoConfig, PretrainedConfig


class DiscreteDiffusionConfig(PretrainedConfig):
    model_type = "discrete_diffusion"

    def __init__(
        self,
        backbone_config=None,
        num_diffusion_timesteps=50,
        diffusion_type="absorbing",
        attention_strategy="full",
        vocab_pad_to_multiple=1,
        lora=False,
        lora_target_modules=["query", "value"],
        lora_alpha=16,
        lora_rank=16,
        lora_bias="none",
        lora_dropout=0,
        mask_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        argmax_decoding=True,  # Default to True for deterministic inference
        pretrained_audio_encoder=False,
        audio_fusion_strategy="prefix",
        # === Streaming-specific fields ===
        num_ergodic_layers=6,
        ergodic_window_left=64,
        ergodic_window_right=16,
        num_position_layers=6,
        position_window_left=128,
        position_window_right=32,
        rope_theta=10000.0,
        active_window_size=64,
        frozen_cache_size=128,
        streaming_denoise_steps=3,
        freeze_confidence_threshold=0.92,
        remask_confidence_threshold=0.30,
        loss_weight_supported=2.0,
        loss_weight_unsupported=0.3,
        use_confidence_calibration=True,
        audio_chunk_duration=2.0,
        audio_overlap_duration=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.diffusion_type = diffusion_type
        self.attention_strategy = attention_strategy
        self.vocab_pad_to_multiple = vocab_pad_to_multiple
        self.lora = lora
        self.lora_target_modules = lora_target_modules
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank
        self.lora_bias = lora_bias
        self.lora_dropout = lora_dropout
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.argmax_decoding = argmax_decoding
        self.pretrained_audio_encoder = pretrained_audio_encoder
        self.audio_fusion_strategy = audio_fusion_strategy

        # Streaming fields
        self.num_ergodic_layers = num_ergodic_layers
        self.ergodic_window_left = ergodic_window_left
        self.ergodic_window_right = ergodic_window_right
        self.num_position_layers = num_position_layers
        self.position_window_left = position_window_left
        self.position_window_right = position_window_right
        self.rope_theta = rope_theta
        self.active_window_size = active_window_size
        self.frozen_cache_size = frozen_cache_size
        self.streaming_denoise_steps = streaming_denoise_steps
        self.freeze_confidence_threshold = freeze_confidence_threshold
        self.remask_confidence_threshold = remask_confidence_threshold
        self.loss_weight_supported = loss_weight_supported
        self.loss_weight_unsupported = loss_weight_unsupported
        self.use_confidence_calibration = use_confidence_calibration
        self.audio_chunk_duration = audio_chunk_duration
        self.audio_overlap_duration = audio_overlap_duration

        if backbone_config is None:
            self.backbone_config = AutoConfig.from_pretrained(
                "FacebookAI/xlm-roberta-large"
            ).to_dict()
        elif isinstance(backbone_config, PretrainedConfig):
            self.backbone_config = backbone_config.to_dict()
        else:
            self.backbone_config = backbone_config

        # Expose backbone attributes
        self.hidden_size = self.backbone_config.get("hidden_size", 1024)
        self.num_attention_heads = self.backbone_config.get("num_attention_heads", 16)
        self.intermediate_size = self.backbone_config.get("intermediate_size", 4096)
        self.max_position_embeddings = self.backbone_config.get(
            "max_position_embeddings", 514
        )
        self.tie_word_embeddings = self.backbone_config.get("tie_word_embeddings", True)
