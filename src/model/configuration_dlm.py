"""Configuration classes for discrete diffusion models."""

from transformers import AutoConfig, PretrainedConfig


class DiscreteDiffusionConfig(PretrainedConfig):
    """Configuration class to store the configuration of a DiscreteDiffusionModel."""

    model_type = "discrete_diffusion"

    def __init__(
        self,
        backbone_config=None,
        num_diffusion_timesteps=50,
        diffusion_type="absorbing",
        attention_strategy="full",
        vocab_pad_to_multiple=1,
        lora=False,
        lora_target_modules=None,
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
        """Initialize DiscreteDiffusionConfig.

        Args:
            backbone_config: Backbone model configuration.
            num_diffusion_timesteps: Total number of diffusion timesteps.
            diffusion_type: Type of diffusion process.
            attention_strategy: Attention strategy ('full', 'local', etc.).
            vocab_pad_to_multiple: Pad vocabulary size to multiple of this.
            lora: Whether to use LoRA fine-tuning.
            lora_target_modules: Module names to apply LoRA to.
            lora_alpha: Scaling factor for LoRA.
            lora_rank: Rank of LoRA update matrices.
            lora_bias: LoRA bias configuration.
            lora_dropout: Dropout probability for LoRA layers.
            mask_token_id: Mask token identifier.
            bos_token_id: Beginning-of-sequence token ID.
            eos_token_id: End-of-sequence token ID.
            pad_token_id: Padding token ID.
            argmax_decoding: Whether to use argmax decoding.
            pretrained_audio_encoder: Whether audio encoder is pretrained.
            audio_fusion_strategy: Strategy for audio conditioning fusion.
            num_ergodic_layers: Number of ergodic layers in streaming backbone.
            ergodic_window_left: Left context window for ergodic attention.
            ergodic_window_right: Right context window for ergodic attention.
            num_position_layers: Number of position-aware layers.
            position_window_left: Left context window for position-aware attention.
            position_window_right: Right context window for position-aware attention.
            rope_theta: Base period for rotary embeddings.
            active_window_size: Size of active streaming window.
            frozen_cache_size: Size of frozen context cache.
            streaming_denoise_steps: Denoising steps per chunk in streaming.
            freeze_confidence_threshold: Threshold to freeze tokens.
            remask_confidence_threshold: Threshold to remask low-confidence tokens.
            loss_weight_supported: Loss multiplier for supported tokens.
            loss_weight_unsupported: Loss multiplier for unsupported tokens.
            use_confidence_calibration: Whether to calibrate confidence.
            audio_chunk_duration: Audio chunk duration in seconds.
            audio_overlap_duration: Audio overlap duration in seconds.
            **kwargs: Additional keyword arguments passed to PretrainedConfig.
        """
        super().__init__(**kwargs)
        self.backbone_config = backbone_config
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.diffusion_type = diffusion_type
        self.attention_strategy = attention_strategy
        self.vocab_pad_to_multiple = vocab_pad_to_multiple
        self.lora = lora
        self.lora_target_modules = (
            lora_target_modules
            if lora_target_modules is not None
            else ["query", "value"]
        )
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
