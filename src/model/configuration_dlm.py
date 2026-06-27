from transformers import PretrainedConfig, AutoConfig

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
        **kwargs
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

        if backbone_config is None:
            self.backbone_config = AutoConfig.from_pretrained("FacebookAI/xlm-roberta-large").to_dict()
        elif isinstance(backbone_config, PretrainedConfig):
            self.backbone_config = backbone_config.to_dict()
        else:
            self.backbone_config = backbone_config
            
        # Expose backbone attributes
        self.hidden_size = self.backbone_config.get("hidden_size", 1024)
        self.num_attention_heads = self.backbone_config.get("num_attention_heads", 16)
        self.intermediate_size = self.backbone_config.get("intermediate_size", 4096)
        self.max_position_embeddings = self.backbone_config.get("max_position_embeddings", 514)
        self.tie_word_embeddings = self.backbone_config.get("tie_word_embeddings", True)

