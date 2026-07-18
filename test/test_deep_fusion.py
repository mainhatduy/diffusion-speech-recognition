import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel
from model.dd_model import DiscreteDiffusionXLMRModel, DiscreteDiffusionModelArguments
from model.cross_attn_roberta import CrossAttnRobertaLayer
from transformers import RobertaConfig, AutoModelForMaskedLM

class MockAudioConfig:
    def __init__(self):
        self.hidden_size = 64

class MockAudioEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = MockAudioConfig()
    def forward(self, x, attention_mask=None):
        class Output:
            last_hidden_state = torch.zeros(x.size(0), 10, 64, device=x.device)
        return Output()
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        return torch.ones(attention_mask.size(0), feature_vector_length, dtype=torch.int, device=attention_mask.device)

class MockMoonshineModel:
    def __init__(self, *args, **kwargs):
        self.encoder = MockAudioEncoder()

def get_dummy_tokenizer():
    tokenizer = MagicMock()
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    tokenizer.mask_token_id = 3
    tokenizer.get_vocab = lambda: {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    return tokenizer

@patch('transformers.MoonshineStreamingModel.from_pretrained')
@patch('transformers.Wav2Vec2Model.from_pretrained')
@patch('transformers.AutoConfig.from_pretrained')
def test_deep_fusion_xlmr_model(mock_autoconfig, mock_wav2vec2, mock_moonshine):
    # Setup mocks
    mock_moonshine.return_value = MockMoonshineModel()
    mock_wav2vec2.return_value = MockAudioEncoder()
    mock_autoconfig.return_value = MockAudioConfig()

    # 1. Setup config and backbone
    backbone_config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128
    )
    backbone_model = AutoModelForMaskedLM.from_config(backbone_config)
    tokenizer = get_dummy_tokenizer()

    # 2. Setup arguments with deep_cross_attn
    args = DiscreteDiffusionModelArguments(
        num_diffusion_timesteps=10,
        diffusion_type="absorbing",
        attention_strategy="full",
        audio_fusion_strategy="deep_cross_attn",
        cache_dir="./cache",
        pretrained_audio_encoder=True
    )
    args.dataset_type = "speech_recognition"

    # Instantiate model
    model = DiscreteDiffusionXLMRModel(args, tokenizer, backbone_model)
    
    # Verify the layers were replaced
    assert isinstance(model.model.roberta.encoder.layer[0], CrossAttnRobertaLayer)
    assert isinstance(model.model.roberta.encoder.layer[1], CrossAttnRobertaLayer)
    
    # Verify weight initialization
    weight_data = model.model.roberta.encoder.layer[0].cross_attention.out_proj.weight.data
    assert weight_data.abs().max() < 0.2

    # 3. Test forward pass
    batch_size = 2
    seq_len = 8
    prev_output_tokens = torch.randint(10, 90, (batch_size, seq_len))
    partial_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    audio_features = torch.zeros(batch_size, 40)
    audio_attention_mask = torch.ones(batch_size, 40)

    # Run forward
    logits = model(
        prev_output_tokens=prev_output_tokens,
        partial_mask=partial_mask,
        audio_features=audio_features,
        audio_attention_mask=audio_attention_mask
    )
    
    assert logits.shape == (batch_size, seq_len, 100)

@patch('transformers.MoonshineStreamingModel.from_pretrained')
@patch('transformers.Wav2Vec2Model.from_pretrained')
@patch('transformers.AutoConfig.from_pretrained')
def test_deep_fusion_dlm_model(mock_autoconfig, mock_wav2vec2, mock_moonshine):
    mock_moonshine.return_value = MockMoonshineModel()
    mock_wav2vec2.return_value = MockAudioEncoder()
    mock_autoconfig.return_value = MockAudioConfig()

    # 1. Setup config
    backbone_config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128
    )
    
    config = DiscreteDiffusionConfig(
        backbone_config=backbone_config.to_dict(),
        num_diffusion_timesteps=10,
        diffusion_type="absorbing",
        attention_strategy="full",
        dataset_type="speech_recognition",
        audio_fusion_strategy="deep_cross_attn",
        mask_token_id=3,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        cache_dir="./cache",
        pretrained_audio_encoder=True
    )

    # Instantiate model
    model = DiscreteDiffusionModel(config)
    
    # Verify layers replaced
    assert isinstance(model.model.roberta.encoder.layer[0], CrossAttnRobertaLayer)
    
    # Run forward
    batch_size = 2
    seq_len = 8
    prev_output_tokens = torch.randint(10, 90, (batch_size, seq_len))
    partial_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    audio_features = torch.zeros(batch_size, 40)
    audio_attention_mask = torch.ones(batch_size, 40)

    logits = model(
        prev_output_tokens=prev_output_tokens,
        partial_mask=partial_mask,
        audio_features=audio_features,
        audio_attention_mask=audio_attention_mask
    )
    
    assert logits.shape == (batch_size, seq_len, 100)

@patch('transformers.MoonshineStreamingModel.from_pretrained')
@patch('transformers.Wav2Vec2Model.from_pretrained')
@patch('transformers.AutoConfig.from_pretrained')
def test_prefix_backward_compatibility(mock_autoconfig, mock_wav2vec2, mock_moonshine):
    mock_moonshine.return_value = MockMoonshineModel()
    mock_wav2vec2.return_value = MockAudioEncoder()
    mock_autoconfig.return_value = MockAudioConfig()

    # Setup config and backbone
    backbone_config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128
    )
    backbone_model = AutoModelForMaskedLM.from_config(backbone_config)
    tokenizer = get_dummy_tokenizer()

    # Setup prefix strategy
    args = DiscreteDiffusionModelArguments(
        num_diffusion_timesteps=10,
        diffusion_type="absorbing",
        attention_strategy="full",
        audio_fusion_strategy="prefix",
        cache_dir="./cache",
        pretrained_audio_encoder=True
    )
    args.dataset_type = "speech_recognition"

    model = DiscreteDiffusionXLMRModel(args, tokenizer, backbone_model)
    
    # Verify layers are standard RobertaLayers (not CrossAttnRobertaLayer)
    assert not isinstance(model.model.roberta.encoder.layer[0], CrossAttnRobertaLayer)

    # Run forward
    batch_size = 2
    seq_len = 8
    prev_output_tokens = torch.randint(10, 90, (batch_size, seq_len))
    partial_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    audio_features = torch.zeros(batch_size, 40)
    audio_attention_mask = torch.ones(batch_size, 40)

    logits = model(
        prev_output_tokens=prev_output_tokens,
        partial_mask=partial_mask,
        audio_features=audio_features,
        audio_attention_mask=audio_attention_mask
    )
    
    assert logits.shape == (batch_size, seq_len, 100)

@patch('numpy.load')
def test_llada_eos_padding(mock_np_load):
    import numpy as np
    mock_np_load.return_value = np.zeros((10, 64))

    from data.precomputed_multitask import PrecomputedMultiTaskDataset
    
    class DummyArgs:
        def __init__(self):
            self.max_length = 20
            
    args = DummyArgs()
    index = [{"idx": 0, "wav_id": "dummy", "embed_file": "dummy.npy"}]
    token_ids_map = {"english": [[10, 11, 12]]}
    task_configs = [("english", 100)]
    tokenizer = get_dummy_tokenizer()
    
    dataset = PrecomputedMultiTaskDataset(
        args=args,
        index=index,
        token_ids_map=token_ids_map,
        task_configs=task_configs,
        tokenizer=tokenizer,
        embed_dir="dummy_dir",
        save_dtype="float32",
        is_train=True
    )
    
    sample = dataset[0]
    source = sample["source"]
    
    # Target sequence should be exactly max_length (20)
    assert len(source) == 20
    
    # Contents should be: [BOS (0), task_token (100), word1 (10), word2 (11), word3 (12)] + 15 EOS tokens (2)
    expected = [0, 100, 10, 11, 12] + [2] * 15
    assert source.tolist() == expected

if __name__ == "__main__":
    print("Running test_deep_fusion_xlmr_model...")
    test_deep_fusion_xlmr_model()
    print("test_deep_fusion_xlmr_model passed!")

    print("Running test_deep_fusion_dlm_model...")
    test_deep_fusion_dlm_model()
    print("test_deep_fusion_dlm_model passed!")

    print("Running test_prefix_backward_compatibility...")
    test_prefix_backward_compatibility()
    print("test_prefix_backward_compatibility passed!")
    
    print("Running test_llada_eos_padding...")
    test_llada_eos_padding()
    print("test_llada_eos_padding passed!")

    print("All tests passed successfully!")
