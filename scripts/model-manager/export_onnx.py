import sys
import os
import torch
import json
import numpy as np

# Add src to Python path
sys.path.append(os.path.abspath("src"))

import transformers
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig, AutoModel
from model.modeling_dlm import DiscreteDiffusionModel
from model.configuration_dlm import DiscreteDiffusionConfig

# Patch configurations to force "eager" attention implementation (bypasses SDPA SymBool issue in PyTorch 2.x)
orig_auto_config_from_pretrained = AutoConfig.from_pretrained
orig_pretrained_config_from_pretrained = PretrainedConfig.from_pretrained

def patched_auto_config_from_pretrained(*args, **kwargs):
    kwargs["attn_implementation"] = "eager"
    config = orig_auto_config_from_pretrained(*args, **kwargs)
    if isinstance(config, tuple):
        config[0]._attn_implementation = "eager"
        config[0].attn_implementation = "eager"
    else:
        config._attn_implementation = "eager"
        config.attn_implementation = "eager"
    return config

def patched_pretrained_config_from_pretrained(cls, *args, **kwargs):
    kwargs["attn_implementation"] = "eager"
    config = orig_pretrained_config_from_pretrained.__func__(cls, *args, **kwargs)
    if isinstance(config, tuple):
        config[0]._attn_implementation = "eager"
        config[0].attn_implementation = "eager"
    else:
        config._attn_implementation = "eager"
        config.attn_implementation = "eager"
    return config

AutoConfig.from_pretrained = patched_auto_config_from_pretrained
PretrainedConfig.from_pretrained = classmethod(patched_pretrained_config_from_pretrained)

class DiscreteDiffusionONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prev_output_tokens, partial_mask, precomputed_audio_embeds, precomputed_audio_mask):
        return self.model(
            prev_output_tokens=prev_output_tokens,
            partial_mask=partial_mask,
            precomputed_audio_embeds=precomputed_audio_embeds,
            precomputed_audio_mask=precomputed_audio_mask
        )

class AudioEncoderONNXWrapper(torch.nn.Module):
    def __init__(self, audio_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, audio_features, audio_attention_mask):
        outputs = self.audio_encoder(
            audio_features,
            attention_mask=audio_attention_mask
        )
        return outputs.last_hidden_state

def main():
    repo_id = "aiai-laboratory/diffusion-speech-translation-from-vi-v1"
    os.makedirs("onnx", exist_ok=True)
    
    print(f"Loading model config from Hugging Face Hub: {repo_id}")
    # Load config locally
    config = DiscreteDiffusionConfig.from_pretrained(repo_id)
    # Ensure attention implementation is eager for backbone config as well
    if isinstance(config.backbone_config, dict):
        config.backbone_config["attn_implementation"] = "eager"
    
    # Load tokenizer with trust_remote_code=True
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    
    # Instantiate the local model (forces using local src/model/modeling_dlm.py)
    print("Instantiating local DiscreteDiffusionModel...")
    # Force pretrained_audio_encoder=True so it downloads Moonshine streaming weights
    config.pretrained_audio_encoder = True
    model = DiscreteDiffusionModel(config)
    
    # Download weights and load them
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    print("Downloading model weights from Hugging Face Hub...")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"Loading weights from: {weights_path}")
    state_dict = load_file(weights_path)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights with strict=False.")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    model.eval()
    
    # Check attention implementation
    print(f"Model backbone attention implementation: {model.model.config._attn_implementation}")
    if hasattr(model, "audio_encoder"):
        print(f"Audio encoder attention implementation: {model.audio_encoder.config._attn_implementation}")

    # ==========================================
    # 1. Export Diffusion Backbone
    # ==========================================
    print("\n--- Exporting Diffusion Backbone Model ---")
    backbone_wrapper = DiscreteDiffusionONNXWrapper(model)
    
    # Dummy inputs for backbone
    batch_size = 1
    seq_len = 32
    audio_len = 96  # e.g., 3 seconds of audio at 50Hz frame rate
    hidden_size = model.audio_encoder.config.hidden_size if hasattr(model, "audio_encoder") else 1024
    
    dummy_prev_output_tokens = torch.randint(0, len(tokenizer), (batch_size, seq_len), dtype=torch.long)
    dummy_partial_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)
    dummy_audio_embeds = torch.randn(batch_size, audio_len, hidden_size, dtype=torch.float32)
    dummy_audio_mask = torch.ones(batch_size, audio_len, dtype=torch.int32)
    
    backbone_path = "onnx/diffusion_backbone.onnx"
    
    try:
        torch.onnx.export(
            backbone_wrapper,
            (dummy_prev_output_tokens, dummy_partial_mask, dummy_audio_embeds, dummy_audio_mask),
            backbone_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["prev_output_tokens", "partial_mask", "precomputed_audio_embeds", "precomputed_audio_mask"],
            output_names=["logits"],
            dynamic_axes={
                "prev_output_tokens": {0: "batch_size", 1: "seq_len"},
                "partial_mask": {0: "batch_size", 1: "seq_len"},
                "precomputed_audio_embeds": {0: "batch_size", 1: "audio_len"},
                "precomputed_audio_mask": {0: "batch_size", 1: "audio_len"},
                "logits": {0: "batch_size", 1: "seq_len"}
            }
        )
        print(f"Successfully exported diffusion backbone to: {backbone_path}")
    except Exception as e:
        print(f"Failed to export diffusion backbone: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================
    # 2. Export Audio Encoder (if present)
    # ==========================================
    if hasattr(model, "audio_encoder") and model.audio_encoder is not None:
        print("\n--- Exporting Audio Encoder Model ---")
        audio_wrapper = AudioEncoderONNXWrapper(model.audio_encoder)
        
        # Moonshine expects inputs of shape (batch, audio_len)
        # Typically audio_len is multiple of 80
        dummy_audio_features = torch.randn(batch_size, 2400, dtype=torch.float32)
        dummy_audio_attention_mask = torch.ones(batch_size, 2400, dtype=torch.long)
        
        audio_encoder_path = "onnx/audio_encoder.onnx"
        
        try:
            torch.onnx.export(
                audio_wrapper,
                (dummy_audio_features, dummy_audio_attention_mask),
                audio_encoder_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["audio_features", "audio_attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "audio_features": {0: "batch_size", 1: "audio_len"},
                    "audio_attention_mask": {0: "batch_size", 1: "audio_len"},
                    "last_hidden_state": {0: "batch_size", 1: "audio_seq_len"}
                }
            )
            print(f"Successfully exported audio encoder to: {audio_encoder_path}")
        except Exception as e:
            print(f"Failed to export audio encoder: {e}")
            import traceback
            traceback.print_exc()
            
    # ==========================================
    # 3. Verification with ONNX Runtime
    # ==========================================
    print("\n--- Verifying ONNX Models with ONNX Runtime ---")
    import onnxruntime as ort
    
    if os.path.exists(backbone_path):
        try:
            ort_sess = ort.InferenceSession(backbone_path)
            
            # Prepare inputs
            ort_inputs = {
                "prev_output_tokens": dummy_prev_output_tokens.numpy(),
                "partial_mask": dummy_partial_mask.numpy(),
                "precomputed_audio_embeds": dummy_audio_embeds.numpy(),
                "precomputed_audio_mask": dummy_audio_mask.numpy()
            }
            
            # Run inference
            ort_outputs = ort_sess.run(None, ort_inputs)
            print(f"ONNX Runtime verification successful for Diffusion Backbone!")
            print(f"Output shape: {ort_outputs[0].shape}")
            
            # Compare outputs with PyTorch
            with torch.no_grad():
                torch_outputs = backbone_wrapper(
                    dummy_prev_output_tokens,
                    dummy_partial_mask,
                    dummy_audio_embeds,
                    dummy_audio_mask
                )
            
            max_diff = np.max(np.abs(torch_outputs.numpy() - ort_outputs[0]))
            print(f"Max difference between PyTorch and ONNX Runtime: {max_diff}")
            if max_diff < 1e-4:
                print("Verification PASSED!")
            else:
                print("Verification WARNING: High difference, please check precision.")
        except Exception as e:
            print(f"ONNX Runtime verification failed: {e}")
            import traceback
            traceback.print_exc()
            
    if hasattr(model, "audio_encoder") and os.path.exists(audio_encoder_path):
        try:
            ort_sess = ort.InferenceSession(audio_encoder_path)
            ort_inputs = {
                "audio_features": dummy_audio_features.numpy(),
                "audio_attention_mask": dummy_audio_attention_mask.numpy()
            }
            ort_outputs = ort_sess.run(None, ort_inputs)
            print(f"ONNX Runtime verification successful for Audio Encoder!")
            print(f"Output shape: {ort_outputs[0].shape}")
            
            with torch.no_grad():
                torch_outputs = audio_wrapper(dummy_audio_features, dummy_audio_attention_mask)
                
            max_diff = np.max(np.abs(torch_outputs.numpy() - ort_outputs[0]))
            print(f"Max difference between PyTorch and ONNX Runtime: {max_diff}")
            if max_diff < 1e-4:
                print("Verification PASSED!")
            else:
                print("Verification WARNING: High difference, please check precision.")
        except Exception as e:
            print(f"ONNX Runtime verification failed for Audio Encoder: {e}")

if __name__ == "__main__":
    main()
