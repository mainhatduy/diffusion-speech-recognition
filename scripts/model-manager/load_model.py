import sys
import os
import json
import torch
import numpy as np
import miniaudio
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.abspath("src"))

from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel
from dd_generator import DiscreteDiffusionGenerator, DiscreteDiffusionGeneratorArguments

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/model-manager/load_model.py <model_path_or_repo_id> [audio_path] [json_path]")
        print("Example local: python scripts/model-manager/load_model.py outputs/vi_multitask/checkpoint-60000 test/test_data/test_sample.mp3")
        print("Example Hugging Face: python scripts/model-manager/load_model.py aiai-laboratory/discrete-diffusion-vi-multitask test/test_data/test_sample.mp3")
        sys.exit(1)
        
    model_path_or_id = sys.argv[1]
    audio_path = sys.argv[2] if len(sys.argv) > 2 else "test/test_data/test_sample.mp3"
    json_path = sys.argv[3] if len(sys.argv) > 3 else "test/test_data/test_sample.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load model and tokenizer
    print(f"Loading model and tokenizer from: {model_path_or_id}")
    
    if os.path.exists(model_path_or_id):
        # Local checkpoint loading
        # Resolve parent directory containing args.json and tokenizer files if path is a checkpoint subdirectory
        if "checkpoint-" in model_path_or_id:
            experiment_dir = os.path.dirname(model_path_or_id)
            checkpoint_dir = model_path_or_id
        else:
            experiment_dir = model_path_or_id
            # Find the highest checkpoint in experiment_dir
            checkpoints = [d for d in os.listdir(experiment_dir) if d.startswith("checkpoint-")]
            if not checkpoints:
                print(f"No checkpoints found in {experiment_dir}")
                sys.exit(1)
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(experiment_dir, checkpoints[-1])
            
        print(f"Using checkpoint directory: {checkpoint_dir}")
        args_path = os.path.join(experiment_dir, "args.json")
        with open(args_path, "r") as f:
            all_args = json.load(f)
            
        model_args = all_args["model"]
        data_args = all_args["data"]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(experiment_dir, use_fast=False)
        
        # Load config
        backbone_config = AutoConfig.from_pretrained(model_args['pretrained'])
        if len(tokenizer) != backbone_config.vocab_size:
            backbone_config.vocab_size = len(tokenizer)
            
        config = DiscreteDiffusionConfig(
            backbone_config=backbone_config,
            num_diffusion_timesteps=model_args["num_diffusion_timesteps"],
            diffusion_type=model_args["diffusion_type"],
            attention_strategy=model_args["attention_strategy"],
            vocab_pad_to_multiple=model_args["vocab_pad_to_multiple"],
            lora=model_args["lora"],
            lora_target_modules=model_args["lora_target_modules"],
            lora_alpha=model_args["lora_alpha"],
            lora_rank=model_args["lora_rank"],
            lora_bias=model_args["lora_bias"],
            lora_dropout=model_args["lora_dropout"],
            mask_token_id=tokenizer.mask_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            argmax_decoding=True,
            dataset_type=data_args.get("dataset_type", "bilingual"),
            audio_encoder_name=data_args.get("audio_encoder_name", "facebook/mms-300m"),
        )
        
        model = DiscreteDiffusionModel(config)
        ckpt_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        print(f"Loading weights from {ckpt_file}")
        state_dict = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load from Hugging Face hub
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        model = AutoModel.from_pretrained(model_path_or_id, trust_remote_code=True)
        
    model = model.to(device)
    model.eval()
    
    # 2. Process audio input
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        sys.exit(1)
        
    print(f"Processing audio: {audio_path}")
    data = miniaudio.decode_file(audio_path)
    waveform = np.array(data.samples, dtype=np.float32)
    if data.nchannels > 1:
        waveform = waveform.reshape(-1, data.nchannels).mean(axis=1)
    waveform = waveform / 32768.0  # normalize
    
    target_sample_rate = 16000
    if data.sample_rate != target_sample_rate:
        ratio = target_sample_rate / data.sample_rate
        new_length = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform)
        
    audio_values = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Handle Moonshine audio length rounding if applicable
    audio_encoder_name = getattr(model.config, "audio_encoder_name", "facebook/mms-300m")
    if "moonshine" in audio_encoder_name.lower():
        audio_len = audio_values.size(-1)
        padded_len = ((audio_len + 79) // 80) * 80
        padded_audio = torch.zeros(1, padded_len, device=device)
        padded_audio[0, :audio_len] = audio_values[0]
        audio_values = padded_audio
        
        audio_attention_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
        audio_attention_mask[0, :audio_len] = 1
    else:
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long).to(device)
        
    # 3. Load ground truth if exists
    sample_info = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            sample_info = json.load(f)
            
    # 4. Initialize Generator
    generator_args = DiscreteDiffusionGeneratorArguments(
        max_iterations=10,
        mbr=1,
        length_beam=1,
        oracle_length=True,  # Match validation setting during training
        strategy="reparam-uncond-deterministic-cosine",
        argmax_decoding=True,
        temperature=1.0,
    )
    generator = DiscreteDiffusionGenerator(generator_args, tokenizer=tokenizer)
    
    # Define tasks to test
    # Key in json -> (task_token_str, task_token_id)
    # If task token is None, it is speech recognition (Vietnamese transcript)
    tasks = {
        "text": ("Transcription (VI)", None),
        "english": ("Translation (EN)", "<vi_en>"),
        "chinese": ("Translation (ZH)", "<vi_zh>"),
        "korean": ("Translation (KO)", "<vi_ko>"),
    }
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print(f"Audio path: {audio_path}")
    print("="*80)
    
    for key, (label, task_token_str) in tasks.items():
        # Encode prefix
        if task_token_str is None:
            # Speech recognition: [BOS]
            src = [tokenizer.bos_token_id]
        else:
            # Multitask translation: [BOS, task_token_id]
            task_token_id = tokenizer.convert_tokens_to_ids(task_token_str)
            src = [tokenizer.bos_token_id, task_token_id]
            
        # Get target reference for oracle length (which is required by oracle_length=True generator setup)
        ref_text = sample_info.get(key, "")
        if not ref_text and key == "text":
            ref_text = sample_info.get("transcription", "")
            
        # If reference text is empty, we fall back to a dummy target length of 30 tokens
        if ref_text:
            from data.utils import normalize_text
            ref_text = normalize_text(ref_text)
            tgt = tokenizer.encode(ref_text, add_special_tokens=True)
            if len(tgt) > 0 and tgt[0] == tokenizer.bos_token_id:
                tgt = tgt[1:]
        else:
            tgt = [tokenizer.mask_token_id] * 30 + [tokenizer.eos_token_id]
            
        sources = [torch.tensor(src + tgt)]
        targets = [torch.tensor([tokenizer.bos_token_id] + tgt)]
        src_lengths = [len(src)]
        
        source_padded = torch.nn.utils.rnn.pad_sequence(
            sources, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        target_padded = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        
        batch_size_dummy, seq_len = source_padded.size()
        src_lengths_tensor = torch.tensor(src_lengths, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size_dummy, -1)
        partial_masks = (position_ids < src_lengths_tensor.unsqueeze(1)).to(device)
        
        test_batch = {
            "id": torch.tensor([0]).to(device),
            "net_input": {
                "src_tokens": source_padded,
                "src_lengths": torch.tensor([len(s) for s in sources]).to(device),
                "partial_masks": partial_masks,
                "audio_features": audio_values,
                "audio_attention_mask": audio_attention_mask
            },
            "target": target_padded,
            "nsentences": 1,
            "ntokens": len(targets[0])
        }
        
        # Run generation
        hyps, _ = generator.generate(model, test_batch)
        pred_text = generator.decode(hyps)[0]
        
        print(f"\n--- Task: {label} ---")
        if ref_text:
            print(f"Reference: {ref_text}")
        print(f"Prediction: {pred_text}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
