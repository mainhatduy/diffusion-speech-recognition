import os
import logging
import multiprocessing as mp
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
from .base import PromptDataset
from .utils import _decode_wav_bytes

class SpeechDataset(PromptDataset):
    """Dataset for speech recognition: audio -> text transcription.
    Uses NhutP/VietSpeech or similar datasets with 'audio' and 'transcription' columns.
    """
    
    def __init__(self, args, raw_data, tokenizer, feature_extractor):
        super().__init__(args, raw_data, tokenizer)
        self.feature_extractor = feature_extractor
        self.target_sample_rate = 16000  # MMS expects 16kHz
    
    def __getitem__(self, index):
        item = self.raw_data[index]
        
        # Decode audio from raw bytes (dataset loaded with decode=False)
        audio_info = item['audio']
        wav_bytes = audio_info['bytes']
        waveform, sample_rate = _decode_wav_bytes(wav_bytes)
        
        # Resample if needed (unlikely for VietSpeech which is 16kHz, but handle it)
        if sample_rate != self.target_sample_rate:
            # Simple linear interpolation resampling
            ratio = self.target_sample_rate / sample_rate
            new_length = int(len(waveform) * ratio)
            indices = np.linspace(0, len(waveform) - 1, new_length)
            waveform = np.interp(indices, np.arange(len(waveform)), waveform)
        
        # Extract audio features using Wav2Vec2FeatureExtractor
        audio_inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )
        audio_values = audio_inputs.input_values.squeeze(0)  # (T_samples,)
        
        # Tokenize transcription as target
        transcription = item['transcription']
        tgt = self.tokenizer.encode(transcription, add_special_tokens=True)
        if len(tgt) > self.max_length:
            tgt = tgt[:self.max_length]
        
        # For speech recognition with audio prefix fusion:
        # - source = [BOS] (minimal text prompt, since audio carries the source info)
        # - target = tokenized transcription
        # - The model receives audio via audio_features, not via text tokens
        src = [self.tokenizer.bos_token_id]
        
        # Remove BOS from tgt if present (will be in src)
        if tgt[0] == self.tokenizer.bos_token_id:
            tgt = tgt[1:]
        
        src_length = len(src)
        concatenated = src + tgt
        
        return {
            "id": index,
            "source": torch.tensor(concatenated)[-self.max_length:],
            "target": torch.tensor([self.tokenizer.bos_token_id] + tgt if tgt[0] != self.tokenizer.bos_token_id else tgt),
            "src_length": src_length,
            "audio_values": audio_values,  # Raw waveform tensor for Wav2Vec2
        }
    
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        hf_token = args.hf_token or os.getenv('HF_TOKEN')
        if hf_token is None:
            raise ValueError(
                "HF_TOKEN is required. Set it via --hf_token argument or HF_TOKEN environment variable"
            )
        
        # Load feature extractor for audio preprocessing
        audio_encoder_name = getattr(args, 'audio_encoder_name', 'UsefulSensors/moonshine-streaming-medium')
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            audio_encoder_name, cache_dir=getattr(args, 'cache_dir', None)
        )
        
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        num_proc = max(1, int(mp.cpu_count() / world_size))
        
        print(f"Loading speech dataset from {args.data_path}")
        try:
            full_dataset = load_dataset(
                args.data_path,
                token=hf_token,
                cache_dir=getattr(args, 'cache_dir', None),
                split='train[:100%]',
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Disable audio decoding to avoid torchcodec dependency
        full_dataset = full_dataset.cast_column('audio', Audio(decode=False))
        
        full_dataset = full_dataset.shuffle(seed=42)
        split = full_dataset.train_test_split(test_size=0.01, seed=42)
        
        train_raw = split['train']
        valid_raw = split['test']
        
        print(f"Speech dataset loaded: {len(train_raw)} train, {len(valid_raw)} validation samples")
        
        def filter_fn(example):
            text = example.get('transcription', '')
            if not text:
                return False
            t_ids = tokenizer.encode(text, add_special_tokens=True)
            # BOS + transcription tokens must fit in max_length
            return (1 + len(t_ids)) <= args.max_length
        
        if train:
            train_raw = train_raw.filter(filter_fn, num_proc=num_proc)
        if valid or test:
            valid_raw = valid_raw.filter(filter_fn, num_proc=num_proc)
        
        print(f"After filtering: {len(train_raw) if train else 0} train, {len(valid_raw) if (valid or test) else 0} validation samples")
        
        train_dataset = SpeechDataset(args, train_raw, tokenizer, feature_extractor) if train else None
        valid_dataset = SpeechDataset(args, valid_raw, tokenizer, feature_extractor) if valid else None
        test_dataset = SpeechDataset(args, valid_raw, tokenizer, feature_extractor) if test else None
        
        return train_dataset, valid_dataset, test_dataset
