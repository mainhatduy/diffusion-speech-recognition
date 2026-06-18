import os
import logging
import multiprocessing as mp
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
from .base import PromptDataset
from .utils import _decode_wav_bytes, normalize_text

class TranslatedSpeechDataset(PromptDataset):
    """Dataset for translated speech recognition: audio -> translated text.
    Loads translations from aiai-laboratory/vietspeech-train-translated
    and maps them to audio files from NhutP/VietSpeech.
    """
    
    def __init__(self, args, raw_data, vietspeech_dataset, path_to_vs_idx, tgt_field, tokenizer, feature_extractor):
        super().__init__(args, raw_data, tokenizer)
        self.vietspeech_dataset = vietspeech_dataset
        self.path_to_vs_idx = path_to_vs_idx
        self.tgt_field = tgt_field
        self.feature_extractor = feature_extractor
        self.target_sample_rate = 16000  # MMS expects 16kHz
        
    def __getitem__(self, index):
        # raw_data is the translated dataset
        translated_item = self.raw_data[index]
        wav_id = translated_item["id"]
        
        vs_idx = self.path_to_vs_idx.get(wav_id)
        if vs_idx is None:
            raise ValueError(f"WAV ID {wav_id} not found in NhutP/VietSpeech")
            
        vs_item = self.vietspeech_dataset[vs_idx]
        
        # Decode audio from raw bytes
        audio_info = vs_item['audio']
        wav_bytes = audio_info['bytes']
        waveform, sample_rate = _decode_wav_bytes(wav_bytes)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
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
        
        # Get target text and normalize it
        text = translated_item[self.tgt_field]
        normalized_text = normalize_text(text)
        
        tgt = self.tokenizer.encode(normalized_text, add_special_tokens=True)
        if len(tgt) > self.max_length:
            tgt = tgt[:self.max_length]
        
        src = [self.tokenizer.bos_token_id]
        
        # Remove BOS from tgt if present (will be in src)
        if len(tgt) > 0 and tgt[0] == self.tokenizer.bos_token_id:
            tgt = tgt[1:]
        
        src_length = len(src)
        concatenated = src + tgt
        
        return {
            "id": index,
            "source": torch.tensor(concatenated)[-self.max_length:],
            "target": torch.tensor([self.tokenizer.bos_token_id] + tgt if len(tgt) == 0 or tgt[0] != self.tokenizer.bos_token_id else tgt),
            "src_length": src_length,
            "audio_values": audio_values,
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
        
        # 1. Load translated dataset from aiai-laboratory/vietspeech-train-translated
        print("Loading translated speech dataset from aiai-laboratory/vietspeech-train-translated")
        try:
            translated_dataset = load_dataset(
                'aiai-laboratory/vietspeech-train-translated',
                token=hf_token,
                cache_dir=getattr(args, 'cache_dir', None),
                split='train[:100%]',
            )
        except Exception as e:
            print(f"Error loading translated dataset: {e}")
            raise
            
        # 2. Load speech dataset from NhutP/VietSpeech
        print("Loading audio dataset from NhutP/VietSpeech")
        try:
            vietspeech_dataset = load_dataset(
                'NhutP/VietSpeech',
                token=hf_token,
                cache_dir=getattr(args, 'cache_dir', None),
                split='train[:100%]',
            )
        except Exception as e:
            print(f"Error loading VietSpeech dataset: {e}")
            raise
            
        # Disable audio decoding to avoid torchcodec dependency
        vietspeech_dataset = vietspeech_dataset.cast_column('audio', Audio(decode=False))
        
        # 3. Create mapping from path to index in vietspeech_dataset
        print("Mapping paths in VietSpeech")
        audio_column = vietspeech_dataset.data.column('audio')
        vs_paths = []
        for chunk in audio_column.chunks:
            vs_paths.extend(chunk.field('path').to_pylist())
        path_to_vs_idx = {path: idx for idx, path in enumerate(vs_paths)}
        
        # Determine the target field based on config
        tgt_col = args.tgt_column or args.tgt_lang or 'english'
        if tgt_col in ['vi', 'vietnamese']:
            tgt_field = 'vietnamese'
        elif tgt_col in ['en', 'english']:
            tgt_field = 'english'
        elif tgt_col in ['zh', 'cn', 'chinese']:
            tgt_field = 'chinese'
        elif tgt_col in ['ko', 'kr', 'korean']:
            tgt_field = 'korean'
        else:
            tgt_field = tgt_col
            
        # Check if tgt_field exists in translated_dataset features
        if tgt_field not in translated_dataset.features:
            raise ValueError(
                f"Target column/language '{tgt_col}' (mapped to '{tgt_field}') not found in "
                f"translated dataset features: {list(translated_dataset.features.keys())}"
            )
            
        print(f"Using target column: '{tgt_field}' for translation")
        
        # 4. Filter or split the dataset.
        translated_dataset = translated_dataset.shuffle(seed=42)
        split = translated_dataset.train_test_split(test_size=0.01, seed=42)
        
        train_raw = split['train']
        valid_raw = split['test']
        
        print(f"Dataset split: {len(train_raw)} train, {len(valid_raw)} validation samples")
        
        # Filter function for target text length
        def filter_fn(example):
            text = example.get(tgt_field, '')
            if not text:
                return False
            # Normalize and tokenize to check length
            normalized_text = normalize_text(text)
            t_ids = tokenizer.encode(normalized_text, add_special_tokens=True)
            return (1 + len(t_ids)) <= args.max_length
            
        if train:
            translated_dataset_train = train_raw.filter(filter_fn, num_proc=num_proc)
        else:
            translated_dataset_train = None
            
        if valid:
            translated_dataset_valid = valid_raw.filter(filter_fn, num_proc=num_proc)
        else:
            translated_dataset_valid = None
            
        print(f"After filtering: {len(translated_dataset_train) if train else 0} train, {len(translated_dataset_valid) if valid else 0} validation samples")
        
        train_dataset = TranslatedSpeechDataset(
            args, translated_dataset_train, vietspeech_dataset, path_to_vs_idx, tgt_field, tokenizer, feature_extractor
        ) if train else None
        
        valid_dataset = TranslatedSpeechDataset(
            args, translated_dataset_valid, vietspeech_dataset, path_to_vs_idx, tgt_field, tokenizer, feature_extractor
        ) if valid else None
        
        test_dataset = None
        
        return train_dataset, valid_dataset, test_dataset
