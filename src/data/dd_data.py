import torch
from torch.utils.data import Dataset, IterableDataset, BatchSampler, DistributedSampler



from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, get_length_grouped_indices

# from src.task.partial_discrete_diffusion_task import PartialDiffusionLanguagePairDataset, concat_func

from typing import Any, Dict, Iterator, List, Optional, Union

from dataclasses import dataclass, field

from functools import partial

import math

import numpy as np

from datasets import load_dataset, load_from_disk, Audio

from tqdm import tqdm

import json

import multiprocessing as mp

import wave
import io
import struct


@dataclass
class DiscreteDiffusionDataArguments:
    dataset_type: str = field(
        default="bilingual"   # fairseq | flan | flanv2 | pair | bilingual | speech_recognition
    )
    audio_encoder_name: str = field(
        default="UsefulSensors/moonshine-streaming-medium",
        metadata={"help": "pretrained audio encoder model name for speech_recognition"}
    )
    data_path: str = field(
        default=""
    )
    src_lang: str = field(
        default=""
    )
    tgt_lang: str = field(
        default=""
    )
    prompt_built: bool = field(
        default=False
        # help="only for flan"
    )
    # batch_by_tokens: bool = field(
    #     default=False
    # )
    max_length: int = field(
        default=2048
    )
    packing: bool = field(
        default=False,
        metadata={"help": "whether to pack the output data"}
    )
    hf_token: str = field(
        default=None,
        metadata={"help": "Hugging Face token for private datasets"}
    )
    src_column: str = field(
        default="en",
        metadata={"help": "Source column name in dataset"}
    )
    tgt_column: str = field(
        default="vi",
        metadata={"help": "Target column name in dataset"}
    )
    dedupe: bool = field(
        default=False,
        metadata={"help": "whether to deduplicate the data"}
    )
    remove_wiki: bool = field(
        default=False,
        metadata={"help": "whether to remove wiki from the AMR entries"}
    )
    fix_ftfy: bool = field(
        default=False,
        metadata={"help": "whether to fix text issues"}
    )
    normalize_punct: bool = field(
        default=False,
        metadata={"help": "whether to normalize punctuation"}
    )
    detokenize: bool = field(
        default=False,
        metadata={"help": "whether to detokenize"}
    )
    remove_bracketed: bool = field(
        default=False,
        metadata={"help": "whether to remove sentences that start and end with punctuation"}
    )
    dereify: bool = field(
        default=False,
        metadata={"help": "whether to dereify AMR graph"}
    )


class PromptDataset(Dataset):
    def __init__(self, args, raw_data, tokenizer):
        super().__init__()
        self.args = args
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.item_size = {}
        self.max_length = args.max_length
    
    # def set_max_length(self, max_length):
    #     self.max_length = max_length
        
    def __len__(self):
        return len(self.raw_data)
    
    def size(self, index):
        if index not in self.item_size:
            item = self.__getitem__(index)
            self.item_size[index] = len(item["source"])
        return self.item_size[index]
    
    def ordered_indices(self):
        raise NotImplementedError
    
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        raise NotImplementedError

    def build_prompt(self, item):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    



class BilingualDataset(PromptDataset):
    """Dataset for bilingual data with two columns (e.g., en and vi) from Hugging Face"""
    
    def __getitem__(self, index):
        item = self.raw_data[index]
        src_text = item[self.args.src_column]
        tgt_text = item[self.args.tgt_column]
        
        src = self.tokenizer.encode(src_text, add_special_tokens=True)
        tgt = self.tokenizer.encode(tgt_text, add_special_tokens=True)[-self.max_length:]
        
        # Remove EOS from src if present, and BOS from tgt if present
        if src[-1] == self.tokenizer.eos_token_id:
            src = src[:-1]
        if tgt[0] == self.tokenizer.bos_token_id:
            tgt = tgt[1:]
        
        src_length = len(src)
        concatenated = src + tgt
            
        return {
            "id": index,
            "source": torch.tensor(concatenated)[-self.max_length:],
            "target": torch.tensor([self.tokenizer.bos_token_id] + tgt if tgt[0] != self.tokenizer.bos_token_id else tgt),
            "src_length": src_length  # Length of source part in concatenated sequence
        }
    
    @staticmethod    
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        import os
        import logging
        
        # Set tokenizer model_max_length from config and suppress warnings during filtering
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        # Get HF token from args or environment variable
        hf_token = args.hf_token or os.getenv('HF_TOKEN')
        
        if hf_token is None:
            raise ValueError(
                "HF_TOKEN is required. Set it via --hf_token argument or HF_TOKEN environment variable"
            )
        
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        num_proc = max(1, int(mp.cpu_count() / world_size))
        
        try:
            # Load dataset from Hugging Face with token
            datasets = load_dataset(
                args.data_path, 
                token=hf_token,
                cache_dir=getattr(args, 'cache_dir', None),
                num_proc=num_proc,
            )
            datasets = datasets.shuffle(seed=42)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Try loading from disk as fallback
            datasets = load_from_disk(args.data_path)
        
        # Filter datasets based on length
        def filter_fn(example):
            src = example[args.src_column]
            tgt = example[args.tgt_column]
            if not src or not tgt: return False
            
            # Tokenize to check length
            s_ids = tokenizer.encode(src, add_special_tokens=True)
            t_ids = tokenizer.encode(tgt, add_special_tokens=True)
            
            # Adjust for concatenation logic in __getitem__
            # if src[-1] == eos: src = src[:-1]
            # if tgt[0] == bos: tgt = tgt[1:]
            if len(s_ids) > 0 and s_ids[-1] == tokenizer.eos_token_id:
                s_ids = s_ids[:-1]
            if len(t_ids) > 0 and t_ids[0] == tokenizer.bos_token_id:
                t_ids = t_ids[1:]
                
            return (len(s_ids) + len(t_ids)) <= args.max_length

        if train and "train" in datasets:
            # print(f"Filtering train dataset (max_length={args.max_length})...")
            datasets["train"] = datasets["train"].filter(filter_fn, num_proc=num_proc)
        
        if valid and "validation" in datasets:
            print(f"Filtering validation dataset (max_length={args.max_length})...")
            datasets["validation"] = datasets["validation"].filter(filter_fn, num_proc=num_proc)

        if test and "test" in datasets:
            print(f"Filtering test dataset (max_length={args.max_length})...")
            datasets["test"] = datasets["test"].filter(filter_fn, num_proc=num_proc)

        # Create dataset splits
        train_dataset = BilingualDataset(args, datasets["train"], tokenizer) if train and "train" in datasets else None
        valid_dataset = BilingualDataset(args, datasets["validation"], tokenizer) if valid and "validation" in datasets else None
        test_dataset = BilingualDataset(args, datasets["test"], tokenizer) if test and "test" in datasets else None
        
        return (train_dataset, valid_dataset, test_dataset)


class FlanV2Dataset(IterableDataset):
    def __init__(self, args, data_path, tokenizer) -> None:
        super().__init__()
        self._full_file_name = []
        with open(f"{data_path}/ratio.json", "r") as f:
            ratios = json.load(f)
        self.sampler = torch.distributions.Categorical(torch.tensor([ratio for _, ratio in ratios.items()]))
        self._full_file_name = [f"{data_path}/{file}" for file in ratios]
        self.idx2dataset = [open(file, "r") for file in self._full_file_name]
        self.args = args
        self.tokenizer = tokenizer
        
        self.counter = 0
        
        self.rank = rank =0 if not torch.distributed.is_initialized() else  torch.distributed.get_rank()
        world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size() 

        self.global_batch_size = args.per_device_batch_size * world_size
        self.read_step_size = world_size
        self.counter_range = set(range(
            rank * args.per_device_batch_size, (rank + 1) * args.per_device_batch_size
        ))
        
        for dataset in self.idx2dataset:
            for _ in range(rank):
                dataset.readline()       
    
    def read_step(self, dataset=None, index=None):
        # select a dataset first
        if dataset is None:
            index = self.sampler.sample()
            dataset = self.idx2dataset[index]
        for _ in range(self.read_step_size):
            line = dataset.readline()
            if not line:
                dataset.close()
                dataset = self.idx2dataset[index] = open(self._full_file_name[index], "r")
                line = dataset.readline()
        # tokenize
        item = json.loads(line)
        if "inputs_ids" in item and "targets_ids" in item:
            src = item["inputs_ids"] 
            tgt = item["targets_ids"]
        else:
            src = self.tokenizer.encode(item["inputs"])
            tgt = self.tokenizer.encode(item["targets"])

        if len(src) + len(tgt) - 2 > self.args.max_length:
            # lets do some concat
            tgt = tgt[-self.args.max_length:]
            # return self.read_step(dataset, index)
        if src[-1] == self.tokenizer.eos_token_id:
            src = src[:-1]
        return {
            "id": self.counter,
            "source": torch.tensor(src+ tgt[1:])[-self.args.max_length:],
            "target": torch.tensor(tgt)
        }
                
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        assert not test
        return (FlanV2Dataset(args, args.data_path, tokenizer=tokenizer), None, None)
    
    def __iter__(self) -> Iterator:
        # lets do some hacking
        # hf trainer wrap the dataset with IterableDatasetShard to avoid duplicas in DDP
        # this iter should only yield valid sample during the range its sample will be used
        # and it should skip N 
        
        # assume single worker 
        assert torch.utils.data.get_worker_info() is None
        while True:
            if (self.counter % self.global_batch_size) not in self.counter_range:
                yield None
            else:
                yield self.read_step()
            self.counter += 1


@dataclass
class MemoryMapTokensDataset(Dataset):
    def __init__(self, args, data_path, tokenizer):
        super().__init__()
        self.args = args
        self.tokens = np.memmap(data_path, dtype="ushort", mode="r")
        self.num_total_tokens = self.tokens.shape[0]
        self.length = args.max_length
      
    def __len__(self):
        return self.num_total_tokens // self.length
    
    def __getitem__(self, index):
        start, end = index * self.length, (index + 1) * self.length
        data = np.array(self.tokens[start:end], dtype=int)
        return {
            "id": index,
            "source": torch.tensor(data),
            "target": torch.tensor(data)
        }
      
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=False, test=False):
        assert (not test)
        return (MemoryMapTokensDataset(args, args.data_path, tokenizer=tokenizer), None, None)
        
         
@dataclass
class DiscreteDiffusionDataCollator(object):
    
    bos_id: int
    eos_id: int
    pad_id: int
    
    def __call__(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        # Extract data from samples
        # 'source' contains concatenated [src_tokens + tgt_tokens]
        # 'src_length' tells us where src ends and tgt begins
        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        src_lengths = [s["src_length"] for s in samples]
        ids = torch.tensor([s["id"] for s in samples])

        # Pad the concatenated source+target sequences
        # Use pad_id for proper padding (not eos_id)
        source_padded = torch.nn.utils.rnn.pad_sequence(
            sources, batch_first=True, padding_value=self.pad_id
        )
        target_padded = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_id
        )

        # Create partial_masks to mark source vs target positions
        # partial_masks[i, j] = True if position j is in the source part (don't mask during training)
        # partial_masks[i, j] = False if position j is in the target part (can be masked during training)
        batch_size, seq_len = source_padded.size()
        src_lengths_tensor = torch.tensor(src_lengths, dtype=torch.long)
        
        # Create position indices [0, 1, 2, ..., seq_len-1] for each sample
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # partial_masks[i, j] = True if j < src_length[i]
        partial_masks = position_ids < src_lengths_tensor.unsqueeze(1)

        # Create net_input
        net_input = {
            "src_tokens": source_padded,
            "src_lengths": torch.tensor([len(s) for s in sources]),
            "partial_masks": partial_masks
        }
        
        # Handle audio features if present (for speech_recognition)
        has_audio = "audio_values" in samples[0]
        if has_audio:
            audio_values_list = [s["audio_values"] for s in samples]
            # Pad audio to max length in batch, rounded up to a multiple of 80 (frame_len of Moonshine)
            max_audio_len = max(av.size(-1) for av in audio_values_list)
            max_audio_len = ((max_audio_len + 79) // 80) * 80
            padded_audio = torch.zeros(batch_size, max_audio_len)
            audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long)
            for i, av in enumerate(audio_values_list):
                length = av.size(-1)
                padded_audio[i, :length] = av
                audio_attention_mask[i, :length] = 1
            net_input["audio_features"] = padded_audio
            net_input["audio_attention_mask"] = audio_attention_mask

        batch = {
            "id": ids,
            "net_input": net_input,
            "target": target_padded,
            "nsentences": len(samples),
            "ntokens": sum(len(s) for s in targets),
        }
        
        return batch


    
class TokenSizeDistributedLengthGroupSampler(DistributedLengthGroupedSampler):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        dataset: Optional[Dataset],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        infinite: bool = False
    ):
        super().__init__(batch_size, dataset, num_replicas, rank, seed, drop_last, lengths, model_input_name)
        self.max_length = max_length
        self.dataset = dataset
        self.infinite = infinite
        
        self.num_batches = None
    
    def __len__(self):
        return self.num_batches if self.num_batches is not None else 0x7fffffff 
        
    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.dataset.ordered_indices()
        # indices, _ = self.dataset.filter_indices_by_size(indices, self.max_length)
        indices = [index for index in indices if self.lengths[index] <= self.max_length]
        
        # Custom batch_by_size implementation
        # Groups indices into batches such that total tokens in batch <= max_tokens (batch_size)
        # Assumes indices are sorted by length (or roughly sorted) for efficiency if needed, 
        # but here we just iterate and pack.
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Note: self.batch_size here seems to be treated as max_tokens in the original code
        # "max_tokens=self.batch_size" in data_utils.batch_by_size call.
        max_tokens = self.batch_size
        
        for idx in indices:
            length = self.lengths[idx]
            # Check if adding this sample exceeds max_tokens
            # Usually max_tokens logic includes some overhead or padding calculation
            # data_utils.batch_by_size uses: (len(batch) + 1) * max(len(s) for s in batch) if padding is considered
            # But simpler logic: just sum of lengths or max_len * batch_size
            
            # Let's implement a simple max_tokens bucket strategy:
            # If we add this sample, will the batch size (in tokens) exceed max_tokens?
            # We approximate batch size as: max_len_in_batch * num_samples (standard for Transformers/Fairseq)
            
            new_max_len = max(length, max([self.lengths[i] for i in current_batch]) if current_batch else 0)
            new_batch_size = new_max_len * (len(current_batch) + 1)
            
            if new_batch_size > max_tokens and len(current_batch) > 0:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
        
        if current_batch:
            batches.append(current_batch)

        num_good_batches = math.floor(len(batches) / self.num_replicas) * self.num_replicas
        total_batches = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
        
        while sum(len(batch) for batch in batches[num_good_batches:]) < total_batches - num_good_batches:
            num_good_batches -= self.num_replicas
        
        new_batches = batches[:num_good_batches]
        reallocate_batches = batches[num_good_batches:]
        reallocate_batches.extend([[] for _ in range(total_batches - len(batches))])
        
        i, j = 0, len(reallocate_batches) - 1
        while i < j:
            while len(reallocate_batches[i]) <= 1 and i < j:
                i = i + 1
            while len(reallocate_batches[j]) > 0 and i < j:
                j = j - 1
            if i >= j:
                break 
            reallocate_batches[j] = [reallocate_batches[i][0]]
            reallocate_batches[i] = reallocate_batches[i][1:]
        new_batches.extend(reallocate_batches)
        assert (len(new_batches) % self.num_replicas == 0)
        batches = new_batches[self.rank : len(new_batches) : self.num_replicas]
        i, num_batches = 0, len(batches)
        self.num_batches = num_batches

        while True:
            yield batches[i]
            i = (i + 1) % num_batches
            if not self.infinite and i <= 0:
                break


class AMRDataset(BilingualDataset):
    """Dataset for AMR parsing, preprocessing the AMR graph"""
    
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        from .amr_process.prepare_dataset import prepare_dataset
        from .amr_process.additional_tokens import get_added_vocabulary
        import os
        import logging
        
        # Set tokenizer model_max_length from config and suppress warnings during filtering
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        # Add additional tokens
        new_tokens = get_added_vocabulary()
        num_added_toks = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added_toks} AMR tokens to tokenizer")
        
        # Prepare dataset
        print(f"Preparing AMR dataset from {args.data_path}...")
        datasets = prepare_dataset(
            dataset_name=args.data_path,
            src_column=args.src_column,
            tgt_column=args.tgt_column,
            output_dir=None, # In-memory processing
            dedupe=getattr(args, 'dedupe', False),
            remove_wiki=getattr(args, 'remove_wiki', True),
            fix_ftfy=getattr(args, 'fix_ftfy', False),
            normalize_punct=getattr(args, 'normalize_punct', False),
            detokenize=getattr(args, 'detokenize', False),
            remove_bracketed=getattr(args, 'remove_bracketed', False),
            dereify=getattr(args, 'dereify', False),
            lang=getattr(args, 'src_lang', 'vi') or 'vi',
            cache_dir=getattr(args, 'cache_dir', None)
        )
        
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        num_proc = max(1, int(mp.cpu_count() / world_size))

        # Filter datasets based on length
        def filter_fn(example):
            src = example[args.src_column]
            tgt = example[args.tgt_column]
            if not src or not tgt: return False
            
            # Tokenize to check length
            s_ids = tokenizer.encode(src, add_special_tokens=True)
            t_ids = tokenizer.encode(tgt, add_special_tokens=True)
            
            if len(s_ids) > 0 and s_ids[-1] == tokenizer.eos_token_id:
                s_ids = s_ids[:-1]
            if len(t_ids) > 0 and t_ids[0] == tokenizer.bos_token_id:
                t_ids = t_ids[1:]
                
            return (len(s_ids) + len(t_ids)) <= args.max_length

        if train and "train" in datasets:
            datasets["train"] = datasets["train"].filter(filter_fn, num_proc=num_proc)
        
        if valid and "validation" in datasets:
            print(f"Filtering validation dataset (max_length={args.max_length})...")
            datasets["validation"] = datasets["validation"].filter(filter_fn, num_proc=num_proc)

        if test and "test" in datasets:
            print(f"Filtering test dataset (max_length={args.max_length})...")
            datasets["test"] = datasets["test"].filter(filter_fn, num_proc=num_proc)

        # Create dataset splits
        train_dataset = AMRDataset(args, datasets["train"], tokenizer) if train and "train" in datasets else None
        valid_dataset = AMRDataset(args, datasets["validation"], tokenizer) if valid and "validation" in datasets else None
        test_dataset = AMRDataset(args, datasets["test"], tokenizer) if test and "test" in datasets else None
        
        return train_dataset, valid_dataset, test_dataset


def _decode_wav_bytes(wav_bytes):
    """Decode raw WAV bytes to float32 numpy array using python's built-in wave module.
    This avoids dependency on torchcodec/soundfile/librosa."""
    f = wave.open(io.BytesIO(wav_bytes), 'rb')
    n_channels = f.getnchannels()
    sampwidth = f.getsampwidth()
    n_frames = f.getnframes()
    sample_rate = f.getframerate()
    raw_frames = f.readframes(n_frames)
    f.close()
    
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    data = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)
    
    # Normalize to [-1.0, 1.0]
    if sampwidth == 2:
        data = data / 32768.0
    elif sampwidth == 4:
        data = data / 2147483648.0
    
    # Convert stereo to mono by averaging channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    
    return data, sample_rate


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
        import os
        import logging
        
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
        if valid:
            valid_raw = valid_raw.filter(filter_fn, num_proc=num_proc)
        
        print(f"After filtering: {len(train_raw)} train, {len(valid_raw)} validation samples")
        
        train_dataset = SpeechDataset(args, train_raw, tokenizer, feature_extractor) if train else None
        valid_dataset = SpeechDataset(args, valid_raw, tokenizer, feature_extractor) if valid else None
        test_dataset = None  # No test set for now
        
        return train_dataset, valid_dataset, test_dataset
