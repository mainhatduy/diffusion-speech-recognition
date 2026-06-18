import os
import logging
import multiprocessing as mp
import torch
from datasets import load_dataset, load_from_disk
from .base import PromptDataset

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
        train_dataset = BilingualDataset(args, datasets["train"], tokenizer) if train and "train" in datasets else None
        valid_dataset = BilingualDataset(args, datasets["validation"], tokenizer) if valid and "validation" in datasets else None
        test_dataset = BilingualDataset(args, datasets["test"], tokenizer) if test and "test" in datasets else None
        
        return (train_dataset, valid_dataset, test_dataset)
