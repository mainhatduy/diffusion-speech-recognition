"""Memory-mapped dataset implementation for efficient large token file reading."""

import numpy as np
import torch
from torch.utils.data import Dataset


class MemoryMapTokensDataset(Dataset):
    """Dataset reading token IDs directly from binary memory-mapped files."""

    def __init__(self, args, data_path, tokenizer):
        """Initialize memory-mapped token dataset."""
        super().__init__()
        self.args = args
        self.tokens = np.memmap(data_path, dtype="ushort", mode="r")
        self.num_total_tokens = self.tokens.shape[0]
        self.length = args.max_length

    def __len__(self):
        """Return number of token chunks of max_length."""
        return self.num_total_tokens // self.length

    def __getitem__(self, index):
        """Read a slice of token IDs from the memory-mapped file."""
        start, end = index * self.length, (index + 1) * self.length
        data = np.array(self.tokens[start:end], dtype=int)
        return {"id": index, "source": torch.tensor(data), "target": torch.tensor(data)}

    @staticmethod
    def load_data(args, tokenizer, train=True, valid=False, test=False):
        """Load memory-mapped token dataset split."""
        assert not test
        return (
            MemoryMapTokensDataset(args, args.data_path, tokenizer=tokenizer),
            None,
            None,
        )
