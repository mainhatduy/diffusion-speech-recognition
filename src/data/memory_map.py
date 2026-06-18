import numpy as np
import torch
from torch.utils.data import Dataset

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
