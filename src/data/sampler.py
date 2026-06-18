import math
import torch
from typing import List, Optional, Iterator
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

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
        indices = [index for index in indices if self.lengths[index] <= self.max_length]
        
        batches = []
        current_batch = []
        
        max_tokens = self.batch_size
        
        for idx in indices:
            length = self.lengths[idx]
            new_max_len = max(length, max([self.lengths[i] for i in current_batch]) if current_batch else 0)
            new_batch_size = new_max_len * (len(current_batch) + 1)
            
            if new_batch_size > max_tokens and len(current_batch) > 0:
                batches.append(current_batch)
                current_batch = []
            
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
