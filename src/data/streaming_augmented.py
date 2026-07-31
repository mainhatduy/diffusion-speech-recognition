"""
Streaming Augmented Dataset.

Wraps existing (full_audio, full_text) datasets to simulate streaming conditions
during training:

1. Split audio into 2s chunks with 0.5s overlap
2. Randomly select k ∈ {1, ..., N} visible chunks (simulate streaming progress)
3. Return FULL text target + support_ratio + supported_text_len
4. Support curriculum training (progressively increase max visible chunks)

This data augmentation ensures the model learns to:
- Produce partial outputs when only partial audio is available
- Be uncertain (low confidence) in unsupported regions
- Progressively improve output as more audio arrives
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class StreamingAugmentedDataset(Dataset):
    """
    Wraps a base dataset to produce streaming-augmented training samples.

    Each call to __getitem__ randomly selects a "streaming moment" —
    how many audio chunks the model has seen so far — producing
    different training signals from the same base sample.

    Args:
        base_dataset: Source dataset providing {"audio": Tensor, "text_ids": Tensor, "task_token_id": int}
        chunk_duration: Duration of each audio chunk in seconds
        overlap_duration: Overlap between consecutive chunks in seconds
        sample_rate: Audio sample rate (Hz)
        max_text_length: Maximum text sequence length
        curriculum_step: Current training step (-1 = no curriculum)
        total_curriculum_steps: Total steps for curriculum completion
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.5,
        sample_rate: int = 16000,
        max_text_length: int = 256,
        curriculum_step: int = -1,
        total_curriculum_steps: int = 100000,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length

        # Chunk sizes in samples
        self.chunk_samples = int(chunk_duration * sample_rate)       # 32000
        self.overlap_samples = int(overlap_duration * sample_rate)   # 8000
        self.hop_samples = self.chunk_samples - self.overlap_samples # 24000

        # Curriculum state
        self.curriculum_step = curriculum_step
        self.total_curriculum_steps = total_curriculum_steps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        # Extract audio and text from base dataset
        # The base dataset should provide raw audio waveform and tokenized text
        audio = sample["audio"]        # 1D Tensor [total_samples]
        text_ids = sample["text_ids"]  # 1D Tensor [text_len]
        task_token_id = sample.get("task_token_id", None)

        if isinstance(audio, list):
            audio = torch.tensor(audio, dtype=torch.float)
        if isinstance(text_ids, list):
            text_ids = torch.tensor(text_ids, dtype=torch.long)

        # Truncate text
        if len(text_ids) > self.max_text_length:
            text_ids = text_ids[: self.max_text_length]

        # 1. Split audio into chunks
        total_samples = len(audio)
        num_chunks = max(1, (total_samples - self.overlap_samples) // self.hop_samples)

        audio_chunks = []
        for i in range(num_chunks):
            start = i * self.hop_samples
            end = start + self.chunk_samples
            chunk = audio[start:end]
            # Pad last chunk if needed
            if len(chunk) < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - len(chunk)))
            audio_chunks.append(chunk)

        # 2. Curriculum: limit maximum visible chunks
        if self.curriculum_step >= 0 and self.total_curriculum_steps > 0:
            progress = self.curriculum_step / self.total_curriculum_steps
            max_visible = max(1, int(progress * num_chunks))
        else:
            max_visible = num_chunks

        # 3. Randomly select number of visible chunks
        num_visible = random.randint(1, max(1, max_visible))

        # 4. Stack visible chunks
        visible_chunks = torch.stack(audio_chunks[:num_visible])  # [num_visible, chunk_samples]

        # 5. Compute support ratio and supported text length
        support_ratio = num_visible / num_chunks
        supported_text_len = int(support_ratio * len(text_ids))

        return {
            "audio_chunks": visible_chunks,          # [num_visible, chunk_samples]
            "text_ids": text_ids,                    # [text_len] — FULL text
            "support_ratio": support_ratio,          # float in (0, 1]
            "supported_text_len": supported_text_len,# int
            "num_visible_chunks": num_visible,       # int
            "total_chunks": num_chunks,              # int
            "task_token_id": task_token_id,          # int or None
        }
