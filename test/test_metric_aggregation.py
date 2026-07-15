import numpy as np
import torch

# Let's simulate what Hugging Face Trainer does when prediction_step returns 1D vs 2D tensors.

# Let's say we have 3 batches.
# Each batch computes BLEU (5 stats) and WER (1 stat).
# Total slots: 6.
active_metrics = ["bleu", "wer"]
_METRIC_SLOT_SIZES = {
    "bleu": 5,
    "wer": 1,
}

print("--- Scenario 1: 1D tensors (Current Implementation) ---")
# Current implementation returns 1D tensors:
batch_outputs_1d = [
    np.array(
        [1.0, 2.0, 3.0, 4.0, 10.0, 0.5]
    ),  # Batch 1: BLEU count=1,2,3,4 len=10, WER edit_dist=0.5
    np.array(
        [1.5, 2.5, 3.5, 4.5, 11.0, 0.6]
    ),  # Batch 2: BLEU count=1.5,2.5,3.5,4.5 len=11, WER edit_dist=0.6
    np.array(
        [1.8, 2.8, 3.8, 4.8, 12.0, 0.7]
    ),  # Batch 3: BLEU count=1.8,2.8,3.8,4.8 len=12, WER edit_dist=0.7
]

# Trainer accumulates by concatenating along dimension 0.
gathered_1d = np.concatenate(batch_outputs_1d, axis=0)
print("Gathered 1D array:", gathered_1d)
print("Gathered 1D shape:", gathered_1d.shape)

# Slicing in MultiMetric:
offset = 0
sliced_metrics_1d = {}
for metric in active_metrics:
    size = _METRIC_SLOT_SIZES[metric]
    sys_slice = gathered_1d[..., offset : offset + size]
    sliced_metrics_1d[metric] = sys_slice
    print(f"Slice for {metric}:", sys_slice)
    offset += size

print("\n--- Scenario 2: 2D tensors (Proposed Fix) ---")
# Proposed implementation returns 2D tensors of shape [1, 6]:
batch_outputs_2d = [
    np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 0.5]]),  # Shape [1, 6]
    np.array([[1.5, 2.5, 3.5, 4.5, 11.0, 0.6]]),  # Shape [1, 6]
    np.array([[1.8, 2.8, 3.8, 4.8, 12.0, 0.7]]),  # Shape [1, 6]
]

# Trainer accumulates by concatenating along dimension 0.
gathered_2d = np.concatenate(batch_outputs_2d, axis=0)
print("Gathered 2D array:\n", gathered_2d)
print("Gathered 2D shape:", gathered_2d.shape)

# Slicing in MultiMetric:
offset = 0
sliced_metrics_2d = {}
for metric in active_metrics:
    size = _METRIC_SLOT_SIZES[metric]
    sys_slice = gathered_2d[..., offset : offset + size]
    sliced_metrics_2d[metric] = sys_slice
    print(f"Slice for {metric}:\n", sys_slice)
    offset += size

# Test reshape and sum for BLEU:
bleu_slice = sliced_metrics_2d["bleu"]
bleu_summed = bleu_slice.reshape(-1, 5).astype("long").sum(0).tolist()
print("\nSummed BLEU stats (2D):", bleu_summed)

# Test sum for WER:
wer_slice = sliced_metrics_2d["wer"]
wer_summed = wer_slice.astype("float64").sum()
print("Summed WER stats (2D):", wer_summed)
