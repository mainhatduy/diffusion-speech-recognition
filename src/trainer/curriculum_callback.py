"""
Curriculum Training Callback for Streaming Diffusion.

Implements a 4-stage curriculum that progressively increases the difficulty
of streaming training by allowing more audio chunks to be visible:

    Stage 1 (0–20%  progress): max 20% of chunks visible → learn basic mapping
    Stage 2 (20–50% progress): max 50% of chunks visible → learn context
    Stage 3 (50–80% progress): max 80% of chunks visible → learn long-range
    Stage 4 (80–100% progress): 100% of chunks visible   → learn full sequence

This is critical because:
- Starting with full audio = model never learns partial-input behavior
- Starting with too few chunks = model can't learn audio-text alignment
- Gradual increase = stable curriculum learning
"""

from __future__ import annotations

from transformers.trainer_callback import TrainerCallback


class CurriculumCallback(TrainerCallback):
    """
    Adjusts the StreamingAugmentedDataset's curriculum_step each training step.

    The dataset uses curriculum_step to limit max_visible_chunks,
    progressively allowing more audio context during training.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """Update curriculum step in the training dataset."""
        train_dataloader = kwargs.get("train_dataloader")
        if train_dataloader is None:
            return

        dataset = getattr(train_dataloader, "dataset", None)
        if dataset is None:
            return

        # Support wrapped datasets (e.g., IterableDatasetShard)
        actual_dataset = dataset
        while hasattr(actual_dataset, "dataset"):
            actual_dataset = actual_dataset.dataset

        if hasattr(actual_dataset, "curriculum_step"):
            actual_dataset.curriculum_step = state.global_step
            actual_dataset.total_curriculum_steps = state.max_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log curriculum progress for monitoring."""
        if logs is None or state.max_steps <= 0:
            return

        progress = state.global_step / state.max_steps

        if progress < 0.2:
            stage = "Stage 1 (basic mapping)"
            max_ratio = 0.2
        elif progress < 0.5:
            stage = "Stage 2 (context learning)"
            max_ratio = 0.5
        elif progress < 0.8:
            stage = "Stage 3 (long-range)"
            max_ratio = 0.8
        else:
            stage = "Stage 4 (full sequence)"
            max_ratio = 1.0

        logs["curriculum_stage"] = stage
        logs["curriculum_max_chunks_ratio"] = max_ratio
