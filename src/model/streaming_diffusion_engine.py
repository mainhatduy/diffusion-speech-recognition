"""
Streaming Diffusion Engine — 3-Zone Management.

Manages the entire streaming diffusion process with three zones:

┌────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ FROZEN     │  │ ACTIVE           │  │ FUTURE             │
│ (KV Cache) │  │ (Denoise Zone)   │  │ (MASK Placeholder) │
│ ≤128 tok   │  │ ≤64 tok          │  │ No compute         │
└────────────┘  └──────────────────┘  └────────────────────┘

Key operations:
- on_new_audio_chunk(): Extend active zone, denoise, freeze confident tokens
- _denoise_active_zone(): K=3 denoise steps, remask bottom 30% (except last step)
- _freeze_confident_tokens(): Monotonic left-to-right freeze at θ=0.92
- flush(): End-of-stream cleanup with 10 denoise steps
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class StreamingDiffusionEngine:
    """
    Manages the 3-zone streaming diffusion process.

    This class is stateful — it tracks frozen/active zones across audio chunks.
    Call reset() between samples.
    """

    def __init__(
        self,
        mask_id: int,
        eos_id: int,
        pad_id: int,
        # Zone sizes
        active_window_size: int = 64,
        frozen_cache_size: int = 128,
        # Denoise parameters
        streaming_denoise_steps: int = 3,
        # Confidence thresholds
        freeze_confidence_threshold: float = 0.92,
        remask_confidence_threshold: float = 0.30,
        # Device
        device: torch.device | str = "cpu",
    ):
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.active_window_size = active_window_size
        self.frozen_cache_size = frozen_cache_size
        self.streaming_denoise_steps = streaming_denoise_steps
        self.freeze_confidence_threshold = freeze_confidence_threshold
        self.remask_confidence_threshold = remask_confidence_threshold
        self.device = torch.device(device)

        # State (initialized in reset())
        self.reset()

    def reset(self):
        """Reset all state for a new sample."""
        # === FROZEN ZONE ===
        self.frozen_tokens: list[int] = []
        self.frozen_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        # === ACTIVE ZONE ===
        self.active_tokens: list[int] = []
        self.active_confidence: list[float] = []

        # === AUDIO BUFFER ===
        self.audio_buffer: list[torch.Tensor] = []

        # === COUNTERS ===
        self.total_audio_chunks_received: int = 0
        self.total_tokens_yielded: int = 0

    def on_new_audio_chunk(
        self,
        audio_embeds: torch.Tensor,
        backbone: torch.nn.Module,
        audio_adapter: torch.nn.Module,
        length_predictor: torch.nn.Module,
    ) -> list[int]:
        """
        Process a new audio chunk.

        Args:
            audio_embeds: [frames, D_audio] — from Moonshine encoder
            backbone: StreamingDiffusionBackbone
            audio_adapter: StreamingAudioAdapter
            length_predictor: StreamingLengthPredictor

        Returns:
            list of token IDs that were frozen (yielded as output)
        """
        self.total_audio_chunks_received += 1

        # 1. Project audio through adapter
        audio_projected = audio_adapter(audio_embeds.unsqueeze(0))  # [1, tokens, D_text]
        self.audio_buffer.append(audio_projected.squeeze(0))  # [tokens, D_text]

        # 2. Predict number of new text tokens for this chunk
        context_ids = self.frozen_tokens[-32:] if self.frozen_tokens else None
        if context_ids is not None:
            ctx_tensor = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        else:
            ctx_tensor = None
        num_new_tokens = length_predictor.predict_chunk(audio_projected, ctx_tensor)
        num_new_tokens = max(1, min(num_new_tokens, 20))  # Clamp

        # 3. Extend active zone with MASK tokens
        self.active_tokens.extend([self.mask_id] * num_new_tokens)
        self.active_confidence.extend([0.0] * num_new_tokens)

        # 4. Truncate active zone if too long
        if len(self.active_tokens) > self.active_window_size:
            self.active_tokens = self.active_tokens[: self.active_window_size]
            self.active_confidence = self.active_confidence[: self.active_window_size]

        # 5. Denoise active zone
        self._denoise_active_zone(backbone)

        # 6. Freeze confident tokens
        newly_frozen = self._freeze_confident_tokens(backbone)

        return newly_frozen

    def _get_audio_context(self) -> torch.Tensor | None:
        """Get concatenated audio context (last 2 chunks) for cross-attention."""
        if not self.audio_buffer:
            return None

        # Keep last 2 chunks for context
        recent = self.audio_buffer[-2:]
        audio_context = torch.cat(recent, dim=0)  # [total_frames, D_text]
        return audio_context.unsqueeze(0)  # [1, total_frames, D_text]

    @torch.no_grad()
    def _denoise_active_zone(self, backbone: torch.nn.Module):
        """
        Run K denoise steps on the active zone only.

        Uses KV cache from frozen zone for context.
        Remasking: bottom 30% confidence tokens → MASK (except on last step).
        """
        if not self.active_tokens:
            return

        K = self.streaming_denoise_steps
        audio_context = self._get_audio_context()

        for step in range(K):
            # Prepare input
            active_ids = torch.tensor(
                [self.active_tokens], dtype=torch.long, device=self.device
            )

            # Forward through backbone with frozen KV cache
            logits, _ = backbone(
                input_ids=active_ids,
                past_kv_caches=self.frozen_kv_caches,
                audio_hidden=audio_context,
            )
            # logits: [1, active_len, vocab_size]

            # Never predict MASK token
            logits[:, :, self.mask_id] = float("-inf")

            # Get predictions and confidence
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)  # [1, active_len]

            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()

            # Remask: bottom-k least confident → MASK (except on final step)
            if step < K - 1:
                num_remask = int(len(predicted) * self.remask_confidence_threshold)
                if num_remask > 0:
                    conf_tensor = torch.tensor(confidence)
                    _, remask_indices = conf_tensor.topk(num_remask, largest=False)

                    for idx in remask_indices.tolist():
                        predicted[idx] = self.mask_id
                        confidence[idx] = 0.0

            # Update active zone
            self.active_tokens = predicted
            self.active_confidence = confidence

    @torch.no_grad()
    def _freeze_confident_tokens(self, backbone: torch.nn.Module) -> list[int]:
        """
        Freeze tokens that exceed confidence threshold θ.

        Strategy: monotonic_left — only freeze from left, stop at first uncertain token.
        This prevents outputting tokens out of order.

        Returns:
            list of newly frozen token IDs
        """
        θ = self.freeze_confidence_threshold

        # Count consecutive confident tokens from the left
        freeze_count = 0
        for token, conf in zip(self.active_tokens, self.active_confidence):
            if conf >= θ and token != self.mask_id:
                freeze_count += 1
            else:
                break  # Monotonic: stop at first uncertain token

        if freeze_count == 0:
            return []

        # 1. Split tokens
        newly_frozen_tokens = self.active_tokens[:freeze_count]
        self.active_tokens = self.active_tokens[freeze_count:]
        self.active_confidence = self.active_confidence[freeze_count:]

        # 2. Compute KV cache for newly frozen tokens
        frozen_ids = torch.tensor(
            [newly_frozen_tokens], dtype=torch.long, device=self.device
        )
        _, new_kv = backbone(
            input_ids=frozen_ids,
            past_kv_caches=self.frozen_kv_caches,
            audio_hidden=None,  # Frozen tokens don't need audio anymore
        )

        # 3. Merge into frozen KV cache
        if self.frozen_kv_caches is None:
            self.frozen_kv_caches = new_kv
        else:
            self.frozen_kv_caches = [
                (
                    torch.cat([old_k, new_k], dim=2),
                    torch.cat([old_v, new_v], dim=2),
                )
                for (old_k, old_v), (new_k, new_v) in zip(
                    self.frozen_kv_caches, new_kv
                )
            ]

        # 4. Update frozen tokens list
        self.frozen_tokens.extend(newly_frozen_tokens)

        # 5. Evict oldest tokens if cache exceeds max size
        max_cache = self.frozen_cache_size
        if len(self.frozen_tokens) > max_cache:
            excess = len(self.frozen_tokens) - max_cache
            self.frozen_tokens = self.frozen_tokens[excess:]

            # Trim KV cache
            self.frozen_kv_caches = [
                (k[:, :, excess:, :], v[:, :, excess:, :])
                for k, v in self.frozen_kv_caches
            ]

        self.total_tokens_yielded += freeze_count
        return newly_frozen_tokens

    @torch.no_grad()
    def flush(self, backbone: torch.nn.Module) -> list[int]:
        """
        End-of-stream: denoise remaining active tokens with more steps.

        Called when audio stream ends. Uses 10 denoise steps (offline quality)
        and freezes everything regardless of confidence.

        Returns:
            list of remaining token IDs
        """
        if not self.active_tokens:
            return []

        # Add EOS at the end
        self.active_tokens.append(self.eos_id)
        self.active_confidence.append(1.0)

        # Denoise with more steps for final quality
        original_steps = self.streaming_denoise_steps
        self.streaming_denoise_steps = 10
        self._denoise_active_zone(backbone)
        self.streaming_denoise_steps = original_steps

        # Freeze everything (no confidence check)
        remaining = self.active_tokens.copy()
        self.active_tokens = []
        self.active_confidence = []
        self.frozen_tokens.extend(remaining)

        return remaining
