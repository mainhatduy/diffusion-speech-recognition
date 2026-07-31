from dataclasses import dataclass

import torch


@dataclass
class DiscreteDiffusionDataCollator:
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
            "partial_masks": partial_masks,
        }

        # Handle audio features if present (for speech_recognition)
        has_audio = "audio_values" in samples[0]
        has_precomputed = "precomputed_audio_embeds" in samples[0]
        if has_precomputed:
            # Pre-computed audio embeddings: 2D tensors (T_frames, D_audio)
            embeds_list = [s["precomputed_audio_embeds"] for s in samples]
            max_T = max(e.size(0) for e in embeds_list)
            D = embeds_list[0].size(1)
            padded_embeds = torch.zeros(batch_size, max_T, D)
            embed_mask = torch.zeros(batch_size, max_T, dtype=torch.long)
            for i, e in enumerate(embeds_list):
                T = e.size(0)
                padded_embeds[i, :T, :] = e
                embed_mask[i, :T] = 1
            net_input["precomputed_audio_embeds"] = padded_embeds
            net_input["precomputed_audio_mask"] = embed_mask
        elif has_audio:
            audio_values_list = [s["audio_values"] for s in samples]
            # Pad audio to max length in batch, rounded up to a multiple of 80 (frame_len of Moonshine)
            max_audio_len = max(av.size(-1) for av in audio_values_list)
            max_audio_len = ((max_audio_len + 79) // 80) * 80
            padded_audio = torch.zeros(batch_size, max_audio_len)
            audio_attention_mask = torch.zeros(
                batch_size, max_audio_len, dtype=torch.long
            )
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


@dataclass
class StreamingCollator:
    """
    Collator for StreamingAugmentedDataset.

    Handles batching of:
    - Variable-length audio chunks (padded to max chunks in batch)
    - Variable-length text sequences (padded to max text length)
    - Support ratios and supported text lengths

    Output keys:
        audio_chunks: [B, max_chunks, chunk_samples]
        audio_mask: [B, max_chunks] — True where chunk is real (not padding)
        text_ids: [B, max_text_len]
        text_mask: [B, max_text_len] — True where token is real
        support_ratios: [B]
        supported_lens: [B]
        task_token_ids: [B] or None
    """

    pad_token_id: int

    def __call__(self, batch):
        # Filter None samples
        batch = [s for s in batch if s is not None]
        if len(batch) == 0:
            return {}

        B = len(batch)

        # --- Audio chunks ---
        max_chunks = max(s["audio_chunks"].shape[0] for s in batch)
        chunk_len = batch[0]["audio_chunks"].shape[1]

        audio_batch = torch.zeros(B, max_chunks, chunk_len)
        audio_mask = torch.zeros(B, max_chunks, dtype=torch.bool)

        for i, s in enumerate(batch):
            n = s["audio_chunks"].shape[0]
            audio_batch[i, :n] = s["audio_chunks"]
            audio_mask[i, :n] = True

        # --- Text ---
        max_text = max(len(s["text_ids"]) for s in batch)
        text_batch = torch.full((B, max_text), self.pad_token_id, dtype=torch.long)
        text_mask = torch.zeros(B, max_text, dtype=torch.bool)

        for i, s in enumerate(batch):
            n = len(s["text_ids"])
            text_batch[i, :n] = s["text_ids"]
            text_mask[i, :n] = True

        # --- Support info ---
        support_ratios = torch.tensor([s["support_ratio"] for s in batch])
        supported_lens = torch.tensor([s["supported_text_len"] for s in batch])
        num_visible_chunks = torch.tensor([s["num_visible_chunks"] for s in batch])
        total_chunks = torch.tensor([s["total_chunks"] for s in batch])

        # --- Task token IDs ---
        task_token_ids = None
        if batch[0].get("task_token_id") is not None:
            task_token_ids = torch.tensor(
                [s["task_token_id"] for s in batch], dtype=torch.long
            )

        out = {
            "audio_chunks": audio_batch,
            "audio_mask": audio_mask,
            "text_ids": text_batch,
            "text_mask": text_mask,
            "support_ratios": support_ratios,
            "supported_lens": supported_lens,
            "num_visible_chunks": num_visible_chunks,
            "total_chunks": total_chunks,
            "task_token_ids": task_token_ids,
        }
        
        if batch[0].get("is_precomputed"):
            out["is_precomputed"] = True
            
        return out
