import torch
from dataclasses import dataclass

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
