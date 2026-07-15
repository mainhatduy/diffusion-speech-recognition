import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# Re-exporting all dataset components for backward compatibility
from .base import PromptDataset
from .bilingual import BilingualDataset
from .memory_map import MemoryMapTokensDataset
from .speech import SpeechDataset
from .translated_speech import TranslatedSpeechDataset
from .multitask import MultiTaskTranslatedSpeechDataset
from .precomputed_multitask import PrecomputedMultiTaskDataset
from .collator import DiscreteDiffusionDataCollator
from .sampler import TokenSizeDistributedLengthGroupSampler
from .utils import normalize_text, _decode_wav_bytes

# Legacy placeholders to avoid breaking imports
PairDataset = BilingualDataset
AMRDataset = BilingualDataset


@dataclass
class DiscreteDiffusionDataArguments:
    dataset_type: str = field(
        default="bilingual"  # bilingual | speech_recognition | speech_translation | speech_translation_multitask
    )
    audio_encoder_name: str = field(
        default="UsefulSensors/moonshine-streaming-medium",
        metadata={"help": "pretrained audio encoder model name for speech_recognition"},
    )
    data_path: str = field(default="")
    src_lang: str = field(default="")
    tgt_lang: str = field(default="")
    max_length: int = field(default=2048)
    packing: bool = field(
        default=False, metadata={"help": "whether to pack the output data"}
    )
    hf_token: str = field(
        default=None, metadata={"help": "Hugging Face token for private datasets"}
    )
    src_column: str = field(
        default="en", metadata={"help": "Source column name in dataset"}
    )
    tgt_column: str = field(
        default="vi", metadata={"help": "Target column name in dataset"}
    )
    dedupe: bool = field(
        default=False, metadata={"help": "whether to deduplicate the data"}
    )
    remove_wiki: bool = field(
        default=False, metadata={"help": "whether to remove wiki from the AMR entries"}
    )
    fix_ftfy: bool = field(
        default=False, metadata={"help": "whether to fix text issues"}
    )
    normalize_punct: bool = field(
        default=False, metadata={"help": "whether to normalize punctuation"}
    )
    detokenize: bool = field(default=False, metadata={"help": "whether to detokenize"})
    remove_bracketed: bool = field(
        default=False,
        metadata={
            "help": "whether to remove sentences that start and end with punctuation"
        },
    )
    dereify: bool = field(
        default=False, metadata={"help": "whether to dereify AMR graph"}
    )
    task_tokens: List[str] = field(
        default_factory=lambda: ["<vi_en>", "<vi_zh>", "<vi_ko>"],
        metadata={
            "help": "List of task token strings (with <>) for multi-task speech translation. E.g. ['<vi_en>', '<vi_zh>', '<vi_ko>']"
        },
    )
    precomputed_data_dir: str = field(
        default="",
        metadata={
            "help": "Path to pre-computed audio embeddings & token IDs. If set, uses fast PrecomputedMultiTaskDataset."
        },
    )


def load_data(
    data_args: DiscreteDiffusionDataArguments,
    model_args,
    tokenizer,
    train: bool = True,
    valid: bool = True,
    test: bool = False,
) -> Tuple[
    Tuple[Optional[PromptDataset], Optional[PromptDataset], Optional[PromptDataset]],
    DiscreteDiffusionDataCollator,
]:
    """Unified dataset loader entry point. Sets appropriate configs and returns data splits and the collator."""
    setattr(data_args, "cache_dir", model_args.cache_dir)

    # Dispatch dataset loading
    if data_args.dataset_type in ["bilingual", "pair"]:
        datasets = BilingualDataset.load_data(data_args, tokenizer, train, valid, test)
    elif data_args.dataset_type == "speech_recognition":
        if (
            not hasattr(data_args, "audio_encoder_name")
            or not data_args.audio_encoder_name
        ):
            data_args.audio_encoder_name = getattr(
                model_args, "audio_encoder_name", "facebook/mms-300m"
            )
        datasets = SpeechDataset.load_data(data_args, tokenizer, train, valid, test)
    elif data_args.dataset_type == "speech_translation":
        if (
            not hasattr(data_args, "audio_encoder_name")
            or not data_args.audio_encoder_name
        ):
            data_args.audio_encoder_name = getattr(
                model_args, "audio_encoder_name", "facebook/mms-300m"
            )
        datasets = TranslatedSpeechDataset.load_data(
            data_args, tokenizer, train, valid, test
        )
    elif data_args.dataset_type == "speech_translation_multitask":
        if (
            not hasattr(data_args, "audio_encoder_name")
            or not data_args.audio_encoder_name
        ):
            data_args.audio_encoder_name = getattr(
                model_args,
                "audio_encoder_name",
                "UsefulSensors/moonshine-streaming-medium",
            )
        # Use precomputed dataset if available
        if getattr(data_args, "precomputed_data_dir", "") and os.path.exists(
            data_args.precomputed_data_dir
        ):
            print(
                f"[load_data] Using PrecomputedMultiTaskDataset from '{data_args.precomputed_data_dir}'"
            )
            datasets = PrecomputedMultiTaskDataset.load_data(
                data_args, tokenizer, train, valid, test
            )
        else:
            datasets = MultiTaskTranslatedSpeechDataset.load_data(
                data_args, tokenizer, train, valid, test
            )
    else:
        raise ValueError(
            f"Unknown or unsupported dataset type: {data_args.dataset_type}"
        )

    # Build collator
    collator = DiscreteDiffusionDataCollator(
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
    )

    return datasets, collator
