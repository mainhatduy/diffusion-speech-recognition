import os
from dataclasses import dataclass, field

# Re-exporting all dataset components for backward compatibility
from .base import PromptDataset
from .bilingual import BilingualDataset
from .collator import DiscreteDiffusionDataCollator
from .multitask import MultiTaskTranslatedSpeechDataset
from .precomputed_multitask import PrecomputedMultiTaskDataset
from .speech import SpeechDataset
from .translated_speech import TranslatedSpeechDataset

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
    task_tokens: list[str] = field(
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
    stream_dataset_from_hub: bool = field(
        default=False,
        metadata={
            "help": "Stream data from Hugging Face Hub instead of downloading/loading locally."
        },
    )
    streaming_repo_id: str = field(
        default="aiai-laboratory/vietspeech-train-streaming",
        metadata={"help": "HF Hub repo ID for streaming dataset."},
    )
    streaming_buffer_size: int = field(
        default=1000,
        metadata={"help": "Shuffle buffer size for streaming dataset."},
    )
    val_streaming_size: int = field(
        default=500,
        metadata={"help": "Number of validation samples for streaming dataset."},
    )
    use_ram_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache raw audio bytes in RAM if free RAM > threshold_ratio."},
    )
    ram_free_threshold_ratio: float = field(
        default=0.30,
        metadata={"help": "Minimum required free RAM ratio after preloading audio bytes into RAM."},
    )
    enable_streaming_architecture: bool = field(
        default=False,
        metadata={
            "help": "Enable chunk augmentation, the streaming backbone, and the streaming trainer."
        },
    )
    curriculum_training: bool = field(
        default=False,
        metadata={"help": "Whether to enable curriculum training for streaming augmentation."}
    )
    audio_chunk_duration: float = field(
        default=2.0,
        metadata={"help": "Audio chunk duration in seconds for streaming augmentation."}
    )
    audio_overlap_duration: float = field(
        default=0.5,
        metadata={"help": "Audio overlap duration in seconds for streaming augmentation."}
    )


def load_data(
    data_args: DiscreteDiffusionDataArguments,
    model_args,
    tokenizer,
    train: bool = True,
    valid: bool = True,
    test: bool = False,
) -> tuple[
    tuple[PromptDataset | None, PromptDataset | None, PromptDataset | None],
    DiscreteDiffusionDataCollator,
]:
    """Unified dataset loader entry point. Sets appropriate configs and returns data splits and the collator."""
    data_args.cache_dir = model_args.cache_dir

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
        # Stream the dataset from Hugging Face Hub if enabled.
        if getattr(data_args, "stream_dataset_from_hub", False):
            print(
                f"[load_data] Using StreamingPrecomputedMultiTaskDataset from HF repo '{data_args.streaming_repo_id}'"
            )
            from .streaming_precomputed_multitask import (
                StreamingPrecomputedMultiTaskDataset,
            )

            datasets = StreamingPrecomputedMultiTaskDataset.load_data(
                data_args, tokenizer, train, valid, test
            )
        # Use precomputed local dataset if available
        elif getattr(data_args, "precomputed_data_dir", "") and os.path.exists(
            data_args.precomputed_data_dir
        ):
            print(
                f"[load_data] Using PrecomputedMultiTaskDataset from '{data_args.precomputed_data_dir}'"
            )
            datasets = PrecomputedMultiTaskDataset.load_data(
                data_args, tokenizer, train, valid, test
            )
        else:
            # Check RAM Cache capacity if enabled
            if getattr(data_args, "use_ram_cache", False):
                threshold_ratio = getattr(data_args, "ram_free_threshold_ratio", 0.30)
                from .utils import check_ram_capacity_for_dataset

                hf_token = data_args.hf_token or os.getenv("HF_TOKEN")
                audio_repo_id = getattr(data_args, "data_path", None) or "NhutP/VietSpeech"
                is_approved, est_gb, proj_ratio, total_examples = (
                    check_ram_capacity_for_dataset(
                        audio_repo_id,
                        hf_token=hf_token,
                        threshold_ratio=threshold_ratio,
                    )
                )
                if not is_approved:
                    print(
                        f"\n[load_data] AUTOMATIC FALLBACK TO STREAMING METHOD:\n"
                        f"  Projected free RAM ({proj_ratio * 100:.1f}%) <= {threshold_ratio * 100:.1f}% threshold.\n"
                        f"  Automatically switching to StreamingPrecomputedMultiTaskDataset ('{data_args.streaming_repo_id}') (ZERO DISK CACHE)!\n"
                    )
                    from .streaming_precomputed_multitask import (
                        StreamingPrecomputedMultiTaskDataset,
                    )

                    datasets = StreamingPrecomputedMultiTaskDataset.load_data(
                        data_args, tokenizer, train, valid, test
                    )
                    collator = DiscreteDiffusionDataCollator(
                        bos_id=tokenizer.bos_token_id,
                        eos_id=tokenizer.eos_token_id,
                        pad_id=tokenizer.pad_token_id,
                    )
                    return datasets, collator

            datasets = MultiTaskTranslatedSpeechDataset.load_data(
                data_args, tokenizer, train, valid, test
            )
    else:
        raise ValueError(
            f"Unknown or unsupported dataset type: {data_args.dataset_type}"
        )

    # Apply chunk augmentation when the streaming architecture is enabled.
    enable_streaming_architecture = getattr(
        data_args, "enable_streaming_architecture", False
    )

    if enable_streaming_architecture and datasets[0] is not None:
        from .streaming_augmented import StreamingAugmentedDataset

        datasets = (
            StreamingAugmentedDataset(
                base_dataset=datasets[0],
                tokenizer=tokenizer,
                chunk_duration=getattr(data_args, "audio_chunk_duration", 2.0),
                overlap_duration=getattr(data_args, "audio_overlap_duration", 0.5),
            ),
            datasets[1],  # Validation typically uses full audio
            datasets[2],  # Test typically uses full audio
        )

    # Build collator
    if enable_streaming_architecture:
        from .collator import StreamingCollator

        collator = StreamingCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        collator = DiscreteDiffusionDataCollator(
            bos_id=tokenizer.bos_token_id,
            eos_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id,
        )

    return datasets, collator
