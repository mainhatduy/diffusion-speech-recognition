import io
import re
import unicodedata
import wave

import numpy as np


def _decode_wav_bytes(wav_bytes: bytes):
    """Decode raw WAV bytes to float32 numpy array using python's built-in wave module.
    This avoids dependency on torchcodec/soundfile/librosa."""
    f = wave.open(io.BytesIO(wav_bytes), "rb")
    n_channels = f.getnchannels()
    sampwidth = f.getsampwidth()
    n_frames = f.getnframes()
    sample_rate = f.getframerate()
    raw_frames = f.readframes(n_frames)
    f.close()

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    data = np.frombuffer(raw_frames, dtype=dtype).astype(np.float32)

    # Normalize to [-1.0, 1.0]
    if sampwidth == 2:
        data = data / 32768.0
    elif sampwidth == 4:
        data = data / 2147483648.0

    # Convert stereo to mono by averaging channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, sample_rate


def normalize_text(text: str) -> str:
    """Normalize and clean text by lowercasing and removing punctuation/symbols."""
    if not text:
        return ""
    normalized_text = text.lower()
    chars = []
    for char in normalized_text:
        cat = unicodedata.category(char)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        else:
            chars.append(char)
    normalized_text = "".join(chars)
    return re.sub(r"\s+", " ", normalized_text).strip()


def check_ram_capacity_for_dataset(
    repo_id: str, hf_token: str | None = None, threshold_ratio: float = 0.30
) -> tuple[bool, float, float, int | None]:
    """Inspect dataset metadata from Hugging Face Hub upfront to pre-calculate if total raw audio size fits in RAM memory.

    Returns:
        (is_approved, estimated_size_gb, projected_free_ratio, total_examples)
    """
    import logging
    import psutil
    from datasets import load_dataset_builder

    logger = logging.getLogger(__name__)
    total_examples = None
    try:
        builder = load_dataset_builder(repo_id, token=hf_token)
        info = builder.info
        est_bytes = info.download_size or info.dataset_size
        if not est_bytes and info.splits:
            est_bytes = sum(
                split.num_bytes for split in info.splits.values() if split.num_bytes
            )
        est_gb = est_bytes / (1024**3)

        if info.splits and "train" in info.splits:
            total_examples = info.splits["train"].num_examples
        elif info.splits:
            total_examples = sum(s.num_examples for s in info.splits.values() if s.num_examples)
    except Exception as e:
        logger.warning(
            f"[RAM Pre-Check] Could not fetch metadata for '{repo_id}': {e}. Skipping RAM pre-check."
        )
        return True, 0.0, 1.0, None

    vm = psutil.virtual_memory()
    tot_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    proj_free_gb = avail_gb - est_gb
    proj_free_ratio = proj_free_gb / tot_gb

    print(
        f"\n[RAM Pre-Check] Upfront Dataset Metadata Calculation for '{repo_id}':\n"
        f"  - Total Audio Samples:          {total_examples if total_examples else 'Unknown':,}\n"
        f"  - Estimated Audio Dataset Size: {est_gb:.2f} GB\n"
        f"  - System Total RAM:             {tot_gb:.2f} GB\n"
        f"  - Currently Available RAM:      {avail_gb:.2f} GB\n"
        f"  - Projected Free RAM:           {proj_free_gb:.2f} GB ({proj_free_ratio * 100:.1f}% free remaining)"
    )

    if proj_free_ratio > threshold_ratio:
        print(
            f"[RAM Pre-Check] DECISION: APPROVED! Projected free RAM ({proj_free_ratio * 100:.1f}%) > {threshold_ratio * 100:.1f}% safety threshold.\n"
            f"  Proceeding to stream raw audio directly into RAM...\n"
        )
        return True, est_gb, proj_free_ratio, total_examples
    else:
        print(
            f"[RAM Pre-Check] DECISION: REJECTED! Dataset size ({est_gb:.2f} GB) would leave only {proj_free_ratio * 100:.1f}% free RAM (<= {threshold_ratio * 100:.1f}% threshold).\n"
            f"  Safely bypassing RAM cache to prevent OOM/memory overflow!\n"
        )
        return False, est_gb, proj_free_ratio, total_examples


