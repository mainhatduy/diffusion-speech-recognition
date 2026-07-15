"""
inference.py — Speech translation inference for the diffusion model.

Loads aiai-laboratory/diffusion-speech-translation-from-vi-v1 from HuggingFace Hub
and translates Vietnamese speech audio into 3 target languages: English, Chinese, Korean.

Usage:
    uv run python scripts/model-manager/inference.py <audio_file> [--iterations 10] [--device cuda]
    uv run python scripts/model-manager/inference.py path/to/audio.wav
"""

import sys
import os
import argparse
import torch
import numpy as np

# ─── Resolve project root so we can import src/ modules ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# ─── Audio loading helpers ────────────────────────────────────────────────────


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and return a float32 numpy array at target_sr Hz."""
    waveform = None
    sr = None

    # 1. Try soundfile (wav, flac, ogg, aiff — NOT mp3 without libsndfile mp3 plugin)
    try:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
    except Exception:
        pass

    # 2. Try librosa (handles mp3 via audioread/ffmpeg)
    if waveform is None:
        try:
            import librosa

            waveform, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        except Exception:
            pass

    # 3. Try pydub (handles mp3/aac/ogg via ffmpeg)
    if waveform is None:
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            waveform = samples / (2 ** (audio.sample_width * 8 - 1))
            sr = target_sr  # already resampled by pydub
        except Exception:
            pass

    # 4. Try torchaudio
    if waveform is None:
        try:
            import torchaudio

            waveform_t, sr = torchaudio.load(path)
            waveform = waveform_t.mean(dim=0).numpy().astype(np.float32)
        except Exception:
            pass

    if waveform is None:
        raise RuntimeError(
            f"Could not load audio file '{path}'.\n"
            "Install one of: librosa, pydub (+ ffmpeg), torchaudio, or soundfile (for WAV/FLAC)."
        )

    # Resample if needed
    if sr != target_sr:
        ratio = target_sr / sr
        new_length = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(
            np.float32
        )

    return waveform


# ─── Main inference function ──────────────────────────────────────────────────


def translate(
    audio_path: str,
    repo_id: str = "aiai-laboratory/diffusion-speech-translation-from-vi-v1",
    max_iterations: int = 10,
    max_length: int = 64,
    canvas_len_override: int = None,
    strategy: str = "reparam-uncond-deterministic-cosine",
    device: str = None,
) -> dict:
    """
    Translate a Vietnamese audio file into English, Chinese, and Korean.

    Args:
        audio_path: Path to a .wav / .mp3 / .flac file (Vietnamese speech).
        repo_id: HuggingFace model repository.
        max_iterations: Number of diffusion denoising steps.
        max_length: Max output token length per translation.
        canvas_len_override: Force a specific canvas length.
        strategy: Decoding strategy passed to the diffusion model.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        dict with keys 'english', 'chinese', 'korean'.
    """
    from transformers import AutoTokenizer
    from transformers import Wav2Vec2FeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference] Using device: {device}")

    # ── 1. Load tokenizer & model ─────────────────────────────────────────────
    print(f"[inference] Loading tokenizer from {repo_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, use_fast=False
    )

    print(f"[inference] Loading model from {repo_id} ...")
    # Use local source classes directly to avoid meta-device issues with nested from_pretrained calls
    from model.configuration_dlm import DiscreteDiffusionConfig
    from model.modeling_dlm import DiscreteDiffusionModel
    from huggingface_hub import hf_hub_download
    import json

    # Download just the config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = DiscreteDiffusionConfig(
        **{
            k: v
            for k, v in config_dict.items()
            if not k.startswith("_")
            and k != "model_type"
            and k != "transformers_version"
            and k != "auto_map"
        }
    )

    # Build the model from config then load weights from Hub
    model = DiscreteDiffusionModel(config)

    # Download weights
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
    except Exception:
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # lm_head.decoder.weight is a tied weight (omitted from safetensors) — re-tie explicitly
    if config.tie_word_embeddings:
        model.model.lm_head.decoder.weight = (
            model.model.roberta.embeddings.word_embeddings.weight
        )
    missing = [k for k in (missing or []) if "lm_head.decoder.weight" not in k]
    if missing:
        print(
            f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    model = model.eval().to(device)

    # Use bfloat16 if CUDA is available (matches training setup)
    if device == "cuda":
        model = model.to(torch.bfloat16)
        print("[inference] Using bfloat16 precision")

    # ── 2. Resolve task token IDs ─────────────────────────────────────────────
    TASKS = {
        "english": "<vi_en>",
        "chinese": "<vi_zh>",
        "korean": "<vi_ko>",
    }
    task_token_ids = {}
    for lang, token in TASKS.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid == tokenizer.unk_token_id:
            raise RuntimeError(
                f"Task token '{token}' not found in tokenizer vocab. "
                "Make sure this tokenizer was saved with the task tokens added."
            )
        task_token_ids[lang] = tid
        print(f"  {token} → id={tid}")

    # ── 3. Preprocess audio ───────────────────────────────────────────────────
    print(f"[inference] Loading audio: {audio_path}")
    waveform = load_audio(audio_path, target_sr=16000)
    audio_duration = len(waveform) / 16000
    print(f"  Duration: {audio_duration:.2f}s  ({len(waveform)} samples @ 16kHz)")

    # Load the Moonshine feature extractor (same as used during training)
    audio_encoder_name = model.config.audio_encoder_name
    print(f"[inference] Loading feature extractor from {audio_encoder_name} ...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_name)

    # Process and pad features to a multiple of 80 frames (Moonshine requirement)
    audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    audio_values_raw = audio_inputs.input_values.to(device)

    audio_len = audio_values_raw.size(-1)
    padded_len = ((audio_len + 79) // 80) * 80
    padded_audio = torch.zeros(1, padded_len, device=device)
    padded_audio[0, :audio_len] = audio_values_raw[0]
    audio_values = padded_audio

    padded_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
    padded_mask[0, :audio_len] = 1
    audio_attention_mask = padded_mask

    if device == "cuda":
        audio_values = audio_values.to(torch.bfloat16)

    # ── 4. Run translation for each target language ───────────────────────────
    results = {}
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    print("\n[inference] Translating ...")
    with torch.no_grad():
        for lang, task_token_id in task_token_ids.items():
            # Determine canvas length
            if canvas_len_override is not None:
                canvas_len = canvas_len_override
            else:
                # Dynamic duration-based heuristic:
                # - Vietnamese to English: ~4.0 tokens per second of speech
                # - Vietnamese to Chinese/Korean: ~2.5 tokens per second of speech
                if lang == "english":
                    canvas_len = int(audio_duration * 4.0)
                else:
                    canvas_len = int(audio_duration * 2.5)
                # Bound to [5, max_length]
                canvas_len = max(5, min(max_length, canvas_len))

            print(f"  → {lang} (canvas length = {canvas_len}) ...", end=" ", flush=True)

            # Source prefix: [BOS, <vi_XX>]
            src_tokens = torch.tensor(
                [[bos_id, task_token_id]], dtype=torch.long, device=device
            )  # (1, 2)
            src_length = src_tokens.size(1)

            # Canvas: [BOS, <vi_XX>, MASK...MASK, EOS]
            canvas = torch.cat(
                [
                    src_tokens,
                    torch.full(
                        (1, canvas_len), mask_id, dtype=torch.long, device=device
                    ),
                    torch.tensor([[eos_id]], dtype=torch.long, device=device),
                ],
                dim=1,
            )  # (1, 2 + canvas_len + 1)

            # partial_mask: True for fixed source prefix, False for generated part
            partial_mask = torch.zeros_like(canvas, dtype=torch.bool)
            partial_mask[:, :src_length] = True

            # non_fixed_sym_masks: positions the model is allowed to fill (exclude BOS, EOS, task tokens)
            non_fixed_sym_masks = (
                canvas.ne(pad_id)
                & canvas.ne(bos_id)
                & canvas.ne(eos_id)
                & ~partial_mask
            )

            output_scores = torch.zeros_like(canvas, dtype=torch.float32)
            output_mask = canvas.eq(mask_id)

            # Run diffusion denoising loop
            from model.modeling_dlm import decoder_out_t

            decoder_out = decoder_out_t(
                output_tokens=canvas.clone(),
                output_scores=output_scores,
                output_masks=output_mask,
                non_fixed_sym_masks=non_fixed_sym_masks,
                attn=None,
                step=0,
                max_step=max_iterations,
                history=None,
            )

            for _ in range(max_iterations):
                decoder_out = model.denoise_step(
                    decoder_out,
                    partial_mask,
                    temperature=1.0,
                    strategy=strategy,
                    audio_features=audio_values,
                    audio_attention_mask=audio_attention_mask,
                )

            # Extract generated tokens (exclude source prefix, EOS, PAD)
            out_tokens = decoder_out.output_tokens[0]  # (seq_len,)
            cutoff = (
                out_tokens.ne(pad_id)
                & out_tokens.ne(bos_id)
                & out_tokens.ne(eos_id)
                & ~partial_mask[0]
            )
            gen_tokens = out_tokens[cutoff]

            text = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True).strip()
            results[lang] = text
            print(repr(text))

    return results


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Translate Vietnamese speech to EN / ZH / KO using the diffusion model."
    )
    parser.add_argument(
        "audio",
        help="Path to input audio file (WAV / MP3 / FLAC). Must be Vietnamese speech.",
    )
    parser.add_argument(
        "--repo",
        default="aiai-laboratory/diffusion-speech-translation-from-vi-v1",
        help="HuggingFace model repository ID.",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=10,
        help="Number of diffusion denoising steps (default: 10).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Max output tokens per translation (default: 64).",
    )
    parser.add_argument(
        "--canvas-len",
        type=int,
        default=None,
        help="Force a specific canvas length (number of generated tokens). If None, calculated dynamically from audio duration.",
    )
    parser.add_argument(
        "--strategy",
        default="reparam-uncond-deterministic-cosine",
        help="Decoding strategy (default: reparam-uncond-deterministic-cosine).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: 'cuda' or 'cpu'. Auto-detects if not set.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)

    results = translate(
        audio_path=args.audio,
        repo_id=args.repo,
        max_iterations=args.iterations,
        max_length=args.max_length,
        canvas_len_override=args.canvas_len,
        strategy=args.strategy,
        device=args.device,
    )

    # ── Pretty-print results ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SPEECH TRANSLATION RESULTS")
    print("═" * 60)
    flag = {"english": "🇬🇧 English", "chinese": "🇨🇳 Chinese", "korean": "🇰🇷 Korean"}
    for lang, text in results.items():
        print(f"\n  {flag[lang]}:")
        print(f"    {text if text else '(empty — model may need more training steps)'}")
    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()
