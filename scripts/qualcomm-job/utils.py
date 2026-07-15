import os
import sys
import time
import numpy as np
import torch
import onnx

# ─────────────────────────────── Constants ────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
MAX_SEQ_LEN = 32

# ─────────────────────────────── Helpers ──────────────────────────────────


def setup_qualcomm_token():
    """Load environment variables and set QAI_HUB_API_TOKEN from QUALCOMM_TOKEN."""
    import dotenv

    dotenv.load_dotenv()
    token = os.getenv("QUALCOMM_TOKEN")
    if not token:
        print("[!] Error: QUALCOMM_TOKEN not found in .env")
        sys.exit(1)
    os.environ["QAI_HUB_API_TOKEN"] = token
    return token


def load_audio(audio_path: str, target_sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Load audio file and return as float32 numpy array at target sample rate."""
    try:
        import librosa

        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio.astype(np.float32)
    except ImportError:
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if sr != target_sr:
            import scipy.signal

            audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        return audio.astype(np.float32)


def prepare_audio_inputs(audio: np.ndarray, stride: int = 80) -> dict:
    """Prepare audio inputs for the audio_encoder ONNX model."""
    original_len = len(audio)
    if original_len % stride != 0:
        padded_len = ((original_len // stride) + 1) * stride
        audio_padded = np.zeros(padded_len, dtype=np.float32)
        audio_padded[:original_len] = audio
        mask = np.zeros(padded_len, dtype=np.int64)
        mask[:original_len] = 1
    else:
        audio_padded = audio
        mask = np.ones(original_len, dtype=np.int64)

    audio_features = audio_padded[np.newaxis, :]  # (1, audio_len)
    audio_attention_mask = mask[np.newaxis, :]  # (1, audio_len)
    return {
        "audio_features": audio_features,
        "audio_attention_mask": audio_attention_mask,
    }


def prepare_backbone_inputs(
    audio_embeds: np.ndarray,
    audio_len: int,
    mask_id: int = None,
    eos_id: int = None,
    seq_len: int = MAX_SEQ_LEN,
) -> dict:
    """Prepare precomputed audio embeddings/mask, and optionally initial input tokens for compile/inference."""
    batch_size = 1
    precomputed_audio_mask = np.ones((batch_size, audio_len), dtype=np.int32)
    inputs = {
        "precomputed_audio_embeds": audio_embeds.astype(np.float32),
        "precomputed_audio_mask": precomputed_audio_mask,
    }
    if mask_id is not None and eos_id is not None:
        tokens = np.full((batch_size, seq_len), mask_id, dtype=np.int64)
        tokens[0, -1] = eos_id
        inputs["prev_output_tokens"] = tokens
        inputs["partial_mask"] = np.zeros((batch_size, seq_len), dtype=bool)
    return inputs


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """Select top-k lowest scores and create mask."""
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def reparam_decoding(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    decoding_strategy,
    xt_neq_x0,
    non_special_sym_mask,
    t,
    max_step,
    noise_id,
):
    """Reparameterized decoding step."""
    _, condition, topk_mode, schedule = decoding_strategy.split("-")

    if schedule == "linear":
        rate = 1 - t / max_step
    elif schedule == "cosine":
        rate = np.cos(t / max_step * np.pi * 0.5)
    else:
        raise NotImplementedError

    cutoff_len = (
        non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
    ).long()
    _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)

    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(
            _scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate
        )
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError

    if condition == "cond":
        not_v1_t = (
            (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        )
    elif condition == "uncond":
        not_v1_t = lowest_k_mask
    else:
        raise NotImplementedError

    not_v2_t = lowest_k_mask

    masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
    output_tokens.masked_fill_(masked_to_noise, noise_id)
    output_scores.masked_fill_(masked_to_noise, -float("inf"))

    masked_to_x0 = xt_neq_x0 & ~not_v2_t
    output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
    output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])

    new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
    return new_xt_neq_x0


def decode_tokens(tokens, tokenizer, eos_id):
    """Truncate sequence at first EOS and decode."""
    tokens_list = tokens.tolist()[0] if hasattr(tokens, "tolist") else list(tokens[0])
    if eos_id in tokens_list:
        eos_idx = tokens_list.index(eos_id)
        tokens_list = tokens_list[:eos_idx]
    return tokenizer.decode(tokens_list, skip_special_tokens=True)


def repackage_model(model_path, output_dir, model_name, data_name):
    """Repackage ONNX model with compliant external weight offloading."""
    print(f"[*] Repackaging {model_path} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = onnx.load(model_path)

    # Save the model with external data
    target_model_path = os.path.join(output_dir, model_name)
    target_data_path = os.path.join(output_dir, data_name)
    if os.path.exists(target_data_path):
        os.remove(target_data_path)

    onnx.save(
        model,
        target_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    print(f"    -> Successfully saved to {target_model_path} and {target_data_path}")


def monitor_jobs(jobs, label="Jobs"):
    """Monitor a set of Qualcomm AI Hub jobs until completion."""
    print(f"\n[*] Monitoring {label}...")
    completed = {name: False for name in jobs}
    failures = {name: None for name in jobs}

    while not all(completed.values()):
        for name, job in jobs.items():
            if completed[name]:
                continue

            status = job.get_status()
            code = status.code
            message = status.message

            if code in ["SUCCESS", "FAILED"]:
                completed[name] = True
                if code == "FAILED":
                    failures[name] = message
                    print(f"    [!] {name} failed: {message}")
                else:
                    print(f"    [+] {name} completed successfully!")
            else:
                print(f"    [-] {name} is in status: {code}")

        if not all(completed.values()):
            time.sleep(30)

    return failures
