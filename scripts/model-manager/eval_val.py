import os
import sys
import time
import argparse
import json
import torch
import numpy as np
import io
import soundfile as sf
from datasets import load_dataset
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# Resolve project root so we can import src/ modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel
from model.modeling_dlm import decoder_out_t
from data.utils import normalize_text
import sacrebleu
import jiwer

def load_audio_from_bytes(raw_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Load audio from raw bytes and resample to target_sr if needed."""
    waveform, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    
    if sr != target_sr:
        ratio = target_sr / sr
        new_length = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
    return waveform

def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Speech Translation model with batching.")
    parser.add_argument("--repo-id", type=str, default="aiai-laboratory/diffusion-speech-translation-from-vi-v1", help="Model repo ID")
    parser.add_argument("--dataset-id", type=str, default="aiai-laboratory/vietspeech-validation-translated", help="Dataset repo ID")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--tasks", type=str, default="vietnamese,english,chinese,korean", help="Comma-separated tasks to evaluate")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of diffusion iterations")
    parser.add_argument("--max-length", type=int, default=64, help="Max length of target text")
    parser.add_argument("--oracle-length", action="store_true", help="Use oracle target length")
    parser.add_argument("--strategy", type=str, default="reparam-uncond-deterministic-cosine", help="Decoding strategy")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading tokenizer from {args.repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True, use_fast=False)

    print(f"Loading config and model from {args.repo_id}...")
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = DiscreteDiffusionConfig(**{
        k: v for k, v in config_dict.items()
        if not k.startswith("_") and k != "model_type" and k != "transformers_version"
        and k != "auto_map"
    })

    model = DiscreteDiffusionModel(config)

    try:
        weights_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device="cpu")
    except Exception:
        weights_path = hf_hub_download(repo_id=args.repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    if config.tie_word_embeddings:
        model.model.lm_head.decoder.weight = model.model.roberta.embeddings.word_embeddings.weight

    model = model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
        print("Model converted to bfloat16")

    if args.compile:
        print("Compiling model (this might take a few minutes)...")
        model = torch.compile(model)

    print(f"Loading feature extractor from {config.audio_encoder_name}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_encoder_name)

    print(f"Loading dataset {args.dataset_id}...")
    ds = load_dataset(args.dataset_id)['validation']
    total_samples = len(ds)
    limit = args.limit if args.limit > 0 else total_samples
    print(f"Loaded {total_samples} samples. Evaluating on the first {limit} samples.")

    # Task configuration
    all_tasks = {
        "vietnamese": ("ASR (VI)", None, "wer"),
        "english": ("Translation (EN)", "<vi_en>", "bleu"),
        "chinese": ("Translation (ZH)", "<vi_zh>", "bleu"),
        "korean": ("Translation (KO)", "<vi_ko>", "bleu"),
    }
    
    selected_task_names = [t.strip() for t in args.tasks.split(",") if t.strip() in all_tasks]
    tasks = {t: all_tasks[t] for t in selected_task_names}
    print(f"Selected tasks: {list(tasks.keys())}")

    # Setup tokens
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    task_token_ids = {}
    for task_name, (label, token, metric_name) in tasks.items():
        if token is not None:
            task_token_ids[task_name] = tokenizer.convert_tokens_to_ids(token)

    results = {task_name: {"refs": [], "hyps": [], "times": [], "lengths": []} for task_name in tasks}
    audio_durations = []

    # Batching loop
    batch_size = args.batch_size
    for batch_idx in tqdm(range(0, limit, batch_size), desc="Evaluating batches"):
        actual_batch_size = min(batch_size, limit - batch_idx)
        batch_rows = [ds[batch_idx + i] for i in range(actual_batch_size)]
        
        # 1. Load and process audio for the batch
        batch_waveforms = []
        batch_durations = []
        skip_indices = []
        
        for idx, row in enumerate(batch_rows):
            raw_bytes = row["audio"]["bytes"]
            try:
                waveform = load_audio_from_bytes(raw_bytes, target_sr=16000)
                batch_waveforms.append(waveform)
                duration = len(waveform) / 16000
                batch_durations.append(duration)
            except Exception as e:
                print(f"\nFailed to load audio for index {batch_idx + idx}: {e}. Skipping sample.")
                skip_indices.append(idx)
                
        if not batch_waveforms:
            continue
            
        audio_durations.extend(batch_durations)

        # Pad audio inputs to the maximum audio length in the batch (rounded to multiple of 80)
        max_audio_len = max(w.shape[0] for w in batch_waveforms)
        padded_audio_len = ((max_audio_len + 79) // 80) * 80
        
        audio_values = torch.zeros(len(batch_waveforms), padded_audio_len, device=device)
        audio_attention_mask = torch.zeros(len(batch_waveforms), padded_audio_len, dtype=torch.long, device=device)
        
        for idx, waveform in enumerate(batch_waveforms):
            audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            raw_val = audio_inputs.input_values.to(device)[0]
            audio_values[idx, :raw_val.size(0)] = raw_val
            audio_attention_mask[idx, :raw_val.size(0)] = 1
            
        if device == "cuda":
            audio_values = audio_values.to(torch.bfloat16)

        # Filter batch rows to remove skipped ones
        valid_batch_rows = [r for idx, r in enumerate(batch_rows) if idx not in skip_indices]

        # 2. Run each task
        for task_name, (label, token, metric_name) in tasks.items():
            # Setup prompts
            if token is None:
                src_length = 1
                prefix = [bos_id]
            else:
                src_length = 2
                prefix = [bos_id, task_token_ids[task_name]]

            # Determine target length for each sample in the batch
            sequences = []
            for idx, row in enumerate(valid_batch_rows):
                ref_text = row[task_name].strip()
                audio_dur = batch_durations[idx]
                
                if args.oracle_length:
                    ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
                    canvas_len = len(ref_tokens)
                else:
                    if task_name == "vietnamese":
                        canvas_len = int(audio_dur * 4.5)
                    elif task_name == "english":
                        canvas_len = int(audio_dur * 4.0)
                    else:
                        canvas_len = int(audio_dur * 2.5)
                    canvas_len = max(5, min(args.max_length, canvas_len))
                    
                seq = prefix + [mask_id] * canvas_len + [eos_id]
                sequences.append(torch.tensor(seq, dtype=torch.long))

            # Pad text sequences to the maximum length in the batch
            canvas = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=pad_id
            ).to(device)

            # Create masks
            prefix_lengths = torch.tensor([src_length] * len(valid_batch_rows), dtype=torch.long, device=device)
            position_ids = torch.arange(canvas.size(1), device=device).unsqueeze(0).expand(canvas.size(0), -1)
            partial_mask = position_ids < prefix_lengths.unsqueeze(1)

            non_fixed_sym_masks = (
                canvas.ne(pad_id) &
                canvas.ne(bos_id) &
                canvas.ne(eos_id) &
                ~partial_mask
            )

            output_scores = torch.zeros_like(canvas, dtype=torch.float32)
            output_mask = canvas.eq(mask_id)

            # Denoising loop
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            decoder_out = decoder_out_t(
                output_tokens=canvas.clone(),
                output_scores=output_scores,
                output_masks=output_mask,
                non_fixed_sym_masks=non_fixed_sym_masks,
                attn=None,
                step=0,
                max_step=args.iterations,
                history=None,
            )

            with torch.no_grad():
                for _ in range(args.iterations):
                    decoder_out = model.denoise_step(
                        decoder_out,
                        partial_mask,
                        temperature=1.0,
                        strategy=args.strategy,
                        audio_features=audio_values,
                        audio_attention_mask=audio_attention_mask,
                    )

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time

            # Decode outputs
            for idx, row in enumerate(valid_batch_rows):
                out_tokens = decoder_out.output_tokens[idx]
                p_mask = partial_mask[idx]
                cutoff = (
                    out_tokens.ne(pad_id) &
                    out_tokens.ne(bos_id) &
                    out_tokens.ne(eos_id) &
                    ~p_mask
                )
                gen_tokens = out_tokens[cutoff]
                pred_text = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True).strip()

                results[task_name]["refs"].append(row[task_name].strip())
                results[task_name]["hyps"].append(pred_text)
                results[task_name]["times"].append(elapsed_time / len(valid_batch_rows))
                results[task_name]["lengths"].append(len(gen_tokens))

    # Calculate metrics
    metrics_summary = {}
    print("\n" + "="*60)
    print(" EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    total_audio_time = sum(audio_durations)
    print(f"Total audio duration: {total_audio_time:.2f}s")
    print(f"Number of samples evaluated: {len(audio_durations)}")
    print("-"*60)

    for task_name, (label, _, metric_type) in tasks.items():
        task_data = results[task_name]
        refs = task_data["refs"]
        hyps = task_data["hyps"]
        times = task_data["times"]
        lengths = task_data["lengths"]

        if not refs:
            continue

        total_task_time = sum(times)
        avg_time = np.mean(times)
        rtf = total_task_time / total_audio_time if total_audio_time > 0 else 0.0
        total_tokens = sum(lengths)
        tokens_per_sec = total_tokens / total_task_time if total_task_time > 0 else 0.0

        if task_name == "vietnamese":
            # For ASR (Vietnamese), calculate WER
            norm_refs = [normalize_text(r) for r in refs]
            norm_hyps = [normalize_text(h) for h in hyps]
            wer_score = jiwer.wer(norm_refs, norm_hyps) * 100
            metric_str = f"WER: {wer_score:.2f}%"
            metrics_summary[task_name] = {"wer": wer_score}
        elif task_name == "english":
            # For English translation, calculate both BLEU and WER
            bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score
            norm_refs = [normalize_text(r) for r in refs]
            norm_hyps = [normalize_text(h) for h in hyps]
            wer_score = jiwer.wer(norm_refs, norm_hyps) * 100
            metric_str = f"BLEU: {bleu_score:.2f} | WER: {wer_score:.2f}%"
            metrics_summary[task_name] = {"bleu": bleu_score, "wer": wer_score}
        else:
            # For Chinese and Korean, calculate BLEU
            tok = "zh" if task_name == "chinese" else "13a"
            bleu_score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok).score
            metric_str = f"BLEU: {bleu_score:.2f}"
            metrics_summary[task_name] = {"bleu": bleu_score}

        metrics_summary[task_name].update({
            "avg_time_per_sample_sec": float(avg_time),
            "rtf": float(rtf),
            "tokens_per_sec": float(tokens_per_sec)
        })

        print(f"{label}:")
        print(f"  {metric_str}")
        print(f"  Avg inference time per sample: {avg_time:.3f}s")
        print(f"  Real-Time Factor (RTF): {rtf:.3f}")
        print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")
        print("-"*60)

    # Save to json
    output_data = {
        "config": vars(args),
        "summary": metrics_summary,
        "details": {
            k: {
                "refs": v["refs"],
                "hyps": v["hyps"],
                "times": v["times"],
                "lengths": v["lengths"]
            } for k, v in results.items()
        }
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
