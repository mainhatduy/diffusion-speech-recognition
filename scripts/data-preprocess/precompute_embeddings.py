#!/usr/bin/env python3
"""Pre-compute audio embeddings and tokenized text targets for fast training.

Usage:
    python scripts/data-preprocess/precompute_embeddings.py \
        --output_dir precomputed_data \
        --audio_encoder_name UsefulSensors/moonshine-streaming-medium \
        --pretrained FacebookAI/xlm-roberta-base \
        --batch_size 32 --max_length 128 [--resume]
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent.parent / "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TASK_TO_FIELD = {"<vi_en>": "english", "<vi_zh>": "chinese", "<vi_ko>": "korean"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--audio_encoder_name",
        type=str,
        default="UsefulSensors/moonshine-streaming-medium",
    )
    p.add_argument("--pretrained", type=str, default="FacebookAI/xlm-roberta-base")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--task_tokens", nargs="+", default=["<vi_en>", "<vi_zh>", "<vi_ko>"]
    )
    return p.parse_args()


class RawAudioDataset(Dataset):
    def __init__(self, vietspeech_dataset, feature_extractor, target_sr=16000):
        self.dataset = vietspeech_dataset
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from data.utils import _decode_wav_bytes

        item = self.dataset[idx]
        audio_info = item["audio"]
        waveform, sr = _decode_wav_bytes(audio_info["bytes"])
        if sr != self.target_sr:
            ratio = self.target_sr / sr
            new_len = int(len(waveform) * ratio)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, new_len),
                np.arange(len(waveform)),
                waveform,
            )
        audio_inputs = self.feature_extractor(
            waveform, sampling_rate=self.target_sr, return_tensors="pt", padding=False
        )
        return {
            "audio_values": audio_inputs.input_values.squeeze(0),
            "path": audio_info.get("path", f"sample_{idx}"),
            "idx": idx,
        }


def collate_audio(batch):
    audio_list = [b["audio_values"] for b in batch]
    max_len = max(a.size(-1) for a in audio_list)
    max_len = ((max_len + 79) // 80) * 80
    padded = torch.zeros(len(audio_list), max_len)
    mask = torch.zeros(len(audio_list), max_len, dtype=torch.long)
    for i, a in enumerate(audio_list):
        L = a.size(-1)
        padded[i, :L] = a
        mask[i, :L] = 1
    return {
        "audio_values": padded,
        "attention_mask": mask,
        "paths": [b["path"] for b in batch],
        "indices": [b["idx"] for b in batch],
    }


def load_audio_encoder(name, cache_dir, device):
    if "moonshine" in name:
        from transformers import MoonshineStreamingModel

        encoder = MoonshineStreamingModel.from_pretrained(
            name, cache_dir=cache_dir
        ).encoder
    else:
        from transformers import Wav2Vec2Model

        encoder = Wav2Vec2Model.from_pretrained(name, cache_dir=cache_dir)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def precompute_audio(args, vietspeech_ds, feat_ext, path_to_vs_idx, out_dir, done_ids):
    embed_dir = os.path.join(out_dir, "audio_embeds")
    os.makedirs(embed_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    encoder = load_audio_encoder(args.audio_encoder_name, args.cache_dir, device)
    hidden_size = encoder.config.hidden_size
    raw_ds = RawAudioDataset(vietspeech_ds, feat_ext)

    if done_ids:
        idxs = [vs_idx for p, vs_idx in path_to_vs_idx.items() if p not in done_ids]
    else:
        idxs = list(range(len(vietspeech_ds)))
    if not idxs:
        print("[Precompute] All audio embeddings done!")
        return hidden_size

    loader = DataLoader(
        Subset(raw_ds, idxs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(args.dtype == "float16")):
        for batch in tqdm(loader, desc="Audio embeddings"):
            av = batch["audio_values"].to(device)
            am = batch["attention_mask"].to(device)
            out = encoder(av, attention_mask=am)
            embeds = out.last_hidden_state
            if hasattr(out, "attention_mask") and out.attention_mask is not None:
                vmask = out.attention_mask
            else:
                vmask = encoder._get_feature_vector_attention_mask(embeds.size(1), am)
            for i, path in enumerate(batch["paths"]):
                vlen = vmask[i].sum().item() if vmask is not None else embeds.size(1)
                e = embeds[i, :vlen].cpu().numpy().astype(save_dtype)
                safe = path.replace("/", "_").replace("\\", "_")
                safe = (
                    (safe.rsplit(".", 1)[0] + ".npy")
                    if "." in safe
                    else (safe + ".npy")
                )
                np.save(os.path.join(embed_dir, safe), e)
    return hidden_size


def precompute_text(args, translated_ds, tokenizer, task_tokens, out_dir):
    from data.utils import normalize_text

    tokens_dir = os.path.join(out_dir, "token_ids")
    os.makedirs(tokens_dir, exist_ok=True)
    for tt in task_tokens:
        field = TASK_TO_FIELD.get(tt, tt.strip("<>"))
        out_path = os.path.join(tokens_dir, f"{field}.json")
        if os.path.exists(out_path):
            print(f"[Precompute] {field} tokens exist, skipping")
            continue
        all_toks = []
        for i in tqdm(range(len(translated_ds)), desc=f"Tokenize {field}"):
            text = translated_ds[i].get(field, "") or ""
            tids = tokenizer.encode(normalize_text(text), add_special_tokens=True)
            if len(tids) > args.max_length:
                tids = tids[: args.max_length]
            all_toks.append(tids)
        with open(out_path, "w") as f:
            json.dump(all_toks, f)
        print(f"[Precompute] Saved {len(all_toks)} seqs → {out_path}")


def build_index(translated_ds, path_to_vs_idx, out_dir):
    idx_path = os.path.join(out_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)
    index = []
    for i in tqdm(range(len(translated_ds)), desc="Build index"):
        wav_id = translated_ds[i]["id"]
        safe = wav_id.replace("/", "_").replace("\\", "_")
        safe = (safe.rsplit(".", 1)[0] + ".npy") if "." in safe else (safe + ".npy")
        index.append({"idx": i, "wav_id": wav_id, "embed_file": safe})
    with open(idx_path, "w") as f:
        json.dump(index, f)
    return index


def main():
    args = parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required")
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained, padding_side="right", use_fast=False, cache_dir=args.cache_dir
    )
    from utils import TASK_SPECIAL_TOKENS

    new_toks = [t for t in TASK_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_toks:
        tokenizer.add_special_tokens({"additional_special_tokens": new_toks})
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_name, cache_dir=args.cache_dir
    )

    from datasets import load_dataset, Audio

    translated_ds = load_dataset(
        "aiai-laboratory/vietspeech-train-translated",
        token=hf_token,
        cache_dir=args.cache_dir,
        split="train[:100%]",
    )
    vietspeech_ds = load_dataset(
        "NhutP/VietSpeech",
        token=hf_token,
        cache_dir=args.cache_dir,
        split="train[:100%]",
    )
    vietspeech_ds = vietspeech_ds.cast_column("audio", Audio(decode=False))

    audio_col = vietspeech_ds.data.column("audio")
    vs_paths = []
    for chunk in audio_col.chunks:
        vs_paths.extend(chunk.field("path").to_pylist())
    path_to_vs_idx = {p: i for i, p in enumerate(vs_paths)}

    done_ids = set()
    embed_dir = os.path.join(args.output_dir, "audio_embeds")
    if args.resume and os.path.exists(embed_dir):
        existing = set(os.listdir(embed_dir))
        for p in path_to_vs_idx:
            safe = p.replace("/", "_").replace("\\", "_")
            safe = (safe.rsplit(".", 1)[0] + ".npy") if "." in safe else (safe + ".npy")
            if safe in existing:
                done_ids.add(p)

    t0 = time.time()
    hs = precompute_audio(
        args, vietspeech_ds, feat_ext, path_to_vs_idx, args.output_dir, done_ids
    )
    print(f"[Precompute] Audio phase: {time.time()-t0:.1f}s")

    t0 = time.time()
    precompute_text(args, translated_ds, tokenizer, args.task_tokens, args.output_dir)
    print(f"[Precompute] Text phase: {time.time()-t0:.1f}s")

    build_index(translated_ds, path_to_vs_idx, args.output_dir)

    vocab_hash = hashlib.md5(
        json.dumps(sorted(tokenizer.get_vocab().items())).encode()
    ).hexdigest()
    meta = {
        "audio_encoder_name": args.audio_encoder_name,
        "audio_hidden_size": hs,
        "pretrained_model": args.pretrained,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_hash": vocab_hash,
        "max_length": args.max_length,
        "save_dtype": args.dtype,
        "task_tokens": args.task_tokens,
        "num_samples": len(translated_ds),
        "num_audio_files": len(path_to_vs_idx),
        "embed_stage": "before_projector",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Precompute] Done! Use precomputed_data_dir={args.output_dir} in config.")


if __name__ == "__main__":
    main()
