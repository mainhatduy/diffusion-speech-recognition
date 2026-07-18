import os
import json
import logging
import numpy as np
import torch
from .base import PromptDataset


class PrecomputedMultiTaskDataset(PromptDataset):
    """Fast multi-task dataset that reads pre-computed audio embeddings and token IDs.

    Replaces MultiTaskTranslatedSpeechDataset for training when precomputed data
    is available. __getitem__ reads from either parquet memory maps or individual
    numpy files depending on what is available.
    """

    TASK_TO_FIELD = {"<vi_en>": "english", "<vi_zh>": "chinese", "<vi_ko>": "korean"}

    def __init__(
        self,
        args,
        index,
        token_ids_map,
        task_configs,
        tokenizer,
        embed_dir,
        save_dtype,
        is_train=True,
        embed_dataset=None,
        embed_file_to_row=None,
    ):
        # PromptDataset expects raw_data; pass index as raw_data
        super().__init__(args, index, tokenizer)
        self.index = index  # list of {"idx", "wav_id", "embed_file"}
        self.token_ids_map = token_ids_map  # {field_name: list[list[int]]}
        self.task_configs = task_configs  # [(field_name, token_id), ...]
        self.embed_dir = embed_dir
        self.n_tasks = len(task_configs)
        self.save_dtype = np.float16 if save_dtype == "float16" else np.float32
        self.is_train = is_train
        self.embed_dataset = embed_dataset
        self.embed_file_to_row = embed_file_to_row

    def __len__(self):
        return len(self.index) * self.n_tasks

    def __getitem__(self, flat_index):
        sample_idx = flat_index // self.n_tasks
        task_idx = flat_index % self.n_tasks
        tgt_field, task_token_id = self.task_configs[task_idx]

        entry = self.index[sample_idx]
        data_idx = entry["idx"]  # original dataset index

        # ── Load pre-computed audio embedding ──
        if self.embed_dataset is not None:
            # Load from parquet dataset (memory-mapped arrow)
            row_idx = self.embed_file_to_row[entry["embed_file"]]
            row = self.embed_dataset[row_idx]

            shape = row.get("shape", None)
            embedding_bytes = row["embedding_bytes"]
            audio_embeds = np.frombuffer(embedding_bytes, dtype=self.save_dtype)
            if shape is not None:
                audio_embeds = audio_embeds.reshape(shape)
            else:
                audio_embeds = audio_embeds.reshape(-1, 768)
            audio_embeds = torch.from_numpy(audio_embeds.astype(np.float32))
        else:
            # Load from individual numpy file (old way)
            embed_path = os.path.join(self.embed_dir, entry["embed_file"])
            audio_embeds = np.load(embed_path)  # (T_frames, D_audio)
            audio_embeds = torch.from_numpy(audio_embeds.astype(np.float32))

        # ── Load pre-tokenized text target ──
        tgt = list(self.token_ids_map[tgt_field][data_idx])
        if len(tgt) > self.max_length:
            tgt = tgt[: self.max_length]

        # ── Build source prefix with task token ──
        src = [self.tokenizer.bos_token_id, task_token_id]

        # Strip BOS from tgt if present
        if len(tgt) > 0 and tgt[0] == self.tokenizer.bos_token_id:
            tgt = tgt[1:]

        src_length = len(src)
        concatenated = src + tgt
        if len(concatenated) > self.max_length:
            concatenated = concatenated[:self.max_length]
        remaining = self.max_length - len(concatenated)
        if remaining > 0:
            concatenated = concatenated + [self.tokenizer.eos_token_id] * remaining

        target_tgt = (
            [self.tokenizer.bos_token_id] + tgt
            if len(tgt) == 0 or tgt[0] != self.tokenizer.bos_token_id
            else tgt
        )

        return {
            "id": flat_index,
            "source": torch.tensor(concatenated),
            "target": torch.tensor(target_tgt),
            "src_length": src_length,
            "precomputed_audio_embeds": audio_embeds,  # (T_frames, D_audio)
        }

    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        """Load pre-computed dataset from disk."""
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        precomputed_dir = args.precomputed_data_dir
        if not os.path.exists(precomputed_dir):
            raise FileNotFoundError(
                f"Precomputed data dir '{precomputed_dir}' not found. "
                f"Run scripts/data-preprocess/precompute_embeddings.py first."
            )

        # Load metadata
        with open(os.path.join(precomputed_dir, "metadata.json")) as f:
            metadata = json.load(f)

        embed_dir = os.path.join(precomputed_dir, "audio_embeds")
        save_dtype = metadata.get("save_dtype", "float16")

        print(f"[PrecomputedMultiTask] Loading from {precomputed_dir}")
        print(
            f"[PrecomputedMultiTask] Encoder: {metadata['audio_encoder_name']}, "
            f"hidden_size: {metadata['audio_hidden_size']}, dtype: {save_dtype}"
        )

        # Check if parquet sharded files exist in embed_dir
        import glob

        parquet_files = sorted(glob.glob(os.path.join(embed_dir, "*.parquet")))
        if parquet_files:
            print(
                f"[PrecomputedMultiTask] Found {len(parquet_files)} parquet files. Using Parquet loading."
            )
            from datasets import load_dataset

            num_cores = os.cpu_count() or 1
            num_proc = max(1, int(num_cores * 0.5))
            print(
                f"[PrecomputedMultiTask] Loading parquet with {num_proc} CPU processes (50% of {num_cores} cores)..."
            )
            embed_dataset = load_dataset(
                "parquet", data_files=parquet_files, num_proc=num_proc
            )["train"]
            print("[PrecomputedMultiTask] Building embed_file index map...")
            embed_file_col = embed_dataset["embed_file"]
            embed_file_to_row = {name: i for i, name in enumerate(embed_file_col)}
            print(
                f"[PrecomputedMultiTask] Built map for {len(embed_file_to_row)} unique embedding files."
            )
        else:
            print(
                "[PrecomputedMultiTask] No parquet files found. Using standard numpy loading."
            )
            embed_dataset = None
            embed_file_to_row = None

        # Parse task tokens
        task_tokens = getattr(args, "task_tokens", ["<vi_en>", "<vi_zh>", "<vi_ko>"])
        TASK_TO_FIELD = PrecomputedMultiTaskDataset.TASK_TO_FIELD
        task_configs = []
        for token in task_tokens:
            field_name = TASK_TO_FIELD.get(token, token.strip("<>"))
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                raise RuntimeError(f"Task token '{token}' not in tokenizer vocab.")
            task_configs.append((field_name, token_id))
            print(f"  {token} → column='{field_name}', token_id={token_id}")

        # Load index
        with open(os.path.join(precomputed_dir, "index.json")) as f:
            full_index = json.load(f)
        print(f"[PrecomputedMultiTask] Loaded index: {len(full_index)} samples")

        # Load pre-tokenized text
        token_ids_map = {}
        tokens_dir = os.path.join(precomputed_dir, "token_ids")
        for field_name, _ in task_configs:
            tok_path = os.path.join(tokens_dir, f"{field_name}.json")
            if not os.path.exists(tok_path):
                raise FileNotFoundError(f"Token file {tok_path} not found.")
            with open(tok_path) as f:
                token_ids_map[field_name] = json.load(f)
            print(
                f"[PrecomputedMultiTask] Loaded {len(token_ids_map[field_name])} "
                f"token sequences for '{field_name}'"
            )

        # Filter: check that all tasks have non-empty text and fit in max_length
        valid_indices = []
        for entry in full_index:
            data_idx = entry["idx"]
            ok = True
            for field_name, _ in task_configs:
                tids = token_ids_map[field_name][data_idx]
                if not tids:
                    ok = False
                    break
                if (2 + len(tids)) > args.max_length:
                    ok = False
                    break

            # Check embed file exists
            if ok:
                if embed_file_to_row is not None:
                    if entry["embed_file"] not in embed_file_to_row:
                        ok = False
                else:
                    if not os.path.exists(os.path.join(embed_dir, entry["embed_file"])):
                        ok = False

            if ok:
                valid_indices.append(entry)

        print(
            f"[PrecomputedMultiTask] After filtering: {len(valid_indices)}/{len(full_index)} valid"
        )

        # Split train/val (same seed as original)
        import random

        rng = random.Random(42)
        shuffled = list(valid_indices)
        rng.shuffle(shuffled)
        split_point = max(1, int(len(shuffled) * 0.99))
        train_index = shuffled[:split_point]
        val_index = shuffled[split_point:]

        n_tasks = len(task_configs)
        print(
            f"[PrecomputedMultiTask] Split: {len(train_index)} train / {len(val_index)} val"
        )
        print(
            f"[PrecomputedMultiTask] Effective (×{n_tasks}): "
            f"{len(train_index)*n_tasks} train / {len(val_index)*n_tasks} val"
        )

        train_ds = (
            PrecomputedMultiTaskDataset(
                args,
                train_index,
                token_ids_map,
                task_configs,
                tokenizer,
                embed_dir,
                save_dtype,
                is_train=True,
                embed_dataset=embed_dataset,
                embed_file_to_row=embed_file_to_row,
            )
            if train
            else None
        )

        val_ds = (
            PrecomputedMultiTaskDataset(
                args,
                val_index,
                token_ids_map,
                task_configs,
                tokenizer,
                embed_dir,
                save_dtype,
                is_train=False,
                embed_dataset=embed_dataset,
                embed_file_to_row=embed_file_to_row,
            )
            if valid
            else None
        )

        test_ds = (
            PrecomputedMultiTaskDataset(
                args,
                val_index,
                token_ids_map,
                task_configs,
                tokenizer,
                embed_dir,
                save_dtype,
                is_train=False,
                embed_dataset=embed_dataset,
                embed_file_to_row=embed_file_to_row,
            )
            if test
            else None
        )

        return train_ds, val_ds, test_ds
