import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class StreamingPrecomputedMultiTaskDataset(IterableDataset):
    """Streaming version of PrecomputedMultiTaskDataset.

    Streams parquet shards directly from Hugging Face Hub using `datasets.load_dataset(..., streaming=True)`.
    Yields individual task samples without downloading or loading full dataset into memory.
    """

    TASK_TO_FIELD = {"<vi_en>": "english", "<vi_zh>": "chinese", "<vi_ko>": "korean"}

    def __init__(
        self,
        args,
        hf_dataset,
        task_configs,
        tokenizer,
        save_dtype=np.float16,
        is_train=True,
        buffer_size=10000,
        max_samples=None,
    ):
        super().__init__()
        self.args = args
        self.hf_dataset = hf_dataset
        self.task_configs = task_configs  # [(field_name, task_token_id), ...]
        self.tokenizer = tokenizer
        self.save_dtype = save_dtype
        self.is_train = is_train
        self.max_length = args.max_length
        self.n_tasks = len(task_configs)
        self.buffer_size = buffer_size
        self.max_samples = max_samples
        self._rainbow_pad_ids = None

    def _get_rainbow_pad_ids(self):
        """Cache rainbow pad token IDs."""
        if self._rainbow_pad_ids is None:
            self._rainbow_pad_ids = [
                self.tokenizer.convert_tokens_to_ids(f"<rpad_{i}>")
                for i in range(7)
            ]
        return self._rainbow_pad_ids

    def _process_row(self, row, sample_counter):
        """Process a single parquet row and yield samples for each task."""
        # 1. Parse audio embeddings
        embedding_bytes = row["embedding_bytes"]
        shape = row.get("shape", None)
        audio_embeds = np.frombuffer(embedding_bytes, dtype=self.save_dtype)
        if shape is not None and len(shape) > 0:
            audio_embeds = audio_embeds.reshape(shape)
        else:
            audio_embeds = audio_embeds.reshape(-1, 768)
        audio_embeds_tensor = torch.from_numpy(audio_embeds.astype(np.float32))

        # 2. Iterate through configured tasks
        for task_idx, (tgt_field, task_token_id) in enumerate(self.task_configs):
            tgt = list(row.get(tgt_field, []))
            if not tgt:
                continue
            if (2 + len(tgt)) > self.max_length:
                continue

            if len(tgt) > self.max_length:
                tgt = tgt[: self.max_length]

            # Build source prefix with task token
            src = [self.tokenizer.bos_token_id, task_token_id]

            # Strip BOS from tgt if present
            if len(tgt) > 0 and tgt[0] == self.tokenizer.bos_token_id:
                tgt = tgt[1:]

            src_length = len(src)
            concatenated = src + tgt
            if len(concatenated) > self.max_length:
                concatenated = concatenated[: self.max_length]
            remaining = self.max_length - len(concatenated)
            if remaining > 0:
                eos_id = self.tokenizer.eos_token_id
                rpad_ids = self._get_rainbow_pad_ids()
                pad_seq = [eos_id]
                for j in range(remaining - 1):
                    pad_seq.append(rpad_ids[j % len(rpad_ids)])
                concatenated = concatenated + pad_seq[:remaining]

            target_tgt = (
                [self.tokenizer.bos_token_id] + tgt
                if len(tgt) == 0 or tgt[0] != self.tokenizer.bos_token_id
                else tgt
            )

            flat_id = sample_counter * self.n_tasks + task_idx

            yield {
                "id": flat_id,
                "source": torch.tensor(concatenated),
                "target": torch.tensor(target_tgt),
                "src_length": src_length,
                "precomputed_audio_embeds": audio_embeds_tensor,
            }

    def __iter__(self):
        dataset_stream = self.hf_dataset
        if self.is_train and self.buffer_size > 0:
            dataset_stream = dataset_stream.shuffle(
                buffer_size=self.buffer_size, seed=42
            )

        if self.max_samples is not None:
            dataset_stream = dataset_stream.take(self.max_samples)

        sample_counter = 0
        yielded_count = 0

        for row in dataset_stream:
            sample_counter += 1
            for sample in self._process_row(row, sample_counter):
                yield sample
                yielded_count += 1

    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        """Load streaming dataset from Hugging Face Hub."""
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        repo_id = getattr(
            args, "streaming_repo_id", "aiai-laboratory/vietspeech-train-streaming"
        )
        buffer_size = getattr(args, "streaming_buffer_size", 10000)
        val_samples = getattr(args, "val_streaming_size", 500)
        hf_token = getattr(args, "hf_token", None) or os.getenv("HF_TOKEN")

        print(f"[StreamingPrecomputedMultiTask] Connecting to HF Hub repo: {repo_id}")

        # Download metadata for save_dtype & info
        try:
            meta_file = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                repo_type="dataset",
                token=hf_token,
            )
            with open(meta_file) as f:
                metadata = json.load(f)
            save_dtype_str = metadata.get("save_dtype", "float16")
        except Exception as e:
            print(f"[StreamingPrecomputedMultiTask] Warning: metadata.json fetch failed ({e}). Defaulting float16.")
            save_dtype_str = "float16"

        save_dtype = np.float16 if save_dtype_str == "float16" else np.float32

        # Parse task tokens
        task_tokens = getattr(args, "task_tokens", ["<vi_en>", "<vi_zh>", "<vi_ko>"])
        TASK_TO_FIELD = StreamingPrecomputedMultiTaskDataset.TASK_TO_FIELD
        task_configs = []
        for token in task_tokens:
            field_name = TASK_TO_FIELD.get(token, token.strip("<>"))
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                raise RuntimeError(f"Task token '{token}' not in tokenizer vocab.")
            task_configs.append((field_name, token_id))
            print(f"  {token} → column='{field_name}', token_id={token_id}")

        # Load streaming HF dataset
        hf_dataset = load_dataset(
            "parquet",
            data_files=f"hf://datasets/{repo_id}/data/*.parquet",
            streaming=True,
            split="train",
        )

        train_ds = (
            StreamingPrecomputedMultiTaskDataset(
                args=args,
                hf_dataset=hf_dataset,
                task_configs=task_configs,
                tokenizer=tokenizer,
                save_dtype=save_dtype,
                is_train=True,
                buffer_size=buffer_size,
            )
            if train
            else None
        )

        val_ds = (
            StreamingPrecomputedMultiTaskDataset(
                args=args,
                hf_dataset=hf_dataset,
                task_configs=task_configs,
                tokenizer=tokenizer,
                save_dtype=save_dtype,
                is_train=False,
                buffer_size=0,
                max_samples=val_samples,
            )
            if valid
            else None
        )

        test_ds = (
            StreamingPrecomputedMultiTaskDataset(
                args=args,
                hf_dataset=hf_dataset,
                task_configs=task_configs,
                tokenizer=tokenizer,
                save_dtype=save_dtype,
                is_train=False,
                buffer_size=0,
                max_samples=val_samples,
            )
            if test
            else None
        )

        return train_ds, val_ds, test_ds
