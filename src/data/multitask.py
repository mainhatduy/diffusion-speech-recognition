import os
import logging
import multiprocessing as mp
import numpy as np
import torch
from typing import List
from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
from .base import PromptDataset
from .utils import _decode_wav_bytes, normalize_text

class MultiTaskTranslatedSpeechDataset(PromptDataset):
    """Multi-task speech translation dataset: audio (Vietnamese) -> text in N target languages.

    Each base audio sample is exposed N times — once per task — identified by a special
    task token prepended to both source prefix and target sequence:

        src   = [BOS, <vi_en>]                    (src_length = 2)
        tgt   = [BOS, word1, word2, ..., EOS]     (translation, no task token)
        input = [BOS, <vi_en>, word1, word2, ...]  (concatenated for diffusion)

    The task tokens (<vi_en>, <vi_zh>, <vi_ko>) are added as special tokens to the
    tokenizer before training; call model.resize_token_embeddings() afterwards.

    Args:
        task_configs: list of (field_name, task_token_id) pairs, e.g.
                      [("english", 250054), ("chinese", 250055), ("korean", 250056)]
    """

    # Mapping from full task token string to dataset column name
    TASK_TO_FIELD: dict = {
        "<vi_en>": "english",
        "<vi_zh>": "chinese",
        "<vi_ko>": "korean",
    }

    def __init__(
        self,
        args,
        raw_data,
        vietspeech_dataset,
        path_to_vs_idx: dict,
        task_configs: list,   # list[tuple[str, int]]
        tokenizer,
        feature_extractor,
    ):
        super().__init__(args, raw_data, tokenizer)
        self.vietspeech_dataset = vietspeech_dataset
        self.path_to_vs_idx = path_to_vs_idx
        self.task_configs = task_configs   # [(field_name, token_id), ...]
        self.feature_extractor = feature_extractor
        self.target_sample_rate = 16000
        self.n_tasks = len(task_configs)

    def __len__(self):
        # Each base sample appears n_tasks times (one per target language)
        return len(self.raw_data) * self.n_tasks

    def __getitem__(self, index):
        # Decompose flat index → (base sample, task)
        sample_idx = index // self.n_tasks
        task_idx   = index % self.n_tasks
        tgt_field, task_token_id = self.task_configs[task_idx]

        # ---- Fetch translated text ----
        translated_item = self.raw_data[sample_idx]
        wav_id = translated_item["id"]

        # ---- Fetch matching audio from VietSpeech ----
        vs_idx = self.path_to_vs_idx.get(wav_id)
        if vs_idx is None:
            raise ValueError(f"WAV ID '{wav_id}' not found in NhutP/VietSpeech index.")

        vs_item = self.vietspeech_dataset[vs_idx]
        audio_info = vs_item["audio"]
        waveform, sample_rate = _decode_wav_bytes(audio_info["bytes"])

        # Resample to 16 kHz if necessary
        if sample_rate != self.target_sample_rate:
            ratio = self.target_sample_rate / sample_rate
            new_length = int(len(waveform) * ratio)
            indices = np.linspace(0, len(waveform) - 1, new_length)
            waveform = np.interp(indices, np.arange(len(waveform)), waveform)

        audio_inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )
        audio_values = audio_inputs.input_values.squeeze(0)  # (T_samples,)

        # ---- Normalize target text ----
        text = translated_item.get(tgt_field, "") or ""
        normalized = normalize_text(text)

        tgt = self.tokenizer.encode(normalized, add_special_tokens=True)
        if len(tgt) > self.max_length:
            tgt = tgt[: self.max_length]

        # ---- Build source prefix with task token ----
        # src  = [BOS, <vi_XX>]   — the diffusion model sees this as the fixed prefix
        # tgt  = [word1, ...]     — what the diffusion model predicts / unmasks
        src = [self.tokenizer.bos_token_id, task_token_id]

        # Strip BOS from encoded tgt (BOS already lives in src)
        if len(tgt) > 0 and tgt[0] == self.tokenizer.bos_token_id:
            tgt = tgt[1:]

        src_length = len(src)            # 2: BOS + task_token
        concatenated = src + tgt         # [BOS, <vi_XX>, word1, ...]

        # ground-truth target tensor expected by diffusion loss
        target_tgt = ([self.tokenizer.bos_token_id] + tgt
                      if len(tgt) == 0 or tgt[0] != self.tokenizer.bos_token_id
                      else tgt)

        return {
            "id":           index,
            "source":       torch.tensor(concatenated)[-self.max_length :],
            "target":       torch.tensor(target_tgt),
            "src_length":   src_length,
            "audio_values": audio_values,
        }

    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        tokenizer.model_max_length = args.max_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "HF_TOKEN is required. Set it via --hf_token argument or HF_TOKEN env variable."
            )

        # 1. Parse task token names and resolve their IDs from tokenizer
        task_tokens: List[str] = getattr(args, "task_tokens", ["<vi_en>", "<vi_zh>", "<vi_ko>"])
        if not task_tokens:
            raise ValueError("args.task_tokens is empty; expected e.g. ['<vi_en>', '<vi_zh>', '<vi_ko>'].")

        print(f"[MultiTask] Using task tokens (pre-registered in tokenizer): {task_tokens}")

        # 2. Build task_configs: [(field_name, token_id), ...]
        TASK_TO_FIELD = MultiTaskTranslatedSpeechDataset.TASK_TO_FIELD
        task_configs = []
        for token in task_tokens:
            field_name = TASK_TO_FIELD.get(token, token.strip("<>"))
            token_id   = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                raise RuntimeError(
                    f"Task token '{token}' was not found in tokenizer vocab "
                    f"(returned unk_token_id). Make sure load_model_tokenizer() ran first."
                )
            task_configs.append((field_name, token_id))
            print(f"  {token} → column='{field_name}', token_id={token_id}")

        # 3. Load audio feature extractor
        audio_encoder_name = getattr(
            args, "audio_encoder_name", "UsefulSensors/moonshine-streaming-medium"
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            audio_encoder_name, cache_dir=getattr(args, "cache_dir", None)
        )

        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        num_proc = max(1, int(mp.cpu_count() / world_size))

        # 4. Load aiai-laboratory/vietspeech-train-translated
        print("[MultiTask] Loading translated dataset from aiai-laboratory/vietspeech-train-translated")
        try:
            translated_dataset = load_dataset(
                "aiai-laboratory/vietspeech-train-translated",
                token=hf_token,
                cache_dir=getattr(args, "cache_dir", None),
                split="train[:100%]",
            )
        except Exception as e:
            print(f"Error loading translated dataset: {e}")
            raise

        # Validate that all target fields exist in the dataset
        for field_name, _ in task_configs:
            if field_name not in translated_dataset.features:
                raise ValueError(
                    f"Field '{field_name}' not found in vietspeech-train-translated. "
                    f"Available columns: {list(translated_dataset.features.keys())}"
                )

        # 5. Load NhutP/VietSpeech for audio
        print("[MultiTask] Loading audio from NhutP/VietSpeech")
        try:
            vietspeech_dataset = load_dataset(
                "NhutP/VietSpeech",
                token=hf_token,
                cache_dir=getattr(args, "cache_dir", None),
                split="train[:100%]",
            )
        except Exception as e:
            print(f"Error loading VietSpeech dataset: {e}")
            raise

        # Decode=False to avoid torchcodec dependency
        vietspeech_dataset = vietspeech_dataset.cast_column("audio", Audio(decode=False))

        # 6. Build path → index mapping for VietSpeech
        print("[MultiTask] Building audio path→index mapping")
        audio_column = vietspeech_dataset.data.column("audio")
        vs_paths = []
        for chunk in audio_column.chunks:
            vs_paths.extend(chunk.field("path").to_pylist())
        path_to_vs_idx = {path: idx for idx, path in enumerate(vs_paths)}
        print(f"[MultiTask] VietSpeech index built: {len(path_to_vs_idx)} entries")

        # 7. Shuffle & split translated dataset
        translated_dataset = translated_dataset.shuffle(seed=42)
        split = translated_dataset.train_test_split(test_size=0.01, seed=42)
        train_raw = split["train"]
        valid_raw = split["test"]

        n_tasks = len(task_configs)
        print(
            f"[MultiTask] Base split: {len(train_raw)} train / {len(valid_raw)} val\n"
            f"[MultiTask] Effective (×{n_tasks} tasks): "
            f"{len(train_raw)*n_tasks} train / {len(valid_raw)*n_tasks} val"
        )

        # 8. Filter — sample must fit in max_length for ALL target languages
        def filter_fn(example):
            for field_name, _ in task_configs:
                text = example.get(field_name, "") or ""
                if not text:
                    return False
                normalized = normalize_text(text)
                t_ids = tokenizer.encode(normalized, add_special_tokens=True)
                # 2 tokens for src prefix (BOS + task_token) + target tokens
                if (2 + len(t_ids)) > args.max_length:
                    return False
            return True

        if train:
            train_raw = train_raw.filter(filter_fn, num_proc=num_proc)
        if valid:
            valid_raw = valid_raw.filter(filter_fn, num_proc=num_proc)

        print(
            f"[MultiTask] After filtering: {len(train_raw)} train / {len(valid_raw)} val "
            f"base samples"
        )

        # 9. Construct dataset objects
        train_dataset = (
            MultiTaskTranslatedSpeechDataset(
                args, train_raw, vietspeech_dataset,
                path_to_vs_idx, task_configs, tokenizer, feature_extractor,
            )
            if train else None
        )
        valid_dataset = (
            MultiTaskTranslatedSpeechDataset(
                args, valid_raw, vietspeech_dataset,
                path_to_vs_idx, task_configs, tokenizer, feature_extractor,
            )
            if valid else None
        )

        return train_dataset, valid_dataset, None
