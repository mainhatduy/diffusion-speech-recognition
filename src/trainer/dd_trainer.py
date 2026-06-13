import transformers
from transformers import Trainer
from transformers.utils import logging
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, seed_worker, has_length
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from typing import List, Dict, Any, Union, Optional, Tuple, Callable

from data.dd_data import TokenSizeDistributedLengthGroupSampler
from dd_generator import DiscreteDiffusionGenerator

from dataclasses import dataclass, field
from typing import Optional

from utils import mean_ds, is_master

from tqdm import tqdm

import math

import os

from transformers import TrainingArguments

from dataclasses import dataclass, field

from typing import List

@dataclass
class DiscreteDiffusionArguments(TrainingArguments):
    batch_by_tokens: bool = field(
        default=False
    )
    eval_metric: str = field(
        default="none"
    )
    weighting: str = field(
        default="linear",
        metadata={"help": "weighting for training losses"}
    )
    mask_on_source: bool = field(
        default=False,
        metadata={"help": "whether masking is performed on source side"}
    )
    mask_on_paddings: bool = field(
        default=False,
        metadata={"help":"whether apply masking on paddings"}
    )

@dataclass
class DiscreteDiffusionTrainingArguments(DiscreteDiffusionArguments):
    finetune_from_model: str = field(
        default=None,
        metadata={"help": "results from previous stage, used for multiple stage training"}
    )
    mask_ratio_sampler: str = field(
        default="diffusion",
        metadata={"help": "diffusion|fixed[mask-ratio]. to decide whether fixed mask ratio mlm or diffusion trianing"}
    )
    train_length: bool = field(
        default=False
    )
    wandb_project: str = field(
        default="mlm-to-dlm",
        metadata={"help": "wandb project name"}
    )
    optimizer: torch.optim.Optimizer = field(
        default=None,
        metadata={"help": "optimizer for training"}
    )
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR = field(
        default=None,
        metadata={"help": "learning rate scheduler for training"}
    )



class DiscreteDiffusionTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: DiscreteDiffusionTrainingArguments = None,
        generator: DiscreteDiffusionGenerator = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # SỬA FILE: src/trainer/dd_trainer.py
        
        super().__init__(
            model=model, 
            args=args, 
            data_collator=data_collator, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            processing_class=tokenizer, 
            model_init=model_init, 
            compute_metrics=compute_metrics, 
            callbacks=callbacks, 
            optimizers=optimizers, 
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.generator = generator 
        self.eval_compute_loss = True
        # self.dictionary = generator.dictionary 
        
    def get_token_batched_dataloader(self, dataset, train=False):
        lengths = [dataset.size(i) for i in tqdm(range(len(dataset)))]
        batch_sampler = TokenSizeDistributedLengthGroupSampler(
            self.args.train_batch_size if train else self.args.eval_batch_size, # max_tokens
            self.args.max_length,
            dataset=dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            model_input_name=None,
            lengths=lengths,
            infinite=train
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker
        )
        return dataloader

    def get_train_dataloader(self):
        # self.train_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            return self.get_token_batched_dataloader(self.train_dataset, train=True)
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        # if eval_dataset is not None:
        #     eval_dataset.set_max_length(self.args.max_length)
        # else:
        #     self.eval_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            return self.get_token_batched_dataloader(eval_dataset)
        else:
            return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # test_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            return self.get_token_batched_dataloader(test_dataset)
        else:
            return super().get_test_dataloader(test_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raw_model = model.module if hasattr(model, "module") else model
        target = inputs["net_input"]["src_tokens"]
        partial_masks = (
            inputs["net_input"]["partial_masks"] 
            if not self.args.mask_on_source
            else torch.zeros_like(target).bool()
        )
        
        # couple
        if self.args.mask_ratio_sampler == "diffusion":
            t1, t2 = torch.randint(
                1, raw_model.args.num_diffusion_timesteps + 1, (2 * target.size(0), ), device=target.device
            ).chunk(2)
        elif self.args.mask_ratio_sampler.startswith("fixed"):
            ratio = float(self.args.mask_ratio_sampler[5:])
            t1, t2 = torch.tensor(
                [math.ceil(ratio * raw_model.args.num_diffusion_timesteps)] * (2 * target.size(0)),
                device=target.device, dtype=torch.long
            ).chunk(2)

        maskable_mask = (~partial_masks)
        if not self.args.mask_on_paddings: 
            maskable_mask = maskable_mask & target.ne(self.generator.pad_id)
        x_t, t, loss_mask = list(
            raw_model.q_sample_coupled(
                target, t1, t2,
                maskable_mask=maskable_mask
            ).values()
        )
        
        target = target.repeat(2, 1)
        partial_masks = partial_masks.repeat(2, 1)
        
        # Extract audio features if present (for speech_recognition)
        audio_features = inputs["net_input"].get("audio_features", None)
        audio_attention_mask = inputs["net_input"].get("audio_attention_mask", None)
        if audio_features is not None:
            audio_features = audio_features.repeat(2, 1)
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.repeat(2, 1)
        
        attention_mask = torch.ones_like(x_t) if self.args.mask_on_paddings else None
        logits = model(x_t, partial_masks, attention_mask=attention_mask, loss_mask=loss_mask,
                       audio_features=audio_features, audio_attention_mask=audio_attention_mask)
        
        num_timesteps = raw_model.args.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps
        }[self.args.weighting][:, None].float()
        weight = weight.expand(loss_mask.size())[loss_mask]
        # cnt_weight = loss_mask.sum(-1)[:, None].expand(loss_mask.size())[loss_mask]
        # cnt_weight = x_t.size(-1)
        cnt_weight = maskable_mask.repeat(2, 1).sum(dim=-1)[:, None].expand(loss_mask.size())[loss_mask]
        ce = F.cross_entropy(logits, target[loss_mask], reduction="none").float()   # num_masked samples
        ce = (ce * weight / cnt_weight).sum() / x_t.size(0) 
        # /  mean_ds(ce.mean(-1) * weight * num_timesteps)
        ls = self.args.label_smoothing_factor
        if ls > 0:
            logit_loss = -F.log_softmax(logits, dim=-1).mean(dim=-1).float()
            logit_loss = num_timesteps * (logit_loss / cnt_weight).sum() / x_t.size(0)
            
            diffusion_loss = ((1 - ls) * ce + ls * logit_loss)
        else:
            diffusion_loss = ce
        return (diffusion_loss, logits) if return_outputs else diffusion_loss
    
    def set_eval_compute_loss(self, value):
        self.eval_compute_loss = value
    
    def begin_write_prediction(self, prediction_write_to):
        if prediction_write_to is None:
            return
        assert not hasattr(self, "write_to"), "writo file already exists"
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            file_name = f"{prediction_write_to}.{rank}"
        else:
            file_name = prediction_write_to
        self.prediction_write_to = prediction_write_to
        self.write_to = open(file_name, "w")
    
    def end_write_prediction(self):
        if not hasattr(self, "write_to"):
            return 
        self.write_to.close()
        delattr(self, "write_to")
        if not torch.distributed.is_initialized():
            return 
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:   # aggregate all 
            lines = []
            for i in range(torch.distributed.get_world_size()):
                with open(f"{self.prediction_write_to}.{i}", "r") as f:
                    lines = lines + [line.strip() for line in f]
                os.remove(f"{self.prediction_write_to}.{i}")
            
            with open(self.prediction_write_to, "w") as f:
                f.write('\n'.join(lines))
        
        delattr(self, "prediction_write_to")
 
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, Any]] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger = logging.get_logger(__name__)
        logger.info(f"Saving model checkpoint to {output_dir}")

        from transformers.modeling_utils import PreTrainedModel
        from transformers.utils import is_peft_available, WEIGHTS_NAME
        
        if is_peft_available():
            from peft import PeftModel
            supported_classes = (PreTrainedModel, PeftModel)
        else:
            supported_classes = (PreTrainedModel,)

        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model, keep_torch_compile=False), supported_classes):
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, saving its state dict with torch.save to avoid safetensors shared memory error.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
         
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.has_printed_sample = False
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    @torch.no_grad()
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if not self.eval_compute_loss:
            loss = torch.tensor([0.]).to(inputs['target'].device)
        else:
            loss = self.compute_loss(model, inputs)

        hyps, history = self.generator.generate(model, inputs)        
        refs = inputs["target"]

        if is_master() and (not hasattr(self, "has_printed_sample") or not self.has_printed_sample):
            try:
                import os
                import json
                import miniaudio
                import numpy as np
                
                json_path = "test/test_data/test_sample.json"
                mp3_path = "test/test_data/test_sample.mp3"
                
                raw_model = model.module if hasattr(model, "module") else model
                is_speech = getattr(raw_model, "has_audio_encoder", False) or "audio_features" in inputs["net_input"] or (hasattr(self, "eval_dataset") and self.eval_dataset is not None and hasattr(self.eval_dataset, "feature_extractor"))
                
                if is_speech and os.path.exists(json_path) and os.path.exists(mp3_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        sample_info = json.load(f)
                    target_text = sample_info["text"]
                    
                    data = miniaudio.decode_file(mp3_path)
                    waveform = np.array(data.samples, dtype=np.float32)
                    if data.nchannels > 1:
                        waveform = waveform.reshape(-1, data.nchannels).mean(axis=1)
                    waveform = waveform / 32768.0
                    
                    target_sample_rate = 16000
                    if data.sample_rate != target_sample_rate:
                        ratio = target_sample_rate / data.sample_rate
                        new_length = int(len(waveform) * ratio)
                        indices = np.linspace(0, len(waveform) - 1, new_length)
                        waveform = np.interp(indices, np.arange(len(waveform)), waveform)
                        
                    if hasattr(self, "eval_dataset") and self.eval_dataset is not None and hasattr(self.eval_dataset, "feature_extractor"):
                        feature_extractor = self.eval_dataset.feature_extractor
                    else:
                        from transformers import Wav2Vec2FeatureExtractor
                        audio_encoder_name = getattr(raw_model.args, "audio_encoder_name", "facebook/mms-300m")
                        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_name)
                        
                    audio_inputs = feature_extractor(
                        waveform,
                        sampling_rate=target_sample_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    
                    device = inputs["net_input"]["src_tokens"].device
                    audio_values = audio_inputs.input_values.to(device)
                    
                    # Round up audio length to a multiple of 80 if we are using Moonshine encoder
                    audio_encoder_name = getattr(raw_model.args, "audio_encoder_name", "facebook/mms-300m")
                    if "moonshine" in audio_encoder_name.lower():
                        audio_len = audio_values.size(-1)
                        padded_len = ((audio_len + 79) // 80) * 80
                        
                        padded_audio = torch.zeros(1, padded_len, device=device)
                        padded_audio[0, :audio_len] = audio_values[0]
                        audio_values = padded_audio
                        
                        padded_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
                        padded_mask[0, :audio_len] = 1
                        audio_attention_mask = padded_mask
                    else:
                        if "attention_mask" in audio_inputs:
                            audio_attention_mask = audio_inputs.attention_mask.to(device)
                        else:
                            audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long).to(device)
                    
                    tokenizer = self.generator.tokenizer
                    tgt = tokenizer.encode(target_text, add_special_tokens=True)
                    if len(tgt) > 0 and tgt[0] == tokenizer.bos_token_id:
                        tgt = tgt[1:]
                    src = [tokenizer.bos_token_id]
                    
                    sources = [torch.tensor(src + tgt)]
                    targets = [torch.tensor([tokenizer.bos_token_id] + tgt)]
                    src_lengths = [len(src)]
                    
                    source_padded = torch.nn.utils.rnn.pad_sequence(
                        sources, batch_first=True, padding_value=tokenizer.pad_token_id
                    ).to(device)
                    target_padded = torch.nn.utils.rnn.pad_sequence(
                        targets, batch_first=True, padding_value=tokenizer.pad_token_id
                    ).to(device)
                    
                    batch_size, seq_len = source_padded.size()
                    src_lengths_tensor = torch.tensor(src_lengths, dtype=torch.long)
                    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                    partial_masks = (position_ids < src_lengths_tensor.unsqueeze(1)).to(device)
                    
                    test_batch = {
                        "id": torch.tensor([0]).to(device),
                        "net_input": {
                            "src_tokens": source_padded,
                            "src_lengths": torch.tensor([len(s) for s in sources]).to(device),
                            "partial_masks": partial_masks,
                            "audio_features": audio_values,
                            "audio_attention_mask": audio_attention_mask
                        },
                        "target": target_padded,
                        "nsentences": 1,
                        "ntokens": len(targets[0])
                    }
                    
                    test_hyps, _ = self.generator.generate(raw_model, test_batch)
                    
                    ref_str = target_text
                    hyp_str = self.generator.decode(test_hyps)[0]
                    
                    print("\n" + "="*80)
                    print(f"VALIDATION SAMPLE PREDICTION (Fixed Audio Sample from {mp3_path})")
                    print(f"INPUT: [Audio File: {mp3_path}]")
                    print(f"TARGET:", ref_str)
                    print(f"PRED:",  hyp_str)
                    print("="*80 + "\n")
                    
                    self.has_printed_sample = True
                else:
                    batch_size = inputs["net_input"]["src_tokens"].size(0)
                    random_idx = torch.randint(0, batch_size, (1,)).item()
                    
                    mask = inputs["net_input"]["partial_masks"][random_idx]
                    src_tokens = inputs["net_input"]["src_tokens"][random_idx]
                    real_src_tokens = src_tokens[mask]
                    
                    src_str = self.generator.decode(real_src_tokens[None, :])[0]
                    ref_str = self.generator.decode(refs[random_idx:random_idx+1])[0]
                    hyp_str = self.generator.decode(hyps[random_idx:random_idx+1])[0]
                    
                    print("\n" + "="*80)
                    print(f"VALIDATION SAMPLE PREDICTION (Random sample {random_idx} from batch of {batch_size})")
                    print(f"INPUT:",  src_str)
                    print(f"TARGET:", ref_str)
                    print(f"PRED:",  hyp_str)
                    print("="*80 + "\n")
                    
                    self.has_printed_sample = True
            except Exception as e:
                print(f"Failed to print validation sample: {e}")
                try:
                    batch_size = inputs["net_input"]["src_tokens"].size(0)
                    random_idx = torch.randint(0, batch_size, (1,)).item()
                    
                    mask = inputs["net_input"]["partial_masks"][random_idx]
                    src_tokens = inputs["net_input"]["src_tokens"][random_idx]
                    real_src_tokens = src_tokens[mask]
                    
                    src_str = self.generator.decode(real_src_tokens[None, :])[0]
                    ref_str = self.generator.decode(refs[random_idx:random_idx+1])[0]
                    hyp_str = self.generator.decode(hyps[random_idx:random_idx+1])[0]
                    
                    print("\n" + "="*80)
                    print(f"VALIDATION SAMPLE PREDICTION (Fallback Random sample {random_idx} from batch of {batch_size})")
                    print(f"INPUT:",  src_str)
                    print(f"TARGET:", ref_str)
                    print(f"PRED:",  hyp_str)
                    print("="*80 + "\n")
                    self.has_printed_sample = True
                except Exception as e2:
                    print(f"Fallback failed too: {e2}")

        if hasattr(self, "write_to") or not prediction_loss_only:
            hyps_seqs = self.generator.decode(hyps)
            refs_seqs = self.generator.decode(refs)

            if hasattr(self, "write_to"):
                inputs_seqs = self.generator.decode(inputs["net_input"]["src_tokens"])
                if history is not None:
                    # import ipdb; ipdb.set_trace()
                    history_seqs = [
                        [self.generator.decode(step[None, :], preserve_special=True)[0] for step in his] 
                        for his in history
                    ]
                    for index, src, hyp, ref, his in zip(inputs["id"], inputs_seqs, hyps_seqs, refs_seqs, history_seqs):
                        index = index.item()
                        self.write_to.write(f"SRC-{index}\t{src}\nHYP-{index}\t{hyp}\nREF-{index}\t{ref}\n") 
                        for i, his_seq in enumerate(his):
                            self.write_to.write(f"STEP{i}-{index}\t{his_seq}\n")
                else:
                    for index, src, hyp, ref in zip(inputs["id"], inputs_seqs, hyps_seqs, refs_seqs):
                        index = index.item()
                        self.write_to.write(f"SRC-{index}\t{src}\nHYP-{index}\t{hyp}\nREF-{index}\t{ref}\n")
                    
            if (not prediction_loss_only) and (self.compute_metrics is not None):
                if self.args.eval_metric == "bleu":
                    bleu = self.generator.compute_bleu(hyps_seqs, refs_seqs)
                    
                    sys_stat = torch.tensor([*bleu.counts, bleu.sys_len]).to(loss)
                    ref_stat = torch.tensor([*bleu.totals, bleu.ref_len]).to(loss)
                elif self.args.eval_metric == "rouge":
                    rouge = self.generator.compute_rouge(hyps_seqs, refs_seqs)
                    sys_stat = torch.tensor([rouge]).to(loss)
                    ref_stat = torch.tensor([len(hyps_seqs)]).to(loss)
                elif self.args.eval_metric == "smatchpp":
                    smatchpp_scores = self.generator.compute_smatchpp(hyps_seqs, refs_seqs)
                    # Return all metrics: F1, Precision, Recall
                    sys_stat = torch.tensor([
                        smatchpp_scores['f1'],
                        smatchpp_scores['precision'],
                        smatchpp_scores['recall']
                    ]).to(loss)
                    ref_stat = torch.tensor([1.0, 1.0, 1.0]).to(loss)
                    
                return (loss, sys_stat, ref_stat)
        return (loss, None, None)

class DiscreteDiffusionLengthTrainer(DiscreteDiffusionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # global global_step_step 
        # global_step_step += 1
        # if global_step_step % 2 == 0:
        #     return super().compute_loss(model, inputs)  
        raw_model = model.module if hasattr(model, "module") else model
        partial_masks = inputs["net_input"]["partial_masks"] 
        partial_masks[:, 0] = True
        input_tokens = inputs["net_input"]["src_tokens"].masked_fill(~partial_masks, raw_model.pad_id)
        max_index = input_tokens.ne(raw_model.pad_id).sum(dim=-1).max()
        input_tokens = input_tokens[:, :max_index]
        
        target = (
            (~partial_masks) &
            inputs["net_input"]["src_tokens"].ne(raw_model.pad_id)
        ).sum(-1).clamp(2) - 2 # -eos, 1->0
        logits = raw_model.forward_length(input_tokens)
        loss = F.cross_entropy(logits, target)
        return (loss, logits) if return_outputs else loss 