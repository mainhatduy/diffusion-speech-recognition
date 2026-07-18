import torch
import torch.distributed as dist

from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from model.dd_model import DiscreteDiffusionXLMRModel

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.utils import logging

import os

from typing import Dict, List, Union, get_args

import math

logger = logging.get_logger(__name__)


def is_master():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def serialized_func(enable=False):
    def _serialized_func(func):
        def wrapped_func(*args, **kwargs):
            local_rank = int(os.environ["LOCAL_RANK"]) if dist.is_initialized() else 0
            local_world_size = (
                int(os.environ["LOCAL_WORLD_SIZE"]) if dist.is_initialized() else 1
            )
            ret = None
            for i in range(local_world_size):
                if dist.is_initialized() and enable:
                    dist.barrier()
                if i == local_rank:
                    ret = func(*args, **kwargs)
                    logger.info(f"Local rank {local_rank} is done")
            return ret

        return wrapped_func

    return _serialized_func


def mean_ds(x, dim=None):
    return (
        x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)
    )


def argument_filter(arguments):
    if isinstance(arguments, List):
        arg_list = []
        for item in arguments:
            if isinstance(item, get_args(Union[int, float, str])):
                arg_list.append(item)
            elif isinstance(item, get_args(List)):
                arg_list.append(argument_filter(item))
        return arg_list
    elif isinstance(arguments, Dict):
        arg_dict = {}
        for key, value in arguments.items():
            assert isinstance(key, str)
            if isinstance(value, get_args(Union[int, float, str])):
                arg_dict[key] = value
            elif isinstance(value, get_args(Union[Dict, List])):
                arg_dict[key] = argument_filter(value)
        return arg_dict


@serialized_func()
def load_ckpt(model, ckpt_path, do_train=False):
    files = os.listdir(ckpt_path)
    # lora
    if "adapter_model.bin" in files:
        model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=do_train)
    # pytorch_model.bin
    else:
        if "pytorch_model.bin" in files:
            state_dict = torch.load(
                f"{ckpt_path}/pytorch_model.bin", map_location="cpu"
            )
        # deepspeed
        else:
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )

            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        if isinstance(model, DiscreteDiffusionXLMRModel) and state_dict[
            "model.lm_head.decoder.weight"
        ].shape == torch.Size([0]):
            state_dict["model.lm_head.decoder.weight"] = state_dict[
                "model.roberta.embeddings.word_embeddings.weight"
            ]

        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(incompatible)
    return model  # , tokenizer


def _get_missing_special_tokens(tokenizer, tokenizer_pad_to_multiple):
    # add special tokens
    special_token_dict, padding_tokens = dict(), []
    if tokenizer.pad_token is None:
        special_token_dict["pad_token"] = "<pad>"
    if tokenizer.bos_token is None:
        special_token_dict["bos_token"] = "<s>"
    if tokenizer.eos_token is None:
        special_token_dict["eos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_token_dict["unk_token"] = "<unk>"
    if tokenizer.mask_token is None:
        special_token_dict["mask_token"] = "<mask>"
    current_vocab_size = len(tokenizer.get_vocab()) + len(special_token_dict)
    target_vocab_size = (
        math.ceil(current_vocab_size / tokenizer_pad_to_multiple)
        * tokenizer_pad_to_multiple
    )
    for i in range(target_vocab_size - current_vocab_size):
        assert (
            f"<unused{i}>" not in tokenizer.get_vocab()
        ), f"unused_{i} already exists in the vocabulary"
        padding_tokens.append(f"<unused{i}>")
    return special_token_dict, padding_tokens


# Task tokens that are permanently reserved for multi-task speech translation.
# These are added to EVERY tokenizer load so that model vocab and tokenizer
# are always in sync regardless of dataset_type or checkpoint resume.
RAINBOW_PAD_TOKENS = [f"<rpad_{i}>" for i in range(7)]
TASK_SPECIAL_TOKENS = ["<vi_en>", "<vi_zh>", "<vi_ko>"] + RAINBOW_PAD_TOKENS


# @serialized_func
def load_model_tokenizer(model_args, do_train):
    pretrained, config = model_args.pretrained, model_args.config
    model_type = pretrained if pretrained is not None else config
    model_type = "xlm-roberta"
    if pretrained is not None:
        model = {"xlm-roberta": AutoModelForMaskedLM}[model_type].from_pretrained(
            pretrained, cache_dir=model_args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            padding_side="right",
            use_fast=False,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = {"xlm-roberta": AutoModelForMaskedLM}[model_type].from_config(
            config, cache_dir=model_args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config, padding_side="right", use_fast=False, cache_dir=model_args.cache_dir
        )

    # ---------------------------------------------------------------
    # Permanently add the task special tokens to the tokenizer and
    # resize model embeddings accordingly.  We do this here — before
    # any dataset is loaded — so that:
    #   1. The vocab size is fixed and deterministic.
    #   2. Resuming from a checkpoint (where the tokenizer was saved
    #      with these tokens) stays consistent.
    #   3. Inference scripts that call load_model_tokenizer() directly
    #      automatically get the correct vocabulary.
    # ---------------------------------------------------------------
    tokens_to_add = [
        tok for tok in TASK_SPECIAL_TOKENS if tok not in tokenizer.get_vocab()
    ]
    if tokens_to_add:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        logger.info(
            f"[load_model_tokenizer] Added {num_added} task token(s) as special tokens: "
            f"{tokens_to_add}  (new vocab size: {len(tokenizer)})"
        )
    else:
        logger.info(
            f"[load_model_tokenizer] Task tokens already in vocab: {TASK_SPECIAL_TOKENS}"
        )

    dd_model = {"xlm-roberta": DiscreteDiffusionXLMRModel}[model_type](
        model_args, tokenizer, model
    )

    # Resize model token embeddings to match the updated tokenizer vocab.
    dd_model.resize_token_embeddings(len(tokenizer))
    logger.info(
        f"[load_model_tokenizer] Model embeddings resized to vocab size: {len(tokenizer)}"
    )

    if model_args.lora:
        lora_config = LoraConfig(
            TaskType.TOKEN_CLS,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            bias=model_args.lora_bias,
            lora_dropout=model_args.lora_dropout,
            inference_mode=(not do_train),
        )
        dd_model = get_peft_model(dd_model, lora_config)

    return dd_model, tokenizer
