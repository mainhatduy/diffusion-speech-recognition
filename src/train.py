import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import torch

import transformers 
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint

import os
from dotenv import load_dotenv

from model.dd_model import DiscreteDiffusionModelArguments
from trainer.dd_trainer import DiscreteDiffusionTrainingArguments
from dd_generator import DiscreteDiffusionGenerator, DiscreteDiffusionGeneratorArguments
from dd_generator import MergeBLEU, MergeRouge, MergeSmatchPP, MergeWER, MultiMetric
from trainer.dd_trainer import DiscreteDiffusionTrainer, DiscreteDiffusionLengthTrainer
from utils import load_ckpt, is_master, argument_filter, load_model_tokenizer
from data.dd_data import (
    DiscreteDiffusionDataArguments, DiscreteDiffusionDataCollator, load_data
)

import json
import sys

def parse_args():
    parser = transformers.HfArgumentParser((
        DiscreteDiffusionDataArguments,   # data
        DiscreteDiffusionModelArguments,   # model
        DiscreteDiffusionTrainingArguments,
        DiscreteDiffusionGeneratorArguments,   # generation
    ))
    
    # Check if a config file is provided as the first argument
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        config_file = sys.argv[1]
        data_args, model_args, train_args, gen_args = parser.parse_json_file(json_file=config_file, allow_extra_keys=True)
    else:
        # Parse from command line arguments (original behavior)
        data_args, model_args, train_args, gen_args = parser.parse_args_into_dataclasses()
    
    if train_args.resume_from_checkpoint is None:
        if os.path.exists(train_args.output_dir):
            train_args.resume_from_checkpoint = get_last_checkpoint(train_args.output_dir)
        
    # dump the arguments
    if is_master():
        if not os.path.exists(train_args.output_dir):
            os.makedirs(train_args.output_dir)
        d = {
            "data": argument_filter(data_args.__dict__),
            "generator": argument_filter(gen_args.__dict__),
            "model": argument_filter(model_args.__dict__),
            "train": argument_filter(train_args.__dict__)
        }
        if train_args.deepspeed is not None:
            with open(train_args.deepspeed, "r") as f:
                ds = json.load(f)
            d["deepspeed"] = ds
        with open(f"{train_args.output_dir}/args.json", "w") as f:
            json.dump(d, f, indent=4)
    return data_args, model_args, train_args, gen_args

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Set WandB API key from environment
    if os.getenv("WANDB_TOKEN"):
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_TOKEN")
    
    data_args, model_args, train_args, gen_args = parse_args()
    
    # Initialize WandB if enabled
    if "wandb" in train_args.report_to:
        import wandb
        # Set WandB project and run name from config
        if not os.getenv("WANDB_PROJECT"):
            wandb_project = getattr(train_args, "wandb_project", "mlm-to-dlm")
            os.environ["WANDB_PROJECT"] = wandb_project
        if not os.getenv("WANDB_RUN_NAME"):
            wandb_run_name = getattr(train_args, "run_name", None)
            if wandb_run_name:
                os.environ["WANDB_RUN_NAME"] = wandb_run_name
    
    # init model
    # Pass dataset_type info to model_args so model knows whether to initialize audio encoder
    model_args.dataset_type = data_args.dataset_type
    if hasattr(data_args, 'audio_encoder_name'):
        model_args.audio_encoder_name = data_args.audio_encoder_name
    model, tokenizer = load_model_tokenizer(model_args, do_train=True)
    
    # load datasets 
    (train_set, valid_set, test_set), collator = load_data(data_args, model_args, tokenizer)
    generator = DiscreteDiffusionGenerator(gen_args, tokenizer=tokenizer)
    
    # resume checkpoint
    if train_args.finetune_from_model is not None:
        model = load_ckpt(model, train_args.finetune_from_model)
    
    # build trainer
    metric = None
    if train_args.eval_metrics:  # multi-metric path (eval_metrics list takes priority)
        metric = MultiMetric(train_args.eval_metrics)
    elif train_args.eval_metric == "bleu":
        metric = MergeBLEU()
    elif train_args.eval_metric == "rouge":
        metric = MergeRouge()
    elif train_args.eval_metric == "smatchpp":
        metric = MergeSmatchPP()
    elif train_args.eval_metric == "wer":
        metric = MergeWER()
    Trainer = DiscreteDiffusionTrainer if not train_args.train_length else DiscreteDiffusionLengthTrainer
    trainer = Trainer(
        model=model, args=train_args, 
        train_dataset=train_set, eval_dataset=valid_set,
        generator=generator,
        data_collator=collator,
        compute_metrics=metric,
        
    )
    # Save tokenizer alongside the model so that the task special tokens
    # (<vi_en>, <vi_zh>, <vi_ko>) are preserved in every checkpoint directory.
    if is_master():
        tokenizer.save_pretrained(train_args.output_dir)
        print(f"[train] Tokenizer saved to '{train_args.output_dir}' (vocab size: {len(tokenizer)})")

    # train!
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

if __name__ == '__main__':
    main()