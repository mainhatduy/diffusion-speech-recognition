import transformers
import dataclasses
from dataclasses import dataclass, field

import json

from model.dd_model import DiscreteDiffusionModelArguments
from data.dd_data import DiscreteDiffusionDataArguments, DiscreteDiffusionDataCollator, load_data
from trainer.dd_trainer import DiscreteDiffusionArguments, DiscreteDiffusionTrainingArguments, DiscreteDiffusionTrainer
from dd_generator import DiscreteDiffusionGeneratorArguments, DiscreteDiffusionGenerator, MergeBLEU, MergeWER

from copy import deepcopy
from typing import List

from utils import load_model_tokenizer, load_ckpt

import os

@dataclass
class DiscreteDiffusionEvalArguments:
    ckpt_args_file: str = field(
        default="",
        metadata={"help": "args file to load config"}
    )
    no_compute_loss: bool = field(
        default=False,
        metadata={"help": "whether to ignore computing loss"}
    )
    prediction_write_to: str = field(
        default=None
    )


@dataclass
class DiscreteDiffusionEvalDataArguments(DiscreteDiffusionDataArguments):
    data_path: List[str] = field(
        default_factory=lambda: []
    )
    
def main():
    parser = transformers.HfArgumentParser((
        DiscreteDiffusionEvalArguments,
        DiscreteDiffusionTrainingArguments, 
        DiscreteDiffusionEvalDataArguments,
        DiscreteDiffusionGeneratorArguments
    ))
    eval_args, train_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    with open(eval_args.ckpt_args_file, "r") as f:
        config = json.load(f)
    model_args = config['model']
    acceptable_model_args_keys = {item.name for item in dataclasses.fields(DiscreteDiffusionModelArguments)}
    for key in list(model_args.keys()):
        if key not in acceptable_model_args_keys:
            del model_args[key]
    model_args = DiscreteDiffusionModelArguments(**model_args)

    model, tokenizer = load_model_tokenizer(model_args, do_train=False)
    
    
    metric = {
        "none": None,
        "bleu": MergeBLEU(),
        "wer": MergeWER(),
    }.get(train_args.eval_metric, None)
    
    model = load_ckpt(model, train_args.resume_from_checkpoint)
    
    if eval_args.prediction_write_to is not None:
        os.makedirs(eval_args.prediction_write_to, exist_ok=True)
        
    for data_path in data_args.data_path:
        data_item_args_dict = deepcopy(data_args.__dict__)
        data_item_args_dict["data_path"] = data_path
        data_item_args = DiscreteDiffusionDataArguments(**data_item_args_dict)
        (train_set, valid_set, testset), collator = load_data(
            data_item_args, model_args, tokenizer, train=False, valid=False, test=True
        )
        generator = DiscreteDiffusionGenerator(gen_args, tokenizer=tokenizer) 
        

        trainer = DiscreteDiffusionTrainer(
            model=model, args=train_args, 
            generator=generator,
            data_collator=collator,
            compute_metrics=metric
        )
        trainer.set_eval_compute_loss(~eval_args.no_compute_loss)
        write_to_file_name = data_path.replace('/', '_') + ".txt"
        write_to = (
            f"{eval_args.prediction_write_to}/{write_to_file_name}"
            if eval_args.prediction_write_to is not None
            else None
        )
        trainer.begin_write_prediction(write_to) 
        result = trainer.evaluate(testset)
        trainer.end_write_prediction()
    
        print(result)
    
if __name__ == '__main__':
    main()