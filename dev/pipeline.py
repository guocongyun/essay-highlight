import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    QuestionAnsweringPipeline,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

tok_name = "../../tok"
# model_name = "../../output_dir/checkpoint-31902"
model_name = "./output_dir/checkpoint-14800"
# file_path = "../../data/sed/test.json"
# tok_name = model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(tok_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)

ctx ="asdf. potato. hello. what. potato"
question = 'Quel est la taille de la personne ?'
res = pipeline({'question': question, 'context': ctx}, topk=5)
print(res)