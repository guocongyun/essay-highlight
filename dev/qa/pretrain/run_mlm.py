import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForMaskedLM
import datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Dict, List, Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data.dataset import Dataset
import regex as re
import os
import click
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast, RobertaTokenizer
from transformers import RobertaConfig
import json
import torch
torch.cuda.is_available()


class MyLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        file_list = [os.path.join(file_path, i) for i in os.listdir(file_path)]
        lines = []
        for i in file_list:
            with open(i, encoding="utf-8") as f:
                lines += [line for line in f.read().splitlines()
                          if (len(line) > 0 and not line.isspace())]
        print("load txt")
        batch_encoding = tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(
            e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class MyRobertaTokenizer(RobertaTokenizer):
    def _tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # token = "".join(
            #     self.byte_encoder[b] for b in token
            # )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

def main():
    data_files = {}
    data_files["train"] = "./all.txt"
#     data_files["validation"] = "./fast-data/vali.txt"
    extension = "text"
    cache_dir = "./cache"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     model = RobertaForMaskedLM(config=config)
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=512,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

#     tokenized_datasets = raw_datasets.map(
#         tokenize_function,
#         batched=True,
#         batch_size=2000,
#         num_proc=24,
#         remove_columns=[text_column_name],
#         desc="Running tokenizer on dataset line_by_line",
#     )
#     tokenized_datasets.save_to_disk("./all")
#     kaggle_datasets = load_from_disk("./kaggle")
    sed_datasets = load_from_disk("./sed")
#     squad_datasets = load_from_disk("./squad")
    all_datasets = load_from_disk("./all")
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    train_args = TrainingArguments(per_gpu_train_batch_size=4, num_train_epochs=100, output_dir="./output_dir", save_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=all_datasets["train"],
        eval_dataset=sed_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
#     logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)

    trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
