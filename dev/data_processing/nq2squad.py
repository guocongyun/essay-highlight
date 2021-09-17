#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for question answering.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
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
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

def create_data(title, ques, ids, context, ans, ans_start, is_impossible):
    data = {
            "title":title,
            "paragraphs":[
                {
                    "qas":[
                        {
                            "question":ques,
                            "id":ids,
                            "answers":[
                                    {
                                        "text":ans,
                                        "answer_start":ans_start
                                    }
                                
                            ],
                            "is_impossible":is_impossible
                        }
                    ],
                    "context":context
                }
            ]
        }
    return data

raw_datasets = load_dataset("natural_questions", cache_dir="../../data/natural_questions")
count = 0
datas = []
for idx, data in enumerate(raw_datasets["train"]):
#     with open("./test2.json", "w+") as f:
#         json.dump(data, f)

    s, e = data["annotations"]["long_answer"][0]["start_token"], data["annotations"]["long_answer"][0]["end_token"]
    ans_start = s-sum(data["document"]["tokens"]["is_html"][:s])
    ne = e-s+ans_start
    texts = []
    for idx, t in enumerate(data["document"]["tokens"]["token"]):
        if not data["document"]["tokens"]["is_html"][idx]: texts.append(t)
    context = " ".join(texts)
    ans = " ".join(texts[ans_start:ne])
    ids = data["id"]
    idx = idx
    title = data["document"]["title"]
    ques = data["question"]["text"]
    is_impossible = True if e >= s else False
    
    data = create_data(title=title, ques=ques, ids=ids, context=context, ans=ans, ans_start=ans_start, is_impossible=is_impossible)
    datas.append(data)
    print(data)
#     print(" ".join(context[ans_start:ne]))
#     print("------")
#     print(" ".join(data["document"]["tokens"]["token"][s:e]))
    # count += 1
    # if count > 2: exit()
ds = datasets.from_dict({"data":datas})
ds.save_to_disk("./test")