from functools import lru_cache
from dataclasses import asdict
import json
from typing import Callable, Dict

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    RobertaConfig,
    AutoModelForQuestionAnswering,
    RobertaForQuestionAnswering,
    AutoTokenizer,
    RobertaTokenizer,
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

from flask import Flask, Request, Response, after_this_request, request, jsonify
from allennlp.version import VERSION
from allennlp.predictors.predictor import JsonDict
from allennlp.interpret.saliency_interpreters import (
    SaliencyInterpreter,
    SimpleGradient,
    SmoothGradient,
    IntegratedGradient,
)
from allennlp.interpret.attackers import Attacker, Hotflip, InputReduction

# from allennlp_demo.common import config
# from allennlp_demo.common.logs import configure_logging
# from allennlp_demo.common import config, http

model_path = "./api/allennlp_demo/common/qa_models/roberta/deepsettosed_kf2/checkpoint-14800"
# model_path = "deepset/roberta-base-squad2"
config = RobertaConfig.from_pretrained(
    model_path,
    cache_dir="./cache",
    use_auth_token=None,
)
tokenizer = RobertaTokenizer.from_pretrained(
    model_path,
    cache_dir="./cache",
    use_auth_token= None,
#         force_download=True,
)
model = RobertaForQuestionAnswering.from_pretrained(
    model_path,
    cache_dir="./cache",
    config=config,
    # use_auth_token=None,
)