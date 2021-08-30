from functools import lru_cache
from dataclasses import asdict
import json
from typing import Callable, Deque, Dict, List

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

from transformers.file_utils import ModelOutput
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.dummy_pt_objects import RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST
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
import torch
from allennlp_demo.common import config
from allennlp_demo.common.logs import configure_logging
from allennlp_demo.common import config, http
import zipfile
import glob

class RobertaModelEndpoint(http.MyModelEndpoint):
    def __init__(self):
        # pass
        model_path = "/app/allennlp_demo/common/qa_models/roberta/deepsettosed_kf2/checkpoint-14800"
        c = config.Model.from_file(os.path.join(os.path.dirname(__file__), "model.json"))
        super().__init__(model_path, c)

if __name__ == "__main__":
    endpoint = RobertaModelEndpoint()
    endpoint.run()
    # inputs = {
    #     "passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
    #     "question": "How many partially reusable launch systems were developed?",
    # }
    # prediction = endpoint.predict(inputs)
    # print(prediction)
