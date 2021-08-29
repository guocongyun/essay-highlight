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

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
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
import numpy as np
from allennlp_demo.common import config
from allennlp_demo.common.logs import configure_logging
from allennlp_demo.common import config, http
import zipfile
import glob

class RobertaModelEndpoint(http.MyModelEndpoint):
    def __init__(self):
        # pass
        model_path = "/app/allennlp_demo/common/qa_models/roberta/deepsettosed_kf2/checkpoint-14800"
        self.max_seq_length = 512
        self.stride = 128
        self.pad_on_right = True
        self.start_weight = 0.5
        self.nbest = 10
        self.max_answer_length = 300
        self.min_answer_length = 5
        self.config = AutoConfig.from_pretrained(
            model_path,
            cache_dir="./cache",
            use_auth_token=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            cache_dir="./cache",
            use_auth_token= None,
    #         force_download=True,
        )
        self.model2 = AutoModelForQuestionAnswering.from_pretrained(
            model_path,
            cache_dir="./cache",
            revision="main",
            config=self.config,
            # use_auth_token=None,
        )
        c = config.Model.from_file(os.path.join(os.path.dirname(__file__), "model.json"))
        super().__init__(c)
    
    def preprocess(self, question: str, contexts: List[str]) -> Dict:
        all_inputs = []
        for context in contexts:
            inputs = self.tokenizer(
                question.lower(), 
                context.lower(), 
                truncation="only_second",
                max_length=self.max_seq_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length")
            for i in range(len(inputs["input_ids"])):
                sequence_ids = inputs.sequence_ids(i)
                context_index = 1 if self.pad_on_right else 0
                inputs["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(inputs["offset_mapping"][i])
                ]
            all_inputs.append(inputs)
        input_dict = {
            "input_ids": torch.tensor([inputs["input_ids"][0] for inputs in all_inputs]),
            "attention_mask": torch.tensor([inputs["attention_mask"][0] for inputs in all_inputs])
                    }
        return input_dict, all_inputs

    def get_kbest(self, valid_answers, rank="score", best_k=1):
        if len(valid_answers) > 0: 
            return sorted(valid_answers, key=lambda x: x[rank], reverse=True)[:best_k]
        else: 
            return [{
                            "qa_score": -999,
                            "score": -999,
                            "start": -999,
                            "end": -999,
                            "text": "",
                        }]

    def get_best_ans(self, inputs: Dict, outputs: Dict, contexts:List[str], all_inputs: List) -> List[Dict]:
        standardize = lambda x: (x - x.mean())/(x.std())
        end_sent = set([".", ",", "?", "!", ":"])
        ret = []
        for idx in range(len(outputs["start_logits"])):
            start_logits = standardize(outputs.start_logits[idx].detach().numpy())
            end_logits = standardize(outputs.end_logits[idx].detach().numpy())
#             start_logits = standardize(outputs.start_logits[idx].detach().numpy()[0]) # For GPU usage
#             end_logits = standardize(outputs.end_logits[idx].detach().numpy()[0])
            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []
            offset_mapping = all_inputs[idx]["offset_mapping"][0]
            cls_index = inputs["input_ids"].tolist()[idx].index(self.tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index]*self.start_weight + end_logits[cls_index]*(1-self.start_weight)
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -self.nbest - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -self.nbest - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or cls_index == start_index
                        or cls_index == end_index
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index 
                        or end_index - start_index + 1 > self.max_answer_length
                        or end_index - start_index + 1 < self.min_answer_length
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score":(start_logits[start_index]*(self.start_weight) + end_logits[end_index]*(1-self.start_weight)),
                            "start":start_logits[start_index],
                            "end":end_logits[end_index],
                            "text": contexts[idx][start_char: end_char],
                        }
                    )
            ret.extend(self.get_kbest(
                    valid_answers, 
                    rank="score",
                    best_k=1))
        return ret
    
    def process_zip(self, file):
        file = zipfile.ZipFile(f"/tmp/{file.filename}")
        file.extractall(path="/tmp/zip")
        files = glob.glob("/tmp/zip/*")
        contexts = []
        for fp in files:
            with open(fp, "r") as f:
                contexts += [inp.strip() for inp in f.read().strip().split("\n") if (inp.strip() != "" and inp.strip() != "\n")]
                # contexts += ["\n--------------------------------------------------------------------------------------------\n"]
        return contexts

    def predict(self, inputs: JsonDict) -> JsonDict:
        """
        Returns predictions.
        """
        question = inputs["question"].strip()
        if self.file: contexts = self.process_zip(self.file)
        else: contexts = [inp.strip() for inp in inputs["passage"].strip().split("\n") if (inp.strip() != "" and inp.strip() != "\n")]
        input_dict, all_inputs = self.preprocess(question, contexts)
        ouputs = self.model2(**input_dict)
        answer = self.get_best_ans(input_dict, ouputs, contexts, all_inputs)
        answer = [ans['text'] for ans in answer[:5]]
        return {

  "best_span_str": answer,
#   "question": question,
  "question": question,
  "context": contexts,
  "answer": "\n".join(answer),
#   "passage_question_attention": [],
#   "passage_tokens": [],
#   "question_tokens": [],
#   "span_end_logits": [],
#   "span_end_probs": [],
#   "span_start_logits": [],
#   "span_start_probs": [],
#   "token_offsets": []
}


if __name__ == "__main__":
    endpoint = RobertaModelEndpoint()
    endpoint.run()
    # inputs = {
    #     "passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
    #     "question": "How many partially reusable launch systems were developed?",
    # }
    # prediction = endpoint.predict(inputs)
    # print(prediction)
