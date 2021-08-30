from functools import lru_cache
from dataclasses import asdict
import json
from typing import Callable, Dict

from flask import Flask, Request, Response, after_this_request, request, jsonify
from allennlp.version import VERSION
from allennlp.predictors.predictor import JsonDict
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from allennlp_demo.common import config
from allennlp_demo.common.logs import configure_logging
from typing import List
import numpy as np
import torch
import zipfile
import glob
import os

def no_cache(request: Request) -> bool:
    """
    Returns True if the "no_cache" query string argument is present in the provided request.

    This provides a consistent mechanism across all endpoints for disabling the cache.
    """
    return "no_cache" in request.args


def with_cache_hit_response_headers(fn: Callable, *args):
    """
    Calls the provided function with the given arguments and returns the results. If the results
    are produced by a cache a HTTP header is added to the response.

    The provided function must be memoized using the functools.lru_cache decorator.
    """
    # This allows us to determine if the response we're serving was cached. It's safe to
    # do because we use a single-threaded server.
    pre_hits = fn.cache_info().hits  # type: ignore
    r = fn(*args)

    # If it was a cache hit add a HTTP header to the response
    if fn.cache_info().hits - pre_hits == 1:  # type: ignore

        @after_this_request
        def add_header(resp: Response) -> Response:
            resp.headers["X-Cache-Hit"] = "1"
            return resp

    return r


class NotFoundError(RuntimeError):
    pass


class UnknownInterpreterError(NotFoundError):
    def __init__(self, interpreter_id: str):
        super().__init__(f"No interpreter with id '{interpreter_id}'")


class InvalidInterpreterError(NotFoundError):
    def __init__(self, interpreter_id: str):
        super().__init__(f"Interpreter with id '{interpreter_id}' is not supported for this model")


class UnknownAttackerError(NotFoundError):
    def __init__(self, attacker_id: str):
        super().__init__(f"No attacker with id '{attacker_id}'")


class InvalidAttackerError(NotFoundError):
    def __init__(self, attacker_id: str):
        super().__init__(f"Attacker with id '{attacker_id}' is not supported for this model")


class MyModelEndpoint:
    """
    Class capturing a single model endpoint which provides a HTTP API suitable for use by
    the AllenNLP demo.

    This class can be extended to implement custom functionality.
    """

    def __init__(self, model: config.Model, log_payloads: bool = False):
        self.model = model
        self.app = Flask(model.id)
        self.configure_logging(log_payloads)
        self.configure_error_handling()
        self.file = None

        # By creating the LRU caches when the class is instantiated, we can
        # be sure that the caches are specific to the instance, and not the class,
        # i.e. every instance will have its own set of caches.

        @lru_cache(maxsize=1024)
        def predict_with_cache(inputs: str) -> JsonDict:
            return self.predict(json.loads(inputs))

        self.predict_with_cache = predict_with_cache

        self.setup_routes()
        self.config = AutoConfig.from_pretrained(
            self.model.model_path,
            cache_dir="./cache",
            use_auth_token=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.model_path,
            use_fast=True,
            cache_dir="./cache",
            use_auth_token= None,
    #         force_download=True,
        )
        self.model2 = AutoModelForQuestionAnswering.from_pretrained(
            self.model.model_path,
            cache_dir="./cache",
            revision="main",
            config=self.config,
        )
    
    def preprocess(self, question: str, contexts: List[str]) -> Dict:
        all_inputs = []
        for context in contexts:
            inputs = self.tokenizer(
                question.lower(), 
                context.lower(), 
                truncation="only_second",
                max_length=self.model.max_seq_length,
                stride=self.model.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length")
            for i in range(len(inputs["input_ids"])):
                sequence_ids = inputs.sequence_ids(i)
                context_index = 1 if self.model.pad_on_right else 0
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
            feature_null_score = start_logits[cls_index]*self.model.start_weight + end_logits[cls_index]*(1-self.model.start_weight)
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -self.model.nbest - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -self.model.nbest - 1 : -1].tolist()
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
                        or end_index - start_index + 1 > self.model.max_answer_length
                        or end_index - start_index + 1 < self.model.min_answer_length
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score":(start_logits[start_index]*(self.model.start_weight) + end_logits[end_index]*(1-self.model.start_weight)),
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


    def info(self) -> str:
        """
        Returns basic information about the model and the version of AllenNLP.
        """
        return jsonify({**asdict(self.model), "allennlp": VERSION})

    def configure_logging(self, log_payloads: bool = False) -> None:
        configure_logging(self.app, log_payloads=log_payloads)

    def configure_error_handling(self) -> None:
        def handle_invalid_json(err: json.JSONDecodeError):
            return jsonify({"error": str(err)}), 400

        self.app.register_error_handler(json.JSONDecodeError, handle_invalid_json)

        def handle_404(err: NotFoundError):
            return jsonify({"error": str(err)}), 404

        self.app.register_error_handler(NotFoundError, handle_404)

    def setup_routes(self) -> None:
        """
        Binds HTTP paths to verbs supported by a standard model endpoint. You can override this
        method to define additional routes or change the default ones.
        """

        @self.app.route("/")
        def info_handler():
            return self.info()

        @self.app.route("/upload", methods=["GET", "POST"])
        def upload_handler():
            if request.method == 'POST':

                file = request.files['file']
                # If the user does not select a file, the browser submits an
                # empty file without a filename.
                filename = file.filename
                file.save(os.path.join("/tmp", filename))
                self.file = file
            return ""

        # noop post for image upload, we need an endpoint, but we don't need to save the image
        @self.app.route("/noop", methods=["POST"])
        def noop():
            return ""

        @self.app.route("/predict", methods=["POST"])
        def predict_handler():
            if no_cache(request):
                return jsonify(self.predict(request.get_json()))
            return jsonify(with_cache_hit_response_headers(self.predict_with_cache, request.data))

        @self.app.route("/interpret/<string:interpreter_id>", methods=["POST"])
        def interpet_handler(interpreter_id: str):
            if no_cache(request):
                return jsonify(self.interpret(interpreter_id, request.get_json()))
            return jsonify(
                with_cache_hit_response_headers(
                    self.interpret_with_cache, interpreter_id, request.data
                )
            )

        @self.app.route("/attack/<string:attacker_id>", methods=["POST"])
        def attack_handler(attacker_id: str):
            if no_cache(request):
                return jsonify(self.attack(attacker_id, request.get_json()))
            return jsonify(
                with_cache_hit_response_headers(self.attack_with_cache, attacker_id, request.data)
            )

    def run(self, port: int = 8000) -> None:
        self.app.run(host="0.0.0.0", port=port)
