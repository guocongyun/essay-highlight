from functools import lru_cache
from dataclasses import asdict
import json
import re
from typing import Callable, Dict

from flask import Flask, Request, Response, after_this_request, request, jsonify
from allennlp.version import VERSION
from allennlp.predictors.predictor import JsonDict
from numpy.core.numeric import Inf
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BartForConditionalGeneration,
)
from allennlp_demo.common import config
from allennlp_demo.common.logs import configure_logging
from typing import List
import numpy as np
import torch
import zipfile
import glob
import os
import tempfile
import shutil
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer, util
import spacy

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
        self.contexts = []

        # By creating the LRU caches when the class is instantiated, we can
        # be sure that the caches are specific to the instance, and not the class,
        # i.e. every instance will have its own set of caches.

        @lru_cache(maxsize=1024)
        def predict_with_cache(inputs: str) -> JsonDict:
            return self.predict(json.loads(inputs))

        self.predict_with_cache = predict_with_cache

        self.setup_routes()

        if self.model.similarity_model_weight != 1:
            self.config = AutoConfig.from_pretrained(
                self.model.qa_model_path,
                cache_dir="./cache",
                use_auth_token=None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.qa_model_path,
                use_fast=True,
                cache_dir="./cache",
                use_auth_token= None,
        #         force_download=True,
            )
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                self.model.qa_model_path,
                cache_dir="./cache",
                revision="main",
                config=self.config,
            )
        
        if self.model.similarity_model_weight != 0:
            self.nlp = spacy.load('en_core_web_sm')
            self.similarity_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", cache_folder=self.model.similarity_model_path)
        
        if self.model.summerization_model:
            self.summarize_tokenizer = AutoTokenizer.from_pretrained(self.model.summarize_model_path)
            self.summarize_model = BartForConditionalGeneration.from_pretrained(self.model.summarize_model_path)

        self.standardize = lambda x: (x - x.mean())/(x.std())

    def similarity_score(self, target: str, texts: List[str]) -> str:

        tar_embed = self.similarity_model.encode(target, convert_to_tensor=True)
        sent_embed = self.similarity_model.encode(texts, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(tar_embed, sent_embed)
        scores = self.standardize(np.array(cosine_scores.tolist()[0]))
        return scores

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
                            "qa_score": -Inf,
                            "score": -Inf,
                            "start": -Inf,
                            "end": -Inf,
                            "text": "",
                        }]

    def get_best_ans(self, inputs: Dict, outputs: Dict, contexts:List[str], question: str, all_inputs: List) -> List[Dict]:
        end_sent = set([".", ",", "?", "!", ":"])
        ret = []
        for idx in range(len(outputs["start_logits"])):
            start_logits = self.standardize(outputs.start_logits[idx].detach().numpy())
            end_logits = self.standardize(outputs.end_logits[idx].detach().numpy())
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
            if self.model.similarity_model_weight != 0:
                texts = [text["text"] for text in valid_answers]
                scores = self.similarity_score(question, texts)
                for idx, answer in enumerate(valid_answers):
                    answer["score"] += self.model.similarity_model_weight*scores[idx]

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
                contexts += [[inp.strip() for inp in f.read().strip().split("\n") if (inp.strip() != "" and inp.strip() != "\n")]]
        return contexts

    def find_similar_texts(self, question: str, context: List[List[str]]) -> List[Dict]:
        ret = []
        for paragraph in context:
            texts = [s.text.strip() for s in self.nlp(paragraph).sents if s.text.strip() != ""]
            if not texts: continue
            score = self.similarity_score(question, texts)
            valid_answers = []
            for idx, s in enumerate(score):
                valid_answers.append(
                    {
                        "score":s,
                        "start":-1,
                        "end":-1,
                        "text": texts[idx],
                    }
                )
                ret.extend(self.get_kbest(
                    valid_answers, 
                    rank="score",
                    best_k=1))
            
            return ret
    
    def summarize(self, inputs: List[str]) -> List[str]:
        ret = []
        for text in inputs:
            inp = self.summarize_tokenizer(text, return_tensors="pt")
            pred = self.summarize_model.generate(**inp)
            pred = self.tokenizer.batch_decode(pred)
            ret.extend(pred)
        return ret

    def get_tags(self, ret: JsonDict) -> List[List[List[str]]]:
        k = len(ret["context"])
        tags = [[] for _ in range(k)]
        contexts = [[] for _ in range(k)]
        summarizations = [[] for _ in range(k)]
        p = 0

        for i in range(k):
            n = len(ret["context"][i])
            for j in range(n):
                start = ret["context"][i][j].find(ret["best_span_str"][i][j])
                if start != -1:
                    end = start + len(ret["best_span_str"][i][j])
                    if len(ret["context"][i][j]) > end:
                        list_context = [ret["context"][i][j][:start], ret["context"][i][j][start:end], ret["context"][i][j][end:]]
                        list_tag = ["O","B- ","O"]
                    else:
                        list_context = [ret["context"][i][j][:start], ret["context"][i][j][start:end]]
                        list_tag = ["O","B- "]
                    summary = ret["summarization"][p]
                    p += 1
                else:
                    list_context = ret["context"][i][j]
                    list_tag = ["O"]
                    summary = ""

                summarizations[i].append(summary)
                contexts[i].append(list_context)
                tags[i].append(list_tag)
        return tags, contexts, summarizations

        

    def predict(self, inputs: JsonDict) -> JsonDict:
        """
        Returns predictions.
        """
        print(inputs)
        question = inputs["question"].strip()
        if self.contexts: contexts = self.contexts
        else: contexts = [[inp.strip() for inp in inputs["passage"].strip().split("\n") if (inp.strip() != "" and inp.strip() != "\n")]]

        ret = {
            "best_span_str": [],
            "summarization": [],
            "question": [],
            "context": [],
            "answer": [],
            "tag": [],
        }       
        for context in contexts:
            if self.model.similarity_model_weight != 1: 
                input_dict, all_inputs = self.preprocess(question, context)
                ouputs = self.qa_model(**input_dict)
                answer = self.get_best_ans(input_dict, ouputs, context, question, all_inputs)
            else:
                answer = self.find_similar_texts(question, context)

            answer = [ans['text'] for ans in answer[:5]]
            ret["best_span_str"].append(answer)
            ret["question"].append(question)
            ret["context"].append(context)
            ret["answer"].append("\n".join(answer))

            if self.model.summerization_model: ret["summarization"].extend(self.summarize(ret["best_span_str"][-1]))

        self.contexts = []
        if not self.model.summerization_model:        
            return {
                "best_span_str": ret["best_span_str"],
                "question": ret["question"],
                "context": ret["context"],
                "answer": ret["answer"],
            }
        else:
            tags, contexts, summary = self.get_tags(ret)
            return {
                "best_span_str": ret["best_span_str"],
                "summarization": summary,
                "question": ret["question"],
                "context": contexts,
                "answer": ret["answer"],
                "tag": tags
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
        @cross_origin(supports_credentials=True)
        def upload_handler():
            if request.method == 'POST':
                file = request.files['file']
                file = zipfile.ZipFile(file)
                tmpdir = tempfile.mkdtemp()
                file.extractall(tmpdir)
                files = glob.glob(f"{tmpdir}/*")
                self.contexts = []
                for fp in files:
                    with open(fp, "r") as f:
                        self.contexts += [[inp.strip() for inp in f.read().strip().split("\n") if (inp.strip() != "" and inp.strip() != "\n")]]
                shutil.rmtree(tmpdir)
            return ""

        @self.app.route("/remove", methods=["GET", "POST"])
        def remove_handler():
            if self.contexts: self.contexts = []
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
