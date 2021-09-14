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
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions
import collections
from tqdm.auto import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import timeit
import torch
import nltk
import spacy

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# sudo python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29523 run_qa_eval.py
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

nlp = spacy.load('en_core_web_sm')

def main():
    # Detecting last checkpoint.
    output_dir = "./eval"

    set_seed(42)

    data_files = {}
    extension = "json"
    pad_on_right=True
    data_files["train"] = "../../data/sed/train.json"
    data_files["validation"] = "../../data/sed/test.json"
    similarity_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    model = get_last_checkpoint("./roberta_model/deepsettosed")
    model_baseline = "roberta"
    raw_datasets = load_dataset("./squad_loader.py", data_files=data_files)
    train_args = TrainingArguments(per_gpu_train_batch_size=8, num_train_epochs=0, output_dir=output_dir, save_strategy="epoch")
#     raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir="./cache")

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
#         "../../pretrain/models/checkpoint-85072",
        model,
#         "deepset/roberta-base-squad2",
        cache_dir="./cache",
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
#         "../../tok",
        model,
#         "deepset/roberta-base-squad2",
        cache_dir="./cache",
        use_fast=True,
        revision="main",
        use_auth_token= None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
#         "../../pretrain/models/checkpoint-85072",
        model,
#         "deepset/roberta-base-squad2",
        from_tf=False,
        config=config,
        cache_dir="./cache",
        revision="main",
        use_auth_token=None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    column_names = raw_datasets["train"].column_names
    column_names = raw_datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    max_seq_length = 512
    stride = 128
    num_proc = 2
    # Training preprocessing
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    train_dataset = raw_datasets["train"]
    
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_examples = raw_datasets["validation"]
    print(len(eval_examples))
    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )


    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
#         default_data_collator
#         if data_args.pad_to_max_length
#         else 
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    metric = load_metric("./squad_v2/squad_v2.py")
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    # Initialize our Trainer
    trainer = Trainer(
#     trainer = QuestionAnsweringTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
#         eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
#         post_process_function=post_processing_function,
#         compute_metrics=compute_metrics,
    )
    def similarity_score(question: str, answer: str, similarity_weights: float=0) -> int:
        if not similarity_weights: return 0
        ques_embed = similarity_model.encode(question, convert_to_tensor=True)
        ans_embed = similarity_model.encode(answer, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(ques_embed, ans_embed)
        return cosine_scores
    
    def postprocess_qa_predictions(examples, features, raw_predictions, similarity_weights=0.2, null_score_diff_threshold=-0.5, n_best_size=20, max_answer_length=512, method="HF"):
        print(raw_predictions)
        all_start_logits, all_end_logits = raw_predictions
        
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
        
        # Let's loop over all the examples!
        useful_example = []
        softmax = lambda x: np.exp(x)/sum(np.exp(x))
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            
            # Some example have empty features
            if not feature_indices: continue
            else: useful_example.append(example)
                
            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]
            ques = example["question"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = softmax(all_start_logits[feature_index])-np.full((max_answer_length,), 0.5)
                end_logits = softmax(all_end_logits[feature_index])-np.full((max_answer_length,), 0.5)

                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if method == "HF":
                            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                            # Don't consider answers with a length that is either < 0 or > max_answer_length.
                            if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                                or end_index < start_index 
                                or end_index - start_index + 1 > max_answer_length
                            ):
                                continue
                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]
                            valid_answers.append(
                                {
                                    "score": start_logits[start_index] + end_logits[end_index] + similarity_weights*similarity_score(ques, context[start_char: end_char], similarity_weights),
                                    "text": context[start_char: end_char]
                                }
                            )
                        if method == "AIED":
                            if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                                or end_index < start_index 
                                or end_index - start_index + 1 > max_answer_length
                            ):
                                valid_answers.append(
                                    {
                                        "score": start_logits[start_index] + end_logits[end_index] + similarity_weights*similarity_score(ques, "", similarity_weights),
                                        "text": ""
                                    }
                                )
                            else:
                                start_char = offset_mapping[start_index][0]
                                end_char = offset_mapping[end_index][1]
                                valid_answers.append(
                                    {
                                        "score": start_logits[start_index] + end_logits[end_index] + similarity_weights*similarity_score(ques, context[start_char: end_char], similarity_weights),
                                        "text": context[start_char: end_char]
                                    }
                                )
            # Calc probability

            if len(valid_answers) > 0: best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else: best_answer = {"text": "", "score": 0.0} # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
            
            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            answer = best_answer["text"] if best_answer["score"]-min_null_score > null_score_diff_threshold else ""
            predictions[example["id"]] = answer

        return predictions, useful_example
    
#     raw_predictions = trainer.evaluate(eval_dataset, eval_examples)
    
#     raw_predictions = trainer.predict(eval_examples, eval_dataset)
    start = timeit.timeit()
    raw_predictions = trainer.predict(eval_dataset)
    end = timeit.timeit()
#     for th in [-1, -0.75, -0.5,-0.25,-0.1, 0.1, 0.25, 0.5 , 0.75, 1]:
#     for sweight in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     for sweight in [0.2, 0.1, 0.4, 0.5, 0.7]:
    for sweight in [0]:
        print(f"sweight: {sweight}")
        if model_baseline == "bart":
            prediction = raw_predictions.predictions[:2]
#             print(len(prediction))
            print(np.shape(raw_predictions.predictions[0]))
            print(np.shape(raw_predictions.predictions[1]))
            print(np.shape(raw_predictions.predictions[2]))
        if model_baseline == "roberta":
            prediction = raw_predictions.predictions
        final_predictions, useful_example = postprocess_qa_predictions(
            eval_examples, 
            eval_dataset, 
            prediction, 
            similarity_weights=sweight, 
            null_score_diff_threshold=-0.5, 
            method="HF"
        )
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        references = [{ "answers": ex["answers"], "id": ex["id"]} for ex in useful_example]
        print(len(formatted_predictions))
        print(len(references))
        final_score = metric.compute(predictions=formatted_predictions, references=references)
        print(final_score)

        num_correct = 0
        total = 0
        for i in formatted_predictions:
            for j in references:
                if i["id"] == j["id"]:
                    total += 1
                    if bool(i["prediction_text"]) == bool(j["answers"]["text"]):
                        num_correct += 1
                    break
        print(num_correct, total, num_correct/total)
        
    


if __name__ == "__main__":
    main()
    