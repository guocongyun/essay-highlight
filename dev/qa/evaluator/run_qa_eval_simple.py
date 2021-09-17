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
    BigBirdForQuestionAnswering,
    BigBirdTokenizer,
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
from torch.utils.data import DataLoader
import scipy
from keybert import KeyBERT
import pdb 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
kw_extractor = KeyBERT('paraphrase-mpnet-base-v2')

# sudo python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29523 run_qa_eval.py
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

nlp = spacy.load('en_core_web_sm')
similarity_model = SentenceTransformer('paraphrase-mpnet-base-v2')
softmax = lambda x: np.exp(x)/sum(np.exp(x))
standardize = lambda x: (x - x.mean())/(x.std())
calc_prob = lambda lmetistx: [1-scipy.stats.norm.sf(x) for x in listx]

def find_similar_sentence(target, paragraph: str) -> str:

    # Single list of sentences
    sentences = [s.text.strip() for s in nlp(paragraph).sents if s.text.strip() != ""]

    #Compute ebeddings
    tar_embed = similarity_model.encode(target, convert_to_tensor=True)
    sent_embed = similarity_model.encode(sentences, convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(tar_embed, sent_embed)
    sorted_sent = sorted(zip(cosine_scores.tolist()[0],sentences), key=lambda x: x[0], reverse=True)
    return np.array(sorted_sent)[:5]

def calc_score(target, vali_ans) -> str:
    tar_embed = similarity_model.encode(target, convert_to_tensor=True)
    sent_embed = similarity_model.encode(vali_ans, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(tar_embed, sent_embed)
    return standardize(np.array(cosine_scores.tolist()[0]))


def predict(model,
            similarity_model, 
            tokenizer,
            question, 
            contexts,
            similarity_weights = 0,
            start_weight = 0.8,
            max_answer_length = 200,
            min_answer_length = 2,
            stride = 128,
            num_proc = 2,
            nbest = 10,
            null_score_diff_threshold = 0,
            max_seq_length = 512,
            pad_on_right = True):
        
#         pdb.set_trace()
        all_input = []
        for context in contexts:
            inputs = tokenizer(
                question.lower(), 
                context.lower(), 
                truncation="only_second",
                max_length=max_seq_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length")
            for i in range(len(inputs["input_ids"])):
                sequence_ids = inputs.sequence_ids(i)
                context_index = 1 if pad_on_right else 0
                inputs["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(inputs["offset_mapping"][i])
                ]
            all_input.append(inputs)
            
        input_dict = {
            "input_ids": torch.tensor([inputs["input_ids"][0] for inputs in all_input]),
            "attention_mask": torch.tensor([[inputs["attention_mask"][0]] for inputs in all_input])
                     }
#         n = len(inputs["input_ids"])
#         for i in range(n):
#             tmp = {"input_ids":torch.tensor([inputs["input_ids"][i]]), "attention_mask":torch.tensor([inputs["attention_mask"][i]])}
        
        outputs = model(**input_dict)
        best_answer = []
        for idx in range(len(outputs["start_logits"])):
            start_logits = standardize(outputs.start_logits[idx].detach().numpy())
            end_logits = standardize(outputs.end_logits[idx].detach().numpy())
#             start_logits = standardize(outputs.start_logits[idx].detach().numpy()[0])
#             end_logits = standardize(outputs.end_logits[idx].detach().numpy()[0])
            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []
            offset_mapping = inputs["offset_mapping"][i]
            cls_index = inputs["input_ids"][i].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index]*start_weight + end_logits[cls_index]*(1-start_weight)
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -nbest - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -nbest - 1 : -1].tolist()
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
                        or end_index - start_index + 1 > max_answer_length
                        or end_index - start_index + 1 < min_answer_length
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "qa_score":(start_logits[start_index]*(start_weight) + end_logits[end_index]*(1-start_weight)),
                            "start":start_logits[start_index],
                            "end":end_logits[end_index],
                            "text": context[start_char: end_char],
                        }
                    )
        #     valid_answers.append(
        #         {
        #             "qa_score":(start_logits[1]*(start_weight) + end_logits[-1]*(1-start_weight)),
        #             "text": context[1: (len(start_logits))],
        #         }
        #     )
            texts = []
            if similarity_weights > 0:
                for answer in valid_answers:
                    texts.append(answer["text"])
                if texts: scores = calc_score(question, texts)
                else: scores = -999
                for idx, answer in enumerate(valid_answers):
                    answer["similarity_score"] = scores[idx]
                    answer["score"] = (answer["similarity_score"])*similarity_weights + answer["qa_score"]*(1-similarity_weights)
            else:
                for answer in valid_answers:
                    answer["score"] = answer["qa_score"]
    #     if len(valid_answers) > 0: best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:best_k]
    #     else: best_answer = [{
    #                         "qa_score": -999,
    #                         "score": -999,
    #                         "start": -999,
    #                         "end": -999,
    #                         "text": "",
    #                     }]
    # #     for answer in best_answer[:1]:
    # #         print(answer["text"])

    #     qa_confidence = np.average([best_answer[i]["qa_score"] for i in range(len(best_answer))])
    #     qa_have_ans = "have answer" if qa_confidence-min_null_score > null_score_diff_threshold else "no answer"
    # #     print(f"QA model thinks {qa_have_ans} with {round(qa_confidence, 5)} confident of having answers, null threshold is {min_null_score}")
    #     if similarity_weights > 0:
    #         ts_confidence = np.average([best_answer[i]["qa_score"] for i in range(len(best_answer))])
    #         ts_have_ans = "have answer" if ts_confidence-0.95 > null_score_diff_threshold else "no answer"
    #         confidence = qa_confidence*(1-similarity_weights)+ts_confidence*similarity_weights
    #         have_ans = "have answer" if confidence-min_null_score*(1-similarity_weights)-0.95*similarity_weights > null_score_diff_threshold else "no answer"
    # #         print(f"Similarity model thinks {ts_have_ans} with {round(ts_confidence, 5)} confident of having answers, null threshold is {0.95}")
    # #         print(f"Overall model thinks {have_ans} with {round(confidence, 5)} overall score of having answers")
    #     else:
    #         confidence = qa_confidence
            best_answer.extend(
                get_kbest(
                    valid_answers, 
                    rank="score",
                    best_k=1)
            )


        return best_answer

def get_kbest(valid_answers, rank="score", best_k=1):
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

# def exist_n_set(num, sol):
    
    
def get_quarter_best(num, valid_ans):
    best_score =  score = sum([x["score"] for x in valid_ans[0:num]])
    best_start = 0
    for idx, ans in enumerate(valid_ans):
        if idx < num:
            continue
        else:
            score += valid_ans[idx]["score"] 
            score -= valid_ans[idx-num]["score"] 
        if score > best_score: best_start = idx - num + 1
    
    return list(range(best_start,best_start+num))
    
def main():
    # Detecting last checkpoint.
    output_dir = "./eval"

    set_seed(42)
    max_seq_length = 512

    data_files = {}
    extension = "json"
    pad_on_right=True
    data_files["train"] = "../../data/sed/train.json"
    data_files["validation"] = "../../data/sed/test.json"
#     model = "vasudevgupta/bigbird-roberta-natural-questions"
    model = get_last_checkpoint("./roberta_model/deepsettosed_kf2")
    model_baseline = "roberta"
    raw_datasets = load_dataset("./squad_loader.py", data_files=data_files)
    
    train_args = TrainingArguments(per_gpu_train_batch_size=8, num_train_epochs=0, output_dir=output_dir, save_strategy="epoch")

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model,
        cache_dir="./cache",
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        cache_dir="./cache",
        use_fast=True,
        revision="main",
        use_auth_token= None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model,
        from_tf=False,
        config=config,
        cache_dir="./cache",
        revision="main",
        use_auth_token=None,
    )
    ans_map = [
        {"introduction":list(range(5)), "methodology":list(range(5,9)), "experiment":list(range(9, 14)), "analysis": list(range(14,21)), "conclusion": list(range(21,27))},
        {"introduction":list(range(5)), "methodology":list(range(5,7)), "experiment":list(range(7, 14)), "analysis": list(range(14,21)), "conclusion": []},
        {"introduction":list(range(7)), "methodology":list(range(7,13)), "experiment":list(range(13, 22)), "analysis": list(range(22,25)), "conclusion": []},
        {"introduction":list(range(2)), "methodology":list(range(2,5)), "experiment":list(range(5, 9)), "analysis": list(range(9,14)), "conclusion": []},
        {"introduction":list(range(6)), "methodology":list(range(6,16)), "experiment":list(range(16, 23)), "analysis": list(range(23,27)), "conclusion": list(range(27,28))},
        {"introduction":list(range(3)), "methodology":list(range(3,6)), "experiment":list(range(6, 8)), "analysis": list(range(8,12)), "conclusion": list(range(12,13))},
        {"introduction":list(range(3)), "methodology":list(range(3,6)), "experiment":list(range(6, 11)), "analysis": list(range(11,16)), "conclusion": list(range(16,17))},
        {"introduction":list(range(2)), "methodology":list(range(2,10)), "experiment":list(range(10, 12)), "analysis": list(range(12,15)), "conclusion": []},
        {"introduction":list(range(3)), "methodology":list(range(3,8)), "experiment":list(range(8, 18)), "analysis": list(range(18,27)), "conclusion": []},
        {"introduction":list(range(1)), "methodology":list(range(1,9)), "experiment":list(range(9, 16)), "analysis": list(range(16,21)), "conclusion": []},
        {"introduction":list(range(4)), "methodology":list(range(4,11)), "experiment":list(range(11, 15)), "analysis": list(range(15,17)), "conclusion": list(range(17,18))},
        {"introduction":list(range(1)), "methodology":list(range(1,3)), "experiment":list(range(3, 7)), "analysis": list(range(7,13)), "conclusion": list(range(13,14))},
        {"introduction":list(range(6)), "methodology":list(range(6,11)), "experiment":list(range(11, 19)), "analysis": list(range(19,25)), "conclusion": list(range(25,26))},
        {"introduction":list(range(3)), "methodology":list(range(3,8)), "experiment":list(range(8, 11)), "analysis": list(range(11,16)), "conclusion": []},
        {"introduction":list(range(3)), "methodology":list(range(3,15)), "experiment":list(range(15, 26)), "analysis": list(range(26,28)), "conclusion": []},
        {"introduction":list(range(7)), "methodology":list(range(7,13)), "experiment":list(range(13, 24)), "analysis": list(range(24,35)), "conclusion": list(range(35,36))},
        {"introduction":list(range(4)), "methodology":list(range(4,19)), "experiment":list(range(19, 21)), "analysis": list(range(21,23)), "conclusion": list(range(23,25))},
        {"introduction":list(range(3)), "methodology":list(range(3,9)), "experiment":list(range(9, 17)), "analysis": list(range(17,27)), "conclusion": []},
        {"introduction":list(range(3)), "methodology":list(range(3,6)), "experiment":list(range(6, 18)), "analysis": [], "conclusion": []},
        {"introduction":list(range(2)), "methodology":list(range(2,10)), "experiment":list(range(10, 18)), "analysis": list(range(18,26)), "conclusion": []},
        {"introduction":list(range(3)), "methodology":list(range(3,5)), "experiment":list(range(5, 9)), "analysis": [], "conclusion": []},
        {"introduction":list(range(4)), "methodology":list(range(4,10)), "experiment":list(range(10, 13)), "analysis": list(range(13,17)), "conclusion": []},
        {"introduction":list(range(4)), "methodology":list(range(4,8)), "experiment":list(range(8, 11)), "analysis": list(range(11,17)), "conclusion": list(range(17,27))},
              ]
    intro = "The paper provides an introduction to the proposed topic and presents the analytical questions that the project seeks to answer."
    meth = "The paper provides sufficient and clear details on the group's methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models."
    exp = "The paper provides sufficient and clear details about experiments, including experimental data, preprocessing steps, and the evaluation metrics."
    ana = "The group explained their findings and interpretation by providing supporting examples or making use of suitable visualisation."
    
    intro2 = "introduction to the proposed topic and presents the analytical questions that the project seeks to answer."
    meth2 = "methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models."
    exp2 = "experiments, including experimental data, preprocessing steps, and the evaluation metrics."
    ana2 = "findings and interpretation by providing supporting examples or making use of suitable visualisation."
    
    hand_craft_key_map = {
        "introduction" : "introduction background abstract questions",
        "methodology" : "methodology BiLSTM BOW vectors",
        "experiment" : "experiments data preprocessing evaluation metrics",
        "analysis" : "findings interpretation examples results",
    }
    auto_key_map = {
        "introduction" : " ".join([x[0] for x in kw_extractor.extract_keywords(intro,  keyphrase_ngram_range=(1, 1), top_n=8, stop_words=None)]),
        "methodology": " ".join([x[0] for x in kw_extractor.extract_keywords(meth,  keyphrase_ngram_range=(1, 1), top_n=8, stop_words=None)]),
        "experiment": " ".join([x[0] for x in kw_extractor.extract_keywords(exp,  keyphrase_ngram_range=(1, 1), top_n=8, stop_words=None)]),
        "analysis": " ".join([x[0] for x in kw_extractor.extract_keywords(ana,  keyphrase_ngram_range=(1, 1), top_n=8, stop_words=None)]),
    }
    kf_to_ques = {
        "introduction" : intro,
        "methodology" : meth,
        "experiment" : exp,
        "analysis" : ana,
    }
    kf_to_handcraft_ques = {
        "introduction" : intro2,
        "methodology" : meth2,
        "experiment" : exp2,
        "analysis" : ana2,
    }
    ratio=0.8
    precisions, recalls = [[] for _ in range(4)], [[] for _ in range(4)]
    for idx in range(22):
        with open("../../data/uni_clean_v3/"+str(idx), encoding="utf-8") as f:
            contexts = f.read().split("\n")
        total_score = []
        for idx, (question, value) in enumerate(ans_map[idx].items()):
            if question not in ["introduction", "methodology", "experiment", "analysis"]:
                continue
            
            question = hand_craft_key_map[question]
            best_answer, best_start, best_end = [], [], []
            best_scores = []
            
            for context in contexts:
                sentences = [s.text.strip() for s in nlp(context).sents if s.text.strip() != ""]
                tar_embed = similarity_model.encode(question, convert_to_tensor=True)
                sent_embed = similarity_model.encode(sentences, convert_to_tensor=True)
                scores = util.pytorch_cos_sim(tar_embed, sent_embed)
                best_scores.append(np.max(scores.cpu().tolist()[0]))

#                 best_answer.extend(
#                     get_kbest(
#                         valid_ans, 
#                         rank="score",
#                         best_k=1)
#                 )
#             best_answer = predict(model, 
#                         similarity_model, 
#                         tokenizer,
#                         question, 
#                         contexts,
#                         similarity_weights=0,)
#                 best_start.extend(
#                     get_kbest(
#                         valid_ans, 
#                         rank="start",
#                         best_k=1)
#                 )
#                 best_end.extend(
#                     get_kbest(
#                         valid_ans, 
#                         rank="end",
#                         best_k=1)
#                 )
#             best_scores = [x["score"] for x in best_answer]
#             best_scores = get_quarter_best(len(best_answer)//4, best_answer)

#             best_start = [x["start"] for x in best_start]
#             best_end = [x["end"] for x in best_end]
            
#             p_scores = []
#             for ids, start in enumerate(best_start):
#                 for ide, end in enumerate(best_end):
#                     if ids > ide: continue
#                     p_scores.append(
#                         {
#                             "score": start*ratio+end*(1-ratio),
#                             "start": ids,
#                             "end": ide,
#                         }
#                     )
            
            best_answer = np.argsort(best_scores)
            best_answer = best_answer[::-1]
            hal_size = len(best_answer)//4
            
#             best_p_answer = []
#             for ps in sorted(p_scores, key=lambda x: x["score"])[::-1]:
#                 for i in range(ps["start"], ps["end"]):
#                     if i not in best_p_answer: best_p_answer.append(i)
#             best_answer = best_p_answer 
            
            correct = 0
            for ans in best_answer[:hal_size]:
#             for ans in best_scores:
                if ans in value:
                    correct += 1
#             print(best_scores)
#             print(value)
#             print(f"{question}:{recall/len(value)}")
#             print(best_answer)
#             precision = correct/len(best_answer) if len(best_answer) > 0 else 0
            precision = correct/hal_size if hal_size > 0 else 0
            recall = correct/len(value) if len(value) > 0 else 0
            precisions[idx].append(precision)
            recalls[idx].append(recall)
            total_score.append((precision, recall))
        print(total_score)
    for i in range(4):
        print(np.mean(precisions[i]))
        print(np.mean(recalls[i]))
#         print(np.average(total_score))

#     for answer in best_answer:
#         print(answer["text"])
#         print(answer["score"])


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     [(0.6666666666666666, 0.8), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
# [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.7142857142857143)]
# [(0.0, 0.0), (0.0, 0.0), (0.5, 0.3333333333333333), (0.5, 1.0)]
# [(0.3333333333333333, 0.5), (0.0, 0.0), (0.0, 0.0), (1.0, 0.6)]
# [(0.8571428571428571, 1.0), (0.0, 0.0), (0.2857142857142857, 0.2857142857142857), (0.5714285714285714, 1.0)]
# [(1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (0.6666666666666666, 0.5)]
# [(0.5, 0.6666666666666666), (0.25, 0.3333333333333333), (0.0, 0.0), (0.75, 0.6)]
# [(0.6666666666666666, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
# [(0.5, 1.0), (0.0, 0.0), (0.16666666666666666, 0.1), (1.0, 0.6666666666666666)]
# [(0.0, 0.0), (1.0, 0.625), (0.0, 0.0), (0.2, 0.2)]
# [(0.0, 0.0), (0.0, 0.0), (0.25, 0.25), (0.5, 1.0)]
# [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.6666666666666666, 0.3333333333333333)]
# [(0.3333333333333333, 0.3333333333333333), (0.0, 0.0), (0.0, 0.0), (0.8333333333333334, 0.8333333333333334)]
# [(0.75, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.8)]
# [(0.0, 0.0), (0.0, 0.0), (0.7142857142857143, 0.45454545454545453), (0.0, 0.0)]
# [(0.5555555555555556, 0.7142857142857143), (0.0, 0.0), (0.0, 0.0), (0.8888888888888888, 0.7272727272727273)]
# [(0.6666666666666666, 1.0), (0.0, 0.0), (0.3333333333333333, 1.0), (0.3333333333333333, 1.0)]
# [(0.0, 0.0), (0.0, 0.0), (0.3333333333333333, 0.25), (0.0, 0.0)]
# [(0.75, 1.0), (0.0, 0.0), (1.0, 0.3333333333333333), (0.0, 0)]
# [(0.3333333333333333, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
# [(0.0, 0.0), (0.0, 0.0), (1.0, 0.5), (0.0, 0)]
# [(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    
    
    
    
    
    
    
    
    
    
    
    
    
# KF para
# methodology:0.0
# experiment:0.5
# analysis:0.3333333333333333
# [0.0, 0.5, 0.3333333333333333]
# 0.27777777777777773
# methodology:0.5
# experiment:0.2857142857142857
# analysis:0.0
# [0.5, 0.2857142857142857, 0.0]
# 0.2619047619047619
# methodology:0.4
# experiment:0.2
# analysis:0.25
# [0.4, 0.2, 0.25]
# 0.2833333333333334
# methodology:0.16666666666666666
# experiment:0.0
# analysis:0.1
# [0.16666666666666666, 0.0, 0.1]
# 0.08888888888888889
# methodology:1.0
# experiment:0.16666666666666666
# [1.0, 0.16666666666666666]
# 0.5833333333333334

# Similarity para
# methodology:1.0
# experiment:0.5
# analysis:0.16666666666666666
# [1.0, 0.5, 0.16666666666666666]
# 0.5555555555555556
# methodology:0.0
# experiment:0.2857142857142857
# analysis:0.25
# [0.0, 0.2857142857142857, 0.25]
# 0.17857142857142858
# methodology:0.0
# experiment:0.2
# analysis:0.125
# [0.0, 0.2, 0.125]
# 0.10833333333333334
# methodology:0.16666666666666666
# experiment:0.5
# analysis:0.3
# [0.16666666666666666, 0.5, 0.3]
# 0.3222222222222222
# methodology:0.0
# experiment:0.25
# [0.0, 0.25]
# 0.125

# KF
# methodology:1.0
# experiment:0.16666666666666666
# analysis:0.3333333333333333
# [1.0, 0.16666666666666666, 0.3333333333333333]
# 0.5
# methodology:0.5
# experiment:0.14285714285714285
# analysis:0.25
# [0.5, 0.14285714285714285, 0.25]
# 0.2976190476190476
# methodology:0.2
# experiment:0.0
# analysis:0.375
# [0.2, 0.0, 0.375]
# 0.19166666666666665
# methodology:0.3333333333333333
# experiment:0.0
# analysis:0.3
# [0.3333333333333333, 0.0, 0.3]
# 0.2111111111111111
# methodology:0.5
# experiment:0.16666666666666666
# [0.5, 0.16666666666666666]
# 0.3333333333333333



# jovyan@guocongyun-safety3-0:~/AIED_2021_TRMRC_code-main/models/roberta$ python3 run_qa_eval_simple.py 
# Using custom data configuration default-18091e4902c5ce2c
# Reusing dataset squad (/home/jovyan/.cache/huggingface/datasets/squad/default-18091e4902c5ce2c/0.0.0/3c7aa1dd57aca723f9bb028dc1ace4e6ea954f3ddf9a43c36b55d929a68a9c2f)
# [0, 1, 2, 3, 4, 5]
# [3]
# methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.:1.0
# [19, 20, 21, 22, 23, 24]
# [7, 8, 9, 10, 11, 12]
# experiments, including experimental data, preprocessing steps, and the evaluation metrics.:0.0
# [0, 1, 2, 3, 4, 5]
# [13, 14, 15, 16, 17, 18]
# findings and interpretation by providing supporting examples or making use of suitable visualisation.:0.0
# [1.0, 0.0, 0.0]
# 0.3333333333333333
# [3, 4, 5, 6, 7]
# [4, 5]
# methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.:1.0
# [6, 7, 8, 9, 10]
# [6, 7, 8, 9, 10, 11, 12]
# experiments, including experimental data, preprocessing steps, and the evaluation metrics.:0.7142857142857143
# [15, 16, 17, 18, 19]
# [16, 17, 18, 19]
# findings and interpretation by providing supporting examples or making use of suitable visualisation.:1.0
# [1.0, 0.7142857142857143, 1.0]
# 0.9047619047619048
# [15, 16, 17, 18, 19]
# [0, 1, 2, 3, 4]
# methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.:0.0
# [15, 16, 17, 18, 19]
# [6, 7, 8, 9, 10]
# experiments, including experimental data, preprocessing steps, and the evaluation metrics.:0.0
# [10, 11, 12, 13, 14]
# [11, 12, 13, 14, 15, 16, 17, 18]
# findings and interpretation by providing supporting examples or making use of suitable visualisation.:0.5
# [0.0, 0.0, 0.5]
# 0.16666666666666666
# [0, 1, 2, 3, 4, 5]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.:0.5
# [0, 1, 2, 3, 4, 5]
# [12, 13, 14, 15]
# experiments, including experimental data, preprocessing steps, and the evaluation metrics.:0.0
# [5, 6, 7, 8, 9, 10]
# [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# findings and interpretation by providing supporting examples or making use of suitable visualisation.:0.0
# [0.5, 0.0, 0.0]
# 0.16666666666666666
# [0, 1, 2]
# [0, 1]
# methodology, including how to turn sentences into vectors and description about the BOW and BiLSTM models.:1.0
# [0, 1, 2]
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# experiments, including experimental data, preprocessing steps, and the evaluation metrics.:0.08333333333333333
# [1.0, 0.08333333333333333]
# 0.5416666666666666
# jovyan@guocongyun-safety3-0:~/AIED_2021_TRMRC_code-main/models/roberta$ 