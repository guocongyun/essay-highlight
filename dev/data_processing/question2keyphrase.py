from nltk import data
from numpy.lib.function_base import average, quantile
from sentence_transformers import SentenceTransformer, util
from numpy.core.fromnumeric import argmax
import nltk
from nltk import tokenize
import spacy
from typing import List
import json
from tqdm import tqdm
from transformers import utils
import numpy as np
from keybert import KeyBERT
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
kw_extractor = KeyBERT('paraphrase-mpnet-base-v2')
# nlp = spacy.load("en_core_web_sm")
def repharse_question(ques):
    doc = nlp(ques)
    ques = " ".join([t.text for t in doc])
    map_dict = {"you":"I", "your":"my"}
    ques_token_list = ques.split()
    for i in range(len(ques_token_list)):
        _token = ques_token_list[i]
        _lower_token = _token.lower()
        if _lower_token in map_dict:
            sub_token = map_dict[_lower_token]
            ques_token_list[i] = sub_token
    return " ".join(ques_token_list)

with open("../../data/squad_v2/train.json", encoding='utf-8') as f:
    squad = json.load(f)
    for article in tqdm(squad["data"]):
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]  # do not strip leading blank spaces GH-2585
            for qa in paragraph["qas"]:
                ques = qa["question"]
#                 ques = repharse_question(ques)
                keywords = [x[0] for x in kw_extractor.extract_keywords(ques,  keyphrase_ngram_range=(1, 1), top_n=2, stop_words=None)]
                qa["question"] = " ".join(keywords)
#                 qa["question"] = ques

with open("../../data/squad_v2_kf/train.json", 'w+', encoding='utf-8') as f:
    json.dump(squad, f)
